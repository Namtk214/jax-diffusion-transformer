"""
Evaluation script for DiT model - generates samples and computes FID.
Automatically computes reference FID stats if they don't exist.

Usage:
    python eval_fid.py --dataset_name=cifar100 --load_dir=checkpoints/cifar100
    python eval_fid.py --dataset_name=cifar10 --load_dir=checkpoints/cifar10 --num_samples=50000
    python eval_fid.py --dataset_name=imagenet256 --load_dir=checkpoints/imagenet256 --use_stable_vae=1
"""

try:
    from localutils.debugger import enable_debug
    enable_debug()
except ImportError:
    pass

from absl import app, flags
import os
import numpy as np
import tqdm

import jax
import jax.numpy as jnp
import flax
import optax

import tensorflow_datasets as tfds
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")

import matplotlib.pyplot as plt
from PIL import Image

from utils.train_state import TrainState
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from schedulers import GaussianDiffusion
from diffusion_transformer import DiT
from train_diffusion import DiffusionTrainer
from utils.fid import get_fid_network, fid_from_stats


# ============================================================
# FLAGS (SAFE): no hasattr/delattr, no duplicate defines
# ============================================================
FLAGS = flags.FLAGS

def define_if_missing(name, define_fn, *args, **kwargs):
    # SAFE: does not trigger UnparsedFlagAccessError
    if name not in FLAGS._flags():
        define_fn(name, *args, **kwargs)

# NOTE: dataset_name is defined in train_diffusion.py (per your error),
# so we MUST NOT redefine it here.

define_if_missing('load_dir', flags.DEFINE_string, None, 'Checkpoint directory to load from.')
define_if_missing('fid_stats', flags.DEFINE_string, None, 'Path to FID statistics file (.npz). Auto-computed if not exists.')
define_if_missing('output_dir', flags.DEFINE_string, 'generated_samples', 'Directory to save generated samples.')
define_if_missing('cfg_weight', flags.DEFINE_float, 4.0, 'Classifier-free guidance weight.')
define_if_missing('use_cfg', flags.DEFINE_integer, 1, 'Whether to use classifier-free guidance.')
define_if_missing('batch_size', flags.DEFINE_integer, 128, 'Total batch size (global). Must be divisible by num devices.')
define_if_missing('diffusion_timesteps', flags.DEFINE_integer, 500, 'Number of diffusion timesteps.')
define_if_missing('num_samples', flags.DEFINE_integer, 50000, 'Total number of samples to generate for FID.')
define_if_missing('seed', flags.DEFINE_integer, 42, 'Random seed.')
define_if_missing('use_stable_vae', flags.DEFINE_integer, 0, 'Whether to use Stable Diffusion VAE.')
define_if_missing('save_images', flags.DEFINE_integer, 0, 'Whether to save generated images to disk.')
define_if_missing('debug_overfit', flags.DEFINE_integer, 0, 'Debug mode.')


# ============================================================
# Dataset config
# ============================================================
def get_dataset_config(dataset_name):
    configs = {
        'cifar10': {
            'image_size': 32,
            'num_classes': 10,
            'tfds_name': 'cifar10',
            'test_split': 'test',
            'train_size': 50000,
        },
        'cifar100': {
            'image_size': 32,
            'num_classes': 100,
            'tfds_name': 'cifar100',
            'test_split': 'test',
            'train_size': 50000,
        },
        'imagenet128': {
            'image_size': 128,
            'num_classes': 1000,
            'tfds_name': 'imagenet2012',
            'test_split': 'validation',
            'train_size': 1281167,
        },
        'imagenet256': {
            'image_size': 256,
            'num_classes': 1000,
            'tfds_name': 'imagenet2012',
            'test_split': 'validation',
            'train_size': 1281167,
        },
    }
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(configs.keys())}")
    return configs[dataset_name]


# ============================================================
# Reference FID stats (REAL images) - PMAP SAFE
# ============================================================
def compute_reference_fid_stats(dataset_name, per_device_batch, get_fid_activations, output_path):
    """
    get_fid_activations is typically pmapped in this repo, so inputs MUST be:
      [n_devices, per_device_batch, 299, 299, 3]
    """
    print(f"\n{'='*60}")
    print(f"Computing reference FID stats for {dataset_name}...")
    print("This only needs to be done once and will be cached.")
    print(f"{'='*60}\n")

    cfg = get_dataset_config(dataset_name)
    total_samples = cfg['train_size']

    n_devices = jax.local_device_count()
    global_batch = per_device_batch * n_devices

    # Load dataset
    ds = tfds.load(cfg['tfds_name'], split='train')

    def preprocess(data):
        image = data['image']
        # Handle ImageNet cropping (keep same style as your original)
        if 'imagenet' in dataset_name:
            min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
            image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)

        image = tf.cast(image, tf.float32) / 255.0
        image = (image - 0.5) / 0.5  # normalize to [-1, 1] (kept as in your original)
        image = tf.image.resize(image, (299, 299), antialias=True)
        image = 2 * image - 1        # keep as in your original
        return image

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(global_batch, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    all_acts = []
    num_processed = 0

    steps = total_samples // global_batch
    pbar = tqdm.tqdm(ds, desc="Real images", total=steps)

    for batch in pbar:
        images = batch.numpy()  # [global_batch, 299, 299, 3]
        # shard for pmap
        images = images.reshape((n_devices, per_device_batch, 299, 299, 3))

        acts = get_fid_activations(images)          # typically [n_devices, per_device_batch, 1,1,2048]
        acts = np.array(acts)[..., 0, 0, :]         # [n_devices, per_device_batch, 2048]
        acts = acts.reshape((-1, acts.shape[-1]))   # [global_batch, 2048]

        all_acts.append(acts)
        num_processed += acts.shape[0]
        pbar.set_postfix({'processed': num_processed})

        # hard stop to respect expected total steps (tfds iterator might be infinite-ish)
        if num_processed >= steps * global_batch:
            break

    all_acts = np.concatenate(all_acts, axis=0)
    mu = np.mean(all_acts, axis=0)
    sigma = np.cov(all_acts, rowvar=False)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(output_path, mu=mu, sigma=sigma)
    print(f"Saved reference FID stats to {output_path}")
    print(f"Processed {num_processed} real images\n")

    return {'mu': mu, 'sigma': sigma}


# ============================================================
# Dataset iterator (labels)
# ============================================================
def get_dataset(dataset_name, per_device_batch, is_train=False, debug_overfit=False):
    cfg = get_dataset_config(dataset_name)
    n_devices = jax.local_device_count()
    global_batch = per_device_batch * n_devices

    if 'imagenet' in dataset_name:
        def deserialization_fn(data):
            image = data['image']
            min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
            image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
            image = tf.image.resize(image, (cfg['image_size'], cfg['image_size']), antialias=True)
            if is_train:
                image = tf.image.random_flip_left_right(image)
            image = tf.cast(image, tf.float32) / 255.0
            image = (image - 0.5) / 0.5
            return image, data['label']

        split = tfds.split_for_jax_process('train' if (is_train or debug_overfit) else cfg['test_split'], drop_remainder=True)
        dataset = tfds.load(cfg['tfds_name'], split=split)

    elif 'cifar' in dataset_name:
        def deserialization_fn(data):
            image = data['image']
            if is_train:
                image = tf.image.random_flip_left_right(image)
            image = tf.cast(image, tf.float32) / 255.0
            image = (image - 0.5) / 0.5
            return image, data['label']

        split = tfds.split_for_jax_process('train' if (is_train or debug_overfit) else cfg['test_split'], drop_remainder=True)
        dataset = tfds.load(cfg['tfds_name'], split=split)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if debug_overfit:
        dataset = dataset.take(global_batch)

    dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(global_batch, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    dataset = tfds.as_numpy(dataset)
    return iter(dataset)


def save_images(images, output_dir, start_idx):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        img = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(output_dir, f'{start_idx + i:06d}.png'))


def visualize_samples(images, labels, label_names=None, num_samples=16, save_path=None):
    num_samples = min(num_samples, len(images))
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i in range(num_samples):
        img = ((images[i] + 1) / 2).clip(0, 1)
        axes[i].imshow(img)
        axes[i].axis('off')
        if label_names is not None and labels is not None:
            axes[i].set_title(f'{label_names[labels[i]]}', fontsize=8)
        elif labels is not None:
            axes[i].set_title(f'Class {labels[i]}', fontsize=8)

    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    plt.show()
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main(_):
    np.random.seed(FLAGS.seed)

    print("\n" + "="*60)
    print("DiT FID Evaluation Script")
    print("="*60)
    print(f"Using devices: {jax.local_devices()}")

    device_count = jax.local_device_count()
    global_device_count = jax.device_count()
    print(f"Device count: {device_count}")
    print(f"Global device count: {global_device_count}")

    # local batch (host) derived from global batch
    local_batch_size = FLAGS.batch_size // (global_device_count // device_count)

    print(f"Global Batch: {FLAGS.batch_size}")
    print(f"Node Batch:   {local_batch_size}")
    print(f"Device Batch: {local_batch_size // device_count}")

    # dataset
    dataset_cfg = get_dataset_config(FLAGS.dataset_name)
    image_size = dataset_cfg['image_size']
    num_classes = dataset_cfg['num_classes']

    print(f"\nDataset: {FLAGS.dataset_name}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Num classes: {num_classes}")

    # per-device batch for pmapped model
    per_device_batch = local_batch_size // device_count
    assert per_device_batch > 0, "per_device_batch must be > 0"
    assert FLAGS.batch_size % global_device_count == 0, "--batch_size must be divisible by device_count"

    # example tensors (host local batch)
    example_obs = jnp.zeros((local_batch_size, image_size, image_size, 3))
    example_labels = jnp.zeros((local_batch_size,), dtype=jnp.int32)

    # Optional: Stable VAE
    vae = None
    vae_decode_pmap = None
    if FLAGS.use_stable_vae:
        vae = StableVAE.create()
        example_obs = vae.encode(jax.random.PRNGKey(0), example_obs)
        vae_decode_pmap = jax.pmap(vae.decode)
        print(f"Using Stable VAE, latent shape: {example_obs.shape}")

    # RNGs
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, param_key, dropout_key = jax.random.split(rng, 3)

    # Load checkpoint
    print("\n" + "-"*40)
    print("Loading Model")
    print("-"*40)

    assert FLAGS.load_dir is not None, "Must specify --load_dir"
    cp = Checkpoint(FLAGS.load_dir)
    cp_dict = cp.load_as_dict()

    if 'config' in cp_dict:
        model_config = cp_dict['config']
        print("Loaded config from checkpoint")
    else:
        raise ValueError("No config found in checkpoint")

    # Update config
    model_config.image_channels = example_obs.shape[-1]
    model_config.image_size = example_obs.shape[1]
    model_config['diffusion_timesteps'] = FLAGS.diffusion_timesteps

    dit_args = {
        'patch_size': model_config['patch_size'],
        'hidden_size': model_config['hidden_size'],
        'depth': model_config['depth'],
        'num_heads': model_config['num_heads'],
        'mlp_ratio': model_config['mlp_ratio'],
        'class_dropout_prob': model_config['class_dropout_prob'],
        'num_classes': model_config['num_classes'],
    }

    print("Model config:")
    for k, v in dit_args.items():
        print(f"  {k}: {v}")

    if dit_args['num_classes'] != num_classes:
        print(f"\nWARNING: Model num_classes ({dit_args['num_classes']}) != dataset num_classes ({num_classes})")

    model_def = DiT(**dit_args)

    # init model
    example_t = jnp.zeros((local_batch_size,))
    model_rngs = {'params': param_key, 'label_dropout': dropout_key}
    params = model_def.init(model_rngs, example_obs, example_t, example_labels)['params']

    tx = optax.adam(learning_rate=model_config['lr'], b1=model_config['beta1'], b2=model_config['beta2'])
    model_ts = TrainState.create(model_def, params, tx=tx)
    model_ts_eps = TrainState.create(model_def, params)

    scheduler = GaussianDiffusion(model_config['diffusion_timesteps'])
    model = DiffusionTrainer(rng, model_ts, model_ts_eps, model_config, scheduler)

    model = cp.load_model(model)
    print(f"Loaded model from step {model.model.step}")

    # replicate model across local devices
    model = flax.jax_utils.replicate(model, devices=jax.local_devices())

    # =========================================================
    # Setup FID
    # =========================================================
    print("\n" + "-"*40)
    print("Setting up FID computation")
    print("-"*40)

    get_fid_activations = get_fid_network()

    if FLAGS.fid_stats is None:
        FLAGS.fid_stats = f'data/{FLAGS.dataset_name}_fidstats.npz'
        print(f"Using default FID stats path: {FLAGS.fid_stats}")

    if os.path.exists(FLAGS.fid_stats):
        truth_fid_stats = dict(np.load(FLAGS.fid_stats))
        print(f"Loaded existing FID stats from {FLAGS.fid_stats}")
    else:
        print(f"FID stats not found at {FLAGS.fid_stats}")
        truth_fid_stats = compute_reference_fid_stats(
            FLAGS.dataset_name,
            per_device_batch,
            get_fid_activations,
            FLAGS.fid_stats
        )
        print("Reference stats computed and saved!")

    # =========================================================
    # Setup generation
    # =========================================================
    print("\n" + "-"*40)
    print("Setting up generation")
    print("-"*40)

    dataset = get_dataset(FLAGS.dataset_name, per_device_batch, is_train=True, debug_overfit=FLAGS.debug_overfit)

    # reshape obs for pmap: [n_devices, per_device_batch, ...]
    example_obs = example_obs.reshape((device_count, -1, *example_obs.shape[1:]))

    # ✅ per-device RNGs for pmapped denoise_step:
    # rng_dev shape: [n_devices, 2]
    rng_dev = jax.random.PRNGKey(FLAGS.seed + jax.process_index())
    rng_dev = jax.random.split(rng_dev, device_count)

    # =========================================================
    # Generate samples + compute activations
    # =========================================================
    print("\n" + "-"*40)
    print("Generating Samples")
    print("-"*40)
    print(f"Total samples to generate: {FLAGS.num_samples}")
    print(f"CFG weight: {FLAGS.cfg_weight}")
    print(f"CFG enabled: {bool(FLAGS.use_cfg)}")
    print(f"Diffusion timesteps: {FLAGS.diffusion_timesteps}")

    activations = []
    total_generated = 0

    num_iters = FLAGS.num_samples // FLAGS.batch_size

    # For sampling noise (host-level key is OK here)
    host_key = jax.random.PRNGKey(FLAGS.seed + 12345 + jax.process_index())

    for i in tqdm.tqdm(range(num_iters), desc="Generating samples"):
        host_key, noise_key = jax.random.split(host_key)

        # noise: [n_devices, per_device_batch, ...]
        x = jax.random.normal(noise_key, example_obs.shape)

        # labels from dataset: global_batch then reshape
        _, labels = next(dataset)
        labels = labels.reshape((device_count, -1))

        # Reverse diffusion
        for ti in range(FLAGS.diffusion_timesteps):
            # ✅ split per-device RNG each step:
            rng_pair = jax.vmap(jax.random.split)(rng_dev)  # [n_devices, 2, 2]
            rng_dev = rng_pair[:, 0, :]                     # next state rng
            step_rng = rng_pair[:, 1, :]                    # rng used this step

            t = jnp.full((x.shape[0], x.shape[1]), FLAGS.diffusion_timesteps - ti)
            cfg_weight_array = jnp.full((x.shape[0],), FLAGS.cfg_weight)

            if FLAGS.use_cfg:
                x = model.denoise_step(x, t, labels, step_rng, cfg_weight_array)
            else:
                x = model.denoise_step_no_cfg(x, t, labels, step_rng, cfg_weight_array * 0.0)

        # Decode VAE if used
        if FLAGS.use_stable_vae and vae_decode_pmap is not None:
            x = vae_decode_pmap(x)

        # Preview
        if i == 0:
            vis_images = np.array(x.reshape(-1, *x.shape[2:]))[:16]
            vis_labels = np.array(labels.reshape(-1))[:16]
            save_path = os.path.join(FLAGS.output_dir, 'samples_preview.png')
            visualize_samples(vis_images, vis_labels, num_samples=16, save_path=save_path)

        # Optionally save images
        if FLAGS.save_images:
            images_flat = np.array(x.reshape(-1, *x.shape[2:]))
            save_images(images_flat, os.path.join(FLAGS.output_dir, 'images'), total_generated)

        # FID activations
        x_resized = jax.image.resize(x, (x.shape[0], x.shape[1], 299, 299, 3), method='bilinear', antialias=False)
        x_resized = 2 * x_resized - 1  # keep as original

        acts = get_fid_activations(x_resized)[..., 0, 0, :]          # [n_devices, per_device_batch, 2048]
        acts = np.array(acts).reshape((-1, acts.shape[-1]))          # [global_batch, 2048]
        activations.append(acts)

        total_generated += FLAGS.batch_size

    # =========================================================
    # Compute FID
    # =========================================================
    print("\n" + "-"*40)
    print("Computing FID Score")
    print("-"*40)

    activations = np.concatenate(activations, axis=0)
    mu_gen = np.mean(activations, axis=0)
    sigma_gen = np.cov(activations, rowvar=False)

    gen_stats_path = os.path.join(FLAGS.output_dir, f'{FLAGS.dataset_name}_gen_fidstats.npz')
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    np.savez(gen_stats_path, mu=mu_gen, sigma=sigma_gen)
    print(f"Saved generated FID stats to {gen_stats_path}")

    fid = fid_from_stats(mu_gen, sigma_gen, truth_fid_stats['mu'], truth_fid_stats['sigma'])

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Dataset:            {FLAGS.dataset_name}")
    print(f"Checkpoint:         {FLAGS.load_dir}")
    print(f"Model step:         {int(flax.jax_utils.unreplicate(model.model.step))}")
    print(f"Num samples:        {total_generated}")
    print(f"CFG weight:         {FLAGS.cfg_weight}")
    print(f"Diffusion steps:    {FLAGS.diffusion_timesteps}")
    print("-"*60)
    print(f"FID Score:          {fid:.4f}")
    print("="*60 + "\n")


if __name__ == '__main__':
    app.run(main)
