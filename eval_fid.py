"""
Evaluation script for DiT model - generates samples and computes FID.
Automatically computes reference FID stats if they don't exist.

Usage:
    python eval_fid.py --dataset_name=cifar100 --load_dir=checkpoints/cifar100
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
import matplotlib.pyplot as plt
from PIL import Image

tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")

from utils.train_state import TrainState
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from schedulers import GaussianDiffusion
from diffusion_transformer import DiT
from train_diffusion import DiffusionTrainer
from utils.fid import get_fid_network, fid_from_stats

# -----------------------
# FLAGS (safe)
# -----------------------
FLAGS = flags.FLAGS

def define_if_missing(name, define_fn, *args, **kwargs):
    if name not in FLAGS._flags():
        define_fn(name, *args, **kwargs)

# dataset_name is defined in train_diffusion.py → do not redefine here

define_if_missing('load_dir', flags.DEFINE_string, None, 'Checkpoint directory to load from.')
define_if_missing('fid_stats', flags.DEFINE_string, None, 'Path to FID statistics file (.npz). Auto-computed if not exists.')
define_if_missing('output_dir', flags.DEFINE_string, 'generated_samples', 'Directory to save generated samples.')
define_if_missing('cfg_weight', flags.DEFINE_float, 4.0, 'Classifier-free guidance weight.')
define_if_missing('use_cfg', flags.DEFINE_integer, 1, 'Whether to use classifier-free guidance.')
define_if_missing('batch_size', flags.DEFINE_integer, 128, 'Total (global) batch size.')
define_if_missing('diffusion_timesteps', flags.DEFINE_integer, 500, 'Number of diffusion timesteps.')
define_if_missing('num_samples', flags.DEFINE_integer, 50000, 'Total number of samples to generate for FID.')
define_if_missing('seed', flags.DEFINE_integer, 42, 'Random seed.')
define_if_missing('use_stable_vae', flags.DEFINE_integer, 0, 'Whether to use Stable Diffusion VAE.')
define_if_missing('save_images', flags.DEFINE_integer, 0, 'Whether to save generated images to disk.')
define_if_missing('debug_overfit', flags.DEFINE_integer, 0, 'Debug mode.')

# -----------------------
# Dataset config
# -----------------------
def get_dataset_config(dataset_name):
    configs = {
        'cifar10': dict(image_size=32, num_classes=10, tfds_name='cifar10', test_split='test', train_size=50000),
        'cifar100': dict(image_size=32, num_classes=100, tfds_name='cifar100', test_split='test', train_size=50000),
        'imagenet128': dict(image_size=128, num_classes=1000, tfds_name='imagenet2012', test_split='validation', train_size=1281167),
        'imagenet256': dict(image_size=256, num_classes=1000, tfds_name='imagenet2012', test_split='validation', train_size=1281167),
    }
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(configs.keys())}")
    return configs[dataset_name]

# -----------------------
# FID stats for real images (pmap-safe)
# -----------------------
def compute_reference_fid_stats(dataset_name, local_batch_size, get_fid_activations, output_path):
    print(f"\n{'='*60}")
    print(f"Computing reference FID stats for {dataset_name}...")
    print(f"{'='*60}\n")

    cfg = get_dataset_config(dataset_name)
    total_samples = cfg['train_size']

    n_devices = jax.local_device_count()
    global_batch = local_batch_size * n_devices

    ds = tfds.load(cfg['tfds_name'], split='train')

    def preprocess(data):
        image = data['image']
        if 'imagenet' in dataset_name:
            min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
            image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = tf.image.resize(image, (299, 299), antialias=True)
        image = 2 * image - 1
        return image

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(global_batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    all_acts = []
    num_processed = 0

    pbar = tqdm.tqdm(ds, desc="Real images", total=(total_samples // global_batch))
    for batch in pbar:
        images = batch.numpy()  # [global_batch, 299,299,3]

        # shard for pmap: [n_devices, local_batch, 299,299,3]
        images = images.reshape((n_devices, local_batch_size, 299, 299, 3))

        acts = get_fid_activations(images)              # [n_devices, local_batch, 1,1,2048] (typical)
        acts = np.array(acts)[..., 0, 0, :]             # [n_devices, local_batch, 2048]
        acts = acts.reshape((-1, acts.shape[-1]))       # [global_batch, 2048]

        all_acts.append(acts)
        num_processed += acts.shape[0]
        pbar.set_postfix({'processed': num_processed})

    all_acts = np.concatenate(all_acts, axis=0)
    mu = np.mean(all_acts, axis=0)
    sigma = np.cov(all_acts, rowvar=False)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(output_path, mu=mu, sigma=sigma)
    print(f"Saved reference FID stats to {output_path} ({num_processed} images)")
    return {'mu': mu, 'sigma': sigma}

# -----------------------
# Dataset for label sampling
# -----------------------
def get_dataset(dataset_name, local_batch_size, is_train=True, debug_overfit=False):
    cfg = get_dataset_config(dataset_name)
    n_devices = jax.local_device_count()
    global_batch = local_batch_size * n_devices

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

    else:
        def deserialization_fn(data):
            image = data['image']
            if is_train:
                image = tf.image.random_flip_left_right(image)
            image = tf.cast(image, tf.float32) / 255.0
            image = (image - 0.5) / 0.5
            return image, data['label']

        split = tfds.split_for_jax_process('train' if (is_train or debug_overfit) else cfg['test_split'], drop_remainder=True)
        dataset = tfds.load(cfg['tfds_name'], split=split)

    dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if debug_overfit:
        dataset = dataset.take(global_batch)

    dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
    dataset = dataset.repeat().batch(global_batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    dataset = tfds.as_numpy(dataset)
    return iter(dataset)

def save_images(images, output_dir, start_idx):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        img = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(output_dir, f'{start_idx + i:06d}.png'))

# -----------------------
# MAIN
# -----------------------
def main(_):
    np.random.seed(FLAGS.seed)

    print("\n" + "="*60)
    print("DiT FID Evaluation Script")
    print("="*60)
    print(f"Using devices: {jax.local_devices()}")

    device_count = jax.local_device_count()
    global_device_count = jax.device_count()

    # local batch per host = global_batch / global_device_count * device_count
    local_batch_size = FLAGS.batch_size // (global_device_count // device_count)

    print(f"Global Batch: {FLAGS.batch_size}")
    print(f"Node Batch:   {local_batch_size}")
    print(f"Device Batch: {local_batch_size // device_count}")

    dataset_cfg = get_dataset_config(FLAGS.dataset_name)
    image_size = dataset_cfg['image_size']
    num_classes = dataset_cfg['num_classes']

    example_obs = jnp.zeros((local_batch_size, image_size, image_size, 3))
    example_labels = jnp.zeros((local_batch_size,), dtype=jnp.int32)

    vae = None
    vae_decode_pmap = None
    if FLAGS.use_stable_vae:
        vae = StableVAE.create()
        example_obs = vae.encode(jax.random.PRNGKey(0), example_obs)
        vae_decode_pmap = jax.pmap(vae.decode)

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, param_key, dropout_key = jax.random.split(rng, 3)

    assert FLAGS.load_dir is not None, "Must specify --load_dir"

    cp = Checkpoint(FLAGS.load_dir)
    cp_dict = cp.load_as_dict()

    if 'config' not in cp_dict:
        raise ValueError("No config found in checkpoint")
    model_config = cp_dict['config']

    model_config.image_channels = example_obs.shape[-1]
    model_config.image_size = example_obs.shape[1]
    model_config['diffusion_timesteps'] = FLAGS.diffusion_timesteps

    dit_args = dict(
        patch_size=model_config['patch_size'],
        hidden_size=model_config['hidden_size'],
        depth=model_config['depth'],
        num_heads=model_config['num_heads'],
        mlp_ratio=model_config['mlp_ratio'],
        class_dropout_prob=model_config['class_dropout_prob'],
        num_classes=model_config['num_classes'],
    )

    if dit_args['num_classes'] != num_classes:
        print(f"\nWARNING: Model num_classes ({dit_args['num_classes']}) != dataset num_classes ({num_classes})")

    model_def = DiT(**dit_args)

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

    model = flax.jax_utils.replicate(model, devices=jax.local_devices())

    # FID network (likely pmapped)
    get_fid_activations = get_fid_network()

    if FLAGS.fid_stats is None:
        FLAGS.fid_stats = f'data/{FLAGS.dataset_name}_fidstats.npz'
        print(f"Using default FID stats path: {FLAGS.fid_stats}")

    if os.path.exists(FLAGS.fid_stats):
        truth_fid_stats = dict(np.load(FLAGS.fid_stats))
        print(f"Loaded existing FID stats from {FLAGS.fid_stats}")
    else:
        truth_fid_stats = compute_reference_fid_stats(
            FLAGS.dataset_name,
            local_batch_size // device_count,   # local per-device batch
            get_fid_activations,
            FLAGS.fid_stats
        )

    dataset = get_dataset(
        FLAGS.dataset_name,
        local_batch_size // device_count,   # per-device batch for label sampling
        is_train=True,
        debug_overfit=FLAGS.debug_overfit
    )

    # reshape example_obs for pmap
    example_obs = example_obs.reshape((device_count, -1, *example_obs.shape[1:]))

    key = jax.random.PRNGKey(FLAGS.seed + jax.process_index())

    num_iters = FLAGS.num_samples // FLAGS.batch_size
    total_generated = 0
    activations = []

    for i in tqdm.tqdm(range(num_iters), desc="Generating samples"):
        noise_key, iter_key, key = jax.random.split(key, 3)

        x = jax.random.normal(noise_key, example_obs.shape)

        _, labels = next(dataset)   # labels is global_batch
        labels = labels.reshape((device_count, -1))

        # reverse diffusion
        for ti in range(FLAGS.diffusion_timesteps):
            t = jnp.full((x.shape[0], x.shape[1]), FLAGS.diffusion_timesteps - ti)
            cfg_weight_array = jnp.full((x.shape[0],), FLAGS.cfg_weight)

            if FLAGS.use_cfg:
                x = model.denoise_step(x, t, labels, iter_key, cfg_weight_array)
            else:
                x = model.denoise_step_no_cfg(x, t, labels, iter_key, cfg_weight_array * 0.0)

        if FLAGS.use_stable_vae and vae_decode_pmap is not None:
            x = vae_decode_pmap(x)

        if FLAGS.save_images:
            images_flat = np.array(x.reshape(-1, *x.shape[2:]))
            save_images(images_flat, os.path.join(FLAGS.output_dir, 'images'), total_generated)

        # FID activations: resize to 299
        x_resized = jax.image.resize(x, (x.shape[0], x.shape[1], 299, 299, 3), method='bilinear', antialias=False)
        x_resized = 2 * x_resized - 1

        acts = get_fid_activations(x_resized)[..., 0, 0, :]
        acts = np.array(acts).reshape((-1, acts.shape[-1]))
        activations.append(acts)

        total_generated += FLAGS.batch_size

    activations = np.concatenate(activations, axis=0)
    mu_gen = np.mean(activations, axis=0)
    sigma_gen = np.cov(activations, rowvar=False)

    fid = fid_from_stats(mu_gen, sigma_gen, truth_fid_stats['mu'], truth_fid_stats['sigma'])

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Dataset: {FLAGS.dataset_name}")
    print(f"Checkpoint: {FLAGS.load_dir}")
    print(f"Num samples: {total_generated}")
    print(f"FID: {fid:.4f}")
    print("="*60 + "\n")

if __name__ == '__main__':
    app.run(main)
