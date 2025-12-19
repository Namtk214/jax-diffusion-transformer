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

from typing import Any
import jax.numpy as jnp
from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import flax
import optax
from ml_collections import config_flags
import ml_collections
import tensorflow_datasets as tfds
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")
import matplotlib.pyplot as plt
import os
from PIL import Image

from utils.train_state import TrainState, target_update
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from schedulers import GaussianDiffusion
from diffusion_transformer import DiT
from train_diffusion import DiffusionTrainer
from utils.fid import get_fid_network, fid_from_stats

# Clear any existing flags that might conflict
# for flag_name in ['dataset_name', 'load_dir', 'batch_size', 'seed']:
#     if hasattr(flags.FLAGS, flag_name):
#         delattr(flags.FLAGS, flag_name)

FLAGS = flags.FLAGS
# flags.DEFINE_string('dataset_name', 'cifar100', 'Dataset name (cifar10, cifar100, imagenet128, imagenet256).')
flags.DEFINE_string('load_dir', None, 'Checkpoint directory to load from.')
flags.DEFINE_string('fid_stats', None, 'Path to FID statistics file (.npz). Auto-computed if not exists.')
flags.DEFINE_string('output_dir', 'generated_samples', 'Directory to save generated samples.')
flags.DEFINE_float('cfg_weight', 4.0, 'Classifier-free guidance weight.')
flags.DEFINE_integer('use_cfg', 1, 'Whether to use classifier-free guidance.')
flags.DEFINE_integer('batch_size', 128, 'Total batch size.')
flags.DEFINE_integer('diffusion_timesteps', 500, 'Number of diffusion timesteps.')
flags.DEFINE_integer('num_samples', 50000, 'Total number of samples to generate for FID.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('use_stable_vae', 0, 'Whether to use Stable Diffusion VAE.')
flags.DEFINE_integer('save_images', 0, 'Whether to save generated images to disk.')
flags.DEFINE_integer('debug_overfit', 0, 'Debug mode.')


def get_dataset_config(dataset_name):
    """Get dataset-specific configuration."""
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


def compute_reference_fid_stats(dataset_name, batch_size, get_fid_activations, output_path):
    """Compute and save FID reference statistics for real dataset."""
    print(f"\n{'='*60}")
    print(f"Computing reference FID stats for {dataset_name}...")
    print("This only needs to be done once and will be cached.")
    print(f"{'='*60}\n")
    
    config = get_dataset_config(dataset_name)
    total_samples = config['train_size']
    
    # Load dataset
    ds = tfds.load(config['tfds_name'], split='train')
    
    def preprocess(data):
        image = data['image']
        # Handle ImageNet cropping
        if 'imagenet' in dataset_name:
            min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
            image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        # Resize to 299x299 for InceptionV3
        image = tf.image.resize(image, (299, 299), antialias=True)
        image = 2 * image - 1  # Rescale for InceptionV3 [-1, 1]
        return image
    
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    all_acts = []
    num_processed = 0
    
    pbar = tqdm.tqdm(ds, desc="Computing reference stats", total=(total_samples + batch_size - 1) // batch_size)
    for batch in pbar:
        images = batch.numpy()
        # Add device dimension for get_fid_activations [1, batch, 299, 299, 3]
        images = images[None, ...]
        acts = get_fid_activations(images)[0, :, 0, 0, :]  # [batch, 2048]
        all_acts.append(np.array(acts))
        num_processed += acts.shape[0]
        pbar.set_postfix({'processed': num_processed})
    
    all_acts = np.concatenate(all_acts, axis=0)
    print(f"\nComputing statistics from {all_acts.shape[0]} samples...")
    
    mu = np.mean(all_acts, axis=0)
    sigma = np.cov(all_acts, rowvar=False)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    np.savez(output_path, mu=mu, sigma=sigma)
    print(f"Saved reference FID stats to {output_path}")
    print(f"Processed {num_processed} real images\n")
    
    return {'mu': mu, 'sigma': sigma}


def get_dataset(dataset_name, local_batch_size, is_train=False, debug_overfit=False):
    """Load dataset for label sampling."""
    config = get_dataset_config(dataset_name)
    
    if 'imagenet' in dataset_name:
        def deserialization_fn(data):
            image = data['image']
            min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
            image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
            image = tf.image.resize(image, (config['image_size'], config['image_size']), antialias=True)
            if is_train:
                image = tf.image.random_flip_left_right(image)
            image = tf.cast(image, tf.float32) / 255.0
            image = (image - 0.5) / 0.5
            return image, data['label']
        
        split = tfds.split_for_jax_process('train' if (is_train or debug_overfit) else config['test_split'], drop_remainder=True)
        dataset = tfds.load(config['tfds_name'], split=split)
    
    elif 'cifar' in dataset_name:
        def deserialization_fn(data):
            image = data['image']
            if is_train:
                image = tf.image.random_flip_left_right(image)
            image = tf.cast(image, tf.float32) / 255.0
            image = (image - 0.5) / 0.5
            return image, data['label']
        
        split = tfds.split_for_jax_process('train' if (is_train or debug_overfit) else config['test_split'], drop_remainder=True)
        dataset = tfds.load(config['tfds_name'], split=split)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    if debug_overfit:
        dataset = dataset.take(8)
    
    dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(local_batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = tfds.as_numpy(dataset)
    return iter(dataset)


def save_images(images, output_dir, start_idx):
    """Save generated images to disk."""
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        # Convert from [-1, 1] to [0, 255]
        img = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(output_dir, f'{start_idx + i:06d}.png'))


def visualize_samples(images, labels, label_names=None, num_samples=16, save_path=None):
    """Visualize a grid of generated samples."""
    num_samples = min(num_samples, len(images))
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        img = ((images[i] + 1) / 2).clip(0, 1)  # Convert from [-1,1] to [0,1]
        axes[i].imshow(img)
        axes[i].axis('off')
        if label_names is not None and labels is not None:
            axes[i].set_title(f'{label_names[labels[i]]}', fontsize=8)
        elif labels is not None:
            axes[i].set_title(f'Class {labels[i]}', fontsize=8)
    
    # Hide empty subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    plt.show()
    plt.close()


def main(_):
    # Set random seed
    np.random.seed(FLAGS.seed)
    
    # Device setup
    print("\n" + "="*60)
    print("DiT FID Evaluation Script")
    print("="*60)
    print(f"Using devices: {jax.local_devices()}")
    device_count = len(jax.local_devices())
    global_device_count = jax.device_count()
    print(f"Device count: {device_count}")
    print(f"Global device count: {global_device_count}")
    
    local_batch_size = FLAGS.batch_size // (global_device_count // device_count)
    print(f"Global Batch: {FLAGS.batch_size}")
    print(f"Node Batch: {local_batch_size}")
    print(f"Device Batch: {local_batch_size // device_count}")
    
    # Get dataset configuration
    dataset_config = get_dataset_config(FLAGS.dataset_name)
    image_size = dataset_config['image_size']
    num_classes = dataset_config['num_classes']
    print(f"\nDataset: {FLAGS.dataset_name}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Num classes: {num_classes}")
    
    # Create example tensors
    example_obs = jnp.zeros((local_batch_size, image_size, image_size, 3))
    example_labels = jnp.zeros((local_batch_size,), dtype=jnp.int32)
    
    # Optional: Use Stable Diffusion VAE
    vae = None
    vae_decode_pmap = None
    if FLAGS.use_stable_vae:
        vae = StableVAE.create()
        example_obs = vae.encode(jax.random.PRNGKey(0), example_obs)
        vae_decode_pmap = jax.pmap(vae.decode)
        print(f"Using Stable VAE, latent shape: {example_obs.shape}")
    
    # Initialize RNG
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, param_key, dropout_key = jax.random.split(rng, 3)
    
    ###################################
    # Load Model from Checkpoint
    ###################################
    print("\n" + "-"*40)
    print("Loading Model")
    print("-"*40)
    
    assert FLAGS.load_dir is not None, "Must specify --load_dir"
    
    cp = Checkpoint(FLAGS.load_dir)
    cp_dict = cp.load_as_dict()
    
    # Load config from checkpoint if available
    if 'config' in cp_dict:
        model_config = cp_dict['config']
        print("Loaded config from checkpoint")
    else:
        raise ValueError("No config found in checkpoint")
    
    # Update config with current settings
    model_config.image_channels = example_obs.shape[-1]
    model_config.image_size = example_obs.shape[1]
    model_config['diffusion_timesteps'] = FLAGS.diffusion_timesteps
    
    # Build model
    dit_args = {
        'patch_size': model_config['patch_size'],
        'hidden_size': model_config['hidden_size'],
        'depth': model_config['depth'],
        'num_heads': model_config['num_heads'],
        'mlp_ratio': model_config['mlp_ratio'],
        'class_dropout_prob': model_config['class_dropout_prob'],
        'num_classes': model_config['num_classes'],
    }
    
    print(f"Model config:")
    for k, v in dit_args.items():
        print(f"  {k}: {v}")
    
    # Verify num_classes matches dataset
    if dit_args['num_classes'] != num_classes:
        print(f"\nWARNING: Model num_classes ({dit_args['num_classes']}) != dataset num_classes ({num_classes})")
    
    model_def = DiT(**dit_args)
    
    # Initialize model
    example_t = jnp.zeros((local_batch_size,))
    example_label = jnp.zeros((local_batch_size,), dtype=jnp.int32)
    model_rngs = {'params': param_key, 'label_dropout': dropout_key}
    params = model_def.init(model_rngs, example_obs, example_t, example_label)['params']
    
    tx = optax.adam(learning_rate=model_config['lr'], b1=model_config['beta1'], b2=model_config['beta2'])
    model_ts = TrainState.create(model_def, params, tx=tx)
    model_ts_eps = TrainState.create(model_def, params)
    
    scheduler = GaussianDiffusion(model_config['diffusion_timesteps'])
    model = DiffusionTrainer(rng, model_ts, model_ts_eps, model_config, scheduler)
    
    # Load checkpoint weights
    model = cp.load_model(model)
    print(f"Loaded model from step {model.model.step}")
    
    # Replicate model across devices
    model = flax.jax_utils.replicate(model, devices=jax.local_devices())
    
    ###################################
    # Setup FID - Auto-compute reference stats if needed
    ###################################
    print("\n" + "-"*40)
    print("Setting up FID computation")
    print("-"*40)
    
    get_fid_activations = get_fid_network()
    
    # Default FID stats path if not provided
    if FLAGS.fid_stats is None:
        FLAGS.fid_stats = f'data/{FLAGS.dataset_name}_fidstats.npz'
        print(f"Using default FID stats path: {FLAGS.fid_stats}")
    
    # Check if reference stats exist, compute if not
    if os.path.exists(FLAGS.fid_stats):
        truth_fid_stats = dict(np.load(FLAGS.fid_stats))
        print(f"Loaded existing FID stats from {FLAGS.fid_stats}")
    else:
        print(f"FID stats not found at {FLAGS.fid_stats}")
        truth_fid_stats = compute_reference_fid_stats(
            FLAGS.dataset_name,
            local_batch_size,
            get_fid_activations,
            FLAGS.fid_stats
        )
        print("Reference stats computed and saved!")
    
    ###################################
    # Setup for Generation
    ###################################
    print("\n" + "-"*40)
    print("Setting up generation")
    print("-"*40)
    
    # Load dataset for label sampling
    dataset = get_dataset(FLAGS.dataset_name, local_batch_size, is_train=True, debug_overfit=FLAGS.debug_overfit)
    
    # Reshape for pmap
    example_obs = example_obs.reshape((device_count, -1, *example_obs.shape[1:]))
    example_labels = example_labels.reshape((device_count, -1))
    
    vmap_split = jax.vmap(jax.random.split, in_axes=(0))
    
    # Create output directory
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    if FLAGS.save_images:
        print(f"Will save images to {FLAGS.output_dir}")
    
    ###################################
    # Generate Samples
    ###################################
    print("\n" + "-"*40)
    print("Generating Samples")
    print("-"*40)
    print(f"Total samples to generate: {FLAGS.num_samples}")
    print(f"CFG weight: {FLAGS.cfg_weight}")
    print(f"CFG enabled: {bool(FLAGS.use_cfg)}")
    print(f"Diffusion timesteps: {FLAGS.diffusion_timesteps}")
    
    activations = []
    all_labels = []
    key = jax.random.PRNGKey(FLAGS.seed + jax.process_index())
    
    num_iters = FLAGS.num_samples // FLAGS.batch_size
    total_generated = 0
    
    for i in tqdm.tqdm(range(num_iters), desc="Generating samples"):
        noise_key, iter_key, key = jax.random.split(key, 3)
        
        # Sample noise
        x = jax.random.normal(noise_key, example_obs.shape)
        
        # Get labels from dataset
        _, labels = next(dataset)
        labels = labels.reshape((device_count, -1))
        
        # Setup iteration key for each device
        iter_key = flax.jax_utils.replicate(iter_key, devices=jax.local_devices())
        iter_key += jnp.arange(device_count, dtype=jnp.uint32)[:, None] * 1000
        
        # Reverse diffusion process
        for ti in range(FLAGS.diffusion_timesteps):
            rng, iter_key = jnp.split(vmap_split(iter_key), 2, axis=-1)
            rng, iter_key = rng[..., 0], iter_key[..., 0]
            t = jnp.full((x.shape[0], x.shape[1]), FLAGS.diffusion_timesteps - ti)
            cfg_weight_array = jnp.full((x.shape[0],), FLAGS.cfg_weight)
            
            if FLAGS.use_cfg:
                x = model.denoise_step(x, t, labels, rng, cfg_weight_array)
            else:
                x = model.denoise_step_no_cfg(x, t, labels, rng, cfg_weight_array * 0.0)
        
        # Decode VAE if used
        if FLAGS.use_stable_vae and vae_decode_pmap is not None:
            x = vae_decode_pmap(x)
        
        # Save some images for visualization (first batch only)
        if i == 0:
            vis_images = np.array(x.reshape(-1, *x.shape[2:]))[:16]
            vis_labels = np.array(labels.reshape(-1))[:16]
            save_path = os.path.join(FLAGS.output_dir, 'samples_preview.png')
            visualize_samples(
                vis_images,
                vis_labels,
                num_samples=16,
                save_path=save_path
            )
        
        # Save images to disk if requested
        if FLAGS.save_images:
            images_flat = np.array(x.reshape(-1, *x.shape[2:]))
            save_images(images_flat, os.path.join(FLAGS.output_dir, 'images'), total_generated)
        
        # Compute FID activations
        x_resized = jax.image.resize(x, (x.shape[0], x.shape[1], 299, 299, 3), method='bilinear', antialias=False)
        x_resized = 2 * x_resized - 1  # Rescale for InceptionV3
        
        acts = get_fid_activations(x_resized)[..., 0, 0, :]
        acts = jax.pmap(lambda x: jax.lax.all_gather(x, 'i', axis=0), axis_name='i')(acts)[0]
        acts = np.array(acts)
        activations.append(acts)
        
        # Store labels
        all_labels.append(np.array(labels.reshape(-1)))
        
        total_generated += FLAGS.batch_size
    
    ###################################
    # Compute FID
    ###################################
    print("\n" + "-"*40)
    print("Computing FID Score")
    print("-"*40)
    
    activations = np.concatenate(activations, axis=0)
    activations = activations.reshape((-1, activations.shape[-1]))
    print(f"Total activations shape: {activations.shape}")
    
    # Compute FID statistics for generated samples
    mu_gen = np.mean(activations, axis=0)
    sigma_gen = np.cov(activations, rowvar=False)
    
    # Save generated statistics
    gen_stats_path = os.path.join(FLAGS.output_dir, f'{FLAGS.dataset_name}_gen_fidstats.npz')
    np.savez(gen_stats_path, mu=mu_gen, sigma=sigma_gen)
    print(f"Saved generated FID stats to {gen_stats_path}")
    
    # Compute FID
    fid = fid_from_stats(mu_gen, sigma_gen, truth_fid_stats['mu'], truth_fid_stats['sigma'])
    
    ###################################
    # Print and Save Results
    ###################################
    print("\n" + "="*60)
    print(f"RESULTS")
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
    
    # Save results to file
    results = {
        'fid': float(fid),
        'num_samples': total_generated,
        'cfg_weight': FLAGS.cfg_weight,
        'use_cfg': FLAGS.use_cfg,
        'diffusion_timesteps': FLAGS.diffusion_timesteps,
        'dataset': FLAGS.dataset_name,
        'checkpoint': FLAGS.load_dir,
        'model_step': int(flax.jax_utils.unreplicate(model.model.step)),
        'seed': FLAGS.seed,
    }
    
    results_path = os.path.join(FLAGS.output_dir, 'fid_results.txt')
    with open(results_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FID Evaluation Results\n")
        f.write("="*60 + "\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
        f.write("="*60 + "\n")
    print(f"Saved results to {results_path}")
    
    # Also save as npz for easy loading
    results_npz_path = os.path.join(FLAGS.output_dir, 'fid_results.npz')
    np.savez(results_npz_path, **results)
    print(f"Saved results to {results_npz_path}")
    
    print("\nDone!")
    
    return fid


if __name__ == '__main__':
    app.run(main)
