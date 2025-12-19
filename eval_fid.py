"""
Evaluation script for DiT model - generates samples and computes FID.
Automatically computes reference FID stats if they don't exist.

Usage:
    python eval_fid.py --dataset_name=cifar100 --load_dir=checkpoints/cifar100
"""

# ============================================================
# Optional debugger
# ============================================================
try:
    from localutils.debugger import enable_debug
    enable_debug()
except ImportError:
    pass


# ============================================================
# Imports
# ============================================================
from absl import app, flags
import os
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import flax
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
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


# ============================================================
# FLAGS (SAFE: no duplicate, no unparsed access)
# ============================================================
FLAGS = flags.FLAGS


def define_if_missing(name, define_fn, *args, **kwargs):
    if name not in FLAGS._flags():
        define_fn(name, *args, **kwargs)


# dataset_name đã được DEFINE trong train_diffusion.py
# → KHÔNG DEFINE lại ở đây

define_if_missing("load_dir", flags.DEFINE_string,
                  None, "Checkpoint directory to load from")

define_if_missing("fid_stats", flags.DEFINE_string,
                  None, "Path to FID statistics file (.npz)")

define_if_missing("output_dir", flags.DEFINE_string,
                  "generated_samples", "Directory to save generated samples")

define_if_missing("cfg_weight", flags.DEFINE_float,
                  4.0, "Classifier-free guidance weight")

define_if_missing("use_cfg", flags.DEFINE_integer,
                  1, "Whether to use classifier-free guidance")

define_if_missing("batch_size", flags.DEFINE_integer,
                  128, "Total batch size")

define_if_missing("diffusion_timesteps", flags.DEFINE_integer,
                  500, "Number of diffusion timesteps")

define_if_missing("num_samples", flags.DEFINE_integer,
                  50000, "Total number of samples for FID")

define_if_missing("seed", flags.DEFINE_integer,
                  42, "Random seed")

define_if_missing("use_stable_vae", flags.DEFINE_integer,
                  0, "Whether to use Stable Diffusion VAE")

define_if_missing("save_images", flags.DEFINE_integer,
                  0, "Whether to save generated images")

define_if_missing("debug_overfit", flags.DEFINE_integer,
                  0, "Debug mode")


# ============================================================
# Dataset helpers
# ============================================================
def get_dataset_config(dataset_name):
    configs = {
        "cifar10": {
            "image_size": 32,
            "num_classes": 10,
            "tfds_name": "cifar10",
            "test_split": "test",
            "train_size": 50000,
        },
        "cifar100": {
            "image_size": 32,
            "num_classes": 100,
            "tfds_name": "cifar100",
            "test_split": "test",
            "train_size": 50000,
        },
        "imagenet128": {
            "image_size": 128,
            "num_classes": 1000,
            "tfds_name": "imagenet2012",
            "test_split": "validation",
            "train_size": 1281167,
        },
        "imagenet256": {
            "image_size": 256,
            "num_classes": 1000,
            "tfds_name": "imagenet2012",
            "test_split": "validation",
            "train_size": 1281167,
        },
    }
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return configs[dataset_name]


def compute_reference_fid_stats(dataset_name, batch_size, get_fid_activations, output_path):
    print("=" * 60)
    print(f"Computing reference FID stats for {dataset_name}")
    print("=" * 60)

    cfg = get_dataset_config(dataset_name)
    ds = tfds.load(cfg["tfds_name"], split="train")

    def preprocess(data):
        image = tf.cast(data["image"], tf.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = tf.image.resize(image, (299, 299), antialias=True)
        image = 2 * image - 1
        return image

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    acts_all = []
    for batch in tqdm.tqdm(ds, desc="Real images"):
        batch = batch.numpy()[None, ...]
        acts = get_fid_activations(batch)[0, :, 0, 0, :]
        acts_all.append(np.array(acts))

    acts_all = np.concatenate(acts_all, axis=0)
    mu = acts_all.mean(axis=0)
    sigma = np.cov(acts_all, rowvar=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, mu=mu, sigma=sigma)
    print("Saved FID stats to", output_path)

    return {"mu": mu, "sigma": sigma}


def get_dataset(dataset_name, batch_size, is_train=True):
    cfg = get_dataset_config(dataset_name)
    split = "train" if is_train else cfg["test_split"]

    def preprocess(data):
        image = tf.cast(data["image"], tf.float32) / 255.0
        image = (image - 0.5) / 0.5
        return image, data["label"]

    ds = tfds.load(cfg["tfds_name"], split=split)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(10000).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return iter(tfds.as_numpy(ds))


def save_images(images, output_dir, start_idx):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        img = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(
            os.path.join(output_dir, f"{start_idx + i:06d}.png")
        )


# ============================================================
# MAIN
# ============================================================
def main(_):
    np.random.seed(FLAGS.seed)

    print("=" * 60)
    print("DiT FID Evaluation")
    print("=" * 60)
    print("Devices:", jax.local_devices())

    assert FLAGS.load_dir is not None, "--load_dir must be specified"

    cfg = get_dataset_config(FLAGS.dataset_name)
    image_size = cfg["image_size"]
    num_classes = cfg["num_classes"]

    device_count = jax.device_count()
    local_batch = FLAGS.batch_size // device_count

    example_x = jnp.zeros((local_batch, image_size, image_size, 3))
    example_y = jnp.zeros((local_batch,), dtype=jnp.int32)

    # VAE
    vae = None
    vae_decode = None
    if FLAGS.use_stable_vae:
        vae = StableVAE.create()
        example_x = vae.encode(jax.random.PRNGKey(0), example_x)
        vae_decode = jax.pmap(vae.decode)

    # Load checkpoint
    ckpt = Checkpoint(FLAGS.load_dir)
    ckpt_dict = ckpt.load_as_dict()
    model_cfg = ckpt_dict["config"]

    model_cfg.image_size = image_size
    model_cfg.image_channels = example_x.shape[-1]
    model_cfg.diffusion_timesteps = FLAGS.diffusion_timesteps

    model_def = DiT(
        patch_size=model_cfg.patch_size,
        hidden_size=model_cfg.hidden_size,
        depth=model_cfg.depth,
        num_heads=model_cfg.num_heads,
        mlp_ratio=model_cfg.mlp_ratio,
        class_dropout_prob=model_cfg.class_dropout_prob,
        num_classes=model_cfg.num_classes,
    )

    params = model_def.init(
        {"params": jax.random.PRNGKey(0)},
        example_x,
        jnp.zeros((local_batch,)),
        example_y,
    )["params"]

    tx = optax.adam(model_cfg.lr)
    ts = TrainState.create(model_def, params, tx=tx)
    ts_ema = TrainState.create(model_def, params)

    scheduler = GaussianDiffusion(FLAGS.diffusion_timesteps)
    model = DiffusionTrainer(jax.random.PRNGKey(FLAGS.seed), ts, ts_ema, model_cfg, scheduler)
    model = ckpt.load_model(model)
    model = flax.jax_utils.replicate(model)

    print("Loaded model step:",
          flax.jax_utils.unreplicate(model.model.step))

    # ========================================================
    # FID setup
    # ========================================================
    get_fid_activations = get_fid_network()

    if FLAGS.fid_stats is None:
        FLAGS.fid_stats = f"data/{FLAGS.dataset_name}_fidstats.npz"

    if os.path.exists(FLAGS.fid_stats):
        truth = dict(np.load(FLAGS.fid_stats))
        print("Loaded FID stats:", FLAGS.fid_stats)
    else:
        truth = compute_reference_fid_stats(
            FLAGS.dataset_name,
            local_batch,
            get_fid_activations,
            FLAGS.fid_stats,
        )

    # ========================================================
    # Generate samples
    # ========================================================
    dataset = get_dataset(FLAGS.dataset_name, local_batch)
    activations = []
    total_generated = 0

    key = jax.random.PRNGKey(FLAGS.seed)

    num_iters = FLAGS.num_samples // FLAGS.batch_size

    for _ in tqdm.tqdm(range(num_iters), desc="Generating samples"):
        key, noise_key = jax.random.split(key)
        x = jax.random.normal(noise_key, example_x.shape)

        _, labels = next(dataset)
        labels = labels.reshape((device_count, -1))

        for t in range(FLAGS.diffusion_timesteps):
            x = model.denoise_step(
                x,
                jnp.full((x.shape[0], x.shape[1]), FLAGS.diffusion_timesteps - t),
                labels,
                key,
                jnp.full((x.shape[0],), FLAGS.cfg_weight),
            )

        if vae_decode is not None:
            x = vae_decode(x)

        if FLAGS.save_images:
            imgs = np.array(x.reshape(-1, *x.shape[2:]))
            save_images(imgs, FLAGS.output_dir, total_generated)

        x = jax.image.resize(x, (x.shape[0], x.shape[1], 299, 299, 3), "bilinear")
        x = 2 * x - 1
        acts = get_fid_activations(x)[..., 0, 0, :]
        acts = np.array(acts.reshape(-1, acts.shape[-1]))
        activations.append(acts)

        total_generated += FLAGS.batch_size

    # ========================================================
    # Compute FID
    # ========================================================
    activations = np.concatenate(activations, axis=0)
    mu_gen = activations.mean(axis=0)
    sigma_gen = np.cov(activations, rowvar=False)

    fid = fid_from_stats(mu_gen, sigma_gen, truth["mu"], truth["sigma"])

    print("=" * 60)
    print("FID RESULT")
    print("=" * 60)
    print("FID:", fid)
    print("=" * 60)


if __name__ == "__main__":
    app.run(main)
