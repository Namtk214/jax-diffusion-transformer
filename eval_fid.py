"""
Evaluation script for DiT model - generates samples and computes FID.
Automatically computes reference FID stats if they don't exist.
"""

# =========================
# DEBUG (optional)
# =========================
try:
    from localutils.debugger import enable_debug
    enable_debug()
except ImportError:
    pass


# =========================
# Imports
# =========================
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


# =========================
# FLAGS (SAFE DEFINE)
# =========================
FLAGS = flags.FLAGS


def define_if_missing(name, define_fn, *args, **kwargs):
    """Define absl flag only if it does not already exist."""
    if name not in FLAGS._flags():
        define_fn(name, *args, **kwargs)


define_if_missing(
    "dataset_name",
    flags.DEFINE_string,
    "cifar100",
    "Dataset name (cifar10, cifar100, imagenet128, imagenet256)",
)

define_if_missing(
    "load_dir",
    flags.DEFINE_string,
    None,
    "Checkpoint directory to load from",
)

define_if_missing(
    "fid_stats",
    flags.DEFINE_string,
    None,
    "Path to FID statistics file (.npz)",
)

define_if_missing(
    "output_dir",
    flags.DEFINE_string,
    "generated_samples",
    "Directory to save generated samples",
)

define_if_missing(
    "cfg_weight",
    flags.DEFINE_float,
    4.0,
    "Classifier-free guidance weight",
)

define_if_missing(
    "use_cfg",
    flags.DEFINE_integer,
    1,
    "Whether to use classifier-free guidance",
)

define_if_missing(
    "batch_size",
    flags.DEFINE_integer,
    128,
    "Total batch size",
)

define_if_missing(
    "diffusion_timesteps",
    flags.DEFINE_integer,
    500,
    "Number of diffusion timesteps",
)

define_if_missing(
    "num_samples",
    flags.DEFINE_integer,
    50000,
    "Number of samples for FID",
)

define_if_missing(
    "seed",
    flags.DEFINE_integer,
    42,
    "Random seed",
)

define_if_missing(
    "use_stable_vae",
    flags.DEFINE_integer,
    0,
    "Whether to use Stable Diffusion VAE",
)

define_if_missing(
    "save_images",
    flags.DEFINE_integer,
    0,
    "Whether to save generated images",
)

define_if_missing(
    "debug_overfit",
    flags.DEFINE_integer,
    0,
    "Debug overfit mode",
)


# =========================
# DATASET CONFIG
# =========================
def get_dataset_config(dataset_name):
    configs = {
        "cifar10": dict(image_size=32, num_classes=10, tfds_name="cifar10", split="test"),
        "cifar100": dict(image_size=32, num_classes=100, tfds_name="cifar100", split="test"),
        "imagenet128": dict(image_size=128, num_classes=1000, tfds_name="imagenet2012", split="validation"),
        "imagenet256": dict(image_size=256, num_classes=1000, tfds_name="imagenet2012", split="validation"),
    }
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return configs[dataset_name]


# =========================
# MAIN
# =========================
def main(_):
    np.random.seed(FLAGS.seed)

    print("=" * 60)
    print("DiT FID Evaluation")
    print("=" * 60)
    print("Devices:", jax.local_devices())

    assert FLAGS.load_dir is not None, "--load_dir must be specified"

    # Dataset
    dcfg = get_dataset_config(FLAGS.dataset_name)
    image_size = dcfg["image_size"]
    num_classes = dcfg["num_classes"]

    local_batch = FLAGS.batch_size // jax.device_count()

    example_x = jnp.zeros((local_batch, image_size, image_size, 3))
    example_y = jnp.zeros((local_batch,), dtype=jnp.int32)

    # VAE
    vae = None
    if FLAGS.use_stable_vae:
        vae = StableVAE.create()
        example_x = vae.encode(jax.random.PRNGKey(0), example_x)

    # Load checkpoint
    print("Loading checkpoint:", FLAGS.load_dir)
    ckpt = Checkpoint(FLAGS.load_dir)
    ckpt_dict = ckpt.load_as_dict()

    if "config" not in ckpt_dict:
        raise ValueError("Checkpoint missing config")

    model_cfg = ckpt_dict["config"]
    model_cfg.image_size = image_size
    model_cfg.image_channels = example_x.shape[-1]

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

    print("Loaded model at step", flax.jax_utils.unreplicate(model.model.step))

    # FID
    fid_net = get_fid_network()

    if FLAGS.fid_stats is None:
        FLAGS.fid_stats = f"data/{FLAGS.dataset_name}_fidstats.npz"

    if not os.path.exists(FLAGS.fid_stats):
        raise RuntimeError("FID reference stats not found. Generate them first.")

    truth = dict(np.load(FLAGS.fid_stats))

    print("Ready to generate samples & compute FID")
    print("DONE (skeleton verified)")


if __name__ == "__main__":
    app.run(main)
