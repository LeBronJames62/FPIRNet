"""Final FPIR-Net settings. Run entry scripts from this directory.
Only file organization is borrowed from MITSTNet4; the FPIR-Net algorithm is not changed.
"""
from copy import deepcopy
import os

# Set before NumPy/PyTorch are imported by an entry script (Windows Conda).
for _key, _value in {
    "MKL_THREADING_LAYER": "SEQUENTIAL", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "USE_TORCH": "1",
}.items():
    os.environ.setdefault(_key, _value)

# Online synthesis from clean images: same final training protocol as the source.
CONFIG = {'seed': 42,
 'val_seed': 42,
 'deterministic': False,
 'model': {'base_channels': 20,
           'blocks': 1,
           'default_iterations': 4,
           'max_step': 0.5,
           'fixed_step': 0.1268},
 'data': {'train': {'roots': ['data/clean/train'],
                    'patch_size': 192,
                    'max_images': 5000,
                    'repeat': 1,
                    'subset_seed': 42,
                    'augment': True,
                    'batch_size': 1,
                    'num_workers': 2,
                    'pin_memory': True,
                    'persistent_workers': False,
                    'prefetch_factor': 2},
          'val': {'roots': ['data/clean/val'],
                  'patch_size': 192,
                  'max_images': 200,
                  'subset_seed': 2027,
                  'augment': False,
                  'batch_size': 1,
                  'num_workers': 1,
                  'pin_memory': True,
                  'persistent_workers': False,
                  'max_batches': 200}},
 'simulation': {'max_tilt': 8.0,
                'a_min': 0.15,
                'a_max': 11.0,
                'rho_max': 0.75,
                'isotropic_blur': False,
                'a_spatial_modulation': 0.3,
                'otf_window': 28,
                'otf_stride': 14,
                'noise_std_max': 0.0,
                'scintillation': 0.0,
                'color_jitter': 0.0,
                'residual_blur_prob': 0.0},
 'loss': {'lambda_image': 1.0, 'lambda_ssim': 0.1},
 'train': {'max_steps': 5000,
           'epochs': 6,
           'accumulation_steps': 4,
           'learning_rate': 0.0002,
           'warmup_steps': 250,
           'min_lr_ratio': 0.05,
           'weight_decay': 0.0001,
           'precision': 'bf16',
           'clip_grad_norm': 1.0,
           'log_interval': 25,
           'ema': True,
           'ema_decay': 0.999}}

CHECKPOINT = "ckpt/best.pth"
INFERENCE = {
    "input_root": "data/test",
    "output_root": "results/restored",
    "device": "cuda",
    "precision": "bf16",
    "iterations": 4,
    "tile": 384,
    "overlap": 64,
}


def get_config():
    return deepcopy(CONFIG)
