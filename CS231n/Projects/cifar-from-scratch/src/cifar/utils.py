"""Small shared utilities: reproducibility and config loading."""

import random

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA) for reproducible runs.

    Also forces deterministic cuDNN (slightly slower, but removes nondeterminism
    from conv kernels) so two runs with the same seed give the same result.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    """Load a YAML config file into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
