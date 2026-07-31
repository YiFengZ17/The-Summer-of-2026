"""CIFAR-10 data module.

Reads the cached ``cifar-10-batches-py`` (the format shipped with CS231n
assignment1), wraps it in a torch ``Dataset`` with per-channel normalization,
and builds train/test ``DataLoader``s. **No download.** M2 adds random
crop / flip augmentation to the TRAIN loader only.
"""

from __future__ import annotations

import os
import pickle
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


# Default location of the cached CIFAR-10 batches, resolved from this file so it
# works regardless of the current working directory.
#   <repo>/CS231n/Projects/cifar-from-scratch/src/cifar/data.py
#   -> up 4 levels -> <repo>/CS231n/Assignments/assignment1/cs231n/datasets/cifar-10-batches-py
_DEFAULT_CIFAR_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        *([ ".." ] * 4),
        "Assignments", "assignment1", "cs231n", "datasets", "cifar-10-batches-py",
    )
)


def load_cifar10(cifar_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw CIFAR-10 from the cached batches.

    Returns ``(X_train, y_train, X_test, y_test)`` where each ``X`` is ``uint8``
    of shape ``(N, 3072)`` in channel-first order (a 1024 R block, then G, then B),
    and each ``y`` is ``int64`` of shape ``(N,)``.
    """
    cifar_dir = cifar_dir or _DEFAULT_CIFAR_DIR
    if not os.path.isdir(cifar_dir):
        raise FileNotFoundError(
            f"CIFAR-10 batches not found at {cifar_dir!r}. Set `cifar_dir` in the "
            "config to the folder that contains data_batch_1..5 and test_batch."
        )

    def _load(fname: str) -> dict:
        with open(os.path.join(cifar_dir, fname), "rb") as f:
            # encoding="bytes" unpickles both the original Stanford format
            # (py2 -> byte keys b"data") and the HF-converted format used here
            # (py3 -> str keys "data"). _field handles either key spelling.
            return pickle.load(f, encoding="bytes")

    def _field(batch: dict, name: str):
        if name in batch:
            return batch[name]
        key = name.encode()
        if key in batch:
            return batch[key]
        raise KeyError(f"{name!r} not in batch keys {list(batch.keys())}")

    train_X, train_y = [], []
    for i in range(1, 6):
        batch = _load(f"data_batch_{i}")
        train_X.append(_field(batch, "data"))
        train_y.extend(_field(batch, "labels"))
    X_train = np.concatenate(train_X).astype(np.uint8)
    y_train = np.asarray(train_y, dtype=np.int64)

    test = _load("test_batch")
    X_test = _field(test, "data").astype(np.uint8)
    y_test = np.asarray(_field(test, "labels"), dtype=np.int64)
    return X_train, y_train, X_test, y_test


class CIFAR10Dataset(Dataset):
    """A torch ``Dataset`` over raw CIFAR-10 batches.

    Each image is normalized per-channel toward ~N(0,1). The optional
    ``transform`` is applied to the [0,1] tensor *before* normalization -- this is
    where random crop / flip is plugged in (train only).
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 mean, std, transform: Optional[Callable] = None):
        assert images.ndim == 2 and images.shape[1] == 3072, f"bad images shape {images.shape}"
        self.images = images
        self.labels = labels
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.as_tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # uint8 (3072,) -> float32 (3, 32, 32) in [0, 1]
        img = (
            torch.from_numpy(self.images[idx])
            .view(3, 32, 32)
            .to(torch.float32)
            .div_(255.0)
        )
        if self.transform is not None:
            img = self.transform(img)
        img = (img - self.mean) / self.std  # per-channel normalize
        return img, int(self.labels[idx])


def get_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """Build train/test DataLoaders from a config dict.

    If cfg['augment'] is true, the TRAIN loader gets random crop + horizontal flip
    (applied to the [0,1] image before normalization). The TEST loader never augments.
    """
    X_train, y_train, X_test, y_test = load_cifar10(cfg.get("cifar_dir"))
    mean, std = cfg["mean"], cfg["std"]
    num_workers = cfg.get("num_workers", 0)

    train_transform = None
    if cfg.get("augment", False):
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
        ])

    train_set = CIFAR10Dataset(X_train, y_train, mean, std, transform=train_transform)
    test_set = CIFAR10Dataset(X_test, y_test, mean, std)   # no augmentation at eval

    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=num_workers, drop_last=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader
