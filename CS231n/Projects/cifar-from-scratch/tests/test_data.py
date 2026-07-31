"""Smoke tests for the data module: shapes, dtypes, normalization sanity."""

import torch

from cifar.data import get_dataloaders, load_cifar10


CFG = {
    "cifar_dir": None,  # use the default cached location
    "mean": [0.4914, 0.4822, 0.4465],
    "std": [0.2470, 0.2435, 0.2616],
    "batch_size": 64,
}


def test_load_cifar10_shapes():
    X_train, y_train, X_test, y_test = load_cifar10()
    assert X_train.shape == (50000, 3072)
    assert y_train.shape == (50000,)
    assert X_test.shape == (10000, 3072)
    assert y_test.shape == (10000,)
    assert X_train.dtype == X_test.dtype == y_train.dtype == y_test.dtype


def test_dataloader_batch():
    train_loader, _ = get_dataloaders(CFG)
    x, y = next(iter(train_loader))
    assert tuple(x.shape) == (64, 3, 32, 32)
    assert x.dtype == torch.float32
    assert tuple(y.shape) == (64,)
    assert torch.isfinite(x).all()
    # per-channel normalized -> spread around 0, not all zero
    assert x.std().item() > 0
