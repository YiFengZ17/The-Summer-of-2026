"""Shape smoke tests for all models (build_model + forward pass)."""

import pytest
import torch

from cifar.model import build_model

BASE = {"channels": 3, "image_size": 32, "num_classes": 10}


@pytest.mark.parametrize("name", ["linear", "two-layer", "cnn"])
def test_forward_output_shape(name):
    cfg = {**BASE, "model": name, "hidden_dim": 100, "base_channels": 32, "dropout": 0.5}
    net = build_model(cfg)
    out = net(torch.randn(4, 3, 32, 32))   # batch of 4
    assert tuple(out.shape) == (4, 10)
