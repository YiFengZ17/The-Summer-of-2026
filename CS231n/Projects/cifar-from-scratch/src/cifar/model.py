"""Models for CIFAR-10.

M1: linear classifier / two-layer net (nn.Module, autograd handles layer backward).
M2: SmallCNN.
"""
# This part can be done by using PyTorch API directly, model = nn.Sequential()
import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """Single linear layer: flatten the image, linear map to class scores."""

    def __init__(self, input_dim=3072, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))   # (N,3,32,32) -> (N,3072) -> (N,C)


class TwoLayerNet(nn.Module):
    """Linear -> ReLU -> Linear (the A1-style 2-layer net, in proper PyTorch)."""

    def __init__(self, input_dim=3072, hidden_dim=100, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc2(torch.relu(self.fc1(x)))


class SmallCNN(nn.Module):
    """A small VGG-style CNN for CIFAR-10 (32x32x3).

    Three conv stages (Conv-BN-ReLU-MaxPool) shrink the map 32 -> 16 -> 8 -> 4,
    then an FC head with dropout. With augmentation + weight decay this reaches
    ~72-76% in ~30 epochs. (Autograd handles every layer's backward here.)
    """

    def __init__(self, num_classes=10, base_ch=32, dropout=0.5):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4   # e.g. 32, 64, 128
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 16->8
            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 8->4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4 * 4, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(cfg):
    """Factory: pick the model from cfg['model']."""
    name = cfg.get("model", "two-layer")
    input_dim = cfg.get("channels", 3) * cfg.get("image_size", 32) ** 2
    num_classes = cfg.get("num_classes", 10)
    if name == "linear":
        return LinearClassifier(input_dim, num_classes)
    if name in ("two-layer", "two_layer"):
        return TwoLayerNet(input_dim, cfg.get("hidden_dim", 100), num_classes)
    if name == "cnn":
        return SmallCNN(
            num_classes=num_classes,
            base_ch=cfg.get("base_channels", 32),
            dropout=cfg.get("dropout", 0.5),
        )
    raise ValueError(f"unknown model {name!r}")
