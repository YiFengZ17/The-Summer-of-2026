"""Evaluation metrics. Runs under torch.no_grad (no graph, no grad)."""

import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy for one batch."""
    return (logits.argmax(dim=1) == labels).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, loss_fn, device="cpu"):
    """Average (loss, accuracy) over a whole loader."""
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        # loss_fn averages over the batch, so weight by batch size for the mean
        loss_sum += loss_fn(logits, y).item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)
    model.train()
    return loss_sum / n, correct / n
