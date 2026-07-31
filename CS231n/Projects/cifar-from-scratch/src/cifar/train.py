"""Training loop + CLI.

The loop wires together everything hand-written in M1:

    forward   logits = model(x)
    loss      loss = softmax_ce(logits, y)   # our autograd.Function
    backward  loss.backward()                # autograd fills each p.grad
    step      optimizer.step()                # our SGD / Momentum / Adam
    zero_grad optimizer.zero_grad()           # torch ACCUMULATES grads -> clear each step

M3 additions: checkpoint (best + last) with resume, CSV logging, CLI overrides.

Run:    python -m cifar.train --config configs/baseline.yaml
Resume: python -m cifar.train --resume checkpoints/two-layer-adam-last.pt
"""

import argparse
import csv
import os
import time

import torch

from cifar.data import get_dataloaders
from cifar.evaluate import evaluate
from cifar.losses import softmax_ce
from cifar.model import build_model
from cifar.optim import SGD, SGDMomentum, Adam
from cifar.utils import load_config, seed_everything


def build_optimizer(cfg, params):
    name = cfg.get("optimizer", "adam")
    lr, wd = cfg["learning_rate"], cfg.get("weight_decay", 0.0)
    if name == "sgd":
        return SGD(params, lr=lr, weight_decay=wd)
    if name == "momentum":
        return SGDMomentum(params, lr=lr, momentum=cfg.get("momentum", 0.9), weight_decay=wd)
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=wd)
    raise ValueError(f"unknown optimizer {name!r}")


def train_epoch(model, loader, loss_fn, optimizer, device):
    """One pass over the training set: forward -> scores -> loss -> backward
    -> gradient descent step -> update running loss/accuracy."""
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                       # forward
        loss = loss_fn(logits, y)               # our softmax-CE
        optimizer.zero_grad()                   # clear last step's grads
        loss.backward()                         # populate p.grad via autograd
        optimizer.step()                        # our update rule on p.data
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)
    return loss_sum / n, correct / n


def save_checkpoint(path, model, optimizer, epoch, best_acc, cfg):
    """Save model + optimizer state so a run can resume exactly."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {"epoch": epoch, "model": model.state_dict(),
         "optimizer": optimizer.state_dict(), "best_acc": best_acc, "cfg": cfg},
        path,
    )


def train(cfg, config_path=None, resume=None):
    device = cfg.get("device", "cpu")
    seed_everything(cfg.get("seed", 0))

    train_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(cfg, model.parameters())

    epochs = cfg.get("epochs", 10)
    # Resolve io dirs relative to the PROJECT ROOT (derived from this file's
    # location), so checkpoints/logs land in the project regardless of the
    # current working directory -- a common "where did my files go?" gotcha.
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints/")
    log_dir = cfg.get("log_dir", "logs/")
    if not os.path.isabs(ckpt_dir): ckpt_dir = os.path.join(proj_root, ckpt_dir)
    if not os.path.isabs(log_dir):  log_dir = os.path.join(proj_root, log_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    run_tag = f"{cfg.get('model', 'model')}-{cfg.get('optimizer', 'adam')}"
    csv_path = os.path.join(log_dir, f"{run_tag}.csv")

    # Resume: restore weights, optimizer buffers, and where we left off.
    start_epoch, best_acc = 1, 0.0
    if resume and os.path.isfile(resume):
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        best_acc = ck.get("best_acc", 0.0)
        print(f"[resume] from {resume} | restart at epoch {start_epoch} | best_acc={best_acc:.3f}")

    print(f"model={cfg.get('model')}  optimizer={cfg.get('optimizer', 'adam')}  "
          f"lr={cfg['learning_rate']}  wd={cfg.get('weight_decay', 0.0)}  "
          f"epochs={epochs}  device={device}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    csv_file = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_file)
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        csv_w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "time_s"])

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, softmax_ce, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, softmax_ce, device)
        dt = time.time() - t0
        print(f"epoch {epoch:2d}/{epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {te_loss:.4f} acc {te_acc:.3f} | {dt:.1f}s")
        csv_w.writerow([epoch, f"{tr_loss:.4f}", f"{tr_acc:.4f}",
                        f"{te_loss:.4f}", f"{te_acc:.4f}", f"{dt:.2f}"])
        csv_file.flush()

        # Checkpoint: best (by val acc) for inference + last (for resume).
        if te_acc > best_acc:
            best_acc = te_acc
            save_checkpoint(os.path.join(ckpt_dir, f"{run_tag}-best.pt"),
                            model, optimizer, epoch, best_acc, cfg)
        save_checkpoint(os.path.join(ckpt_dir, f"{run_tag}-last.pt"),
                        model, optimizer, epoch, best_acc, cfg)

    csv_file.close()
    print(f"done. best val acc = {best_acc:.4f}  | checkpoints: {ckpt_dir} | log: {csv_path}")
    return model


def main():
    ap = argparse.ArgumentParser(description="Train CIFAR-10 from scratch.")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override cfg.epochs")
    ap.add_argument("--device", default=None, help="override cfg.device (cpu/cuda)")
    ap.add_argument("--model", default=None, help="override cfg.model")
    ap.add_argument("--optimizer", default=None, help="override cfg.optimizer")
    ap.add_argument("--resume", default=None, help="checkpoint path to resume from")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None: cfg["epochs"] = args.epochs
    if args.device: cfg["device"] = args.device
    if args.model: cfg["model"] = args.model
    if args.optimizer: cfg["optimizer"] = args.optimizer
    train(cfg, args.config, resume=args.resume)


if __name__ == "__main__":
    main()
