# CIFAR-10 from scratch (CS231n Final Project)

A from-scratch **CIFAR-10 image classifier**, built for two goals:
- **(A) Project anatomy** — the structure of a standard ML project
  (config / package / data module / training engine / eval / checkpoint+logging / tests / packaging).
- **(B) Concept mastery** — hand-write the **loss** (as `torch.autograd.Function`),
  the **optimizers** (SGD / Momentum / Adam), and the **training loop**, so every
  line is understood. PyTorch is the framework; the core is **not** a black box.

> Status: ✅ M0–M3 complete. (Full CNN accuracy run best done on GPU.)

## What's hand-written (and why)
| File | Hand-written core | Concept it nails down |
|---|---|---|
| `losses.py` | `SoftmaxCrossEntropy` forward (stable log-sum-exp) + **backward** (`(probs − onehot)/N`) | softmax-CE gradient; verified by `gradcheck` |
| `optim.py` | `SGD` / `SGDMomentum` / `Adam` `step()` (read `.grad`, update `.data`) | update rules + Adam bias correction; verified bit-exact vs `torch.optim` |
| `train.py` | `train_epoch` loop: forward → loss → backward → step → zero_grad | the training mechanism; why `zero_grad` is needed |

Everything else (conv/BN/dropout layers, data loading) uses `torch`/`torchvision` — the
hand-writing is spent where the concepts live, not on boilerplate.

## Project structure
```
cifar-from-scratch/
├── configs/         YAML configs (baseline = two-layer, cnn = conv experiment)
├── src/cifar/
│   ├── data.py      CIFAR-10 Dataset / DataLoader / normalization / augmentation
│   ├── model.py     LinearClassifier / TwoLayerNet / SmallCNN
│   ├── losses.py    ★ hand-written SoftmaxCrossEntropy (autograd.Function)
│   ├── optim.py     ★ hand-written SGD / Momentum / Adam (+ state_dict for resume)
│   ├── train.py     ★ hand-written training loop + checkpoint/CSV/CLI
│   └── evaluate.py  accuracy / metrics
├── tests/           pytest: loss gradcheck, optimizer bit-exact match, model shapes, data
├── docs/ARCHITECTURE.md   full module/function map + data flow
├── notebooks/       exploration only (no core logic)
├── checkpoints/ logs/   run artifacts (gitignored)
├── pyproject.toml   packaging + deps
└── requirements.txt
```

## Setup
```bash
pip install -e ".[dev]"
```
CIFAR-10 is read from the cached `cifar-10-batches-py` (CS231n assignment1 dir) — **no download**.

## Train
```bash
# two-layer net (M1)
python -m cifar.train --config configs/baseline.yaml

# small CNN (M2)
python -m cifar.train --config configs/cnn.yaml

# CLI overrides (no need to edit the YAML):
python -m cifar.train --config configs/cnn.yaml --epochs 50 --device cuda --optimizer momentum

# resume from the last checkpoint
python -m cifar.train --config configs/cnn.yaml --resume checkpoints/cnn-adam-last.pt
```
Each run writes `checkpoints/<run>-best.pt` (best val acc) + `<run>-last.pt` (resumable)
and appends per-epoch metrics to `logs/<run>.csv`.

## Tests
```bash
pytest -q
```
Covers: `SoftmaxCrossEntropy` gradcheck + value match, each optimizer bit-exact vs
`torch.optim`, model output shapes, data shapes/normalization.

## Results (observed)
| Model | Val acc | Notes |
|---|---|---|
| Two-layer net | ~51% | 20 ep, Adam, no augmentation |
| SmallCNN | **74.5%** (best, epoch 9) | 10 ep, Adam lr1e-3 wd5e-4, conv+BN+dropout+augment, CPU ~29 s/ep |

CNN val acc per epoch: `53.6 → 60.6 → 60.7 → 66.7 → 64.3 → 68.8 → 71.6 → 73.0 → 74.5 → 71.8`.
Val loss rose at epoch 10 (0.73 → 0.81) — early overfitting, so the saved `cnn-adam-best.pt`
(epoch 9) is the checkpoint to use for inference. A longer run with lr decay would push higher.

## Concepts this project teaches
- **Softmax + cross-entropy** — forward (log-sum-exp for stability) and backward (grad = probs − onehot).
- **Optimizers** — SGD, momentum, Adam (with bias correction) update rules, written by hand.
- **Training loop** — forward → loss → backward → step → zero_grad.
- **Regularization** — L2 weight decay lives in the optimizer, not the loss (PyTorch convention).
- **CNN** — conv/pool/BN/dropout + data augmentation (random crop/flip).
- **Engineering** — config-driven experiments, checkpoint/resume, CSV logging, tests, packaging.
