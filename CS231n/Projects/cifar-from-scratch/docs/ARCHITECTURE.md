# cifar-from-scratch — 项目架构 / 大纲

> 本文件 = 项目的"地图 + 契约"。**先把每个文件的函数签名固定下来**，实现时按签名填肉。
> 状态图例：✅ 已实现　🔨 M1 待写　🔜 M2　🔧 M3　✍️ = 手写/Socratic（必须能讲清）　⚙️ = 样板

---

## 1. 目录结构

```
cifar-from-scratch/
├── configs/baseline.yaml        ✅  超参 config（数据/模型/loss/optim/训练/IO）
├── src/cifar/                   ✅  可 import、可单测的包（src-layout）
│   ├── __init__.py              ✅
│   ├── utils.py                 ✅  seed + config 加载
│   ├── data.py                  ✅  CIFAR-10 读取 + Dataset + DataLoader
│   ├── losses.py                🔨  ✍️ 手写 SoftmaxCrossEntropy (autograd.Function)
│   ├── optim.py                 🔨  ✍️ 手写 SGD / Momentum / Adam
│   ├── model.py                 🔨  模型（Linear / TwoLayer → CNN）
│   ├── evaluate.py              🔨  准确率 / 评测
│   └── train.py                 🔨  ✍️ 手写训练循环 + CLI 入口
├── tests/
│   ├── test_data.py             ✅  形状/标准化
│   ├── test_losses.py           🔨  gradcheck 验证 loss backward
│   └── test_optim.py            🔧  一步更新 sanity（可选）
├── docs/ARCHITECTURE.md         ✅  本文件
├── notebooks/                   ⚙️  探索，不承载核心逻辑
├── checkpoints/ logs/           ⚙️  产物（gitignore）
├── pyproject.toml  requirements.txt  README.md  .gitignore   ✅
└── scripts/                     🔧  一次性脚本（M3 视需要）
```

---

## 2. 函数级地图（file → 符号 → 签名 → 职责 → 状态）

### `src/cifar/utils.py` ✅
| 符号 | 签名 | 职责 | 状态 |
|---|---|---|---|
| `seed_everything` | `(seed:int)->None` | 固定 random/numpy/torch + `cudnn.deterministic` | ✅ ⚙️ |
| `load_config` | `(path:str)->dict` | YAML → dict | ✅ ⚙️ |

### `src/cifar/data.py` ✅
| 符号 | 签名 | 职责 | 状态 |
|---|---|---|---|
| `load_cifar10` | `(cifar_dir:str\|None=None)->(Xtr,ytr,Xte,yte)` | 读缓存批次（兼容 str/bytes key）→ uint8/int64 numpy | ✅ ⚙️ |
| `CIFAR10Dataset` | `(Dataset)` `__init__(images,labels,mean,std,transform=None)`；`__getitem__`: reshape→[0,1]→transform→标准化 | 单样本读取 | ✅ ⚙️ |
| `get_dataloaders` | `(cfg:dict)->(train_loader,test_loader)` | 按 cfg 建 train/test DataLoader | ✅ ⚙️ |

### `src/cifar/losses.py` 🔨 ✍️
| 符号 | 签名 | 职责 | 状态 |
|---|---|---|---|
| `SoftmaxCrossEntropy` | `(autograd.Function)` `forward(ctx, logits(N,C), labels(N,))->标量`；`backward(ctx, grad_output)->(grad_logits, None)` | ✍️ 手写前向(log-sum-exp 稳定) + 反向(grad=(probs−onehot)/N) | 🔨 |
| `softmax_ce` | `(logits, labels)->Tensor` | `SoftmaxCrossEntropy.apply(...)` 薄封装 | 🔨 ⚙️ |

### `src/cifar/optim.py` 🔨 ✍️
| 符号 | 签名 | 职责 | 状态 |
|---|---|---|---|
| `BaseOptimizer` | `__init__(params,lr,weight_decay=0)`；`zero_grad()`；`step()`(抽象) | 公共：持有 params、清 `.grad` | 🔨 |
| `SGD` | `(BaseOptimizer)` `step()`: `p -= lr*(grad + wd*p)` | ✍️ 手写 vanilla SGD | 🔨 |
| `SGDMomentum` | `(BaseOptimizer, momentum=0.9)` `step()`: `v=μv+grad; p-=lr*v` | ✍️ 手写动量 | 🔨 |
| `Adam` | `(BaseOptimizer, betas=(0.9,0.999),eps=1e-8)` `step()`: 一/二阶动量 + bias correction | ✍️ 手写 Adam | 🔨 |

### `src/cifar/model.py`
| 符号 | 签名 | 职责 | 状态 |
|---|---|---|---|
| `LinearClassifier` | `(nn.Module)` `__init__(input_dim=3072,num_classes=10)`；`forward`: flatten→Linear | M1 线性基线（~40%） | 🔨 ⚙️ |
| `TwoLayerNet` | `(nn.Module)` `__init__(input_dim,hidden_dim=100,num_classes)`；fc1→ReLU→fc2 | M1 两层网 | 🔨 ⚙️ |
| `SmallCNN` | `(nn.Module)` Conv-Pool-Conv-Pool-FC + BN + Dropout | M2 | 🔜 |
| `build_model` | `(cfg:dict)->nn.Module` | 按 `cfg['model']` 工厂分发 | 🔨/🔜 ⚙️ |

### `src/cifar/evaluate.py` 🔨
| 符号 | 签名 | 职责 | 状态 |
|---|---|---|---|
| `accuracy` | `(logits, labels)->float` | 一个 batch 的 top-1 准确率 | 🔨 ⚙️ |
| `evaluate` | `(model, loader, loss_fn, device='cpu')->(avg_loss, acc)` | 跑完整个 loader 的 loss/acc（`torch.no_grad`） | 🔨 ⚙️ |

### `src/cifar/train.py` 🔨
| 符号 | 签名 | 职责 | 状态 |
|---|---|---|---|
| `train_epoch` | `(model, loader, loss_fn, optimizer, device)->(avg_loss,acc)` | ✍️ 手写一个 epoch：forward→loss→backward→step→zero_grad | 🔨 |
| `train` | `(cfg:dict, config_path=None)->None` | 编排：seed/loaders/model/loss/optim → epoch 循环(eval+log+checkpoint) | 🔨 ⚙️ |
| `main` | `()->None` | argparse CLI：`--config` | 🔨 ⚙️ |

### `tests/`
| 文件 | 内容 | 状态 |
|---|---|---|
| `test_data.py` | 形状 `(50000,3072)`、标准化 ~N(0,1) | ✅ |
| `test_losses.py` | `torch.autograd.gradcheck` 数值验证 `SoftmaxCrossEntropy.backward` | 🔨 |
| `test_optim.py` | 一步 `step()` 后 loss 下降 | 🔧 可选 |

---

## 3. 模块依赖（谁 import 谁）

```
                 train.py            ← 编排者（composition root），唯一把所有部件 wire 起来的地方
            ┌─────┬───────┬────────┬─────────┐
         data   model  losses   optim    evaluate     ← 全是"叶子"，彼此不 import
            └─────┴───────┴────────┴─────────┘
                          │
                       utils.py                   ← 最底层叶子（seed/load_config），谁都能用
```

要点：
- **data / losses / optim / model 各自只依赖 torch/numpy**，互相不耦合 → 可独立写、独立测。
- **evaluate 不 import model/data**，而是把 `(model, loader)` 当参数收进来 → 避免循环依赖。
- **train.py 是唯一知道"全局"的文件**：它决定用哪个 model、哪个 loss、哪个 optim，把它们拼起来。换实验只动 train + config，不动叶子模块。

---

## 4. 一次训练的数据流（`python -m cifar.train --config configs/baseline.yaml`）

```
1. main() → argparse → load_config(utils) → seed_everything(utils)
2. get_dataloaders(data) ─► load_cifar10 ─► CIFAR10Dataset ─► DataLoader(train/test)
3. build_model(model)     ─► LinearClassifier / TwoLayerNet
4. loss_fn   = softmax_ce(losses)          # = SoftmaxCrossEntropy.apply
5. optimizer = Adam(model.parameters(), lr, ...)(optim)
6. for epoch in range(epochs):
     train_epoch(model, train_loader, loss_fn, optim):
         for x,y in loader:
             logits = model(x)           # 前向（nn.Module 自动建计算图）
             loss   = loss_fn(logits,y)  # autograd.Function.forward + 登记 backward
             loss.backward()             # autograd 沿计算图反传 → 填各参数 .grad
             optimizer.step()            # 读 .grad、改 .data（手写更新规则）
             optimizer.zero_grad()       # 清零（torch 梯度默认累加）
     loss,acc = evaluate(model, test_loader, loss_fn)   # torch.no_grad
     日志 + 存 checkpoint
7. 存最终 checkpoint
```

---

## 5. 实现顺序（M0→M3）

| 阶段 | 做什么 | 手写/Socratic ✍️ | 样板 ⚙️ |
|---|---|---|---|
| ✅ M0 | `utils` → `data`（已实测通过） | — | seed/config/读取/标准化 |
| 🔨 M1 | `losses`(✍️ 推 forward+backward) → **gradcheck 验证** → `optim`(✍️ 3 个) → `model`(Linear/TwoLayer) → `evaluate` → `train`(✍️ loop) → 跑通 ~40% | losses 的 forward/backward；optim.step；train_epoch | model 的 nn.Module、evaluate、main/CLI |
| 🔜 M2 | `SmallCNN` + `data` 增强(RandomCrop/Flip) → ~70–80% | （可选）BN 写成 Function | CNN 结构、增强 |
| 🔧 M3 | config 驱动实验 / checkpoint 续训 / logging(TB 或 CSV) / CLI / README 收尾 | — | 工程化 |

---

## 6. 手写核心契约

- **`SoftmaxCrossEntropy`**
  - forward：用 log-sum-exp 稳定算 `log_softmax`，loss = `−log_softmax[target]` 的均值；`ctx.save_for_backward(probs, labels)`、存 `N`。
  - backward：返回 `grad_output * (probs − onehot)/N`；labels 的梯度返回 `None`（整数标签不需要梯度）。
  - 验证：`torch.autograd.gradcheck`（数值梯度 vs 解析梯度，要求 double precision、小输入）。
- **optim**：读 `p.grad`、改 `p.data`；Adam 的 bias correction（前几步除以 `1−β^t`）；weight_decay 的位置（L2 项加到 grad 上再更新）。
- **train loop**：固定顺序 `forward → loss → backward → step → zero_grad`；讲清为什么 `zero_grad` 必要（torch 梯度默认累加）。
