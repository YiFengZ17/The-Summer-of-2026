# The Summer of 2026

这是我在 2026 年暑期围绕计算机视觉 (Computer Vision)、自然语言处理 (Natural Language Processing) 与深度学习工程 (Deep Learning Engineering) 进行集中学习和实践的记录。

仓库最初是一份学习计划，最终沉淀为两门 Stanford 课程的作业实现、一个 CIFAR-10 图像分类项目、实验报告与课程笔记。重点不只是调用框架完成训练，而是通过手写关键模块理解数据如何流经模型、梯度如何反向传播，以及优化器如何更新参数。

## 最终成果

| 模块 | 完成内容 | 可验证产物 | 状态 |
| --- | --- | --- | --- |
| CS231n | Assignment 1–3：分类器、全连接网络、归一化、卷积网络、图像描述、Transformer、扩散模型等 | [Assignments](CS231n/Assignments/) | 已完成 |
| CS231n Final Project | 从零搭建 CIFAR-10 训练工程，手写损失函数、SGD/Momentum/Adam 与训练循环 | [项目 README](CS231n/Projects/cifar-from-scratch/README.md) · [项目报告](CS231n/Reports/基于cifar10数据集：从零搭建图像分类器.md) | 已完成 |
| CS224n | Assignment 1–4：词向量、依存句法分析、Transformer、LLM 评测流程 | [Assignments](CS224n/cs224n/) | 核心代码已完成；A4 仅用 mock 数据验证流程 |
| LeetCode | 150 道练习，覆盖 16 类常见算法与数据结构 | [分类题单](Leetcode_sheet.md) | 已整理 |

## CS231n：计算机视觉

### Assignment 1：分类基础

- 实现 K 近邻 (K-Nearest Neighbors, KNN)、Softmax 分类器与两层神经网络。
- 练习向量化计算 (vectorization)、数值梯度检查 (numerical gradient check) 与反向传播 (backpropagation)。
- 使用 CIFAR-10 完成图像分类实验与特征分析。

### Assignment 2：神经网络组件

- 实现全连接网络、批归一化 (Batch Normalization)、层归一化 (Layer Normalization) 与随机失活 (Dropout)。
- 实现卷积神经网络 (Convolutional Neural Network, CNN)，并过渡到 PyTorch 训练流程。
- 完成循环神经网络 (Recurrent Neural Network, RNN) 图像描述任务。

### Assignment 3：现代视觉模型

- 实现基于 Transformer 的图像描述模型。
- 完成 CLIP/DINO 表征实验与扩散模型 (Diffusion Model) 练习。
- 自监督学习 (Self-Supervised Learning) 的训练实验尚未完整跑通，保留为后续工作。

## CS231n Final Project：CIFAR-10 from Scratch

项目目标是理解一个机器学习工程的完整数据流，如何从裸数据到训练集到构建模型到调试参数到分析结果。

```text
YAML 配置
   ↓
数据加载与归一化
   ↓
模型前向传播
   ↓
手写 Softmax Cross-Entropy
   ↓
反向传播
   ↓
手写 SGD / Momentum / Adam
   ↓
评估、CSV 日志与 checkpoint
```

核心实现包括：

- 使用 `torch.autograd.Function` 手写数值稳定的 Softmax 交叉熵 (Softmax Cross-Entropy) 及其梯度。
- 手写 SGD、动量法 (Momentum) 与 Adam，包括 Adam 的偏差修正 (bias correction)。
- 手写训练循环，并加入配置管理、断点恢复 (checkpoint/resume)、CSV 日志与自动化测试。
- 构建两层网络与小型 CNN，对比模型容量、数据增强和正则化的影响。
- 本地测试结果：`11 passed`。

### 实验结果

| 模型 | 验证集准确率 | 实验说明 |
| --- | ---: | --- |
| SmallCNN | **74.46%** | 10 epochs；最佳结果出现在 epoch 9 |
| Two-layer network | 49.13% | 当前仓库日志记录到 epoch 3 |

SmallCNN 在第 10 个 epoch 的训练准确率继续上升，但验证损失从 `0.7312` 上升到 `0.8068`，验证准确率从 `74.46%` 降至 `71.75%`。这表明模型开始过拟合 (overfitting)，因此应使用 epoch 9 的最佳 checkpoint，而不是最后一个 checkpoint。

更多实现细节与运行方法见 [项目说明](CS231n/Projects/cifar-from-scratch/README.md) 和 [完整报告](CS231n/Reports/基于cifar10数据集：从零搭建图像分类器.md)。

## CS224n：自然语言处理

### Assignment 1：词向量

- 探索 Word2Vec 与 GloVe 词向量。
- 使用余弦相似度 (cosine similarity)、主成分分析 (Principal Component Analysis, PCA) 等方法观察语义关系与词向量偏差。

### Assignment 2：依存句法分析

- 实现基于转移系统 (transition-based system) 的神经依存句法分析器。
- 练习 minibatch 训练、掩码处理与模型权重保存。

### Assignment 3：Transformer

- 实现多头自注意力 (multi-head self-attention)、解码器块 (decoder block)、因果掩码 (causal mask)、语言模型损失与自回归生成。
- 本地快照测试结果：`6 passed`。

### Assignment 4：大语言模型评测

- 实现 GSM8K、LLM-as-a-Judge 与红队测试 (red teaming) 的评测流程。
- 当前结果来自 mock response，证明管线能够运行，但不能作为真实模型能力结论。


## 仓库结构

```text
.
├── CS231n/
│   ├── Assignments/          # CS231n Assignment 1–3
│   ├── Projects/
│   │   └── cifar-from-scratch/
│   └── Reports/              # CIFAR-10 项目报告与 PDF
├── CS224n/
│   ├── cs224n/               # CS224n Assignment 1–4
│   └── Notes/                # 课程笔记
└── README.md
```

## 当前不足与后续工作

- 补完 CS231n 自监督学习实验，并记录可复现的配置、指标与失败分析。
- 使用真实模型输出重新运行 CS224n A4；目前的 mock 结果只能验证评测管线。
- 补充统一的根目录环境说明与运行入口，减少不同课程环境之间的依赖冲突。

## 时间范围

本次集中学习的仓库记录始于 **2026-07-01**，当前总结截至 **2026-08-23**。
