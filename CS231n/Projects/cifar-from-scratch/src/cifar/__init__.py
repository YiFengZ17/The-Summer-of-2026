"""From-scratch CIFAR-10 classifier (CS231n final project).

Package layout:
    data.py      CIFAR-10 Dataset / DataLoader / normalization / augmentation
    model.py     models (two-layer net, small CNN)
    losses.py    hand-written SoftmaxCrossEntropy (autograd.Function)
    optim.py     hand-written SGD / Momentum / Adam
    train.py     hand-written training loop + CLI (``python -m cifar.train``)
    evaluate.py  accuracy / metrics
"""

__version__ = "0.1.0"
