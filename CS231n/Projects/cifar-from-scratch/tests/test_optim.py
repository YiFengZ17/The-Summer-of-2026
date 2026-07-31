"""Verification for the hand-written optimizers.

Two checks:
  1. Vanilla SGD must match torch.optim.SGD EXACTLY (same params after N steps).
  2. Each optimizer must drive a simple quadratic toward its minimum (loss↓).
"""

import torch

from cifar.optim import SGD, SGDMomentum, Adam


def test_sgd_matches_torch():
    """Vanilla SGD (no momentum / weight decay) == torch.optim.SGD, to the bit."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(3, 4))
    p_ref = torch.nn.Parameter(p.detach().clone())
    opt = SGD([p], lr=0.1)
    opt_ref = torch.optim.SGD([p_ref], lr=0.1)
    for _ in range(5):
        g = torch.randn(3, 4)           # fixed random gradient (detached leaf)
        p.grad = g
        p_ref.grad = g.clone()
        opt.step()
        opt_ref.step()
    assert torch.allclose(p.data, p_ref.data, atol=1e-7), float((p.data - p_ref.data).abs().max())


def test_momentum_matches_torch():
    """SGDMomentum must match torch.optim.SGD(momentum=...) to the bit.

    Catches the classic 'velocity buffer never updated' bug: rebinding a local
    variable instead of updating the buffer in place silently degenerates to
    vanilla SGD -- loss still goes down, but momentum is fake. This fails it.
    """
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(3, 4))
    p_ref = torch.nn.Parameter(p.detach().clone())
    opt = SGDMomentum([p], lr=0.1, momentum=0.9)
    opt_ref = torch.optim.SGD([p_ref], lr=0.1, momentum=0.9)
    for _ in range(5):
        g = torch.randn(3, 4)
        p.grad = g
        p_ref.grad = g.clone()
        opt.step()
        opt_ref.step()
    assert torch.allclose(p.data, p_ref.data, atol=1e-7), float((p.data - p_ref.data).abs().max())


def test_adam_matches_torch():
    """Adam must match torch.optim.Adam to the bit."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(3, 4))
    p_ref = torch.nn.Parameter(p.detach().clone())
    opt = Adam([p], lr=1e-3)
    opt_ref = torch.optim.Adam([p_ref], lr=1e-3)
    for _ in range(5):
        g = torch.randn(3, 4)
        p.grad = g
        p_ref.grad = g.clone()
        opt.step()
        opt_ref.step()
    assert torch.allclose(p.data, p_ref.data, atol=1e-6), float((p.data - p_ref.data).abs().max())


def test_optimizers_reduce_loss():
    """Each optimizer should shrink loss on f(p) = sum((p - target)^2)."""
    for OptimCls, kw in [
        (SGD, dict(lr=0.1)),
        (SGDMomentum, dict(lr=0.1, momentum=0.9)),
        (Adam, dict(lr=0.05)),
    ]:
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.randn(4, 5))
        target = torch.randn(4, 5)
        opt = OptimCls([p], **kw)

        def loss():
            return ((p - target) ** 2).sum()

        loss0 = loss().item()
        for _ in range(50):
            opt.zero_grad()
            loss().backward()
            opt.step()
        loss1 = loss().item()
        assert loss1 < loss0 / 10, (OptimCls.__name__, loss0, loss1)
