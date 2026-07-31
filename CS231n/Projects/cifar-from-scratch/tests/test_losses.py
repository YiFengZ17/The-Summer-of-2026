"""Verification for the hand-written SoftmaxCrossEntropy.

Two checks:
  1. gradcheck — torch compares our analytic backward against a NUMERICAL gradient
     (finite differences). If they match, your backward is correct.
  2. value check — our forward loss equals torch.nn.functional.cross_entropy on
     the same input (sanity that forward is right).

Run:  python -m pytest tests/test_losses.py -v
"""

import torch

from cifar.losses import SoftmaxCrossEntropy, softmax_ce


def test_softmax_ce_value_matches_torch():
    """Forward sanity: our loss should equal torch's CE to high precision."""
    torch.manual_seed(0)
    logits = torch.randn(8, 10, dtype=torch.float64)
    labels = torch.randint(0, 10, (8,))
    mine = softmax_ce(logits, labels)
    ref = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.allclose(mine, ref, atol=1e-6), (float(mine), float(ref))


def test_softmax_ce_gradcheck():
    """Backward sanity: analytic gradient vs numerical gradient (finite diff).

    gradcheck needs double precision and a differentiable input. `labels` is an
    integer index (non-differentiable), so we check gradients only w.r.t. logits
    and our backward returns None for labels.
    """
    torch.manual_seed(0)
    N, C = 6, 10
    logits = torch.randn(N, C, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, C, (N,))

    ok = torch.autograd.gradcheck(
        lambda x: SoftmaxCrossEntropy.apply(x, labels),
        (logits,),
        eps=1e-6,
        atol=1e-4,
    )
    assert ok, "SoftmaxCrossEntropy backward does not match the numerical gradient"
