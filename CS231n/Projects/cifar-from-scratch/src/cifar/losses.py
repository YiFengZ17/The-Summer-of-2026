"""Hand-written softmax + cross-entropy loss, as a torch.autograd.Function.

This is a hand-written core: YOU implement `forward` and `backward`.
The surrounding plumbing (staticmethods, ctx, .apply, the wrapper) is provided.

Contract
--------
forward(ctx, logits, labels)
    logits : (N, C) float        raw scores
    labels : (N,) long           integer class indices in [0, C)
    -> scalar tensor             mean cross-entropy over the batch

backward(ctx, grad_output)
    grad_output : scalar tensor  upstream gradient dL_total/d(loss)
    -> (grad_logits, None)       grad w.r.t. logits; None for labels (it's an index)

See docs/ARCHITECTURE.md §6 for the derivation.
Verify with:  python -m pytest tests/test_losses.py
"""

import torch


class SoftmaxCrossEntropy(torch.autograd.Function):
    """Numerically stable softmax + cross-entropy, averaged over the batch."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # TODO: implement, then delete `raise NotImplementedError`.
        #
        # 1. Compute log_softmax STABLY:
        #      - subtract the per-row max of logits (along the class dim) to avoid
        #        exp() overflow;
        #      - log_softmax = (shifted logits) - log(sum(exp(shifted logits))).
        # 2. Pick the log-prob of the CORRECT class per sample
        #      (use torch.arange(N) to index: log_softmax[arange(N), labels]).
        # 3. loss = mean of -that- over the batch  ->  scalar tensor.
        # 4. Save for backward:
        #      ctx.save_for_backward(<softmax probs = exp(log_softmax)>, labels)
        #      ctx.N = N                      (or save a 1-element tensor)
        #    return loss.
        N = logits.shape[0]
        logits_norm = logits - torch.amax(logits,dim = 1,keepdim = True)
        logits_log = -logits_norm + torch.log(torch.sum(torch.exp(logits_norm),dim=1,keepdim=True))
        logits_idx = logits_log[torch.arange(N),labels]
        loss = torch.sum(logits_idx) / N
        probs = torch.exp(-logits_log)
        ctx.save_for_backward(probs,labels)
        ctx.N = N
        return loss
        

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # TODO: implement, then delete `raise NotImplementedError`.
        #
        # Recover what forward saved, then return:
        #     grad_logits = grad_output * (probs - one_hot(labels)) / N
        #     return grad_logits, None      # one grad per forward input; None for labels
        #
        # Hint on one-hot: you can build it, OR note that
        #     probs - one_hot(labels)  ==  probs with 1.0 subtracted at column `labels`.
        probs,labels = ctx.saved_tensors
        N = ctx.N
        grad = probs.clone()
        grad[torch.arange(N),labels] -= 1
        grad_logits = grad_output * (grad) / N
        return grad_logits, None


def softmax_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper around SoftmaxCrossEntropy.apply(logits, labels)."""
    return SoftmaxCrossEntropy.apply(logits, labels)
