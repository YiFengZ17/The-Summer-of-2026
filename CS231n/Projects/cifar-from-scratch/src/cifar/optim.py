"""Hand-written optimizers: SGD, SGD+Momentum, Adam.

YOU implement each `step()`. The plumbing (BaseOptimizer, buffer init in
`__init__`, `zero_grad`) is ⚙️ provided.

How an optimizer works
----------------------
- `zero_grad()` clears `p.grad` on every parameter (torch *accumulates* grads, so
  you must clear them each step).
- `step()` reads each param's `p.grad` (filled by `loss.backward()`) and updates
  `p.data` (the weights) in place. The update itself must NOT be tracked by
  autograd, so we touch `p.data` directly.

Weight decay (L2 reg) lives HERE, not in the loss:
    g = p.grad + weight_decay * p.data      # add the L2 gradient as a NEW tensor
                                            # (do NOT mutate p.grad in place)

⚠️ Buffers (velocity / m / v) MUST update in place (mul_/add_/addcmul_) —
   rebinding a local variable does NOT change the stored buffer.
"""

import torch

# This part can be done directly by PyTorch API, optimizer = optim._()
class BaseOptimizer:
    """Minimal optimizer base: holds params, lr, weight_decay; clears grads."""

    def __init__(self, params, lr, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None

    def step(self):
        raise NotImplementedError

    def state_dict(self):
        """Snapshot buffers + scalars for resume (NOT the params themselves).

        Generic: saves every list-of-tensor attr (velocity / m / v) and every
        scalar attr (lr, t, momentum, ...). Params are skipped -- on resume the
        optimizer is re-created from the model, then buffers are restored here.
        """
        out = {}
        for k, v in self.__dict__.items():
            if k == "params":
                continue
            if isinstance(v, list):
                out[k] = [t.detach().clone() for t in v]
            elif isinstance(v, torch.Tensor):
                out[k] = v.detach().clone()
            else:
                out[k] = v
        return out

    def load_state_dict(self, sd):
        for k, v in sd.items():
            cur = getattr(self, k, None)
            if isinstance(v, list) and isinstance(cur, list):
                for i, t in enumerate(v):
                    cur[i].copy_(t)
            elif isinstance(v, torch.Tensor) and isinstance(cur, torch.Tensor):
                cur.copy_(t)
            else:
                setattr(self, k, v)


class SGD(BaseOptimizer):
    """Vanilla SGD:  p -= lr * (grad + wd*p)."""

    def __init__(self, params, lr, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)

    def step(self):
        for p in self.params:
            g = p.grad + self.weight_decay * p.data
            p.data -= self.lr * g



class SGDMomentum(BaseOptimizer):
    """SGD with classic (CS231n) momentum:
        v <- mu*v + g
        p -= lr * v
    """

    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p) for p in self.params]  # one buffer per param

    def step(self):
        for i, p in enumerate(self.params):
            g = p.grad + self.weight_decay * p.data
            v = self.velocity[i]
            # change on place, not the copy v
            v.mul_(self.momentum).add_(g)
            p.data -= self.lr * v


class Adam(BaseOptimizer):
    """Adam with bias correction:
        t <- t + 1
        m <- b1*m + (1-b1)*g
        v <- b2*v + (1-b2)*g^2
        m_hat = m / (1 - b1^t)
        v_hat = v / (1 - b2^t)
        p -= lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]   # first moment
        self.v = [torch.zeros_like(p) for p in self.params]   # second moment
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad + self.weight_decay * p.data
            m, v = self.m[i], self.v[i]
            # change on place, not the copy m/v
            m.mul_(self.beta1).add_(g,alpha = 1-self.beta1)
            v.mul_(self.beta2).addcmul_(g,g,value = 1-self.beta2)
            m_hat = m /(1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
