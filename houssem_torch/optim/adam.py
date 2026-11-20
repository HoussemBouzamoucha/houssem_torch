from __future__ import annotations
import numpy as np
from houssem_torch.backends.cpu import fast
from houssem_torch.optim.optimizer import Optimizer


class Adam(Optimizer):
    """Adam: each parameter gets its own adaptive step size.

    For every parameter, two running averages are kept:
        m = β1·m + (1-β1)·grad        (average direction: momentum)
        v = β2·v + (1-β2)·grad²       (average squared size)
    Early on both are biased towards 0, so they are corrected:
        m̂ = m / (1-β1^t),   v̂ = v / (1-β2^t)        (t = step count)
    and the update is
        param -= lr · m̂ / (sqrt(v̂) + eps)
    Dividing by sqrt(v̂) gives parameters with consistently large gradients
    smaller steps, and rarely updated ones (rare tokens' embeddings) larger
    steps.

    Defaults are the transformer paper's: betas=(0.9, 0.98), eps=1e-9.
    weight_decay > 0 adds the classic L2 penalty to the gradient
    (grad += wd·param); prefer AdamW for decoupled weight decay.
    Each parameter keeps its own step count, so one that receives no gradient
    on a step is left untouched.
    """

    _state_names = ("steps", "m", "v")
    _decoupled_decay = False  # AdamW sets True: decay the weights, not the grad

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.98),
                 eps: float = 1e-9, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.steps = [0] * len(self.params)
        self.m = [np.zeros_like(p.numpy()) for p in self.params]
        self.v = [np.zeros_like(p.numpy()) for p in self.params]

    def _decay(self, p) -> None:
        """Hook for AdamW's decoupled decay; Adam does nothing here."""

    def step(self) -> None:
        b1, b2 = self.beta1, self.beta2
        for i, p in enumerate(self.params):
            if p._grad is None:
                continue
            self.steps[i] += 1
            t = self.steps[i]
            bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t  # bias corrections
            pv, m, v = p.numpy(), self.m[i], self.v[i]
            if fast.use(pv, p._grad) and _contiguous(pv, m, v):
                # one fused pass: L2 penalty, moments and update together
                self._decay(p)
                l2 = 0.0 if self._decoupled_decay else self.weight_decay
                fast.adam_step(pv.reshape(-1), fast.flat(p._grad), m.reshape(-1),
                               v.reshape(-1), self.lr, b1, b2, bc1, bc2,
                               self.eps, l2)
                continue
            grad = p._grad
            if self.weight_decay and not self._decoupled_decay:  # L2 penalty
                grad = grad + self.weight_decay * pv
            m *= b1
            m += (1 - b1) * grad
            v *= b2
            v += (1 - b2) * np.square(grad)
            m_hat = m / bc1
            v_hat = v / bc2
            self._decay(p)
            pv[...] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _contiguous(*arrays) -> bool:
    return all(a.flags.c_contiguous for a in arrays)
