from __future__ import annotations
from houssem_torch.optim.adam import Adam


class AdamW(Adam):
    """Adam with decoupled weight decay (Loshchilov & Hutter).

    Instead of adding wd·param to the gradient (where Adam's adaptive scaling
    would distort it), the weights are shrunk directly each step:
        param -= lr · weight_decay · param
    then the normal Adam update is applied. This is the standard choice for
    modern transformer training. Typical weight_decay: 0.01 to 0.1.
    """

    _decoupled_decay = True

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.98),
                 eps: float = 1e-9, weight_decay: float = 0.01):
        super().__init__(params, lr, betas, eps, weight_decay)

    def _decay(self, p) -> None:
        if self.weight_decay:
            p.numpy()[...] *= 1 - self.lr * self.weight_decay
