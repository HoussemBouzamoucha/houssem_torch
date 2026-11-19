from __future__ import annotations
import numpy as np
from houssem_torch.optim.optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic gradient descent, optionally with momentum.

        velocity = momentum · velocity + grad
        param   -= lr · velocity

    With momentum=0 this is plain `param -= lr · grad`. Momentum smooths the
    updates by remembering a running direction. Mostly useful as a baseline;
    transformers are trained with Adam.
    """

    _state_names = ("velocity",)

    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.numpy()) for p in self.params]

    def step(self) -> None:
        for p, vel in zip(self.params, self.velocity):
            if p._grad is None:
                continue
            if self.momentum:
                vel *= self.momentum
                vel += p._grad
                update = vel
            else:
                update = p._grad
            p.numpy()[...] -= self.lr * update
