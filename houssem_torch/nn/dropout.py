from __future__ import annotations
from houssem_torch.nn.module import Module
from houssem_torch.ops.activation import dropout


class Dropout(Module):
    """Randomly zero a fraction p of values during training (scaling the rest
    by 1/(1-p) so the expected value is unchanged); do nothing in eval mode.

    Stops the network from relying on any single unit, reducing overfitting.
    Which mode it is in follows `model.train()` / `model.eval()`.
    """

    def __init__(self, p: float = 0.1):
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}.")
        self.p = p

    def forward(self, x):
        return dropout(x, self.p, training=self.training)
