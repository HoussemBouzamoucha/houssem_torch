from __future__ import annotations
from houssem_torch.nn.dropout import Dropout
from houssem_torch.nn.linear import Linear
from houssem_torch.nn.module import Module
from houssem_torch.ops.activation import gelu
from houssem_torch.ops.pointwise import relu

_ACTIVATIONS = {"relu": relu, "gelu": gelu}


class FeedForward(Module):
    """Position-wise feed-forward block: Linear -> activation -> dropout -> Linear.

    Applied to every position independently: (..., d) -> (..., d_ff) -> (..., d).
    d_ff is usually 4·d. This is where most of a transformer layer's
    parameters live. activation: "relu" (the original paper) or "gelu".
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0,
                 activation: str = "relu"):
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}, "
                             f"got '{activation}'.")
        self.up = Linear(d_model, d_ff)
        self.down = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        return self.down(self.dropout(_ACTIVATIONS[self.activation](self.up(x))))
