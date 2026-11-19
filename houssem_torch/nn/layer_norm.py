from __future__ import annotations
from houssem_torch.nn import init
from houssem_torch.nn.module import Module
from houssem_torch.ops.normalization import layer_norm
from houssem_torch.parameter import Parameter


class LayerNorm(Module):
    """Normalize each position's feature vector to mean 0, variance 1, then
    apply a learned scale (weight, starts at 1) and shift (bias, starts at 0).

    Input (..., d) -> output (..., d). Keeps activations at a stable scale
    through a deep stack of layers, which is what makes deep transformers
    trainable.
    """

    def __init__(self, d: int, eps: float = 1e-5):
        self.d = d
        self.eps = eps
        self.weight = Parameter(init.ones(d))
        self.bias = Parameter(init.zeros(d))

    def forward(self, x):
        return layer_norm(x, self.weight, self.bias, self.eps)
