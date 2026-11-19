from __future__ import annotations
from houssem_torch.nn import init
from houssem_torch.nn.module import Module
from houssem_torch.parameter import Parameter


class Linear(Module):
    """Fully connected layer: y = x @ weight + bias.

    weight: (in_features, out_features), Xavier-uniform initialized.
    bias:   (out_features,), zero initialized (or None with bias=False).

    Input (..., in_features) -> output (..., out_features). All leading
    dimensions are flattened into one before the multiply, so (B, T, d) runs
    as a single (B·T, d) @ (d, out) matrix product. That is faster than a
    batched matmul, and the weight gradient needs no broadcast reduction.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(init.xavier_uniform((in_features, out_features)))
        self.bias = Parameter(init.zeros(out_features)) if bias else None

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Linear expected last dim {self.in_features}, "
                             f"got input shape {x.shape}")
        leading = x.shape[:-1]
        y = x.reshape(-1, self.in_features) @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*leading, self.out_features)
