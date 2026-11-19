"""
Houssem Torch — a small deep-learning framework built from scratch.
"""

__version__ = "0.1.0"

from houssem_torch.tensor import Tensor
from houssem_torch.device import Device
from houssem_torch.dtypes import float32, int64

from houssem_torch.autograd import no_grad
from houssem_torch.parameter import Parameter
from houssem_torch.fusion import fusion_mode


from houssem_torch.ops.creation import (
    zeros, ones, full, arange, randn, tensor
)
from houssem_torch.ops.pointwise import (
    add, subtract, multiply, divide, relu, exp, log, sqrt, pow, masked_fill
)
from houssem_torch.ops.reductions import sum, mean, max, min
from houssem_torch.ops.linalg import matmul
from houssem_torch.ops.shape import reshape, permute, swapaxes, getitem, split
from houssem_torch.ops.activation import gelu, softmax, log_softmax, dropout
from houssem_torch.ops.normalization import layer_norm
from houssem_torch.ops.embedding import embedding
from houssem_torch.ops.loss import cross_entropy
from houssem_torch.random import manual_seed
from houssem_torch import nn, optim  # noqa: E402  (after core, they use it)

__all__ = [
    "Tensor", "Device", "float32", "int64",
    "__version__",
    # creation
    "zeros", "ones", "full", "arange", "randn", "tensor",
    # pointwise
    "add", "subtract", "multiply", "divide", "relu", "exp", "log",
    "sqrt", "pow", "masked_fill",
    # reductions
    "sum", "mean", "max", "min",
    # linalg
    "matmul",
    # shape
    "reshape", "permute", "swapaxes", "getitem", "split",
    # transformer ops
    "gelu", "softmax", "log_softmax", "dropout", "layer_norm", "embedding",
    "cross_entropy",
    # misc
    "manual_seed", "no_grad", "Parameter", "fusion_mode",
    # packages
    "nn", "optim",
]

