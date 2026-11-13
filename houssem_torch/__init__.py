"""
Houssem Torch — a small deep-learning framework built from scratch.
"""

__version__ = "0.1.0"

from houssem_torch.tensor import Tensor
from houssem_torch.device import Device
from houssem_torch.dtypes import float32, int64

from houssem_torch.ops.creation import (
    zeros, ones, full, arange, randn, tensor
)
from houssem_torch.ops.pointwise import add, subtract, multiply, divide, relu, exp, log
from houssem_torch.ops.reductions import sum, mean, max, min
from houssem_torch.ops.linalg import matmul

__all__ = [
    "Tensor", "Device", "float32", "int64",
    "__version__",
    # creation
    "zeros", "ones", "full", "arange", "randn", "tensor",
    # pointwise
    "add", "subtract", "multiply", "divide", "relu", "exp", "log",
    # reductions
    "sum", "mean", "max", "min",
    # linalg
    "matmul",
]