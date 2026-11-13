from __future__ import annotations
import numpy as np
from houssem_torch.autograd.function import record
from houssem_torch.autograd import functions as F
from houssem_torch.fusion.engine import try_fuse


def _wrap(storage, ref_tensor):
    from houssem_torch.tensor import Tensor
    return Tensor(_storage=storage, device=ref_tensor.device,
                  dtype=ref_tensor.dtype)


def _coerce(other, ref_tensor):
    """Convert scalar or array to Tensor if needed."""
    from houssem_torch.tensor import Tensor
    from houssem_torch.backends.cpu import kernels as cpu
    if isinstance(other, Tensor):
        return other
    arr = np.array(other, dtype=ref_tensor.dtype.numpy_name)
    return Tensor(_storage=cpu.CPUStorage(arr),
                  device=ref_tensor.device, dtype=ref_tensor.dtype)


def _pair(a, b):
    if not hasattr(b, '_storage'):
        b = _coerce(b, a)
    if not hasattr(a, '_storage'):
        a = _coerce(a, b)
    return a, b


def _compute(op, *tensors):
    fused = try_fuse(op, *tensors)
    if fused is not None:
        return fused

    from houssem_torch.backends.cpu import kernels as cpu
    storages = tuple(tensor._storage for tensor in tensors)
    return getattr(cpu, op)(*storages)


def add(a, b):
    a, b = _pair(a, b)
    return record(_wrap(_compute("add", a, b), a), F.AddBackward, a, b)


def subtract(a, b):
    a, b = _pair(a, b)
    return record(_wrap(_compute("subtract", a, b), a), F.SubBackward, a, b)


def multiply(a, b):
    a, b = _pair(a, b)
    return record(_wrap(_compute("multiply", a, b), a), F.MulBackward, a, b)


def divide(a, b):
    a, b = _pair(a, b)
    return record(_wrap(_compute("divide", a, b), a), F.DivBackward, a, b)


def negate(a):
    return record(_wrap(_compute("negate", a), a), F.NegBackward, a)


def relu(a):
    return record(_wrap(_compute("relu", a), a), F.ReluBackward, a)


def exp(a):
    return record(_wrap(_compute("exp", a), a), F.ExpBackward, a)


def log(a):
    return record(_wrap(_compute("log", a), a), F.LogBackward, a)
