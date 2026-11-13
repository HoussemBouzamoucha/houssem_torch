from __future__ import annotations
import numpy as np
from houssem_torch.dtypes import float32


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


def add(a, b):
    from houssem_torch.backends.cpu import kernels as cpu
    b = _coerce(b, a)
    return _wrap(cpu.add(a._storage, b._storage), a)


def subtract(a, b):
    from houssem_torch.backends.cpu import kernels as cpu
    if not hasattr(b, '_storage'):
        b = _coerce(b, a)
    if not hasattr(a, '_storage'):
        a = _coerce(a, b)
    return _wrap(cpu.subtract(a._storage, b._storage), a)


def multiply(a, b):
    from houssem_torch.backends.cpu import kernels as cpu
    b = _coerce(b, a)
    return _wrap(cpu.multiply(a._storage, b._storage), a)


def divide(a, b):
    from houssem_torch.backends.cpu import kernels as cpu
    if not hasattr(b, '_storage'):
        b = _coerce(b, a)
    if not hasattr(a, '_storage'):
        a = _coerce(a, b)
    return _wrap(cpu.divide(a._storage, b._storage), a)

def relu(a):
    from houssem_torch.backends.cpu import kernels as cpu
    return _wrap(cpu.relu(a._storage), a)


def exp(a):
    from houssem_torch.backends.cpu import kernels as cpu
    return _wrap(cpu.exp(a._storage), a)


def log(a):
    from houssem_torch.backends.cpu import kernels as cpu
    return _wrap(cpu.log(a._storage), a)