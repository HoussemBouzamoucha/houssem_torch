from __future__ import annotations
from houssem_torch.autograd.function import record
from houssem_torch.autograd import functions as F


def _wrap(storage, ref_tensor):
    from houssem_torch.tensor import Tensor
    return Tensor(_storage=storage, device=ref_tensor.device,
                  dtype=ref_tensor.dtype)


def sum(a, axis=None, keepdims=False):
    from houssem_torch.backends.cpu import kernels as cpu
    out = _wrap(cpu.sum(a._storage, axis=axis, keepdims=keepdims), a)
    return record(out, F.SumBackward, a, axis=axis)


def mean(a, axis=None, keepdims=False):
    from houssem_torch.backends.cpu import kernels as cpu
    out = _wrap(cpu.mean(a._storage, axis=axis, keepdims=keepdims), a)
    count = a.numel // out.numel
    return record(out, F.SumBackward, a, axis=axis, scale=1.0 / count)


# No backward yet: results of max/min never require grad.
def max(a, axis=None, keepdims=False):
    from houssem_torch.backends.cpu import kernels as cpu
    return _wrap(cpu.max(a._storage, axis=axis, keepdims=keepdims), a)


def min(a, axis=None, keepdims=False):
    from houssem_torch.backends.cpu import kernels as cpu
    return _wrap(cpu.min(a._storage, axis=axis, keepdims=keepdims), a)