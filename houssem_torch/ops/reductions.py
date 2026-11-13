from __future__ import annotations


def _wrap(storage, ref_tensor):
    from houssem_torch.tensor import Tensor
    return Tensor(_storage=storage, device=ref_tensor.device,
                  dtype=ref_tensor.dtype)


def sum(a, axis=None, keepdims=False):
    from houssem_torch.backends.cpu import kernels as cpu
    return _wrap(cpu.sum(a._storage, axis=axis, keepdims=keepdims), a)


def mean(a, axis=None, keepdims=False):
    from houssem_torch.backends.cpu import kernels as cpu
    return _wrap(cpu.mean(a._storage, axis=axis, keepdims=keepdims), a)


def max(a, axis=None, keepdims=False):
    from houssem_torch.backends.cpu import kernels as cpu
    return _wrap(cpu.max(a._storage, axis=axis, keepdims=keepdims), a)


def min(a, axis=None, keepdims=False):
    from houssem_torch.backends.cpu import kernels as cpu
    return _wrap(cpu.min(a._storage, axis=axis, keepdims=keepdims), a)