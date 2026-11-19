from __future__ import annotations
import numpy as np
from houssem_torch.autograd.graph import result


def _wrap(storage, ref_tensor):
    from houssem_torch.tensor import Tensor
    return Tensor(_storage=storage, device=ref_tensor.device,
                  dtype=ref_tensor.dtype)


def _kept_shape(shape, axis):
    """Shape of the reduction result with keepdims=True."""
    if axis is None:
        return (1,) * len(shape)
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = {ax % len(shape) for ax in axes}
    return tuple(1 if i in axes else s for i, s in enumerate(shape))


def sum(a, axis=None, keepdims=False):
    from houssem_torch.backends.cpu import kernels as cpu

    def _backward(g):
        # every input element contributes once: broadcast g back over the axes
        g = g.reshape(_kept_shape(a.shape, axis))
        a._accumulate_grad(np.broadcast_to(g, a.shape))

    out = cpu.sum(a._storage, axis=axis, keepdims=keepdims)
    return result(out, a, (a,), "sum", _backward)


def mean(a, axis=None, keepdims=False):
    from houssem_torch.backends.cpu import kernels as cpu
    out = cpu.mean(a._storage, axis=axis, keepdims=keepdims)
    out_numel = int(np.prod(out.shape, dtype=np.int64))
    count = a.numel // out_numel if out_numel else 1

    def _backward(g):
        # like sum, scaled by 1/count
        g = g.reshape(_kept_shape(a.shape, axis)) * (1.0 / count)
        a._accumulate_grad(np.broadcast_to(g, a.shape))

    return result(out, a, (a,), "mean", _backward)


# No backward yet: results of max/min never require grad.
def max(a, axis=None, keepdims=False):
    from houssem_torch.backends.cpu import kernels as cpu
    return _wrap(cpu.max(a._storage, axis=axis, keepdims=keepdims), a)


def min(a, axis=None, keepdims=False):
    from houssem_torch.backends.cpu import kernels as cpu
    return _wrap(cpu.min(a._storage, axis=axis, keepdims=keepdims), a)
