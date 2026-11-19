"""Activations, softmax and dropout.

Each `_backward` hands the saved forward values to its function in
ops/grads.py, where the gradient math lives.
"""
from __future__ import annotations
import numpy as np
from houssem_torch.autograd.graph import result
from houssem_torch.ops import grads


def gelu(x):
    """GELU, tanh approximation: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))."""
    from houssem_torch.backends.cpu import kernels as cpu
    out = cpu.gelu(x._storage)

    def _backward(g):
        x._accumulate_grad(grads.gelu_backward(g, x.numpy()))

    return result(out, x, (x,), "gelu", _backward)


def softmax(x, axis: int = -1):
    """Numerically stable softmax along `axis` (subtracts the max first)."""
    from houssem_torch.backends.cpu import kernels as cpu
    out = cpu.softmax(x._storage, axis=axis)  # saved: out.array = probs

    def _backward(g):
        x._accumulate_grad(grads.softmax_backward(g, out.array, axis))

    return result(out, x, (x,), "softmax", _backward)


def log_softmax(x, axis: int = -1):
    """log(softmax(x)) computed directly, so it never takes log(0)."""
    from houssem_torch.backends.cpu import kernels as cpu
    out = cpu.log_softmax(x._storage, axis=axis)  # saved: out.array = log p

    def _backward(g):
        x._accumulate_grad(grads.log_softmax_backward(g, out.array, axis))

    return result(out, x, (x,), "log_softmax", _backward)


def dropout(x, p: float = 0.1, training: bool = True):
    """Zero each element with probability p; scale survivors by 1/(1-p).

    Identity (and no graph node) when not training or p == 0.
    """
    from houssem_torch.backends.cpu import kernels as cpu
    from houssem_torch.random import get_rng

    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout probability must be in [0, 1), got {p}.")
    if not training or p == 0.0:
        return x

    dtype = x.dtype.numpy_name
    keep = get_rng().random(x.shape, dtype=np.float32) >= p
    mask = keep.astype(dtype) / np.asarray(1.0 - p, dtype=dtype)  # saved
    out = cpu.CPUStorage(x.numpy() * mask)

    def _backward(g):
        x._accumulate_grad(grads.dropout_backward(g, mask))

    return result(out, x, (x,), "dropout", _backward)
