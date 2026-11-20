"""Activations, softmax and dropout.

Each `_backward` hands the saved forward values to its function in
ops/grads.py, where the gradient math lives. When Numba is available,
big inputs take the fused multi-core kernels in backends/cpu/fast.py
instead (same math, checked against the reference in the tests).
"""
from __future__ import annotations
import numpy as np
from houssem_torch.autograd.graph import result
from houssem_torch.backends.cpu import fast
from houssem_torch.ops import grads


def _last_axis(axis: int, array: np.ndarray) -> bool:
    return array.ndim >= 1 and axis % array.ndim == array.ndim - 1


def gelu(x):
    """GELU, tanh approximation: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))."""
    from houssem_torch.backends.cpu import kernels as cpu
    xv = x.numpy()
    if fast.use(xv):
        out = cpu.CPUStorage(fast.gelu_forward(fast.flat(xv)).reshape(xv.shape))

        def _backward(g):
            dx = fast.gelu_backward(fast.flat(g), fast.flat(xv))
            x._accumulate_grad(dx.reshape(xv.shape))
    else:
        out = cpu.gelu(x._storage)

        def _backward(g):
            x._accumulate_grad(grads.gelu_backward(g, xv))

    return result(out, x, (x,), "gelu", _backward)


def softmax(x, axis: int = -1):
    """Numerically stable softmax along `axis` (subtracts the max first)."""
    from houssem_torch.backends.cpu import kernels as cpu
    xv = x.numpy()
    if _last_axis(axis, xv) and fast.use(xv):
        probs = fast.softmax_forward(fast.rows(xv))  # saved for backward
        out = cpu.CPUStorage(probs.reshape(xv.shape))

        def _backward(g):
            dx = fast.softmax_backward(fast.rows(g), probs)
            x._accumulate_grad(dx.reshape(xv.shape))
    else:
        out = cpu.softmax(x._storage, axis=axis)  # saved: out.array = probs

        def _backward(g):
            x._accumulate_grad(grads.softmax_backward(g, out.array, axis))

    return result(out, x, (x,), "softmax", _backward)


def log_softmax(x, axis: int = -1):
    """log(softmax(x)) computed directly, so it never takes log(0)."""
    from houssem_torch.backends.cpu import kernels as cpu
    xv = x.numpy()
    if _last_axis(axis, xv) and fast.use(xv):
        log_probs = fast.log_softmax_forward(fast.rows(xv))  # saved
        out = cpu.CPUStorage(log_probs.reshape(xv.shape))

        def _backward(g):
            dx = fast.log_softmax_backward(fast.rows(g), log_probs)
            x._accumulate_grad(dx.reshape(xv.shape))
    else:
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
