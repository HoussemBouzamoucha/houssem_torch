from __future__ import annotations
import numpy as np
from houssem_torch.autograd.graph import result, unbroadcast
from houssem_torch.ops import grads
from houssem_torch.fusion.engine import try_fuse


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


# Each `_backward(g)` receives g = dLoss/dout and pushes dLoss/dinput into
# each input. Input values are read with `.numpy()` at backward time.

def add(a, b):
    a, b = _pair(a, b)

    def _backward(g):
        # d(a+b)/da = 1, d(a+b)/db = 1
        a._accumulate_grad(unbroadcast(g, a.shape))
        b._accumulate_grad(unbroadcast(g, b.shape))

    return result(_compute("add", a, b), a, (a, b), "add", _backward)


def subtract(a, b):
    a, b = _pair(a, b)

    def _backward(g):
        # d(a-b)/da = 1, d(a-b)/db = -1
        a._accumulate_grad(unbroadcast(g, a.shape))
        b._accumulate_grad(unbroadcast(-g, b.shape))

    return result(_compute("subtract", a, b), a, (a, b), "subtract", _backward)


def multiply(a, b):
    a, b = _pair(a, b)

    def _backward(g):
        # d(a*b)/da = b, d(a*b)/db = a
        a._accumulate_grad(unbroadcast(g * b.numpy(), a.shape))
        b._accumulate_grad(unbroadcast(g * a.numpy(), b.shape))

    return result(_compute("multiply", a, b), a, (a, b), "multiply", _backward)


def divide(a, b):
    a, b = _pair(a, b)

    def _backward(g):
        # d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
        bv = b.numpy()
        a._accumulate_grad(unbroadcast(g / bv, a.shape))
        b._accumulate_grad(unbroadcast(-g * a.numpy() / (bv * bv), b.shape))

    return result(_compute("divide", a, b), a, (a, b), "divide", _backward)


def negate(a):
    def _backward(g):
        a._accumulate_grad(-g)

    return result(_compute("negate", a), a, (a,), "negate", _backward)


def relu(a):
    def _backward(g):
        # d relu(a)/da = 1 where a > 0, else 0
        a._accumulate_grad(g * (a.numpy() > 0))

    return result(_compute("relu", a), a, (a,), "relu", _backward)


def exp(a):
    out = _compute("exp", a)

    def _backward(g):
        # d exp(a)/da = exp(a) = out (reuse the forward value)
        a._accumulate_grad(g * out.array)

    return result(out, a, (a,), "exp", _backward)


def log(a):
    def _backward(g):
        # d log(a)/da = 1/a
        a._accumulate_grad(g / a.numpy())

    return result(_compute("log", a), a, (a,), "log", _backward)


# --- Backward math lives in ops/grads.py ------------------------------------

def sqrt(a):
    from houssem_torch.backends.cpu import kernels as cpu
    out = cpu.sqrt(a._storage)

    def _backward(g):
        a._accumulate_grad(grads.sqrt_backward(g, out.array))

    return result(out, a, (a,), "sqrt", _backward)


def pow(a, exponent: float):
    """a ** exponent for a Python-number exponent."""
    from houssem_torch.backends.cpu import kernels as cpu
    if hasattr(exponent, "_storage"):
        raise TypeError("pow only supports a scalar (Python number) exponent.")
    out = cpu.pow(a._storage, exponent)

    def _backward(g):
        a._accumulate_grad(grads.pow_backward(g, a.numpy(), exponent))

    return result(out, a, (a,), "pow", _backward)


def masked_fill(a, mask, value: float):
    """Replace a's elements with `value` wherever `mask` is true.

    `mask` (bool/0-1 array or Tensor) must broadcast to a.shape; the output
    has a's shape. Used for padding masks and the decoder's causal mask
    (value = -inf before softmax).
    """
    from houssem_torch.backends.cpu import kernels as cpu
    m = mask.numpy() if hasattr(mask, "_storage") else np.asarray(mask)
    m = m.astype(bool, copy=False)
    try:
        np.broadcast_to(m, a.shape)
    except ValueError:
        raise ValueError(
            f"mask shape {m.shape} does not broadcast to tensor shape {a.shape}."
        ) from None
    out = cpu.masked_fill(a._storage, m, value)  # `m` saved for backward

    def _backward(g):
        a._accumulate_grad(grads.masked_fill_backward(g, m))

    return result(out, a, (a,), "masked_fill", _backward)
