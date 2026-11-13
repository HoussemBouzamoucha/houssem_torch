from __future__ import annotations
from houssem_torch.autograd.function import Function


def _ones(shape, ref):
    from houssem_torch.ops.creation import ones
    return ones(shape, dtype=ref.dtype, device=ref.device)


def unbroadcast(grad, shape):
    """Sum `grad` down to `shape` (undo broadcasting)."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class AddBackward(Function):
    def backward(self, g):
        a, b = self.inputs
        return unbroadcast(g, a.shape), unbroadcast(g, b.shape)


class SubBackward(Function):
    def backward(self, g):
        a, b = self.inputs
        return unbroadcast(g, a.shape), unbroadcast(-g, b.shape)


class MulBackward(Function):
    def backward(self, g):
        a, b = self.inputs
        return unbroadcast(g * b, a.shape), unbroadcast(g * a, b.shape)


class DivBackward(Function):
    def backward(self, g):
        a, b = self.inputs
        return (unbroadcast(g / b, a.shape),
                unbroadcast(-(g * a) / (b * b), b.shape))


class NegBackward(Function):
    def backward(self, g):
        return (-g,)


class ExpBackward(Function):
    def backward(self, g):
        return (g * self.out,)


class LogBackward(Function):
    def backward(self, g):
        (a,) = self.inputs
        return (g / a,)


class ReluBackward(Function):
    def backward(self, g):
        from houssem_torch.tensor import Tensor
        (a,) = self.inputs
        mask = Tensor(a.numpy() > 0, device=a.device, dtype=a.dtype)  # CPU only
        return (g * mask,)


class MatmulBackward(Function):
    def backward(self, g):
        a, b = self.inputs
        return g @ b.transpose(), a.transpose() @ g


class SumBackward(Function):
    """Also used for mean, via `scale` = 1 / count."""

    def __init__(self, a, axis=None, scale=1.0):
        super().__init__(a)
        self.axis = axis
        self.scale = scale

    def backward(self, g):
        (a,) = self.inputs
        if self.axis is None:
            kept = (1,) * a.ndim
        else:
            axes = (self.axis,) if isinstance(self.axis, int) else tuple(self.axis)
            axes = tuple(ax % a.ndim for ax in axes)
            kept = tuple(1 if i in axes else s for i, s in enumerate(a.shape))
        out = g.reshape(kept) * _ones(a.shape, a)
        return (out * self.scale if self.scale != 1.0 else out,)