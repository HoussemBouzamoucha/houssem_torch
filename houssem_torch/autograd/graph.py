"""micrograd-style graph building, on whole NumPy arrays instead of scalars.

Every op computes its forward value, then hands `result()` a `_backward(g)`
closure that you write by hand. `g` is the NumPy gradient of the loss w.r.t.
the op's output; the closure pushes gradients into its parents with
`parent._accumulate_grad(...)`, which is a no-op for parents that don't
require grad. See docs/autograd.md.
"""
from __future__ import annotations
import numpy as np
from houssem_torch.autograd.grad_mode import is_grad_enabled


def result(storage, ref, parents, op: str, backward_fn):
    """Wrap `storage` in a Tensor and attach `backward_fn` if grad is needed."""
    from houssem_torch.tensor import Tensor

    out = Tensor(_storage=storage, device=ref.device, dtype=ref.dtype)
    if is_grad_enabled():
        prev = tuple(p for p in parents if getattr(p, "requires_grad", False))
        if prev:
            out.requires_grad = True
            out._prev = prev
            out._op = op
            out._backward = backward_fn
    return out


def backward_todo(op: str):
    """Raise from a `_backward` whose math hasn't been written yet."""
    raise NotImplementedError(
        f"backward for '{op}' is not written yet: the forward pass works, "
        f"but gradients can't flow through '{op}' until its _backward is filled in."
    )


def unbroadcast(g: np.ndarray, shape) -> np.ndarray:
    """Sum `g` down to `shape`, undoing NumPy broadcasting, in one reduction."""
    shape = tuple(shape)
    if g.shape == shape:
        return g
    extra = g.ndim - len(shape)
    axes = tuple(range(extra)) + tuple(
        extra + i for i, size in enumerate(shape)
        if size == 1 and g.shape[extra + i] != 1
    )
    return g.sum(axis=axes, keepdims=True).reshape(shape)
