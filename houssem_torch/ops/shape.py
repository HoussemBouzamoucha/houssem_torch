"""Shape/view operations. Their gradients just undo the shape change."""
from __future__ import annotations
import numpy as np
from houssem_torch.autograd.graph import result
from houssem_torch.ops import grads


def _normalize_axes(axes, ndim):
    axes = tuple(int(ax) for ax in axes)
    if len(axes) != ndim:
        raise ValueError(f"permutation {axes} does not match {ndim} dimensions.")
    normalized = tuple(ax % ndim if ndim else ax for ax in axes)
    if sorted(normalized) != list(range(ndim)):
        raise ValueError(f"invalid permutation {axes} for {ndim} dimensions.")
    return normalized


def _restore_shape(a):
    """Backward for any op that only changes shape (same element order)."""
    def _backward(g):
        a._accumulate_grad(g.reshape(a.shape))
    return _backward


def reshape(a, shape):
    from houssem_torch.backends.cpu import kernels as cpu
    out = cpu.reshape(a._storage, tuple(shape))
    return result(out, a, (a,), "reshape", _restore_shape(a))


def permute(a, axes=None):
    """Reorder dimensions. ``axes=None`` reverses them (NumPy ``transpose``)."""
    from houssem_torch.backends.cpu import kernels as cpu
    if axes is None:
        axes = tuple(reversed(range(a.ndim)))
    axes = _normalize_axes(axes, a.ndim)

    def _backward(g):
        # apply the inverse permutation to route g back to a's layout
        a._accumulate_grad(np.transpose(g, np.argsort(axes)))

    out = cpu.transpose(a._storage, axes)
    return result(out, a, (a,), "permute", _backward)


def swapaxes(a, axis0: int, axis1: int):
    """Swap two dimensions (PyTorch ``transpose(dim0, dim1)`` semantics)."""
    axes = list(range(a.ndim))
    axes[axis0], axes[axis1] = axes[axis1], axes[axis0]
    return permute(a, axes)


def squeeze(a, axis=None):
    from houssem_torch.backends.cpu import kernels as cpu
    out = cpu.squeeze(a._storage, axis)
    return result(out, a, (a,), "squeeze", _restore_shape(a))


def unsqueeze(a, axis: int):
    from houssem_torch.backends.cpu import kernels as cpu
    out = cpu.unsqueeze(a._storage, axis)
    return result(out, a, (a,), "unsqueeze", _restore_shape(a))


def _normalize_index(index):
    """Tensor indices -> NumPy arrays, so NumPy indexing rules apply."""
    if isinstance(index, tuple):
        return tuple(_normalize_index(i) for i in index)
    return index.numpy() if hasattr(index, "_storage") else index


def getitem(a, index):
    """NumPy-style indexing: a[0], a[:, 1:3], a[..., :d], a[ids], ..."""
    from houssem_torch.backends.cpu import kernels as cpu
    index = _normalize_index(index)  # saved for backward
    out = cpu.getitem(a._storage, index)

    def _backward(g):
        a._accumulate_grad(grads.getitem_backward(g, a.shape, index))

    return result(out, a, (a,), "getitem", _backward)


def split(a, split_size_or_sections, axis: int = -1):
    """Split into pieces along `axis` (PyTorch semantics).

    int    -> pieces of that size (the last may be smaller)
    list   -> pieces with exactly those sizes (must sum to the axis length)
    e.g. q, k, v = split(qkv, d, axis=-1) on a (B, S, 3d) tensor.
    """
    length = a.shape[axis]
    if isinstance(split_size_or_sections, int):
        size = split_size_or_sections
        if size <= 0:
            raise ValueError(f"split size must be positive, got {size}.")
        sizes = [min(size, length - start) for start in range(0, length, size)]
    else:
        sizes = list(split_size_or_sections)
        if sum(sizes) != length:
            raise ValueError(f"split sizes {sizes} do not sum to {length}.")
    axis = axis % a.ndim
    pieces, start = [], 0
    for size in sizes:
        index = (slice(None),) * axis + (slice(start, start + size),)
        pieces.append(getitem(a, index))
        start += size
    return pieces
