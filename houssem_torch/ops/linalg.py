from __future__ import annotations
import numpy as np
from houssem_torch.autograd.graph import result, unbroadcast


def matmul(a, b):
    """Matrix product of tensors with ndim >= 2.

    Leading (batch) dimensions broadcast like NumPy, e.g.
    (B, H, T, D) @ (B, H, D, T) or (B, T, C) @ (C, N).
    """
    from houssem_torch.backends.cpu import kernels as cpu

    if a.ndim < 2 or b.ndim < 2:
        raise ValueError(
            "matmul requires tensors with at least 2 dims, "
            f"got {a.ndim}D and {b.ndim}D."
        )
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"matmul shape mismatch: {a.shape} @ {b.shape}.")
    try:
        np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    except ValueError:
        raise ValueError(
            f"matmul batch dims do not broadcast: {a.shape} @ {b.shape}."
        ) from None

    def _backward(g):
        # out = a @ b  ->  da = g @ b^T,  db = a^T @ g   (^T = swap last 2 dims)
        # then sum over any batch dims that were broadcast.
        av, bv = a.numpy(), b.numpy()
        a._accumulate_grad(unbroadcast(g @ np.swapaxes(bv, -1, -2), a.shape))
        b._accumulate_grad(unbroadcast(np.swapaxes(av, -1, -2) @ g, b.shape))

    out = cpu.matmul(a._storage, b._storage)
    return result(out, a, (a, b), "matmul", _backward)
