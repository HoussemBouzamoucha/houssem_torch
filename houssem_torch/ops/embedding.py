from __future__ import annotations
import numpy as np
from houssem_torch.autograd.graph import result
from houssem_torch.ops import grads


def _as_index_array(ids) -> np.ndarray:
    arr = ids.numpy() if hasattr(ids, "_storage") else np.asarray(ids)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"token ids must be integers, got dtype {arr.dtype}.")
    return arr


def embedding(ids, weight):
    """Look up rows of `weight` by token id.

    ids: (...) integers in [0, V)   weight: (V, d)   ->   (..., d)
    `ids` never receives a gradient. If the same `weight` is used several
    times (shared encoder/decoder/output table), its gradients accumulate.
    """
    from houssem_torch.backends.cpu import kernels as cpu

    idx = _as_index_array(ids)  # saved for backward
    vocab = weight.shape[0]
    if idx.size and (idx.min() < 0 or idx.max() >= vocab):
        raise IndexError(
            f"token id out of range [0, {vocab}): min {idx.min()}, max {idx.max()}."
        )
    out = cpu.embedding(weight._storage, idx)

    def _backward(g):
        weight._accumulate_grad(grads.embedding_backward(g, idx, vocab))

    return result(out, weight, (weight,), "embedding", _backward)
