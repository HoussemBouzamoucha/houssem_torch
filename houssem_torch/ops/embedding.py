from __future__ import annotations
import numpy as np
from houssem_torch.autograd.graph import result
from houssem_torch.backends.cpu import fast
from houssem_torch.ops import grads


def _as_index_array(ids) -> np.ndarray:
    arr = ids.numpy() if hasattr(ids, "_storage") else np.asarray(ids)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"token ids must be integers, got dtype {arr.dtype}.")
    return arr


def embedding(ids, weight, padding_idx: int | None = None):
    """Look up rows of `weight` by token id.

    ids: (...) integers in [0, V)   weight: (V, d)   ->   (..., d)
    `ids` never receives a gradient. If the same `weight` is used several
    times (shared encoder/decoder/output table), its gradients accumulate.

    padding_idx: this lookup sends no gradient to that row (PyTorch's
    `padding_idx`). Other uses of the same table, such as a tied output
    projection, still do.
    """
    from houssem_torch.backends.cpu import kernels as cpu

    idx = _as_index_array(ids)  # saved for backward
    vocab = weight.shape[0]
    if idx.size and (idx.min() < 0 or idx.max() >= vocab):
        raise IndexError(
            f"token id out of range [0, {vocab}): min {idx.min()}, max {idx.max()}."
        )
    if padding_idx is not None and not 0 <= padding_idx < vocab:
        raise IndexError(f"padding_idx {padding_idx} out of range [0, {vocab}).")
    out = cpu.embedding(weight._storage, idx)

    def _backward(g):
        if fast.use(g):
            d = g.shape[-1]
            ids = np.ascontiguousarray(idx.reshape(-1), dtype=np.int64)
            dweight = fast.embedding_backward(fast.rows(g), ids, vocab)
            dweight = dweight.reshape(vocab, d)
        else:
            dweight = grads.embedding_backward(g, idx, vocab)
        if padding_idx is not None:
            dweight[padding_idx] = 0
        weight._accumulate_grad(dweight)

    return result(out, weight, (weight,), "embedding", _backward)
