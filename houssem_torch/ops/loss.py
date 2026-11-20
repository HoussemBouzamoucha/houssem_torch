from __future__ import annotations
import numpy as np
from houssem_torch.autograd.graph import result
from houssem_torch.backends.cpu import fast
from houssem_torch.ops import grads
from houssem_torch.ops.embedding import _as_index_array


def cross_entropy(logits, targets, label_smoothing: float = 0.0,
                  ignore_index: int | None = None):
    """Mean cross-entropy over all non-ignored positions (fused log-softmax).

    logits: (..., V)   targets: (...) integers   ->   scalar ()

    With label smoothing ε the target distribution is
    (1 - ε)·one_hot(target) + ε/V on every class (PyTorch's definition), so
        loss_i = (1 - ε)·(-log p[target]) + ε·mean_c(-log p[c]).
    Positions where target == ignore_index (padding) contribute nothing and
    are excluded from the mean.
    """
    from houssem_torch.backends.cpu import kernels as cpu

    vocab = logits.shape[-1]
    t = _as_index_array(targets)
    if t.shape != logits.shape[:-1]:
        raise ValueError(
            f"targets shape {t.shape} must equal logits shape without the "
            f"last axis {logits.shape[:-1]}."
        )
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}.")

    t = t.reshape(-1)
    valid = np.ones(t.shape, dtype=bool) if ignore_index is None else t != ignore_index
    safe_t = np.where(valid, t, 0)
    if valid.any() and (safe_t[valid].min() < 0 or safe_t[valid].max() >= vocab):
        raise IndexError(f"target out of range [0, {vocab}).")

    if fast.use(logits.numpy()):
        return _cross_entropy_fast(logits, safe_t, valid, label_smoothing)

    log_probs = cpu.log_softmax(logits._storage, axis=-1).array.reshape(-1, vocab)
    nll = -log_probs[np.arange(t.size), safe_t]
    if label_smoothing:
        per_token = (1.0 - label_smoothing) * nll \
            + label_smoothing * -log_probs.mean(axis=1)
    else:
        per_token = nll
    count = int(valid.sum())  # saved for backward
    loss = (per_token * valid).sum() / max(count, 1)
    out = cpu.CPUStorage(np.asarray(loss, dtype=logits.dtype.numpy_name))
    # saved for backward: log_probs (N, V), safe_t (N,), valid (N,), count

    def _backward(g):
        dlogits = grads.cross_entropy_backward(g, log_probs, safe_t, valid,
                                               count, label_smoothing)
        logits._accumulate_grad(dlogits.reshape(logits.shape))

    return result(out, logits, (logits,), "cross_entropy", _backward)


def _cross_entropy_fast(logits, safe_t, valid, label_smoothing):
    """Same loss in two fused passes, straight from the logits.

    Saves only the logits and one log-sum-exp per row for backward, instead
    of a full (N, V) log_probs array.
    """
    from houssem_torch.backends.cpu import kernels as cpu

    x = fast.rows(logits.numpy())
    targets = np.ascontiguousarray(safe_t, dtype=np.int64)
    valid = np.ascontiguousarray(valid)
    lse, losses = fast.cross_entropy_forward(x, targets, valid, label_smoothing)
    count = int(valid.sum())
    loss = losses.sum() / max(count, 1)
    out = cpu.CPUStorage(np.asarray(loss, dtype=logits.dtype.numpy_name))

    def _backward(g):
        scale = float(g) / max(count, 1)
        dlogits = fast.cross_entropy_backward(x, lse, targets, valid, scale,
                                              label_smoothing)
        logits._accumulate_grad(dlogits.reshape(logits.shape))

    return result(out, logits, (logits,), "cross_entropy", _backward)
