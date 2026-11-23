from __future__ import annotations
import numpy as np
from houssem_torch.autograd.graph import result
from houssem_torch.backends.cpu import fast
from houssem_torch.ops import grads
from houssem_torch.ops.embedding import _as_index_array


SMOOTHING_MODES = ("uniform", "non_target")


def cross_entropy(logits, targets, label_smoothing: float = 0.0,
                  ignore_index: int | None = None,
                  label_smoothing_mode: str = "uniform"):
    """Mean cross-entropy over all non-ignored positions (fused log-softmax).

    logits: (..., V)   targets: (...) integers   ->   scalar ()

    Label smoothing ε moves ε of the target probability onto other classes.
    label_smoothing_mode says where it goes:

    "uniform" (PyTorch's definition): ε/V on every class, so the target
        distribution is (1 - ε)·one_hot(target) + ε/V and
            loss_i = (1 - ε)·(-log p[target]) + ε·mean_c(-log p[c]).
    "non_target" (Tensor2Tensor-style): the target keeps exactly 1 - ε, the
        ignore_index class (padding) gets 0, and the K remaining classes
        get ε/K each (K = V - 2 with a padding class, V - 1 without), so
            loss_i = (1 - ε)·(-log p[target]) + ε·mean_{c in others}(-log p[c]).
        The model is never taught to put probability on padding.

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
    if label_smoothing_mode not in SMOOTHING_MODES:
        raise ValueError(f"label_smoothing_mode must be one of {SMOOTHING_MODES}, "
                         f"got {label_smoothing_mode!r}.")
    non_target = label_smoothing_mode == "non_target"
    # The class that receives no smoothing mass: the ignore_index, if it is
    # a real class id.
    in_range = ignore_index is not None and 0 <= ignore_index < vocab
    pad = ignore_index if in_range else None

    t = t.reshape(-1)
    valid = np.ones(t.shape, dtype=bool) if ignore_index is None else t != ignore_index
    safe_t = np.where(valid, t, 0)
    if valid.any() and (safe_t[valid].min() < 0 or safe_t[valid].max() >= vocab):
        raise IndexError(f"target out of range [0, {vocab}).")

    if fast.use(logits.numpy()):
        return _cross_entropy_fast(logits, safe_t, valid, label_smoothing,
                                   non_target, pad)

    log_probs = cpu.log_softmax(logits._storage, axis=-1).array.reshape(-1, vocab)
    rows = np.arange(t.size)
    nll = -log_probs[rows, safe_t]
    if label_smoothing and non_target:
        others = log_probs.sum(axis=1) - log_probs[rows, safe_t]
        if pad is not None:
            others -= np.where(safe_t != pad, log_probs[:, pad], 0.0)
        k = grads.non_target_counts(vocab, safe_t, pad)
        per_token = (1.0 - label_smoothing) * nll + label_smoothing * -others / k
    elif label_smoothing:
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
                                               count, label_smoothing,
                                               non_target, pad)
        logits._accumulate_grad(dlogits.reshape(logits.shape))

    return result(out, logits, (logits,), "cross_entropy", _backward)


def _cross_entropy_fast(logits, safe_t, valid, label_smoothing, non_target, pad):
    """Same loss in two fused passes, straight from the logits.

    Saves only the logits and one log-sum-exp per row for backward, instead
    of a full (N, V) log_probs array.
    """
    from houssem_torch.backends.cpu import kernels as cpu

    x = fast.rows(logits.numpy())
    targets = np.ascontiguousarray(safe_t, dtype=np.int64)
    valid = np.ascontiguousarray(valid)
    pad = -1 if pad is None else int(pad)
    lse, losses = fast.cross_entropy_forward(x, targets, valid, label_smoothing,
                                             non_target, pad)
    count = int(valid.sum())
    loss = losses.sum() / max(count, 1)
    out = cpu.CPUStorage(np.asarray(loss, dtype=logits.dtype.numpy_name))

    def _backward(g):
        scale = float(g) / max(count, 1)
        dlogits = fast.cross_entropy_backward(x, lse, targets, valid, scale,
                                              label_smoothing, non_target, pad)
        logits._accumulate_grad(dlogits.reshape(logits.shape))

    return result(out, logits, (logits,), "cross_entropy", _backward)
