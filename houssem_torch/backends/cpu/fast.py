"""Fast CPU kernels: fused, multi-core loops compiled with Numba.

The NumPy code in kernels.py and ops/grads.py is the reference: short,
readable, and the source of truth. Each kernel here computes the same thing
in one pass over the data, with one output array and the rows spread over
all CPU cores. tests/test_cpu_fast.py checks every kernel against its
reference.

Numba is optional (`pip install houssem_torch[fusion]`). Without it, or with
the environment variable HOUSSEM_TORCH_NUMBA=0, every op uses the NumPy
reference. `use(array)` tells an op whether to take the fast path: Numba is
on, the dtype is float32/float64, and the array is big enough that thread
start-up costs less than it saves.

Shapes: row kernels take 2-D C-contiguous (rows, d) arrays and work along the
last axis; the op reshapes (..., d) to (rows, d) before calling. Elementwise
kernels take 1-D contiguous arrays.

Accumulations (row sums, the loss) run in float64, then round once.
"""
from __future__ import annotations
import math
import os
import numpy as np

try:
    import numba
    from numba import njit, prange
except ImportError:  # pragma: no cover - exercised only without numba
    numba = None
    prange = range

    def njit(*_args, **_kwargs):  # the kernels stay plain (unused) Python
        return lambda fn: fn

# Numba's default "omp" threading layer keeps its threads spinning after each
# loop, fighting OpenBLAS's threads (matmuls ran up to 2x slower between
# fast kernels). "workqueue" threads sleep instead. NUMBA_THREADING_LAYER in
# the environment still wins.
if numba is not None and "NUMBA_THREADING_LAYER" not in os.environ:
    numba.config.THREADING_LAYER = "workqueue"

MIN_ELEMENTS = 4096  # below this, thread start-up costs more than it saves
_FLOATS = (np.dtype(np.float32), np.dtype(np.float64))
_enabled = numba is not None and os.environ.get("HOUSSEM_TORCH_NUMBA", "1") != "0"


def available() -> bool:
    """True when Numba is installed (the fast kernels can run at all)."""
    return numba is not None


def enabled() -> bool:
    """True when ops take the fast path."""
    return _enabled


def set_enabled(flag: bool) -> bool:
    """Turn the fast path on/off at runtime; returns the previous setting."""
    global _enabled
    previous = _enabled
    _enabled = bool(flag) and numba is not None
    return previous


def use(*arrays) -> bool:
    """Should an op take the fast path for these input arrays?"""
    return (_enabled and arrays[0].size >= MIN_ELEMENTS
            and all(a.dtype in _FLOATS for a in arrays))


def rows(a: np.ndarray) -> np.ndarray:
    """(..., d) -> C-contiguous (rows, d) view (copies only if needed)."""
    return np.ascontiguousarray(a.reshape(-1, a.shape[-1]))


def flat(a: np.ndarray) -> np.ndarray:
    """Any shape -> C-contiguous 1-D view (copies only if needed)."""
    return np.ascontiguousarray(a).reshape(-1)


def _jit(fn):
    """Compile a multi-core kernel (cached on disk after the first run)."""
    return njit(parallel=True, cache=True)(fn)


_GELU_C = math.sqrt(2.0 / math.pi)
_GELU_K = 0.044715


# ---------------------------------------------------------------------------
# GELU (tanh approximation), elementwise
# ---------------------------------------------------------------------------

@_jit
def gelu_forward(x):
    out = np.empty_like(x)
    for i in prange(x.size):
        v = x[i]
        out[i] = 0.5 * v * (1.0 + math.tanh(_GELU_C * (v + _GELU_K * v * v * v)))
    return out


@_jit
def gelu_backward(g, x):
    dx = np.empty_like(x)
    for i in prange(x.size):
        v = x[i]
        t = math.tanh(_GELU_C * (v + _GELU_K * v * v * v))
        du = _GELU_C * (1.0 + 3.0 * _GELU_K * v * v)
        dx[i] = g[i] * 0.5 * ((1.0 + t) + v * (1.0 - t * t) * du)
    return dx


# ---------------------------------------------------------------------------
# Softmax family, along the last axis of (rows, d)
# ---------------------------------------------------------------------------

@_jit
def softmax_forward(x):
    n, d = x.shape
    out = np.empty_like(x)
    for r in prange(n):
        m = x[r, 0]
        for j in range(1, d):
            if x[r, j] > m:
                m = x[r, j]
        s = 0.0
        for j in range(d):
            e = math.exp(x[r, j] - m)
            out[r, j] = e
            s += e
        inv = 1.0 / s
        for j in range(d):
            out[r, j] *= inv
    return out


@_jit
def softmax_backward(g, p):
    n, d = p.shape
    dx = np.empty_like(p)
    for r in prange(n):
        dot = 0.0
        for j in range(d):
            dot += g[r, j] * p[r, j]
        for j in range(d):
            dx[r, j] = p[r, j] * (g[r, j] - dot)
    return dx


@_jit
def log_softmax_forward(x):
    n, d = x.shape
    out = np.empty_like(x)
    for r in prange(n):
        m = x[r, 0]
        for j in range(1, d):
            if x[r, j] > m:
                m = x[r, j]
        s = 0.0
        for j in range(d):
            s += math.exp(x[r, j] - m)
        lse = m + math.log(s)
        for j in range(d):
            out[r, j] = x[r, j] - lse
    return out


@_jit
def log_softmax_backward(g, log_probs):
    n, d = g.shape
    dx = np.empty_like(g)
    for r in prange(n):
        sg = 0.0
        for j in range(d):
            sg += g[r, j]
        for j in range(d):
            dx[r, j] = g[r, j] - math.exp(log_probs[r, j]) * sg
    return dx


# ---------------------------------------------------------------------------
# Cross-entropy straight from the logits (never builds log_probs)
# ---------------------------------------------------------------------------

@_jit
def cross_entropy_forward(logits, targets, valid, label_smoothing,
                          non_target=False, pad=-1):
    """Returns (lse (N,) float64, per-row losses (N,) float64).

    uniform:    loss_n = (1-ε)·(lse - x[t]) + ε·(lse - mean(x))
    non_target: loss_n = (1-ε)·(lse - x[t]) + ε·(lse - mean of x over the K
                classes other than t and pad)        (pad = -1: none)
    0 for ignored rows.
    """
    n, v = logits.shape
    lse = np.empty(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    for r in prange(n):
        m = logits[r, 0]
        for j in range(1, v):
            if logits[r, j] > m:
                m = logits[r, j]
        s = 0.0
        total = 0.0
        for j in range(v):
            x = logits[r, j]
            s += math.exp(x - m)
            total += x
        lse_r = m + math.log(s)
        lse[r] = lse_r
        if valid[r]:
            t = targets[r]
            nll = lse_r - logits[r, t]
            if non_target:
                k = v - 1
                total -= logits[r, t]
                if pad >= 0 and pad != t:
                    k -= 1
                    total -= logits[r, pad]
                smooth = lse_r - total / max(k, 1)
            else:
                smooth = lse_r - total / v
            losses[r] = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    return lse, losses


@_jit
def cross_entropy_backward(logits, lse, targets, valid, scale, label_smoothing,
                           non_target=False, pad=-1):
    """dL/dlogits = (softmax - q) · scale on valid rows, 0 elsewhere.

    uniform:    q = ε/V everywhere plus (1-ε) at the target.
    non_target: q = (1-ε) at the target, 0 at pad, ε/K on the K others.
    scale = g / count.
    """
    n, v = logits.shape
    out = np.empty_like(logits)
    for r in prange(n):
        if not valid[r]:
            for j in range(v):
                out[r, j] = 0.0
            continue
        t = targets[r]
        if non_target:
            k = v - 1
            if pad >= 0 and pad != t:
                k -= 1
            floor = label_smoothing / max(k, 1)
        else:
            floor = label_smoothing / v
        l_r = lse[r]
        for j in range(v):
            out[r, j] = (math.exp(logits[r, j] - l_r) - floor) * scale
        if non_target:
            out[r, t] += (floor - (1.0 - label_smoothing)) * scale
            if pad >= 0 and pad != t:
                out[r, pad] += floor * scale
        else:
            out[r, t] -= (1.0 - label_smoothing) * scale
    return out


# ---------------------------------------------------------------------------
# LayerNorm over the last axis of (rows, d)
# ---------------------------------------------------------------------------

@_jit
def layer_norm_forward(x, weight, bias, eps):
    """Returns (out, x_hat, rstd (rows, 1)), like kernels.layer_norm."""
    n, d = x.shape
    out = np.empty_like(x)
    x_hat = np.empty_like(x)
    rstd = np.empty((n, 1), dtype=x.dtype)
    for r in prange(n):
        mean = 0.0
        for j in range(d):
            mean += x[r, j]
        mean /= d
        var = 0.0
        for j in range(d):
            c = x[r, j] - mean
            var += c * c
        inv = 1.0 / math.sqrt(var / d + eps)
        rstd[r, 0] = inv
        for j in range(d):
            h = (x[r, j] - mean) * inv
            x_hat[r, j] = h
            out[r, j] = h * weight[j] + bias[j]
    return out, x_hat, rstd


@_jit
def layer_norm_backward(g, x_hat, rstd, weight):
    """Returns (dx, dweight, dbias), like grads.layer_norm_backward.

    Rows are split into chunks; each chunk sums its own dweight/dbias
    partials (no two threads write the same memory), added up at the end.
    """
    n, d = g.shape
    dx = np.empty_like(g)
    chunks = min(n, 64)
    part_w = np.zeros((chunks, d), dtype=np.float64)
    part_b = np.zeros((chunks, d), dtype=np.float64)
    for c in prange(chunks):
        start = c * n // chunks
        stop = (c + 1) * n // chunks
        for r in range(start, stop):
            mean_gh = 0.0
            mean_ghx = 0.0
            for j in range(d):
                gj = g[r, j]
                hj = x_hat[r, j]
                gh = gj * weight[j]
                mean_gh += gh
                mean_ghx += gh * hj
                part_w[c, j] += gj * hj
                part_b[c, j] += gj
            mean_gh /= d
            mean_ghx /= d
            inv = rstd[r, 0]
            for j in range(d):
                hj = x_hat[r, j]
                dx[r, j] = inv * (g[r, j] * weight[j] - mean_gh - hj * mean_ghx)
    dweight = np.empty(d, dtype=g.dtype)
    dbias = np.empty(d, dtype=g.dtype)
    for j in range(d):
        sw = 0.0
        sb = 0.0
        for c in range(chunks):
            sw += part_w[c, j]
            sb += part_b[c, j]
        dweight[j] = sw
        dbias[j] = sb
    return dx, dweight, dbias


# ---------------------------------------------------------------------------
# Embedding backward: scatter-add rows (sequential; repeated ids would race)
# ---------------------------------------------------------------------------

@njit(cache=True)
def embedding_backward(g, ids, vocab_size):
    n, d = g.shape
    grad = np.zeros((vocab_size, d), dtype=g.dtype)
    for i in range(n):
        row = ids[i]
        for j in range(d):
            grad[row, j] += g[i, j]
    return grad


# ---------------------------------------------------------------------------
# Adam / AdamW update, in place, one pass over each parameter
# ---------------------------------------------------------------------------

@_jit
def adam_step(p, g, m, v, lr, b1, b2, bc1, bc2, eps, l2):
    """In place on 1-D p, m, v (g read-only). Same formula as optim.Adam:

        grad = g + l2·p   (classic L2 penalty; l2 = 0 for none / AdamW)
        m = b1·m + (1-b1)·grad,  v = b2·v + (1-b2)·grad²
        p -= lr · (m / bc1) / (sqrt(v / bc2) + eps)
    """
    for i in prange(p.size):
        gi = g[i] + l2 * p[i]
        mi = b1 * m[i] + (1.0 - b1) * gi
        vi = b2 * v[i] + (1.0 - b2) * gi * gi
        m[i] = mi
        v[i] = vi
        p[i] -= lr * (mi / bc1) / (math.sqrt(vi / bc2) + eps)
