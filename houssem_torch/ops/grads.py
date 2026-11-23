"""Backward math for the transformer ops: one pure NumPy function per op.

Every function receives `g`, the gradient of the loss with respect to the op's
OUTPUT, plus whatever the forward pass saved, and returns the gradient with
respect to the op's INPUT(S). No Tensors, no graph: just arrays in, arrays out.
The ops in ops/*.py call these and accumulate the results for you.

Rules for every function:
  * Return exactly the input's shape (a wrong shape raises an error naming
    the op).
  * Keep float32: build new arrays with `dtype=g.dtype`, or `np.zeros_like`.
  * Check yourself: `pytest tests/test_transformer_ops.py -k grad -v`
      SKIPPED = not written yet, PASSED = matches PyTorch, FAILED = wrong.

Suggested order (easiest first): pow, sqrt, masked_fill, dropout, getitem,
embedding, gelu, log_softmax, softmax, cross_entropy, layer_norm.
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# 1. Single-variable derivatives
# ---------------------------------------------------------------------------

def pow_backward(g: np.ndarray, a: np.ndarray, exponent: float) -> np.ndarray:
    """out = a ** exponent, elementwise.

    g: same shape as a. Return: dL/da, same shape as a.

    What: the chain rule, dL/da = g * d(out)/da.
    How:  write down d(a^n)/da for a constant n. One line.
    """
    return g * exponent * np.power(a, exponent - 1, dtype=g.dtype)


def sqrt_backward(g: np.ndarray, out: np.ndarray) -> np.ndarray:
    """out = sqrt(a), elementwise. `out` is the saved forward result.

    g: same shape as out. Return: dL/da, same shape.

    What: g * d sqrt(a)/da.
    How:  differentiate a^(1/2), then express the result using `out`
          instead of `a` (cheaper: no second sqrt).
    """
    return g * np.power(out, -1, dtype=g.dtype) * 0.5


# ---------------------------------------------------------------------------
# 2. Masks: gradients that pass through or get blocked
# ---------------------------------------------------------------------------

def masked_fill_backward(g: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """out = where(mask, value, a). `mask` is bool and broadcasts to g's shape.

    g: shape of a. Return: dL/da, shape of a.

    What: where the mask replaced `a` with a constant, `a` had no influence
          on the output. Everywhere else the output IS `a`.
    How:  a single np.where. Its output must have g's full shape even when
          `mask` is smaller (e.g. (B,1,1,S) against (B,h,T,S)).
    """
    # Masked positions were overwritten by a constant -> gradient 0 there.
    # np.where broadcasts the smaller mask against g's full shape.
    return np.where(mask, g.dtype.type(0), g)


def dropout_backward(g: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """out = x * mask, where mask holds 0 (dropped) or 1/(1-p) (kept).

    g, mask: shape of x. Return: dL/dx, shape of x.

    What: the forward pass is a multiplication by a constant array.
    How:  what is d(x * c)/dx for a constant c?
    """
    return g * mask


# ---------------------------------------------------------------------------
# 3. Scatter: sending gradients back to where values were read from
# ---------------------------------------------------------------------------

def getitem_backward(g: np.ndarray, input_shape: tuple, index) -> np.ndarray:
    """out = a[index], any NumPy index (slices, ints, integer arrays, ...).

    g: shape of a[index]. Return: dL/da, shape `input_shape`.

    What: each output element was read from one element of `a`; its gradient
          goes back to that element. Elements never read get 0. An element
          read twice (e.g. index [2, 0, 2]) gets the SUM of both gradients.
    How:  start from zeros of `input_shape`, then scatter-add g at `index`.
          Careful: `grad[index] += g` silently drops repeats with integer
          array indices; look at `np.add.at`.
    Bonus: `split` is built on getitem, so this also makes split work.
    """
    grad = np.zeros(input_shape, dtype=g.dtype)
    if _is_basic_index(index):
        grad[index] = g        # slices/ints never repeat an element: exact
    else:
        np.add.at(grad, index, g)  # in place, sums repeated indices
    return grad


def _is_basic_index(index) -> bool:
    """True for slices, ints, None and Ellipsis only (no arrays or lists)."""
    parts = index if isinstance(index, tuple) else (index,)
    return all(p is None or p is Ellipsis or isinstance(p, (slice, int, np.integer))
               for p in parts)


def embedding_backward(g: np.ndarray, ids: np.ndarray,
                       vocab_size: int) -> np.ndarray:
    """out = weight[ids]: picks one row of weight (V, d) per token id.

    g: (..., d), one gradient row per token. ids: (...) integers.
    Return: dL/dweight, shape (vocab_size, d).

    What: the same idea as getitem. Row v of the result is the sum of the
          g rows of every position whose token id is v. A token used 50 times
          in the batch gets 50 rows added; unused tokens get zeros.
    How:  flatten ids to (N,) and g to (N, d), then scatter-add into zeros.
    """
    d = g.shape[-1]
    grad = np.zeros((vocab_size, d), dtype=g.dtype)
    np.add.at(grad, ids.reshape(-1), g.reshape(-1, d))  # in place
    return grad


# ---------------------------------------------------------------------------
# 4. Chain rule through a formula
# ---------------------------------------------------------------------------

def gelu_backward(g: np.ndarray, x: np.ndarray) -> np.ndarray:
    """out = 0.5 · x · (1 + tanh(u)),  with  u = c · (x + 0.044715 · x³),
    c = sqrt(2/π).  (tanh approximation, same as the forward pass.)

    g: shape of x. Return: dL/dx, shape of x.

    What: g * d(out)/dx.
    How:  product rule on 0.5·x·(1 + tanh u), then chain rule through u.
          Useful facts: d tanh(u)/du = 1 - tanh(u)²; du/dx = c·(1 + 3·0.044715·x²).
          Compute tanh(u) once and reuse it.
    """
    c = x.dtype.type(np.sqrt(2.0 / np.pi))
    x2 = x * x                                # x*x is much faster than x**2
    t = np.tanh(c * (x + 0.044715 * x2 * x))  # computed once, reused
    du_dx = c * (1.0 + 3 * 0.044715 * x2)
    # product rule: d/dx [0.5·x·(1 + t)] = 0.5·(1 + t) + 0.5·x·(1 - t²)·du/dx
    return g * 0.5 * ((1.0 + t) + x * (1.0 - t * t) * du_dx)


# ---------------------------------------------------------------------------
# 5. Softmax family: every output depends on every input along `axis`
# ---------------------------------------------------------------------------

def log_softmax_backward(g: np.ndarray, log_probs: np.ndarray,
                         axis: int) -> np.ndarray:
    """out_i = x_i - log(sum_j exp(x_j)) along `axis`. `log_probs` = out.

    g, log_probs: shape of x. Return: dL/dx, shape of x.

    What: out_i depends on x_i directly AND on every x_j through the
          log-sum-exp term.
    How:  first find d(log sum_j exp(x_j))/dx_k: it is a familiar quantity
          you can get from log_probs. Then combine:
          dL/dx_k = sum_i g_i · d out_i / d x_k.
          You'll need one sum over `axis` (keepdims=True).
    """
    return g - np.exp(log_probs) * np.sum(g, axis=axis, keepdims=True)


def softmax_backward(g: np.ndarray, probs: np.ndarray, axis: int) -> np.ndarray:
    """p_i = exp(x_i) / sum_j exp(x_j) along `axis`. `probs` = p.

    g, probs: shape of x. Return: dL/dx, shape of x.

    What: dL/dx_k = sum_i g_i · dp_i/dx_k, where
          dp_i/dx_k = p_i · (δ_ik − p_k)   (δ = 1 if i == k else 0).
    How:  don't build the (n × n) Jacobian. Expand the sum: the δ term and
          the other term each simplify. The result uses one elementwise
          product and one sum over `axis` (keepdims=True).
    """
    # sum_i g_i·p_i·(δ_ik − p_k) = p_k·g_k − p_k·sum_i(g_i·p_i)
    return probs * (g - np.sum(g * probs, axis=axis, keepdims=True))


def cross_entropy_backward(g: np.ndarray, log_probs: np.ndarray,
                           targets: np.ndarray, valid: np.ndarray, count: int,
                           label_smoothing: float, non_target: bool = False,
                           pad: int | None = None) -> np.ndarray:
    """Forward recap (N positions, V classes):
        p = softmax(logits) row-wise, log_probs = log p,       (N, V)
        q = target distribution: (1-ε)·one_hot(target) + ε/V    (N, V)
        loss_n = -sum_v q[n, v] · log_probs[n, v]
        loss   = sum over valid n of loss_n / count

    g: shape () (scalar upstream gradient, usually 1.0).
    targets: (N,) ints (ignored positions already set to 0, safe to index).
    valid: (N,) bool, False at ignored/padding positions.
    Return: dL/dlogits, shape (N, V). The op reshapes it back to (..., V).

    What: the gradient of loss_n with respect to its logits row.
    How:  use log_softmax's result: since sum_v q = 1, dloss_n/dlogits
          collapses to something very short in terms of p and q. Then:
          zero the rows where `valid` is False, divide by `count`
          (use max(count, 1)), and multiply by g.
          Build q with np.full(..., ε/V) and add (1-ε) at the targets.

    non_target=True switches q to the "non_target" smoothing: (1-ε) at the
    target, 0 at `pad` (the ignore_index, or None), and ε spread evenly over
    the K remaining classes. The gradient is still p - q.
    """
    n, vocab = log_probs.shape
    rows = np.arange(n)
    # d loss_n / d logits = p - q   (prediction minus target): subtract the
    # parts of q in place instead of building it.
    dlogits = np.exp(log_probs)
    if non_target:
        w = (label_smoothing / non_target_counts(vocab, targets, pad))
        w = w.astype(dlogits.dtype)
        dlogits -= w[:, None]
        dlogits[rows, targets] += w - (1.0 - label_smoothing)
        if pad is not None:
            other = targets != pad          # pad == target: already handled
            dlogits[rows[other], pad] += w[other]
    else:
        dlogits -= label_smoothing / vocab
        dlogits[rows, targets] -= 1.0 - label_smoothing
    dlogits *= (valid * (g / max(count, 1)))[:, None].astype(dlogits.dtype)
    return dlogits                                # padding rows -> 0


def non_target_counts(vocab: int, targets: np.ndarray, pad: int | None) -> np.ndarray:
    """K per row for "non_target" smoothing: classes other than the target
    and `pad`. (N,) float64, at least 1."""
    k = np.full(targets.shape, vocab - 1, dtype=np.float64)
    if pad is not None:
        k -= targets != pad
    return np.maximum(k, 1.0)


# ---------------------------------------------------------------------------
# 6. LayerNorm: the boss level
# ---------------------------------------------------------------------------

def layer_norm_backward(g: np.ndarray, x_hat: np.ndarray, rstd: np.ndarray,
                        weight: np.ndarray):
    """Forward recap, over the last axis (size d):
        mean = mean(x),  var = mean((x - mean)²),  rstd = 1/sqrt(var + eps)
        x_hat = (x - mean) · rstd
        out = x_hat · weight + bias

    g, x_hat: (..., d).  rstd: (..., 1).  weight: (d,).
    Return: (dx, dweight, dbias) with shapes (..., d), (d,), (d,).

    What and how, three gradients:
      dbias:   bias is added to every position, so sum g over all axes
               except the last.
      dweight: like dbias, but each position contributed weight·x_hat.
      dx:      first let gh = g · weight (the gradient reaching x_hat).
               x_hat depends on x directly, but ALSO through `mean` and
               `var`, which depend on every element of the row. The result
               is rstd times (gh minus two correction terms): one from the
               mean, one from the variance; each is a mean over the last
               axis. Derive them, then check: pytest will tell you.
    """
    d = g.shape[-1]
    dbias = g.reshape(-1, d).sum(axis=0)
    dweight = (g * x_hat).reshape(-1, d).sum(axis=0)

    gh = g * weight  # gradient arriving at x_hat
    # x_hat = (x - mean)·rstd: the direct path gives gh; the path through
    # `mean` subtracts mean(gh); the path through `var` subtracts
    # x_hat·mean(gh·x_hat). Everything is scaled by rstd.
    dx = rstd * (gh - gh.mean(axis=-1, keepdims=True)
                 - x_hat * (gh * x_hat).mean(axis=-1, keepdims=True))
    return dx, dweight, dbias

