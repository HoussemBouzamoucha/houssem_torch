"""The Numba fast kernels must match the NumPy reference (and PyTorch).

Each test runs the same op twice, fast path on and off, at sizes above
fast.MIN_ELEMENTS, and compares the output and every input gradient.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import houssem_torch as ht
from houssem_torch import optim
from houssem_torch.backends.cpu import fast

pytestmark = pytest.mark.skipif(not fast.available(), reason="numba not installed")


@pytest.fixture
def toggle():
    """Yield a function that runs `fn` with the fast path forced on/off."""
    previous = fast.enabled()

    def run(flag, fn):
        fast.set_enabled(flag)
        try:
            return fn()
        finally:
            fast.set_enabled(previous)
    yield run
    fast.set_enabled(previous)


def _forward_backward(op, arrays, dtype=np.float32, upstream_seed=7):
    leaves = [ht.tensor(a.astype(dtype), requires_grad=True) for a in arrays]
    out = op(*leaves)
    if out.ndim == 0:
        out.backward()
    else:  # random upstream gradient, not just ones
        g = np.random.default_rng(upstream_seed).standard_normal(out.shape)
        (out * ht.tensor(g.astype(dtype))).sum().backward()
    return out.numpy(), [leaf.grad.numpy() for leaf in leaves]


def _check_same(toggle, op, arrays, rtol=1e-5, atol=1e-5):
    out_f, grads_f = toggle(True, lambda: _forward_backward(op, arrays))
    out_r, grads_r = toggle(False, lambda: _forward_backward(op, arrays))
    np.testing.assert_allclose(out_f, out_r, rtol=rtol, atol=atol)
    for gf, gr in zip(grads_f, grads_r):
        np.testing.assert_allclose(gf, gr, rtol=rtol, atol=atol)


R = np.random.default_rng(0)
BIG = (37, 129)        # 4773 elements: above the threshold, odd row length
BIG3 = (3, 41, 64)


def test_fast_path_is_actually_taken(toggle):
    assert fast.MIN_ELEMENTS <= BIG[0] * BIG[1]
    assert toggle(True, lambda: fast.use(np.zeros(BIG, np.float32)))
    assert not toggle(True, lambda: fast.use(np.zeros(16, np.float32)))
    assert not toggle(True, lambda: fast.use(np.zeros(BIG, np.int64)))
    assert not toggle(False, lambda: fast.use(np.zeros(BIG, np.float32)))


def test_gelu(toggle):
    # The fast kernel works in float64 internally, so it is the MORE accurate
    # of the two (error vs exact ~2e-7, reference ~3e-6 where 1 + tanh(u)
    # cancels for very negative x): allow that reference error.
    _check_same(toggle, ht.gelu, [3 * R.standard_normal(BIG)], atol=5e-6)


@pytest.mark.parametrize("shape", [BIG, BIG3])
def test_softmax_and_log_softmax(toggle, shape):
    x = 5 * R.standard_normal(shape)
    _check_same(toggle, lambda t: ht.softmax(t, axis=-1), [x], atol=1e-6)
    _check_same(toggle, lambda t: ht.log_softmax(t, axis=-1), [x])


def test_softmax_not_last_axis_uses_reference(toggle):
    x = R.standard_normal(BIG3)
    _check_same(toggle, lambda t: ht.softmax(t, axis=1), [x])


def test_masked_softmax_rows(toggle):
    x = R.standard_normal((4, 2, 33, 33))
    mask = np.triu(np.ones((33, 33), dtype=bool), 1)
    _check_same(toggle, lambda t: t.masked_fill(mask, -1e9).softmax(axis=-1), [x])


def test_layer_norm(toggle):
    d = BIG3[-1]
    arrays = [3 * R.standard_normal(BIG3) + 1, 1 + 0.1 * R.standard_normal(d),
              0.1 * R.standard_normal(d)]
    _check_same(toggle, lambda x, w, b: ht.layer_norm(x, w, b), arrays,
                rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_cross_entropy_matches_reference_and_torch(toggle, label_smoothing):
    n, v = 40, 300
    logits = 4 * R.standard_normal((n, v))
    targets = R.integers(1, v, size=n)
    targets[::5] = 0

    def op(x):
        return ht.cross_entropy(x, targets, label_smoothing=label_smoothing,
                                ignore_index=0)
    _check_same(toggle, op, [logits], atol=1e-7)

    lt = torch.tensor(logits.astype(np.float32), requires_grad=True)
    ref = F.cross_entropy(lt, torch.tensor(targets), ignore_index=0,
                          label_smoothing=label_smoothing)
    ref.backward()
    out, (grad,) = toggle(True, lambda: _forward_backward(op, [logits]))
    assert float(out) == pytest.approx(ref.item(), rel=1e-5)
    np.testing.assert_allclose(grad, lt.grad.numpy(), rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("ignore", [0, None])
def test_cross_entropy_non_target_smoothing(toggle, ignore):
    n, v = 40, 300
    logits = 4 * R.standard_normal((n, v))
    targets = R.integers(1, v, size=n)
    targets[::5] = 0     # padding when ignore = 0, a real class when None

    def op(x):
        return ht.cross_entropy(x, targets, label_smoothing=0.1, ignore_index=ignore,
                                label_smoothing_mode="non_target")
    _check_same(toggle, op, [logits], atol=1e-7)

    # Independent PyTorch formula: 1-ε on the target, 0 on pad, ε/K elsewhere.
    lt = torch.tensor(logits.astype(np.float32), requires_grad=True)
    t = torch.tensor(targets)
    log_probs = lt.log_softmax(-1)
    k = v - 1 - (0 if ignore is None else 1)
    q = torch.full_like(log_probs, 0.1 / k)
    q.scatter_(1, t[:, None], 0.9)
    keep = torch.ones(n)
    if ignore is not None:
        q[:, ignore] = 0.0
        keep = (t != ignore).float()
    ref = (-(q * log_probs).sum(-1) * keep).sum() / keep.sum()
    ref.backward()
    out, (grad,) = toggle(True, lambda: _forward_backward(op, [logits]))
    assert float(out) == pytest.approx(ref.item(), rel=1e-5)
    np.testing.assert_allclose(grad, lt.grad.numpy(), rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("padding_idx", [None, 3])
def test_embedding_padding_idx_fast_and_reference(toggle, padding_idx):
    vocab, d = 50, 128
    ids = R.integers(0, vocab, size=(4, 30))
    ids[:, -5:] = 3
    w = R.standard_normal((vocab, d))
    _check_same(toggle, lambda t: ht.embedding(ids, t, padding_idx=padding_idx), [w])
    _, (grad,) = toggle(True, lambda: _forward_backward(
        lambda t: ht.embedding(ids, t, padding_idx=padding_idx), [w]))
    assert grad[3].any() == (padding_idx is None)


def test_cross_entropy_all_ignored_gives_zero(toggle):
    logits = R.standard_normal((40, 300))
    targets = np.zeros(40, dtype=np.int64)

    def op(x):
        return ht.cross_entropy(x, targets, ignore_index=0)
    out, (grad,) = toggle(True, lambda: _forward_backward(op, [logits]))
    assert float(out) == 0.0
    assert not grad.any()


def test_embedding_backward_with_repeated_ids(toggle):
    vocab, d = 50, 128
    ids = R.integers(0, vocab, size=(4, 30))
    ids[0, :10] = 7  # many repeats of one id
    table = R.standard_normal((vocab, d))
    _check_same(toggle, lambda w: ht.embedding(ids, w), [table])


@pytest.mark.parametrize("opt_cls,wd", [(optim.Adam, 0.0), (optim.Adam, 0.01),
                                        (optim.AdamW, 0.01)])
def test_adam_step_matches_reference(toggle, opt_cls, wd):
    shape = (64, 100)
    w0 = R.standard_normal(shape).astype(np.float32)
    grads_seq = [R.standard_normal(shape).astype(np.float32) for _ in range(4)]

    def train():
        p = ht.Parameter(w0.copy())
        opt = opt_cls([p], lr=1e-2, weight_decay=wd)
        for g in grads_seq:
            p.grad = g
            opt.step()
        return p.numpy().copy(), opt.m[0].copy(), opt.v[0].copy()

    for a, b in zip(toggle(True, train), toggle(False, train)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)
