"""Forward passes of the transformer ops, checked against PyTorch.

Backward passes are written step by step; until each one exists, its op must
fail loudly (naming the op) as soon as backward reaches it. When a backward
is filled in, move its op from TODO_BACKWARD to a gradient test.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import houssem_torch as ht

rng = np.random.default_rng(0)


def rand(*shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def close(h, t, rtol=1e-5, atol=1e-6):
    np.testing.assert_allclose(h.numpy(), t.detach().numpy(), rtol=rtol, atol=atol)


# ------------------------------------------------------------------
# Activations / softmax
# ------------------------------------------------------------------

def test_gelu_matches_torch_tanh_approximation():
    x = rand(4, 7, scale=3.0)
    close(ht.gelu(ht.tensor(x)), F.gelu(torch.tensor(x), approximate="tanh"))


@pytest.mark.parametrize("axis", [-1, 1, 0])
def test_softmax_matches_torch(axis):
    x = rand(3, 5, 6)
    close(ht.softmax(ht.tensor(x), axis=axis), torch.softmax(torch.tensor(x), dim=axis))


def test_softmax_is_stable_for_large_values():
    x = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
    out = ht.tensor(x).softmax().numpy()
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out.sum(axis=-1), 1.0, rtol=1e-6)


def test_log_softmax_matches_torch():
    x = rand(2, 4, 9, scale=5.0)
    close(ht.tensor(x).log_softmax(), torch.log_softmax(torch.tensor(x), dim=-1),
          atol=1e-5)


def test_causal_mask_then_softmax_zeroes_future_positions():
    T = 5
    scores = ht.tensor(rand(2, 3, T, T))
    causal = np.triu(np.ones((T, T), dtype=bool), k=1)  # True = future = masked
    probs = scores.masked_fill(causal, float("-inf")).softmax().numpy()
    assert np.all(probs[..., causal] == 0.0)
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, rtol=1e-6)


def test_padding_mask_broadcasts_over_heads_and_queries():
    B, h, T, S = 2, 3, 4, 5
    scores = rand(B, h, T, S)
    pad = np.zeros((B, S), dtype=bool)
    pad[0, 3:] = True
    out = ht.tensor(scores).masked_fill(pad[:, None, None, :], -1e9)
    expected = torch.tensor(scores).masked_fill(
        torch.tensor(pad)[:, None, None, :], -1e9)
    close(out, expected)


def test_masked_fill_rejects_non_broadcastable_mask():
    with pytest.raises(ValueError):
        ht.tensor(rand(2, 3)).masked_fill(np.zeros((4,), dtype=bool), 0.0)


# ------------------------------------------------------------------
# Dropout
# ------------------------------------------------------------------

def test_dropout_eval_mode_is_identity():
    x = ht.tensor(rand(4, 4), requires_grad=True)
    assert ht.dropout(x, p=0.5, training=False) is x
    assert ht.dropout(x, p=0.0) is x


def test_dropout_drops_about_p_and_rescales_survivors():
    ht.manual_seed(123)
    x = np.ones((200, 500), dtype=np.float32)
    out = ht.dropout(ht.tensor(x), p=0.3).numpy()
    dropped = np.mean(out == 0.0)
    assert abs(dropped - 0.3) < 0.01
    np.testing.assert_allclose(out[out != 0.0], 1.0 / 0.7, rtol=1e-6)


def test_dropout_is_reproducible_with_manual_seed():
    x = ht.tensor(rand(8, 8))
    ht.manual_seed(7)
    a = ht.dropout(x, p=0.5).numpy()
    ht.manual_seed(7)
    b = ht.dropout(x, p=0.5).numpy()
    np.testing.assert_array_equal(a, b)


def test_dropout_rejects_invalid_p():
    with pytest.raises(ValueError):
        ht.dropout(ht.tensor(rand(2)), p=1.0)


# ------------------------------------------------------------------
# LayerNorm, embedding, loss
# ------------------------------------------------------------------

def test_layer_norm_matches_torch():
    x, w, b = rand(2, 5, 16, scale=3.0), rand(16), rand(16)
    out = ht.layer_norm(ht.tensor(x), ht.tensor(w), ht.tensor(b), eps=1e-5)
    expected = F.layer_norm(torch.tensor(x), (16,), torch.tensor(w),
                            torch.tensor(b), eps=1e-5)
    close(out, expected, rtol=1e-4, atol=1e-5)


def test_layer_norm_rejects_wrong_weight_shape():
    with pytest.raises(ValueError):
        ht.layer_norm(ht.tensor(rand(2, 8)), ht.tensor(rand(4)), ht.tensor(rand(8)))


def test_embedding_matches_torch():
    V, d = 11, 6
    w = rand(V, d)
    ids = rng.integers(0, V, size=(3, 4))
    out = ht.embedding(ht.tensor(ids, dtype=ht.int64), ht.tensor(w))
    assert out.shape == (3, 4, d)
    close(out, F.embedding(torch.tensor(ids), torch.tensor(w)))


def test_embedding_rejects_bad_ids():
    w = ht.tensor(rand(5, 3))
    with pytest.raises(IndexError):
        ht.embedding(np.array([0, 5]), w)
    with pytest.raises(IndexError):
        ht.embedding(np.array([-1]), w)
    with pytest.raises(TypeError):
        ht.embedding(np.array([0.0, 1.0]), w)


@pytest.mark.parametrize("smoothing,ignore", [(0.0, None), (0.1, None),
                                              (0.0, 0), (0.1, 0)])
def test_cross_entropy_matches_torch(smoothing, ignore):
    B, T, V = 3, 5, 13
    logits = rand(B, T, V, scale=2.0)
    targets = rng.integers(0, V, size=(B, T))
    targets[0, 3:] = 0  # padding positions when ignore_index = 0
    out = ht.cross_entropy(ht.tensor(logits), targets,
                           label_smoothing=smoothing, ignore_index=ignore)
    assert out.shape == ()
    expected = F.cross_entropy(
        torch.tensor(logits).reshape(-1, V), torch.tensor(targets).reshape(-1),
        label_smoothing=smoothing,
        ignore_index=-100 if ignore is None else ignore)
    close(out, expected, rtol=1e-5, atol=1e-5)


def test_cross_entropy_rejects_mismatched_targets():
    with pytest.raises(ValueError):
        ht.cross_entropy(ht.tensor(rand(2, 3, 5)), np.zeros((2, 4), dtype=np.int64))


# ------------------------------------------------------------------
# Pointwise and indexing
# ------------------------------------------------------------------

def test_sqrt_and_pow_match_numpy():
    x = np.abs(rand(3, 4)) + 0.1
    np.testing.assert_allclose(ht.tensor(x).sqrt().numpy(), np.sqrt(x), rtol=1e-6)
    np.testing.assert_allclose((ht.tensor(x) ** 3).numpy(), x ** 3, rtol=1e-6)
    np.testing.assert_allclose((ht.tensor(x) ** -0.5).numpy(), x ** -0.5, rtol=1e-6)
    assert (ht.tensor(x) ** 2).numpy().dtype == np.float32


def test_pow_rejects_tensor_exponent():
    with pytest.raises(TypeError):
        ht.tensor(rand(2)) ** ht.tensor(rand(2))


@pytest.mark.parametrize("index", [
    0, (slice(None), 1), (Ellipsis, slice(0, 3)), (slice(None), slice(1, None, 2)),
    np.array([2, 0, 2]),
])
def test_getitem_matches_numpy(index):
    x = rand(4, 5, 6)
    np.testing.assert_array_equal(ht.tensor(x)[index].numpy(), x[index])


def test_split_matches_torch():
    x = rand(2, 3, 12)
    for arg in (4, 5, [3, 4, 5]):
        hs = ht.tensor(x).split(arg, axis=-1)
        ts = torch.tensor(x).split(arg, dim=-1)
        assert len(hs) == len(ts)
        for h, t in zip(hs, ts):
            close(h, t)


def test_split_qkv_from_fused_projection():
    B, S, d = 2, 3, 8
    qkv = ht.tensor(rand(B, S, 3 * d))
    q, k, v = qkv.split(d, axis=-1)
    assert q.shape == k.shape == v.shape == (B, S, d)


def test_split_rejects_bad_sections():
    with pytest.raises(ValueError):
        ht.tensor(rand(2, 6)).split([2, 2], axis=-1)


# ------------------------------------------------------------------
# Gradients vs PyTorch
#
# Each case SKIPS while its function in houssem_torch/ops/grads.py still
# raises NotImplementedError, and starts checking against PyTorch as soon as
# it is written. Run:  pytest tests/test_transformer_ops.py -k grad -v
# ------------------------------------------------------------------

def _positive(*shape):
    return np.abs(rand(*shape)) + 0.5


def _case_dropout():
    # Same random mask on both sides: rebuild it from houssem's own output.
    x = rand(4, 6) + 3.0  # nonzero, so the mask can be recovered as out / x
    ht.manual_seed(0)
    mask = ht.dropout(ht.tensor(x), p=0.4).numpy() / x
    ht.manual_seed(0)
    return ([x], lambda a: ht.dropout(a, p=0.4),
            lambda a: a * torch.tensor(mask))


def _combine_ht(pieces):
    """Reduce split pieces to one tensor so each piece gets a distinct grad."""
    return (pieces[0].sum(axis=-1) * 1.0 + pieces[1].sum(axis=-1) * 2.0
            + pieces[2].sum(axis=-1) * 3.0)


def _combine_t(pieces):
    return pieces[0].sum(-1) * 1.0 + pieces[1].sum(-1) * 2.0 + pieces[2].sum(-1) * 3.0


_IDS = np.array([[1, 4, 1], [0, 4, 4]])            # repeated tokens on purpose
_TARGETS = np.array([[2, 0, 5, 0], [1, 3, 0, 4]])  # 0 = padding
_FILL_MASK = rng.random((3, 1, 5)) > 0.5           # broadcasts over axis 1
_CAUSAL = np.triu(np.ones((4, 4), dtype=bool), k=1)


def _ce_torch(z, **kw):
    return F.cross_entropy(z.reshape(-1, 7), torch.tensor(_TARGETS).reshape(-1), **kw)


# name -> () -> (input arrays, houssem fn, torch fn)
GRAD_CASES = {
    "pow": lambda: ([_positive(3, 4)], lambda a: a ** 3, lambda a: a ** 3),
    "pow_fractional": lambda: ([_positive(3, 4)], lambda a: a ** -0.5,
                               lambda a: a ** -0.5),
    "sqrt": lambda: ([_positive(3, 4)], ht.sqrt, torch.sqrt),
    "masked_fill": lambda: (
        [rand(3, 4, 5)],
        lambda a: a.masked_fill(_FILL_MASK, 0.0),
        lambda a: a.masked_fill(torch.tensor(_FILL_MASK), 0.0)),
    "masked_fill_causal_softmax": lambda: (
        [rand(2, 4, 4)],
        lambda a: a.masked_fill(_CAUSAL, float("-inf")).softmax(),
        lambda a: a.masked_fill(torch.tensor(_CAUSAL), float("-inf")).softmax(-1)),
    "dropout": _case_dropout,
    "getitem_slice": lambda: ([rand(4, 5, 6)], lambda a: a[:, 1:4, ::2],
                              lambda a: a[:, 1:4, ::2]),
    "getitem_repeats": lambda: ([rand(4, 3)], lambda a: a[np.array([2, 0, 2])],
                                lambda a: a[torch.tensor([2, 0, 2])]),
    "split": lambda: ([rand(2, 3, 9)],
                      lambda a: _combine_ht(a.split([2, 3, 4], axis=-1)),
                      lambda a: _combine_t(a.split([2, 3, 4], dim=-1))),
    "embedding": lambda: ([rand(6, 4)], lambda w: ht.embedding(_IDS, w),
                          lambda w: F.embedding(torch.tensor(_IDS), w)),
    "gelu": lambda: ([rand(3, 7, scale=3.0)], ht.gelu,
                     lambda a: F.gelu(a, approximate="tanh")),
    "log_softmax": lambda: ([rand(3, 4, 6)], lambda a: a.log_softmax(),
                            lambda a: torch.log_softmax(a, dim=-1)),
    "log_softmax_axis1": lambda: ([rand(3, 4, 6)], lambda a: a.log_softmax(1),
                                  lambda a: torch.log_softmax(a, dim=1)),
    "softmax": lambda: ([rand(3, 4, 6)], lambda a: a.softmax(),
                        lambda a: torch.softmax(a, dim=-1)),
    "softmax_axis1": lambda: ([rand(3, 4, 6)], lambda a: a.softmax(1),
                              lambda a: torch.softmax(a, dim=1)),
    "cross_entropy": lambda: ([rand(2, 4, 7, scale=2.0)],
                              lambda z: ht.cross_entropy(z, _TARGETS), _ce_torch),
    "cross_entropy_smooth_pad": lambda: (
        [rand(2, 4, 7, scale=2.0)],
        lambda z: ht.cross_entropy(z, _TARGETS, label_smoothing=0.1, ignore_index=0),
        lambda z: _ce_torch(z, label_smoothing=0.1, ignore_index=0)),
    "layer_norm": lambda: ([rand(2, 3, 8, scale=2.0), rand(8), rand(8)],
                           lambda x, w, b: ht.layer_norm(x, w, b),
                           lambda x, w, b: F.layer_norm(x, (8,), w, b)),
}

# which grads.py function each case needs (named in the skip message)
_NEEDS = {"pow_fractional": "pow", "masked_fill_causal_softmax": "masked_fill",
          "getitem_slice": "getitem", "getitem_repeats": "getitem",
          "split": "getitem", "log_softmax_axis1": "log_softmax",
          "softmax_axis1": "softmax", "cross_entropy_smooth_pad": "cross_entropy"}


@pytest.mark.parametrize("case", sorted(GRAD_CASES))
def test_grad_matches_torch(case):
    arrays, fn_ht, fn_t = GRAD_CASES[case]()
    hs = [ht.tensor(a, requires_grad=True) for a in arrays]
    ts = [torch.tensor(a, requires_grad=True) for a in arrays]

    out_h = fn_ht(*hs)
    # Weight the output by fixed random numbers so every gradient element is
    # distinct (a plain .sum() would hide many routing mistakes).
    weights = np.asarray(rng.standard_normal(out_h.shape), dtype=np.float32)
    try:
        (out_h * ht.tensor(weights)).sum().backward()
    except NotImplementedError as exc:
        missing = str(exc).split("'")[1]  # the op whose backward is missing
        pytest.skip(f"{missing}_backward in ops/grads.py is not written yet")

    (fn_t(*ts) * torch.tensor(weights)).sum().backward()
    for i, (h, t) in enumerate(zip(hs, ts)):
        assert h.grad is not None, f"{case}: input {i} got no gradient"
        np.testing.assert_allclose(h.grad.numpy(), t.grad.numpy(),
                                   rtol=1e-4, atol=1e-5,
                                   err_msg=f"{case}: gradient of input {i}")
        assert h.grad.numpy().dtype == np.float32, f"{case}: grad must stay float32"


# ------------------------------------------------------------------
# Graph bookkeeping (independent of the gradient math)
# ------------------------------------------------------------------

def _w(*shape):
    return ht.tensor(rand(*shape), requires_grad=True)


OPS = {
    "gelu": lambda: ht.gelu(_w(3, 4)),
    "softmax": lambda: ht.softmax(_w(3, 4)),
    "log_softmax": lambda: ht.log_softmax(_w(3, 4)),
    "dropout": lambda: ht.dropout(_w(3, 4), p=0.5),
    "layer_norm": lambda: ht.layer_norm(_w(3, 4), _w(4), _w(4)),
    "embedding": lambda: ht.embedding(np.array([[0, 2]]), _w(5, 3)),
    "cross_entropy": lambda: ht.cross_entropy(_w(2, 5), np.array([1, 3])),
    "sqrt": lambda: ht.sqrt(ht.tensor(np.abs(rand(3)) + 1.0, requires_grad=True)),
    "pow": lambda: _w(3) ** 2,
    "masked_fill": lambda: _w(2, 3).masked_fill(np.eye(2, 3, dtype=bool), 0.0),
    "getitem": lambda: _w(4, 3)[1:],
}


@pytest.mark.parametrize("op", sorted(OPS))
def test_op_records_graph_node(op):
    out = OPS[op]()
    assert out.requires_grad and out._op == op


@pytest.mark.parametrize("op", sorted(OPS))
def test_forward_under_no_grad_builds_no_graph(op):
    with ht.no_grad():
        out = OPS[op]()
    assert out.is_leaf and not out.requires_grad


def test_backward_returning_none_gives_clear_error():
    from houssem_torch.autograd import result

    x = ht.tensor(rand(3), requires_grad=True)

    def forgot_return(g):
        x._accumulate_grad(None)

    out = result(x._storage, x, (x,), "my_op", forgot_return)
    with pytest.raises(TypeError, match="forget to `return`"):
        out.sum().backward()
