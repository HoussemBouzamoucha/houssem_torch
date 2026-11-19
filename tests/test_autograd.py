import numpy as np
import pytest
import torch
import houssem_torch as ht

rng = np.random.default_rng(0)


def rand(*shape, positive=False):
    x = rng.standard_normal(shape).astype(np.float32)
    return np.abs(x) + 0.5 if positive else x


# (name, ht_fn, torch_fn, input_shapes, positive_inputs)
CASES = [
    ("add",          lambda a, b: a + b,  lambda a, b: a + b,  [(3, 4), (3, 4)], False),
    ("sub",          lambda a, b: a - b,  lambda a, b: a - b,  [(3, 4), (3, 4)], False),
    ("mul",          lambda a, b: a * b,  lambda a, b: a * b,  [(3, 4), (3, 4)], False),
    ("div",          lambda a, b: a / b,  lambda a, b: a / b,  [(3, 4), (3, 4)], True),
    ("matmul",       lambda a, b: a @ b,  lambda a, b: a @ b,  [(3, 4), (4, 5)], False),
    ("add_bcast",    lambda a, b: a + b,  lambda a, b: a + b,  [(3, 4), (4,)],   False),
    ("mul_bcast",    lambda a, b: a * b,  lambda a, b: a * b,  [(3, 1), (1, 4)], False),
    ("neg",          lambda a: -a,        lambda a: -a,        [(3, 4)], False),
    ("scalar_mix",   lambda a: 2.0 * a + 1.0 - a / 4.0,
                     lambda a: 2.0 * a + 1.0 - a / 4.0,        [(3, 4)], False),
    ("rsub",         lambda a: 1.0 - a,   lambda a: 1.0 - a,   [(3, 4)], False),
    ("rtruediv",     lambda a: 2.0 / a,   lambda a: 2.0 / a,   [(3, 4)], True),
    ("exp",          lambda a: ht.exp(a), lambda a: torch.exp(a), [(3, 4)], False),
    ("log",          lambda a: ht.log(a), lambda a: torch.log(a), [(3, 4)], True),
    ("relu",         lambda a: ht.relu(a), lambda a: torch.relu(a), [(3, 4)], False),
    ("sum_axis0",    lambda a: a.sum(axis=0) * 2.0,
                     lambda a: a.sum(dim=0) * 2.0,             [(3, 4)], False),
    ("sum_keepdims", lambda a: a.sum(axis=1, keepdims=True) * a,
                     lambda a: a.sum(dim=1, keepdim=True) * a, [(3, 4)], False),
    ("mean_all",     lambda a: a.mean() * a,  lambda a: a.mean() * a, [(3, 4)], False),
    ("mean_axis",    lambda a: a.mean(axis=0) * 3.0,
                     lambda a: a.mean(dim=0) * 3.0,            [(3, 4)], False),
    ("reuse_x",      lambda a: a * a * a, lambda a: a * a * a, [(3, 4)], False),
    ("chain",        lambda a, b: ht.relu(a @ b).sum() * ht.exp(a.mean() * 0.1),
                     lambda a, b: torch.relu(a @ b).sum() * torch.exp(a.mean() * 0.1),
                     [(3, 4), (4, 5)], False),
    # --- shape ops (multiplied by b so gradient routing is actually checked)
    ("reshape",      lambda a, b: a.reshape(2, 6) * b,
                     lambda a, b: a.reshape(2, 6) * b,        [(3, 4), (2, 6)], False),
    ("reshape_neg1", lambda a, b: a.reshape(-1) * b,
                     lambda a, b: a.reshape(-1) * b,          [(2, 3, 2), (12,)], False),
    ("transpose_2d", lambda a, b: a.transpose() * b,
                     lambda a, b: a.T * b,                    [(3, 4), (4, 3)], False),
    ("transpose_nd", lambda a, b: a.transpose() * b,
                     lambda a, b: a.permute(2, 1, 0) * b,     [(2, 3, 4), (4, 3, 2)], False),
    ("permute",      lambda a, b: a.permute(2, 0, 1) * b,
                     lambda a, b: a.permute(2, 0, 1) * b,     [(2, 3, 4), (4, 2, 3)], False),
    ("permute_neg",  lambda a, b: a.permute(0, -1, 1) * b,
                     lambda a, b: a.permute(0, -1, 1) * b,    [(2, 3, 4), (2, 4, 3)], False),
    ("swapaxes",     lambda a, b: a.swapaxes(1, 2) * b,
                     lambda a, b: a.transpose(1, 2) * b,      [(2, 3, 4), (2, 4, 3)], False),
    ("mT",           lambda a, b: a.mT * b,
                     lambda a, b: a.mT * b,                   [(2, 3, 4), (2, 4, 3)], False),
    ("squeeze",      lambda a, b: a.squeeze() * b,
                     lambda a, b: a.squeeze() * b,            [(1, 3, 1), (3,)], False),
    ("squeeze_axis", lambda a, b: a.squeeze(0) * b,
                     lambda a, b: a.squeeze(0) * b,           [(1, 3, 1), (3, 1)], False),
    ("unsqueeze",    lambda a, b: a.unsqueeze(1) * b,
                     lambda a, b: a.unsqueeze(1) * b,         [(3, 4), (3, 1, 4)], False),
    # --- batched / broadcast matmul
    ("bmm_3d",       lambda a, b: a @ b,  lambda a, b: a @ b,  [(2, 3, 4), (2, 4, 5)], False),
    ("bmm_4d",       lambda a, b: a @ b,  lambda a, b: a @ b,  [(2, 3, 4, 5), (2, 3, 5, 6)], False),
    ("matmul_w",     lambda a, b: a @ b,  lambda a, b: a @ b,  [(2, 3, 4), (4, 5)], False),
    ("matmul_w_lhs", lambda a, b: a @ b,  lambda a, b: a @ b,  [(4, 5), (2, 5, 3)], False),
    ("matmul_bcast1", lambda a, b: a @ b, lambda a, b: a @ b,  [(1, 3, 4, 5), (2, 1, 5, 6)], False),
]


def _heads_ht(x, B, T, H, D):
    return x.reshape(B, T, H, D).swapaxes(1, 2)


def _heads_t(x, B, T, H, D):
    return x.reshape(B, T, H, D).transpose(1, 2)


def test_multihead_attention_grads_match_torch():
    """Attention-shaped graph: projections, head split, QK^T, AV, merge."""
    B, T, H, D = 2, 5, 3, 4
    C = H * D
    arrays = {
        "x": rand(B, T, C), "wq": rand(C, C), "wk": rand(C, C),
        "wv": rand(C, C), "wo": rand(C, C),
    }
    hs = {k: ht.tensor(v, requires_grad=True) for k, v in arrays.items()}
    ts = {k: torch.tensor(v, requires_grad=True) for k, v in arrays.items()}

    def forward(p, heads, exp):
        q = heads(p["x"] @ p["wq"], B, T, H, D)
        k = heads(p["x"] @ p["wk"], B, T, H, D)
        v = heads(p["x"] @ p["wv"], B, T, H, D)
        # exp(scores) stands in for softmax until it has its own op.
        weights = exp((q @ k.mT) * (D ** -0.5) * 0.1)
        out = (weights @ v).swapaxes(1, 2) if heads is _heads_ht \
            else (weights @ v).transpose(1, 2)
        return (out.reshape(B, T, C) @ p["wo"]).sum()

    forward(hs, _heads_ht, ht.exp).backward()
    forward(ts, _heads_t, torch.exp).backward()

    for name in arrays:
        assert hs[name].grad is not None, name
        np.testing.assert_allclose(hs[name].grad.numpy(), ts[name].grad.numpy(),
                                   rtol=1e-4, atol=1e-4, err_msg=name)


def test_shape_ops_propagate_requires_grad():
    x = ht.tensor(rand(2, 3, 4), requires_grad=True)
    for y in (x.reshape(6, 4), x.transpose(), x.permute(1, 0, 2),
              x.swapaxes(0, 2), x.mT, x.unsqueeze(0), x.unsqueeze(0).squeeze(0)):
        assert y.requires_grad and not y.is_leaf


def test_shape_ops_under_no_grad_record_nothing():
    x = ht.tensor(rand(2, 3), requires_grad=True)
    with ht.no_grad():
        y = x.reshape(3, 2).transpose()
    assert not y.requires_grad and y.is_leaf


def test_permute_rejects_invalid_axes():
    x = ht.tensor(rand(2, 3, 4))
    with pytest.raises(ValueError):
        x.permute(0, 0, 1)
    with pytest.raises(ValueError):
        x.permute(0, 1)


def test_matmul_batch_mismatch_raises():
    with pytest.raises(ValueError):
        ht.zeros((2, 3, 4)) @ ht.zeros((3, 4, 5))


@pytest.mark.parametrize("name,fn_ht,fn_t,shapes,positive", CASES,
                         ids=[c[0] for c in CASES])
def test_grad_matches_torch(name, fn_ht, fn_t, shapes, positive):
    arrays = [rand(*s, positive=positive) for s in shapes]
    hs = [ht.tensor(a, requires_grad=True) for a in arrays]
    ts = [torch.tensor(a, requires_grad=True) for a in arrays]

    fn_ht(*hs).sum().backward()
    fn_t(*ts).sum().backward()

    for h, t in zip(hs, ts):
        assert h.grad is not None
        assert h.grad.shape == tuple(t.grad.shape)
        np.testing.assert_allclose(h.grad.numpy(), t.grad.numpy(),
                                   rtol=1e-4, atol=1e-5)


def test_no_grad_blocks_graph():
    x = ht.tensor([1.0, 2.0], requires_grad=True)
    with ht.no_grad():
        y = x * 2.0
    assert not y.requires_grad and y.is_leaf


def test_requires_grad_propagates():
    x = ht.tensor([1.0, 2.0], requires_grad=True)
    c = ht.tensor([3.0, 4.0])
    assert (x * c).requires_grad
    assert not (c * c).requires_grad


def test_backward_nonscalar_raises():
    x = ht.tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(RuntimeError):
        (x * 2.0).backward()


def test_grad_accumulates_across_backward_calls():
    x = ht.tensor([1.0, 2.0, 3.0], requires_grad=True)
    (x * x).sum().backward()
    (x * x).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), 2 * 2 * np.array([1, 2, 3]))


def test_constants_get_no_grad():
    x = ht.tensor([1.0, 2.0], requires_grad=True)
    c = ht.tensor([3.0, 4.0])
    (x * c).sum().backward()
    assert c.grad is None


def test_parameter_is_trainable_leaf():
    p = ht.Parameter([[1.0, 2.0], [3.0, 4.0]])
    assert p.requires_grad and p.is_leaf
    
    
    

# ------------------------------------------------------------------
# micrograd-style engine behaviour
# ------------------------------------------------------------------

def test_backward_twice_on_same_graph_accumulates_only_leaves():
    x = ht.tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss = (x * x).sum()
    loss.backward()
    loss.backward()  # intermediates are recomputed, not double-counted
    np.testing.assert_allclose(x.grad.numpy(), 2 * 2 * np.array([1, 2, 3]))


def test_intermediate_grads_freed_unless_retained():
    x = ht.tensor([1.0, 2.0], requires_grad=True)
    y = x * 3.0
    y.sum().backward()
    assert y.grad is None

    x.grad = None
    y = x * 3.0
    y.sum().backward(retain_grads=True)
    np.testing.assert_allclose(y.grad.numpy(), [1.0, 1.0])
    np.testing.assert_allclose(x.grad.numpy(), [3.0, 3.0])


def test_graph_fields_record_op_and_parents():
    a = ht.tensor([1.0], requires_grad=True)
    b = ht.tensor([2.0])  # no grad -> not a parent
    c = a * b
    assert c._op == "multiply" and c._prev == (a,)


def test_wrong_gradient_shape_from_custom_backward_raises():
    from houssem_torch.autograd import result

    x = ht.tensor(rand(2, 3), requires_grad=True)

    def bad_backward(g):
        x._accumulate_grad(g.sum(axis=0))  # forgot to keep the shape

    out = result(x._storage, x, (x,), "bad_op", bad_backward)
    with pytest.raises(RuntimeError, match="gradient shape"):
        out.sum().backward()


def test_custom_op_via_result():
    """How users add their own op: forward in NumPy + hand-written backward."""
    from houssem_torch.autograd import result
    from houssem_torch.backends.cpu.kernels import CPUStorage

    def square(a):
        def _backward(g):
            a._accumulate_grad(g * 2.0 * a.numpy())
        return result(CPUStorage(a.numpy() ** 2), a, (a,), "square", _backward)

    x = ht.tensor([1.0, -2.0, 3.0], requires_grad=True)
    square(x).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [2.0, -4.0, 6.0])


def test_graph_has_no_reference_cycles():
    """Activations should be freed by refcounting, without waiting for gc."""
    import gc
    import weakref

    x = ht.tensor(rand(4, 4), requires_grad=True)
    gc.disable()
    try:
        y = ht.exp(x @ x).reshape(16)
        ref = weakref.ref(y)
        y.sum().backward()
        del y
        assert ref() is None
    finally:
        gc.enable()


def test_leaf_grad_is_private_writable_copy():
    a = ht.tensor([1.0, 2.0], requires_grad=True)
    b = ht.tensor([3.0, 4.0], requires_grad=True)
    (a + b).sum().backward()  # same g array is pushed to both parents
    a.grad.numpy()[...] = 0.0
    np.testing.assert_allclose(b.grad.numpy(), [1.0, 1.0])
