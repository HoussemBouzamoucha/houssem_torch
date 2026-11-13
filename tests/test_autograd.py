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
]


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
    assert not y.requires_grad and y._grad_fn is None


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
    
    
    