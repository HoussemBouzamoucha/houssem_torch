"""Modules and optimizers, checked against PyTorch."""
import numpy as np
import pytest
import torch
import houssem_torch as ht
from houssem_torch import nn, optim

rng = np.random.default_rng(0)


def rand(*shape):
    return rng.standard_normal(shape).astype(np.float32)


# ------------------------------------------------------------------
# Module bookkeeping
# ------------------------------------------------------------------

class Tiny(nn.Module):
    def __init__(self):
        self.a = nn.Linear(4, 3)
        self.blocks = nn.ModuleList([nn.LayerNorm(3), nn.LayerNorm(3)])
        self.shared = self.a.weight  # same Parameter registered twice

    def forward(self, x):
        return self.blocks[1](self.blocks[0](self.a(x)))


def test_named_parameters_follow_attribute_paths_and_dedupe_shared():
    names = [n for n, _ in Tiny().named_parameters()]
    assert names == ["a.weight", "a.bias", "blocks.0.weight", "blocks.0.bias",
                     "blocks.1.weight", "blocks.1.bias"]


def test_train_eval_propagates_to_all_submodules():
    m = Tiny()
    m.eval()
    assert all(not mod.training for mod in m.modules())
    m.train()
    assert all(mod.training for mod in m.modules())


def test_state_dict_roundtrip_and_strict_checks():
    a, b = Tiny(), Tiny()
    b.load_state_dict(a.state_dict())
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        np.testing.assert_array_equal(p.numpy(), q.numpy(), err_msg=n)
    state = a.state_dict()
    state["a.weight"] = np.zeros((2, 2), np.float32)
    with pytest.raises(ValueError):
        b.load_state_dict(state)
    with pytest.raises(KeyError):
        b.load_state_dict({"a.weight": a.state_dict()["a.weight"]})


def test_state_dict_is_a_copy():
    m = Tiny()
    state = m.state_dict()
    state["a.bias"][...] = 99.0
    assert not np.any(m.a.bias.numpy() == 99.0)


def test_zero_grad_clears_gradients():
    m = Tiny()
    m(ht.tensor(rand(2, 4))).sum().backward()
    assert all(p.grad is not None for p in m.parameters())
    m.zero_grad()
    assert all(p.grad is None for p in m.parameters())


# ------------------------------------------------------------------
# Layers vs PyTorch
# ------------------------------------------------------------------

def test_linear_matches_torch_forward_and_grads():
    lin = nn.Linear(5, 3)
    x = rand(2, 4, 5)
    xh = ht.tensor(x, requires_grad=True)
    lin(xh).sum().backward()

    xt = torch.tensor(x, requires_grad=True)
    w = torch.tensor(lin.weight.numpy(), requires_grad=True)
    b = torch.tensor(lin.bias.numpy(), requires_grad=True)
    out_t = xt @ w + b
    out_t.sum().backward()

    np.testing.assert_allclose(lin(ht.tensor(x)).numpy(), out_t.detach().numpy(),
                               rtol=1e-5, atol=1e-6)
    for h, t in [(xh, xt), (lin.weight, w), (lin.bias, b)]:
        np.testing.assert_allclose(h.grad.numpy(), t.grad.numpy(), rtol=1e-5, atol=1e-5)


def test_dropout_module_follows_train_eval():
    d = nn.Dropout(0.5)
    x = ht.tensor(np.ones((50, 50), np.float32))
    assert np.any(d(x).numpy() == 0.0)
    d.eval()
    assert d(x) is x


@pytest.mark.parametrize("cross,causal,padded", [
    (False, False, False), (False, True, False), (False, False, True),
    (True, False, True),
])
def test_attention_matches_torch(cross, causal, padded):
    B, T, S, d, h = 2, 4, 5, 8, 2
    mha = nn.MultiHeadAttention(d, h, cross=cross)
    x, mem = rand(B, T, d), rand(B, S, d)
    k_len = S if cross else T
    pad = np.zeros((B, k_len), dtype=bool)
    if padded:
        pad[1, -2:] = True

    xh = ht.tensor(x, requires_grad=True)
    out_h = mha(xh, memory=ht.tensor(mem) if cross else None,
                key_padding_mask=pad if padded else None, causal=causal)

    xt = torch.tensor(x, requires_grad=True)
    P = {n: torch.tensor(p.numpy()) for n, p in mha.named_parameters()}

    def lin(t, name):
        return t @ P[f"{name}.weight"] + P[f"{name}.bias"]

    if cross:
        q = lin(xt, "w_q")
        k, v = lin(torch.tensor(mem), "w_kv").split(d, dim=-1)
    else:
        q, k, v = lin(xt, "w_qkv").split(d, dim=-1)
    heads = [t.reshape(B, -1, h, d // h).transpose(1, 2) for t in (q, k, v)]
    scores = heads[0] @ heads[1].transpose(-2, -1) / np.sqrt(d // h)
    if causal:
        scores = scores.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1),
                                    -1e9)
    if padded:
        scores = scores.masked_fill(torch.tensor(pad)[:, None, None, :], -1e9)
    out_t = lin((scores.softmax(-1) @ heads[2]).transpose(1, 2).reshape(B, T, d), "w_o")

    np.testing.assert_allclose(out_h.numpy(), out_t.detach().numpy(),
                               rtol=1e-4, atol=1e-5)
    w = rand(*out_h.shape)
    (out_h * ht.tensor(w)).sum().backward()
    (out_t * torch.tensor(w)).sum().backward()
    np.testing.assert_allclose(xh.grad.numpy(), xt.grad.numpy(), rtol=1e-4, atol=1e-5)


def test_attention_rejects_misuse():
    with pytest.raises(ValueError):
        nn.MultiHeadAttention(10, 3)
    x = ht.tensor(rand(1, 2, 4))
    with pytest.raises(ValueError):
        nn.MultiHeadAttention(4, 2, cross=True)(x)
    with pytest.raises(ValueError):
        nn.MultiHeadAttention(4, 2)(x, memory=x)


# ------------------------------------------------------------------
# Optimizers vs PyTorch
# ------------------------------------------------------------------

def _run_optimizer(make_ht, make_t, steps=5):
    """Minimize sum((w·x - y)²) for a few steps with both frameworks."""
    w0, x, y = rand(4, 3), rand(6, 4), rand(6, 3)
    wh = nn.Parameter(w0)
    wt = torch.tensor(w0, requires_grad=True)
    opt_h, opt_t = make_ht([wh]), make_t([wt])
    for _ in range(steps):
        opt_h.zero_grad()
        ((ht.tensor(x) @ wh - ht.tensor(y)) ** 2).sum().backward()
        opt_h.step()
        opt_t.zero_grad()
        ((torch.tensor(x) @ wt - torch.tensor(y)) ** 2).sum().backward()
        opt_t.step()
    np.testing.assert_allclose(wh.numpy(), wt.detach().numpy(), rtol=1e-5, atol=1e-6)


def test_adam_matches_torch():
    _run_optimizer(lambda p: optim.Adam(p, lr=0.01, betas=(0.9, 0.98), eps=1e-9),
                   lambda p: torch.optim.Adam(p, lr=0.01, betas=(0.9, 0.98), eps=1e-9))


def test_adam_l2_weight_decay_matches_torch():
    _run_optimizer(lambda p: optim.Adam(p, lr=0.01, weight_decay=0.1),
                   lambda p: torch.optim.Adam(p, lr=0.01, betas=(0.9, 0.98), eps=1e-9,
                                              weight_decay=0.1))


def test_adamw_matches_torch():
    _run_optimizer(lambda p: optim.AdamW(p, lr=0.01, weight_decay=0.1),
                   lambda p: torch.optim.AdamW(p, lr=0.01, betas=(0.9, 0.98),
                                               eps=1e-9, weight_decay=0.1))


def test_sgd_momentum_matches_torch():
    _run_optimizer(lambda p: optim.SGD(p, lr=0.01, momentum=0.9),
                   lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9))


def test_optimizer_updates_shared_parameter_once():
    p = nn.Parameter(np.ones(3, np.float32))
    opt = optim.SGD([p, p], lr=1.0)
    p.grad = np.ones(3, np.float32)
    opt.step()
    np.testing.assert_array_equal(p.numpy(), np.zeros(3))


def test_optimizer_state_dict_resumes_identically():
    w0, x = rand(3, 3), rand(5, 3)

    def run(steps, resume_from=None):
        w = nn.Parameter(w0)
        opt = optim.Adam([w], lr=0.05)
        if resume_from is not None:
            w.numpy()[...] = resume_from[0]
            opt.load_state_dict(resume_from[1])
        for _ in range(steps):
            opt.zero_grad()
            ((ht.tensor(x) @ w) ** 2).sum().backward()
            opt.step()
        return w.numpy().copy(), opt.state_dict()

    straight, _ = run(6)
    w_mid, state = run(3)
    resumed, _ = run(3, resume_from=(w_mid, state))
    np.testing.assert_allclose(resumed, straight, rtol=1e-6)


def test_noam_schedule_shape():
    lrs = [optim.noam_lr(s, d_model=512, warmup_steps=4000)
           for s in (1, 2000, 4000, 16000)]
    assert lrs[0] < lrs[1] < lrs[2]          # warmup: rising
    assert lrs[3] == pytest.approx(lrs[2] / 2)  # decay: 1/sqrt(step)
    w = nn.Parameter(np.ones(1, np.float32))
    opt = optim.Adam([w])
    sched = optim.NoamSchedule(opt, d_model=512, warmup_steps=4000)
    sched.step()
    assert opt.lr == pytest.approx(lrs[0])


def test_clip_grad_norm_matches_torch():
    arrays = [rand(3, 4) * 5, rand(4) * 5]
    ps = [nn.Parameter(a) for a in arrays]
    ts = [torch.tensor(a, requires_grad=True) for a in arrays]
    grads = [rand(3, 4) * 10, rand(4) * 10]
    for p, t, g in zip(ps, ts, grads):
        p.grad = g
        t.grad = torch.tensor(g)
    norm_h = nn.clip_grad_norm(ps, max_norm=1.0)
    norm_t = torch.nn.utils.clip_grad_norm_(ts, max_norm=1.0)
    assert norm_h == pytest.approx(float(norm_t), rel=1e-5)
    for p, t in zip(ps, ts):
        np.testing.assert_allclose(p.grad.numpy(), t.grad.numpy(), rtol=1e-5)
