"""The compiled GPU training step against the CPU Transformer.

With dropout off, loss, every parameter gradient and the AdamW update must
match the CPU model. With dropout on, the GPU's gradients are checked
against finite differences of its own loss (same dropout masks).
Skipped without CuPy and a CUDA GPU.
"""
import numpy as np
import pytest

import houssem_torch as ht
from houssem_torch.backends.cuda import driver

pytestmark = pytest.mark.skipif(not driver.is_available(),
                                reason="needs CuPy and a CUDA GPU")

from houssem_torch.models import Transformer, TransformerConfig  # noqa: E402

B, S, T = 3, 7, 5


def small_config(**over):
    base = dict(vocab_size=61, d_model=64, n_heads=4, d_ff=96,
                n_encoder_layers=2, n_decoder_layers=2, dropout=0.0,
                max_len=16, label_smoothing=0.1)
    base.update(over)
    return TransformerConfig(**base)


def batch(cfg, seed=0):
    """Padded batch: varied lengths, one target row entirely padding."""
    rng = np.random.default_rng(seed)
    src = rng.integers(3, cfg.vocab_size, (B, S))
    tgt = rng.integers(3, cfg.vocab_size, (B, T + 1))
    src[0, 4:] = cfg.pad_id
    src[2, 2:] = cfg.pad_id
    tgt[1, 3:] = cfg.pad_id
    tgt[2, :] = cfg.pad_id
    tgt[0, 0] = cfg.bos_id
    tgt[1, 0] = cfg.bos_id
    return src, tgt[:, :-1], tgt[:, 1:]


def build(cfg, precision="fp32", **kw):
    ht.manual_seed(0)
    model = Transformer(cfg)
    from houssem_torch.backends.cuda.train_step import TrainStep
    return model, TrainStep(model, B, S, T, precision=precision, **kw)


def cpu_loss_and_grads(model, src, tgt_in, tgt_out):
    model.train()
    model.zero_grad()
    loss = model.loss(src, tgt_in, tgt_out)
    loss.backward()
    return float(loss.numpy()), {n: p.grad.numpy() for n, p in
                                 model.named_parameters()}


CONFIGS = {
    "base": {},
    "gelu_no_attn_bias": dict(activation="gelu", attention_bias=False),
    "non_target_padding_idx": dict(label_smoothing_mode="non_target",
                                   embedding_padding_idx=True,
                                   embedding_init="xavier"),
}


@pytest.mark.parametrize("variant", CONFIGS)
def test_fp32_matches_cpu(variant):
    cfg = small_config(**CONFIGS[variant])
    model, step = build(cfg)
    src, tgt_in, tgt_out = batch(cfg)
    ref_loss, ref_grads = cpu_loss_and_grads(model, src, tgt_in, tgt_out)

    step.forward_backward(src, tgt_in, tgt_out)
    assert step.last_loss() == pytest.approx(ref_loss, rel=1e-5)
    grads = step.gradients()
    assert set(grads) == set(ref_grads)
    for name, ref in ref_grads.items():
        scale = np.abs(ref).max() + 1e-6
        np.testing.assert_allclose(grads[name], ref, rtol=1e-3,
                                   atol=1e-4 * scale, err_msg=name)


def test_adamw_update_matches_cpu():
    cfg = small_config()
    model, step = build(cfg, weight_decay=0.01, max_grad_norm=0.5)
    src, tgt_in, tgt_out = batch(cfg)
    opt = ht.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    for i in range(3):
        _, g = cpu_loss_and_grads(model, src, tgt_in, tgt_out)
        if i == 0:
            first_grads = g
        ht.nn.clip_grad_norm(model.parameters(), max_norm=0.5)
        opt.step()
        step(src, tgt_in, tgt_out, lr=1e-3)
    assert step.stats()["steps"] == 3
    gpu = step.state_dict()
    for name, ref in model.state_dict().items():
        # Elements whose true gradient is 0 (the key bias: softmax ignores a
        # per-row constant) get rounding noise that Adam scales up to full
        # ±lr steps in either engine; compare only real gradients.
        g = np.abs(first_grads[name])
        real = g > 1e-6 * (g.max() + 1e-30)
        np.testing.assert_allclose(gpu[name][real], ref[real], rtol=1e-4,
                                   atol=1e-5, err_msg=name)


def test_fp16_close_to_cpu():
    cfg = small_config()
    model, step = build(cfg, precision="fp16")
    src, tgt_in, tgt_out = batch(cfg)
    ref_loss, ref_grads = cpu_loss_and_grads(model, src, tgt_in, tgt_out)
    step.forward_backward(src, tgt_in, tgt_out)
    assert step.last_loss() == pytest.approx(ref_loss, rel=1e-2)
    grads = step.gradients()
    for name, ref in ref_grads.items():
        cos = np.dot(grads[name].ravel(), ref.ravel()) / (
            np.linalg.norm(grads[name]) * np.linalg.norm(ref) + 1e-12)
        assert cos > 0.99, (name, cos)


@pytest.mark.parametrize("attention_bias", [True, False])
def test_fused_attention_matches_unfused(attention_bias):
    """fp16, dropout on: both attention paths use the same dropout masks
    (same seeds, same element indices), so they must agree to rounding.
    GELU, because with ReLU and a tiny batch one unit sitting at ~0 can flip
    sign between two roundings and move a few gradients by several %."""
    cfg = small_config(dropout=0.1, activation="gelu",
                       attention_bias=attention_bias)
    _, fused = build(cfg, precision="fp16")
    _, plain = build(cfg, precision="fp16", fused_attention=False)
    assert fused.k_attn and not plain.k_attn      # the paths really differ
    src, tgt_in, tgt_out = batch(cfg)
    fused.forward_backward(src, tgt_in, tgt_out)
    plain.forward_backward(src, tgt_in, tgt_out)
    assert fused.last_loss() == pytest.approx(plain.last_loss(), rel=2e-3)
    a, b = fused.gradients(), plain.gradients()
    for name in a:
        err = np.linalg.norm(a[name] - b[name]) / (np.linalg.norm(b[name]) + 1e-8)
        # zero-gradient key biases are pure rounding noise in both paths
        if np.linalg.norm(b[name]) > 1e-4:
            assert err < 2e-2, (name, err)


def test_dropout_gradients_match_finite_differences():
    """Dropout on: the gradient must be the derivative of the GPU's own loss
    under the same masks (checks every dropout backward path)."""
    cfg = small_config(dropout=0.2, activation="gelu")
    _, step = build(cfg)
    src, tgt_in, tgt_out = batch(cfg)

    def loss_at(flat):
        step.master.copy_from(flat)
        step._rng = np.random.default_rng(123)   # same dropout masks
        step.forward_backward(src, tgt_in, tgt_out)
        return step.last_loss()

    p0 = step.master.numpy()
    loss_at(p0)
    g = step.grad.numpy()
    assert np.isfinite(g).all() and np.abs(g).max() > 0
    rng = np.random.default_rng(1)
    for _ in range(3):
        v = rng.standard_normal(p0.shape).astype(np.float32)
        v /= np.linalg.norm(v)
        eps = 1e-2
        fd = (loss_at(p0 + eps * v) - loss_at(p0 - eps * v)) / (2 * eps)
        assert fd == pytest.approx(float(g @ v), rel=3e-2, abs=1e-4)


def test_dropout_changes_loss_but_eval_does_not():
    cfg = small_config(dropout=0.3)
    model, step = build(cfg)
    src, tgt_in, tgt_out = batch(cfg)
    a = step.evaluate(src, tgt_in, tgt_out)
    b = step.evaluate(src, tgt_in, tgt_out)
    model.eval()
    ref = float(model.loss(src, tgt_in, tgt_out).numpy())
    # The loss is summed with float atomics, whose order varies run to run:
    # equal to rounding, not bit for bit.
    assert a == pytest.approx(b, rel=1e-6)
    assert a == pytest.approx(ref, rel=1e-5)
    step.forward_backward(src, tgt_in, tgt_out)
    assert step.last_loss() != pytest.approx(ref, rel=1e-4)


def test_fp16_training_reduces_loss():
    cfg = small_config()
    _, step = build(cfg, precision="fp16")
    src, tgt_in, tgt_out = batch(cfg)
    first = step.evaluate(src, tgt_in, tgt_out)
    for _ in range(30):
        step(src, tgt_in, tgt_out, lr=3e-3)
    stats = step.stats()
    assert stats["steps"] > 0 and np.isfinite(stats["loss"])
    assert step.evaluate(src, tgt_in, tgt_out) < 0.5 * first


def test_rejects_wrong_batch_shape():
    cfg = small_config()
    _, step = build(cfg)
    src, tgt_in, tgt_out = batch(cfg)
    with pytest.raises(ValueError):
        step(src[:, :-1], tgt_in, tgt_out, lr=1e-3)
