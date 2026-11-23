"""The full encoder-decoder model.

1. Same numbers as PyTorch: an independent PyTorch implementation, loaded
   with identical weights, gives the same loss and gradients.
2. It learns: a tiny model learns to reverse sequences on CPU.
"""
import json
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import houssem_torch as ht
from houssem_torch import optim
from houssem_torch.models import Transformer, TransformerConfig
from houssem_torch.models.transformer import sinusoidal_position_encoding

PAD, BOS, EOS = 0, 1, 2


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

def test_config_from_json_roundtrip(tmp_path):
    cfg = TransformerConfig(vocab_size=8192, d_model=256, n_heads=4, d_ff=1024)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg.to_dict()))
    assert TransformerConfig.from_json(str(path)) == cfg


@pytest.mark.parametrize("bad", [
    {"vocab_size": 100, "d_model": 30, "n_heads": 4},   # not divisible
    {"vocab_size": 100, "activation": "tanh"},
    {"vocab_size": 100, "pad_id": 1},                   # pad == bos
    {"vocab_size": 100, "dmodel": 64},                  # typo'd key
])
def test_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        TransformerConfig.from_dict(bad)


def test_position_encoding_matches_formula():
    pe = sinusoidal_position_encoding(10, 8)
    pos, i = 7, 2  # dimension 2i = 4 (sin) and 2i+1 = 5 (cos)
    angle = pos / 10000 ** (2 * i / 8)
    assert pe[pos, 4] == pytest.approx(np.sin(angle), rel=1e-6)
    assert pe[pos, 5] == pytest.approx(np.cos(angle), rel=1e-6)


# ------------------------------------------------------------------
# Same numbers as an independent PyTorch implementation
# ------------------------------------------------------------------

def torch_smoothed_ce(logits, targets, eps, mode, pad=PAD):
    """Label-smoothed cross-entropy written out in PyTorch for both modes."""
    if mode == "uniform":
        return F.cross_entropy(logits, targets, label_smoothing=eps, ignore_index=pad)
    # non_target: 1-eps on the target, 0 on pad, eps/(V-2) on the others
    log_probs = logits.log_softmax(-1)
    q = torch.full_like(log_probs, eps / (logits.shape[-1] - 2))
    q.scatter_(1, targets[:, None], 1.0 - eps)
    q[:, pad] = 0.0
    keep = (targets != pad).float()
    return (-(q * log_probs).sum(-1) * keep).sum() / keep.sum()


def _torch_reference_loss(P, cfg, src, tgt_in, tgt_out):
    """The same architecture written directly in PyTorch, from the weights P."""
    d, h = cfg.d_model, cfg.n_heads
    pre = cfg.norm_position == "pre"

    def lin(x, name):
        y = x @ P[f"{name}.weight"]
        return y + P[f"{name}.bias"] if f"{name}.bias" in P else y

    def ln(x, name):
        return F.layer_norm(x, (d,), P[f"{name}.weight"], P[f"{name}.bias"])

    def block(x, name, sublayer):
        if pre:
            return x + sublayer(ln(x, name))
        return ln(x + sublayer(x), name)

    def attn(x, name, memory=None, pad=None, causal=False):
        B, Tq, _ = x.shape
        if memory is None:
            q, k, v = lin(x, f"{name}.w_qkv").split(d, -1)
        else:
            q = lin(x, f"{name}.w_q")
            k, v = lin(memory, f"{name}.w_kv").split(d, -1)
        q, k, v = (t.reshape(B, -1, h, d // h).transpose(1, 2) for t in (q, k, v))
        s = q @ k.transpose(-2, -1) / np.sqrt(d // h)
        if causal:
            s = s.masked_fill(torch.triu(torch.ones(Tq, Tq, dtype=torch.bool), 1), -1e9)
        if pad is not None:
            s = s.masked_fill(torch.tensor(pad)[:, None, None, :], -1e9)
        out = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, Tq, d)
        return lin(out, f"{name}.w_o")

    def ffn(x, name):
        return lin(torch.relu(lin(x, f"{name}.up")), f"{name}.down")

    pe = torch.tensor(sinusoidal_position_encoding(cfg.max_len, d))
    E = P["embedding.weight"]
    padding_idx = cfg.pad_id if cfg.embedding_padding_idx else None

    def embed(ids):
        return (F.embedding(torch.tensor(ids), E, padding_idx=padding_idx) * np.sqrt(d)
                + pe[: ids.shape[1]])

    def final(x, name):
        return ln(x, name) if pre else x

    src_pad, tgt_pad = src == PAD, tgt_in == PAD
    x = embed(src)
    for i in range(cfg.n_encoder_layers):
        n = f"encoder.{i}"
        x = block(x, f"{n}.norm_attn",
                  lambda z: attn(z, f"{n}.self_attn", pad=src_pad))
        x = block(x, f"{n}.norm_ffn", lambda z: ffn(z, f"{n}.ffn"))
    memory = final(x, "encoder_norm")

    y = embed(tgt_in)
    for i in range(cfg.n_decoder_layers):
        n = f"decoder.{i}"
        y = block(y, f"{n}.norm_self",
                  lambda z: attn(z, f"{n}.self_attn", pad=tgt_pad, causal=True))
        y = block(y, f"{n}.norm_cross",
                  lambda z: attn(z, f"{n}.cross_attn", memory=memory, pad=src_pad))
        y = block(y, f"{n}.norm_ffn", lambda z: ffn(z, f"{n}.ffn"))
    logits = final(y, "decoder_norm") @ E.T
    return torch_smoothed_ce(logits.reshape(-1, cfg.vocab_size),
                             torch.tensor(tgt_out).reshape(-1),
                             cfg.label_smoothing, cfg.label_smoothing_mode)


# Every architecture switch, alone and all together (the paper's settings).
ARCH_VARIANTS = {
    "default": {},
    "post_norm": {"norm_position": "post"},
    "no_attention_bias": {"attention_bias": False},
    "non_target_smoothing": {"label_smoothing_mode": "non_target"},
    "padding_idx_xavier": {"embedding_padding_idx": True, "embedding_init": "xavier"},
    "paper": {"norm_position": "post", "attention_bias": False,
              "label_smoothing_mode": "non_target", "embedding_padding_idx": True,
              "embedding_init": "xavier", "ffn_dropout": 0.0},
}


@pytest.mark.parametrize("variant", ARCH_VARIANTS)
def test_full_model_matches_torch_loss_and_every_gradient(variant):
    ht.manual_seed(0)
    cfg = TransformerConfig(vocab_size=17, d_model=16, n_heads=4, d_ff=32,
                            n_encoder_layers=2, n_decoder_layers=2, dropout=0.0,
                            max_len=12, label_smoothing=0.1,
                            **ARCH_VARIANTS[variant])
    model = Transformer(cfg)
    r = np.random.default_rng(1)
    src = r.integers(3, 17, size=(3, 6))
    src[1, 4:] = PAD                                  # padded source
    tgt = r.integers(3, 17, size=(3, 6))
    tgt_in = np.concatenate([np.full((3, 1), BOS), tgt[:, :-1]], axis=1)
    tgt_out = tgt.copy()
    tgt_in[2, 4:], tgt_out[2, 4:] = PAD, PAD          # padded target

    loss_h = model.loss(src, tgt_in, tgt_out)
    loss_h.backward()

    P = {n: torch.tensor(p.numpy(), requires_grad=True)
         for n, p in model.named_parameters()}
    loss_t = _torch_reference_loss(P, cfg, src, tgt_in, tgt_out)
    loss_t.backward()

    assert float(loss_h.numpy()) == pytest.approx(loss_t.item(), rel=1e-5)
    for name, p in model.named_parameters():
        np.testing.assert_allclose(p.grad.numpy(), P[name].grad.numpy(),
                                   rtol=1e-3, atol=1e-5, err_msg=name)


def _tiny(**kw):
    return TransformerConfig(vocab_size=11, d_model=8, n_heads=2, d_ff=16,
                             n_encoder_layers=1, n_decoder_layers=1, **kw)


def test_post_norm_has_no_final_layer_norms():
    pre = {n for n, _ in Transformer(_tiny()).named_parameters()}
    post = {n for n, _ in Transformer(_tiny(norm_position="post")).named_parameters()}
    assert pre - post == {"encoder_norm.weight", "encoder_norm.bias",
                          "decoder_norm.weight", "decoder_norm.bias"}
    assert post <= pre


def test_attention_bias_off_removes_only_attention_biases():
    names = [n for n, _ in Transformer(_tiny(attention_bias=False)).named_parameters()]
    assert not any(("self_attn." in n or "cross_attn." in n) and n.endswith(".bias")
                   for n in names)
    assert "encoder.0.norm_attn.bias" in names        # LayerNorms keep theirs
    assert "encoder.0.ffn.up.bias" in names           # FFN keeps its bias


def test_ffn_dropout_is_separate_from_dropout():
    model = Transformer(_tiny(dropout=0.3, ffn_dropout=0.0))
    layer = model.encoder[0]
    assert layer.ffn.dropout.p == 0.0
    assert layer.dropout.p == 0.3 and layer.self_attn.attn_dropout.p == 0.3
    assert Transformer(_tiny(dropout=0.3)).encoder[0].ffn.dropout.p == 0.3


@pytest.mark.parametrize("bad", [
    {"norm_position": "middle"}, {"label_smoothing_mode": "smooth"},
    {"embedding_init": "zeros"}, {"ffn_dropout": 1.0},
])
def test_config_rejects_bad_switches(bad):
    with pytest.raises(ValueError):
        _tiny(**bad)


def test_old_config_dicts_still_load_with_default_switches():
    old = {"vocab_size": 11, "d_model": 8, "n_heads": 2, "d_ff": 16}
    cfg = TransformerConfig.from_dict(old)
    switches = (cfg.norm_position, cfg.attention_bias, cfg.label_smoothing_mode,
                cfg.embedding_init, cfg.embedding_padding_idx, cfg.ffn_dropout)
    assert switches == ("pre", True, "uniform", "normal", False, None)


def test_shared_embedding_is_one_parameter():
    model = Transformer(TransformerConfig(vocab_size=11, d_model=8, n_heads=2,
                                          d_ff=16, n_encoder_layers=1,
                                          n_decoder_layers=1))
    names = [n for n, _ in model.named_parameters()]
    assert names.count("embedding.weight") == 1
    assert not any("embed" in n and n != "embedding.weight" for n in names)


def test_dropout_only_active_in_training():
    ht.manual_seed(0)
    model = Transformer(TransformerConfig(vocab_size=11, d_model=8, n_heads=2,
                                          d_ff=16, n_encoder_layers=1,
                                          n_decoder_layers=1, dropout=0.5))
    src, tgt = np.array([[3, 4, 5]]), np.array([[1, 6, 7]])
    model.eval()
    a, b = model(src, tgt).numpy(), model(src, tgt).numpy()
    np.testing.assert_array_equal(a, b)
    model.train()
    assert not np.array_equal(model(src, tgt).numpy(), model(src, tgt).numpy())


# ------------------------------------------------------------------
# It learns: reverse a sequence
# ------------------------------------------------------------------

def make_reverse_batch(rng, batch, max_len, vocab):
    """src: random tokens (variable length, padded); target: them reversed."""
    lengths = rng.integers(3, max_len + 1, size=batch)
    src = np.full((batch, max_len), PAD)
    tgt_in = np.full((batch, max_len + 1), PAD)
    tgt_out = np.full((batch, max_len + 1), PAD)
    for i, n in enumerate(lengths):
        seq = rng.integers(3, vocab, size=n)
        src[i, :n] = seq
        tgt_in[i, 0], tgt_in[i, 1:n + 1] = BOS, seq[::-1]
        tgt_out[i, :n], tgt_out[i, n] = seq[::-1], EOS
    return src, tgt_in, tgt_out


def test_tiny_transformer_learns_to_reverse_sequences():
    ht.manual_seed(0)
    rng = np.random.default_rng(0)
    vocab, max_len = 12, 6
    cfg = TransformerConfig(vocab_size=vocab, d_model=32, n_heads=4, d_ff=64,
                            n_encoder_layers=2, n_decoder_layers=2, dropout=0.0,
                            max_len=max_len + 2, label_smoothing=0.0)
    model = Transformer(cfg)
    opt = optim.Adam(model.parameters(), lr=3e-3)

    for _ in range(600):
        src, tgt_in, tgt_out = make_reverse_batch(rng, 64, max_len, vocab)
        opt.zero_grad()
        loss = model.loss(src, tgt_in, tgt_out)
        loss.backward()
        ht.nn.clip_grad_norm(model.parameters(), 1.0)
        opt.step()
    assert float(loss.numpy()) < 0.1, f"final training loss {float(loss.numpy())}"

    src, _, tgt_out = make_reverse_batch(rng, 200, max_len, vocab)
    pred = model.translate_greedy(src, max_new_tokens=max_len + 1)
    exact = [np.array_equal(p[:np.argmax(t == EOS) + 1], t[:np.argmax(t == EOS) + 1])
             for p, t in zip(pred, tgt_out)]
    assert np.mean(exact) > 0.9, f"exact-match accuracy {np.mean(exact):.2f}"
