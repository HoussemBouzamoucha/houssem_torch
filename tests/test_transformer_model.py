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

def _torch_reference_loss(P, cfg, src, tgt_in, tgt_out):
    """The same architecture written directly in PyTorch, from the weights P."""
    d, h = cfg.d_model, cfg.n_heads

    def lin(x, name):
        return x @ P[f"{name}.weight"] + P[f"{name}.bias"]

    def ln(x, name):
        return F.layer_norm(x, (d,), P[f"{name}.weight"], P[f"{name}.bias"])

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

    def embed(ids):
        return E[torch.tensor(ids)] * np.sqrt(d) + pe[: ids.shape[1]]

    src_pad, tgt_pad = src == PAD, tgt_in == PAD
    x = embed(src)
    for i in range(cfg.n_encoder_layers):
        n = f"encoder.{i}"
        x = x + attn(ln(x, f"{n}.norm_attn"), f"{n}.self_attn", pad=src_pad)
        x = x + ffn(ln(x, f"{n}.norm_ffn"), f"{n}.ffn")
    memory = ln(x, "encoder_norm")

    y = embed(tgt_in)
    for i in range(cfg.n_decoder_layers):
        n = f"decoder.{i}"
        y = y + attn(ln(y, f"{n}.norm_self"), f"{n}.self_attn", pad=tgt_pad,
                     causal=True)
        y = y + attn(ln(y, f"{n}.norm_cross"), f"{n}.cross_attn", memory=memory,
                     pad=src_pad)
        y = y + ffn(ln(y, f"{n}.norm_ffn"), f"{n}.ffn")
    logits = ln(y, "decoder_norm") @ E.T
    return F.cross_entropy(logits.reshape(-1, cfg.vocab_size),
                           torch.tensor(tgt_out).reshape(-1),
                           label_smoothing=cfg.label_smoothing, ignore_index=PAD)


def test_full_model_matches_torch_loss_and_every_gradient():
    ht.manual_seed(0)
    cfg = TransformerConfig(vocab_size=17, d_model=16, n_heads=4, d_ff=32,
                            n_encoder_layers=2, n_decoder_layers=2, dropout=0.0,
                            max_len=12, label_smoothing=0.1)
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
