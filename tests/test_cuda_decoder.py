"""GPU decoding with a key/value cache against the CPU Transformer.

The cache-based, one-token-at-a-time logits must equal a full teacher-forced
forward pass, and greedy translation must give the CPU model's tokens.
Skipped without CuPy and a CUDA GPU.
"""
import numpy as np
import pytest

import houssem_torch as ht
from houssem_torch.backends.cuda import driver

pytestmark = pytest.mark.skipif(not driver.is_available(),
                                reason="needs CuPy and a CUDA GPU")

from houssem_torch.backends.cuda.decoder import Translator  # noqa: E402
from houssem_torch.backends.cuda.train_step import TrainStep  # noqa: E402
from houssem_torch.models import Transformer, TransformerConfig  # noqa: E402

B, S, T = 3, 7, 5


def config(**over):
    base = dict(vocab_size=61, d_model=64, n_heads=4, d_ff=96,
                n_encoder_layers=2, n_decoder_layers=2, dropout=0.0,
                max_len=16, label_smoothing=0.1)
    base.update(over)
    return TransformerConfig(**base)


def data(cfg, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.integers(3, cfg.vocab_size, (B, S))
    src[0, 4:] = cfg.pad_id
    src[2, 2:] = cfg.pad_id
    tgt = rng.integers(3, cfg.vocab_size, (B, T + 1))
    tgt[:, 0] = cfg.bos_id
    tgt[1, 4:] = cfg.pad_id
    return src, tgt[:, :-1], tgt[:, 1:]


def trained(cfg, precision, steps=30):
    """A step trained a little (peaked, not uniform, logits) and a CPU model
    holding the same weights."""
    ht.manual_seed(0)
    model = Transformer(cfg)
    step = TrainStep(model, B, S, T, precision=precision)
    for i in range(steps):
        step(*data(cfg, i % 3), lr=3e-3)
    model.load_state_dict(step.state_dict())
    model.eval()
    return model, step


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
@pytest.mark.parametrize("attention_bias", [True, False])
def test_cached_logits_match_full_forward(precision, attention_bias):
    cfg = config(attention_bias=attention_bias)
    model, step = trained(cfg, precision)
    src, tgt_in, _ = data(cfg)
    tr = Translator(step, B, S)
    got = tr.decode_logits(src, tgt_in)
    with ht.no_grad():
        ref = model(src, tgt_in).numpy()
    # Padded target positions: the full pass masks them as keys, decoding
    # never feeds them to an unfinished sentence. Compare real tokens only.
    real = tgt_in != cfg.pad_id
    got, ref = got[real], ref[real]
    if precision == "fp32":
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)
    else:
        err = np.abs(got - ref).max() / np.abs(ref).max()
        assert err < 2e-2, err
        assert (got.argmax(-1) == ref.argmax(-1)).mean() > 0.95


def test_greedy_matches_cpu_fp32():
    cfg = config()
    model, step = trained(cfg, "fp32")
    src, _, _ = data(cfg, 1)
    tr = Translator(step, B, S)
    got = tr.translate_greedy(src)
    ref = model.translate_greedy(src)
    np.testing.assert_array_equal(got, ref)


def test_greedy_pads_short_batches_and_is_repeatable():
    """Fewer sentences than batch_size and narrower sources are padded; the
    cache from one call doesn't leak into the next."""
    cfg = config()
    model, step = trained(cfg, "fp32")
    src, _, _ = data(cfg, 1)
    tr = Translator(step, B + 2, S + 3)
    one = tr.translate_greedy(src[:2, :5], max_new_tokens=8)
    two = tr.translate_greedy(src[:2, :5], max_new_tokens=8)
    np.testing.assert_array_equal(one, two)
    assert one.shape[0] == 2 and one.shape[1] <= 8
    ref = model.translate_greedy(src[:2, :5], max_new_tokens=8)
    np.testing.assert_array_equal(one, ref)
    with pytest.raises(ValueError):
        tr.translate_greedy(np.zeros((B + 3, S), np.int64))


def cpu_beam(model, src, beam, alpha, max_new):
    """Reference: the same bookkeeping, with the CPU model re-decoding every
    prefix in full each step."""
    from houssem_torch.models.beam import beam_search
    cfg = model.cfg
    n = src.shape[0]
    rep = np.repeat(src, beam, axis=0)
    with ht.no_grad():
        memory, src_pad = model.encode(rep)
    state = {"ys": np.full((n * beam, 1), cfg.bos_id, np.int64)}
    kk = 2 * beam

    def step_fn(tokens, parents, t):
        if parents is not None:
            state["ys"] = np.concatenate([state["ys"][parents],
                                          tokens[:, None].astype(np.int64)], 1)
        with ht.no_grad():
            z = model.decode(state["ys"], memory, src_pad).numpy()[:, -1]
        z = z.astype(np.float64)
        lp = z - (z.max(1, keepdims=True)
                  + np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)))
        order = np.argsort(-lp, axis=1, kind="stable")[:, :kk]
        return np.take_along_axis(lp, order, 1), order
    return beam_search(step_fn, n, beam, max_new, bos=cfg.bos_id,
                       eos=cfg.eos_id, pad=cfg.pad_id, alpha=alpha)


def test_beam_one_matches_greedy():
    cfg = config()
    _, step = trained(cfg, "fp32")
    src, _, _ = data(cfg, 2)
    tr = Translator(step, B, S)
    np.testing.assert_array_equal(tr.translate_beam(src, beam=1),
                                  tr.translate_greedy(src))


@pytest.mark.parametrize("alpha", [0.0, 0.6])
def test_beam_matches_cpu_reference(alpha):
    """GPU log-softmax/top-k and the cache reordering, against the CPU
    model with the same bookkeeping (fp32)."""
    cfg = config()
    model, step = trained(cfg, "fp32")
    src, _, _ = data(cfg, 1)
    tr = Translator(step, B * 4, S)
    got = tr.translate_beam(src, beam=4, alpha=alpha, max_new_tokens=10)
    ref = cpu_beam(model, src, 4, alpha, 10)
    np.testing.assert_array_equal(got, ref)
    # and the cache is reset between calls (greedy after beam is unchanged)
    np.testing.assert_array_equal(tr.translate_greedy(src),
                                  model.translate_greedy(src))
