"""Train a tiny encoder-decoder transformer to reverse sequences, on CPU.

A complete training loop on a toy "translation" task: the source is a random
sequence of tokens and the target is the same tokens reversed. It exercises
everything a translation model needs (padding masks, the causal
mask, cross-attention, label-smoothed loss, Adam with warmup, greedy
decoding) and finishes in about two minutes. Expect ~99% exact-match
accuracy; the loss levels off near 0.6, not 0, because label smoothing keeps
the model from ever being fully certain (that floor is intended).

Run from the repository root:
    python examples/train_toy_reverse.py
"""
import time
import numpy as np
import houssem_torch as ht
from houssem_torch import optim
from houssem_torch.models import Transformer, TransformerConfig

PAD, BOS, EOS = 0, 1, 2
VOCAB, MAX_LEN, BATCH, STEPS = 16, 8, 64, 1500


def make_batch(rng, batch):
    """Variable-length random sequences (padded) and their reversals."""
    lengths = rng.integers(3, MAX_LEN + 1, size=batch)
    src = np.full((batch, MAX_LEN), PAD)
    tgt_in = np.full((batch, MAX_LEN + 1), PAD)
    tgt_out = np.full((batch, MAX_LEN + 1), PAD)
    for i, n in enumerate(lengths):
        seq = rng.integers(3, VOCAB, size=n)
        src[i, :n] = seq
        tgt_in[i, 0], tgt_in[i, 1:n + 1] = BOS, seq[::-1]   # <bos> + target
        tgt_out[i, :n], tgt_out[i, n] = seq[::-1], EOS      # target + <eos>
    return src, tgt_in, tgt_out


def main():
    ht.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = TransformerConfig(vocab_size=VOCAB, d_model=64, n_heads=4, d_ff=128,
                            n_encoder_layers=2, n_decoder_layers=2, dropout=0.1,
                            max_len=MAX_LEN + 2, label_smoothing=0.1)
    model = Transformer(cfg)
    opt = optim.Adam(model.parameters())
    schedule = optim.NoamSchedule(opt, d_model=cfg.d_model, warmup_steps=200,
                                  factor=2.0)
    print(f"model: {model.num_parameters():,} parameters")

    start = time.time()
    for step in range(1, STEPS + 1):
        src, tgt_in, tgt_out = make_batch(rng, BATCH)
        model.train()
        opt.zero_grad()
        loss = model.loss(src, tgt_in, tgt_out)
        loss.backward()
        ht.nn.clip_grad_norm(model.parameters(), max_norm=1.0)
        lr = schedule.step()
        opt.step()
        if step % 100 == 0:
            print(f"step {step:5d}  loss {float(loss.numpy()):.4f}  "
                  f"lr {lr:.2e}  {time.time() - start:.0f}s")

    src, _, tgt_out = make_batch(rng, 500)
    pred = model.translate_greedy(src, max_new_tokens=MAX_LEN + 1)
    ends = np.argmax(tgt_out == EOS, axis=1)
    exact = np.mean([np.array_equal(p[:e + 1], t[:e + 1])
                     for p, t, e in zip(pred, tgt_out, ends)])
    print(f"\nexact-match accuracy on 500 new sequences: {exact:.1%}")
    for i in range(3):
        n = ends[i]
        print(f"  {src[i, :n].tolist()} -> {pred[i, :n].tolist()}")


if __name__ == "__main__":
    main()
