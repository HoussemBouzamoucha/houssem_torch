"""Beam search bookkeeping, independent of where the model runs.

The model side is a callback, so the GPU Translator and a plain CPU
reference share this exact logic:

    step_fn(tokens, parents, t) -> (logprobs, ids)

tokens (n·beam,) are fed at position t; parents (n·beam,) says which row's
state (key/value cache, prefix) each row continues (None at t = 0, when
every row is fresh); the result holds, per row, the best few log-probs of
the next token, best first, and their token ids, shape (n·beam, kk), with
kk >= 2·beam so that `beam` live hypotheses can always be refilled.

Rules (as in common NMT toolkits):
  - candidates = live score + next-token log-prob, ranked over all beams of
    a sentence; ranks < beam ending in <eos> become finished hypotheses,
    scored by sum log-prob / ((5 + length) / 6) ** alpha (GNMT length
    penalty; length counts the <eos>);
  - the best non-<eos> candidates refill the `beam` live hypotheses;
  - a sentence is done once it has `beam` finished hypotheses; at the end
    the live ones still standing are finished as they are.
"""
from __future__ import annotations

import numpy as np


def length_penalty(length: int, alpha: float) -> float:
    return ((5.0 + length) / 6.0) ** alpha


def beam_search(step_fn, n: int, beam: int, max_new: int, *, bos: int,
                eos: int, pad: int, alpha: float = 0.6) -> np.ndarray:
    """Best hypothesis per sentence: int ids (n, ≤ max_new), without
    <bos>, ending in <eos> when one was produced, padded with `pad`."""
    K = int(beam)
    if K < 1 or max_new < 1:
        raise ValueError("beam and max_new must be at least 1")
    rows = n * K
    scores = np.full((n, K), -np.inf)
    scores[:, 0] = 0.0                       # all beams start identical
    hist = np.zeros((n, K, 0), np.int64)
    tokens = np.full(rows, bos, np.int32)
    parents = None
    finished = [[] for _ in range(n)]
    done = np.zeros(n, bool)
    for t in range(max_new):
        vals, ids = step_fn(tokens, parents, t)
        kk = vals.shape[1]
        cand = scores[:, :, None] + vals.reshape(n, K, kk)
        ids = ids.reshape(n, K, kk)
        new_scores = np.full((n, K), -np.inf)
        new_hist = np.full((n, K, t + 1), pad, np.int64)
        new_hist[:, :, :t] = hist
        new_par = np.arange(rows, dtype=np.int32)
        new_tok = np.full(rows, pad, np.int32)
        for i in range(n):
            if done[i]:
                continue
            flat = cand[i].reshape(-1)
            alive = 0
            for rank, c in enumerate(np.argsort(-flat, kind="stable")):
                sc = flat[c]
                if not np.isfinite(sc):
                    break
                k, j = divmod(int(c), kk)
                tok = int(ids[i, k, j])
                if tok == eos:
                    if rank < K:
                        seq = hist[i, k].tolist() + [eos]
                        finished[i].append((sc / length_penalty(len(seq), alpha),
                                            seq))
                    continue
                new_scores[i, alive] = sc
                new_hist[i, alive, :t] = hist[i, k]
                new_hist[i, alive, t] = tok
                new_par[i * K + alive] = i * K + k
                new_tok[i * K + alive] = tok
                alive += 1
                if alive == K:
                    break
            if len(finished[i]) >= K or alive == 0:
                done[i] = True
        scores, hist, parents, tokens = new_scores, new_hist, new_par, new_tok
        if done.all():
            break
    for i in range(n):
        if not done[i]:
            for k in range(K):
                if np.isfinite(scores[i, k]):
                    seq = hist[i, k].tolist()
                    finished[i].append((scores[i, k]
                                        / length_penalty(len(seq), alpha), seq))
    best = [max(f, key=lambda h: h[0])[1] if f else [] for f in finished]
    width = max(1, max(len(s) for s in best))
    out = np.full((n, width), pad, np.int64)
    for i, s in enumerate(best):
        out[i, :len(s)] = s
    return out
