"""Beam search bookkeeping (CPU, toy scorer): exhaustive search when the beam
is wide enough, greedy when it is 1."""
import itertools

import numpy as np

from houssem_torch.models.beam import beam_search, length_penalty

PAD, BOS, EOS, V = 0, 1, 2, 5          # words: 3 and 4


def toy_logprobs(prefix, sentence):
    """Deterministic next-token log-probs for a prefix (bos excluded);
    pad and bos are impossible."""
    seed = hash((sentence, tuple(prefix))) % (2 ** 32)
    z = np.random.default_rng(seed).standard_normal(V)
    z[PAD] = z[BOS] = -np.inf
    m = z[np.isfinite(z)].max()
    return z - (m + np.log(np.exp(z[np.isfinite(z)] - m).sum()))


def make_step(n, beam):
    state = {"prefix": [[] for _ in range(n * beam)]}

    def step_fn(tokens, parents, t):
        if parents is not None:
            state["prefix"] = [state["prefix"][p] + [int(tokens[r])]
                               for r, p in enumerate(parents)]
        kk = min(2 * beam, V)
        vals = np.zeros((n * beam, kk))
        ids = np.zeros((n * beam, kk), np.int64)
        for r, pre in enumerate(state["prefix"]):
            lp = toy_logprobs(pre, r // beam)
            order = np.argsort(-lp, kind="stable")[:kk]
            vals[r], ids[r] = lp[order], order
        return vals, ids
    return step_fn


def score(seq, sentence, alpha):
    total = sum(toy_logprobs(seq[:i], sentence)[tok] for i, tok in enumerate(seq))
    return total / length_penalty(len(seq), alpha)


def brute_force(sentence, max_new, alpha):
    best, best_seq = -np.inf, None
    for L in range(1, max_new + 1):
        for words in itertools.product((3, 4), repeat=L - 1):
            for last in (EOS, 3, 4):
                if last != EOS and L < max_new:
                    continue          # unfinished only at the length limit
                seq = list(words) + [last]
                s = score(seq, sentence, alpha)
                if s > best:
                    best, best_seq = s, seq
    return best_seq


def test_wide_beam_is_exhaustive():
    n, max_new, alpha = 4, 4, 0.6
    beam = 3 ** max_new                  # never prunes
    out = beam_search(make_step(n, beam), n, beam, max_new, bos=BOS, eos=EOS,
                      pad=PAD, alpha=alpha)
    for i in range(n):
        got = [t for t in out[i] if t != PAD]
        assert got == brute_force(i, max_new, alpha), i


def test_beam_one_is_greedy():
    n, max_new = 5, 6
    out = beam_search(make_step(n, 1), n, 1, max_new, bos=BOS, eos=EOS,
                      pad=PAD, alpha=0.6)
    for i in range(n):
        seq = []
        for _ in range(max_new):
            tok = int(np.argmax(toy_logprobs(seq, i)))
            seq.append(tok)
            if tok == EOS:
                break
        assert [t for t in out[i] if t != PAD] == seq, i
