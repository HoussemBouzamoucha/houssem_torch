# Training quality on opus-100 fr->en (Kaggle T4, 2026-11-28)

Branch `train/quality` (dbb1421, merged into main as 8f4a60a),
`benchmarks/quality_kaggle.py`, HT_QUICK=0. houssem_torch only.

933,333 training pairs (16.4M target tokens, 17.6 per pair), 28 val batches,
2000 test sentences; data ready in 119 s. transformer_base, fp16, token-budget
buckets 16/32/48/64 ({16: 720, 32: 280, 48: 168, 64: 128} pairs per x4
micro-batch), 10000 optimizer steps per run, Noam warmup 2000, same initial
weights and batch order for every run. x1 = 1472 target tokens per step.
BLEU: sacrebleu on the 2000 test sentences, our GPU decoder (greedy; beam 4,
alpha 0.6).

| run | tokens / step | epochs | ms / step | val loss | BLEU greedy | BLEU beam 4 | wall s | memory GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x4 lr1 | 5,872 | 3.6 | 211.2 | 3.3180 | 26.62 | 27.18 | 2112 | 5.61 |
| x4 lr2 | 5,872 | 3.6 | 210.8 | 3.3137 | 26.18 | 26.97 | 2108 | 5.61 |
| x8 lr2.8 (2 x x4 accumulated) | 11,745 | 7.2 | 413.9 | 3.1695 | 28.07 | 28.87 | 4139 | 5.61 |

Val loss (step: val / train):

| step | x4 lr1 | x4 lr2 | x8 lr2.8 |
|---:|---:|---:|---:|
| 1000 | 5.668 / 5.758 | 5.319 / 5.473 | 4.529 / 4.741 |
| 2000 | 4.435 / 4.676 | 4.427 / 4.617 | 3.934 / 4.213 |
| 3000 | 3.922 / 4.164 | 3.907 / 4.151 | 3.562 / 3.800 |
| 4000 | 3.707 / 3.980 | 3.694 / 3.939 | 3.415 / 3.647 |
| 5000 | 3.573 / 3.820 | 3.560 / 3.817 | 3.332 / 3.580 |
| 6000 | 3.496 / 3.767 | 3.483 / 3.687 | 3.275 / 3.508 |
| 7000 | 3.438 / 3.693 | 3.424 / 3.649 | 3.235 / 3.455 |
| 8000 | 3.376 / 3.622 | 3.366 / 3.591 | 3.199 / 3.437 |
| 9000 | 3.346 / 3.582 | 3.343 / 3.557 | 3.183 / 3.415 |
| 10000 | 3.318 / 3.587 | 3.314 / 3.564 | 3.169 / 3.383 |

Greedy BLEU at step 5000 / 10000: x4 lr1 22.77 / 26.62, x4 lr2 22.78 / 26.18,
x8 25.72 / 28.07. Greedy decoding of the 2000 test sentences takes 3 s.

## Findings

- **LR factor 1 vs 2 at x4: no difference.** Val 3.318 vs 3.314 is noise, and
  BLEU goes slightly the other way (26.62 vs 26.18 greedy, 27.18 vs 26.97 beam).
  LR 2 only helps early on (step 1000: 5.32 vs 5.67), and the gap is gone by
  step 2000. Keep the default Noam factor 1.0.
- **x8 accumulated is the best model, but only because it trains on twice the
  tokens.** At equal compute (x8 step 5000, 2072 s vs x4 step 10000, 2112 s)
  x8 is slightly worse: val 3.332 vs 3.318, BLEU 25.72 vs 26.62. That point is
  mid-schedule for x8 (its LR is still 1.77e-3), so this does not penalize it
  much. Bigger batches do not buy quality per GPU-second here. They are worth
  it only when you want the best model in a fixed number of updates.
- **Not converged.** Every run is still improving at step 10000 (x8: -0.014
  val per 1000 steps). After 7.2 epochs there is no overfitting yet (opus_books
  overfit at about 13). Longer training will help.
- **Beam 4 adds +0.6 to +0.8 BLEU** consistently, and no looping was seen in
  the samples. Keep alpha 0.6.
- **Accumulation is free in memory:** x8 uses the same 5.61 GB as x4, and its
  step time is exactly 2x (413.9 vs 211.2 ms).
- Typical sample errors: pronoun/reflexive mistakes ("You were kissing you at
  the bus stop"), and "chez lui" taken literally ("stay with him").
