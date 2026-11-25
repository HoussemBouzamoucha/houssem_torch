# houssem_torch TrainStep vs PyTorch, Kaggle T4 (2026-11-25): attention v3, GPU swap, async uploads

Four runs of [`benchmarks/cuda_train_kaggle.py`](../cuda_train_kaggle.py) on
branch `gpu/attention-v3`. Kaggle, 2 × Tesla T4 (driver 580.178), PyTorch
2.11.0+cu130, transformer_base (52.5M parameters), batch 64 × 64/64, fp16, opus_books
fr→en. Machine-readable numbers for every run, including 2026-11-24's: [`history.json`](history.json).

## Bottom line

On the same card, houssem_torch is **1.20–1.22× faster per step than torch.compile**,
and since the async-upload fix **training wall time shows the same 1.21×**. Attention now
beats PyTorch's. Matrix multiplies (cuBLAS, ~10% slower than torch's calls) are
the only category we lose.

| run | commit | GPUs (ht / torch) | ms/step b64 ht | torch.compile | eager | speedup | 3000 steps ht / torch | val loss ht / torch |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A 16:31 | `3156790` | 0 / 1 | 81.1 | 102.4 | 122.5 | 1.26× | 260 / 333 s | 4.769 / 4.791 |
| B 17:43 | `35525b2` | 0 / 1 | 77.2 | 94.9 | 117.8 | 1.23× | 289 / 302 s | 4.781 / 4.782 |
| C 18:03 | `8abba38` | **1 / 0** (swapped) | 78.1 | 110.1 | 129.3 | 1.41×* | 261 / 348 s | 4.776 / 4.788 |
| D 18:24 | `e82fad4` | **1 / 1** (same card) | 87.7 | 104.9 | 130.9 | **1.20×** | **263 / 317 s (1.21×)** | 4.775 / 4.796 |

\* inflated: PyTorch ran on the slower card (cuda:0).

Startup: houssem_torch 0.5 s every run; torch.compile 30–31 s (2.1 s in run D:
warm compile cache). Memory: houssem_torch 2.03 GB, torch.compile 2.19–2.20,
eager 2.68–2.69.

## What changed between runs

- **A, `3156790`**: attention kernels load B fragments with `ldmatrix`, backward 8 warps.
- **B, `35525b2`**: attention loads staged in registers (every global load issued before any
  shared store; sm_75 has no `cp.async`), D = rowsum(dO·O) fused into the backward's
  loads, maximum shared-memory carveout, backward 16 warps. Then `f0e2cfb`: forward keeps
  Q in shared memory (QREG=0), since Q-in-registers was ~5 µs slower.
- **C, `8abba38`**: no engine change; `HT_GPU` / `TORCH_GPU` env vars, GPU clock /
  temperature readout around training.
- **D, `e82fad4`**: batch ids are uploaded from two pinned host slots with
  `copy_from_async` (event-guarded). Before, uploads came from pageable memory, so CUDA
  waited for the previous step to finish first. Training logs show host batch time and SM clock.

## Attention kernels alone (`benchmarks/attn_bench.py`, B=64 H=8 L=64 DK=64 p=0.1, µs)

| variant | fwd non-causal | bwd non-causal | fwd causal | bwd causal |
|---|---:|---:|---:|---:|
| `afbb189`, bwd 4 warps (11-24) | 243.6 | 873.4 | 155.8 | 525.1 |
| `3156790`, bwd 4 warps | 156.8 | 495.3 | 152.2 | 483.5 |
| `3156790`, bwd 8 warps | 156.5 | 346.6 | 153.0 | 347.4 |
| `3156790`, bwd 16 warps | 157.0 | 303.4 | 152.7 | 302.3 |
| `35525b2`, bwd 16, QREG=0 (default) | 162.4 | 279.4 | 154.8 | 278.1 |
| `35525b2`, bwd 16, QREG=1 | 167.0 | 279.9 | 162.5 | 263.6 |

The backward went from 873 to ~279 µs. The forward is stuck at ~155 µs even with
staged loads and 3 blocks per SM, so it is not limited by load latency. The memory-traffic
lower bound is ~55 µs forward and ~105 µs backward.

## Where the time goes (ms/step at batch 64, scaled to each engine's step time)

| work | A ht / tc | B ht / tc | C ht / tc | D ht / tc |
|---|---:|---:|---:|---:|
| matrix multiplies | 49.3 / 52.3 | 47.5 / 42.9 | 48.6 / 57.5 | 56.3 / 50.8 |
| attention | 11.5 / 10.2 | **8.9 / 9.6** | 9.0 / 10.9 | 10.0 / 10.6 |
| LayerNorm + residual | 5.3 / 11.1 | 5.4 / 12.2 | 5.3 / 11.9 | 5.5 / 12.5 |
| activation, dropout, bias | 5.4 / 12.9 | 5.5 / 12.9 | 5.4 / 13.7 | 5.7 / 13.2 |
| loss | 1.1 / 4.6 | 1.1 / 4.7 | 1.1 / 4.9 | 1.1 / 5.0 |
| optimizer | 7.5 / 11.1 | 7.7 / 12.1 | 7.6 / 10.7 | 7.9 / 12.4 |
| embedding, other | 1.1 / 0.4 | 1.1 / 0.4 | 1.1 / 0.4 | 1.2 / 0.4 |

(ht = houssem_torch, tc = torch.compile. In runs C and D, the matrix multiplies
on the same card: ours are ~10% slower.)

## Findings to remember

1. **Kaggle's cuda:0 is the slower card.** houssem_torch ran 77–78 ms/step on either
   card. torch.compile ran 95 ms on cuda:1 and 110 ms on cuda:0. Run B's houssem slowdown
   during training (20 → 25 s per 250 steps) was the card, not the code.
2. **T4s run power- or heat-capped under sustained load.** In run D the SM clock was
   975–1290 MHz out of 1590, and the whole session was ~12% slower. Compare engines only
   within one run, ideally both on the same card (`HT_GPU=1 TORCH_GPU=1`).
3. **Training wall now matches the step benchmark** (run D: 87.5 vs 87.7 ms/step). Host
   batch building costs 0.4 ms/step. The upload fix was not A/B-tested within one session.
4. **Training quality is identical** in every run: the loss curves track step for step,
   and the validation losses are within ±0.02.

## Next

Matrix multiplies: cuBLAS algorithm choice, bias in the GEMM epilogue (cuBLASLt),
and layouts. That is ~5 ms/step at stake. Then length buckets.
