# Matmul layouts and output dtype vs torch.matmul (Kaggle T4, 2026-11-27)

Branch `gpu/faster-gemms`, commit 94242c6, `benchmarks/gemm_bench.py --device 1`,
batch 64 and 256 x 64/64, transformer_base. Every multiply that one training step
makes is timed in isolation. All the variants of a multiply run back to back in the
same round, and each value is the median of 7 rounds of 20.

## Totals (µs of matmul per training step)

| variant | batch 64 | batch 256 |
|---|---:|---:|
| ours (as the step runs now) | 58343 | 234939 |
| ours alt (weights stored (out, in), like nn.Linear) | 58736 | 239343 |
| ours fp16 (weight grads written as fp16, beta 0) | 56857 | 231869 |
| torch.matmul, our layout | 56879 | 231066 |
| torch.matmul, nn.Linear layout | 57207 | 235743 |
| best of the ours variants, picked per multiply | 56525 | 231414 |

## Findings

- **Our multiplies match PyTorch's.** With the same layout and dtypes, the times
  agree with torch.matmul to within noise. The "matmuls ~10% / ~3% slower"
  measured earlier came from timing the two engines at different moments while
  the T4's power-capped clock drifted.
- **The weight layout makes no difference.** Storing weights (out, in) made the
  total slightly slower at both batch sizes, and much slower for the logits
  forward at batch 256 (15.6 vs 12.8 ms). Dropped.
- **The fp32 output with accumulation is the only difference** from what torch
  runs. It costs 1.5 ms per step at batch 64 (2.5% of matmul time) and 3.1 ms at
  batch 256 (1.3%). PyTorch pays for it differently: its fp16 weight gradient
  goes through a separate cast-and-accumulate into the fp32 `.grad`.
- **Why the speedup shrinks at big batches:** the matmuls are the same cuBLAS
  kernels in both engines and take a growing share of the step, so this is
  Amdahl's law, not slower matmuls. Matmul time is ~58 ms of the ~90 ms step at
  batch 64.

## Raw tables

See the Kaggle output in this session (pasted into the chat). The per-multiply
rows go into `gemm_bench_b64_results.json` and `gemm_bench_b256_results.json`
on the Kaggle machine.
