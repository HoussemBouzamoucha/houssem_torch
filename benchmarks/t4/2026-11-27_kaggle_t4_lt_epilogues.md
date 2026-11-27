# cuBLASLt FFN-backward epilogues (Kaggle T4, 2026-11-27)

Branch `gpu/faster-gemms`, commit 84422d1. 63 GPU tests pass, including the
ReLU-mask bit layout (row-major, least significant bit first) and both
fused modes matching the plain kernels' gradients.
`benchmarks/epilogue_bench.py --device 1`, transformer_base, 64/64 tokens,
median of 7 rounds x 10 steps, all modes on the same step, alternating.

| batch | plain ms | drelu_bgrad ms | drelu_wgrad ms |
|---|---:|---:|---:|
| 64 | 79.3 | 81.3 (0.976x) | 81.8 (0.970x) |
| 256 | 289.9 | 296.3 (0.978x) | 303.2 (0.956x) |

The modes:
- **plain:** cuBLAS multiply, then the `act_bwd` kernel (applies the mask and
  dropout, sums the bias gradient).
- **drelu_bgrad:** one cuBLASLt multiply that applies the mask and writes the
  fp16 bias gradient.
- **drelu_wgrad:** a cuBLASLt multiply that applies the mask, plus the fp32 bias
  gradient from the weight-gradient multiply (BGRADA).

## Why it loses

The cuBLASLt epilogue kernels on Turing run much slower than plain cuBLAS for
the same shape. That holds after timing all 8 heuristic candidates; they fall
within ~10% of each other.

| multiply (µs per call) | plain cuBLAS | best cuBLASLt |
|---|---:|---:|
| 4096x2048x512 B^T (input grad) | ~367 | 642 (drelu_bgrad), 938 (drelu) |
| 512x2048x4096 A^T fp32 +C (weight grad) | ~412 | 665 (bgrad_b) |
| 16384x2048x512 B^T | ~1630 | 2203 (drelu), 2792 (drelu_bgrad) |
| 512x2048x16384 A^T fp32 +C | ~1440 | 1253 (bgrad_b) |

The kernel they replace, `act_bwd`, costs only 2.6 ms per step at batch 64
(9.8 ms at batch 256), less than the extra multiply time.

## Decision

`TrainStep(fused_epilogues=False)` stays the default. The ctypes cuBLASLt
binding (`driver.gemm_epilogue`, autotuned, `lt_algorithms()`) stays in, tested,
for newer GPUs (Ampere and later have fast epilogue kernels). Combined with the
layout run earlier the same day
(`2026-11-27_kaggle_t4_gemm_layouts.md`), matmul-side speedups on the T4 are
exhausted.
