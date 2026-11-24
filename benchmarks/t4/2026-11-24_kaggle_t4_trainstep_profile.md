# houssem_torch TrainStep profile on a Kaggle T4 (2026-11-24)

`python benchmarks/trainstep_profile.py --batch-sizes 64 32 96 128` at commit
`cb67c53`. Synthetic batches (random lengths, min fill 0.5, so ~2990 target
tokens per batch of 64 instead of opus_books' 1471; same shapes and work).

Step time here is 111.4 ms vs 101.8 ms in the benchmark run earlier the same
day on another Kaggle session (same shapes, same code): T4 clocks vary
between sessions, so compare shares, and re-time PyTorch in the same session.

- Step time (unprofiled, median of 3 x 20): **111.4 ms**
- Sum of launch times (profiled, per step): 112.4 ms
- Launches per step: 592
- Matmuls: 66.4 ms, 1.32 TFLOP/step, 19.8 TFLOP/s (31% of 65 TFLOP/s fp16 peak)

## By kind of work

| work | ms/step | share | launches/step |
|---|---:|---:|---:|
| matmul: linear layers | 45.22 | 40.2% | 198 |
| head split / merge | 13.84 | 12.3% | 84 |
| matmul: attention (QK^T, PV) | 10.74 | 9.6% | 108 |
| optimizer (norm, scaling, AdamW) | 8.95 | 8.0% | 4 |
| residual + dropout + LayerNorm | 7.84 | 7.0% | 64 |
| FFN bias + activation + dropout | 7.40 | 6.6% | 24 |
| matmul: output projection (logits) | 6.98 | 6.2% | 3 |
| loss (cross-entropy fwd+bwd) | 4.03 | 3.6% | 1 |
| matmul: bias gradients (ones x g) | 3.48 | 3.1% | 66 |
| attention softmax + dropout | 2.78 | 2.5% | 36 |
| memset (zero gradients) | 0.94 | 0.8% | 2 |
| embedding backward | 0.18 | 0.2% | 2 |

## By phase and block (ms/step)

| block | forward | backward | total |
|---|---:|---:|---:|
| embed | 0.18 | 0.18 | 0.36 |
| self | 12.07 | 20.96 | 33.03 |
| cross | 6.24 | 10.73 | 16.97 |
| ffn | 11.96 | 21.50 | 33.46 |
| residual_ln | 3.28 | 4.38 | 7.66 |
| head | 6.59 | 4.42 | 11.01 |
| optimizer |  |  | 9.87 |

## Matrix multiplies by shape (m x n x k, row-major)

| shape | kind | calls/step | ms/step | TFLOP/s |
|---|---:|---:|---:|---:|
| 512x2048x4096 A^T | linear layers | 12 | 4.63 | 22.3 |
| 4096x2048x512 B^T | linear layers | 12 | 4.13 | 25.0 |
| 2048x512x4096 A^T | linear layers | 12 | 4.10 | 25.1 |
| 4096x2048x512 | linear layers | 12 | 4.02 | 25.6 |
| 4096x512x2048 B^T | linear layers | 12 | 4.01 | 25.7 |
| 4096x512x2048 | linear layers | 12 | 3.91 | 26.4 |
| 64x64x64 x512 B^T | attention (QK^T, PV) | 36 | 3.66 | 2.6 |
| 64x64x64 x512 | attention (QK^T, PV) | 36 | 3.57 | 2.7 |
| 64x64x64 x512 A^T | attention (QK^T, PV) | 36 | 3.51 | 2.8 |
| 4096x512x1536 B^T | linear layers | 12 | 3.19 | 24.2 |
| 512x1536x4096 A^T | linear layers | 12 | 3.12 | 24.8 |
| 4096x1536x512 | linear layers | 12 | 3.01 | 25.7 |
| 4096x512x512 B^T | linear layers | 24 | 2.64 | 19.5 |
| 4096x16384x512 B^T | output projection (logits) | 1 | 2.56 | 26.8 |
| 512x512x4096 A^T | linear layers | 24 | 2.55 | 20.2 |
| 4096x512x512 | linear layers | 24 | 2.54 | 20.3 |
| 16384x512x4096 A^T | output projection (logits) | 1 | 2.28 | 30.1 |
| 4096x512x16384 | output projection (logits) | 1 | 2.14 | 32.1 |
| 1x1536x4096 | bias gradients (ones x g) | 12 | 1.32 | 0.1 |
| 4096x512x1024 B^T | linear layers | 6 | 1.15 | 22.4 |
| 512x1024x4096 A^T | linear layers | 6 | 1.13 | 22.7 |
| 4096x1024x512 | linear layers | 6 | 1.07 | 24.0 |
| 1x512x4096 | bias gradients (ones x g) | 36 | 0.98 | 0.2 |
| 1x2048x4096 | bias gradients (ones x g) | 12 | 0.92 | 0.2 |
| 1x1024x4096 | bias gradients (ones x g) | 6 | 0.26 | 0.2 |

## Our kernels

| kernel | ms/step | calls/step | us/call |
|---|---:|---:|---:|
| adamw | 8.01 | 1 | 8014 |
| merge_heads | 7.24 | 42 | 172 |
| split_heads | 6.60 | 42 | 157 |
| ln_bwd | 4.38 | 32 | 137 |
| bias_act_fwd | 4.03 | 12 | 336 |
| cross_entropy | 4.03 | 1 | 4027 |
| add_ln_fwd | 3.47 | 32 | 108 |
| act_bwd | 3.37 | 12 | 281 |
| softmax_fwd | 1.41 | 18 | 78 |
| softmax_bwd | 1.37 | 18 | 76 |
| fill | 0.94 | 2 | 472 |
| sum_squares | 0.92 | 1 | 925 |
| embed_bwd | 0.18 | 2 | 89 |
| step_prepare | 0.01 | 1 | 6 |
| reset_scalars | 0.01 | 1 | 6 |

## Batch sizes

| batch | ms/step | tokens/s | setup s | memory GB |
|---|---:|---:|---:|---:|
| 64 | 111.4 | 26800 | 1.1 | 2.26 |
| 32 | 61.1 | 25334 | 0.6 | 1.57 |
| 96 | 163.2 | 28499 | 0.6 | 2.95 |
| 128 | 226.1 | 26692 | 0.6 | 3.64 |
