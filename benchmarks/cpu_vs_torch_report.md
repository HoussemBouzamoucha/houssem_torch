# Houssem Torch CPU vs PyTorch CPU

- **Benchmark:** [`cpu_vs_torch.py`](./cpu_vs_torch.py)
- **CPU:** Intel64 Family 6 Model 165 Stepping 2, GenuineIntel
- **Runtime:** Python 3.13.11, NumPy 2.5.4, PyTorch 2.15.0+cpu
- **Method:** autograd on for both; 3 warmups, medians of 15 runs (5 for the full model). Every case is first checked against PyTorch (forward output and all input gradients).
- **Ratio** = Houssem time / PyTorch time; below 1.00x means Houssem is faster.
- **f+b** = forward, then `.sum().backward()` (scalar outputs used as-is). The Adam row times one optimizer step.

## 1 thread(s)

| Case | Shape | HT fwd ms | PT fwd ms | fwd ratio | HT f+b ms | PT f+b ms | f+b ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| add | (1024, 512) | 1.131 | 0.800 | 1.41x | 2.486 | 1.820 | 1.37x |
| multiply | (1024, 512) | 1.057 | 0.770 | 1.37x | 5.268 | 2.144 | 2.46x |
| relu | (1024, 512) | 1.178 | 0.661 | 1.78x | 3.906 | 1.780 | 2.19x |
| gelu (tanh) | (1024, 512) | 12.102 | 3.702 | 3.27x | 26.673 | 8.002 | 3.33x |
| exp | (1024, 512) | 2.052 | 0.746 | 2.75x | 3.398 | 1.466 | 2.32x |
| log | (1024, 512) | 1.864 | 0.637 | 2.93x | 4.267 | 1.434 | 2.98x |
| sum(axis=-1) | (1024, 512) | 0.386 | 0.385 | 1.00x | 1.109 | 0.750 | 1.48x |
| mean(axis=-1) | (1024, 512) | 0.480 | 0.332 | 1.45x | 1.259 | 1.256 | 1.00x |
| matmul | (1024, 512) @ (512, 2048) | 33.570 | 32.193 | 1.04x | 116.344 | 120.758 | 0.96x |
| batched q @ kT | 2 x (16, 8, 64, 64) | 3.201 | 2.431 | 1.32x | 10.839 | 7.846 | 1.38x |
| softmax | (16, 8, 64, 64) | 4.784 | 2.100 | 2.28x | 9.282 | 3.788 | 2.45x |
| causal mask + softmax | (16, 8, 64, 64) | 12.904 | 3.291 | 3.92x | 14.779 | 6.435 | 2.30x |
| log_softmax | (1024, 16384) | 155.379 | 55.556 | 2.80x | 365.701 | 156.605 | 2.34x |
| layer_norm | (1024, 512) | 4.100 | 1.415 | 2.90x | 8.998 | 2.599 | 3.46x |
| embedding | ids (16, 64), table (16384, 512) | 1.154 | 0.810 | 1.42x | 25.902 | 10.555 | 2.45x |
| cross_entropy (ls=0.1) | (1024, 16384) | 144.803 | 69.555 | 2.08x | 348.958 | 387.708 | 0.90x |
| Linear | (16, 64, 512) -> 2048 | 65.801 | 120.425 | 0.55x | 325.334 | 222.673 | 1.46x |
| FeedForward | (16, 64, 512), d_ff 2048 | 143.152 | 81.752 | 1.75x | 272.947 | 230.314 | 1.19x |
| self-attention (causal) | (16, 64, 512), 8 heads | 66.241 | 45.038 | 1.47x | 184.267 | 140.614 | 1.31x |
| Transformer loss | base config (52.5M params), batch 8x32 | 931.135 | 687.759 | 1.35x | 2553.223 | 1819.479 | 1.40x |
| Adam step | 52.5M params | - | - | - | 259.497 | 607.194 | 0.43x |

## 6 thread(s)

| Case | Shape | HT fwd ms | PT fwd ms | fwd ratio | HT f+b ms | PT f+b ms | f+b ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| add | (1024, 512) | 0.985 | 0.658 | 1.50x | 3.102 | 1.979 | 1.57x |
| multiply | (1024, 512) | 1.345 | 0.727 | 1.85x | 11.349 | 3.412 | 3.33x |
| relu | (1024, 512) | 1.762 | 0.539 | 3.27x | 3.582 | 1.373 | 2.61x |
| gelu (tanh) | (1024, 512) | 3.429 | 1.082 | 3.17x | 9.656 | 3.324 | 2.91x |
| exp | (1024, 512) | 2.272 | 0.604 | 3.76x | 5.584 | 2.171 | 2.57x |
| log | (1024, 512) | 3.503 | 0.879 | 3.98x | 11.729 | 1.756 | 6.68x |
| sum(axis=-1) | (1024, 512) | 0.785 | 0.470 | 1.67x | 2.105 | 0.901 | 2.34x |
| mean(axis=-1) | (1024, 512) | 0.604 | 0.335 | 1.80x | 1.538 | 0.878 | 1.75x |
| matmul | (1024, 512) @ (512, 2048) | 14.419 | 16.030 | 0.90x | 60.538 | 54.949 | 1.10x |
| batched q @ kT | 2 x (16, 8, 64, 64) | 6.542 | 2.039 | 3.21x | 11.608 | 6.403 | 1.81x |
| softmax | (16, 8, 64, 64) | 1.862 | 0.821 | 2.27x | 5.886 | 2.880 | 2.04x |
| causal mask + softmax | (16, 8, 64, 64) | 6.840 | 2.499 | 2.74x | 14.054 | 3.856 | 3.64x |
| log_softmax | (1024, 16384) | 55.350 | 27.388 | 2.02x | 156.434 | 70.642 | 2.21x |
| layer_norm | (1024, 512) | 1.778 | 0.569 | 3.12x | 5.895 | 1.715 | 3.44x |
| embedding | ids (16, 64), table (16384, 512) | 1.213 | 0.565 | 2.15x | 23.610 | 6.786 | 3.48x |
| cross_entropy (ls=0.1) | (1024, 16384) | 37.896 | 32.118 | 1.18x | 116.855 | 91.906 | 1.27x |
| Linear | (16, 64, 512) -> 2048 | 18.380 | 15.999 | 1.15x | 60.331 | 51.083 | 1.18x |
| FeedForward | (16, 64, 512), d_ff 2048 | 34.412 | 36.427 | 0.94x | 132.939 | 120.607 | 1.10x |
| self-attention (causal) | (16, 64, 512), 8 heads | 33.760 | 20.980 | 1.61x | 142.450 | 60.450 | 2.36x |
| Transformer loss | base config (52.5M params), batch 8x32 | 468.561 | 558.162 | 0.84x | 3523.687 | 1242.212 | 2.84x |
| Adam step | 52.5M params | - | - | - | 199.085 | 598.004 | 0.33x |
