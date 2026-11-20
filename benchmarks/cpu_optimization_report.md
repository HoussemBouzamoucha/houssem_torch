# CPU optimization: Houssem Torch vs PyTorch, before and after

**Date:** 2026-11-20
**Benchmark:** [`cpu_vs_torch.py`](./cpu_vs_torch.py) (raw per-run tables in
[`cpu_vs_torch_report.md`](./cpu_vs_torch_report.md))

This report compares Houssem Torch's CPU engine with PyTorch's CPU eager
mode **before** and **after** the CPU optimization work. That work added
fused multi-core Numba kernels, cleaned up the NumPy reference and made
autograd allocate less.

## Summary

A full training step (forward, backward and Adam update) on the base
Transformer now runs at about the same speed as PyTorch:

| Full training step, base config | Houssem (fast path) | Houssem (NumPy only) | PyTorch |
|---|---:|---:|---:|
| 6 threads | **1.03x** | 1.88x | 1.00x |
| 1 thread | **1.09x** | 1.17x | 1.00x |

Values are time relative to PyTorch: below 1.00x is faster than PyTorch,
above is slower. The "NumPy only" column (`HOUSSEM_TORCH_NUMBA=0`) already
includes the NumPy cleanups, so the original code was slower than that column
shows.

The largest single-op gains, forward + backward at 6 threads:

| Op | Before | After | Speedup of Houssem itself |
|---|---:|---:|---:|
| gelu (tanh) | 16.53x | 2.91x | 4.3x |
| layer_norm | 10.07x | 3.44x | 2.7x |
| softmax | 4.94x | 2.04x | 1.5x |
| log_softmax (16k vocab) | 5.30x | 2.21x | 1.9x |
| cross_entropy (16k vocab) | 3.96x | 1.27x | 2.7x |
| Adam step (52.5M params) | 1.95x | **0.33x** | 6.0x |

Adam is now **3x faster than PyTorch's** `torch.optim.Adam`.

## Setup

- **CPU:** Intel Core i7-10750H (6 cores / 12 threads, AVX2), laptop
- **Runtime:** Windows, Python 3.13.11, NumPy 2.5.4 (OpenBLAS 0.3.34),
  Numba 0.68.1, PyTorch 2.15.0+cpu
- **Shapes:** from `configs/transformer_base.json`: d_model 512, d_ff 2048,
  8 heads, 16,384-token vocabulary. Ops use 16 sentences x 64 tokens (1,024
  rows). The full model uses 8 x 32 tokens (52.5M parameters, dropout off).
- **Method:** autograd on in both libraries. Each case is first checked
  against PyTorch (forward output and every input gradient); only matching
  computations are timed. Timings are medians: 15 runs per op, 5 for the
  model, after 3 warmups. Thread counts are pinned through
  `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
  `NUMBA_NUM_THREADS` and `torch.set_num_threads`.
- **f+b:** forward, then `.sum().backward()` (scalar outputs used as-is).

### Measurement noise

This laptop throttles under sustained load, so absolute times drift by up to
2x across a long run. PyTorch's own full-model time, for example, ranged from
818 to 1,242 ms at 6 threads between runs. Consequences:

- The **summary** full-step numbers are the reliable ones. They come from one
  process that runs Houssem fast, Houssem NumPy and PyTorch in turn, 6 times
  each, so all three see the same machine conditions.
- In the **per-op tables**, the "before" and "after" columns come from
  separate runs. Ratios against PyTorch hold up better than raw
  milliseconds, but rows that the work never touched still moved a lot. For
  example, multiply at 6 threads went from 0.39x to 3.33x and log from
  2.37x to 6.68x, with no code change behind either. Treat moves below
  about 1.5x as noise.
- The "after" 6-thread **Transformer loss** row (2.84x) was measured during a
  throttled stretch. A same-process A/B taken just before it gave 1,604 ms
  with the fast path and 1,782 ms without, against roughly 820 to 1,240 ms
  for PyTorch.

## What changed

1. **Fused Numba kernels** (`houssem_torch/backends/cpu/fast.py`). Each
   kernel makes one pass over the data, writes one output array and spreads
   rows over all cores. Accumulations run in float64.
   - gelu forward and backward
   - softmax and log_softmax forward and backward (along the last axis)
   - layer_norm forward and backward: dweight and dbias are summed from
     per-chunk partials, so no two threads write the same memory
   - cross_entropy computed straight from the logits: it stores one
     log-sum-exp per row instead of a full (N, V) log-probability array
   - embedding backward (sequential scatter-add)
   - the Adam / AdamW update, in place in a single pass

   The ops take this path for float inputs with at least 4,096 elements when
   Numba is installed. `HOUSSEM_TORCH_NUMBA=0` turns it off.
2. **Threading layer set to `workqueue`.** Numba's default OpenMP layer keeps
   its threads spinning after each loop, and they fought OpenBLAS's threads.
   With the OpenMP layer, the fast path made the model **slower**: 2,139 ms
   vs 1,854 ms without it. With `workqueue` it is faster: 1,780 ms vs
   1,957 ms. A `NUMBA_THREADING_LAYER` set in the environment still takes
   precedence.
3. **NumPy reference cleanups**: the same math with fewer temporary arrays.
   - `x*x*x` instead of `x**3` (12 ms vs 2.5 ms on 512K elements)
   - in-place softmax and log_softmax
   - layer_norm centres x only once
   - cross_entropy backward no longer builds the full target matrix q
4. **Autograd allocations**:
   - Leaf gradients accumulate in place. The `.grad` setter now copies the
     array it is given, which keeps that safe.
   - Plain-slice backward (used by `split`, for example for q/k/v) assigns
     directly instead of calling the slow `np.add.at`.

Correctness: all 223 tests pass with the fast path on and off.
`tests/test_cpu_fast.py` checks every fast kernel against the NumPy
reference, and against PyTorch for cross_entropy. The fast gelu is the more
accurate of the two: its error against exact float64 values is about 2e-7,
against about 3e-6 for the reference.

## Per-op results

"Ratio" is Houssem time divided by PyTorch time, both measured in the same
run.

### 1 thread

| Case | Shape | Before HT ms | Before ratio | After HT ms | After ratio | PyTorch ms (after run) |
|---|---|---:|---:|---:|---:|---:|
| add | (1024, 512) | 3.798 | 1.88x | 2.486 | 1.37x | 1.820 |
| multiply | (1024, 512) | 6.234 | 2.47x | 5.268 | 2.46x | 2.144 |
| relu | (1024, 512) | 5.610 | 3.60x | 3.906 | 2.19x | 1.780 |
| gelu (tanh) | (1024, 512) | 50.612 | 6.91x | 26.673 | 3.33x | 8.002 |
| exp | (1024, 512) | 6.512 | 3.10x | 3.398 | 2.32x | 1.466 |
| log | (1024, 512) | 4.259 | 2.58x | 4.267 | 2.98x | 1.434 |
| sum(axis=-1) | (1024, 512) | 1.391 | 1.30x | 1.109 | 1.48x | 0.750 |
| mean(axis=-1) | (1024, 512) | 1.480 | 0.93x | 1.259 | 1.00x | 1.256 |
| matmul | (1024, 512) @ (512, 2048) | 130.865 | 1.18x | 116.344 | 0.96x | 120.758 |
| batched q @ kT | 2 x (16, 8, 64, 64) | 9.743 | 1.34x | 10.839 | 1.38x | 7.846 |
| softmax | (16, 8, 64, 64) | 9.853 | 2.43x | 9.282 | 2.45x | 3.788 |
| causal mask + softmax | (16, 8, 64, 64) | 12.280 | 2.12x | 14.779 | 2.30x | 6.435 |
| log_softmax | (1024, 16384) | 297.116 | 2.20x | 365.701 | 2.34x | 156.605 |
| layer_norm | (1024, 512) | 19.018 | 3.41x | 8.998 | 3.46x | 2.599 |
| embedding | ids (16, 64), table (16384, 512) | 46.391 | 3.90x | 25.902 | 2.45x | 10.555 |
| cross_entropy (ls=0.1) | (1024, 16384) | 316.839 | 1.75x | 348.958 | 0.90x | 387.708 |
| Linear | (16, 64, 512) -> 2048 | 156.370 | 1.04x | 325.334 | 1.46x | 222.673 |
| FeedForward | (16, 64, 512), d_ff 2048 | 330.220 | 1.15x | 272.947 | 1.19x | 230.314 |
| self-attention (causal) | (16, 64, 512), 8 heads | 256.610 | 1.38x | 184.267 | 1.31x | 140.614 |
| Transformer loss | base config (52.5M params), batch 8x32 | 2976.840 | 1.35x | 2553.223 | 1.40x | 1819.479 |
| Adam step | 52.5M params | 1322.330 | 2.77x | 259.497 | 0.43x | 607.194 |

### 6 threads

| Case | Shape | Before HT ms | Before ratio | After HT ms | After ratio | PyTorch ms (after run) |
|---|---|---:|---:|---:|---:|---:|
| add | (1024, 512) | 3.315 | 2.19x | 3.102 | 1.57x | 1.979 |
| multiply | (1024, 512) | 8.373 | 0.39x | 11.349 | 3.33x | 3.412 |
| relu | (1024, 512) | 3.939 | 3.15x | 3.582 | 2.61x | 1.373 |
| gelu (tanh) | (1024, 512) | 41.458 | 16.53x | 9.656 | 2.91x | 3.324 |
| exp | (1024, 512) | 3.093 | 2.08x | 5.584 | 2.57x | 2.171 |
| log | (1024, 512) | 4.444 | 2.37x | 11.729 | 6.68x | 1.756 |
| sum(axis=-1) | (1024, 512) | 1.146 | 1.51x | 2.105 | 2.34x | 0.901 |
| mean(axis=-1) | (1024, 512) | 1.287 | 1.33x | 1.538 | 1.75x | 0.878 |
| matmul | (1024, 512) @ (512, 2048) | 50.402 | 1.10x | 60.538 | 1.10x | 54.949 |
| batched q @ kT | 2 x (16, 8, 64, 64) | 9.816 | 1.66x | 11.608 | 1.81x | 6.403 |
| softmax | (16, 8, 64, 64) | 9.039 | 4.94x | 5.886 | 2.04x | 2.880 |
| causal mask + softmax | (16, 8, 64, 64) | 10.457 | 4.01x | 14.054 | 3.64x | 3.856 |
| log_softmax | (1024, 16384) | 301.581 | 5.30x | 156.434 | 2.21x | 70.642 |
| layer_norm | (1024, 512) | 15.731 | 10.07x | 5.895 | 3.44x | 1.715 |
| embedding | ids (16, 64), table (16384, 512) | 29.905 | 4.83x | 23.610 | 3.48x | 6.786 |
| cross_entropy (ls=0.1) | (1024, 16384) | 317.100 | 3.96x | 116.855 | 1.27x | 91.906 |
| Linear | (16, 64, 512) -> 2048 | 40.791 | 0.84x | 60.331 | 1.18x | 51.083 |
| FeedForward | (16, 64, 512), d_ff 2048 | 99.163 | 1.13x | 132.939 | 1.10x | 120.607 |
| self-attention (causal) | (16, 64, 512), 8 heads | 119.345 | 2.34x | 142.450 | 2.36x | 60.450 |
| Transformer loss | base config (52.5M params), batch 8x32 | 1325.100 | 1.62x | 3523.687 | 2.84x | 1242.212 |
| Adam step | 52.5M params | 1193.286 | 1.95x | 199.085 | 0.33x | 598.004 |

## What the numbers say

- **Ops that are chains of several steps gained most** once fused into a
  single pass: gelu, layer_norm, softmax and cross_entropy. The win grows
  with thread count, because NumPy runs everything except matmul on one
  core, while the fused kernels use all of them.
- **Adam was a hidden cost.** Before the change, one step allocated about six
  temporary arrays the size of the model, and took about as long as the
  whole forward and backward pass. It is now a single in-place pass and
  beats PyTorch.
- **Matmul is now the limit.** It takes about 70% of a model step and is
  already level with PyTorch (both call an optimized BLAS library), so on
  CPU there is little left to gain there.
- **Remaining gaps:**
  - Single-thread softmax and log_softmax stay at about 2.3x, because a
    one-thread Numba loop calls a scalar `exp` while NumPy uses a SIMD one.
  - embedding backward (about 2.5 to 3.5x) is limited by zero-filling the
    16,384 x 512 gradient table.
  - Simple pointwise ops (about 1.5 to 2.5x) pay for allocating a fresh
    array on every call.

## Next steps

1. Bring the GPU fast engine up on the 2x T4. On CPU, the remaining time is
   mostly BLAS matmul, which no Python-level change will speed up.
2. If more CPU speed is ever needed:
   - SIMD-friendly exp in the Numba row kernels (`fastmath` or SVML), to
     close the single-thread softmax gap
   - fusing attention end to end (scores, mask, softmax, @V) so the
     (B, h, T, T) intermediates are never stored
   - reusing buffers between steps, to remove allocation cost in pointwise
     ops

## Reproducing

```powershell
# full benchmark: 1 and 6 threads, writes cpu_vs_torch_report.md
& .\.venv\Scripts\python.exe .\benchmarks\cpu_vs_torch.py

# reference NumPy path only (fast kernels off)
$env:HOUSSEM_TORCH_NUMBA = "0"; & .\.venv\Scripts\python.exe .\benchmarks\cpu_vs_torch.py
```

Run on a cool, idle machine plugged into power. On a laptop, compare ratios
taken within one run instead of milliseconds taken across runs.
