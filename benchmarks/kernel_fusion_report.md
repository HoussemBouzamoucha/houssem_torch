# CPU Kernel Fusion Benchmark Report

**Run date:** 2026-11-13  
**Platform:** Windows, Python 3.13.11, one CPU thread  
**Libraries:** NumPy 2.5.4, Numba 0.68.1, PyTorch 2.15.0+cpu  
**Benchmark:** [`kernel_fusion.py`](./kernel_fusion.py)  
**Settings:** 2 warmups and 7 timed repetitions; medians reported

## Workload and method

The benchmark runs a pointwise chain with two input tensors:

```python
log(exp(relu((a * b + 0.25) / 2.0)) + 1.0)
```

It measures forward execution with autograd recording enabled, backward after
the forward graph is prepared, and combined forward plus backward. Houssem
Torch eager and fused modes, PyTorch eager, and `torch.compile` are attempted
for sizes from 64 to 65,536 elements.

Speedup is `Houssem eager combined time / Houssem fused combined time`; values
below `1.00x` mean fusion was slower. Peak memory is the `tracemalloc` peak
during forward graph construction and materialization. It reflects Python and
NumPy allocations, not PyTorch's native CPU allocations.

## Results

Times are milliseconds in `forward / backward / combined` order.

| Elements | Houssem eager | Houssem fused | PyTorch eager | Fused speedup | Peak traced MiB, eager → fused |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.0716 / 0.2432 / 0.4131 | 1.0306 / 0.5856 / 1.8326 | 0.6345 combined | 0.23x | 0.008 → 0.013 |
| 256 | 0.0970 / 0.2478 / 0.3579 | 0.3739 / 0.3531 / 0.7876 | 0.2523 combined | 0.45x | 0.013 → 0.015 |
| 1,024 | 0.0962 / 0.2753 / 0.4026 | 0.5058 / 0.5643 / 1.2901 | 0.2800 combined | 0.31x | 0.034 → 0.023 |
| 4,096 | 0.2001 / 0.4578 / 0.7205 | 0.4758 / 0.8433 / 1.9739 | 0.2932 combined | 0.37x | 0.116 → 0.058 |
| 16,384 | 0.1552 / 0.3316 / 0.5711 | 0.6697 / 0.6589 / 1.4399 | 0.4793 combined | 0.40x | 0.444 → 0.199 |
| 65,536 | 0.4632 / 1.0628 / 1.3794 | 1.4675 / 1.1710 / 3.5189 | 0.6252 combined | 0.39x | 1.756 → 0.761 |

`torch.compile` could not be benchmarked: PyTorch Inductor reported
`InvalidCxxCompiler` because the MSVC `cl` compiler is unavailable.

## Interpretation

This run does **not** show a speed improvement. Fused combined execution was
slower at every tested size, reaching 0.23–0.45x of eager performance. Forward
graph memory was lower for sizes 1,024 and above, but slightly higher for 64
and 256 elements. PyTorch memory is not comparable in this report because its
native allocations are not captured by `tracemalloc`.

The results support fusion as an opt-in memory optimization for larger
autograd-tracked pointwise chains, not as a general speed optimization.
Re-benchmark after changes to input snapshotting, fused backward, or CPU SIMD,
and on a system where `torch.compile` is supported.
