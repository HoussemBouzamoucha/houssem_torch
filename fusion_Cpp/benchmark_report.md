# C++ CPU optimization benchmark

## Setup

- **CPU:** Intel Core i7-10750H @ 2.60 GHz
- **Compiler:** MinGW-w64 g++ 15.2.0, `-std=c++17 -O3 -fopenmp`
- **Runtime:** Windows, Python 3.13.11
- **Libraries:** NumPy 2.5.4, Numba 0.68.1, PyTorch 2.15.0+cpu
- **Measurements:** 5 warmups, 21 repetitions, medians; framework baselines
  use one thread
- **CPU features:** AVX2 available; OpenMP enabled

The script validates forward outputs and gradients against Houssem Torch and
PyTorch before timing. Validation includes negative and near-zero preactivations,
vector-tail lengths, and arbitrary upstream gradients.

## Combined forward + backward

Times are milliseconds. OpenMP uses static contiguous work sharing. The
parallel region is disabled below 8,192 elements; the one-thread OpenMP entry
point still has some runtime overhead at small sizes.

| Elements | Houssem eager | PyTorch eager | C++ scalar | C++ AVX2 | OpenMP 1 thread | OpenMP 2 threads | OpenMP 4 threads | OpenMP 6 threads |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.3824 | 0.2090 | 0.0031 | 0.0023 | 0.0072 | 0.0076 | 0.0071 | 0.0077 |
| 256 | 0.3160 | 0.3207 | 0.0089 | 0.0039 | 0.0212 | 0.0403 | 0.0214 | 0.0217 |
| 1,024 | 0.6272 | 0.2827 | 0.0226 | 0.0098 | 0.0274 | 0.0267 | 0.0266 | 0.0288 |
| 4,096 | 0.3879 | 0.2929 | 0.0882 | 0.0210 | 0.0953 | 0.0914 | 0.0883 | 0.0884 |
| 16,384 | 0.5782 | 0.3235 | 0.4620 | 0.0779 | 0.4095 | 0.2541 | 0.1669 | 0.1706 |
| 65,536 | 1.1636 | 1.2670 | 1.6765 | 0.2684 | 1.3502 | 0.6904 | 0.4178 | 0.3106 |

This is not an end-to-end comparison: C++ uses preallocated buffers and a
single native call, while Houssem Torch and PyTorch include operation-tensor
and autograd-graph work. The framework combined timing also includes reduction
to a scalar loss; the C++ kernel assumes an all-ones upstream gradient. In
this run the AVX2 kernel-only combined time was 0.2684 ms at 65,536 elements;
six-thread OpenMP was 0.3106 ms. These are measurements of this fixed
workload, not general framework speedups.

At 65,536 elements, the forward/backward medians were:

| Variant | Forward (ms) | Backward (ms) | Combined (ms) |
|---|---:|---:|---:|
| C++ scalar | 1.3627 | 0.8046 | 1.6765 |
| C++ AVX2 | 0.2295 | 0.1789 | 0.2684 |
| C++ OpenMP, 2 threads | 0.5456 | 0.3658 | 0.6904 |
| C++ OpenMP, 4 threads | 0.3705 | 0.2652 | 0.4178 |
| C++ OpenMP, 6 threads | 0.3092 | 0.1912 | 0.3106 |

OpenMP helps at the larger sizes, but thread startup/scheduling dominates
small workloads. The AVX2 variant is single-threaded and computes exponentials
and logarithms using approximations (range-reduced exp polynomial and log1p
series), rather than calling scalar `std::exp`/`std::log` for each element.
Its measured speed should therefore be considered together with the numerical
validation and the approximation limitation.

## Memory and allocation behavior

The combined C++ kernel allocates no intermediate tensors or scratch buffers.
Its preallocated working set is five `float32` arrays (`a`, `b`, output, and
two gradients), or 20 bytes per element. Houssem's Python-traced forward
peaks were:

| Elements | C++ combined buffers | Houssem Python-traced peak |
|---:|---:|---:|
| 64 | 1,280 B | 9,388 B |
| 256 | 5,120 B | 16,300 B |
| 1,024 | 20,480 B | 43,948 B |
| 4,096 | 81,920 B | 154,540 B |
| 16,384 | 327,680 B | 596,908 B |
| 65,536 | 1,310,720 B | 2,366,380 B |

These are not equivalent allocator measurements: C++ reports the known
preallocated input/output/gradient buffers, whereas Python `tracemalloc`
reports Python-tracked allocations and may omit native allocations. Neither
the C++ working-set estimate nor its timing includes a Python wrapper's
allocation of the buffers.

## Implementation and limitations

- `-O3` enables compiler optimizations. The scalar implementation remains
  portable and serves as a reference.
- AVX2 is isolated in target-specific functions and selected only when the
  runtime CPU feature check succeeds; it processes eight `float32` values per
  vector and handles the remainder with the scalar reference.
- OpenMP partitions contiguous ranges statically. A threshold prevents
  parallel-region execution below 8,192 elements.
- The experiment is fixed to contiguous `float32` inputs and this expression.
  AVX2 uses approximate transcendental functions, validated on random values
  and near the ReLU boundary; extreme-value behavior has not been exhaustively
  characterized. It is not a general fusion engine or a replacement for
  Houssem Torch autograd.
- Results vary with CPU load, power management, and runtime scheduling; the
  short benchmark should be rerun on an idle target machine.

The harness also prints Houssem Numba fusion timings. In this run it was
slower than Houssem eager at every tested size. `torch.compile` remains
excluded because this Windows environment lacks the MSVC `cl` compiler
required by the installed PyTorch Inductor setup.

Run [`benchmark.py`](./benchmark.py) to rebuild the DLL, repeat the numerical
checks, and collect new medians. The generated library is kept under
`fusion_Cpp/build/` and ignored by Git.
