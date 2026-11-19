# C++ fusion benchmark

This folder is isolated from the Python fusion implementation. It contains a
standalone C++ fused forward/backward kernel and a Python `ctypes` harness that
compares it with Houssem Torch eager, Houssem Torch's Numba fusion, and PyTorch
eager. The experiment includes a portable scalar baseline, an AVX2 variant
with runtime feature detection, and OpenMP variants with configurable thread
counts.

The C++ kernel implements:

```text
log(exp(relu((a * b + 0.25) / 2.0)) + 1.0)
```

with gradients for the sum of all outputs. Every C++ implementation computes
the operation in one pass into preallocated output/gradient buffers. AVX2
uses an exp range-reduction polynomial and a log1p series; its results are
checked numerically against the scalar, Houssem Torch, and PyTorch results.
The framework timings include operation-tensor and autograd-graph work, while
C++ timings do not. The framework combined timing also includes reduction to
a scalar loss; the C++ kernel assumes an all-ones upstream gradient. This is a
kernel-level comparison rather than a like-for-like framework comparison.

## Requirements and running

- Windows with a MinGW-w64 `g++` compiler on `PATH`
- The repository `.venv` with NumPy, PyTorch, and the optional Numba extra

From the repository root, run:

```powershell
& .\.venv\Scripts\python.exe .\fusion_Cpp\benchmark.py
```

The script compiles `fused_kernel.cpp` with `-O3 -fopenmp` to
`fusion_Cpp\build\fusion_kernel.dll`, validates C++ outputs and gradients
against Houssem Torch and PyTorch, then reports median forward, backward, and
combined times for 64 through 65,536 elements. It measures the scalar and
AVX2 kernels, plus OpenMP at 1, 2, 4, and 6 threads. OpenMP uses static
contiguous work sharing and avoids thread launch overhead below 8,192
elements. Compilation is excluded from the timed measurements.

Set `BENCH_WARMUP`, `BENCH_REPEATS`, and `BENCH_THREADS` to change the
measurement settings. The default is 2 warmups, 7 repetitions, and one thread
for the framework baselines. C++ combined working-set bytes and Houssem's
Python-traced peak are also reported; Python tracing may not include all
native allocations. See [benchmark_report.md](benchmark_report.md) for a
recorded comparison and its caveats.
