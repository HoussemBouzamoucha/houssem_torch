# C++ fusion benchmark

This folder is isolated from the Python fusion implementation. It contains a
standalone C++ fused forward/backward kernel and a Python `ctypes` harness that
compares it with Houssem Torch eager, Houssem Torch's Numba fusion, and PyTorch
eager.

The C++ kernel implements:

```text
log(exp(relu((a * b + 0.25) / 2.0)) + 1.0)
```

with gradients for the sum of all outputs. It uses preallocated output and
gradient buffers; its timings exclude Tensor creation and autograd graph
construction. The framework timings include those costs, so this is a
kernel-level comparison rather than a like-for-like framework comparison.

## Requirements and running

- Windows with a MinGW-w64 `g++` compiler on `PATH`
- The repository `.venv` with NumPy, PyTorch, and the optional Numba extra

From the repository root, run:

```powershell
& .\.venv\Scripts\python.exe .\fusion_Cpp\benchmark.py
```

The script compiles `fused_kernel.cpp` with `-O3 -march=native` to
`fusion_Cpp\build\fusion_kernel.dll`, validates C++ outputs and gradients
against Houssem Torch and PyTorch, then reports median forward, backward, and
combined times for 64 through 65,536 elements. Compilation is excluded from
the timed measurements.

Set `BENCH_WARMUP`, `BENCH_REPEATS`, and `BENCH_THREADS` to change the
measurement settings. The default is 2 warmups, 7 repetitions, and one thread.
See [benchmark_report.md](benchmark_report.md) for a recorded comparison.
