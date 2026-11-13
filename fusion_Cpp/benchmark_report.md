# Standalone C++ Fusion Benchmark

**Run:** 2026-11-13  
**CPU:** Intel Core i7-10750H @ 2.60 GHz  
**Compiler:** MinGW-w64 g++ 15.2.0, `-std=c++17 -O3 -march=native`  
**Runtime:** Windows, Python 3.13.11, one thread  
**Libraries:** NumPy 2.5.4, Numba 0.68.1, PyTorch 2.15.0+cpu  
**Measurement:** 5 warmups, 31 repetitions, medians

The script validated C++ forward outputs and both input gradients against
Houssem Torch eager and PyTorch eager on 257 random `float32` values before
timing.

## Workload

The standalone C++ kernel evaluates:

```text
y = log(exp(relu((a * b + 0.25) / 2.0)) + 1.0)
```

Its backward kernel calculates gradients of `sum(y)` with respect to both
inputs. The C++ combined kernel computes forward values and gradients in one
loop. The Houssem and PyTorch measurements include their operation graphs and
autograd. C++ uses preallocated buffers and a single `ctypes` call; it does not
construct framework tensors or an autograd graph. Therefore the C++ timings
are kernel-level results, not an end-to-end, like-for-like framework
comparison. Compilation is excluded.

Times are milliseconds in `forward / backward / combined` order. The last
column is Houssem eager combined time divided by the C++ combined time.

| Elements | Houssem eager | Houssem Numba fusion | PyTorch eager | C++ kernel | HT eager / C++ |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.0739 / 0.2214 / 0.4822 | 0.3575 / 0.4121 / 1.0093 | 0.1229 / 0.1670 / 0.2602 | 0.0029 / 0.0030 / 0.0035 | 137.77x |
| 256 | 0.0738 / 0.2030 / 0.4095 | 0.4817 / 0.3670 / 1.0971 | 0.1547 / 0.3106 / 0.4704 | 0.0104 / 0.0081 / 0.0119 | 34.41x |
| 1,024 | 0.0920 / 0.2746 / 0.3682 | 0.5162 / 0.6497 / 1.3818 | 0.1708 / 0.2722 / 0.4717 | 0.0217 / 0.0145 / 0.0255 | 14.44x |
| 4,096 | 0.1569 / 0.2841 / 0.4159 | 0.5412 / 0.5747 / 1.5720 | 0.1346 / 0.3275 / 0.4037 | 0.0840 / 0.0859 / 0.1061 | 3.92x |
| 16,384 | 0.3100 / 0.4155 / 0.7386 | 0.6759 / 1.1879 / 4.4134 | 0.2247 / 0.4095 / 0.5386 | 0.4440 / 0.1977 / 0.3801 | 1.94x |
| 65,536 | 1.0558 / 1.5504 / 3.3066 | 2.8209 / 3.6653 / 7.8797 | 0.5303 / 1.2996 / 1.0638 | 1.2738 / 0.6608 / 1.2965 | 2.55x |

`torch.compile` was not included. The current Windows environment reports
that PyTorch Inductor cannot find the MSVC `cl` compiler.

## Interpretation and caveats

- The specialized C++ loop was faster than Houssem eager for the **combined**
  workload at all tested sizes in this run. Its kernel-only combined time was
  also close to PyTorch eager at 65,536 elements (1.2965 ms versus 1.0638 ms).
- C++ forward time was not always faster than Houssem eager, and individual
  measurements show some system noise. Re-run on an otherwise idle machine
  before drawing conclusions about small differences.
- The large C++/framework ratios partly reflect what is timed: C++ has no
  tensor construction, autograd graph, or output-buffer allocation. They
  should not be presented as an end-to-end framework speedup.
- This is a fixed-shape, contiguous `float32` experiment. It does not implement
  general broadcasting, arbitrary expressions, a reusable autograd graph, or
  a Python tensor backend.
- The C++ kernel's backward formula is manually specialized for this workload;
  it does not replace or modify Houssem Torch's autograd implementation.

Run [`benchmark.py`](./benchmark.py) to rebuild the DLL, repeat the numerical
checks, and collect new medians. The generated library is kept under
`fusion_Cpp/build/` and ignored by Git.
