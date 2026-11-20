# Houssem Torch

See [CPU pointwise kernel fusion](docs/kernel_fusion.md) for the opt-in CPU
fusion mode, its autograd behavior, limitations, and benchmark instructions.

See [Modules, optimizers and the Transformer model](docs/modules.md) for
building and training the model, and [Autograd](docs/autograd.md) for how
gradients work and how to write a new op with a hand-written backward.

With Numba installed (`pip install houssem_torch[fusion]`), the heavy
transformer ops (gelu, softmax, log_softmax, layer_norm, cross_entropy,
embedding backward and the Adam update) run as fused multi-core kernels from
`houssem_torch/backends/cpu/fast.py` on large inputs. The NumPy code stays the
reference; set `HOUSSEM_TORCH_NUMBA=0` to use it everywhere. See
[benchmarks/cpu_vs_torch_report.md](benchmarks/cpu_vs_torch_report.md) for
timings against PyTorch.

Internal design notes and planning documents live in [docs/dev/](docs/dev/).
