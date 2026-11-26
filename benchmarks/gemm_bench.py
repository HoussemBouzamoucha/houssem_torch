"""The training step's matrix multiplies: cuBLAS default vs tuned vs PyTorch.

Profiles one `TrainStep` twice, first with cuBLAS's default algorithm for
every multiply and then with startup tuning on (`tune_gemms=True`, the
default), and groups the GPU time by multiply shape. When PyTorch is
installed it also times every shape with torch.matmul (fp16 output, as under
autocast), so each row shows where our multiplies stand against PyTorch's.

Run from the repository root (needs CuPy and a CUDA GPU; PyTorch optional):
    python benchmarks/gemm_bench.py
    python benchmarks/gemm_bench.py --batch 64 --device 1
Writes gemm_bench_report.md and gemm_bench_results.json.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import houssem_torch as ht  # noqa: E402
from houssem_torch.backends.cuda import driver as drv  # noqa: E402
from houssem_torch.backends.cuda.train_step import TrainStep  # noqa: E402
from houssem_torch.models import Transformer, TransformerConfig  # noqa: E402
from trainstep_profile import category, make_batch, table  # noqa: E402

_LABEL = re.compile(r"(\d+)x(\d+)x(\d+)(?: x(\d+))?( A\^T)?( B\^T)?$")


def build(cfg, args, tune):
    ht.manual_seed(0)
    t0 = time.perf_counter()
    step = TrainStep(Transformer(cfg), args.batch, args.src_len, args.tgt_len,
                     max_grad_norm=1.0, device=args.device, tune_gemms=tune)
    drv.synchronize()
    return step, time.perf_counter() - t0


def per_shape(step, batch, args):
    """{(shape, kind): [µs per step, calls per step, flops per step]}."""
    for _ in range(args.warmup):
        step(*batch, lr=1e-4)
    with drv.Profiler() as prof:
        for _ in range(args.steps):
            step(*batch, lr=1e-4)
    out = defaultdict(lambda: [0.0, 0, 0.0])
    for r in prof.records():
        if r.name.startswith("gemm"):
            s = out[(r.shape, category(r))]
            s[0] += r.ms * 1e3 / args.steps
            s[1] += 1 / args.steps
            s[2] += r.flops / args.steps
    return out


def torch_us(shape, device, reps=20):
    """µs of one torch.matmul of this profiler shape label, fp16 in and out."""
    import torch
    m, n, k, batch, ta, tb = _LABEL.match(shape).groups()
    m, n, k = int(m), int(n), int(k)
    lead = (int(batch),) if batch else ()
    dev = torch.device(f"cuda:{device}")
    a = torch.randn(*lead, *((k, m) if ta else (m, k)), device=dev,
                    dtype=torch.float16)
    b = torch.randn(*lead, *((n, k) if tb else (k, n)), device=dev,
                    dtype=torch.float16)
    a, b = (a.mT if ta else a), (b.mT if tb else b)
    for _ in range(3):
        torch.matmul(a, b)
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(reps):
        torch.matmul(a, b)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / reps


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/transformer_base.json")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--src-len", type=int, default=64)
    ap.add_argument("--tgt-len", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="gemm_bench_report.md")
    args = ap.parse_args()
    if not drv.is_available():
        sys.exit("needs CuPy and a CUDA GPU")
    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False

    cfg = TransformerConfig.from_json(args.config)
    batch = make_batch(cfg, args.batch, args.src_len, args.tgt_len, 0.5)
    import cupy

    drv.reset_gemm_algorithms()
    step, setup_default = build(cfg, args, tune=False)
    default = per_shape(step, batch, args)
    del step
    cupy.get_default_memory_pool().free_all_blocks()

    step, setup_tuned = build(cfg, args, tune=True)
    tuned = per_shape(step, batch, args)
    del step
    cupy.get_default_memory_pool().free_all_blocks()
    algos = drv.gemm_algorithms()

    rows, data = [], []
    tot = defaultdict(float)
    for key in sorted(default, key=lambda k: -default[k][0]):
        shape, kind = key
        d_us, calls, flops = default[key]
        t_us = tuned.get(key, [float("nan")])[0]
        th_us = torch_us(shape, args.device) * calls if has_torch else float("nan")
        chosen = sorted({f"{v['algo']}" for lab, v in algos.items()
                         if lab.split(" float")[0] == shape})
        tot["default"] += d_us
        tot["tuned"] += t_us
        tot["torch"] += th_us
        rows.append([shape, kind, f"{calls:.0f}", f"{d_us:.0f}", f"{t_us:.0f}",
                     f"{th_us:.0f}", f"{flops / t_us / 1e6:.1f}",
                     ",".join(chosen) or "-"])
        data.append(dict(shape=shape, kind=kind, calls=calls, default_us=d_us,
                         tuned_us=t_us, torch_us=th_us, algos=chosen))
    rows.append(["**total**", "", "", f"{tot['default']:.0f}",
                 f"{tot['tuned']:.0f}", f"{tot['torch']:.0f}", "", ""])

    info = drv.device_info(args.device)
    lines = [f"# Training-step matrix multiplies on {info.name}", "",
             f"Batch {args.batch} x {args.src_len}/{args.tgt_len}, fp16, "
             f"`{args.config}`. µs per training step (all calls of the shape). "
             "Algorithm 99 = cuBLAS default; 100-115 = ALGO0..15_TENSOR_OP. "
             "PyTorch column: torch.matmul, fp16 output"
             + ("" if has_torch else " (PyTorch not installed)") + ".",
             f"Setup: {setup_default:.1f} s untuned, {setup_tuned:.1f} s tuned.",
             ""]
    lines += table(["shape (m x n x k)", "kind", "calls", "default µs",
                    "tuned µs", "torch µs", "tuned TFLOP/s", "algo"], rows)
    report = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.splitext(args.out)[0].replace("_report", "")
              + "_results.json", "w", encoding="utf-8") as f:
        json.dump(dict(gpu=info.name, batch=args.batch, src_len=args.src_len,
                       tgt_len=args.tgt_len, setup_s=[setup_default, setup_tuned],
                       rows=data, algorithms=algos), f, indent=2, default=float)
    print(report)


if __name__ == "__main__":
    main()
