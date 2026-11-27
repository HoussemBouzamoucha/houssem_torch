"""Training step: FFN backward plain vs the two cuBLASLt epilogue modes.

One `TrainStep` per batch size; the FFN-backward mode is switched on the
same step (`step.lt_ffn`), every mode once per round, so the T4's drifting
clock hits all of them alike. Reports the median ms per step, the GPU time
of the launches that differ (one profiled step per mode) and the cuBLASLt
algorithm timings.

    plain        gemm g_y·W_downᵀ, act_bwd kernel (mask, dropout, bias grad)
    drelu_bgrad  one cuBLASLt multiply: mask + fp16 bias gradient
    drelu_wgrad  cuBLASLt mask multiply; fp32 bias gradient from the
                 weight-gradient multiply (BGRADA)

Run from the repository root (needs CuPy, a CUDA GPU and cuBLASLt):
    python benchmarks/epilogue_bench.py --device 1 --batch-sizes 64 256
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import houssem_torch as ht  # noqa: E402
from houssem_torch.backends.cuda import driver as drv  # noqa: E402
from houssem_torch.backends.cuda.train_step import TrainStep  # noqa: E402
from houssem_torch.models import Transformer, TransformerConfig  # noqa: E402
from trainstep_profile import make_batch, table  # noqa: E402

MODES = (None, "drelu_bgrad", "drelu_wgrad")
DIFFER = ("act_bwd", "add_into_f32", "acc_f32", "gemm_lt")


def name(mode):
    return mode or "plain"


def time_steps(step, batch, n):
    t = drv.Timer()
    t.start()
    for _ in range(n):
        step(*batch, lr=1e-4)
    return t.stop() / n


def profiled_us(step, batch, d_ff):
    """GPU µs of the launches that differ between modes, plus the plain
    multiplies with a d_ff-wide side (FFN gemms)."""
    with drv.Profiler() as prof:
        step(*batch, lr=1e-4)
    out = defaultdict(float)
    for r in prof.records():
        if r.name in DIFFER:
            out[r.name] += r.ms * 1e3
        elif r.name == "gemm" and str(d_ff) in r.shape.split()[0].split("x"):
            out["ffn gemm"] += r.ms * 1e3
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/transformer_base.json")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--src-len", type=int, default=64)
    ap.add_argument("--tgt-len", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    if not drv.is_available():
        sys.exit("needs CuPy and a CUDA GPU")
    cfg = TransformerConfig.from_json(args.config)
    rows, prof_rows = [], []
    for B in args.batch_sizes:
        batch = make_batch(cfg, B, args.src_len, args.tgt_len, 0.5)
        ht.manual_seed(0)
        step = TrainStep(Transformer(cfg), B, args.src_len, args.tgt_len,
                         max_grad_norm=1.0, device=args.device,
                         fused_epilogues="drelu_wgrad")
        if not step.lt_ffn:
            sys.exit("the fused FFN epilogue is unavailable on this setup")
        ms = {m: [] for m in MODES}
        for mode in MODES:                      # plans + cuBLASLt tuning
            step.lt_ffn = mode
            time_steps(step, batch, 3)
        for _ in range(args.rounds):
            for mode in MODES:
                step.lt_ffn = mode
                ms[mode].append(time_steps(step, batch, args.steps))
        med = {m: statistics.median(ms[m]) for m in MODES}
        base = med[None]
        rows.append([B] + [f"{med[m]:.1f}" for m in MODES]
                    + [f"{base / med[m]:.3f}" for m in MODES[1:]])
        for mode in MODES:
            step.lt_ffn = mode
            p = profiled_us(step, batch, cfg.d_ff)
            prof_rows.append([B, name(mode), f"{sum(p.values()):.0f}",
                              ", ".join(f"{k} {v:.0f}" for k, v in sorted(p.items()))])
        print(f"batch {B}: " + ", ".join(f"{name(m)} {med[m]:.1f} ms"
                                         for m in MODES), flush=True)
        del step
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
    info = drv.device_info(args.device)
    lines = [f"# FFN-backward epilogues on {info.name}", "",
             f"`{args.config}`, {args.src_len}/{args.tgt_len} tokens, median of "
             f"{args.rounds} rounds x {args.steps} steps, modes alternating.", ""]
    lines += table(["batch"] + [f"{name(m)} ms" for m in MODES]
                   + [f"{name(m)} speedup" for m in MODES[1:]], rows)
    lines += ["", "Profiled step (µs): the launches that differ, and the FFN "
              "plain gemms.", ""]
    lines += table(["batch", "mode", "total µs", "parts"], prof_rows)
    lines += ["", "cuBLASLt candidates (µs per call, chosen = fastest):", ""]
    for label, v in drv.lt_algorithms().items():
        lines.append(f"- `{label}`: chosen #{v['chosen']}, {v['us']}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
