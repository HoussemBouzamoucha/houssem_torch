"""Where does houssem_torch's GPU training step spend its time?

Times every launch of one `TrainStep` (cuBLAS calls and our kernels) with
CUDA events and groups them three ways:
  * by kind of work (matmuls, residual + LayerNorm, softmax, optimizer, ...),
    comparable with the torch.compile breakdown in research.md §7.2
  * by phase and block (forward / backward × embedding, self-attention,
    cross-attention, feed-forward, residual + LayerNorm, output head)
  * matrix multiplies by shape, with achieved TFLOP/s
plus the slowest non-matmul kernels. Optionally times several batch sizes.

Run from the repository root (needs CuPy and a CUDA GPU, not PyTorch):
    python benchmarks/trainstep_profile.py
    python benchmarks/trainstep_profile.py --batch-sizes 32 64 96
Writes trainstep_profile_report.md and trainstep_profile_results.json.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import houssem_torch as ht  # noqa: E402
from houssem_torch.backends.cuda import driver as drv  # noqa: E402
from houssem_torch.backends.cuda.train_step import TrainStep  # noqa: E402
from houssem_torch.models import Transformer, TransformerConfig  # noqa: E402

KERNEL_CATEGORY = {
    "add_ln_fwd": "residual + dropout + LayerNorm",
    "ln_bwd": "residual + dropout + LayerNorm",
    "softmax_fwd": "attention softmax + dropout",
    "softmax_bwd": "attention softmax + dropout",
    "split_heads": "head split / merge",
    "merge_heads": "head split / merge",
    "bias_act_fwd": "FFN bias + activation + dropout",
    "act_bwd": "FFN bias + activation + dropout",
    "cross_entropy": "loss (cross-entropy fwd+bwd)",
    "embed_bwd": "embedding backward",
    "adamw": "optimizer (norm, scaling, AdamW)",
    "sum_squares": "optimizer (norm, scaling, AdamW)",
    "step_prepare": "optimizer (norm, scaling, AdamW)",
    "reset_scalars": "optimizer (norm, scaling, AdamW)",
    "fill": "memset (zero gradients)",
}

# T4 tensor-core fp16 peak (dense), for the efficiency column.
T4_FP16_TFLOPS = 65.0


def category(rec) -> str:
    if rec.name == "gemm_batched":
        return "matmul: attention (QK^T, PV)"
    if rec.name == "gemm":
        if rec.shape.startswith("1x"):
            return "matmul: bias gradients (ones x g)"
        if "/head" in rec.region:
            return "matmul: output projection (logits)"
        return "matmul: linear layers"
    return KERNEL_CATEGORY.get(rec.name, rec.name)


def block(rec) -> tuple[str, str]:
    """(phase, block) from a region like 'bwd/dec/cross'."""
    parts = rec.region.split("/")
    phase = parts[0] if parts and parts[0] else "other"
    blk = parts[-1] if len(parts) > 1 else phase
    return phase, blk


def make_batch(cfg, B, S, T, min_fill, seed=0):
    """Fixed-shape batch with random sentence lengths (realistic padding)."""
    rng = np.random.default_rng(seed)
    V, pad, bos, eos = cfg.vocab_size, cfg.pad_id, cfg.bos_id, cfg.eos_id

    def lengths(L):
        return rng.integers(max(1, int(L * min_fill)), L + 1, size=B)

    src = rng.integers(3, V, (B, S))
    src[np.arange(S)[None, :] >= lengths(S)[:, None]] = pad
    t_len = lengths(T)
    body = rng.integers(3, V, (B, T))
    tgt_out = body.copy()
    tgt_out[np.arange(B), np.minimum(t_len - 1, T - 1)] = eos
    tgt_out[np.arange(T)[None, :] >= t_len[:, None]] = pad
    tgt_in = np.concatenate([np.full((B, 1), bos), body[:, :-1]], axis=1)
    tgt_in[np.arange(T)[None, :] >= t_len[:, None]] = pad
    return src, tgt_in, tgt_out


def build(cfg, B, args):
    ht.manual_seed(0)
    model = Transformer(cfg)
    t0 = time.perf_counter()
    step = TrainStep(model, B, args.src_len, args.tgt_len,
                     precision=args.precision, max_grad_norm=1.0,
                     device=args.device)
    drv.synchronize()
    return step, time.perf_counter() - t0


def time_step(step, batch, args):
    for _ in range(args.warmup):
        step(*batch, lr=1e-4)
    drv.synchronize()
    runs = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        for _ in range(args.steps):
            step(*batch, lr=1e-4)
        drv.synchronize()
        runs.append((time.perf_counter() - t0) / args.steps * 1e3)
    return statistics.median(runs)


def profile(step, batch, n):
    with drv.Profiler() as prof:
        for _ in range(n):
            step(*batch, lr=1e-4)
    return prof.records()


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header)))
           + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/transformer_base.json")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[64])
    ap.add_argument("--src-len", type=int, default=64)
    ap.add_argument("--tgt-len", type=int, default=64)
    ap.add_argument("--min-fill", type=float, default=0.5)
    ap.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--profile-steps", type=int, default=5)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="trainstep_profile_report.md")
    args = ap.parse_args()
    if not drv.is_available():
        sys.exit("needs CuPy and a CUDA GPU")

    cfg = TransformerConfig.from_json(args.config)
    info = drv.device_info(args.device)
    sweep = []
    detail = None
    for i, B in enumerate(args.batch_sizes):
        step, setup = build(cfg, B, args)
        batch = make_batch(cfg, B, args.src_len, args.tgt_len, args.min_fill)
        tokens = int((batch[2] != cfg.pad_id).sum())
        ms = time_step(step, batch, args)
        mem = drv.memory_in_use() / 2 ** 30
        sweep.append(dict(batch=B, ms=ms, tok_s=tokens / ms * 1e3,
                          setup_s=setup, mem_gb=mem))
        print(f"batch {B:4d}: {ms:7.1f} ms/step  {tokens / ms * 1e3:8.0f} tok/s  "
              f"setup {setup:.1f}s  mem {mem:.2f} GB", flush=True)
        if i == 0:
            recs = profile(step, batch, args.profile_steps)
            detail = dict(batch=B, ms=ms, records=recs)
        del step
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()

    n = args.profile_steps
    recs = detail["records"]
    busy = sum(r.ms for r in recs) / n
    launches = len(recs) / n

    by_cat = defaultdict(lambda: [0.0, 0])
    by_block = defaultdict(float)
    by_shape = defaultdict(lambda: [0.0, 0, 0.0, ""])
    by_kernel = defaultdict(lambda: [0.0, 0])
    for r in recs:
        c = by_cat[category(r)]
        c[0] += r.ms / n
        c[1] += 1
        by_block[block(r)] += r.ms / n
        if r.name.startswith("gemm"):
            s = by_shape[(r.shape, category(r))]
            s[0] += r.ms / n
            s[1] += 1
            s[2] += r.flops / n
        else:
            k = by_kernel[r.name]
            k[0] += r.ms / n
            k[1] += 1

    mm_ms = sum(v[0] for k, v in by_cat.items() if k.startswith("matmul"))
    mm_flops = sum(v[2] for v in by_shape.values())
    lines = [
        "# houssem_torch TrainStep profile",
        "",
        f"GPU: {info.name} ({info.multiprocessors} SMs). Config: {args.config}, "
        f"{args.precision}. Batch {detail['batch']} x {args.src_len}/"
        f"{args.tgt_len} tokens, min fill {args.min_fill}.",
        "",
        f"- Step time (unprofiled, median of {args.repeats} x {args.steps}): "
        f"**{detail['ms']:.1f} ms**",
        f"- Sum of launch times (profiled, per step): {busy:.1f} ms "
        f"({busy / detail['ms']:.0%} of the step; event timing adds a little)",
        f"- Launches per step: {launches:.0f}",
        f"- Matmuls: {mm_ms:.1f} ms, {mm_flops / 1e12:.2f} TFLOP/step, "
        f"{mm_flops / (mm_ms / 1e3) / 1e12:.1f} TFLOP/s "
        f"({mm_flops / (mm_ms / 1e3) / 1e12 / T4_FP16_TFLOPS:.0%} of the T4's "
        f"{T4_FP16_TFLOPS:.0f} TFLOP/s fp16 peak)",
        "",
        "## By kind of work",
        "",
    ]
    rows = sorted(by_cat.items(), key=lambda kv: -kv[1][0])
    lines += table(["work", "ms/step", "share", "launches/step"],
                   [(k, f"{v[0]:.2f}", f"{v[0] / busy:.1%}", f"{v[1] / n:.0f}")
                    for k, v in rows])

    lines += ["", "## By phase and block (ms/step)", ""]
    blocks = ["embed", "self", "cross", "ffn", "residual_ln", "head"]
    phases = ["fwd", "bwd"]
    brow = []
    for b in blocks:
        vals = [by_block.get((p, b), 0.0) for p in phases]
        brow.append((b, *(f"{v:.2f}" for v in vals), f"{sum(vals):.2f}"))
    opt = sum(v for (p, _), v in by_block.items() if p == "optimizer")
    brow.append(("optimizer", "", "", f"{opt:.2f}"))
    lines += table(["block", "forward", "backward", "total"], brow)

    lines += ["", "## Matrix multiplies by shape (m x n x k, row-major)", ""]
    rows = sorted(by_shape.items(), key=lambda kv: -kv[1][0])
    lines += table(["shape", "kind", "calls/step", "ms/step", "TFLOP/s"],
                   [(shape, cat.replace("matmul: ", ""), f"{v[1] / n:.0f}",
                     f"{v[0]:.2f}", f"{v[2] / (v[0] / 1e3) / 1e12:.1f}")
                    for (shape, cat), v in rows])

    lines += ["", "## Our kernels", ""]
    rows = sorted(by_kernel.items(), key=lambda kv: -kv[1][0])
    lines += table(["kernel", "ms/step", "calls/step", "us/call"],
                   [(k, f"{v[0]:.2f}", f"{v[1] / n:.0f}",
                     f"{v[0] / (v[1] / n) * 1e3:.0f}") for k, v in rows])

    if len(sweep) > 1:
        lines += ["", "## Batch sizes", ""]
        lines += table(["batch", "ms/step", "tokens/s", "setup s", "memory GB"],
                       [(s["batch"], f"{s['ms']:.1f}", f"{s['tok_s']:.0f}",
                         f"{s['setup_s']:.1f}", f"{s['mem_gb']:.2f}")
                        for s in sweep])
    report = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.splitext(args.out)[0].replace("_report", "") +
              "_results.json", "w", encoding="utf-8") as f:
        json.dump(dict(gpu=info.name, sweep=sweep, step_ms=detail["ms"],
                       busy_ms=busy,
                       categories={k: v[0] for k, v in by_cat.items()},
                       blocks={"/".join(k): v for k, v in by_block.items()}),
                  f, indent=2)
    print("\n" + report)


if __name__ == "__main__":
    main()
