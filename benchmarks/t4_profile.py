"""Profile one training step of the reference Transformer on a GPU (target: T4).

Answers the question that decides what to build first in the GPU backend:
where does a PyTorch training step spend its time, and how much of it does
torch.compile already remove?

Three modes, same model, same batch, same seed:
    eager            plain PyTorch
    compile          torch.compile(...)
    reduce-overhead  torch.compile(mode="reduce-overhead")  (CUDA graphs)

For each mode it measures ms/step, tokens/s, peak memory and compile time,
then profiles a few steps and splits GPU time into: matmuls, attention,
loss, layernorm, softmax, dropout, embedding, optimizer, other, plus the
time the GPU sits idle between kernels. A separate micro-benchmark times
the output projection + cross-entropy alone (the target of a fused,
logits-free loss kernel).

Standalone: needs only PyTorch, so it can be uploaded on its own to Kaggle
or Colab. The model mirrors houssem_torch's Transformer (shared embedding
used as output projection, embedding x sqrt(d) + sinusoidal positions,
pre- or post-norm, fused QKV / KV projections, label-smoothed loss).

Usage (Kaggle notebook cell):
    !python t4_profile.py                       # defaults: transformer_base
    !python t4_profile.py --config my.json --batch-size 96 --src-len 48 --tgt-len 48
    !python t4_profile.py --attention math      # explicit attention, like houssem_torch

Writes t4_profile_report.md (paste it back) and t4_profile_results.json;
--trace-dir also keeps the Chrome traces (open in chrome://tracing or
https://ui.perfetto.dev).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import tempfile
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Same values as configs/transformer_base.json, so the script runs alone.
DEFAULT_CONFIG = {
    "vocab_size": 16384, "d_model": 512, "n_heads": 8, "d_ff": 2048,
    "n_encoder_layers": 6, "n_decoder_layers": 6, "dropout": 0.1,
    "max_len": 128, "activation": "relu", "label_smoothing": 0.1,
    "pad_id": 0, "bos_id": 1, "eos_id": 2,
    "norm_position": "pre", "attention_bias": True, "ffn_dropout": None,
    "label_smoothing_mode": "uniform", "embedding_init": "normal",
    "embedding_padding_idx": False,
}

MODES = ("eager", "compile", "reduce-overhead")


# ---------------------------------------------------------------------------
# Model (PyTorch mirror of houssem_torch.models.Transformer)
# ---------------------------------------------------------------------------

def sinusoidal_position_encoding(max_len, d):
    pos = torch.arange(max_len, dtype=torch.float64)[:, None]
    rates = 1.0 / (10000.0 ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    pe = torch.zeros(max_len, d, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(pos * rates)
    pe[:, 1::2] = torch.cos(pos * rates[: d // 2])
    return pe.float()


class Attention(nn.Module):
    def __init__(self, cfg, cross, impl):
        super().__init__()
        d, bias = cfg["d_model"], cfg["attention_bias"]
        self.h, self.dk, self.p = cfg["n_heads"], d // cfg["n_heads"], cfg["dropout"]
        self.cross, self.impl = cross, impl
        if cross:
            self.w_q = nn.Linear(d, d, bias=bias)
            self.w_kv = nn.Linear(d, 2 * d, bias=bias)
        else:
            self.w_qkv = nn.Linear(d, 3 * d, bias=bias)
        self.w_o = nn.Linear(d, d, bias=bias)

    def forward(self, x, keep, memory=None):
        """keep: bool mask broadcastable to (B, h, Tq, Tk), True = may attend."""
        B, Tq, d = x.shape
        if self.cross:
            q = self.w_q(x)
            k, v = self.w_kv(memory).chunk(2, dim=-1)
        else:
            q, k, v = self.w_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.reshape(B, -1, self.h, self.dk).transpose(1, 2) for t in (q, k, v))
        p = self.p if self.training else 0.0
        if self.impl == "sdpa":
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=keep, dropout_p=p)
        else:  # explicit, the way houssem_torch computes it today
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
            scores = scores.masked_fill(~keep, -1e4)  # fp16-safe mask value
            w = F.dropout(torch.softmax(scores, dim=-1), p)
            o = w @ v
        return self.w_o(o.transpose(1, 2).reshape(B, Tq, d))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.l1 = nn.Linear(cfg["d_model"], cfg["d_ff"])
        self.l2 = nn.Linear(cfg["d_ff"], cfg["d_model"])
        rate = cfg["ffn_dropout"]
        self.drop = nn.Dropout(cfg["dropout"] if rate is None else rate)
        self.act = F.relu if cfg["activation"] == "relu" else F.gelu

    def forward(self, x):
        return self.l2(self.drop(self.act(self.l1(x))))


class Block(nn.Module):
    """pre: x + drop(f(norm(x)));  post: norm(x + drop(f(x)))."""

    def __init__(self, cfg, sublayer):
        super().__init__()
        self.f, self.norm = sublayer, nn.LayerNorm(cfg["d_model"])
        self.drop, self.pre = nn.Dropout(cfg["dropout"]), cfg["norm_position"] == "pre"

    def forward(self, x, *args, **kw):
        if self.pre:
            return x + self.drop(self.f(self.norm(x), *args, **kw))
        return self.norm(x + self.drop(self.f(x, *args, **kw)))


class EncoderLayer(nn.Module):
    def __init__(self, cfg, impl):
        super().__init__()
        self.attn = Block(cfg, Attention(cfg, False, impl))
        self.ffn = Block(cfg, FeedForward(cfg))

    def forward(self, x, keep):
        return self.ffn(self.attn(x, keep))


class DecoderLayer(nn.Module):
    def __init__(self, cfg, impl):
        super().__init__()
        self.self_attn = Block(cfg, Attention(cfg, False, impl))
        self.cross_attn = Block(cfg, Attention(cfg, True, impl))
        self.ffn = Block(cfg, FeedForward(cfg))

    def forward(self, y, memory, self_keep, cross_keep):
        y = self.self_attn(y, self_keep)
        y = self.cross_attn(y, cross_keep, memory=memory)
        return self.ffn(y)


class Transformer(nn.Module):
    def __init__(self, cfg, impl):
        super().__init__()
        self.cfg, d = cfg, cfg["d_model"]
        pad = cfg["pad_id"] if cfg["embedding_padding_idx"] else None
        self.embedding = nn.Embedding(cfg["vocab_size"], d, padding_idx=pad)
        if cfg["embedding_init"] == "xavier":
            nn.init.xavier_uniform_(self.embedding.weight)
        else:
            nn.init.normal_(self.embedding.weight, std=d ** -0.5)
        if pad is not None:
            with torch.no_grad():
                self.embedding.weight[pad].zero_()
        self.register_buffer("pe", sinusoidal_position_encoding(cfg["max_len"], d))
        self.drop = nn.Dropout(cfg["dropout"])
        self.encoder = nn.ModuleList(EncoderLayer(cfg, impl)
                                     for _ in range(cfg["n_encoder_layers"]))
        self.decoder = nn.ModuleList(DecoderLayer(cfg, impl)
                                     for _ in range(cfg["n_decoder_layers"]))
        pre = cfg["norm_position"] == "pre"
        self.enc_norm = nn.LayerNorm(d) if pre else nn.Identity()
        self.dec_norm = nn.LayerNorm(d) if pre else nn.Identity()

    def embed(self, ids):
        x = self.embedding(ids) * math.sqrt(self.cfg["d_model"])
        return self.drop(x + self.pe[: ids.shape[1]])

    def loss(self, src, tgt_in, tgt_out):
        pad = self.cfg["pad_id"]
        src_keep = (src != pad)[:, None, None, :]                      # (B,1,1,S)
        T = tgt_in.shape[1]
        causal = torch.ones(T, T, dtype=torch.bool, device=src.device).tril()
        tgt_keep = (tgt_in != pad)[:, None, None, :] & causal          # (B,1,T,T)
        x = self.embed(src)
        for layer in self.encoder:
            x = layer(x, src_keep)
        memory = self.enc_norm(x)
        y = self.embed(tgt_in)
        for layer in self.decoder:
            y = layer(y, memory, tgt_keep, src_keep)
        h = self.dec_norm(y)
        logits = h @ self.embedding.weight.t()                        # (B,T,V)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                               tgt_out.reshape(-1), ignore_index=pad,
                               label_smoothing=self.cfg["label_smoothing"])


# ---------------------------------------------------------------------------
# Batch, step, timing
# ---------------------------------------------------------------------------

def make_batch(cfg, B, S, T, min_fill, device, seed=0):
    """Fixed-shape batch with random sentence lengths (realistic padding).
    Every sentence keeps at least one real token, so no row is fully masked."""
    g = torch.Generator().manual_seed(seed)
    V, pad, bos, eos = cfg["vocab_size"], cfg["pad_id"], cfg["bos_id"], cfg["eos_id"]

    def lengths(L):
        lo = max(1, int(L * min_fill))
        return torch.randint(lo, L + 1, (B,), generator=g)

    def fill(L, lens):
        ids = torch.randint(3, V, (B, L), generator=g)
        ids[torch.arange(L)[None, :] >= lens[:, None]] = pad
        return ids

    src = fill(S, lengths(S))
    tgt_lens = lengths(T)
    body = fill(T, tgt_lens)
    tgt_in = torch.cat([torch.full((B, 1), bos), body[:, :-1]], dim=1)
    tgt_out = body.clone()
    tgt_out[torch.arange(B), (tgt_lens - 1).clamp(max=T - 1)] = eos
    tgt_in[torch.arange(T)[None, :] >= tgt_lens[:, None]] = pad
    return src.to(device), tgt_in.to(device), tgt_out.to(device)


def build_step(cfg, args, mode, device):
    torch.manual_seed(0)
    model = Transformer(cfg, args.attention).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4,
                            fused=device.type == "cuda")
    use_amp = args.precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if mode == "eager":
        loss_fn = model.loss
    elif mode == "compile":
        loss_fn = torch.compile(model.loss)
    else:
        loss_fn = torch.compile(model.loss, mode="reduce-overhead")
    mark = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
    batch = make_batch(cfg, args.batch_size, args.src_len, args.tgt_len,
                       args.min_fill, device)

    def step():
        if mode == "reduce-overhead" and mark is not None:
            mark()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            loss = loss_fn(*batch)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return loss

    return step, batch


def short_error(exc):
    """One line, table-safe."""
    return " ".join(f"{type(exc).__name__}: {exc}".split())[:300].replace("|", "/")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_steps(step, n, repeats, device):
    """Median wall-clock ms per step over `repeats` runs of n steps (CPU
    overhead included: that's what training pays)."""
    runs = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        for _ in range(n):
            step()
        sync(device)
        runs.append((time.perf_counter() - t0) / n * 1e3)
    return statistics.median(runs)


# ---------------------------------------------------------------------------
# Profiling: classify GPU kernels, measure idle gaps
# ---------------------------------------------------------------------------

# First match wins. Kernel names are cuBLAS/CUTLASS/ATen names in eager and
# triton_{poi,red,per,tem}_fused_<ops> names under torch.compile.
CATEGORIES = [
    ("attention", r"fmha|flash|efficient_attention|attention"),
    ("loss (log_softmax / NLL)", r"log_softmax|nll_loss|cross_entropy"),
    ("optimizer + grad scaling", r"adam|multi_tensor|foreach|unscale|non_finite"),
    ("matmul", r"gemm|xmma|cutlass|cublas|s1688|h1688|s884|h884|sm75|turing_|"
               r"volta_|ampere_|triton_tem|_mm_|bmm|addmm"),
    ("layernorm", r"layer_norm|layernorm"),
    ("softmax", r"softmax"),
    ("dropout / RNG", r"dropout|philox|bernoulli|native_dropout|\brand|_rand"),
    ("embedding", r"embedding|index_select|indexing_backward|index_put|gather|scatter"),
]
MEM_CATS = ("gpu_memcpy", "gpu_memset")


def classify(name, cat):
    if cat in MEM_CATS:
        return "memcpy / memset"
    low = name.lower()
    for label, pattern in CATEGORIES:
        if re.search(pattern, low):
            return label
    return "other (elementwise, reductions, fused)"


def union_length(intervals):
    total, end = 0.0, -math.inf
    for s, e in sorted(intervals):
        if s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def analyze_trace(path, steps):
    """GPU time per category, idle time and top kernels, all per step (ms)."""
    with open(path, encoding="utf-8") as f:
        trace = json.load(f)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    gpu = [e for e in events if e.get("ph") == "X"
           and e.get("cat") in ("kernel", "Kernel", *MEM_CATS)]
    if not gpu:
        return None
    by_cat, by_kernel = defaultdict(float), defaultdict(lambda: [0.0, 0])
    intervals = []
    for e in gpu:
        dur, start = float(e.get("dur", 0.0)), float(e["ts"])
        by_cat[classify(e.get("name", ""), e.get("cat"))] += dur
        k = by_kernel[e.get("name", "?")]
        k[0] += dur
        k[1] += 1
        intervals.append((start, start + dur))
    span = max(e for _, e in intervals) - min(s for s, _ in intervals)
    busy = union_length(intervals)
    us = 1e3 * steps  # trace times are µs; report ms per step
    return {
        "categories": {c: t / us for c, t in
                       sorted(by_cat.items(), key=lambda x: -x[1])},
        "busy_ms": busy / us,
        "span_ms": span / us,
        "idle_ms": (span - busy) / us,
        "kernels_per_step": len(gpu) / steps,
        "top_kernels": [(n[:110], t / us, c / steps) for n, (t, c) in
                        sorted(by_kernel.items(), key=lambda x: -x[1][0])[:12]],
    }


def profile_steps(step, steps, trace_path):
    from torch.profiler import ProfilerActivity, profile, schedule
    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts, schedule=schedule(wait=1, warmup=2, active=steps),
                 on_trace_ready=lambda p: p.export_chrome_trace(trace_path)) as prof:
        for _ in range(1 + 2 + steps):
            step()
            prof.step()
    return analyze_trace(trace_path, steps)


# ---------------------------------------------------------------------------
# Output projection + loss alone
# ---------------------------------------------------------------------------

def bench_head(cfg, args, device):
    """h @ Eᵀ -> label-smoothed CE, forward + backward, eager vs compiled.
    The logits (N, V) are the largest activation of the step; a fused kernel
    that never stores them is the first custom kernel candidate."""
    N = args.batch_size * args.tgt_len
    d, V = cfg["d_model"], cfg["vocab_size"]
    fp16 = args.precision == "fp16" and device.type == "cuda"
    dtype = torch.float16 if fp16 else torch.float32
    g = torch.Generator(device="cpu").manual_seed(0)
    h = torch.randn(N, d, generator=g).to(device, dtype).requires_grad_()
    E = (torch.randn(V, d, generator=g) * d ** -0.5).to(device, dtype).requires_grad_()
    tgt = torch.randint(3, V, (N,), generator=g).to(device)

    def head(h, E, tgt):
        return F.cross_entropy((h @ E.t()).float(), tgt,
                               label_smoothing=cfg["label_smoothing"])

    out = {"logits_mb_fp16": N * V * 2 / 2**20, "logits_mb_fp32": N * V * 4 / 2**20}
    for name, fn in (("eager", head), ("compile", torch.compile(head))):
        def run(fn=fn):
            h.grad = E.grad = None
            fn(h, E, tgt).backward()
        try:
            for _ in range(3):
                run()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            out[f"{name}_ms"] = time_steps(run, args.steps, args.repeats, device)
            if device.type == "cuda":
                out[f"{name}_peak_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
        except Exception as exc:  # noqa: BLE001 - report, don't crash the run
            out[f"{name}_error"] = short_error(exc)
    return out


# ---------------------------------------------------------------------------
# Run + report
# ---------------------------------------------------------------------------

def run_mode(cfg, args, mode, device, trace_dir):
    torch._dynamo.reset()
    step, (src, tgt_in, tgt_out) = build_step(cfg, args, mode, device)
    res = {"mode": mode}
    sync(device)
    t0 = time.perf_counter()
    loss = step()                       # first call compiles
    sync(device)
    res["first_step_s"] = time.perf_counter() - t0
    res["loss"] = float(loss)
    for _ in range(args.warmup):
        step()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    res["ms_per_step"] = time_steps(step, args.steps, args.repeats, device)
    if device.type == "cuda":
        res["peak_mem_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    real_tokens = int((src != cfg["pad_id"]).sum() + (tgt_out != cfg["pad_id"]).sum())
    res["tokens_per_s"] = real_tokens / (res["ms_per_step"] / 1e3)
    trace = os.path.join(trace_dir, f"trace_{mode}.json")
    res["profile"] = profile_steps(step, args.profile_steps, trace)
    return res


def environment(device):
    env = {"torch": torch.__version__, "python": platform.python_version(),
           "device": str(device)}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        env.update(gpu=p.name, capability=f"sm_{p.major}{p.minor}",
                   gpu_mem_gb=round(p.total_memory / 2**30, 1),
                   gpus_visible=torch.cuda.device_count(), cuda=torch.version.cuda,
                   cudnn=torch.backends.cudnn.version())
    try:
        import triton
        env["triton"] = triton.__version__
    except ImportError:
        env["triton"] = None
    return env


def report(env, cfg, args, results, head):
    L = ["# T4 profile: reference Transformer training step", ""]
    L += ["## Environment", "", "| | |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in env.items()]
    L += ["", "## Setup", "",
          f"- batch {args.batch_size} × src {args.src_len} / tgt {args.tgt_len}, "
          f"sentence fill {args.min_fill:.0%}–100% (padding realistic, shapes fixed)",
          f"- precision {args.precision}, attention `{args.attention}`, "
          "fused AdamW, GradScaler",
          f"- timing: median of {args.repeats} × {args.steps} steps after "
          f"{args.warmup} warm-up steps; profile over {args.profile_steps} steps",
          "- config: `" + json.dumps(cfg) + "`", ""]
    if cfg["label_smoothing_mode"] != "uniform":
        L += ["- note: PyTorch only has uniform label smoothing; timed with that.", ""]

    ok = [r for r in results if "error" not in r]
    base = next((r["ms_per_step"] for r in ok if r["mode"] == "eager"), None)
    L += ["## Step time", "",
          "| mode | ms/step | tokens/s | vs eager | peak mem MB "
          "| first step s | loss |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        if "error" in r:
            L.append(f"| {r['mode']} | failed: {r['error']} | | | | | |")
            continue
        speed = f"{base / r['ms_per_step']:.2f}×" if base else "–"
        mem = f"{r['peak_mem_mb']:.0f}" if "peak_mem_mb" in r else "–"
        L.append(f"| {r['mode']} | {r['ms_per_step']:.2f} | {r['tokens_per_s']:,.0f} | "
                 f"{speed} | {mem} | {r['first_step_s']:.1f} | {r['loss']:.3f} |")

    for r in ok:
        p = r["profile"]
        L += ["", f"## GPU time split: {r['mode']}", ""]
        if p is None:
            L.append("No GPU kernels in the trace (CPU run).")
            continue
        span = p["span_ms"]
        L += [f"GPU span {span:.2f} ms/step = busy {p['busy_ms']:.2f} + "
              f"**idle {p['idle_ms']:.2f} ({p['idle_ms'] / span:.0%})**, "
              f"{p['kernels_per_step']:.0f} kernels/step.", "",
              "| category | ms/step | % of busy |", "|---|---:|---:|"]
        L += [f"| {c} | {t:.2f} | {t / p['busy_ms']:.1%} |"
              for c, t in p["categories"].items()]
        L += ["", "<details><summary>Top kernels</summary>", "",
              "| kernel | ms/step | calls/step |", "|---|---:|---:|"]
        L += [f"| `{n}` | {t:.3f} | {c:.0f} |" for n, t, c in p["top_kernels"]]
        L += ["", "</details>"]

    L += ["", "## Output projection + loss alone (fwd + bwd)", "",
          f"N = {args.batch_size * args.tgt_len} target positions, "
          f"V = {cfg['vocab_size']}: "
          f"logits are {head['logits_mb_fp16']:.0f} MB in fp16, "
          f"{head['logits_mb_fp32']:.0f} MB in fp32.", ""]
    for name in ("eager", "compile"):
        if f"{name}_ms" in head:
            share = ""
            step_ms = next((r["ms_per_step"] for r in ok if r["mode"] == name), None)
            if step_ms:
                share = f", {head[f'{name}_ms'] / step_ms:.0%} of the {name} step"
            mem = ""
            if f"{name}_peak_mb" in head:
                mem = f", peak {head[f'{name}_peak_mb']:.0f} MB"
            L.append(f"- {name}: {head[f'{name}_ms']:.2f} ms{share}{mem}")
        elif f"{name}_error" in head:
            L.append(f"- {name}: failed: {head[f'{name}_error']}")

    L += ["", "## What to build first (rules from research.md §5)", ""]
    best = min(ok, key=lambda r: r["ms_per_step"]) if ok else None
    if best and best["profile"]:
        p = best["profile"]
        cats, busy = p["categories"], p["busy_ms"]
        mm = cats.get("matmul", 0) / busy
        loss = cats.get("loss (log_softmax / NLL)", 0) / busy
        idle = p["idle_ms"] / p["span_ms"]
        L += [f"Fastest mode: **{best['mode']}**. Its split: matmul {mm:.0%}, "
              f"loss kernels {loss:.0%} (plus the output-projection matmul, "
              "see above), "
              f"idle {idle:.0%}.", "",
              "- loss + output projection large → first kernel: fused, "
              "logits-free cross-entropy",
              "- idle large → CUDA graphs / megakernel pay most",
              "- matmul dominant (> ~70%) → fusion headroom is small; "
              "rethink the angle"]
    else:
        L.append("No GPU profile available: run this on the T4.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config",
                    help="houssem_torch config JSON (default: transformer_base)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--src-len", type=int, default=64)
    ap.add_argument("--tgt-len", type=int, default=64)
    ap.add_argument("--min-fill", type=float, default=0.5,
                    help="shortest sentence as a fraction of the padded length")
    ap.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    ap.add_argument("--attention", choices=("sdpa", "math"), default="sdpa",
                    help="sdpa: PyTorch's fused attention; "
                         "math: explicit softmax(QKᵀ)V")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--profile-steps", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--trace-dir", help="keep Chrome traces here")
    ap.add_argument("--out", default="t4_profile_report.md")
    args = ap.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            user = json.load(f)
        unknown = sorted(set(user) - set(cfg))
        if unknown:
            raise SystemExit(f"unknown config keys: {unknown}")
        cfg.update(user)
    if max(args.src_len, args.tgt_len) > cfg["max_len"]:
        raise SystemExit(f"sequence length exceeds max_len {cfg['max_len']}")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False  # T4 has no TF32 anyway
    trace_dir = args.trace_dir or tempfile.mkdtemp(prefix="t4_profile_")
    os.makedirs(trace_dir, exist_ok=True)

    env = environment(device)
    print(json.dumps(env, indent=1))
    results = []
    for mode in args.modes:
        print(f"\n== {mode} ...", flush=True)
        try:
            r = run_mode(cfg, args, mode, device, trace_dir)
            print(f"   {r['ms_per_step']:.2f} ms/step, "
                  f"first step {r['first_step_s']:.1f} s")
        except Exception as exc:  # noqa: BLE001 - one failing mode shouldn't kill the run
            r = {"mode": mode, "error": short_error(exc)}
            print("   failed:", r["error"])
        results.append(r)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n== output projection + loss ...", flush=True)
    torch._dynamo.reset()
    head = bench_head(cfg, args, device)

    text = report(env, cfg, args, results, head)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.splitext(args.out)[0].replace("report", "results") + ".json",
              "w", encoding="utf-8") as f:
        json.dump({"env": env, "config": cfg, "args": vars(args),
                   "results": results, "head": head}, f, indent=1)
    print("\n" + text)
    print(f"report: {os.path.abspath(args.out)}")
    if args.trace_dir:
        print(f"traces: {os.path.abspath(trace_dir)}")


if __name__ == "__main__":
    main()
