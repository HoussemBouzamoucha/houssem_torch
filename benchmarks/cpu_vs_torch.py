"""Houssem Torch CPU vs PyTorch CPU: ops, layers, full model, optimizer.

Every case runs with autograd on, on both frameworks, at the sizes of
configs/transformer_base.json (d_model 512, d_ff 2048, 8 heads, 16k vocab).
Before timing, each case checks that the forward output and every input
gradient match PyTorch, so only computations that agree get timed.

Usual entry point (runs 1 and 6 threads, writes cpu_vs_torch_report.md):
    python benchmarks/cpu_vs_torch.py
One thread count only, printed to the terminal:
    BENCH_THREADS=1 python benchmarks/cpu_vs_torch.py --run

Environment knobs: BENCH_THREADS, BENCH_WARMUP (3), BENCH_REPEATS (15),
BENCH_MODEL_REPEATS (5).
"""
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

N_THREADS = int(os.environ.get("BENCH_THREADS", "1"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "3"))
REPEAT = int(os.environ.get("BENCH_REPEATS", "15"))
MODEL_REPEAT = int(os.environ.get("BENCH_MODEL_REPEATS", "5"))
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMBA_NUM_THREADS"):
    os.environ[_var] = str(N_THREADS)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = Path(os.environ.get("BENCH_RESULTS_DIR", tempfile.gettempdir()))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import houssem_torch as ht  # noqa: E402
from houssem_torch import nn, optim  # noqa: E402
from houssem_torch.models import Transformer, TransformerConfig  # noqa: E402
from tests.test_transformer_model import _torch_reference_loss, PAD, BOS  # noqa: E402

torch.set_num_threads(N_THREADS)

# Sizes: a batch of 16 sentences x 64 tokens = 1024 token rows.
B, T, D, H, DFF, V = 16, 64, 512, 8, 2048, 16384
N = B * T


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _median_ms(fn, reset, repeat):
    for _ in range(WARMUP):
        reset()
        fn()
    times = []
    for _ in range(repeat):
        reset()
        gc.collect()
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3


def _close(a, b, rtol, atol, what):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        raise AssertionError(f"{what}: max abs diff {np.max(np.abs(a - b)):.3e}")


# ---------------------------------------------------------------------------
# Op / layer cases
# ---------------------------------------------------------------------------
# make() -> (ht_leaves, torch_leaves, ht_fn, torch_fn). fn(*leaves) returns
# the output; backward is out.sum().backward() (scalar outputs used as-is).

def _pair(arr):
    return (ht.tensor(arr, requires_grad=True),
            torch.tensor(arr, requires_grad=True))


def _leaves(*arrays):
    pairs = [_pair(a) for a in arrays]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _rng():
    return np.random.default_rng(0)


def case_binary(op):
    def make():
        r = _rng()
        h, t = _leaves(r.standard_normal((N, D), dtype=np.float32),
                       r.standard_normal((N, D), dtype=np.float32))
        return h, t, op, op
    return make


def case_unary(op_ht, op_t, positive=False):
    def make():
        x = _rng().standard_normal((N, D), dtype=np.float32)
        if positive:
            x = np.abs(x) + 0.1
        h, t = _leaves(x)
        return h, t, op_ht, op_t
    return make


def make_matmul():
    r = _rng()
    h, t = _leaves(r.standard_normal((N, D), dtype=np.float32),
                   r.standard_normal((D, DFF), dtype=np.float32) * 0.05)
    return h, t, lambda a, b: a @ b, lambda a, b: a @ b


def make_bmm_scores():
    r = _rng()
    shape = (B, H, T, D // H)
    h, t = _leaves(r.standard_normal(shape, dtype=np.float32),
                   r.standard_normal(shape, dtype=np.float32))
    return h, t, lambda q, k: q @ k.mT, lambda q, k: q @ k.transpose(-2, -1)


def make_softmax():
    h, t = _leaves(_rng().standard_normal((B, H, T, T), dtype=np.float32))
    return h, t, lambda x: ht.softmax(x, axis=-1), lambda x: torch.softmax(x, -1)


def make_masked_softmax():
    x = _rng().standard_normal((B, H, T, T), dtype=np.float32)
    mask = np.triu(np.ones((T, T), dtype=bool), 1)
    mask_t = torch.tensor(mask)
    h, t = _leaves(x)
    return (h, t,
            lambda s: s.masked_fill(mask, -1e9).softmax(axis=-1),
            lambda s: s.masked_fill(mask_t, -1e9).softmax(-1))


def make_log_softmax():
    h, t = _leaves(_rng().standard_normal((N, V), dtype=np.float32))
    return (h, t, lambda x: ht.log_softmax(x, axis=-1),
            lambda x: torch.log_softmax(x, -1))


def make_layer_norm():
    r = _rng()
    h, t = _leaves(r.standard_normal((N, D), dtype=np.float32),
                   1 + 0.1 * r.standard_normal(D, dtype=np.float32),
                   0.1 * r.standard_normal(D, dtype=np.float32))
    return (h, t, lambda x, w, b: ht.layer_norm(x, w, b),
            lambda x, w, b: F.layer_norm(x, (D,), w, b))


def make_embedding():
    r = _rng()
    ids = r.integers(0, V, size=(B, T))
    ids_t = torch.tensor(ids)
    h, t = _leaves(r.standard_normal((V, D), dtype=np.float32))
    return h, t, lambda w: ht.embedding(ids, w), lambda w: F.embedding(ids_t, w)


def make_cross_entropy():
    r = _rng()
    tgt = r.integers(3, V, size=N)
    tgt[::7] = PAD
    tgt_t = torch.tensor(tgt)
    h, t = _leaves(r.standard_normal((N, V), dtype=np.float32))
    return (h, t,
            lambda x: ht.cross_entropy(x, tgt, label_smoothing=0.1, ignore_index=PAD),
            lambda x: F.cross_entropy(x, tgt_t, label_smoothing=0.1, ignore_index=PAD))


def make_sum_axis():
    h, t = _leaves(_rng().standard_normal((N, D), dtype=np.float32))
    return h, t, lambda x: ht.sum(x, axis=-1), lambda x: x.sum(-1)


def make_mean_axis():
    h, t = _leaves(_rng().standard_normal((N, D), dtype=np.float32))
    return h, t, lambda x: ht.mean(x, axis=-1), lambda x: x.mean(-1)


# --- layers: the torch side reuses the Houssem weights -------------------

def _torch_params(module):
    params = dict(module.named_parameters())
    return params, {n: torch.tensor(p.numpy(), requires_grad=True)
                    for n, p in params.items()}


def _module_case(module, torch_fn, **call_kwargs):
    hx, tx = _pair(_rng().standard_normal((B, T, D), dtype=np.float32))
    params, P = _torch_params(module)
    return ([hx, *params.values()], [tx, *P.values()],
            lambda *_: module(hx, **call_kwargs),
            lambda *_: torch_fn(tx, P))


def make_linear():
    return _module_case(nn.Linear(D, DFF),
                        lambda x, P: x @ P["weight"] + P["bias"])


def make_feedforward():
    def ref(x, P):
        u = torch.relu(x @ P["up.weight"] + P["up.bias"])
        return u @ P["down.weight"] + P["down.bias"]
    return _module_case(nn.FeedForward(D, DFF, dropout=0.0), ref)


def make_self_attention():
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)

    def ref(x, P):
        q, k, v = (x @ P["w_qkv.weight"] + P["w_qkv.bias"]).split(D, -1)
        q, k, v = (z.reshape(B, T, H, D // H).transpose(1, 2) for z in (q, k, v))
        s = (q @ k.transpose(-2, -1)) / np.sqrt(D // H)
        o = (s.masked_fill(causal, -1e9).softmax(-1) @ v)
        return o.transpose(1, 2).reshape(B, T, D) @ P["w_o.weight"] + P["w_o.bias"]
    return _module_case(nn.MultiHeadAttention(D, H, dropout=0.0), ref, causal=True)


CASES = [
    # name, group, shape label, make, rtol, atol
    ("add", "pointwise", f"({N}, {D})", case_binary(lambda a, b: a + b), 1e-5, 1e-5),
    ("multiply", "pointwise", f"({N}, {D})",
     case_binary(lambda a, b: a * b), 1e-5, 1e-5),
    ("relu", "pointwise", f"({N}, {D})", case_unary(ht.relu, torch.relu), 1e-5, 1e-5),
    ("gelu (tanh)", "pointwise", f"({N}, {D})",
     case_unary(ht.gelu, lambda x: F.gelu(x, approximate="tanh")), 1e-4, 1e-5),
    ("exp", "pointwise", f"({N}, {D})", case_unary(ht.exp, torch.exp), 1e-5, 1e-5),
    ("log", "pointwise", f"({N}, {D})",
     case_unary(ht.log, torch.log, positive=True), 1e-5, 1e-5),
    ("sum(axis=-1)", "reduction", f"({N}, {D})", make_sum_axis, 1e-4, 1e-4),
    ("mean(axis=-1)", "reduction", f"({N}, {D})", make_mean_axis, 1e-4, 1e-5),
    ("matmul", "linalg", f"({N}, {D}) @ ({D}, {DFF})", make_matmul, 1e-4, 1e-3),
    ("batched q @ kT", "linalg", f"2 x ({B}, {H}, {T}, {D // H})",
     make_bmm_scores, 1e-4, 1e-3),
    ("softmax", "transformer op", f"({B}, {H}, {T}, {T})", make_softmax, 1e-4, 1e-5),
    ("causal mask + softmax", "transformer op", f"({B}, {H}, {T}, {T})",
     make_masked_softmax, 1e-4, 1e-5),
    ("log_softmax", "transformer op", f"({N}, {V})", make_log_softmax, 1e-4, 1e-4),
    ("layer_norm", "transformer op", f"({N}, {D})", make_layer_norm, 1e-4, 1e-4),
    ("embedding", "transformer op", f"ids ({B}, {T}), table ({V}, {D})",
     make_embedding, 1e-5, 1e-5),
    ("cross_entropy (ls=0.1)", "transformer op", f"({N}, {V})",
     make_cross_entropy, 1e-4, 1e-6),
    ("Linear", "layer", f"({B}, {T}, {D}) -> {DFF}", make_linear, 1e-4, 1e-3),
    ("FeedForward", "layer", f"({B}, {T}, {D}), d_ff {DFF}",
     make_feedforward, 1e-4, 1e-3),
    ("self-attention (causal)", "layer", f"({B}, {T}, {D}), {H} heads",
     make_self_attention, 1e-3, 1e-3),
]


def run_case(name, group, shape, make, rtol, atol):
    h_leaves, t_leaves, f_h, f_t = make()

    def scalar(out):
        return out if out.ndim == 0 else out.sum()

    # correctness first: forward output and every leaf gradient
    out_h, out_t = f_h(*h_leaves), f_t(*t_leaves)
    _close(out_h.numpy(), out_t.detach().numpy(), rtol, atol, f"{name} forward")
    scalar(out_h).backward()
    scalar(out_t).backward()
    for i, (lh, lt) in enumerate(zip(h_leaves, t_leaves)):
        _close(lh.grad.numpy(), lt.grad.numpy(), rtol, atol, f"{name} grad[{i}]")

    def reset_h():
        for leaf in h_leaves:
            leaf.grad = None

    def reset_t():
        for leaf in t_leaves:
            leaf.grad = None

    return {
        "name": name, "group": group, "shape": shape,
        "ht_fwd": _median_ms(lambda: f_h(*h_leaves), reset_h, REPEAT),
        "pt_fwd": _median_ms(lambda: f_t(*t_leaves), reset_t, REPEAT),
        "ht_fb": _median_ms(lambda: scalar(f_h(*h_leaves)).backward(), reset_h, REPEAT),
        "pt_fb": _median_ms(lambda: scalar(f_t(*t_leaves)).backward(), reset_t, REPEAT),
    }


# ---------------------------------------------------------------------------
# Full model training step and optimizer step
# ---------------------------------------------------------------------------

MB, MT = 8, 32  # model batch: 8 sentences x 32 tokens


def run_model():
    base = TransformerConfig.from_json(str(ROOT / "configs" / "transformer_base.json"))
    cfg = TransformerConfig.from_dict({**base.to_dict(), "dropout": 0.0})
    ht.manual_seed(0)
    model = Transformer(cfg)
    r = _rng()
    src = r.integers(3, cfg.vocab_size, size=(MB, MT))
    tgt = r.integers(3, cfg.vocab_size, size=(MB, MT))
    src[0, -5:] = PAD
    tgt_in = np.concatenate([np.full((MB, 1), BOS), tgt[:, :-1]], axis=1)
    tgt_out = tgt.copy()
    tgt_in[1, -4:], tgt_out[1, -4:] = PAD, PAD

    P = {n: torch.tensor(p.numpy(), requires_grad=True)
         for n, p in model.named_parameters()}

    def loss_h():
        return model.loss(src, tgt_in, tgt_out)

    def loss_t():
        return _torch_reference_loss(P, cfg, src, tgt_in, tgt_out)

    lh, lt = loss_h(), loss_t()
    _close(lh.numpy(), lt.item(), 1e-4, 1e-4, "model loss")
    lh.backward()
    lt.backward()
    worst = max(float(np.max(np.abs(p.grad.numpy() - P[n].grad.numpy())))
                for n, p in model.named_parameters())
    if worst > 1e-3:
        raise AssertionError(f"model grads: max abs diff {worst:.3e}")

    def reset_t():
        for p in P.values():
            p.grad = None

    params = f"{model.num_parameters() / 1e6:.1f}M params"
    rows = [{
        "name": "Transformer loss", "group": "model",
        "shape": f"base config ({params}), batch {MB}x{MT}",
        "ht_fwd": _median_ms(loss_h, model.zero_grad, MODEL_REPEAT),
        "pt_fwd": _median_ms(loss_t, reset_t, MODEL_REPEAT),
        "ht_fb": _median_ms(lambda: loss_h().backward(), model.zero_grad, MODEL_REPEAT),
        "pt_fb": _median_ms(lambda: loss_t().backward(), reset_t, MODEL_REPEAT),
    }]

    # Adam step over the same parameters (gradients populated once).
    loss_h().backward()
    loss_t().backward()
    opt_h = optim.Adam(model.parameters(), lr=1e-4)
    opt_t = torch.optim.Adam(list(P.values()), lr=1e-4)
    rows.append({
        "name": "Adam step", "group": "optimizer", "shape": params,
        "ht_fwd": None, "pt_fwd": None,
        "ht_fb": _median_ms(opt_h.step, lambda: None, REPEAT),
        "pt_fb": _median_ms(opt_t.step, lambda: None, REPEAT),
    })
    return rows


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _fmt(x):
    return "-" if x is None else f"{x:.3f}"


def _ratio(h, t):
    return "-" if h is None or t is None else f"{h / t:.2f}x"


def _print_row(r):
    print(f"{r['name']:<26}{_fmt(r['ht_fwd']):>10}{_fmt(r['pt_fwd']):>10}"
          f"{_ratio(r['ht_fwd'], r['pt_fwd']):>8}{_fmt(r['ht_fb']):>11}"
          f"{_fmt(r['pt_fb']):>11}{_ratio(r['ht_fb'], r['pt_fb']):>8}", flush=True)


def run():
    print(f"threads={N_THREADS} torch={torch.__version__} numpy={np.__version__} "
          f"warmup={WARMUP} repeat={REPEAT}")
    print(f"{'case':<26}{'HT fwd':>10}{'PT fwd':>10}{'ratio':>8}"
          f"{'HT f+b':>11}{'PT f+b':>11}{'ratio':>8}")
    rows = []
    for spec in CASES:
        rows.append(run_case(*spec))
        _print_row(rows[-1])
    for row in run_model():
        rows.append(row)
        _print_row(row)
    out = RESULTS_DIR / f"cpu_vs_torch_{N_THREADS}t.json"
    out.write_text(json.dumps({
        "threads": N_THREADS, "torch": torch.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "warmup": WARMUP, "repeat": REPEAT,
        "model_repeat": MODEL_REPEAT, "rows": rows,
    }, indent=2), encoding="utf-8")


def report(thread_counts):
    results = [json.loads((RESULTS_DIR / f"cpu_vs_torch_{n}t.json")
                          .read_text(encoding="utf-8")) for n in thread_counts]
    meta = results[0]
    lines = [
        "# Houssem Torch CPU vs PyTorch CPU",
        "",
        "- **Benchmark:** [`cpu_vs_torch.py`](./cpu_vs_torch.py)",
        f"- **CPU:** {platform.processor() or platform.machine()}",
        f"- **Runtime:** Python {meta['python']}, NumPy {meta['numpy']}, "
        f"PyTorch {meta['torch']}",
        f"- **Method:** autograd on for both; {meta['warmup']} warmups, medians "
        f"of {meta['repeat']} runs ({meta['model_repeat']} for the full model). "
        "Every case is first checked against PyTorch (forward output and all "
        "input gradients).",
        "- **Ratio** = Houssem time / PyTorch time; below 1.00x means Houssem "
        "is faster.",
        "- **f+b** = forward, then `.sum().backward()` (scalar outputs used "
        "as-is). The Adam row times one optimizer step.",
        "",
    ]
    for res in results:
        lines += [f"## {res['threads']} thread(s)", "",
                  "| Case | Shape | HT fwd ms | PT fwd ms | fwd ratio | "
                  "HT f+b ms | PT f+b ms | f+b ratio |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for r in res["rows"]:
            lines.append(
                f"| {r['name']} | {r['shape']} | {_fmt(r['ht_fwd'])} | "
                f"{_fmt(r['pt_fwd'])} | {_ratio(r['ht_fwd'], r['pt_fwd'])} | "
                f"{_fmt(r['ht_fb'])} | {_fmt(r['pt_fb'])} | "
                f"{_ratio(r['ht_fb'], r['pt_fb'])} |")
        lines.append("")
    path = HERE / "cpu_vs_torch_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


def main():
    if "--run" in sys.argv:
        run()
        return
    threads = (1, 6)
    for n in threads:
        env = {**os.environ, "BENCH_THREADS": str(n)}
        subprocess.run([sys.executable, __file__, "--run"], env=env, check=True)
    report(threads)


if __name__ == "__main__":
    main()
