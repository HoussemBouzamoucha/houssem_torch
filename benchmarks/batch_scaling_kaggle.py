# ruff: noqa: E402  (notebook cells: each cell imports what it uses)
# %% [markdown]
# # Batch-size scaling on Kaggle: houssem_torch vs torch.compile
#
# The best setup so far (2026-11-26): length buckets with a token budget.
# Every batch has about the same number of target tokens, and short buckets
# take more sentence pairs. This script keeps that setup and changes only
# the batch size, in tokens per step. SCALES are multiples of one fixed
# 64-pair batch (~1,470 target tokens). Each size trains TRAIN_STEPS steps
# from the same initial weights with the same LR schedule, on both engines,
# on the same card. The order of the engines alternates between sizes, so
# GPU heat does not always favour the same one.
#
# Reports: validation loss against steps and against training time (the
# fair way to compare sizes), ms/step, target tokens/s, memory, setup time.
# Outputs: /kaggle/working/scaling_results_<date>/ and a .zip of it.
#
# Kaggle settings: Accelerator "GPU T4 x2", Internet "On".
# Run: %run houssem_torch/benchmarks/batch_scaling_kaggle.py

# %% Cell 1: settings
REPO_URL = "https://github.com/HoussemBouzamoucha/houssem_torch.git"
BRANCH = "gpu/gemm-tuning-buckets"
WORK = "/kaggle/working"
REPO_DIR = f"{WORK}/houssem_torch"

SRC_LANG, TGT_LANG = "fr", "en"
BASE_BATCH, MAX_LEN = 64, 64      # one "batch 64" of tokens; longest sentence
LEN_BUCKETS = (16, 32, 48, 64)
SCALES = (1, 2, 4)                # tokens per step = scale x one 64-pair batch
TRAIN_STEPS = 6000
VAL_EVERY = 1000                  # validation loss (excluded from timing)
LOG_EVERY = 500
WARMUP_STEPS, LR_FACTOR = 2000, 1.0     # Noam schedule, same for every size
MAX_GRAD_NORM = 1.0
VAL_FRACTION = 0.02
SEED = 0
RUN_TORCH = True                  # False: houssem_torch only (half the time)

# %% Cell 2: get the code, dependencies, GPU, tests
import os
import subprocess
import sys


def sh(cmd, cwd=None):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


if not os.path.isdir(f"{REPO_DIR}/.git"):
    sh(f"git clone --branch {BRANCH} {REPO_URL} {REPO_DIR}")
else:
    sh(f"git -C {REPO_DIR} fetch origin {BRANCH}")
    sh(f"git -C {REPO_DIR} checkout {BRANCH}")
    sh(f"git -C {REPO_DIR} pull --ff-only origin {BRANCH}")
sh(f"git -C {REPO_DIR} log -1 --oneline")
for p in (REPO_DIR, f"{REPO_DIR}/benchmarks"):
    if p not in sys.path:
        sys.path.insert(0, p)
for name in list(sys.modules):   # use the code just pulled
    if (name == "houssem_torch" or name.startswith("houssem_torch.")
            or name == "t4_profile"):
        del sys.modules[name]

try:
    import cupy  # noqa: F401
except ImportError:
    sh(f"{sys.executable} -m pip install -q cupy-cuda12x nvidia-cublas-cu12 "
       "nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12")
for mod in ("tokenizers", "datasets"):
    try:
        __import__(mod)
    except ImportError:
        sh(f"{sys.executable} -m pip install -q {mod}")

import gc
import json
import time

import cupy as cp
import numpy as np
import torch

from houssem_torch.backends.cuda import driver as drv

assert drv.is_available(), "CuPy cannot see a GPU: enable 'GPU T4 x2'"
GPU = int(os.environ.get("HT_GPU", 1 if torch.cuda.device_count() > 1 else 0))
torch_dev = torch.device(f"cuda:{GPU}")
print(drv.device_info(GPU))
print("PyTorch", torch.__version__, f"on cuda:{GPU} (both engines)")
sh(f"{sys.executable} -m pytest -q -p no:warnings tests/test_cuda_driver.py "
   "tests/test_cuda_attention.py tests/test_cuda_train_step.py", cwd=REPO_DIR)


def sm_clock():
    r = subprocess.run(f"nvidia-smi -i {GPU} --query-gpu=clocks.sm "
                       "--format=csv,noheader", shell=True,
                       capture_output=True, text=True)
    return r.stdout.strip() or "?"


# %% Cell 3: data, tokenizer, buckets
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from tokenizers.trainers import BpeTrainer

from houssem_torch.models import TransformerConfig

raw = load_dataset("Helsinki-NLP/opus_books", f"{min(SRC_LANG, TGT_LANG)}-"
                   f"{max(SRC_LANG, TGT_LANG)}", split="train")
pairs = [(r["translation"][SRC_LANG], r["translation"][TGT_LANG]) for r in raw]
order = np.random.default_rng(SEED).permutation(len(pairs))
n_val = int(len(pairs) * VAL_FRACTION)
val_pairs = [pairs[i] for i in order[:n_val]]
train_pairs = [pairs[i] for i in order[n_val:]]

base_cfg = TransformerConfig.from_json(f"{REPO_DIR}/configs/transformer_base.json")
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.normalizer = normalizers.NFKC()
tok.pre_tokenizer = pre_tokenizers.Metaspace()
tok.decoder = decoders.Metaspace()
tok.train_from_iterator((s for pair in train_pairs for s in pair),
                        BpeTrainer(vocab_size=base_cfg.vocab_size,
                                   special_tokens=SPECIALS, min_frequency=2,
                                   show_progress=False))
VOCAB = tok.get_vocab_size()
PAD, BOS, EOS = base_cfg.pad_id, base_cfg.bos_id, base_cfg.eos_id


def encode(pairs_):
    src = tok.encode_batch([p[0] for p in pairs_])
    tgt = tok.encode_batch([p[1] for p in pairs_])
    return [(s.ids + [EOS], t.ids) for s, t in zip(src, tgt)
            if len(s.ids) + 1 <= MAX_LEN and len(t.ids) + 1 <= MAX_LEN]


train_ids, val_ids = encode(train_pairs), encode(val_pairs)


def to_batch(chunk, S, T):
    B = len(chunk)
    src = np.full((B, S), PAD, np.int64)
    tgt_in = np.full((B, T), PAD, np.int64)
    tgt_out = np.full((B, T), PAD, np.int64)
    for i, (s, t) in enumerate(chunk):
        src[i, :len(s)] = s
        tgt_in[i, 0], tgt_in[i, 1:len(t) + 1] = BOS, t
        tgt_out[i, :len(t)], tgt_out[i, len(t)] = t, EOS
    return src, tgt_in, tgt_out


def bucket_of(pair):
    need = max(len(pair[0]), len(pair[1]) + 1)
    return next(L for L in LEN_BUCKETS if L >= need)


lens = {L: [] for L in LEN_BUCKETS}
for pair in train_ids:
    lens[bucket_of(pair)].append(len(pair[1]) + 1)
mean_tokens = float(np.mean([len(t) + 1 for _, t in train_ids]))


def bucket_batch(scale):
    """Pairs per batch for each bucket: ~scale x BASE_BATCH x mean tokens."""
    return {L: max(8, int(round(scale * BASE_BATCH * mean_tokens
                                / np.mean(v) / 8)) * 8) for L, v in lens.items()}


def bucketed_stream(scale, n_steps):
    """The same batches, in the same order, for both engines at this scale."""
    bb = bucket_batch(scale)
    epoch, made = 0, 0
    while True:
        rng_ = np.random.default_rng(SEED + epoch)
        groups = {L: [] for L in LEN_BUCKETS}
        for j in rng_.permutation(len(train_ids)):
            groups[bucket_of(train_ids[j])].append(j)
        chunks = [(L, idx[i:i + bb[L]]) for L, idx in groups.items()
                  for i in range(0, len(idx) - bb[L] + 1, bb[L])]
        for k in rng_.permutation(len(chunks)):
            L, idx = chunks[k]
            yield to_batch([train_ids[j] for j in idx], L, L)
            made += 1
            if made == n_steps:
                return
        epoch += 1


vidx = np.random.default_rng(SEED).permutation(len(val_ids))
val_batches = [to_batch([val_ids[j] for j in vidx[i:i + BASE_BATCH]], MAX_LEN,
                        MAX_LEN)
               for i in range(0, len(vidx) - BASE_BATCH + 1, BASE_BATCH)]
print(f"{len(train_ids):,} train pairs, {len(val_batches)} val batches, "
      f"{BASE_BATCH * mean_tokens:.0f} target tokens per scale-1 batch")
for s in SCALES:
    print(f"scale {s}: pairs per batch {bucket_batch(s)}")

# %% Cell 4: engines from the same initial weights
import houssem_torch as ht
from houssem_torch.backends.cuda.train_step import BucketedTrainStep
from houssem_torch.models import Transformer
from houssem_torch.optim import noam_lr
import t4_profile as tp

cfg = TransformerConfig.from_dict({**base_cfg.to_dict(), "vocab_size": VOCAB,
                                   "max_len": max(MAX_LEN, base_cfg.max_len)})
ht.manual_seed(SEED)
cpu_model = Transformer(cfg)
init_state = cpu_model.state_dict()
ADAM = dict(betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
HT, TC = "houssem_torch", "torch.compile"


def torch_name(name):
    """houssem_torch parameter name -> t4_profile.Transformer name."""
    parts = name.split(".")
    if parts[0] in ("encoder_norm", "decoder_norm"):
        return {"encoder_norm": "enc_norm", "decoder_norm": "dec_norm"}[parts[0]] \
            + "." + parts[1]
    if parts[0] == "embedding":
        return name
    stack, i, mod, rest = parts[0], parts[1], parts[2], parts[3:]
    norms = {"norm_attn": "attn", "norm_self": "self_attn",
             "norm_cross": "cross_attn", "norm_ffn": "ffn"}
    if mod in norms:
        return ".".join([stack, i, norms[mod], "norm", *rest])
    block = "attn" if (stack == "encoder" and mod == "self_attn") else mod
    rest = [{"up": "l1", "down": "l2"}.get(r, r) for r in rest]
    return ".".join([stack, i, block, "f", *rest])


def free_gpu():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()


class HoussemEngine:
    def __init__(self, scale):
        cpu_model.load_state_dict(init_state)
        bb = bucket_batch(scale)
        shapes = [(bb[L], L, L) for L in LEN_BUCKETS]
        shapes.append((BASE_BATCH, MAX_LEN, MAX_LEN))      # the val batches
        self.step = BucketedTrainStep(cpu_model, BASE_BATCH, shapes,
                                      precision="fp16",
                                      max_grad_norm=MAX_GRAD_NORM, seed=SEED,
                                      device=GPU, **ADAM)
        drv.synchronize()

    def train(self, batch, lr):
        self.step(*batch, lr=lr)

    def loss(self):
        return self.step.last_loss()

    def sync(self):
        drv.synchronize()

    def val(self):
        return float(np.mean([self.step.evaluate(*b) for b in val_batches]))

    def memory_gb(self):
        return drv.memory_in_use() / 2 ** 30


class TorchEngine:
    def __init__(self, scale):
        torch.manual_seed(SEED)
        torch._dynamo.reset()
        model = tp.Transformer(cfg.to_dict(), "sdpa").to(torch_dev).train()
        mapped = {}
        for name, value in init_state.items():
            t = torch.from_numpy(np.ascontiguousarray(value))
            if value.ndim == 2 and name != "embedding.weight":
                t = t.T.contiguous()      # houssem (in, out), torch (out, in)
            mapped[torch_name(name)] = t
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        assert not unexpected and set(missing) <= {"pe"}, (missing, unexpected)
        self.model = model
        self.opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True,
                                     **ADAM)
        self.scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 16,
                                           growth_interval=2000)
        self.loss_fn = torch.compile(model.loss, dynamic=False)  # graph per bucket
        self._loss = None
        torch.cuda.reset_peak_memory_stats(torch_dev)

    def train(self, batch, lr):
        b = tuple(torch.from_numpy(a).to(torch_dev, non_blocking=True)
                  for a in batch)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = self.loss_fn(*b)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
        self.scaler.step(self.opt)
        self.scaler.update()
        self._loss = loss

    def loss(self):
        return float(self._loss.detach().item())

    def sync(self):
        torch.cuda.synchronize(torch_dev)

    def val(self):
        self.model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            v = float(np.mean([float(self.model.loss(*(
                torch.from_numpy(a).to(torch_dev) for a in b)))
                for b in val_batches]))
        self.model.train()
        return v

    def memory_gb(self):
        return torch.cuda.max_memory_allocated(torch_dev) / 2 ** 30


# %% Cell 5: train every (scale, engine)
def run(engine_cls, name, scale):
    t0 = time.perf_counter()
    eng = engine_cls(scale)
    setup = time.perf_counter() - t0
    val_time, tokens = 0.0, 0
    curve, log = [], []
    t_start = time.perf_counter()
    for i, b in enumerate(bucketed_stream(scale, TRAIN_STEPS), start=1):
        eng.train(b, noam_lr(i, cfg.d_model, WARMUP_STEPS, LR_FACTOR))
        tokens += int((b[2] != PAD).sum())
        if i % LOG_EVERY == 0 or i % VAL_EVERY == 0:
            eng.sync()
            el = time.perf_counter() - t_start - val_time
            log.append((i, el))
            line = (f"[{name} x{scale}] step {i:5d}  {el:7.1f}s  "
                    f"batch loss {eng.loss():.3f} (width {b[0].shape[1]})")
            if i % VAL_EVERY == 0:
                tv = time.perf_counter()
                v = eng.val()
                val_time += time.perf_counter() - tv
                curve.append((i, el, v))
                line += f"  val {v:.4f}"
            print(line + f"  {sm_clock()}", flush=True)
    eng.sync()
    wall = time.perf_counter() - t_start - val_time
    (i0, e0), (i1, e1) = log[0], log[-1]
    res = dict(engine=name, scale=scale, pairs_per_batch=bucket_batch(scale),
               setup_s=setup, wall_s=wall, steady_ms=(e1 - e0) / (i1 - i0) * 1e3,
               tokens_per_s=tokens / wall, target_tokens_per_step=tokens / TRAIN_STEPS,
               val_curve=curve, val=curve[-1][2], mem_gb=eng.memory_gb())
    del eng
    free_gpu()
    print(f"== {name} x{scale}: {wall:.0f} s, {res['steady_ms']:.1f} ms/step, "
          f"{res['tokens_per_s']:,.0f} target tok/s, val {res['val']:.4f}, "
          f"{res['mem_gb']:.2f} GB, setup {setup:.1f} s", flush=True)
    return res


engines = [(HoussemEngine, HT)] + ([(TorchEngine, TC)] if RUN_TORCH else [])
results = []
for n, scale in enumerate(SCALES):
    for cls, name in (engines if n % 2 == 0 else engines[::-1]):
        results.append(run(cls, name, scale))

# %% Cell 6: report, chart, download
import matplotlib.pyplot as plt

stamp = time.strftime("%Y%m%d_%H%M")
out = f"{WORK}/scaling_results_{stamp}"
os.makedirs(out, exist_ok=True)
gpu_name = drv.device_info(GPU).name
by = {(r["engine"], r["scale"]): r for r in results}
lines = [f"# Batch-size scaling: houssem_torch vs torch.compile ({gpu_name})", "",
         f"Token-budget length buckets {list(LEN_BUCKETS)}, {TRAIN_STEPS} steps per "
         f"run, same init, data order and Noam schedule (warmup {WARMUP_STEPS}). "
         f"Scale 1 = {BASE_BATCH * mean_tokens:.0f} target tokens per step "
         f"(one 64-pair batch). Both engines on cuda:{GPU}; validation time "
         "excluded.", "",
         "| scale | engine | pairs per batch (16/32/48/64) | ms/step | target tok/s "
         "| wall s | val loss | memory GB | setup s |",
         "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
for r in results:
    bb = "/".join(str(r["pairs_per_batch"][L]) for L in LEN_BUCKETS)
    lines.append(f"| {r['scale']} | {r['engine']} | {bb} | {r['steady_ms']:.1f} | "
                 f"{r['tokens_per_s']:,.0f} | {r['wall_s']:.0f} | {r['val']:.4f} | "
                 f"{r['mem_gb']:.2f} | {r['setup_s']:.1f} |")
if RUN_TORCH:
    lines += ["", "| scale | houssem_torch speedup (ms/step) | val loss ht / torch |",
              "|---:|---:|---:|"]
    for s in SCALES:
        a, b = by[HT, s], by[TC, s]
        lines.append(f"| {s} | {b['steady_ms'] / a['steady_ms']:.2f}x | "
                     f"{a['val']:.4f} / {b['val']:.4f} |")
lines += ["", "## Validation loss curves (step, training seconds, val)", ""]
for r in results:
    lines.append(f"- {r['engine']} x{r['scale']}: " + ", ".join(
        f"{i}: {v:.3f} @ {e:.0f}s" for i, e, v in r["val_curve"]))
report = "\n".join(lines) + "\n"
print(report)
with open(f"{out}/scaling_report.md", "w", encoding="utf-8") as f:
    f.write(report)
with open(f"{out}/scaling_results.json", "w", encoding="utf-8") as f:
    json.dump(dict(gpu=gpu_name, torch=torch.__version__, steps=TRAIN_STEPS,
                   buckets=list(LEN_BUCKETS), base_tokens=BASE_BATCH * mean_tokens,
                   results=results), f, indent=1, default=float)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
colors = {1: "#4c78a8", 2: "#f58518", 4: "#54a24b", 8: "#b279a2"}
for r in results:
    st = "-" if r["engine"] == HT else "--"
    c = colors.get(r["scale"], "#777777")
    steps = [i for i, _, _ in r["val_curve"]]
    secs = [e for _, e, _ in r["val_curve"]]
    vals = [v for _, _, v in r["val_curve"]]
    lab = f"{'ht' if r['engine'] == HT else 'torch'} x{r['scale']}"
    axes[0].plot(steps, vals, st, color=c, marker="o", ms=3, label=lab)
    axes[1].plot(secs, vals, st, color=c, marker="o", ms=3, label=lab)
axes[0].set(xlabel="step", ylabel="val loss", title="Val loss vs steps")
axes[1].set(xlabel="training seconds", ylabel="val loss",
            title="Val loss vs training time")
for name, off in ((HT, -0.2), (TC, 0.2)):
    rs = [by[name, s] for s in SCALES if (name, s) in by]
    axes[2].bar([SCALES.index(r["scale"]) + off for r in rs],
                [r["tokens_per_s"] / 1e3 for r in rs], width=0.4, label=name)
axes[2].set_xticks(range(len(SCALES)), [f"x{s}" for s in SCALES])
axes[2].set(xlabel="tokens per step (x one 64-pair batch)",
            ylabel="thousand target tokens / s", title="Throughput")
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{out}/scaling.png", dpi=120)
plt.show()

import shutil

archive = shutil.make_archive(out, "zip", out)
print("zip:", archive)
try:
    from IPython.display import FileLink, display
    display(FileLink(os.path.relpath(archive, WORK)))
except Exception:  # noqa: BLE001  (not in a notebook)
    pass
