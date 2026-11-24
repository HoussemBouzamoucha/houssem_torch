# ruff: noqa: E402  (notebook cells: each cell imports what it uses)
# %% [markdown]
# # houssem_torch GPU training step vs PyTorch, on Kaggle
#
# Trains the same encoder-decoder Transformer (configs/transformer_base.json)
# on opus_books (fr -> en) with two engines and compares them:
#
#   * houssem_torch `TrainStep`: compiled whole-step engine (cuBLAS + our fused
#     CUDA kernels via CuPy), fp16 mixed precision, no PyTorch
#   * PyTorch: the same model (benchmarks/t4_profile.py), AMP fp16, eager and
#     torch.compile
#
# Both start from identical weights and see identical batches in the same
# order, with the same AdamW settings, LR schedule, clipping and loss scaling.
#
# Kaggle settings: Accelerator "GPU T4 x2", Internet "On".
# Run the cells top to bottom (each `# %%` block is one notebook cell), or
# the whole file with `!python cuda_train_kaggle.py`.
# Outputs in /kaggle/working: cuda_vs_torch_report.md, checkpoints, tokenizer.

# %% Cell 1: settings
REPO_URL = "https://github.com/HoussemBouzamoucha/houssem_torch.git"
BRANCH = "gpu/driver-layer"
WORK = "/kaggle/working"
REPO_DIR = f"{WORK}/houssem_torch"

SRC_LANG, TGT_LANG = "fr", "en"
BATCH, SRC_LEN, TGT_LEN = 64, 64, 64      # fixed batch shape for both engines
VAL_FRACTION = 0.02
SEED = 0

BENCH_WARMUP, BENCH_STEPS, BENCH_REPEATS = 10, 30, 3
TRAIN_STEPS = 3000              # per engine; ~1,900 steps = one epoch
WARMUP_STEPS, LR_FACTOR = 2000, 1.0     # Noam schedule
MAX_GRAD_NORM = 1.0
LOG_EVERY = 250
TORCH_TRAIN_MODE = "compile"    # PyTorch training run: "compile" or "eager"

# %% Cell 2: get the code (git clone the first time, git pull afterwards)
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

# %% Cell 3: dependencies and GPUs
try:
    import cupy  # noqa: F401
except ImportError:
    sh(f"{sys.executable} -m pip install -q cupy-cuda12x nvidia-cublas-cu12 "
       "nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12")
for mod, pkg in (("tokenizers", "tokenizers"), ("datasets", "datasets")):
    try:
        __import__(mod)
    except ImportError:
        sh(f"{sys.executable} -m pip install -q {pkg}")

sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv")
import numpy as np
import torch

from houssem_torch.backends.cuda import driver as drv

assert drv.is_available(), "CuPy cannot see a GPU: enable 'GPU T4 x2'"
N_GPUS = torch.cuda.device_count()
HT_GPU = 0
TORCH_GPU = 1 if N_GPUS > 1 else 0      # one engine per T4 when there are two
print(drv.device_info(HT_GPU))
print("PyTorch", torch.__version__, "on cuda:%d" % TORCH_GPU)

# %% Cell 4: correctness tests on this GPU (GPU step vs CPU Transformer)
sh(f"{sys.executable} -m pytest -q -p no:warnings tests/test_cuda_driver.py "
   "tests/test_cuda_train_step.py", cwd=REPO_DIR)

# %% Cell 5: download opus_books and split
from datasets import load_dataset

raw = load_dataset("Helsinki-NLP/opus_books", f"{min(SRC_LANG, TGT_LANG)}-"
                   f"{max(SRC_LANG, TGT_LANG)}", split="train")
pairs = [(r["translation"][SRC_LANG], r["translation"][TGT_LANG]) for r in raw]
rng = np.random.default_rng(SEED)
order = rng.permutation(len(pairs))
n_val = int(len(pairs) * VAL_FRACTION)
val_pairs = [pairs[i] for i in order[:n_val]]
train_pairs = [pairs[i] for i in order[n_val:]]
print(f"{len(pairs):,} pairs: {len(train_pairs):,} train, {len(val_pairs):,} val")
print("example:", train_pairs[0])

# %% Cell 6: joint BPE tokenizer (ids 0/1/2 = pad/bos/eos, as in the config)
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from tokenizers.trainers import BpeTrainer

from houssem_torch.models import TransformerConfig

base_cfg = TransformerConfig.from_json(f"{REPO_DIR}/configs/transformer_base.json")
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.normalizer = normalizers.NFKC()
tok.pre_tokenizer = pre_tokenizers.Metaspace()
tok.decoder = decoders.Metaspace()
trainer = BpeTrainer(vocab_size=base_cfg.vocab_size, special_tokens=SPECIALS,
                     min_frequency=2, show_progress=False)
tok.train_from_iterator((s for pair in train_pairs for s in pair), trainer)
tok.save(f"{WORK}/tokenizer.json")
assert [tok.token_to_id(t) for t in SPECIALS[:3]] == [
    base_cfg.pad_id, base_cfg.bos_id, base_cfg.eos_id]
VOCAB = tok.get_vocab_size()
print("vocab:", VOCAB)

# %% Cell 7: encode and build fixed-shape batches
PAD, BOS, EOS = base_cfg.pad_id, base_cfg.bos_id, base_cfg.eos_id


def encode(pairs_):
    src = tok.encode_batch([p[0] for p in pairs_])
    tgt = tok.encode_batch([p[1] for p in pairs_])
    out = []
    for s, t in zip(src, tgt):
        s_ids, t_ids = s.ids + [EOS], t.ids
        if len(s_ids) <= SRC_LEN and len(t_ids) + 1 <= TGT_LEN:
            out.append((s_ids, t_ids))
    return out


train_ids, val_ids = encode(train_pairs), encode(val_pairs)
print(f"kept {len(train_ids):,}/{len(train_pairs):,} train pairs that fit "
      f"{SRC_LEN}/{TGT_LEN} tokens")


def to_batch(chunk):
    src = np.full((BATCH, SRC_LEN), PAD, np.int64)
    tgt_in = np.full((BATCH, TGT_LEN), PAD, np.int64)
    tgt_out = np.full((BATCH, TGT_LEN), PAD, np.int64)
    for i, (s, t) in enumerate(chunk):
        src[i, :len(s)] = s
        tgt_in[i, 0], tgt_in[i, 1:len(t) + 1] = BOS, t
        tgt_out[i, :len(t)], tgt_out[i, len(t)] = t, EOS
    return src, tgt_in, tgt_out


def batches(data, epoch_seed):
    """Full batches only (the GPU step has a fixed shape), shuffled per epoch."""
    idx = np.random.default_rng(epoch_seed).permutation(len(data))
    for i in range(0, len(idx) - BATCH + 1, BATCH):
        yield to_batch([data[j] for j in idx[i:i + BATCH]])


def train_stream(n_steps):
    """The same n_steps batches, in the same order, for both engines."""
    epoch, made = 0, 0
    while made < n_steps:
        for b in batches(train_ids, SEED + epoch):
            yield b
            made += 1
            if made == n_steps:
                return
        epoch += 1


val_batches = list(batches(val_ids, SEED))
steps_per_epoch = len(train_ids) // BATCH
print(f"{steps_per_epoch} steps/epoch, {len(val_batches)} val batches")

# %% Cell 8: build both engines from the same initial weights
import gc
import time

import cupy as cp
import houssem_torch as ht
from houssem_torch.backends.cuda.train_step import TrainStep
from houssem_torch.models import Transformer
from houssem_torch.optim import noam_lr
import t4_profile as tp

cfg = TransformerConfig.from_dict({**base_cfg.to_dict(), "vocab_size": VOCAB,
                                   "max_len": max(SRC_LEN, TGT_LEN,
                                                  base_cfg.max_len)})
ht.manual_seed(SEED)
cpu_model = Transformer(cfg)
init_state = cpu_model.state_dict()
print(f"model: {cpu_model.num_parameters():,} parameters")
ADAM = dict(betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
torch_dev = torch.device(f"cuda:{TORCH_GPU}")


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


def build_torch(state):
    torch.manual_seed(SEED)
    model = tp.Transformer(cfg.to_dict(), "sdpa").to(torch_dev).train()
    mapped = {}
    for name, value in state.items():
        t = torch.from_numpy(np.ascontiguousarray(value))
        if value.ndim == 2 and name != "embedding.weight":
            t = t.T.contiguous()          # Linear: houssem (in, out), torch (out, in)
        mapped[torch_name(name)] = t
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    assert not unexpected and set(missing) <= {"pe"}, (missing, unexpected)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True, **ADAM)
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 16,
                                  growth_interval=2000)
    return model, opt, scaler


def torch_step_fn(model, opt, scaler, mode):
    loss_fn = model.loss if mode == "eager" else torch.compile(model.loss)

    def step(batch, lr):
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = loss_fn(*batch)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(opt)
        scaler.update()
        return loss
    return step


def to_torch(batch):
    return tuple(torch.from_numpy(a).to(torch_dev, non_blocking=True) for a in batch)


def build_ht():
    cpu_model.load_state_dict(init_state)
    t0 = time.perf_counter()
    step = TrainStep(cpu_model, BATCH, SRC_LEN, TGT_LEN, precision="fp16",
                     max_grad_norm=MAX_GRAD_NORM, seed=SEED, device=HT_GPU, **ADAM)
    drv.synchronize()
    return step, time.perf_counter() - t0


def free_gpu():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()


# %% Cell 9: same model? loss of both engines on one batch, dropout off
probe = val_batches[0]
ht_step, ht_build_s = build_ht()
ht_eval = ht_step.evaluate(*probe)
tm, _, _ = build_torch(init_state)
tm.eval()
with torch.no_grad():
    torch_fp32 = float(tm.loss(*to_torch(probe)))
    with torch.autocast("cuda", dtype=torch.float16):
        torch_fp16 = float(tm.loss(*to_torch(probe)))
print(f"eval loss  houssem fp16 {ht_eval:.5f} | torch fp32 {torch_fp32:.5f} | "
      f"torch fp16 {torch_fp16:.5f}")
assert abs(ht_eval - torch_fp32) < 2e-2 * torch_fp32, "engines disagree"
del tm, ht_step
free_gpu()

# %% Cell 10: speed benchmark (ms per training step, same real batches)
bench_batches = [b for _, b in zip(range(BENCH_WARMUP + BENCH_STEPS),
                                   batches(train_ids, SEED + 99))]
bench_tokens = np.mean([(b[2] != PAD).sum() for b in bench_batches])


def time_engine(run_step, data, sync):
    """Median ms/step; `data` already in the engine's input format."""
    for b in data[:BENCH_WARMUP]:
        run_step(b)
    sync()
    runs = []
    for _ in range(BENCH_REPEATS):
        t0 = time.perf_counter()
        for b in data[BENCH_WARMUP:]:
            run_step(b)
        sync()
        runs.append((time.perf_counter() - t0) / BENCH_STEPS * 1e3)
    return float(np.median(runs))


results = {}
ht_step, ht_build_s = build_ht()
ms = time_engine(lambda b, st=ht_step: st(*b, lr=1e-4), bench_batches,
                 drv.synchronize)
results["houssem_torch TrainStep (fp16)"] = dict(
    ms=ms, setup_s=ht_build_s, mem_gb=drv.memory_in_use() / 2 ** 30)
del ht_step
free_gpu()

for mode in ("eager", "compile"):
    model, opt, scaler = build_torch(init_state)
    step = torch_step_fn(model, opt, scaler, mode)
    torch.cuda.reset_peak_memory_stats(torch_dev)
    t0 = time.perf_counter()
    step(to_torch(bench_batches[0]), 1e-4)        # includes compile time
    torch.cuda.synchronize(torch_dev)
    setup = time.perf_counter() - t0
    tb = [to_torch(b) for b in bench_batches]
    ms = time_engine(lambda b, st=step: st(b, 1e-4), tb,
                     lambda: torch.cuda.synchronize(torch_dev))
    results[f"PyTorch {mode} (AMP fp16)"] = dict(
        ms=ms, setup_s=setup,
        mem_gb=torch.cuda.max_memory_allocated(torch_dev) / 2 ** 30)
    del model, opt, scaler, step, tb
    free_gpu()

best_torch = min(v["ms"] for k, v in results.items() if k.startswith("PyTorch"))
print(f"\n{'engine':34s} {'ms/step':>9s} {'tok/s':>9s} {'vs best torch':>14s} "
      f"{'setup s':>8s} {'mem GB':>7s}")
for name, r in results.items():
    r["tok_s"] = bench_tokens / (r["ms"] / 1e3)
    r["speedup"] = best_torch / r["ms"]
    print(f"{name:34s} {r['ms']:9.1f} {r['tok_s']:9.0f} {r['speedup']:13.2f}x "
          f"{r['setup_s']:8.1f} {r['mem_gb']:7.2f}")

# %% Cell 11: train houssem_torch
def val_loss_ht(step):
    return float(np.mean([step.evaluate(*b) for b in val_batches]))


def train_ht():
    step, _ = build_ht()
    log, t0 = [], time.perf_counter()
    for i, b in enumerate(train_stream(TRAIN_STEPS), start=1):
        step(*b, lr=noam_lr(i, cfg.d_model, WARMUP_STEPS, LR_FACTOR))
        if i % LOG_EVERY == 0 or i == TRAIN_STEPS:
            loss, el = step.last_loss(), time.perf_counter() - t0
            log.append((i, loss, el))
            print(f"[houssem] step {i:6d}  loss {loss:.4f}  {el:7.1f}s  "
                  f"scale {step.stats()['loss_scale']:.0f}", flush=True)
    drv.synchronize()
    wall = time.perf_counter() - t0
    val = val_loss_ht(step)
    state = step.state_dict()
    np.savez(f"{WORK}/houssem_gpu_checkpoint.npz", **state)
    del step
    free_gpu()
    return dict(log=log, wall=wall, val=val, state=state)


ht_run = train_ht()
print(f"houssem_torch: {TRAIN_STEPS} steps in {ht_run['wall']:.0f}s, "
      f"val loss {ht_run['val']:.4f}")

# %% Cell 12: train PyTorch on the same batches
def train_torch(mode):
    model, opt, scaler = build_torch(init_state)
    step = torch_step_fn(model, opt, scaler, mode)
    log, t0 = [], time.perf_counter()
    for i, b in enumerate(train_stream(TRAIN_STEPS), start=1):
        loss = step(to_torch(b), noam_lr(i, cfg.d_model, WARMUP_STEPS, LR_FACTOR))
        if i % LOG_EVERY == 0 or i == TRAIN_STEPS:
            el, lv = time.perf_counter() - t0, loss.detach().item()
            log.append((i, lv, el))
            print(f"[torch {mode}] step {i:6d}  loss {lv:.4f}  "
                  f"{el:7.1f}s", flush=True)
    torch.cuda.synchronize(torch_dev)
    wall = time.perf_counter() - t0
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        val = float(np.mean([float(model.loss(*to_torch(b))) for b in val_batches]))
    torch.save(model.state_dict(), f"{WORK}/torch_checkpoint.pt")
    del model, opt, scaler, step
    free_gpu()
    return dict(log=log, wall=wall, val=val)


torch_run = train_torch(TORCH_TRAIN_MODE)
print(f"PyTorch {TORCH_TRAIN_MODE}: {TRAIN_STEPS} steps in {torch_run['wall']:.0f}s,"
      f" val loss {torch_run['val']:.4f}")

# %% Cell 13: a few translations from the houssem_torch weights (CPU, greedy)
cpu_model.load_state_dict(ht_run["state"])
sample = val_ids[:5]
src = np.full((len(sample), SRC_LEN), PAD, np.int64)
for i, (s, _) in enumerate(sample):
    src[i, :len(s)] = s
pred = cpu_model.translate_greedy(src, max_new_tokens=TGT_LEN)
translations = []
for (s, t), p in zip(sample, pred):
    p = list(p)
    p = p[:p.index(EOS)] if EOS in p else p
    row = (tok.decode(s[:-1]), tok.decode(t), tok.decode(p))
    translations.append(row)
    print(f"src : {row[0]}\nref : {row[1]}\nhyp : {row[2]}\n")

# %% Cell 14: report
lines = [
    "# houssem_torch TrainStep vs PyTorch",
    "",
    f"GPU: {drv.device_info(HT_GPU).name} (houssem_torch on cuda:{HT_GPU}, "
    f"PyTorch on cuda:{TORCH_GPU}), PyTorch {torch.__version__}",
    f"Model: transformer_base, vocab {VOCAB}, {cpu_model.num_parameters():,} "
    f"parameters. Batch {BATCH} x {SRC_LEN}/{TGT_LEN}, fp16 mixed precision.",
    f"Data: opus_books {SRC_LANG}->{TGT_LANG}, {len(train_ids):,} train pairs.",
    "",
    "## Speed (median of %d x %d steps, real batches, %.0f target tokens/batch)"
    % (BENCH_REPEATS, BENCH_STEPS, bench_tokens),
    "",
    "| engine | ms/step | tokens/s | vs best PyTorch | setup s | memory GB |",
    "|---|---:|---:|---:|---:|---:|",
]
for name, r in results.items():
    lines.append(f"| {name} | {r['ms']:.1f} | {r['tok_s']:.0f} | "
                 f"{r['speedup']:.2f}x | {r['setup_s']:.1f} | {r['mem_gb']:.2f} |")
lines += [
    "",
    f"## Training, {TRAIN_STEPS} steps each, same init and batches",
    "",
    "| engine | wall s | final train loss | val loss |",
    "|---|---:|---:|---:|",
    f"| houssem_torch TrainStep | {ht_run['wall']:.0f} | "
    f"{ht_run['log'][-1][1]:.4f} | {ht_run['val']:.4f} |",
    f"| PyTorch {TORCH_TRAIN_MODE} | {torch_run['wall']:.0f} | "
    f"{torch_run['log'][-1][1]:.4f} | {torch_run['val']:.4f} |",
    "",
    "| step | houssem loss | torch loss |",
    "|---:|---:|---:|",
]
for (i, a, _), (_, b, _) in zip(ht_run["log"], torch_run["log"]):
    lines.append(f"| {i} | {a:.4f} | {b:.4f} |")
lines += ["", "## Sample translations (houssem_torch weights, greedy)", ""]
for s, r, h in translations:
    lines += [f"- src: {s}", f"  - ref: {r}", f"  - hyp: {h}"]
report = "\n".join(lines) + "\n"
with open(f"{WORK}/cuda_vs_torch_report.md", "w", encoding="utf-8") as f:
    f.write(report)
print(report)
