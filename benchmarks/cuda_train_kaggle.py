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
# Outputs in /kaggle/working: houssem_vs_torch.png (the summary chart),
# cuda_vs_torch_report.md, checkpoints, tokenizer.

# %% Cell 1: settings
REPO_URL = "https://github.com/HoussemBouzamoucha/houssem_torch.git"
BRANCH = "gpu/gemm-tuning-buckets"
WORK = "/kaggle/working"
REPO_DIR = f"{WORK}/houssem_torch"

SRC_LANG, TGT_LANG = "fr", "en"
BATCH, SRC_LEN, TGT_LEN = 64, 64, 64      # fixed batch shape for both engines
VAL_FRACTION = 0.02
SEED = 0

BENCH_WARMUP, BENCH_STEPS, BENCH_REPEATS = 10, 30, 3
BENCH_BATCHES = [32, 64, 96]    # speed sweep (BATCH must be one of them)
PROFILE_STEPS = 5               # steps per engine for the time breakdown
TRAIN_STEPS = 3000              # per engine; ~1,900 steps = one epoch
WARMUP_STEPS, LR_FACTOR = 2000, 1.0     # Noam schedule
MAX_GRAD_NORM = 1.0
LOG_EVERY = 250
TORCH_TRAIN_MODE = "compile"    # PyTorch training run: "compile" or "eager"
LEN_BUCKETS = (16, 32, 48, 64)  # Cell 12b: length buckets (both lengths)

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
# A notebook keeps modules imported by an earlier run in memory: drop them so
# the code just pulled is what runs (otherwise git pull changes nothing).
for name in list(sys.modules):
    if (name == "houssem_torch" or name.startswith("houssem_torch.")
            or name in ("t4_profile", "trainstep_profile")):
        del sys.modules[name]

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
# One engine per T4 when there are two; HT_GPU=1 TORCH_GPU=0 swaps them to
# rule out card-to-card differences (clocks, cooling).
HT_GPU = int(os.environ.get("HT_GPU", 0))
TORCH_GPU = int(os.environ.get("TORCH_GPU", 1 if N_GPUS > 1 else 0))


def gpu_status(label):
    """Clocks, temperature, power and throttle reasons of every GPU."""
    print(f"-- GPU status {label}")
    # never fatal: the throttle-reasons field name varies by driver version
    subprocess.run("nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,"
                   "temperature.gpu,power.draw,clocks_throttle_reasons.active "
                   "--format=csv", shell=True)

print(drv.device_info(HT_GPU))
print("PyTorch", torch.__version__, "on cuda:%d" % TORCH_GPU)

# %% Cell 4: correctness tests on this GPU (GPU step vs CPU Transformer)
sh(f"{sys.executable} -m pytest -q -p no:warnings tests/test_cuda_driver.py "
   "tests/test_cuda_attention.py tests/test_cuda_train_step.py", cwd=REPO_DIR)

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


def to_batch(chunk, S=SRC_LEN, T=TGT_LEN):
    B = len(chunk)
    src = np.full((B, S), PAD, np.int64)
    tgt_in = np.full((B, T), PAD, np.int64)
    tgt_out = np.full((B, T), PAD, np.int64)
    for i, (s, t) in enumerate(chunk):
        src[i, :len(s)] = s
        tgt_in[i, 0], tgt_in[i, 1:len(t) + 1] = BOS, t
        tgt_out[i, :len(t)], tgt_out[i, len(t)] = t, EOS
    return src, tgt_in, tgt_out


def batches(data, epoch_seed, B=BATCH):
    """Full batches only (the GPU step has a fixed shape), shuffled per epoch."""
    idx = np.random.default_rng(epoch_seed).permutation(len(data))
    for i in range(0, len(idx) - B + 1, B):
        yield to_batch([data[j] for j in idx[i:i + B]])


class Timed:
    """Iterator wrapper adding up the host time spent producing items."""

    def __init__(self, it):
        self.it, self.s = iter(it), 0.0

    def __iter__(self):
        return self

    def __next__(self):
        t = time.perf_counter()
        try:
            return next(self.it)
        finally:
            self.s += time.perf_counter() - t


def sm_clock(idx):
    """Current SM clock of GPU idx, e.g. '1590 MHz' ('?' if unavailable)."""
    r = subprocess.run(f"nvidia-smi -i {idx} --query-gpu=clocks.sm "
                       "--format=csv,noheader", shell=True,
                       capture_output=True, text=True)
    return r.stdout.strip() or "?"


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
from houssem_torch.backends.cuda.train_step import BucketedTrainStep, TrainStep
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


def torch_step_fn(model, opt, scaler, mode, dynamic=None):
    if mode != "eager":
        torch._dynamo.reset()      # fresh compile for each model / shape
    loss_fn = model.loss if mode == "eager" else torch.compile(model.loss,
                                                               dynamic=dynamic)

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


def build_ht(B=BATCH):
    cpu_model.load_state_dict(init_state)
    t0 = time.perf_counter()
    step = TrainStep(cpu_model, B, SRC_LEN, TGT_LEN, precision="fp16",
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
kinds = {"encoder self-attention": (SRC_LEN, SRC_LEN, 0),
         "decoder self-attention": (TGT_LEN, TGT_LEN, 1),
         "cross-attention": (TGT_LEN, SRC_LEN, 0)}
print("houssem_torch attention: " + ", ".join(
    f"{k} {'fused' if v in ht_step.k_attn else 'UNFUSED'}" for k, v in kinds.items()))
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

# %% Cell 10: speed benchmark: ms per training step, several batch sizes
# Same real batches for every engine. torch.compile recompiles for each batch
# size (its first step, "setup", includes that).
HT, TC, TE = "houssem_torch", "torch.compile", "PyTorch eager"
ENGINES = (HT, TC, TE)


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


def torch_sync():
    torch.cuda.synchronize(torch_dev)


sweep = {}          # sweep[B][engine] = dict(ms, tok_s, setup_s, mem_gb)
bench_data = {}
for B in BENCH_BATCHES:
    data = [b for _, b in zip(range(BENCH_WARMUP + BENCH_STEPS),
                              batches(train_ids, SEED + 99, B))]
    bench_data[B] = data
    tokens = float(np.mean([(b[2] != PAD).sum() for b in data]))
    row = sweep[B] = {}
    step, setup = build_ht(B)
    ms = time_engine(lambda b, st=step: st(*b, lr=1e-4), data, drv.synchronize)
    row[HT] = dict(ms=ms, setup_s=setup, mem_gb=drv.memory_in_use() / 2 ** 30)
    del step
    free_gpu()
    for name, mode in ((TE, "eager"), (TC, "compile")):
        model, opt, scaler = build_torch(init_state)
        step = torch_step_fn(model, opt, scaler, mode)
        tb = [to_torch(b) for b in data]
        torch.cuda.reset_peak_memory_stats(torch_dev)
        t0 = time.perf_counter()
        step(tb[0], 1e-4)                         # includes compile time
        torch_sync()
        setup = time.perf_counter() - t0
        ms = time_engine(lambda b, st=step: st(b, 1e-4), tb, torch_sync)
        row[name] = dict(ms=ms, setup_s=setup,
                         mem_gb=torch.cuda.max_memory_allocated(torch_dev) / 2 ** 30)
        del model, opt, scaler, step, tb
        free_gpu()
    for r in row.values():
        r["tok_s"] = tokens / (r["ms"] / 1e3)
    print(f"batch {B:4d}: " + "  ".join(f"{e} {row[e]['ms']:.1f} ms"
                                        for e in ENGINES), flush=True)
results = sweep[BATCH]
bench_tokens = float(np.mean([(b[2] != PAD).sum() for b in bench_data[BATCH]]))

# %% Cell 10b: where the time goes (same categories for every engine)
# Each engine's GPU time is grouped into the same kinds of work, then scaled
# to its measured (unprofiled) ms/step, so its bars add up to its step time.
import trainstep_profile as htp

BUCKETS = ("matrix multiplies", "attention", "LayerNorm + residual",
           "activation, dropout, bias", "loss", "optimizer", "embedding, other")


def bucket_houssem(cat):
    if "attention" in cat or "head split" in cat:
        return "attention"
    if cat.startswith("matmul"):
        return "matrix multiplies"
    if cat.startswith("residual"):
        return "LayerNorm + residual"
    if cat.startswith("FFN"):
        return "activation, dropout, bias"
    if cat.startswith("loss"):
        return "loss"
    if cat.startswith("optimizer"):
        return "optimizer"
    return "embedding, other"


def bucket_torch(cat):
    if cat == "matmul":
        return "matrix multiplies"
    if cat in ("attention", "softmax"):
        return "attention"
    if cat == "layernorm":
        return "LayerNorm + residual"
    if cat.startswith("dropout") or cat.startswith("other"):
        return "activation, dropout, bias"
    if cat.startswith("loss"):
        return "loss"
    if cat.startswith("optimizer"):
        return "optimizer"
    return "embedding, other"


def scaled(raw, ms):
    total = sum(raw.values()) or 1.0
    return {k: raw.get(k, 0.0) / total * ms for k in BUCKETS}


breakdown = {}
data = bench_data[BATCH]
step, _ = build_ht()
for b in data[:BENCH_WARMUP]:
    step(*b, lr=1e-4)
with drv.Profiler() as prof:
    for b in data[:PROFILE_STEPS]:
        step(*b, lr=1e-4)
raw = {}
for r in prof.records():
    k = bucket_houssem(htp.category(r))
    raw[k] = raw.get(k, 0.0) + r.ms / PROFILE_STEPS
breakdown[HT] = scaled(raw, results[HT]["ms"])
del step, prof
free_gpu()
for name, mode in ((TC, "compile"), (TE, "eager")):
    model, opt, scaler = build_torch(init_state)
    step = torch_step_fn(model, opt, scaler, mode)
    tb = to_torch(data[0])
    for _ in range(3):
        step(tb, 1e-4)
    prof = tp.profile_steps(lambda st=step, tb=tb: st(tb, 1e-4), PROFILE_STEPS,
                            f"{WORK}/trace_{mode}.json")
    raw = {}
    for cat, ms in (prof or {}).get("categories", {}).items():
        k = bucket_torch(cat)
        raw[k] = raw.get(k, 0.0) + ms
    breakdown[name] = scaled(raw, results[name]["ms"])
    del model, opt, scaler, step, tb
    free_gpu()
print(f"{'ms per step':28s}" + "".join(f"{e:>16s}" for e in ENGINES))
for k in BUCKETS:
    print(f"{k:28s}" + "".join(f"{breakdown[e][k]:16.1f}" for e in ENGINES))

# %% Cell 11: train houssem_torch
def val_loss_ht(step):
    return float(np.mean([step.evaluate(*b) for b in val_batches]))


def train_ht(build=build_ht, make_stream=None, tag="houssem", save=True):
    step, setup = build()
    log, t0 = [], time.perf_counter()
    stream = Timed((make_stream or train_stream)(TRAIN_STEPS))
    for i, b in enumerate(stream, start=1):
        step(*b, lr=noam_lr(i, cfg.d_model, WARMUP_STEPS, LR_FACTOR))
        if i % LOG_EVERY == 0 or i == TRAIN_STEPS:
            loss, el = step.last_loss(), time.perf_counter() - t0
            log.append((i, loss, el))
            print(f"[{tag}] step {i:6d}  loss {loss:.4f}  {el:7.1f}s  "
                  f"data {stream.s:5.1f}s  {sm_clock(HT_GPU)}  "
                  f"scale {step.stats()['loss_scale']:.0f}", flush=True)
    drv.synchronize()
    wall = time.perf_counter() - t0
    val = val_loss_ht(step)
    state = step.state_dict()
    mem = drv.memory_in_use() / 2 ** 30
    if save:
        np.savez(f"{WORK}/houssem_gpu_checkpoint.npz", **state)
    del step
    free_gpu()
    return dict(log=log, wall=wall, val=val, state=state, setup=setup, mem_gb=mem)


gpu_status("before houssem_torch training")
ht_run = train_ht()
gpu_status("after houssem_torch training")
print(f"houssem_torch: {TRAIN_STEPS} steps in {ht_run['wall']:.0f}s, "
      f"val loss {ht_run['val']:.4f}")

# %% Cell 12: train PyTorch on the same batches
def train_torch(mode, make_stream=None, dynamic=None, save=True):
    model, opt, scaler = build_torch(init_state)
    step = torch_step_fn(model, opt, scaler, mode, dynamic)
    torch.cuda.reset_peak_memory_stats(torch_dev)
    log, t0 = [], time.perf_counter()
    stream = Timed((make_stream or train_stream)(TRAIN_STEPS))
    for i, b in enumerate(stream, start=1):
        loss = step(to_torch(b), noam_lr(i, cfg.d_model, WARMUP_STEPS, LR_FACTOR))
        if i % LOG_EVERY == 0 or i == TRAIN_STEPS:
            el, lv = time.perf_counter() - t0, loss.detach().item()
            log.append((i, lv, el))
            print(f"[torch {mode}] step {i:6d}  loss {lv:.4f}  "
                  f"{el:7.1f}s  data {stream.s:5.1f}s  "
                  f"{sm_clock(TORCH_GPU)}", flush=True)
    torch.cuda.synchronize(torch_dev)
    wall = time.perf_counter() - t0
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        val = float(np.mean([float(model.loss(*to_torch(b))) for b in val_batches]))
    if save:
        torch.save(model.state_dict(), f"{WORK}/torch_checkpoint.pt")
    mem = torch.cuda.max_memory_allocated(torch_dev) / 2 ** 30
    del model, opt, scaler, step
    free_gpu()
    return dict(log=log, wall=wall, val=val, mem_gb=mem)


gpu_status("before PyTorch training")
torch_run = train_torch(TORCH_TRAIN_MODE)
gpu_status("after PyTorch training")
print(f"PyTorch {TORCH_TRAIN_MODE}: {TRAIN_STEPS} steps in {torch_run['wall']:.0f}s,"
      f" val loss {torch_run['val']:.4f}")

# %% Cell 12b: length buckets: the same sentences, batched by length
# Above, every batch is padded to 64/64 tokens (mostly padding). Here each
# pair goes to the smallest bucket in LEN_BUCKETS that holds it, batches are
# made within a bucket (still BATCH pairs each), and each runs at its
# bucket's width. houssem_torch: BucketedTrainStep (one compiled step per
# bucket, shared weights). PyTorch: torch.compile(dynamic=False), one graph
# per bucket. Both see the same batches in the same order.
def bucket_of(pair):
    need = max(len(pair[0]), len(pair[1]) + 1)
    return next(L for L in LEN_BUCKETS if L >= need)


def bucketed_batches(data, epoch_seed, B=BATCH):
    rng_ = np.random.default_rng(epoch_seed)
    groups = {L: [] for L in LEN_BUCKETS}
    for j in rng_.permutation(len(data)):
        groups[bucket_of(data[j])].append(j)
    made = [(L, idx[i:i + B]) for L, idx in groups.items()
            for i in range(0, len(idx) - B + 1, B)]
    for k in rng_.permutation(len(made)):
        L, idx = made[k]
        yield to_batch([data[j] for j in idx], L, L)


def bucketed_stream(n_steps):
    epoch, made = 0, 0
    while made < n_steps:
        for b in bucketed_batches(train_ids, SEED + epoch):
            yield b
            made += 1
            if made == n_steps:
                return
        epoch += 1


def build_ht_bucketed():
    cpu_model.load_state_dict(init_state)
    t0 = time.perf_counter()
    step = BucketedTrainStep(cpu_model, BATCH, [(L, L) for L in LEN_BUCKETS],
                             precision="fp16", max_grad_norm=MAX_GRAD_NORM,
                             seed=SEED, device=HT_GPU, **ADAM)
    drv.synchronize()
    return step, time.perf_counter() - t0


def steady_ms(log):
    """ms/step between the first and last log points (no compile/setup)."""
    (i0, _, t0_), (i1, _, t1_) = log[0], log[-1]
    return (t1_ - t0_) / (i1 - i0) * 1e3


widths = [b[0].shape[1] for b in bucketed_stream(TRAIN_STEPS)]
fill_fixed = np.mean([(b[2] != PAD).mean() for b in val_batches])
print("bucket share of the training batches: " + ", ".join(
    f"{L}: {widths.count(L) / len(widths):.0%}" for L in LEN_BUCKETS)
    + f"  (fixed 64/64 batches are {fill_fixed:.0%} real target tokens)")
gpu_status("before bucketed training")
ht_bucket_run = train_ht(build_ht_bucketed, bucketed_stream, "houssem buckets",
                         save=False)
torch_bucket_run = train_torch("compile", bucketed_stream, dynamic=False,
                               save=False)
gpu_status("after bucketed training")
bucket_runs = {
    "buckets": list(LEN_BUCKETS),
    "share": {str(L): widths.count(L) / len(widths) for L in LEN_BUCKETS},
    HT: {k: ht_bucket_run[k] for k in ("log", "wall", "val", "setup", "mem_gb")},
    TC: {k: torch_bucket_run[k] for k in ("log", "wall", "val", "mem_gb")},
}
for name, run_fixed, run_b in ((HT, ht_run, ht_bucket_run),
                               (TC, torch_run, torch_bucket_run)):
    bucket_runs[name]["steady_ms"] = steady_ms(run_b["log"])
    print(f"{name:14s} buckets: {run_b['wall']:6.0f} s, "
          f"{steady_ms(run_b['log']):5.1f} ms/step steady, val {run_b['val']:.4f}"
          f"  | fixed 64/64: {run_fixed['wall']:6.0f} s, "
          f"{steady_ms(run_fixed['log']):5.1f} ms/step, val {run_fixed['val']:.4f}")
print(f"houssem_torch bucketed setup {ht_bucket_run['setup']:.1f} s, memory "
      f"{ht_bucket_run['mem_gb']:.2f} GB; torch.compile bucketed peak memory "
      f"{torch_bucket_run['mem_gb']:.2f} GB")

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

# %% Cell 14: scoreboard and charts
import matplotlib.pyplot as plt

ht_ms, tc_ms, te_ms = (results[e]["ms"] for e in ENGINES)
gpu_name = drv.device_info(HT_GPU).name
print("=" * 76)
print(f"  houssem_torch vs PyTorch: {gpu_name}, transformer_base, batch {BATCH}, fp16")
print("=" * 76)
print(f"  Speed      houssem_torch {ht_ms:6.1f} ms/step | torch.compile {tc_ms:6.1f}"
      f" | eager {te_ms:6.1f}")
print(f"             -> houssem_torch is {tc_ms / ht_ms:.2f}x torch.compile's speed"
      f" and {te_ms / ht_ms:.2f}x eager's")
print(f"  Startup    houssem_torch {results[HT]['setup_s']:5.1f} s | torch.compile "
      f"{results[TC]['setup_s']:5.1f} s (compiling)")
print(f"  Memory     houssem_torch {results[HT]['mem_gb']:.2f} GB | torch.compile "
      f"{results[TC]['mem_gb']:.2f} GB")
print(f"  Training   {TRAIN_STEPS} steps, same data: houssem_torch "
      f"{ht_run['wall']:.0f} s, val loss {ht_run['val']:.3f} | PyTorch "
      f"{TORCH_TRAIN_MODE} {torch_run['wall']:.0f} s, val loss {torch_run['val']:.3f}")
print("=" * 76)

# Light chart theme; one fixed color per engine in every panel.
SURF, INK, INK2, MUTED, GRID, BASE = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                      "#e1e0d9", "#c3c2b7")
COLOR = {HT: "#2a78d6", TC: "#eb6834", TE: "#1baf7a"}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": INK2, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def xgrid(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f"houssem_torch vs PyTorch on a {gpu_name}: transformer_base "
             f"({cpu_model.num_parameters() / 1e6:.1f}M parameters), fp16, "
             f"opus_books {SRC_LANG}->{TGT_LANG}",
             x=0.01, ha="left", fontsize=14, fontweight="bold")

# (a) time per step
ax = axs[0, 0]
order = [TE, TC, HT]                                 # barh draws bottom-up
vals = [results[e]["ms"] for e in order]
ax.barh(order, vals, color=[COLOR[e] for e in order], height=0.55)
for y, (e, v) in enumerate(zip(order, vals)):
    note = "" if e == HT else f"   ({v / ht_ms:.2f}x houssem_torch's time)"
    ax.text(v + max(vals) * 0.01, y, f"{v:.1f} ms{note}", va="center",
            color=INK2, fontsize=9)
ax.set_xlim(0, max(vals) * 1.6)
ax.set_xlabel("milliseconds per training step (lower is better)")
ax.set_title(f"Time per training step, batch {BATCH}")
xgrid(ax)

# (b) where the time goes: grouped bars per kind of work
ax = axs[0, 1]
rows = list(BUCKETS)[::-1]
h = 0.26
for i, e in enumerate(ENGINES):
    ys = [r + (1 - i) * (h + 0.02) for r in range(len(rows))]
    v = [breakdown[e][k] for k in rows]
    ax.barh(ys, v, height=h, color=COLOR[e], label=e)
    for y, val in zip(ys, v):
        if val >= 0.5:
            ax.text(val + 0.4, y, f"{val:.1f}", va="center", fontsize=7.5,
                    color=MUTED)
ax.set_yticks(range(len(rows)), rows)
ax.set_xlim(0, max(max(breakdown[e].values()) for e in ENGINES) * 1.12)
ax.set_xlabel("milliseconds per step (each engine's bars add up to its step time)")
ax.set_title("Where the time goes")
ax.legend(loc="lower right")
xgrid(ax)

# (c) throughput vs batch size
ax = axs[1, 0]
for e in ENGINES:
    tok = [sweep[B][e]["tok_s"] for B in BENCH_BATCHES]
    ax.plot(BENCH_BATCHES, tok, color=COLOR[e], linewidth=2, marker="o",
            markersize=8, markeredgecolor=SURF, markeredgewidth=2, label=e)
    ax.annotate(e, (BENCH_BATCHES[-1], tok[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK2)
ax.set_xticks(BENCH_BATCHES)
ax.set_xlim(BENCH_BATCHES[0] - 8, BENCH_BATCHES[-1] + 30)
ax.set_xlabel("batch size (sentences)")
ax.set_ylabel("target tokens per second (higher is better)")
ax.set_title("Throughput vs batch size")
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.legend(loc="lower right")

# (d) training loss against wall-clock time
ax = axs[1, 1]
torch_name_ = TC if TORCH_TRAIN_MODE == "compile" else TE
for e, run in ((HT, ht_run), (torch_name_, torch_run)):
    secs = [t for _, _, t in run["log"]]
    loss = [lv for _, lv, _ in run["log"]]
    ax.plot(secs, loss, color=COLOR[e], linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURF, markeredgewidth=1.5,
            label=f"{e}: {run['wall']:.0f} s, val loss {run['val']:.3f}")
ax.set_xlabel("wall-clock seconds (same batches, same order)")
ax.set_ylabel("training loss")
ax.set_title(f"Training: {TRAIN_STEPS} steps each")
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.legend(loc="upper right")

fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{WORK}/houssem_vs_torch.png", dpi=150)
plt.show()

# %% Cell 15: report (markdown, with the chart)
lines = [
    "# houssem_torch TrainStep vs PyTorch",
    "",
    "![summary](houssem_vs_torch.png)",
    "",
    f"GPU: {gpu_name} (houssem_torch on cuda:{HT_GPU}, PyTorch on cuda:{TORCH_GPU}),"
    f" PyTorch {torch.__version__}",
    f"Model: transformer_base, vocab {VOCAB}, {cpu_model.num_parameters():,} "
    f"parameters. Batch {BATCH} x {SRC_LEN}/{TGT_LEN}, fp16 mixed precision.",
    f"Data: opus_books {SRC_LANG}->{TGT_LANG}, {len(train_ids):,} train pairs.",
    "",
    "## Speed (median of %d x %d steps, real batches, %.0f target tokens/batch)"
    % (BENCH_REPEATS, BENCH_STEPS, bench_tokens),
    "",
    "| engine | ms/step | tokens/s | time vs houssem_torch | setup s | memory GB |",
    "|---|---:|---:|---:|---:|---:|",
]
for name in ENGINES:
    r = results[name]
    lines.append(f"| {name} | {r['ms']:.1f} | {r['tok_s']:.0f} | "
                 f"{r['ms'] / ht_ms:.2f}x | {r['setup_s']:.1f} | {r['mem_gb']:.2f} |")
lines += ["", "## By batch size (ms/step)", "",
          "| batch | " + " | ".join(ENGINES) + " | houssem_torch speedup vs "
          "torch.compile |", "|---:|" + "---:|" * (len(ENGINES) + 1)]
for B in BENCH_BATCHES:
    lines.append(f"| {B} | " + " | ".join(f"{sweep[B][e]['ms']:.1f}" for e in ENGINES)
                 + f" | {sweep[B][TC]['ms'] / sweep[B][HT]['ms']:.2f}x |")
lines += ["", "## Where the time goes (ms/step, scaled to each engine's step time)",
          "", "| work | " + " | ".join(ENGINES) + " |",
          "|---|" + "---:|" * len(ENGINES)]
for k in BUCKETS:
    lines.append(f"| {k} | " + " | ".join(f"{breakdown[e][k]:.1f}" for e in ENGINES)
                 + " |")
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
lines += ["", f"## Length buckets {list(LEN_BUCKETS)}, {TRAIN_STEPS} steps each, "
          "same batches", "",
          "Bucket share: " + ", ".join(f"{L}: {v:.0%}" for L, v in
                                       bucket_runs["share"].items()), "",
          "| engine | wall s | steady ms/step | val loss | fixed 64/64 wall s | "
          "fixed ms/step | fixed val |", "|---|---:|---:|---:|---:|---:|---:|"]
for name, run_fixed, run_b in ((HT, ht_run, ht_bucket_run),
                               (TC, torch_run, torch_bucket_run)):
    lines.append(f"| {name} | {run_b['wall']:.0f} | {steady_ms(run_b['log']):.1f}"
                 f" | {run_b['val']:.4f} | {run_fixed['wall']:.0f} | "
                 f"{steady_ms(run_fixed['log']):.1f} | {run_fixed['val']:.4f} |")
lines += ["", "## Sample translations (houssem_torch weights, greedy)", ""]
for s_, r_, h_ in translations:
    lines += [f"- src: {s_}", f"  - ref: {r_}", f"  - hyp: {h_}"]
report = "\n".join(lines) + "\n"
with open(f"{WORK}/cuda_vs_torch_report.md", "w", encoding="utf-8") as f:
    f.write(report)
print(report)
