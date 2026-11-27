# ruff: noqa: E402  (notebook cells: each cell imports what it uses)
# %% [markdown]
# # Training quality on opus-100 fr->en: big-batch learning-rate schedules
#
# opus_books (116k pairs) overfits at x4 tokens per step after ~13 epochs.
# This script moves to opus-100 en-fr (1M training pairs, official 2k
# validation / 2k test sets) and compares, at large batches, the batch-64
# schedule against learning rates scaled for the batch:
#   RUNS: (name, micro-batches per step, tokens per micro-batch as a
#          multiple of one 64-pair batch, Noam factor)
# x8 is two accumulated x4 micro-batches (accumulate_step). Every run starts
# from the same weights and sees the batches in the same order.
#
# Quality: validation loss every VAL_EVERY steps, and BLEU (sacrebleu) on
# the 2k test sentences with houssem_torch's own GPU decoder (greedy, and
# beam search at the end). houssem_torch only: the engine comparison with
# PyTorch is done.
#
# Kaggle: Accelerator "GPU T4 x2", Internet "On". Long: use Save Version ->
# Save & Run All, or run cell by cell.
# Outputs: /kaggle/working/quality_results_<date>/ and a .zip of it.

# %% Cell 1: settings
REPO_URL = "https://github.com/HoussemBouzamoucha/houssem_torch.git"
BRANCH = "train/quality"
WORK = "/kaggle/working"
REPO_DIR = f"{WORK}/houssem_torch"

SRC_LANG, TGT_LANG = "fr", "en"
BASE_BATCH, MAX_LEN = 64, 64      # one "batch 64" of tokens; longest sentence
LEN_BUCKETS = (16, 32, 48, 64)
TRAIN_STEPS = 10000               # optimizer steps per run
WARMUP_STEPS = 2000
RUNS = [                          # name, micro-batches, scale, Noam factor
    ("x4 lr1", 1, 4, 1.0),
    ("x4 lr2", 1, 4, 2.0),
    ("x8 lr2.8", 2, 4, 2.83),
]
VAL_EVERY = 1000
BLEU_EVERY = 5000                 # greedy BLEU on the test set (and at the end)
BEAM, ALPHA = 4, 0.6              # final beam-search BLEU
MAX_DECODE = 96                   # longest output, tokens
DECODE_ROWS = 128                 # Translator batch (beam: 32 sentences)
MAX_GRAD_NORM = 1.0
SEED = 0

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
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
for name in list(sys.modules):   # use the code just pulled
    if name == "houssem_torch" or name.startswith("houssem_torch."):
        del sys.modules[name]

try:
    import cupy  # noqa: F401
except ImportError:
    sh(f"{sys.executable} -m pip install -q cupy-cuda12x nvidia-cublas-cu12 "
       "nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12")
for mod in ("tokenizers", "datasets", "sacrebleu"):
    try:
        __import__(mod)
    except ImportError:
        sh(f"{sys.executable} -m pip install -q {mod}")

import gc
import json
import time

import cupy as cp
import numpy as np

from houssem_torch.backends.cuda import driver as drv

assert drv.is_available(), "CuPy cannot see a GPU: enable 'GPU T4 x2'"
GPU = int(os.environ.get("HT_GPU",
                         1 if cp.cuda.runtime.getDeviceCount() > 1 else 0))
print(drv.device_info(GPU))
sh(f"CUDA_VISIBLE_DEVICES={GPU} {sys.executable} -m pytest -q -p no:warnings "
   "tests/test_cuda_decoder.py tests/test_cuda_train_step.py", cwd=REPO_DIR)

# %% Cell 3: data, tokenizer, buckets
from datasets import load_dataset
import sacrebleu
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from tokenizers.trainers import BpeTrainer

from houssem_torch.models import TransformerConfig

t0 = time.perf_counter()
ds = load_dataset("Helsinki-NLP/opus-100", "en-fr")


def pairs_of(split):
    return [(r["translation"][SRC_LANG], r["translation"][TGT_LANG])
            for r in ds[split]]


train_pairs = [p for p in pairs_of("train") if p[0].strip() and p[1].strip()]
val_pairs, test_pairs = pairs_of("validation"), pairs_of("test")

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
assert [tok.token_to_id(s) for s in SPECIALS[:3]] == [PAD, BOS, EOS]


def encode(pairs_):
    src = tok.encode_batch([p[0] for p in pairs_])
    tgt = tok.encode_batch([p[1] for p in pairs_])
    return [(s.ids + [EOS], t.ids) for s, t in zip(src, tgt)
            if len(s.ids) + 1 <= MAX_LEN and len(t.ids) + 1 <= MAX_LEN]


train_ids, val_ids = encode(train_pairs), encode(val_pairs)
# Test sources: every sentence (BLEU uses all 2000), cut to MAX_LEN.
test_src = [e.ids[:MAX_LEN - 1] + [EOS]
            for e in tok.encode_batch([p[0] for p in test_pairs])]
test_refs = [p[1] for p in test_pairs]


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
# Token budget: one 64-pair batch of opus_books' ~23 target tokens per pair,
# so "x4" means the same tokens per step as in the opus_books runs.
BASE_TOKENS = BASE_BATCH * 23.0


def bucket_batch(scale):
    """Pairs per micro-batch for each bucket: ~scale x BASE_TOKENS tokens."""
    return {L: max(8, int(round(scale * BASE_TOKENS / np.mean(v) / 8)) * 8)
            for L, v in lens.items() if v}


def bucketed_stream(scale):
    """Micro-batches in a fixed order (the same for every run)."""
    bb = bucket_batch(scale)
    epoch = 0
    while True:
        rng_ = np.random.default_rng(SEED + epoch)
        groups = {L: [] for L in bb}
        for j in rng_.permutation(len(train_ids)):
            groups[bucket_of(train_ids[j])].append(j)
        chunks = [(L, idx[i:i + bb[L]]) for L, idx in groups.items()
                  for i in range(0, len(idx) - bb[L] + 1, bb[L])]
        for k in rng_.permutation(len(chunks)):
            L, idx = chunks[k]
            yield to_batch([train_ids[j] for j in idx], L, L)
        epoch += 1


vidx = np.random.default_rng(SEED).permutation(len(val_ids))
val_batches = [to_batch([val_ids[j] for j in vidx[i:i + BASE_BATCH]], MAX_LEN,
                        MAX_LEN)
               for i in range(0, len(vidx) - BASE_BATCH + 1, BASE_BATCH)]
train_tokens = sum(len(t) + 1 for _, t in train_ids)
print(f"{len(train_ids):,} train pairs ({train_tokens / 1e6:.1f}M target tokens, "
      f"{mean_tokens:.1f} per pair), {len(val_batches)} val batches, "
      f"{len(test_src)} test sentences; data ready in "
      f"{time.perf_counter() - t0:.0f} s")
for name, micro, scale, factor in RUNS:
    print(f"{name}: {micro} x {bucket_batch(scale)} pairs per micro-batch")

# %% Cell 4: model, training step, evaluation
import houssem_torch as ht
from houssem_torch.backends.cuda.decoder import Translator
from houssem_torch.backends.cuda.train_step import BucketedTrainStep
from houssem_torch.models import Transformer
from houssem_torch.optim import noam_lr

cfg = TransformerConfig.from_dict({**base_cfg.to_dict(), "vocab_size": VOCAB,
                                   "max_len": max(MAX_LEN, base_cfg.max_len)})
ht.manual_seed(SEED)
cpu_model = Transformer(cfg)
init_state = cpu_model.state_dict()
ADAM = dict(betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)


def free_gpu():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


def make_step(scale):
    cpu_model.load_state_dict(init_state)
    bb = bucket_batch(scale)
    shapes = [(bb[L], L, L) for L in bb] + [(BASE_BATCH, MAX_LEN, MAX_LEN)]
    return BucketedTrainStep(cpu_model, BASE_BATCH, shapes, precision="fp16",
                             max_grad_norm=MAX_GRAD_NORM, seed=SEED, device=GPU,
                             **ADAM)


def val_loss(step):
    return float(np.mean([step.evaluate(*b) for b in val_batches]))


def detok(ids):
    ids = [int(t) for t in ids]
    if EOS in ids:
        ids = ids[:ids.index(EOS)]
    return tok.decode([t for t in ids if t != PAD])


def translate_test(tr, beam):
    per_call = DECODE_ROWS // beam
    order = np.argsort([len(s) for s in test_src], kind="stable")  # less padding
    out = [None] * len(test_src)
    for i in range(0, len(order), per_call):
        idx = order[i:i + per_call]
        width = max(len(test_src[j]) for j in idx)
        src = np.full((len(idx), width), PAD, np.int64)
        for r, j in enumerate(idx):
            src[r, :len(test_src[j])] = test_src[j]
        ids = (tr.translate_greedy(src, max_new_tokens=MAX_DECODE) if beam == 1
               else tr.translate_beam(src, beam=beam, alpha=ALPHA,
                                      max_new_tokens=MAX_DECODE))
        for r, j in enumerate(idx):
            out[j] = detok(ids[r])
    return out


def bleu(tr, beam=1):
    t = time.perf_counter()
    hyps = translate_test(tr, beam)
    score = sacrebleu.corpus_bleu(hyps, [test_refs]).score
    return score, hyps, time.perf_counter() - t


# %% Cell 5: train every run
def run(name, micro, scale, factor):
    t0 = time.perf_counter()
    step = make_step(scale)
    tr = Translator(step, DECODE_ROWS, MAX_LEN)
    drv.synchronize()
    setup = time.perf_counter() - t0
    stream = bucketed_stream(scale)
    curve, bleus, tokens, eval_time = [], [], 0, 0.0
    t_start = time.perf_counter()
    for i in range(1, TRAIN_STEPS + 1):
        batches = [next(stream) for _ in range(micro)]
        tokens += sum(int((b[2] != PAD).sum()) for b in batches)
        lr = noam_lr(i, cfg.d_model, WARMUP_STEPS, factor)
        if micro == 1:
            step(*batches[0], lr=lr)
        else:
            step.accumulate_step(batches, lr)
        if i % VAL_EVERY == 0 or i == TRAIN_STEPS:
            drv.synchronize()
            te = time.perf_counter()
            el = te - t_start - eval_time
            v = val_loss(step)
            line = (f"[{name}] step {i:5d}  {el:7.1f}s  train {step.last_loss():.3f}"
                    f"  val {v:.4f}  lr {lr:.2e}")
            curve.append((i, el, v, step.last_loss()))
            if i % BLEU_EVERY == 0 or i == TRAIN_STEPS:
                b, _, bt = bleu(tr)
                bleus.append((i, el, b))
                line += f"  BLEU greedy {b:.2f} ({bt:.0f} s)"
            eval_time += time.perf_counter() - te
            print(line, flush=True)
    wall = time.perf_counter() - t_start - eval_time
    greedy_bleu, greedy_hyps, _ = bleu(tr)
    beam_bleu, beam_hyps, beam_s = bleu(tr, BEAM)
    stats = step.stats()
    res = dict(name=name, micro=micro, scale=scale, factor=factor,
               pairs_per_batch=bucket_batch(scale), setup_s=setup, wall_s=wall,
               ms_per_step=wall / TRAIN_STEPS * 1e3, tokens_per_s=tokens / wall,
               tokens_per_step=tokens / TRAIN_STEPS,
               epochs=tokens / train_tokens, val_curve=curve, bleu_curve=bleus,
               val=curve[-1][2], bleu_greedy=greedy_bleu, bleu_beam=beam_bleu,
               beam_seconds=beam_s, loss_scale=stats["loss_scale"],
               mem_gb=drv.memory_in_use() / 2 ** 30,
               samples=[(test_pairs[j][0], test_refs[j], greedy_hyps[j],
                         beam_hyps[j]) for j in range(0, len(test_refs), 400)])
    del tr, step
    free_gpu()
    print(f"== {name}: {wall:.0f} s, {res['ms_per_step']:.1f} ms/step, "
          f"{res['epochs']:.1f} epochs, val {res['val']:.4f}, BLEU greedy "
          f"{greedy_bleu:.2f} / beam{BEAM} {beam_bleu:.2f}, "
          f"{res['mem_gb']:.2f} GB", flush=True)
    return res


results = [run(*r) for r in RUNS]

# %% Cell 6: report, chart, download
import matplotlib.pyplot as plt

stamp = time.strftime("%Y%m%d_%H%M")
out = f"{WORK}/quality_results_{stamp}"
os.makedirs(out, exist_ok=True)
gpu_name = drv.device_info(GPU).name
lines = [f"# Training quality on opus-100 {SRC_LANG}->{TGT_LANG} ({gpu_name})", "",
         f"{len(train_ids):,} training pairs, transformer_base, fp16, buckets "
         f"{list(LEN_BUCKETS)}, {TRAIN_STEPS} optimizer steps per run, Noam "
         f"warmup {WARMUP_STEPS}. x1 = {BASE_TOKENS:.0f} target tokens per step. "
         f"BLEU: sacrebleu on the {len(test_refs)} test sentences, our GPU "
         f"decoder (greedy; beam {BEAM}, alpha {ALPHA}).", "",
         "| run | tokens / step | epochs | ms / step | val loss | BLEU greedy "
         f"| BLEU beam {BEAM} | wall s | memory GB |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in results:
    lines.append(f"| {r['name']} | {r['tokens_per_step']:,.0f} | {r['epochs']:.1f} | "
                 f"{r['ms_per_step']:.1f} | {r['val']:.4f} | {r['bleu_greedy']:.2f} | "
                 f"{r['bleu_beam']:.2f} | {r['wall_s']:.0f} | {r['mem_gb']:.2f} |")
lines += ["", "## Curves (step: val loss / train loss)", ""]
for r in results:
    lines.append(f"- {r['name']}: " + ", ".join(
        f"{i}: {v:.3f}/{tl:.3f}" for i, _, v, tl in r["val_curve"]))
    lines.append("  BLEU greedy: " + ", ".join(
        f"{i}: {b:.2f}" for i, _, b in r["bleu_curve"]))
lines += ["", "## Sample translations (last run)", ""]
for src_, ref, g, b in results[-1]["samples"]:
    lines += [f"- **fr:** {src_}", f"  - ref: {ref}", f"  - greedy: {g}",
              f"  - beam: {b}"]
report = "\n".join(lines) + "\n"
print(report)
with open(f"{out}/quality_report.md", "w", encoding="utf-8") as f:
    f.write(report)
with open(f"{out}/quality_results.json", "w", encoding="utf-8") as f:
    json.dump(dict(gpu=gpu_name, steps=TRAIN_STEPS, warmup=WARMUP_STEPS,
                   base_tokens=BASE_TOKENS, results=results), f, indent=1,
              default=float, ensure_ascii=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for r in results:
    steps = [c[0] for c in r["val_curve"]]
    axes[0].plot(steps, [c[2] for c in r["val_curve"]], marker="o", ms=3,
                 label=r["name"])
    axes[1].plot([c[1] for c in r["val_curve"]], [c[2] for c in r["val_curve"]],
                 marker="o", ms=3, label=r["name"])
    axes[2].plot([b[0] for b in r["bleu_curve"]], [b[2] for b in r["bleu_curve"]],
                 marker="o", label=r["name"])
axes[0].set(xlabel="optimizer step", ylabel="val loss", title="Val loss vs steps")
axes[1].set(xlabel="training seconds", ylabel="val loss",
            title="Val loss vs training time")
axes[2].set(xlabel="optimizer step", ylabel="BLEU (greedy)", title="Test BLEU")
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{out}/quality.png", dpi=120)
plt.show()

import shutil

archive = shutil.make_archive(out, "zip", out)
print("zip:", archive)
try:
    from IPython.display import FileLink, display
    display(FileLink(os.path.relpath(archive, WORK)))
except Exception:  # noqa: BLE001  (not in a notebook)
    pass
