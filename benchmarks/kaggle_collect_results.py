"""Collect a Kaggle run's results into one folder and one zip to download.

Run in the SAME notebook, after benchmarks/cuda_train_kaggle.py finished:

    %run -i benchmarks/kaggle_collect_results.py

(-i lets it read the run's variables: speed sweep, breakdown, training logs.)
Creates /kaggle/working/results_<date>/ and results_<date>.zip, then shows
a download link. Everything also appears in the notebook's Output panel.

Contents:
    houssem_vs_torch.png       the four-panel summary chart
    cuda_vs_torch_report.md    the run's markdown report
    results.json               every number of the run (when the variables exist)
    tokenizer.json             the BPE tokenizer
    trace_*.json               PyTorch profiler traces (open in ui.perfetto.dev)
    checkpoints/               only with INCLUDE_CHECKPOINTS = True (~400 MB)
"""
import json
import os
import shutil
import time

WORK = os.environ.get("RESULTS_WORK_DIR", "/kaggle/working")
INCLUDE_CHECKPOINTS = False     # model weights: ~200 MB per engine
INCLUDE_TRACES = True

stamp = time.strftime("%Y%m%d_%H%M")
out = os.path.join(WORK, f"results_{stamp}")
os.makedirs(out, exist_ok=True)

files = ["houssem_vs_torch.png", "cuda_vs_torch_report.md", "tokenizer.json"]
if INCLUDE_TRACES:
    files += [f for f in os.listdir(WORK)
              if f.startswith("trace_") and f.endswith(".json")]
copied = []
for f in files:
    src = os.path.join(WORK, f)
    if os.path.exists(src):
        shutil.copy2(src, out)
        copied.append(f)
if INCLUDE_CHECKPOINTS:
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    for f in ("houssem_gpu_checkpoint.npz", "torch_checkpoint.pt"):
        if os.path.exists(os.path.join(WORK, f)):
            shutil.copy2(os.path.join(WORK, f), os.path.join(out, "checkpoints"))
            copied.append(f"checkpoints/{f}")

# The run's numbers, if this runs in the notebook that ran the benchmark.
g = globals()
if all(k in g for k in ("sweep", "breakdown", "ht_run", "torch_run")):
    data = {
        "gpu": g["drv"].device_info(g["HT_GPU"]).name if "drv" in g else None,
        "torch": g["torch"].__version__ if "torch" in g else None,
        "batch": g.get("BATCH"), "src_len": g.get("SRC_LEN"),
        "tgt_len": g.get("TGT_LEN"),
        "step_ms": {str(B): {e: r["ms"] for e, r in row.items()}
                    for B, row in g["sweep"].items()},
        "tokens_per_s": {str(B): {e: r["tok_s"] for e, r in row.items()}
                         for B, row in g["sweep"].items()},
        "setup_s": {str(B): {e: r["setup_s"] for e, r in row.items()}
                    for B, row in g["sweep"].items()},
        "memory_gb": {str(B): {e: r["mem_gb"] for e, r in row.items()}
                      for B, row in g["sweep"].items()},
        "breakdown_ms": g["breakdown"],
        "training": {
            "steps": g.get("TRAIN_STEPS"),
            "houssem_torch": {k: g["ht_run"][k] for k in ("log", "wall", "val")},
            "torch": {k: g["torch_run"][k] for k in ("log", "wall", "val")},
        },
        "translations": g.get("translations"),
    }
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=float)
    copied.append("results.json")
else:
    print("note: run variables not found (use %run -i in the benchmark's "
          "notebook); results.json skipped")

archive = shutil.make_archive(out, "zip", out)
print(f"collected {len(copied)} files into {out}:")
for f in copied:
    print("  ", f)
print("zip:", archive, f"({os.path.getsize(archive) / 2**20:.1f} MB)")
try:  # download link (paths are relative to /kaggle/working in the notebook)
    from IPython.display import FileLink, display
    here = os.getcwd()
    os.chdir(WORK)
    display(FileLink(os.path.basename(archive)))
    os.chdir(here)
except Exception:
    pass
print("Or: Output panel (right side) -> /kaggle/working -> "
      f"{os.path.basename(archive)} -> Download")
