"""Regenerate the report's figures from data/results.json.

    python reports/2026-11-24_gpu_trainstep_vs_pytorch/make_figures.py

Needs only matplotlib. Writes figures/*.png next to this file.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
R = json.loads((HERE / "data" / "results.json").read_text(encoding="utf-8"))
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

HT, TC, TE = "houssem_torch", "torch.compile", "PyTorch eager"
ENGINES = (HT, TC, TE)
# One fixed color per engine in every figure.
COLOR = {HT: "#2a78d6", TC: "#eb6834", TE: "#1baf7a"}
SURF, INK, INK2, MUTED, GRID, BASE = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                      "#e1e0d9", "#c3c2b7")
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": INK2, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
F = R["final_run"]


def grid(ax, axis):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print("wrote", OUT / name)


# 1. Time per step, batch 64
fig, ax = plt.subplots(figsize=(8, 3.2))
order = [TE, TC, HT]
vals = [F["step_ms"]["64"][e] for e in order]
ax.barh(order, vals, color=[COLOR[e] for e in order], height=0.55)
ht = F["step_ms"]["64"][HT]
for y, (e, v) in enumerate(zip(order, vals)):
    note = "" if e == HT else f"   ({v / ht:.2f}x houssem_torch's time)"
    ax.text(v + 1.5, y, f"{v:.1f} ms{note}", va="center", color=INK2, fontsize=9)
ax.set_xlim(0, max(vals) * 1.6)
ax.set_xlabel("milliseconds per training step (lower is better)")
ax.set_title("Time per training step (Tesla T4, batch 64, fp16)")
grid(ax, "x")
save(fig, "fig1_step_time.png")

# 2. Step time by batch size
fig, ax = plt.subplots(figsize=(8, 4))
batches = sorted(F["step_ms"], key=int)
w = 0.26
for i, e in enumerate(ENGINES):
    xs = [b + (i - 1) * (w + 0.02) for b in range(len(batches))]
    v = [F["step_ms"][B][e] for B in batches]
    ax.bar(xs, v, width=w, color=COLOR[e], label=e)
    for x, val in zip(xs, v):
        ax.text(x, val + 2, f"{val:.0f}", ha="center", fontsize=8, color=MUTED)
for b, B in enumerate(batches):
    s = F["speedup_vs_compile_reported"][B]   # from unrounded timings
    ax.text(b, max(F["step_ms"][B].values()) + 16, f"{s:.2f}x vs torch.compile",
            ha="center", fontsize=9, color=INK2)
ax.set_xticks(range(len(batches)), [f"batch {B}" for B in batches])
ax.set_ylim(0, max(F["step_ms"][batches[-1]].values()) * 1.2)
ax.set_ylabel("milliseconds per step (lower is better)")
ax.set_title("Step time by batch size")
ax.legend(loc="upper left")
grid(ax, "y")
save(fig, "fig2_batch_sizes.png")

# 3. Where the time goes
bd = F["breakdown_ms_batch64"]
cats = bd["categories"][::-1]
fig, ax = plt.subplots(figsize=(9, 5))
h = 0.26
for i, e in enumerate(ENGINES):
    ys = [r + (1 - i) * (h + 0.02) for r in range(len(cats))]
    v = bd[e][::-1]
    ax.barh(ys, v, height=h, color=COLOR[e], label=e)
    for y, val in zip(ys, v):
        if val >= 0.5:
            ax.text(val + 0.4, y, f"{val:.1f}", va="center", fontsize=7.5, color=MUTED)
ax.set_yticks(range(len(cats)), cats)
ax.set_xlim(0, max(max(bd[e]) for e in ENGINES) * 1.12)
ax.set_xlabel("milliseconds per step (each engine's bars add up to its step time)")
ax.set_title("Where the time goes (batch 64)")
ax.legend(loc="lower right")
grid(ax, "x")
save(fig, "fig3_time_breakdown.png")

# 4. Training: loss vs step (same curve = same math) and vs wall-clock (speed)
T = F["training"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
for e in (HT, TC):
    log = T[e]["log"]
    steps, loss, secs = ([r[i] for r in log] for i in range(3))
    kw = dict(color=COLOR[e], linewidth=2, marker="o", markersize=6,
              markeredgecolor=SURF, markeredgewidth=1.5)
    a1.plot(steps, loss, label=e, **kw)
    a2.plot(secs, loss, label=f"{e}: {T[e]['wall_s']:.0f} s, val loss "
            f"{T[e]['val_loss']:.3f}", **kw)
a1.set_xlabel("training step")
a1.set_title("Loss per step: the two engines learn identically")
a2.set_xlabel("wall-clock seconds")
sooner = T[TC]["wall_s"] / T[HT]["wall_s"]
a2.set_title(f"Loss over time: same {T['steps']} steps, {sooner:.2f}x sooner")
for a in (a1, a2):
    a.set_ylabel("training loss")
    a.legend(loc="upper right")
    grid(a, "y")
save(fig, "fig4_training.png")

# 5. Progress: speedup per version (same-session) and attention time
H = R["history"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.8))
names = [v["name"] for v in H["versions"]]
sp = [v["compile_ms"] / v["houssem_ms"] for v in H["versions"]]
a1.barh(names[::-1], sp[::-1], color=COLOR[HT], height=0.5)
a1.axvline(1.0, color=COLOR[TC], linewidth=2)
a1.text(1.02, -0.45, "torch.compile = 1.0", color=INK2, fontsize=9, va="center")
for y, v in enumerate(sp[::-1]):
    a1.text(max(v, 1.0) + 0.03, y, f"{v:.2f}x", va="center", color=INK2,
            fontweight="bold")
a1.set_xlim(0, 1.5)
a1.set_ylim(-0.65, 1.4)
a1.set_xlabel("speed relative to torch.compile, same session (higher is better)")
a1.set_title("houssem_torch versions")
grid(a1, "x")
att = H["attention_ms"]
labels = list(att)[::-1]
cols = [COLOR[TC] if "torch" in k else COLOR[HT] for k in labels]
a2.barh(labels, [att[k] for k in labels], color=cols, height=0.5)
for y, k in enumerate(labels):
    a2.text(att[k] + 0.3, y, f"{att[k]:.1f} ms", va="center", color=INK2)
a2.set_xlim(0, max(att.values()) * 1.25)
a2.set_xlabel("attention milliseconds per step (lower is better)")
a2.set_title("Attention: still behind PyTorch's sdpa")
grid(a2, "x")
save(fig, "fig5_progress.png")
