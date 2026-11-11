"""Build the accuracy-benchmark figures from the JSONL run logs.

usage: python bench/plot_results.py [--logs bench/run_logs] [--out figures/accuracy]
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--logs", default=os.path.join(os.path.dirname(__file__), "run_logs"))
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "figures", "accuracy"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# chart chrome (light surface) and categorical slots in fixed order; each run keeps its color in every figure
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {
    "adamw-base": ("AdamW", "#2a78d6"),
    "gefen-base": ("Gefen 1.2.3", "#eb6834"),
    "gefen-vprof_col": ("Gefen + column profile", "#1baf7a"),
    "gefen-vprof_full": ("Gefen + per-element profile (ceiling)", "#eda100"),
    "gefen-unquant": ("Gefen + unquantized update (2a)", "#e87ba4"),
}
BAR = "#2a78d6"
TOKENS_PER_STEP = 64 * 1024

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def load(sub):
    runs = defaultdict(dict)
    for path in glob.glob(os.path.join(args.logs, sub, "*.jsonl")):
        recs = [json.loads(line) for line in open(path)]
        if not any(r["type"] == "final" for r in recs):
            continue
        a = recs[0]["args"]
        runs[f"{a['opt']}-{a['variant']}"][a["seed"]] = recs
    return runs


def mean_se(xs):
    m = sum(xs) / len(xs)
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs)) if len(xs) > 1 else 0.0
    return m, se


def final(recs):
    return next(r for r in recs if r["type"] == "final")


def paired(runs, a, b):
    seeds = sorted(set(runs[a]) & set(runs[b]))
    return mean_se([final(runs[a][s])["val_loss"] - final(runs[b][s])["val_loss"] for s in seeds])


def seed_mean_curve(seed_runs, pick):
    """pick(recs) -> {x: y}; returns sorted (x, mean y) over the x values every seed has."""
    per_seed = [pick(r) for r in seed_runs.values()]
    xs = sorted(set.intersection(*(set(d) for d in per_seed)))
    return xs, [sum(d[x] for d in per_seed) / len(per_seed) for x in xs]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, name), dpi=150)
    plt.close(fig)
    print("wrote", name)


adam = load("adam_50M")
muon = load("muon_20M")

# 1. gap to AdamW --------------------------------------------------------------------------------------------
order = ["gefen-base", "gefen-unquant", "gefen-vprof_col", "gefen-vprof_full"]
fig, ax = plt.subplots(figsize=(8.2, 3.4))
for i, key in enumerate(order):
    m, se = paired(adam, key, "adamw-base")
    state = final(adam[key][0])["state_bytes"] / 2**20
    ax.barh(i, m, height=0.6, color=SERIES[key][1], edgecolor=SURFACE, linewidth=2)
    ax.errorbar(m, i, xerr=se, color=INK2, capsize=3, lw=1)
    ax.text(m + se + 0.004, i, f"+{m:.3f} ± {se:.3f}   ({state:.0f} MiB state)", va="center", color=INK2)
ax.set_yticks(range(len(order)), [SERIES[k][0] for k in order])
ax.invert_yaxis()
ax.axvline(0, color=AXIS, lw=1)
ax.set_xlim(0, 0.30)
ax.grid(axis="y", visible=False)
ax.set_xlabel("final validation loss minus AdamW (lower is better; AdamW state: 756 MiB)")
ax.set_title("Gap to AdamW after 50M tokens (mean ± SE, 3 seeds)")
save(fig, "adam_gap_to_adamw.png")

# 2. validation curves ---------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 4.2))
curve_keys = ["adamw-base", "gefen-base", "gefen-vprof_col", "gefen-vprof_full"]


def val_points(recs):
    pts = {(r["step"] + 1) * TOKENS_PER_STEP / 1e6: r["val_loss"] for r in recs if r["type"] == "eval" and r["step"] >= 199}
    last = max(r["step"] for r in recs if r["type"] == "train")
    pts[(last + 1) * TOKENS_PER_STEP / 1e6] = final(recs)["val_loss"]
    return pts


label_y = []
for key in curve_keys:
    xs, ys = seed_mean_curve(adam[key], val_points)
    ax.plot(xs, ys, color=SERIES[key][1], lw=2, marker="o", ms=4, label=SERIES[key][0])
    label_y.append((ys[-1], key))
# direct end labels, nudged apart so they do not collide
placed = []
for y, key in sorted(label_y):
    y_text = max([y] + [p + 0.06 for p in placed])
    placed.append(y_text)
    ax.annotate(f"{y:.3f}", (xs[-1], y), xytext=(xs[-1] + 0.8, y_text), color=INK2, va="center", fontsize=9)
ax.set_xlim(right=xs[-1] + 5)
ax.set_xlabel("training tokens (millions)")
ax.set_ylabel("validation loss")
ax.set_title("Validation loss, mean of 3 seeds")
ax.legend(loc="upper right")
save(fig, "adam_val_loss.png")

# 3. update error over training ----------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 4.0))
for key in ["gefen-base", "gefen-vprof_col", "gefen-vprof_full"]:
    xs, ys = seed_mean_curve(adam[key], lambda recs: {r["step"]: r["err"]["all"]["tot"] for r in recs if r["type"] == "probe"})
    ax.plot(xs, ys, color=SERIES[key][1], lw=2, label=SERIES[key][0])
    ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", color=INK2, va="center", fontsize=9)
ax.set_ylim(0, None)
ax.set_xlim(right=max(xs) * 1.07)
ax.set_xlabel("optimizer step")
ax.set_ylabel("relative update error vs exact AdamW")
ax.set_title("How far each step is from the exact AdamW step (same gradients)")
ax.legend(loc="upper right")
save(fig, "adam_update_error.png")

# 4. second-moment error by layer type ---------------------------------------------------------------------------
cats = ["embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "norm"]


def mean_probe(key, cat, part):
    vals = []
    for recs in adam[key].values():
        pr = [r for r in recs if r["type"] == "probe" and r["step"] >= 50]
        vals.append(sum(p["err"][cat][part] for p in pr) / len(pr))
    return sum(vals) / len(vals)


fig, ax = plt.subplots(figsize=(8.2, 3.8))
width = 0.38
for j, key in enumerate(["gefen-base", "gefen-vprof_col"]):
    ys = [mean_probe(key, c, "v") for c in cats]
    xs = [i + (j - 0.5) * width for i in range(len(cats))]
    ax.bar(xs, ys, width=width, color=SERIES[key][1], edgecolor=SURFACE, linewidth=2, label=SERIES[key][0])
ax.set_xticks(range(len(cats)), cats, rotation=0, fontsize=8.5)
ax.grid(axis="x", visible=False)
ax.set_ylabel("relative update error from v")
ax.set_title("Second-moment error by layer type (mean from step 50, 3 seeds)")
ax.legend(loc="upper left")
save(fig, "adam_v_error_by_layer.png")

# 5. Muon: paired differences --------------------------------------------------------------------------------------
pairs = [("gefenmuon-base", "muon-base", "GefenMuon 1.2.3"),
         ("muon-normuon", "muon-base", "NorMuon (fp32 momentum)"),
         ("gefenmuon-normuon", "muon-base", "GefenMuon + NorMuon")]
fig, ax = plt.subplots(figsize=(8.2, 3.0))
for i, (a, b, label) in enumerate(pairs):
    m, se = paired(muon, a, b)
    state = final(muon[a][0])["state_bytes_muon"] / 2**20
    ax.barh(i, m, height=0.55, color=BAR, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(m, i, xerr=se, color=INK2, capsize=3, lw=1)
    # labels sit right of zero (or of the bar end) so they never collide with the category labels
    ax.text(max(m + se, 0) + 0.0012, i, f"{m:+.4f} ± {se:.4f}   ({state:.0f} MiB state)", va="center",
            ha="left", color=INK2)
ax.set_yticks(range(len(pairs)), [p[2] for p in pairs])
ax.invert_yaxis()
ax.axvline(0, color=AXIS, lw=1)
ax.set_xlim(-0.04, 0.035)
ax.grid(axis="y", visible=False)
ax.set_xlabel("final validation loss minus Muon with fp32 momentum (300 MiB); lower is better")
ax.set_title("Muon family vs Muon after 20M tokens (mean ± SE, 3 seeds)")
save(fig, "muon_vs_reference.png")

# 6. GefenMuon update error over training ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 3.6))
for key, label, color in [("gefenmuon-base", "GefenMuon vs exact Muon", "#eb6834"),
                          ("gefenmuon-normuon", "GefenMuon + NorMuon vs exact NorMuon", "#1baf7a")]:
    xs, ys = seed_mean_curve(muon[key], lambda recs: {r["step"]: r["err"]["all"]["tot"] for r in recs if r["type"] == "probe"})
    ax.plot(xs, ys, color=color, lw=2, label=label)
ax.set_ylim(0, None)
ax.set_xlabel("optimizer step")
ax.set_ylabel("relative update error")
ax.set_title("GefenMuon update error vs exact (Nor)Muon, same gradients")
ax.legend(loc="upper right")
save(fig, "muon_update_error.png")
