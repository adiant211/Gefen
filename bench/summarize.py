"""Summarize benchmark runs into a markdown results table.

Gap = final validation loss minus AdamW's, paired by seed.
Decision rule (from GEFEN_ACCURACY_PLAN.md): keep a variant if it shrinks the paired gap versus gefen-base by more
than 2 standard errors across seeds and its optimizer state is within 2% of gefen-base.
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="/workspace/runs")
ap.add_argument("--base", default="gefen-base")
ap.add_argument("--ref", default="adamw-base", help="reference run for the gap column")
ap.add_argument("--probe_from", type=int, default=50, help="average probe errors from this step on")
args = ap.parse_args()

runs = defaultdict(dict)  # label -> seed -> record
for path in sorted(glob.glob(os.path.join(args.runs, "*.jsonl"))):
    recs = [json.loads(line) for line in open(path)]
    cfg = recs[0]
    final = next((r for r in recs if r["type"] == "final"), None)
    if final is None:
        continue
    a = cfg["args"]
    label = f"{a['opt']}-{a['variant']}"
    probes = [r["err"]["all"] for r in recs if r["type"] == "probe" and r["step"] >= args.probe_from]
    train = [r for r in recs if r["type"] == "train"]
    runs[label][a["seed"]] = {
        "val": final["val_loss"],
        "state": final.get("state_bytes_muon", final["state_bytes"]),
        "wall": final["wall"],
        # median: the step right after each eval reports a meaningless spike (its timer excludes the eval)
        "tok_s": sorted(r["tok_s"] for r in train[1:])[len(train[1:]) // 2] if len(train) > 1 else float("nan"),
        "probe": {k: sum(p[k] for p in probes) / len(probes) for k in probes[0]} if probes else {},
    }


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


adamw = runs.get(args.ref, {})
base = runs.get(args.base, {})
lines = [f"| run | seeds | val loss | gap to {args.ref} | gain vs {args.base} | state MiB | update err | mom err | v err | k tok/s | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for label in sorted(runs):
    seeds = sorted(runs[label])
    vals = [runs[label][s]["val"] for s in seeds]
    v_m, v_se = mean_se(vals)
    paired = [s for s in seeds if s in adamw]
    gap = mean_se([runs[label][s]["val"] - adamw[s]["val"] for s in paired]) if paired else (float("nan"),) * 2
    both = [s for s in seeds if s in base and s in adamw]
    gain = (mean_se([base[s]["val"] - runs[label][s]["val"] for s in both])
            if both and label != args.base and label.startswith("gefen") else None)
    state = sum(runs[label][s]["state"] for s in seeds) / len(seeds) / 2**20
    probe = runs[label][seeds[0]]["probe"]
    tok_s = sum(runs[label][s]["tok_s"] for s in seeds) / len(seeds) / 1e3
    verdict = ""
    if gain is not None and label.startswith("gefen") and base:
        base_state = sum(base[s]["state"] for s in base) / len(base) / 2**20
        mem_ok = state <= base_state * 1.02
        sig = len(both) >= 2 and gain[0] > 2 * gain[1]
        verdict = "keep" if (sig and mem_ok) else ("drop: memory" if not mem_ok else "drop: not significant")
    fmt = lambda ms: f"{ms[0]:+.4f} ± {ms[1]:.4f}" if ms else ""
    lines.append(
        f"| {label} | {len(seeds)} | {v_m:.4f} | {fmt(gap) if paired else ''} | {fmt(gain) if gain else ''} | "
        f"{state:.0f} | {probe.get('tot', float('nan')):.3f} | {probe.get('mom', float('nan')):.3f} | "
        f"{probe.get('v', float('nan')):.3f} | {tok_s:.0f} | {verdict} |")
table = "\n".join(lines)
print(table)
open(os.path.join(args.runs, "results.md"), "w").write(table + "\n")
