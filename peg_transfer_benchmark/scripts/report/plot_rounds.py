"""Reward vs round for every model series in $P/results/peg_transfer_benchmark/comparison/ (one series.json each),
with the stand-in (exact positions, same controller) as the reference if its series exists ("mock").

  python plot_rounds.py [--out comparison/reward_vs_round.png]     (needs matplotlib)"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

P = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CMP = os.path.join(P, "results", "peg_transfer_benchmark", "comparison")
COLORS = {"anthropic_claude-opus-5.5": "#2a78d6", "gpt-6.1-sol": "#eb6834"}     # fixed per model, never by rank
NAMES = {"anthropic_claude-opus-5.5": "Claude Opus 5.5", "gpt-6.1-sol": "GPT-6.1 Sol", "mock": "Stand-in (exact positions)"}

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(CMP, "reward_vs_round.png"))
args = ap.parse_args()

series = {}
for f in sorted(glob.glob(os.path.join(CMP, "*", "series.json"))):
    s = json.load(open(f))
    if s.get("rounds"):
        series[os.path.basename(os.path.dirname(f))] = s
if not series:
    raise SystemExit(f"no series in {CMP}")

fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
ink, muted, grid = "#1b2120", "#5d6764", "#e3e6e4"
for tag, s in series.items():
    xs = [r["round"] for r in s["rounds"]]
    ys = [r["reward"] for r in s["rounds"]]
    if tag == "mock":
        ax.plot(xs, ys, color="#9aa29f", lw=1.5, zorder=1)
        ax.annotate(NAMES[tag], (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=muted)
        continue
    c = COLORS.get(tag, "#6250d6")
    ax.plot(xs, ys, color=c, lw=2, marker="o", ms=6, mec="white", mew=1.5, zorder=3)
    ax.annotate(f"{NAMES.get(tag, tag)}  {ys[-1]:.1f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=ink)
ax.set_xlabel("Round", color=muted, fontsize=9)
ax.set_ylabel("Reward (higher is better)", color=muted, fontsize=9)
ax.set_xticks(range(1, 1 + max(len(s["rounds"]) for s in series.values())))
ax.set_ylim(0, 100)
ax.grid(axis="y", color=grid, lw=0.8)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(grid)
ax.tick_params(colors=muted, labelsize=8)
ax.set_title("FLS peg transfer: reward by round", loc="left", fontsize=11, color=ink)
fig.tight_layout()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
fig.savefig(args.out)

# the numbers behind it
rows = ["| Model | " + " | ".join(f"Round {k}" for k in range(1, 6)) + " | Cost |", "|---|" + "---|" * 6]
for tag, s in series.items():
    vals = {r["round"]: r["reward"] for r in s["rounds"]}
    cost = sum(r.get("cost", 0) + r.get("reflection_cost", 0) for r in s["rounds"])
    rows.append(f"| {NAMES.get(tag, tag)} | " + " | ".join(f"{vals[k]:.1f}" if k in vals else "–" for k in range(1, 6))
                + f" | ${cost:.3f}{' (simulated)' if tag == 'mock' else ''} |")
open(os.path.join(CMP, "rewards.md"), "w").write("\n".join(rows) + "\n")
print("wrote", args.out, "and", os.path.join(CMP, "rewards.md"))
