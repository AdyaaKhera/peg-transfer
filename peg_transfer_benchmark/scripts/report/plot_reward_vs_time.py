"""Reward so far vs robot time, one line per run, every model and round of the v2 comparison on one graph.

The reward is recomputed after every turn (the dashboard's live reward), so a line shows how a run's score built up
and where it fell: e.g. a strong first half followed by contacts and refused places on the return trip.

  python plot_reward_vs_time.py [--root $P/results/peg_transfer_benchmark/comparison_v2] [--out <png>]"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

P = os.environ.get("P", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=f"{P}/results/peg_transfer_benchmark/comparison_v2")
ap.add_argument("--out", default=None)
args = ap.parse_args()
out = args.out or os.path.join(args.root, "reward_vs_robot_time.png")

# one hue per model (categorical slots 1 and 2), rounds as light -> dark steps of it, plus a line style per round
MODELS = {"gpt-6.1-sol": ("GPT-6.1 Sol", ["#f4a27f", "#eb6834", "#b4471b", "#7f3011", "#7f3011"]),     # same hues as the v1 plots
          "anthropic_claude-opus-5.5": ("Claude Opus 5.5", ["#86b6ef", "#2a78d6", "#184f95", "#0d366b", "#0d366b"])}
STYLES = [(0, (1.5, 1.5)), (0, (5, 2)), "solid", (0, (8, 2, 2, 2)), "solid"]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


def run_curve(run_dir):
    f = os.path.join(run_dir, "live", "events.jsonl")
    if not os.path.exists(f):
        f = os.path.join(run_dir, "events.jsonl")
    pts = [(0.0, 0.0)]
    for line in open(f):
        e = json.loads(line)
        if e.get("type") == "score" and e.get("reward"):
            pts.append((float(e["score"]["sim_seconds"]), float(e["reward"]["total"])))
    return pts


fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ends = []
for tag, (name, ramp) in MODELS.items():
    try:
        series = json.load(open(os.path.join(args.root, tag, "series.json")))
    except OSError:
        continue
    for r in series["rounds"]:
        k = r["round"] - 1
        pts = run_curve(r["run_dir"])
        m = r["metrics"]
        pts.append((m["robot_s"], r["reward"]))            # the final reward, as recorded for the series
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=ramp[k], lw=2, ls=STYLES[k], solid_capstyle="round", zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], s=46, color=ramp[k], edgecolor=SURFACE, linewidth=2, zorder=4)
        peak = max(range(len(ys)), key=lambda i: ys[i])
        if ys[peak] - ys[-1] >= 4:                         # mark a peak the run fell from
            ax.scatter([xs[peak]], [ys[peak]], s=30, facecolor=SURFACE, edgecolor=ramp[k], linewidth=1.6, zorder=4)
        ends.append([xs[-1], ys[-1], f"{'GPT' if 'GPT' in name else 'Opus'} r{r['round']}  {r['reward']:.1f}"
                     f"  ({m['transfers']}/12)", ramp[k]])

# direct labels at the run ends, nudged apart vertically so they never overlap
ends.sort(key=lambda e: e[1])
for i in range(1, len(ends)):
    if ends[i][1] - ends[i - 1][1] < 2.6:
        ends[i][1] = ends[i - 1][1] + 2.6
for x, y, text, col in ends:
    ax.annotate(text, (x, y), xytext=(9, 0), textcoords="offset points", va="center", fontsize=9, color=INK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.9))

ax.axvline(300, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
ax.text(302, 2, "FLS time limit (300 s)", color=MUTED, fontsize=8.5, va="bottom")
ax.set_xlabel("Robot time (s): arm motion only, not model thinking or simulator overhead", color=INK, fontsize=10)
ax.set_ylabel("Reward so far (recomputed after every turn)", color=INK, fontsize=10)
ax.set_title("Peg transfer v2: reward as each run unfolds, all models and rounds", color=INK, fontsize=13,
             loc="left", pad=12)
ax.set_ylim(0, max(80, max(e[1] for e in ends) + 6) if ends else 80)
ax.set_xlim(0, max(330, max(e[0] for e in ends) * 1.22) if ends else 330)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=9)

from matplotlib.lines import Line2D  # noqa: E402
handles = [Line2D([], [], color=MODELS[t][1][1], lw=2.5, label=MODELS[t][0]) for t in MODELS]
handles += [Line2D([], [], color=MUTED, lw=2, ls=STYLES[k], label=f"round {k + 1}") for k in range(3)]
handles += [Line2D([], [], color=MUTED, marker="o", ls="", markerfacecolor=SURFACE, label="peak it fell from")]
ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9, ncol=2, labelcolor=INK)
fig.text(0.01, 0.01, "Each corner is one turn; later rounds are darker. Early points run high because precision, safety and efficiency "
         "start full and only fall as misses and contacts come in;\nthe line drops at each refused place or contact and "
         "climbs with each completed transfer.",
         color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(out, facecolor=SURFACE)
print(out)
