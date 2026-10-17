"""One figure that explains the v2 comparison: reward and model-only reward by round with the stand-in ceiling and
the main cause of each change, plus tool calls per turn (the gap that decided the comparison), place aim error
against the 4 mm tolerance, and cost per round.

  python plot_v2_summary.py      (reads comparison_v2/<model>/series.json, writes comparison_v2/v2_summary.png)"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

P = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CMP = os.path.join(P, "results", "peg_transfer_benchmark", "comparison_v2")
M = {"gpt-6.1-sol": ("GPT-6.1 Sol", "#eb6834"), "anthropic_claude-opus-5.5": ("Claude Opus 5.5", "#2a78d6")}
INK, MUTED, GRID = "#1b2120", "#5d6764", "#e3e6e4"
CEIL = (74.6, 76.1)          # stand-in on the same three boards (seeds 0-2): 76.1, 74.6, 75.9
NOTES = {("gpt-6.1-sol", 1): "one call per turn:\nout of turns at 6/12",
         ("gpt-6.1-sol", 2): "lesson said batch calls;\nstill one per turn",
         ("gpt-6.1-sol", 3): "same again: saturated,\nseries stopped",
         ("anthropic_claude-opus-5.5", 1): "12/12, but 17 contacts\n(15 in hand-overs)",
         ("anthropic_claude-opus-5.5", 2): "hand-over contacts 15 -> 6,\nbut 4 missed picks, 9 refused places",
         ("anthropic_claude-opus-5.5", 3): "12/12 after an early\nfailed hand-over"}
OFFS = {("gpt-6.1-sol", 1): (0, -14, "center"), ("gpt-6.1-sol", 2): (0, -14, "center"), ("gpt-6.1-sol", 3): (0, -14, "center"),
        ("anthropic_claude-opus-5.5", 1): (-12, 0, "right"), ("anthropic_claude-opus-5.5", 2): (14, 2, "left"),
        ("anthropic_claude-opus-5.5", 3): (14, 0, "left")}   # (dx, dy, ha) in points

S = {m: json.load(open(os.path.join(CMP, m, "series.json")))["rounds"] for m in M}


def calls_per_turn(r):
    t = json.load(open(os.path.join(r["run_dir"], "model", "turns.json")))
    n = [len(x.get("calls") or []) for x in t]
    return sum(n) / max(1, len(n))


fig = plt.figure(figsize=(13.5, 7.6), dpi=150)
gs = fig.add_gridspec(3, 2, width_ratios=[1.75, 1], hspace=0.6, wspace=0.18)
A = fig.add_subplot(gs[:, 0])
B, Cax, D = (fig.add_subplot(gs[k, 1]) for k in range(3))


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


# A: reward by round (solid) and model-only reward (dashed: robot-caused contacts excluded)
A.axhspan(*CEIL, color="#d9dcda", zorder=0)
A.text(3.7, CEIL[1] + 1.2, "stand-in baseline on the same boards (exact positions): 74.6-76.1", ha="right", fontsize=8, color=MUTED)
for m, (name, col) in M.items():
    xs = [r["round"] for r in S[m]]
    A.plot(xs, [r["reward_model_only"] for r in S[m]], color=col, lw=1.4, ls=(0, (4, 2)), marker="o", ms=5,
           mfc="white", mec=col, mew=1.3, zorder=2)
    A.plot(xs, [r["reward"] for r in S[m]], color=col, lw=2.2, marker="o", ms=8, mec="white", mew=1.5, zorder=3, label=name)
    for r in S[m]:
        mt = r["metrics"]
        dx, dy, ha = OFFS[(m, r["round"])]
        n = NOTES.get((m, r["round"]))
        A.annotate(f"{r['reward']:.1f} · {mt['transfers']}/12", (r["round"], r["reward"]), xytext=(dx, dy),
                   textcoords="offset points", ha=ha, va="top" if dy < 0 else "center", fontsize=8, color=INK)
        if n:
            A.annotate(n, (r["round"], r["reward"]), xytext=(dx, dy - 13 if dy < 0 else dy - 9), textcoords="offset points",
                       ha=ha, va="top", fontsize=7, color=MUTED)
A.set_xticks([1, 2, 3])
A.set_xlim(0.45, 3.75)
A.set_ylim(20, 90)
A.set_xlabel("Round (seed 0, 1, 2: new triangle orientations; each model's own lessons carried in)", fontsize=8, color=MUTED)
A.set_ylabel("Reward (higher is better)", fontsize=9, color=MUTED)
style(A, "Reward by round: Opus ahead every round; neither improved steadily")
from matplotlib.lines import Line2D  # noqa: E402
h = [Line2D([], [], color=c, lw=2.2, marker="o", ms=7, mec="white", label=n) for n, c in M.values()]
h.append(Line2D([], [], color=MUTED, lw=1.4, ls=(0, (4, 2)), marker="o", ms=5, mfc="white", label="model-only reward"))
A.legend(handles=h, frameon=False, fontsize=8.5, loc="upper left")

# B: tool calls per turn
w = 0.36
for k, (m, (name, col)) in enumerate(M.items()):
    xs = [r["round"] + (k - 0.5) * w for r in S[m]]
    ys = [calls_per_turn(r) for r in S[m]]
    B.bar(xs, ys, width=w * 0.92, color=col)
    for x, y in zip(xs, ys):
        B.text(x, y + 0.15, f"{y:.1f}", ha="center", fontsize=6.5, color=INK)
B.axhline(3, color=MUTED, lw=0.8, ls=(0, (3, 3)))
B.set_ylim(0, 5.3)
B.set_xticks([1, 2, 3])
style(B, "Tool calls per turn (up to 6; dashed 3 = a transfer a turn)")

# C: place aim error vs tolerance
Cax.axhspan(0, 4, color="#dcefe3", zorder=0)
Cax.text(0.75, 0.5, "accepted within 4 mm", ha="left", fontsize=7, color=MUTED)
for m, (name, col) in M.items():
    pts = [(r["round"], r["metrics"]["mean_aim_mm"]) for r in S[m] if r["metrics"]["mean_aim_mm"] is not None]
    Cax.plot(*zip(*pts), color=col, lw=2, marker="o", ms=6, mec="white", mew=1.2)
    for x, y in pts:
        Cax.text(x, y + 0.5, f"{y:.1f}", ha="center", fontsize=7, color=INK)
Cax.set_ylim(0, 7)
Cax.set_xlim(0.7, 3.3)
Cax.set_xticks([1, 2, 3])
Cax.set_ylabel("mm", fontsize=8, color=MUTED)
style(Cax, "Mean place aim error (distance to the peg's centre)")

# D: cost per round
for k, (m, (name, col)) in enumerate(M.items()):
    xs = [r["round"] + (k - 0.5) * w for r in S[m]]
    ys = [r["cost"] + (r.get("reflection_cost") or 0) for r in S[m]]
    D.bar(xs, ys, width=w * 0.92, color=col)
    for x, y in zip(xs, ys):
        D.text(x, y + 0.02, f"\\${y:.2f}", ha="center", fontsize=6.5, color=INK)
tot = {M[m][0]: sum(r["cost"] + (r.get("reflection_cost") or 0) for r in S[m]) for m in M}
D.set_ylim(0, 1.1)
D.set_xticks([1, 2, 3])
D.set_ylabel("\\$", fontsize=8, color=MUTED)
style(D, f"Cost per round (total: GPT \\${tot['GPT-6.1 Sol']:.2f}, Opus \\${tot['Claude Opus 5.5']:.2f})")
D.set_xlabel("Round", fontsize=8, color=MUTED)

fig.suptitle("FLS peg transfer v2 (preliminary, one run per round): GPT-6.1 Sol vs Claude Opus 5.5, effort medium, 20 turns",
             x=0.06, ha="left", fontsize=11.5, color=INK)
fig.text(0.06, 0.015, "Model-only reward leaves out contacts during motions the robot plans itself (hand-overs, lowering). "
         "One run per round; round 1 = seed 0, 2 = seed 1, 3 = seed 2 for both models, so the boards match across models.",
         fontsize=7.5, color=MUTED)
out = os.path.join(CMP, "v2_summary.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
