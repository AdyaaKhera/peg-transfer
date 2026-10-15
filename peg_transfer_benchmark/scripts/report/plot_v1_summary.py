"""One figure that explains the v1 comparison: reward by round with the stand-in ceiling and the main cause of each
change, plus pick success, place aim error against the hole's tolerance, and cost per round.

  python plot_v1_summary.py      (reads comparison/<model>/series.json, writes comparison/v1_summary.png)"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

P = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CMP = os.path.join(P, "results", "peg_transfer_benchmark", "comparison")
M = {"gpt-6.1-sol": ("GPT-6.1 Sol", "#eb6834"), "anthropic_claude-opus-5.5": ("Claude Opus 5.5", "#2a78d6")}
INK, MUTED, GRID = "#1b2120", "#5d6764", "#e3e6e4"
CEIL = (75.5, 77.4)          # stand-in, same controller, exact positions, full 12-transfer task, seeds 0-4
OFFS = {("gpt-6.1-sol", 2): (-14, -22, "right"), ("gpt-6.1-sol", 3): (45, -20, "left"), ("anthropic_claude-opus-5.5", 3): (-10, 30, "right")}   # label nudges (points)
NOTES = {("gpt-6.1-sol", 2): "robot: hand-over clashes\n(5 incidents, 508 s)",
         ("gpt-6.1-sol", 3): "4 drops: reused a wrong peg\ncoordinate from its own lesson",
         ("anthropic_claude-opus-5.5", 1): "aim ~13 mm off:\n3 of 24 picks, 3 drops",
         ("anthropic_claude-opus-5.5", 2): "5 drops; picks with the\nleft arm in the right field",
         ("anthropic_claude-opus-5.5", 3): "stopped after 10 missed\npicks (no drops, so no penalty)"}

S = {m: json.load(open(os.path.join(CMP, m, "series.json")))["rounds"] for m in M}
fig = plt.figure(figsize=(13.5, 7.6), dpi=150)
gs = fig.add_gridspec(3, 2, width_ratios=[1.75, 1], hspace=0.55, wspace=0.18)
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


# A: reward by round
A.axhspan(*CEIL, color="#d9dcda", zorder=0)
A.text(3.7, CEIL[1] + 1.5, "stand-in ceiling, same robot, exact positions: 75-77", ha="right", fontsize=8, color=MUTED)
A.axhline(0, color="#b9bebc", lw=1)
for m, (name, col) in M.items():
    xs = [r["round"] for r in S[m]]
    ys = [r["reward"] for r in S[m]]
    A.plot(xs, ys, color=col, lw=2.2, marker="o", ms=8, mec="white", mew=1.5, zorder=3, label=name)
    for r in S[m]:
        mt = r["metrics"]
        dx, dy, ha = OFFS.get((m, r["round"]), (0, 11, "center"))
        lab = f"{mt['transfers']} transfers · {mt['drops']} drop{'' if mt['drops'] == 1 else 's'}"
        n = NOTES.get((m, r["round"]))
        if (m, r["round"]) in OFFS:      # crowded: label and note together, off to one side
            A.annotate(lab + ("\n" + n if n else ""), (r["round"], r["reward"]), xytext=(dx, dy), textcoords="offset points",
                       ha=ha, va="center", fontsize=7.2, color=INK, arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8))
            continue
        A.annotate(lab, (r["round"], r["reward"]), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=7.5, color=INK)
        if n:
            A.annotate(n, (r["round"], r["reward"]), xytext=(0, -26), textcoords="offset points", ha="center", va="top",
                       fontsize=7, color=MUTED)
A.set_xticks([1, 2, 3])
A.set_xlim(0.6, 3.75)
A.set_ylim(-75, 92)
A.set_xlabel("Round (new triangle orientations each round; lessons from earlier rounds carried in)", fontsize=8, color=MUTED)
A.set_ylabel("Reward (higher is better)", fontsize=9, color=MUTED)
style(A, "Reward by round: neither model improved from its own lessons")
A.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 0.9))

# B: pick success
w = 0.36
for k, (m, (name, col)) in enumerate(M.items()):
    xs = [r["round"] + (k - 0.5) * w for r in S[m]]
    ys = [100 * r["metrics"]["picks_ok"] / max(1, r["metrics"]["picks"]) for r in S[m]]
    B.bar(xs, ys, width=w * 0.92, color=col)
    for x, y, r in zip(xs, ys, S[m]):
        B.text(x, y + 3, f"{r['metrics']['picks_ok']}/{r['metrics']['picks']}", ha="center", fontsize=6.5, color=INK)
B.set_ylim(0, 125)
B.set_xticks([1, 2, 3])
B.set_ylabel("%", fontsize=8, color=MUTED)
style(B, "Picks that grasped first time")

# C: place aim error vs tolerance
Cax.axhspan(0, 4, color="#dcefe3", zorder=0)
Cax.text(0.75, 0.7, "hole fits within ~4 mm", ha="left", fontsize=7, color=MUTED)
for m, (name, col) in M.items():
    pts = [(r["round"], r["metrics"]["mean_aim_mm"]) for r in S[m] if r["metrics"]["mean_aim_mm"] is not None]
    Cax.plot(*zip(*pts), color=col, lw=2, marker="o", ms=6, mec="white", mew=1.2)
    for x, y in pts:
        Cax.text(x, y + 1.2, f"{y:.1f}", ha="center", fontsize=7, color=INK)
Cax.text(3, 9, "Opus round 3:\nno places", ha="center", fontsize=6.5, color="#2a78d6")
Cax.set_ylim(0, 19)
Cax.set_xlim(0.7, 3.3)
Cax.set_xticks([1, 2, 3])
Cax.set_ylabel("mm", fontsize=8, color=MUTED)
style(Cax, "Mean place aim error (distance to the peg's centre)")

# D: cost per round
for k, (m, (name, col)) in enumerate(M.items()):
    xs = [r["round"] + (k - 0.5) * w for r in S[m]]
    ys = [r["cost"] + r.get("reflection_cost", 0) for r in S[m]]
    D.bar(xs, ys, width=w * 0.92, color=col)
    for x, y in zip(xs, ys):
        D.text(x, y + 0.02, f"\\${y:.2f}", ha="center", fontsize=6.5, color=INK)
tot = {M[m][0]: sum(r["cost"] + r.get("reflection_cost", 0) for r in S[m]) for m in M}
D.set_ylim(0, 0.75)
D.set_xticks([1, 2, 3])
D.set_ylabel("\\$", fontsize=8, color=MUTED)
style(D, f"Cost per round (total: GPT \\${tot['GPT-6.1 Sol']:.2f}, Opus \\${tot['Claude Opus 5.5']:.2f})")
D.set_xlabel("Round", fontsize=8, color=MUTED)

fig.suptitle("FLS peg transfer v1 (frozen): GPT-6.1 Sol vs Claude Opus 5.5, low effort, 12-transfer task, one run per round",
             x=0.06, ha="left", fontsize=11.5, color=INK)
fig.text(0.06, 0.015, "Robot faults (hand-over clashes, refused places) count against both models. Opus's round 3 was "
         "stopped by hand after 10 turns. Each round is one run, so round-to-round changes are noisy.",
         fontsize=7.5, color=MUTED)
out = os.path.join(CMP, "v1_summary.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
