"""After a round (slurm/run_pegs.sbatch runs this when SERIES is set): debrief the round, ask the same model for its
lessons, and update the series file that the next round reads.

  <series>/series.json   {"model", "effort", "rounds": [{round, seed, run_dir, reward, parts, cost, reflection_cost}],
                          "lessons": [...], "saturated": bool}
  <series>/round_NN_debrief.txt, round_NN_lessons.json

  python reflect_round.py --run_dir <packaged run> --series <dir> --round N --model M [--effort low] [--cap 3]"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "core"))
P = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(P, "sim_pylibs"))
import reflect as RF  # noqa: E402
from openai_client import PRICES, Budget, OpenAIClient, cost_of  # noqa: E402
import openrouter_client as OR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True)
ap.add_argument("--series", required=True)
ap.add_argument("--round", type=int, required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--effort", default="low")
ap.add_argument("--cap", type=float, default=3.00)
args = ap.parse_args()

result = json.load(open(os.path.join(args.run_dir, "score", "result.json")))
turns = json.load(open(os.path.join(args.run_dir, "model", "turns.json")))
os.makedirs(args.series, exist_ok=True)
spath = os.path.join(args.series, "series.json")
series = RF.load_series(spath)
series.update(model=args.model, effort=args.effort)
history = [dict(round=r["round"], reward=r["reward"]) for r in series["rounds"] if r["round"] < args.round]
text = RF.debrief(result, turns, history)
open(os.path.join(args.series, f"round_{args.round:02d}_debrief.txt"), "w").write(text)

user = text + "\n\nYour current lessons:\n" + ("\n".join(f"- {x}" for x in series.get("lessons", [])) or "- (none yet)")
mock = args.model in ("mock", "replay")
cost = 0.0
if mock:
    lessons = series.get("lessons", []) + [f"(stand-in lesson from round {args.round})"]
    lessons, note = lessons[-RF.MAX_LESSONS:], "stand-in: no model call"
else:
    via_or = "/" in args.model
    price = OR.PRICES.get(args.model, OR.PRICES["anthropic/claude-opus-5.5"]) if via_or else PRICES["standard"]
    budget = Budget(os.path.join(P, "results", "openrouter_spend.json" if via_or else "openai_spend.json"), args.cap, price)
    max_out = 6000                                      # room for reasoning at effort medium before the tool call
    worst = budget.worst_case(len(RF.INSTRUCTIONS + user) / 3 + 600, max_out)
    if not budget.allows(worst):   # flagged, not stopped
        print(f"OVER BUDGET: ${budget.spent():.4f} spent + ${worst:.4f} worst case passes ${args.cap:.2f}; reflecting anyway", flush=True)
    client = (OR.OpenRouterClient(args.model, effort=args.effort, max_output_tokens=max_out) if via_or
              else OpenAIClient(args.model, effort=args.effort, max_output_tokens=max_out))
    images = []                                         # v2: the reflection sees the board, not only the text
    try:
        import numpy as np
        from PIL import Image
        photos = [os.path.join(args.run_dir, "photos", "board_final.png")]
        bad = [t["turn"] for t in turns if any(w in r for r in t.get("results") or [] for w in RF.FAIL_WORDS)][:2]
        photos += [os.path.join(args.run_dir, "photos", f"turn_{n:02d}.png") for n in bad]
        images = [np.asarray(Image.open(p).convert("RGB")) for p in photos if os.path.exists(p)]
        user += "\n\nImages: the board at the end of the round" + "".join(f"; the overhead photo at the start of turn {n}" for n in bad) + "."
    except Exception as e:   # never lose the reflection over a photo
        print("reflection photos skipped:", e, flush=True)
    worst = budget.worst_case(len(RF.INSTRUCTIONS + user) / 3 + 600 + 2500 * len(images), max_out)
    if not budget.allows(worst):   # flagged, not stopped
        print(f"OVER BUDGET: ${budget.spent():.4f} spent + ${worst:.4f} worst case passes ${args.cap:.2f}; reflecting anyway", flush=True)
    lessons, raw = None, []
    for attempt in (1, 2):   # Opus cannot be forced to call the tool (tool_choice auto), so ask once more if it does not
        res = client.call(RF.INSTRUCTIONS, user, images, [RF.LESSON_TOOL])
        u, c = res.get("usage"), 0.0
        if u:
            c = u["cost_usd"] if u.get("cost_usd") is not None else cost_of(u, price)
        elif str(res.get("error", "")).startswith("connection"):
            c = worst
        if c:
            budget.record(c, benchmark="peg_transfer_reflection", model=args.model, run=args.run_dir, round=args.round, usage=u)
        cost += c
        raw.append(dict(attempt=attempt, text=res.get("text"), reasoning=res.get("reasoning"), calls=res.get("calls"),
                        error=res.get("error"), usage=u))
        lessons, note = RF.parse_lessons(res)
        if lessons is None:                               # lessons written as a list in the reply instead of the tool
            lessons, note = RF.lessons_from_text(res.get("text") or ""), "taken from the reply text (no write_lessons call)"
        if lessons:
            break
        print(f"reflection attempt {attempt}: no lessons ({res.get('error') or 'no write_lessons call'})", flush=True)
        lessons = None
        user += ("\n\nYour previous reply did not call write_lessons. Call write_lessons now, once, with your lessons "
                 "(at most 10) and a one-sentence note. Do not answer in plain text.")
    json.dump(raw, open(os.path.join(args.series, f"round_{args.round:02d}_reflection_raw.json"), "w"), indent=1)
    if not lessons:                                       # keep the old list rather than lose it
        lessons, note = series.get("lessons", []), f"reflection failed ({note}); lessons kept"

json.dump(dict(lessons=lessons, note=note, cost=cost), open(os.path.join(args.series, f"round_{args.round:02d}_lessons.json"), "w"), indent=1)
rw = result.get("reward") or {}
series["rounds"] = [r for r in series["rounds"] if r["round"] != args.round] + [dict(
    round=args.round, seed=result.get("seed"), run_dir=args.run_dir, reward=rw.get("total", 0.0),
    reward_model_only=rw.get("model_only"), parts=rw.get("parts"),
    metrics=rw.get("metrics"), cost=result.get("cost", 0.0), reflection_cost=cost, time=time.strftime("%Y-%m-%d %H:%M"))]
series["rounds"].sort(key=lambda r: r["round"])
series["lessons"] = lessons
series["saturated"] = RF.saturated([dict(round=r["round"], reward=r["reward"]) for r in series["rounds"]])
json.dump(series, open(spath, "w"), indent=1)
print(f"ROUND {args.round}: reward {rw.get('total', 0):.1f}; {len(lessons)} lessons ({note}); reflection ${cost:.4f}; "
      f"{'SATURATED: stop this series' if series['saturated'] else 'next round: ' + str(args.round + 1)}", flush=True)
