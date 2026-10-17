"""FLS peg transfer benchmark: an AI model (GPT via OpenAI, or e.g. Claude via OpenRouter) runs the two YAM arms
through the peg transfer task in Isaac Sim.

Each turn: park both arms, photograph the board from overhead (orthographic, mm grid drawn on), check the budget, ask
the model for one or more tool calls (pick / hand_over / place / done), run them in order (stopping at the first
failure). Everything the dashboard shows is written to $RUN_DIR; creating $RUN_DIR/STOP ends the run cleanly.

  --model mock      stand-in model that plays the task from the simulator's ground truth; $0, no key needed
  --model clumsy    error-prone stand-in (aim off, wrong arm, bad corners; corrects from the feedback words); $0
  --model replay    replays the calls of an earlier run; $0
  --model a/b       a model id with a slash goes through OpenRouter (e.g. anthropic/claude-opus-5.5), its own key file
  --model <id>      OpenAI model id; key from ~/.config/openai/key (or OPENAI_KEY_FILE); spend over --cap is flagged, not stopped"""
import os
from isaacsim import SimulationApp
# RaytracedLighting + DLSS (NGX bound in by the job script): clean frames on Isaac Sim 6.0
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "anti_aliasing": int(os.environ.get("SIM_AA", 3))})

import sys  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))
import argparse, json, time  # noqa: E401,E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import yam_env as Y  # noqa: E402
import pegboard as PB  # noqa: E402
from ik import Grippers  # noqa: E402
from collide import Monitor  # noqa: E402
from live import Live, StopRun  # noqa: E402
import reward as RW  # noqa: E402
import reflect as RF  # noqa: E402
from openai_client import PRICES, Budget, OpenAIClient, cost_of, estimate_input_tokens  # noqa: E402
import openrouter_client as OR  # noqa: E402
from run_io import RESULTS, new_run_dir  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "mock"))
ap.add_argument("--phases", type=int, default=2, choices=(1, 2), help="1: left to right only; 2: and back (the full FLS task)")
ap.add_argument("--turns", type=int, default=40)
ap.add_argument("--max_calls", type=int, default=6, help="tool calls run per turn (two objects)")
ap.add_argument("--effort", default="medium", help="reasoning effort (reasoning tokens are billed as output)")
ap.add_argument("--max_output", type=int, default=4000, help="hard limit on output tokens per call, reasoning included")
ap.add_argument("--cap", type=float, default=3.00, help="budget per provider (its ledger in $P/results/); passing it is flagged red, the run goes on")
ap.add_argument("--max_minutes", type=float, default=75)
ap.add_argument("--seed", type=int, default=0, help="object orientations")
ap.add_argument("--replay", help="with --model replay: a recorded run's model/turns.json to play back ($0)")
ap.add_argument("--series", help="a rounds series folder: its lessons (written by the model after earlier rounds) are added to the instructions")
args = ap.parse_args()

mock = args.model in ("mock", "replay", "clumsy")      # no API calls, nothing charged
via_openrouter = "/" in args.model             # e.g. anthropic/claude-opus-5.5: OpenRouter, its own key, ledger and cap
d = os.environ.get("RUN_DIR") or new_run_dir("peg_transfer_benchmark", args.model, name=f"seed{args.seed}")
os.makedirs(d, exist_ok=True)
live = Live(d)
price = OR.PRICES.get(args.model, OR.PRICES["anthropic/claude-opus-5.5"]) if via_openrouter else PRICES["standard"]
budget = Budget(os.path.join(RESULTS, "openrouter_spend.json" if via_openrouter else "openai_spend.json"), args.cap, price)
live.emit("meta", title="FLS peg transfer", model=args.model, mock=mock, phases=args.phases, turns=args.turns,
          price=price, cap=args.cap, run_dir=d, job=os.environ.get("SLURM_JOB_ID"), checks=["complete", "across", "back", "drops", "lost", "time"],
          sub=f"{'Stand-in model (mock)' if mock else args.model} → two YAM arms (Isaac Sim) · "
              + ("6 objects across and back" if args.phases == 2 else "6 objects left → right"))
live.emit("status", state="idle")
live.emit("now", text="Starting the simulator", sub="loading the YAM station and the pegboard")

task = PB.PegTransfer(seed=args.seed)
if args.model == "replay":
    client = PB.ReplayClient(args.replay)
elif args.model == "clumsy":
    client = PB.ClumsyClient(task, seed=args.seed)
elif mock:
    client = PB.MockClient(task)
else:
    try:
        client = (OR.OpenRouterClient(args.model, effort=args.effort, max_output_tokens=args.max_output) if via_openrouter
                  else OpenAIClient(args.model, effort=args.effort, max_output_tokens=args.max_output, parallel_tools=True))
    except RuntimeError as e:
        live.emit("stopped", text=str(e))
        live.emit("status", state="stopped")
        print("ERROR", e, flush=True)
        app.close()
        sys.exit(1)

# ---------------------------------------------------------------- scene
sheet = dict(size=(PB.BOARD_MM[0] / 1000, PB.BOARD_MM[1] / 1000), base=PB.board_texture(3.0), px_per_mm=3.0, drape=None)
env = Y.YamDrawEnv(args.usd, sheet=sheet, paint_texture=True, policy_cams=True, render_every=3, tool="none",
                   build_extra=task.build, paper_cam=False)
task.after_reset()
env.tex_gamma, env._tex_version = 3.0, -1      # counters the renderer over-exposing the board texture
env._upload_paint()
SCENE_CAM = ((1.05, -0.305, 0.55), (0.30, -0.305, 0.15))   # facing the robot: its left arm is on the image's right
env.hero.set_world_pose(np.array(SCENE_CAM[0]), Y.look_at(*SCENE_CAM))
g = Grippers(env)
active = ["left"]
contacts = []                    # collisions since the last turn, reported to the model
def on_contact(text):
    contacts.append(text)
    live.emit("collision", text=text)
monitor = Monitor(env, task, on_contact=on_contact)
first_handover = []              # sim time of the first moment both grippers hold an object (for the report's still)
task.on_moment = lambda name: first_handover.append(env.t / 60) if name == "both_holding" and not first_handover else None


def hands():
    out = {}
    for s in ("left", "right"):
        i = task.holding(s)
        out[s] = PB.COLORS[i][0] if i is not None else None
    out["active"] = active[0]
    return out


def frames():
    return {"arm": (Y.rgb(env.hero), (960, 540)), "wrist": (Y.rgb(env.cams[active[0]][0]), (640, 360))}


_step = env.step


def step(*a, **k):  # every sim step: carry held objects, stop check, and (throttled) live state + camera frames
    _step(*a, **k)
    task.carry(g)
    monitor.check()
    live.tick(hands, frames)


env.step = step
who = "Stand-in model" if mock else args.model
shown = dict(title=f"{who}: waiting for the first turn", text="")   # the video's thinking strip
env.panel = lambda: (shown["title"], shown["text"])
env.start_video(f"{d}/video.mp4")
g.hold(60)
live.emit("robot", text="Scene ready: six objects on the left pegs, both grippers open")


def take_photo(turn):
    live.emit("status", state="checking")
    live.emit("now", text="Parking both arms", sub="clearing the overhead camera's view")
    task.doing = "parking for the photo"
    for s in ("left", "right"):
        task.park(g, s)
    live.emit("now", text="Photographing the board", sub="overhead camera, mm grid added")
    img = PB.with_board_grid(task.photo())
    Image.fromarray(img).save(f"{d}/t{turn:02d}_photo.png")
    live.frame(f"photo{turn}", img)
    return img


views = []               # wrist-camera photos taken after picks and hand-overs, shown with the next turn's photo


def wrist_view(turn, side, what):
    """Photo from the wrist camera of the arm now holding the object, for the model to check the grasp next turn."""
    img = np.asarray(Image.fromarray(Y.rgb(env.cams[side][0])).resize((640, 360), Image.LANCZOS))
    name = f"t{turn:02d}_wrist{len(views) + 1}.png"
    Image.fromarray(img).save(f"{d}/{name}")
    live.frame(f"wrist_check{len(views) + 1}", img)
    views.append(dict(caption=f"{side} wrist camera, just after {what}", file=name, img=img))


def run_call(name, a):
    """One tool call -> (ok, message)."""
    if name == "pick":
        side = a.get("arm") if a.get("arm") in ("left", "right") else None
        if side is None:
            return False, f"pick: arm must be left or right, got {a.get('arm')!r}"
        active[0] = side
        live.emit("now", text=f"{side.capitalize()} arm picking", sub=f"closing at ({a['x_mm']:g}, {a['y_mm']:g}) mm")
        return task.pick(g, side, float(a["x_mm"]), float(a["y_mm"]))
    if name == "hand_over":
        h = [s for s in ("left", "right") if task.holding(s) is not None]
        if len(h) == 1:
            active[0] = "right" if h[0] == "left" else "left"
        live.emit("now", text="Mid-air transfer", sub=f"{h[0] if h else '?'} arm to {active[0]} arm")
        return task.hand_over(g)
    if name == "place":
        h = [s for s in ("left", "right") if task.holding(s) is not None]
        if h:
            active[0] = h[0]
        live.emit("now", text=f"{active[0].capitalize()} arm placing", sub=f"hole over ({a['x_mm']:g}, {a['y_mm']:g}) mm")
        return task.place(g, float(a["x_mm"]), float(a["y_mm"]), float(a.get("height_mm", 50.0)))
    return False, f"unknown tool {name}"


# ---------------------------------------------------------------- turns
system = PB.system_prompt(args.turns, args.phases, args.max_calls)
if args.series:
    _s = RF.load_series(os.path.join(args.series, "series.json"))
    system += RF.prompt_section(_s.get("lessons", []), [dict(round=r["round"], reward=r["reward"]) for r in _s.get("rounds", [])])
open(f"{d}/system_prompt.txt", "w").write(system)
turns, results = [], []
run_cost, totals = 0.0, dict(input=0, cached=0, output=0, reasoning=0)
action_s = [0.0]          # simulated time spent on actions (parking for the photos is not part of the task)
stopped, finished, errors_in_row = None, False, 0
t_start = time.time()


def spend_event():
    live.emit("spend", run=run_cost, session=(run_cost if mock else budget.spent()), cap=args.cap, simulated=mock)


HIST = []                    # this model's earlier rounds in the series, for the live reward comparison
if args.series:
    HIST = [r["reward"] for r in RF.load_series(os.path.join(args.series, "series.json")).get("rounds", [])]
CEILING = 77 if args.phases == 2 else 89     # stand-in (exact positions, same robot) on this task


def score_event():
    sc = task.score(action_s[0], args.phases, len(monitor.events))
    sc["collision_log"] = monitor.summary()
    rew = RW.compute(sc, task.actions, max(1, len(turns)), need=6 * args.phases)   # v2: the reward so far, every turn
    live.emit("score", score=sc, partial=True, reward=rew, history=HIST, ceiling=CEILING, cost=run_cost)


try:
    spend_event()
    for turn in range(1, args.turns + 1):
        if time.time() - t_start > args.max_minutes * 60:
            stopped = f"time limit ({args.max_minutes:g} min)"
            break
        photo = take_photo(turn)
        shown_views, views[:] = views[-3:], []
        user = PB.turn_prompt(turn, args.turns, task, results, [v["caption"] for v in shown_views])
        images = [photo] + [v["img"] for v in shown_views]
        worst = budget.worst_case(estimate_input_tokens([system, user], len(images), PB.TOOLS), args.max_output)
        if not mock:   # the cap is a warning, not a stop: the user stops runs from the dashboard
            if not budget.allows(worst):
                live.emit("budget", text=f"Over budget: ${budget.spent():.4f} spent + ${worst:.4f} worst case passes ${args.cap:.2f}; "
                                         "continuing (press STOP to end the run)")
            else:
                live.emit("budget", text=f"${budget.spent():.4f} spent + ${worst:.4f} worst case ≤ ${args.cap:.2f}: OK")
        else:
            live.emit("budget", text=f"Stand-in model: $0 charged (a real call's worst case would be ${worst:.4f})")
        live.emit("status", state="thinking")
        live.emit("now", text=f"Model thinking (turn {turn})", sub="reading the photo")
        live.emit("turn_start", turn=turn)
        res = client.call(system, user, images, PB.TOOLS,
                          on_event=lambda kind, text: live.emit("thinking", turn=turn, text=text) if kind in ("reasoning", "text")
                          else live.emit("model", text=text),
                          should_stop=live.stop_requested)
        usage = res.get("usage")
        if usage:
            cost = usage["cost_usd"] if usage.get("cost_usd") is not None else cost_of(usage, price)
        elif not mock and (res.get("cancelled") or str(res.get("error", "")).startswith("connection")):
            cost = worst  # unknown usage for an interrupted call: book the worst case so the cap stays safe
        else:
            cost = 0.0
        run_cost += cost
        if usage:
            for k in totals:
                totals[k] += usage.get(k, 0)
        if not mock and cost:
            budget.record(cost, benchmark="peg_transfer", model=args.model, run=d, turn=turn, usage=usage,
                          note="estimated (interrupted)" if not usage else "")
        if usage:
            live.emit("usage", turn=turn, latency=res.get("seconds", 0), cost=cost, **usage)
        spend_event()
        rec = dict(turn=turn, reasoning=res.get("reasoning"), text=res.get("text"), calls=res.get("calls"), error=res.get("error"),
                   usage=usage, cost=cost, seconds=res.get("seconds"), results=[],
                   views=[dict(caption=v["caption"], file=v["file"]) for v in shown_views])
        turns.append(rec)
        shown.update(title=f"{who} · turn {turn} · its reasoning", text=(res.get("reasoning") or res.get("text") or "").strip())
        if res.get("cancelled"):
            raise StopRun(open(live.stop_path).read().strip() if os.path.exists(live.stop_path) else "STOP pressed")
        if res.get("http_status") in (401, 403, 404):   # bad key, no access, or unknown model: retrying won't help
            stopped = {401: "the API key was rejected (HTTP 401)", 403: "this key has no access (HTTP 403)",
                       404: f"model '{args.model}' not found (HTTP 404)"}[res["http_status"]]
            live.emit("model", text=f"Turn {turn}: {res.get('error')}"[:300])
            break
        calls = [(n, a) for n, a in (res.get("calls") or []) if n][:args.max_calls]
        if res.get("error") or not calls:
            errors_in_row += 1
            live.emit("model", text=f"Turn {turn}: {res.get('error') or 'no tool call'}"[:300])
            results = ["your reply had no usable tool call; nothing happened"]
            if errors_in_row >= 2:
                stopped = "two turns in a row without a usable tool call"
                break
            continue
        errors_in_row, results = 0, []
        contacts.clear()
        for name, a in calls:
            a = dict(a or {})
            note = str(a.pop("note", ""))[:300]
            live.emit("reply", turn=turn, call=dict(tool=name, **a), note=note)
            shown["title"] = (f"{who} · turn {turn} · doing {name}({', '.join(f'{k}={v}' for k, v in a.items())})"
                              + (f"  \"{note}\"" if note else ""))[:160]
            if name == "done":
                finished = True
                results.append("done")
                break
            t_sim = env.t
            try:
                ok, msg = run_call(name, a)
            except (KeyError, TypeError, ValueError) as e:
                ok, msg = False, f"{name}: bad arguments ({e})"
            action_s[0] += (env.t - t_sim) / 60
            if ok and name == "pick":
                wrist_view(turn, a["arm"], f"pick({a['x_mm']:g}, {a['y_mm']:g})")
            elif ok and name == "hand_over":
                wrist_view(turn, active[0], "hand_over")
            results.append(msg)
            live.emit("robot" if ok else "model", text=msg)
            if not ok:
                results.append("the rest of this turn's calls were skipped")
                break
            if args.phases == 2 and task.phase == 1 and len(task.transfers["left->right"]) >= 6:
                task.phase = 2
                live.emit("robot", text="All six across: phase 2, transfer them back to the left")
                results.append("all six objects are across: now transfer them back to the left pegs")
        if contacts:
            results.append("collisions during this turn: " + "; ".join(c.split(": ", 1)[-1] for c in contacts[:6]))
        rec["results"] = results
        json.dump(turns, open(f"{d}/turns.json", "w"), indent=1, default=str)
        score_event()
        if finished:
            break
except StopRun as e:
    stopped = str(e)

# ---------------------------------------------------------------- wrap up
live.stopping = True
sim_s = action_s[0]
if stopped and not finished:
    live.emit("stopped", text=f"Stopped: {stopped}")
    for s in ("left", "right"):
        t = g.tool(s)
        g.line_to(s, [t[0], t[1], max(t[2], PB.SAFE_Z)])   # lift clear and hold (a held object stays held)
    live.emit("robot", text="Arms lifted clear and holding still")
    live.emit("now", text="Stopped", sub="arms holding; saving results")
else:
    live.emit("now", text="Scoring", sub="checking where every object ended up")
    for s in ("left", "right"):
        task.park(g, s)
score = task.score(sim_s, args.phases, len(monitor.events))
score["collision_log"] = monitor.summary()
rew = RW.compute(score, task.actions, len(turns), need=6 * args.phases)
live.emit("score", score=score, reward=rew, history=HIST, ceiling=CEILING, cost=run_cost)
live.emit("robot", text=f"Reward {rew['total']:.1f} ({', '.join(f'{k} {v:g}' for k, v in rew['parts'].items())})")
for _ in range(20):
    env.step()
env.stop_video()
final = PB.with_board_grid(task.photo())
Image.fromarray(final).save(f"{d}/board_final.png")
result = dict(benchmark="peg_transfer", model=args.model, mock=mock, phases=args.phases, seed=args.seed, finished=finished,
              stopped=stopped, turns=len(turns), usage=totals, cost=run_cost, cost_simulated=mock, score=score,
              reward=rew, actions=task.actions, effort=args.effort,
              sim_seconds=sim_s, sim_seconds_total=env.t / 60, wall_seconds=time.time() - t_start,
              first_handover_sim_s=first_handover[0] if first_handover else None)
json.dump(result, open(f"{d}/result.json", "w"), indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("RESULT", json.dumps(dict(model=args.model, pass_=score["pass"], transfers=score["checks"]["complete"]["text"],
                                drops=score["drops"], sim_s=round(sim_s), cost=round(run_cost, 4), stopped=stopped,
                                reward=rew["total"])), flush=True)
live.emit("robot", text="Saved: video.mp4, board_final.png, turns.json, result.json")
if stopped and not finished:
    live.emit("status", state="stopped")
else:
    live.emit("now", text="Finished", sub=f"{'simulated ' if mock else ''}cost ${run_cost:.4f} for {len(turns)} model turns")
    live.emit("status", state="done")
app.close()
