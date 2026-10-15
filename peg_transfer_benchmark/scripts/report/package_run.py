"""Tidy a finished run into a clean folder (run by slurm/run_pegs.sbatch after the run, in Isaac's python for PIL +
ffmpeg):

  README.md            summary: model, date, score table, cost, tokens, what happened
  video/               full.mp4 (the recording), preview_6x.mp4, handover.jpg (first mid-air transfer, if any)
  photos/              turn_01.png ... exactly what the model saw each turn, and board_final.png
  model/               transcript.md (each turn: reasoning, calls, results, tokens, cost), system_prompt.txt, turns.json
  score/               result.json
  live/                the dashboard's data (events.jsonl, last camera frames, dashboard.txt/log)

and refresh <model folder>/index.md, one line per run.

  python package_run.py <run_dir>"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

d = os.path.abspath(sys.argv[1])
if not os.path.exists(os.path.join(d, "result.json")):
    sys.exit(f"{d}: no result.json (run did not finish); leaving it as it is")
P = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(P, "sim_pylibs"))


def sub(*parts):
    p = os.path.join(d, *parts)
    os.makedirs(os.path.dirname(p) if "." in os.path.basename(p) else p, exist_ok=True)
    return p


def move(src, dst):
    if os.path.exists(src):
        shutil.move(src, dst)


r = json.load(open(os.path.join(d, "result.json")))
turns = json.load(open(os.path.join(d, "turns.json"))) if os.path.exists(os.path.join(d, "turns.json")) else []

# ---- files into place
for f in sorted(glob.glob(os.path.join(d, "t[0-9][0-9]_photo.png"))):
    move(f, sub("photos", f"turn_{os.path.basename(f)[1:3]}.png"))
for f in sorted(glob.glob(os.path.join(d, "t[0-9][0-9]_wrist[0-9]*.png"))):
    move(f, sub("photos", "wrist", os.path.basename(f)))
move(os.path.join(d, "board_final.png"), sub("photos", "board_final.png"))
move(os.path.join(d, "system_prompt.txt"), sub("model", "system_prompt.txt"))
if turns:
    shutil.copy(os.path.join(d, "turns.json"), sub("model", "turns.json"))
for f in ["events.jsonl", "dashboard.txt", "dashboard.log", "STOP"] + [os.path.basename(x) for x in glob.glob(os.path.join(d, "live_*.jpg"))]:
    move(os.path.join(d, f), sub("live", f))
move(os.path.join(d, "video.mp4"), sub("video", "full.mp4"))

# ---- preview video and the hand-over still
ffmpeg = None
try:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
except Exception as e:
    print("no ffmpeg for the preview:", e)
full = os.path.join(d, "video", "full.mp4")
if ffmpeg and os.path.exists(full):
    subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", full, "-vf", "setpts=PTS/6,scale=960:-2", "-r", "30", "-an",
                    "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p", sub("video", "preview_6x.mp4")])
    ho = r.get("first_handover_sim_s")
    if ho is not None:
        subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-ss", f"{ho:.2f}", "-i", full, "-frames:v", "1",
                        "-vf", "crop=1280:720:0:0", sub("video", "handover.jpg")])

# ---- transcript
lines = [f"# Transcript: {r.get('model')} on the FLS peg transfer", ""]
for t in turns:
    u = t.get("usage") or {}
    lines += [f"## Turn {t['turn']}", "",
              f"![what the model saw](../photos/turn_{t['turn']:02d}.png)", ""]
    if t.get("views"):
        lines += ["Wrist-camera photos it also saw (from the turn before):", "",
                  " ".join(f"![{v['caption']}](../photos/wrist/{v['file']})" for v in t["views"]), ""]
    lines += [
              f"*{u.get('input', 0):,} tokens in ({u.get('cached', 0):,} cached), {u.get('output', 0):,} out "
              f"({u.get('reasoning', 0):,} reasoning), ${t.get('cost', 0):.4f}, {t.get('seconds') or 0:.1f} s*", ""]
    if t.get("reasoning"):
        lines += ["**Reasoning summary**", "", t["reasoning"].strip(), ""]
    if t.get("calls"):
        lines += ["**Calls**", ""] + [f"- `{n}({', '.join(f'{k}={v}' for k, v in (a or {}).items() if k != 'note')})` "
                                       f"{(a or {}).get('note', '')}" for n, a in t["calls"]] + [""]
    if t.get("results"):
        lines += ["**What happened**", ""] + [f"- {x}" for x in t["results"]] + [""]
    if t.get("error"):
        lines += [f"**Error:** {t['error']}", ""]
open(sub("model", "transcript.md"), "w").write("\n".join(lines))

# ---- result into score/, README
move(os.path.join(d, "result.json"), sub("score", "result.json"))
if os.path.exists(os.path.join(d, "turns.json")):
    os.remove(os.path.join(d, "turns.json"))          # copied to model/turns.json above
sc = r.get("score", {})
checks = sc.get("checks", {})
u = r.get("usage", {})
when = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(os.path.join(d, "score", "result.json"))))
readme = [f"# {r.get('model')} · FLS peg transfer · {when}", "",
          f"**{'PASS' if sc.get('pass') else 'FAIL'}**: {sc.get('n_pass', 0)} of {len(checks)} checks"
          + (f" · stopped: {r['stopped']}" if r.get("stopped") else ""), "",
          "| Check | Result | Pass |", "|---|---|---|"]
readme += [f"| {v.get('label', k)} | {v.get('text')} | {'yes' if v.get('pass') else 'no'} |" for k, v in checks.items()]
rw = r.get("reward")
if rw:
    readme += ["", f"**Reward: {rw['total']:.1f}** ({rw['version']}; higher is better, 100 would need a perfect, instant run)"
               + (f" · model-only {rw['model_only']:.1f} (robot-caused contacts excluded)" if rw.get("model_only") is not None else ""), "",
               "| Part | Points |", "|---|---|"] + [f"| {k} | {v:g} |" for k, v in rw["parts"].items()]
readme += ["", f"- Cost: **${r.get('cost', 0):.4f}**" + (" (simulated: stand-in model, nothing charged)" if r.get("cost_simulated") else ""),
           f"- Tokens: {u.get('input', 0):,} in ({u.get('cached', 0):,} cached), {u.get('output', 0):,} out ({u.get('reasoning', 0):,} reasoning)",
           f"- Model turns: {r.get('turns')} · robot time {r.get('sim_seconds', 0):.0f} s · wall time {r.get('wall_seconds', 0) / 60:.1f} min",
           "", "## Files", "",
           "- `video/preview_6x.mp4`: the whole run, 6x speed; `video/full.mp4`: full recording" +
           ("; `video/handover.jpg`: the first mid-air transfer" if os.path.exists(os.path.join(d, "video", "handover.jpg")) else ""),
           "- `model/transcript.md`: every turn, with the photo the model saw, its reasoning, its calls and what happened",
           "- `photos/`: the model's view each turn (overhead; `wrist/`: the wrist-camera checks), and the board at the end",
           "- `score/result.json`: every number above", "- `live/`: what the dashboard showed"]
col = sc.get("collision_log", {})
if col.get("events"):
    readme += ["", "## Collisions", ""] + [f"- {e['t']:.1f} s: {e['what']}" for e in col["events"]]
open(os.path.join(d, "README.md"), "w").write("\n".join(readme) + "\n")

# ---- index of runs for this model
model_dir = os.path.dirname(d)
rows = []
for rd in sorted(glob.glob(os.path.join(model_dir, "*", "score", "result.json"))):
    rr = json.load(open(rd))
    run = os.path.basename(os.path.dirname(os.path.dirname(rd)))
    s2 = rr.get("score", {})
    tr = s2.get("checks", {}).get("complete", {}).get("text", "?")
    rows.append(f"| [{run}]({run}/README.md) | {'PASS' if s2.get('pass') else 'fail'} | {tr} | {s2.get('drops', '?')} | "
                f"{s2.get('checks', {}).get('collisions', {}).get('text', '?')} | ${rr.get('cost', 0):.4f} |")
open(os.path.join(model_dir, "index.md"), "w").write(
    f"# {r.get('model')} runs\n\n| Run | Result | Transfers | Drops | Collisions | Cost |\n|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
print("packaged", d)
