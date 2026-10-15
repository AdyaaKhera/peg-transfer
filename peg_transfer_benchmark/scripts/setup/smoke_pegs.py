"""Pegboard check: build the scene, photograph it, then one scripted transfer (pick left, hand over, place on a right
peg) and one deliberately missed placement. Saves every camera view at each stage."""
import os
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "anti_aliasing": int(os.environ.get("SIM_AA", 3))})

import sys, time  # noqa: E401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))
import argparse
import numpy as np
from PIL import Image
import yam_env as Y
import pegboard as PB
from ik import Grippers

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
task = PB.PegTransfer(seed=0)
sheet = dict(size=(PB.BOARD_MM[0] / 1000, PB.BOARD_MM[1] / 1000), base=PB.board_texture(3.0), px_per_mm=3.0, drape=None)
print("SMOKE creating env", flush=True)
env = Y.YamDrawEnv(args.usd, sheet=sheet, paint_texture=True, policy_cams=True, render_every=3, tool="none",
                   build_extra=task.build, paper_cam=False)
task.after_reset()
env.tex_gamma, env._tex_version = 3.0, -1
env._upload_paint()
pos = (1.05, -0.305, 0.55)
env.hero.set_world_pose(np.array(pos), Y.look_at(pos, (0.30, -0.305, 0.15)))
from collide import Monitor
from isaacsim.sensors.camera import Camera
g = Grippers(env)
monitor = Monitor(env, task, on_contact=lambda t: print("SMOKE COLLISION", t, flush=True))
close = Camera("/World/close_cam", name="close_cam", resolution=(960, 540), position=np.array([0.72, -0.305, 0.36]),
               orientation=Y.look_at((0.72, -0.305, 0.36), PB.TRANSFER))
close.initialize()
close.set_clipping_range(0.01, 10.0)
_step = env.step
def step(*a, **k):
    _step(*a, **k)
    task.carry(g)
    monitor.check()
env.step = step
print("SMOKE env ready", flush=True)
for sd in ("left", "right"):     # the URDF kinematics must match the simulator
    q = env.q()[env.idx[sd]]
    p_fk = g.solver.pad_pose(sd, q)[0]
    print(f"SMOKE FK check {sd}: model {np.round(p_fk, 4)} sim {np.round(g.tool(sd), 4)} diff {np.linalg.norm(p_fk - g.tool(sd)) * 1000:.2f} mm", flush=True)
def save(tag):
    Image.fromarray(Y.rgb(env.hero)).save(f"{args.out}/{tag}_hero.png")
    Image.fromarray(Y.rgb(close)).save(f"{args.out}/{tag}_close.png")
    for n in ("left", "right"):
        Image.fromarray(Y.rgb(env.cams[n][0])).save(f"{args.out}/{tag}_{n}.png")
task.on_moment = lambda name: save(f"t{len(task.transfers['left->right'])}_{name}")
g.hold(60)
print("SMOKE settled:", [task.where(i)["state"] for i in range(6)], flush=True)
for s in ("left", "right"):
    task.park(g, s)
img = task.photo()
Image.fromarray(PB.with_board_grid(img)).save(f"{args.out}/photo.png")
save("start")
t0 = time.time()
if os.environ.get("REACH_MAP"):
    from ik import grasp_axes, DOWN
    import math
    for lean_deg in (22, 0):
        for i in range(6):
            row = []
            for k, bar in enumerate(task.bars_world(i)):
                a, c = grasp_axes(bar["dir"], bar["out"], math.radians(lean_deg)) if lean_deg else (DOWN, np.cross(DOWN, bar["dir"]))
                pt = bar["mid"] + np.array([0, 0, 0.03])
                g.set_axes("left", a, c)
                task.lift(g, "left")
                g.line_to("left", pt)
                ok = g.go("left", pt, max_steps=120)
                d = np.linalg.norm(g.tool("left") - pt) * 1000
                aa, cc = g.current_axes("left")
                row.append(f"bar{k} out=({bar['out'][0]:+.2f},{bar['out'][1]:+.2f}) err={d:.0f}mm tilt_err={math.degrees(math.acos(np.clip(aa @ a, -1, 1))):.0f}deg")
                task.park(g, "left")
            print(f"SMOKE REACH lean={lean_deg} obj{i}: " + " | ".join(row), flush=True)
    app.close()
    sys.exit(0)
def mid(i):
    peg = PB.board_to_world([PB.PEGS_MM[task.where(i)["peg"]]])[0]
    b = max(task.bars_world(i), key=lambda b: float(np.linalg.norm(b["mid"][:2] - peg)))
    return PB.world_to_board(b["mid"][:2])[0]
for obj, peg in ((0, 6), (1, 7)):
    x, y = mid(obj)
    print("SMOKE", task.pick(g, "left", round(x, 1), round(y, 1)), flush=True)
    save(f"o{obj}_picked")
    print("SMOKE", task.hand_over(g), flush=True)
    px, py = PB.PEGS_MM[peg]
    print("SMOKE", task.place(g, px, py), flush=True)
    save(f"o{obj}_placed")
for s in ("left", "right"):
    task.park(g, s)
img = task.photo()
Image.fromarray(PB.with_board_grid(img)).save(f"{args.out}/photo_end.png")
print(f"SMOKE OK sim {env.t / 60:.1f}s wall {time.time() - t0:.0f}s states {[task.where(i)['state'] for i in range(6)]}", flush=True)
print("SMOKE collisions", monitor.summary(), flush=True)
app.close()
