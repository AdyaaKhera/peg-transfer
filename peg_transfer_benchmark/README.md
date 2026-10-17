# FLS peg transfer: two robot arms run a surgical skills test, directed by an AI model

The peg transfer is the first task of the FLS exam (Fundamentals of Laparoscopic Surgery). Six small triangular
objects sit on pegs on the left of a board. Each object has to be lifted with the left grasper, passed in mid-air to
the right grasper, and placed over a peg on the right. Then all six go back the same way.

In this benchmark, two I2RT YAM arms in Isaac Sim 6.0.1 do the handling and an AI model (GPT-6.1 Sol or Claude Opus
5.5 in v2) directs them. Each turn the model sees an overhead photo of the board with a millimetre grid and calls tools:

| Tool | What the robot does |
|---|---|
| `pick(arm, x_mm, y_mm, note)` | Moves that gripper to the point and pinches the nearest corner of a triangle there (aim on the corner's centre line, about 12 mm in from its tip), then lifts it. Each arm works only its own half of the board |
| `hand_over(note)` | Brings the held object to the middle of the board; the other gripper takes it by a free corner |
| `place(x_mm, y_mm, height_mm, note)` | Lowers the held object so its hole is over the point, until its underside is `height_mm` (2-60 mm) above the peg top, then lets go. Refused unless the point is within 4 mm of a peg's centre |
| `done(note)` | Ends the run |

`note` is one short sentence for the operator. The model may chain up to 6 calls in one turn, so pick, hand_over and
place can move one object per turn; the first call that fails ends the turn. A pick that misses the grip point closes
on nothing, and a refused place leaves the object in the gripper. After a miss the model is told only roughly which
way it was off, in words ("about a centimetre to the left and a little lower in the photo").

## Physics, grasping and collisions

The YAM gripper is far bigger than a laparoscopic grasper. Its fingers are 94 mm long and widen to 65 mm, and its body
is 10 cm wide. The task is scaled to fit it (all values in `core/pegboard.py`):
- solid triangles with 70 mm sides, 20 mm thick, with a hole through the middle (11 mm inscribed radius)
- 10 mm wide, 40 mm tall pegs: on the left two columns of three (x 130 and 260 mm, y 60, 150 and 240 mm), on the
  right six on a ring of radius 95 mm around (605, 150) mm
- an 810 x 300 mm board, split into a left and a right field

The pegs and objects are rigid bodies. A released object slides down a peg if its hole is over it.

**Grasping.** The robot's URDF has visual meshes but no collision shapes, so grasps are decided geometrically. Every
measurement comes from the meshes (`scripts/setup/robot_geometry.py`), including the finger pads.
- The gripper pinches a corner of the triangle, square to its surface (a tilted triangle is pinched tilted).
- A grasp holds when the finger pads close within 6 mm of the corner's centre line and 4-22 mm in from its tip.
- A triangle tilted more than 35 degrees (standing on its edge) can no longer be pinched.
- While held, the object moves with the gripper on every simulation step. Once the fingers open, physics takes over.

**Mid-air transfer.**
1. The holding arm brings the object to the transfer point over the middle of the board, 11 cm above it.
2. It turns a free corner toward the other arm. If the other arm can't reach it, it tries up to six presentations
   (other corners, a 40 degree turn, or shifted 5 cm aside or 4 cm ahead).
3. The other arm approaches 35 degrees below level (then 15 degrees) and pinches that corner, with the same test as a
   pick. It backs out of any approach that would bring the two arms within 6 mm of each other.
4. Both grippers hold the object for a moment, then the first lets go and backs away. If no presentation works, the
   object stays in the first gripper and the model is told to re-grip it by another corner.

**Placing.** The new holder levels the object, lines its hole up over the peg (several small corrections), lowers it
with a guarded, slowing motion, and lets go `height_mm` above the peg top.

**Collision monitor** (`core/collide.py`). Because PhysX can't see the arms, every other simulation step checks, using
the arms' real mesh surfaces placed at the links' live poses:
- arm against arm, the board, any peg or any object (fingers may touch only the object they hold)
- the held object against pegs, other objects and the board

A contact is a real overlap of more than 0.5 mm. Every contact is logged with its time and the phase it happened in,
shown on the dashboard, reported to the model in the next turn and counted in the score. Contacts less than 1.5 s
apart count as one incident.

## Score

| Check | Pass |
|---|---|
| Transfers | 12 of 12: six across and six back, each placed on a peg on the other side after a mid-air hand-over |
| Left -> right / right -> left | 6 of 6 each way |
| Drops | 0. A drop is a release that doesn't end over a peg |
| Lost off board | 0. In FLS, objects dropped out of the field can't be recovered |
| Robot time | Simulated time the arms spend on actions, within 300 s (the FLS time limit). Model thinking time is not counted |
| Collisions | 0. This check counts every logged contact event; the reward and the write-up group events less than 1.5 s apart into one incident, so their numbers are lower |

Each run also gets a reward out of 100 (`core/reward.py`): completion 40, speed 20, precision 15, gentleness 5,
safety 10 and efficiency 10, minus 10 per drop and 20 per lost object. The model-only reward is the same but counts
only contacts during actions whose geometry the model chose (picks), not ones during motions the robot plans itself.

`--phases 1` runs only the left-to-right half.

## Cost

In the v2 comparison (20 turns, effort medium) a GPT-6.1 Sol round cost about $0.10 and a Claude Opus 5.5 round
(via OpenRouter) about $0.70-0.90, including the reflection between rounds.

Each provider has its own spend ledger (`$P/results/openai_spend.json`, `$P/results/openrouter_spend.json`) and a
budget (`--cap`, default $3.00). Going over the budget does not stop a run: the dashboard's spend turns red and the
log says so, and you stop the run yourself with STOP.

## Run it

You need, in one working folder `$P` (default `~/isaac-bench`):
- `repo/`: this repository (`git clone <this repo> $P/repo`)
- `isaac-sim_6.0.1.sif`: the Isaac Sim 6.0.1 Apptainer image, and a GPU with RTX (ray-tracing) cores, e.g. an A10
- `third_party/i2rt/`: the [I2RT](https://github.com/i2rt-robotics/i2rt) repository, for the YAM station URDF
- `assets/yam_station.usd`: that URDF (`yam_station_linear_4310_d405`) converted to USD with Isaac Sim's URDF importer
- `sim_pylibs/`: extra Python packages for Isaac Sim's Python (`pip install --target $P/sim_pylibs imageio imageio-ffmpeg`)
- an API key for the provider you use, in the key file named below (`chmod 600`). Keys are only ever read from
  that file or the environment; never put one in the repo.

Edit the `#SBATCH -A` line in `slurm/*.sbatch` to your Slurm account, and the partition (`-p a10`) and QOS
(`-q standby`) to ones your cluster has with RTX GPUs. The job log goes to `logs/` in the folder you submit from, so
create it first (`mkdir -p logs`). Then, from `$P/repo`:

```bash
python3 peg_transfer_benchmark/scripts/setup/openai_preflight.py <model id>   # free check of an OpenAI key and model (OpenAI only)
MODEL=mock sbatch peg_transfer_benchmark/slurm/run_pegs.sbatch                # stand-in model: real sim, $0
MODEL=clumsy sbatch peg_transfer_benchmark/slurm/run_pegs.sbatch              # error-prone stand-in, $0
MODEL=<model id> sbatch peg_transfer_benchmark/slurm/run_pegs.sbatch          # a model (default 40 turns; v2 used --turns 20)
# the v2 comparison: round r plays seed r-1 with the lessons from earlier rounds, then writes new lessons
ROUND=1 MODEL=gpt-6.1-sol sbatch peg_transfer_benchmark/slurm/run_pegs.sbatch --turns 20
ROUND=1 MODEL=anthropic/claude-opus-5.5 sbatch peg_transfer_benchmark/slurm/run_pegs.sbatch --turns 20
```

The key goes in `~/.config/openai/key` (GPT) or `~/.config/openrouter/key` (Claude via OpenRouter), `chmod 600`.
`SEED=<n>` changes the objects' starting orientations.

To watch a run, open `dashboard.txt` in its run folder (`live/dashboard.txt` once the run has been packaged). In VS
Code's Ports panel, forward the `<node>:<port>` it lists, then open its link. The dashboard shows:
- the scene camera (facing the robot, so the robot's left arm appears on the right of the image) and the wrist camera
  of whichever arm is working
- the model's reasoning and every call
- what each gripper holds
- spend against the budget
- the running score and reward

To stop a run, press **STOP**, press Esc, or run `touch <run dir>/STOP`. The arms lift clear and hold, and the results
are saved. If the run hasn't stopped after 45 s, the dashboard server cancels the job.

## Outputs

A single run writes to `$P/results/peg_transfer_benchmark/<model>/<date>_seed<N>_j<job>/`; a comparison round
(`ROUND=r`) writes to `$P/results/peg_transfer_benchmark/comparison_v2/<model>/round_<rr>_seed<N>_j<job>/` (`<rr>`
is the round with two digits, e.g. `round_01_seed0_j12345`). When the run
ends, `scripts/report/package_run.py` sorts it into:

| Path | Contents |
|---|---|
| `README.md` | Summary: model, date, score table, cost, tokens, what happened |
| `score/result.json` | Score, reward, cost and token usage |
| `model/turns.json`, `model/transcript.md`, `model/system_prompt.txt` | Each turn's reasoning, calls and their results |
| `photos/turn_01.png`, ..., `photos/board_final.png` | Exactly what the model saw each turn, and the board at the end |
| `photos/wrist/` | The wrist-camera photos taken after grasps |
| `video/full.mp4`, `video/preview_6x.mp4`, `video/handover.jpg` | Recording of the run, a 6x preview, the first mid-air transfer |
| `live/` | The dashboard's data (`events.jsonl`, last camera frames, `dashboard.txt`) |

A comparison series also keeps `series.json` (rewards, metrics and the current lessons) and, per round,
`round_<rr>_debrief.txt`, `round_<rr>_lessons.json` and `round_<rr>_reflection_raw.json` (the model's raw reply) in
`comparison_v2/<model>/`. `scripts/run/reflect_only.sh <round> <model>` re-runs just the reflection for a round.

## Simulator notes

- **Isaac Sim version:** runs on **Isaac Sim 6.0.1** (`$P/isaac-sim_6.0.1.sif`). Since October 2026 the A10 nodes have NVIDIA driver 595.91, and Isaac Sim 5.1 crashes at startup on that driver (`librtx.scenedb`). Choose a different image with `SIF=...`.
- **NGX library:** Apptainer's `--nv` doesn't pass in `libnvidia-ngx.so`. Without it the DLSS denoiser fails and every frame is grainy, so the job scripts bind it in. The renderer is `RaytracedLighting`.
- **Texture brightness:** the renderer over-exposes the board texture and object colours, so the texture is uploaded with an extra gamma of 3 and the materials are pre-darkened.

## Code

| Path | Contents |
|---|---|
| `core/pegboard.py` | Board, pegs and objects; grasp and carry; pick, hand_over and place; where each object ended up; score; prompts and tools; the stand-in model |
| `core/kin.py` | Forward kinematics and global IK (scipy least squares) from the YAM URDF |
| `core/ik.py` | Differential IK for both grippers: finger-pad point plus a chosen approach and closing direction, keeping the wrist inside its ±120° limit |
| `core/collide.py` | Collision monitor |
| `scripts/setup/robot_geometry.py` | Measures the robot from its meshes and writes `core/yam_geometry.npz` (I2RT meshes, MIT License; see THIRD_PARTY_NOTICES.md) |
| `scripts/setup/reach_check.py` | Offline check that every pick, hand-over and place pose is reachable |
| `scripts/setup/openai_preflight.py` | Free check of an OpenAI key and model id |
| `core/openai_client.py`, `core/openrouter_client.py` | OpenAI Responses API and OpenRouter chat API over https (streaming, cancel, several tool calls per reply) and the spend ledger |
| `core/reward.py`, `core/reflect.py` | The reward, and the debrief and lessons between rounds |
| `core/live.py`, `core/run_io.py` | Events, camera frames and the STOP file for the dashboard; saving run files |
| `core/yam_env.py`, `core/paint_sim.py` | The YAM scene, adapted from an earlier robot-painting benchmark of ours (not included), with both grippers and a task hook. The board is the textured "sheet" |
| `scripts/run/gpt_pegs.py`, `scripts/run/reflect_round.py`, `scripts/run/reflect_only.sh` | The benchmark run, the reflection that writes the next round's lessons, and re-running only the reflection |
| `scripts/report/` | Packaging a run, and the plots |
| `scripts/setup/smoke_pegs.py` | Scene check: one scripted transfer and one deliberate miss |
| `scripts/servers/dashboard_server.py`, `dashboard/index.html` | Live dashboard (token-protected) |
| `slurm/run_pegs.sbatch`, `slurm/smoke.sbatch` | Jobs. Both use Isaac Sim 6.0.1 with the NGX library bound in (see Simulator notes) |
