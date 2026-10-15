# FLS peg transfer: can GPT run a surgical skills test with two robot arms?

The peg transfer is the first task of the FLS exam (Fundamentals of Laparoscopic Surgery), which surgeons in the US pass before they can be board certified. Six small triangular objects sit on pegs on the left of a board. Each object has to be lifted with the left grasper, passed in mid-air to the right grasper, and dropped over a peg on the right. Then all six go back the same way.

In this benchmark, two black I2RT YAM arms in Isaac Sim 6.0 do the handling and GPT directs them. Each turn it sees an overhead photo of the board with a millimetre grid and calls tools:

| Tool | What the robot does |
|---|---|
| `pick(arm, x_mm, y_mm)` | Moves that gripper over the point, turns it to close across the nearest bar, descends and closes |
| `hand_over()` | Brings the held object to the centre; the other gripper takes it by the bar farthest from the first |
| `place(x_mm, y_mm)` | Lowers the held object so its hole is over the point, then lets go |
| `done()` | Ends the run |

GPT may chain several calls in one turn, so pick → hand_over → place can move one object per turn. The first call that fails ends the turn. Its coordinates decide where the gripper goes, so a pick a few millimetres off a bar closes on nothing, and a place a few millimetres off a peg drops the object.

## Physics, grasping and collisions

The YAM gripper is far bigger than a laparoscopic grasper. Its fingers are 94 mm long and widen to 65 mm, and its body is 10 cm wide. The task is scaled to fit it:
- 90 mm triangles made of 10 mm bars
- 10 × 40 mm pegs, 130 mm apart
- a 620 × 420 mm board

The pegs and objects are rigid bodies. A released object slides down a peg if its hole is over it. If it isn't, the object hits the peg and falls onto the board or off it.

**Grasping.** The robot's URDF has visual meshes but no collision shapes, so grasps are decided geometrically. Every measurement comes from the meshes (`scripts/setup/robot_geometry.py`), including the finger pads, which sit 14 cm out from the wrist.
- The fingers open only 2 mm either side of a bar.
- A grasp holds when they close with a bar between them: the pad centre must be within 7 mm of the bar's centre line, closing across it.
- While held, the object is moved with the gripper on every simulation step and drawn there. Once the fingers open, physics takes over again.
- One finger goes down inside the hole, so the bar to pick is the one farthest from the peg. The peg can sit anywhere in the hole.

**Mid-air transfer.**
1. The holding arm lifts the object 25 cm above the centre of the board.
2. It turns the object and tilts it 40° toward its own side.
3. The other arm leans 40° toward its own side and closes on a bar near the far corner. That grasp must pass the same test as a pick, or the transfer fails.
4. Both grippers hold the object for a moment, then the first lets go and backs away.

The two arms meet the object from opposite sides; at their closest so far they were 18 mm apart. Before placing, the new holder levels the object.

**Collision monitor** (`core/collide.py`). Because PhysX can't see the arms, every other step checks, using the arms' real mesh surfaces placed at the links' live poses:
- arm against arm, the board, any peg or any object (fingers may touch only the object they hold)
- the held object against pegs, other objects and the board

A contact is real overlap of more than 0.5 mm. Every contact is logged with its time, shown on the dashboard, reported to GPT in the next turn and counted in the score.

## Score

| Check | Pass |
|---|---|
| Transfers | 12 of 12: six across and six back, each placed on a peg on the other side after a mid-air hand-over |
| Left → right / right → left | 6 of 6 each way |
| Drops | 0. A drop is a release that doesn't end over a peg |
| Lost off board | 0. In FLS, objects dropped out of the field can't be recovered |
| Robot time | Simulated time spent on actions within 300 s, the FLS time limit |
| Collisions | 0 |

`--phases 1` runs only the left-to-right half.

## Cost

Prices are for Standard mode: $2 per 1M input tokens, $0.10 per 1M cached input tokens, $10 per 1M output tokens.

Moving one object per turn, the full task is about 13 calls, roughly **$0.25–0.35**. The 20-turn worst case, with every call at the 4,000-token output limit, is about $1.

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

Edit the `#SBATCH -A` line in `slurm/*.sbatch` to your Slurm account (and the partition to one with RTX GPUs), then,
from `$P/repo`:

```bash
python3 peg_transfer_benchmark/scripts/setup/openai_preflight.py <model id>   # free key and model check
MODEL=mock sbatch peg_transfer_benchmark/slurm/run_pegs.sbatch                # stand-in model: real sim, $0
MODEL=clumsy sbatch peg_transfer_benchmark/slurm/run_pegs.sbatch              # error-prone stand-in, $0
MODEL=<model id> sbatch peg_transfer_benchmark/slurm/run_pegs.sbatch          # GPT
# the v2 comparison: round r plays seed r-1 with the lessons from earlier rounds, then writes new lessons
ROUND=1 MODEL=gpt-6.1-sol sbatch peg_transfer_benchmark/slurm/run_pegs.sbatch --turns 20
ROUND=1 MODEL=anthropic/claude-opus-5.5 sbatch peg_transfer_benchmark/slurm/run_pegs.sbatch --turns 20
```

The key goes in `~/.config/openai/key` (`chmod 600`). `SEED=<n>` changes the objects' starting orientations.

To watch a run, open `dashboard.txt` in its run folder (`$P/results/peg_transfer_benchmark/<model>/seed<N>_j<job>/`). In VS Code's Ports panel, forward the `<node>:<port>` it lists, then open its link. The dashboard shows:
- the scene camera (facing the robot, so the robot's left arm appears on the right of the image) and the wrist camera of whichever arm is working
- GPT's reasoning and every call
- what each gripper holds
- spend against the cap
- the running score

To stop a run, press **STOP**, press Esc, or run `touch <run dir>/STOP`. The arms lift clear and hold, and the results are saved. If the run hasn't stopped after 45 s, the dashboard server cancels the job.

## Outputs

Each run writes to `$P/results/peg_transfer_benchmark/...`:

| File | Contents |
|---|---|
| `result.json` | Score, cost and token usage |
| `turns.json` | Each turn's reasoning, calls and their results |
| `t01_photo.png`, `t02_photo.png`, … | What GPT saw each turn |
| `board_final.png` | The board at the end |
| `video.mp4` | Recording of the run |
| `events.jsonl` | Everything the dashboard showed |

## Simulator notes

- **Isaac Sim version:** runs on **Isaac Sim 6.0.1** (`$P/isaac-sim_6.0.1.sif`). Since October 2026 the A10 nodes have NVIDIA driver 595.91, and Isaac Sim 5.1 crashes at startup on that driver (`librtx.scenedb`). Choose a different image with `SIF=...`.
- **NGX library:** Apptainer's `--nv` doesn't pass in `libnvidia-ngx.so`. Without it the DLSS denoiser fails and every frame is grainy, so the job scripts bind it in. The renderer is `RaytracedLighting`.
- **Texture brightness:** the renderer over-exposes the board texture and object colours, so the texture is uploaded with an extra gamma of 3 and the materials are pre-darkened.

## Code

| Path | Contents |
|---|---|
| `core/pegboard.py` | Board, pegs and objects; grasp and carry; pick, hand_over and place; where each object ended up; score; prompts and tools; the stand-in model |
| `core/ik.py` | Differential IK for both grippers: finger-pad point plus a chosen approach and closing direction, keeping the wrist inside its ±120° limit |
| `core/collide.py` | Collision monitor |
| `scripts/setup/robot_geometry.py` | Measures the robot from its meshes and writes `core/yam_geometry.npz` |
| `core/openai_client.py` | Responses API over https (streaming, cancel, several tool calls per reply) and the budget ledger |
| `core/live.py` | Events, camera frames and the STOP file for the dashboard |
| `core/yam_env.py`, `core/paint_sim.py` | The YAM scene, adapted from `art_benchmarks` with both grippers and a task hook. The board is the textured "sheet" |
| `scripts/run/gpt_pegs.py` | The benchmark run |
| `scripts/setup/smoke_pegs.py` | Scene check: one scripted transfer and one deliberate miss |
| `scripts/servers/dashboard_server.py`, `dashboard/index.html` | Live dashboard (token-protected) |
| `slurm/run_pegs.sbatch`, `slurm/smoke.sbatch` | Jobs. Both use Isaac Sim 6.0.1 with the NGX library bound in (see Simulator notes) |
