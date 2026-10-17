#!/bin/bash
#SBATCH -A YOUR_SLURM_ACCOUNT          # edit: your cluster account (partition/qos below are cluster-specific: an RTX GPU is needed)
#SBATCH -p a10
#SBATCH -q standby
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -J pegs
#SBATCH -o logs/%x-%j.out               # relative to where you submit; create logs/ first
# One FLS peg transfer run + its live dashboard. Submit from $P/repo:
#   MODEL=mock sbatch peg_transfer_benchmark/slurm/run_pegs.sbatch                  # stand-in model, $0
#   MODEL=<openai model id> sbatch peg_transfer_benchmark/slurm/run_pegs.sbatch [--phases 1] [gpt_pegs.py args]
#   MODEL=anthropic/claude-opus-5.5 ...                                              # OpenRouter (~/.config/openrouter/key)
#   ROUND=2 MODEL=... sbatch ...      # comparison round 2: seed 1, the model's lessons from round 1, then its reflection
#                                     # (series in $P/results/peg_transfer_benchmark/comparison_v2/<model>/)
# The dashboard link (with its access token) is written to $RUN_DIR/dashboard.txt once the job starts.

P=${P:-$HOME/isaac-bench}   # working folder: repo/ (this repo), assets/, sim_pylibs/, third_party/, results/
SIF=${SIF:-$P/isaac-sim_6.0.1.sif}   # 5.1 crashes on the 595 driver the A10 nodes have had since Oct 2026
KIT_CACHE=$P/cache/kit_$(basename $SIF .sif)
NGX="--bind /usr/lib64/libnvidia-ngx.so.1:/.singularity.d/libs/libnvidia-ngx.so.1"   # --nv leaves NGX out; without it the DLSS denoiser fails and frames are grainy
MODEL=${MODEL:-mock}
TAG=${MODEL//\//_}                     # anthropic/claude-opus-5.5 -> anthropic_claude-opus-5.5
EFFORT=${EFFORT:-medium}                # v2 default (v1 ran at low)
if [ -n "$ROUND" ]; then                # a comparison round: round r plays seed r-1, with the lessons of rounds 1..r-1
  SEED=${SEED:-$((ROUND - 1))}
  SERIES=$P/results/peg_transfer_benchmark/comparison_v2/$TAG   # v1 rounds stay in comparison/ (git tag peg-v1)
  RUN_DIR=$SERIES/round_$(printf %02d $ROUND)_seed${SEED}_j${SLURM_JOB_ID}
  SERIES_ARGS=(--series $SERIES)
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('$SERIES/series.json')).get('saturated') else 1)" 2>/dev/null; then
    echo "series $SERIES has saturated: round $ROUND skipped (nothing run, nothing spent)"; exit 0
  fi
else
  SEED=${SEED:-0}
  RUN_DIR=$P/results/peg_transfer_benchmark/$TAG/$(date +%Y-%m-%d_%H%M)_seed${SEED}_j${SLURM_JOB_ID}
  SERIES_ARGS=()
fi
PORT=$((8800 + SLURM_JOB_ID % 900))
mkdir -p $RUN_DIR $KIT_CACHE $P/isaac_home
echo "seed $SEED model $MODEL run dir $RUN_DIR commit $(git -C $P/repo rev-parse --short HEAD)"

KEY_BIND=(); KEY_ENV=""
if [[ "$MODEL" == */* ]]; then          # an OpenRouter model id (provider/model)
  KEYFILE=${OPENROUTER_KEY_FILE:-$HOME/.config/openrouter/key}
  if [ -r "$KEYFILE" ]; then
    KEY_BIND=(--bind "$KEYFILE:/run/openrouter_key:ro"); KEY_ENV=",OPENROUTER_KEY_FILE=/run/openrouter_key"
  elif [ -n "$OPENROUTER_API_KEY" ]; then
    echo "Using OPENROUTER_API_KEY from the environment"
  else
    echo "No OpenRouter key: put it in $KEYFILE or set OPENROUTER_API_KEY"; exit 1
  fi
elif [ "$MODEL" != mock ] && [ "$MODEL" != replay ] && [ "$MODEL" != clumsy ]; then
  KEYFILE=${OPENAI_KEY_FILE:-$HOME/.config/openai/key}
  if [ -r "$KEYFILE" ]; then
    KEY_BIND=(--bind "$KEYFILE:/run/openai_key:ro"); KEY_ENV=",OPENAI_KEY_FILE=/run/openai_key"
  elif [ -n "$OPENAI_API_KEY" ]; then
    echo "Using OPENAI_API_KEY from the environment"   # passed into the container with the rest of the environment
  else
    echo "No OpenAI key: put it in $KEYFILE or set OPENAI_API_KEY (see peg_transfer_benchmark/README.md)"; exit 1
  fi
fi

LEDGER=$P/results/openai_spend.json; [[ "$MODEL" == */* ]] && LEDGER=$P/results/openrouter_spend.json   # each provider has its own $3
python3 $P/repo/peg_transfer_benchmark/scripts/servers/dashboard_server.py --run_dir $RUN_DIR --port $PORT --ledger $LEDGER > $RUN_DIR/dashboard.log 2>&1 &
SERVER=$!
trap "kill $SERVER 2>/dev/null" EXIT
sleep 2; cat $RUN_DIR/dashboard.log

apptainer exec --nv --writable-tmpfs \
  --home $P/isaac_home \
  --bind $KIT_CACHE:/isaac-sim/kit/cache \
  --bind $P:$P $NGX "${KEY_BIND[@]}" \
  --env ACCEPT_EULA=Y,PRIVACY_CONSENT=Y,PYTHONUNBUFFERED=1,PEG_DEBUG=${PEG_DEBUG:-},SIM_PYLIBS=$P/sim_pylibs,RUN_DIR=$RUN_DIR$KEY_ENV \
  $SIF \
  /isaac-sim/python.sh $P/repo/peg_transfer_benchmark/scripts/run/gpt_pegs.py --usd $P/assets/yam_station.usd \
  --seed $SEED --model $MODEL --effort $EFFORT "${SERIES_ARGS[@]}" "$@"
sleep 15   # let open dashboards receive the last events
kill $SERVER 2>/dev/null
# tidy the run folder: README, video/, photos/, model/ (transcript), score/, live/; and the model's index.md
apptainer exec --bind $P:$P --env PYTHONUNBUFFERED=1 $SIF /isaac-sim/python.sh \
  $P/repo/peg_transfer_benchmark/scripts/report/package_run.py $RUN_DIR
if [ -n "$ROUND" ] && [ -f $RUN_DIR/score/result.json ]; then   # debrief + the model's own lessons for the next round
  apptainer exec --bind $P:$P "${KEY_BIND[@]}" --env PYTHONUNBUFFERED=1$KEY_ENV $SIF /isaac-sim/python.sh \
    $P/repo/peg_transfer_benchmark/scripts/run/reflect_round.py --run_dir $RUN_DIR --series $SERIES --round $ROUND \
    --model $MODEL --effort $EFFORT
fi
