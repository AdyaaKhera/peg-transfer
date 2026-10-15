#!/bin/bash
# Re-run only the between-round reflection for a finished comparison round (no simulator, no GPU: fine on a login
# node). Use it when a round played but its reflection failed. Costs one or two model calls on that provider's budget.
#   peg_transfer_benchmark/scripts/run/reflect_only.sh <round> <model> [effort]
#   e.g. peg_transfer_benchmark/scripts/run/reflect_only.sh 1 anthropic/claude-opus-5.5
set -e
P=${P:-$HOME/isaac-bench}
SIF=${SIF:-$P/isaac-sim_6.0.1.sif}
ROUND=$1; MODEL=$2; EFFORT=${3:-medium}
[ -n "$ROUND" ] && [ -n "$MODEL" ] || { sed -n 2,5p "$0"; exit 1; }
TAG=${MODEL//\//_}
SERIES=$P/results/peg_transfer_benchmark/comparison_v2/$TAG
RUN_DIR=$(ls -d $SERIES/round_$(printf %02d $ROUND)_seed* 2>/dev/null | tail -1)
[ -f "$RUN_DIR/score/result.json" ] || { echo "no finished round $ROUND in $SERIES"; exit 1; }
echo "reflecting on $RUN_DIR"

KEY_BIND=(); KEY_ENV=""
if [[ "$MODEL" == */* ]]; then
  KEYFILE=${OPENROUTER_KEY_FILE:-$HOME/.config/openrouter/key}
  [ -r "$KEYFILE" ] && { KEY_BIND=(--bind "$KEYFILE:/run/openrouter_key:ro"); KEY_ENV=",OPENROUTER_KEY_FILE=/run/openrouter_key"; }
else
  KEYFILE=${OPENAI_KEY_FILE:-$HOME/.config/openai/key}
  [ -r "$KEYFILE" ] && { KEY_BIND=(--bind "$KEYFILE:/run/openai_key:ro"); KEY_ENV=",OPENAI_KEY_FILE=/run/openai_key"; }
fi
apptainer exec --bind $P:$P "${KEY_BIND[@]}" --env PYTHONUNBUFFERED=1$KEY_ENV $SIF /isaac-sim/python.sh \
  $P/repo/peg_transfer_benchmark/scripts/run/reflect_round.py --run_dir $RUN_DIR --series $SERIES --round $ROUND \
  --model $MODEL --effort $EFFORT
