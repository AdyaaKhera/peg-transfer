#!/bin/bash
#SBATCH -A YOUR_SLURM_ACCOUNT          # edit: your cluster account (partition/qos below are cluster-specific: an RTX GPU is needed)
#SBATCH -p a10
#SBATCH -q standby
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J peg-smoke
#SBATCH -o logs/%x-%j.out               # relative to where you submit; create logs/ first
# Pegboard check: sbatch peg_transfer_benchmark/slurm/smoke.sbatch
P=${P:-$HOME/isaac-bench}   # working folder: repo/ (this repo), assets/, sim_pylibs/, third_party/, results/
SIF=${SIF:-$P/isaac-sim_6.0.1.sif}   # 5.1 crashes on the 595 driver the A10 nodes have had since Oct 2026
KIT_CACHE=$P/cache/kit_$(basename $SIF .sif)
NGX="--bind /usr/lib64/libnvidia-ngx.so.1:/.singularity.d/libs/libnvidia-ngx.so.1"   # --nv leaves NGX out; without it the DLSS denoiser fails and frames are grainy
OUT=$P/results/peg_transfer_benchmark/_smoke/j$SLURM_JOB_ID
mkdir -p $OUT $KIT_CACHE $P/isaac_home
apptainer exec --nv --writable-tmpfs --home $P/isaac_home --bind $KIT_CACHE:/isaac-sim/kit/cache --bind $P:$P $NGX \
  --env ACCEPT_EULA=Y,PRIVACY_CONSENT=Y,PYTHONUNBUFFERED=1,PYTHONFAULTHANDLER=1,SIM_AA=${SIM_AA:-3},SIM_PYLIBS=$P/sim_pylibs \
  $SIF /isaac-sim/python.sh $P/repo/peg_transfer_benchmark/scripts/setup/smoke_pegs.py \
  --usd $P/assets/yam_station.usd --out $OUT "$@"
