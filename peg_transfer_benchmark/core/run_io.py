"""Where runs go: $P/results/<benchmark>/..., outside the repo (videos are too big for git); PEG_OUT overrides."""
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # peg_transfer_benchmark/
RESULTS = os.environ.get("PEG_OUT", os.path.join(os.path.dirname(os.path.dirname(ROOT)), "results"))


def new_run_dir(*parts, name=""):
    """A fresh directory per run: <parts>/<name>_j<slurm job id> (a timestamp outside Slurm)."""
    job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    tag = f"j{job}" if job else time.strftime("t%Y%m%d-%H%M%S")
    base = os.path.join(RESULTS, *parts, f"{name}_{tag}" if name else tag)
    d, k = base, 1
    while os.path.exists(d):
        d, k = f"{base}_{k}", k + 1
    os.makedirs(d)
    return d
