# PENDING: gm seed 0

- **Status:** failed
- **Cause:** cuSOLVER error 7 (`cusolverDnCreate`) under GPU memory shortage (the GPU was nearly full: 28 of 32.6 GB in use at the start of attempts 2 and 3). Attempt 1 reached cycle 118 (about 74 min); attempts 2 and 3 failed in cycle 1. A later by-hand run with an 8 h timeout (`GM_TIMEOUT` equivalent) was stopped by hand at cycle 66 of 400 (no crash, plan time about 36 to 43 s per cycle). No usable result was kept; the run is left for a rerun on a free GPU.
- **No data here:** this folder has no `cycles.csv` or `final.json`; `script/final_figs.py` and the summary ignore it.

## Command to produce it (from the repo root, G1 model fetched with `script/fetch_g1_model.sh`, reference downloaded on first use)

```bash
CUKERNELSOS_PYTHON=/path/to/cuKernelSOS_gpu/python XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
python examples/g1_walk.py --headless --seed-start 0 --seed-end 1 --max-cycles 400 --out-dir OUT --gpu globalmppi
```

or through the queue (writes `<queue>/g1_walk/results/gm/s0/`):

```bash
TASKS=g1_walk SEEDS=0 METHODS=gm CUKS_GPU=/path/to/cuKernelSOS_gpu/python PY=$(which python) script/overnight_queue.sh
```

About 4 h on a shared GPU and at least 6 GB of free GPU memory; raise `GM_TIMEOUT` in `script/overnight_queue.sh` first if you use the queue (14400 s is too tight).

Then package it with `script/package_task_results.py` (see the README of this folder, section "Extending this folder") and delete this file.
