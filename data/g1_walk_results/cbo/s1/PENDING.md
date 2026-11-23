# PENDING: cbo seed 1

- **Status:** unfinished
- **Cause:** the queue was stopped by hand while this run was in progress; no `final.json`.
- **No data here:** this folder has no `cycles.csv` or `final.json`; `script/final_figs.py` and the summary ignore it.

## Command to produce it (from the repo root, G1 model fetched with `script/fetch_g1_model.sh`, reference downloaded on first use)

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
python examples/g1_walk.py --headless --seed-start 1 --seed-end 2 --max-cycles 400 --out-dir OUT cbo
```

or through the queue (writes `<queue>/g1_walk/results/cbo/s1/`):

```bash
TASKS=g1_walk SEEDS=1 METHODS=cbo CUKS_GPU=/path/to/cuKernelSOS_gpu/python PY=$(which python) script/overnight_queue.sh
```

Baselines took 26 to 59 minutes per run here.

Then package it with `script/package_task_results.py` (see the README of this folder, section "Extending this folder") and delete this file.
