# PENDING: ps seed 2

- **Status:** not run
- **Cause:** not run yet.
- **No data here:** this folder has no `cycles.csv` or `final.json`; `script/final_figs.py` and the summary ignore it.

## Command to produce it (from the repo root, G1 model fetched with `script/fetch_g1_model.sh`, reference downloaded on first use)

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
python examples/g1_walk.py --headless --seed-start 2 --seed-end 3 --max-cycles 400 --out-dir OUT ps
```

or through the queue (writes `<queue>/g1_walk/results/ps/s2/`):

```bash
TASKS=g1_walk SEEDS=2 METHODS=ps CUKS_GPU=/path/to/cuKernelSOS_gpu/python PY=$(which python) script/overnight_queue.sh
```

Baselines took 26 to 59 minutes per run here.

Then package it with `script/package_task_results.py` (see the README of this folder, section "Extending this folder") and delete this file.
