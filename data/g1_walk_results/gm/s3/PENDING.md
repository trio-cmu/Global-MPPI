# PENDING: gm seed 3

- **Status:** not run
- **Cause:** not run yet.
- **No data here:** this folder has no `cycles.csv` or `final.json`; `script/final_figs.py` and the summary ignore it.

## Command to produce it (from the repo root, G1 model fetched with `script/fetch_g1_model.sh`, reference downloaded on first use)

```bash
CUKERNELSOS_PYTHON=/path/to/cuKernelSOS_gpu/python XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
python examples/g1_walk.py --headless --seed-start 3 --seed-end 4 --max-cycles 400 --out-dir OUT --gpu globalmppi
```

or through the queue (writes `<queue>/g1_walk/results/gm/s3/`):

```bash
TASKS=g1_walk SEEDS=3 METHODS=gm CUKS_GPU=/path/to/cuKernelSOS_gpu/python PY=$(which python) script/overnight_queue.sh
```

About 4 h on a shared GPU and at least 6 GB of free GPU memory; raise `GM_TIMEOUT` in `script/overnight_queue.sh` first if you use the queue (14400 s is too tight).

Then package it with `script/package_task_results.py` (see the README of this folder, section "Extending this folder") and delete this file.
