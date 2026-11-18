# PENDING: ps seed 0

- **Status:** unfinished
- **Cause:** 398 of 400 cycles done, then the queue's 3600 s timeout (about 9 to 10 s per cycle on the loaded GPU) stopped it; no `final.json`, not retried.
- **No data here:** this folder has no `cycles.csv` or `final.json`; `script/final_figs.py` and the summary ignore it.

## Command to produce it (from the repo root, G1 model fetched with `script/fetch_g1_model.sh`, reference downloaded on first use)

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
python examples/g1_walk.py --headless --seed-start 0 --seed-end 1 --max-cycles 400 --out-dir OUT ps
```

or through the queue (writes `<queue>/g1_walk/results/ps/s0/`):

```bash
TASKS=g1_walk SEEDS=0 METHODS=ps CUKS_GPU=/path/to/cuKernelSOS_gpu/python PY=$(which python) script/overnight_queue.sh
```

The queue's `BASE_TIMEOUT` is 3600 s per attempt; on a loaded GPU a baseline run can need longer, raise it first if you use the queue.

Then package it with `script/package_task_results.py` (see the README of this folder, section "Extending this folder") and delete this file.
