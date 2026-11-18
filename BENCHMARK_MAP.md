# Global-MPPI benchmark map

Where everything is. All seven tasks are on **one branch**, [`benchmarks`](https://github.com/trio-cmu/Global-MPPI/tree/benchmarks), of `https://github.com/trio-cmu/Global-MPPI`. It has a `data/<task>_results/` folder per task with per-run data, a figure, a summary and a README that lists the exact settings, the rerun command and the known caveats. Everything (tasks, drivers, queue script, data) runs from this one checkout; there is no branch switching.

Status as of 2026-11-18. Run on a shared RTX 5090 (other users' jobs at 100 % utilisation), so the wall times in the READMEs are not clean timings. The senior's plan is to rerun everything on an idle 4090.

## Tasks

| Task | Data folder (branch `benchmarks`) | README | Figure |
|---|---|---|---|
| cube (LEAP hand) | [data/cube_results_mj336](https://github.com/trio-cmu/Global-MPPI/tree/benchmarks/data/cube_results_mj336) | [README](https://github.com/trio-cmu/Global-MPPI/blob/benchmarks/data/cube_results_mj336/README.md) | [figures/](https://github.com/trio-cmu/Global-MPPI/tree/benchmarks/data/cube_results_mj336/figures) |
| G1 standup | [data/g1_results](https://github.com/trio-cmu/Global-MPPI/tree/benchmarks/data/g1_results) | [README](https://github.com/trio-cmu/Global-MPPI/blob/benchmarks/data/g1_results/README.md) | [figures/](https://github.com/trio-cmu/Global-MPPI/tree/benchmarks/data/g1_results/figures) |
| PushT | [data/pusht_results](https://github.com/trio-cmu/Global-MPPI/tree/benchmarks/data/pusht_results) | [README](https://github.com/trio-cmu/Global-MPPI/blob/benchmarks/data/pusht_results/README.md) | [pusht_final.png](https://github.com/trio-cmu/Global-MPPI/blob/benchmarks/data/pusht_results/pusht_final.png) |
| walker | [data/walker_results](https://github.com/trio-cmu/Global-MPPI/tree/benchmarks/data/walker_results) | [README](https://github.com/trio-cmu/Global-MPPI/blob/benchmarks/data/walker_results/README.md) | [walker_final.png](https://github.com/trio-cmu/Global-MPPI/blob/benchmarks/data/walker_results/walker_final.png) |
| double cart-pole | [data/double_cart_pole_results](https://github.com/trio-cmu/Global-MPPI/tree/benchmarks/data/double_cart_pole_results) | [README](https://github.com/trio-cmu/Global-MPPI/blob/benchmarks/data/double_cart_pole_results/README.md) | [double_cart_pole_final.png](https://github.com/trio-cmu/Global-MPPI/blob/benchmarks/data/double_cart_pole_results/double_cart_pole_final.png) |
| bugtrap | [data/bugtrap_results](https://github.com/trio-cmu/Global-MPPI/tree/benchmarks/data/bugtrap_results) | [README](https://github.com/trio-cmu/Global-MPPI/blob/benchmarks/data/bugtrap_results/README.md) | [bugtrap_final.png](https://github.com/trio-cmu/Global-MPPI/blob/benchmarks/data/bugtrap_results/bugtrap_final.png) |
| G1 walk (mocap tracking) | [data/g1_walk_results](https://github.com/trio-cmu/Global-MPPI/tree/benchmarks/data/g1_walk_results) | [README](https://github.com/trio-cmu/Global-MPPI/blob/benchmarks/data/g1_walk_results/README.md) | [g1_walk_final.png](https://github.com/trio-cmu/Global-MPPI/blob/benchmarks/data/g1_walk_results/g1_walk_final.png) |

Notes on the table:
- cube and G1 standup have all six methods, 6 seeds each (plus a `gm_noise0.02` variant on cube).
- PushT, walker, double cart-pole and bugtrap have all six methods, 6 seeds each (36 runs).
- G1 walk is **not completed on our side**: 8 of 36 runs are done and the remaining 28 (including all gm runs) are left for a rerun on a free GPU; there is no finished GlobalMPPI run. The 28 missing runs have a folder with a `PENDING.md` that gives the status, the cause and the command to produce the run. The README has a 6 x 6 status matrix. Design doc: [docs/g1_walk.md](https://github.com/trio-cmu/Global-MPPI/blob/benchmarks/docs/g1_walk.md). The g1_walk runs that are not done are left for a rerun on a free GPU; the folder can be extended with `script/package_task_results.py` (see its README).
- Older cube and G1 standup folders have a `figures/` subfolder and `args.json` per run. The later tasks keep the figure and summary directly in the data folder.

## Setup needed for GlobalMPPI (gm)

gm calls ksos-tools, which calls cuKernelSOS (GPU ADMM). Use the GPU solve branch:

- `https://github.com/ssisyg/cuKernelSOS`, branch `gpu-solve`, commit `2768612`
- `export CUKERNELSOS_PYTHON=<path to cuKernelSOS_gpu>/python`
- `solve_paths` in a gm run must be `{gpu: N}`. The READMEs and `final.json`/logs show this for the runs we did.
- `ksos-tools` is a submodule pinned by each branch.

## Rerun commands

All commands run from the repo root on the `benchmarks` branch (no branch switching). The README of each task has the exact version.

- cube: `CUKERNELSOS_PYTHON=<...>/python CONFIGS="ps mppi dial mtp cbo gm gm_noise0.02" script/run_cube_benchmark.sh runs/cube_benchmark` (the default runs only `ps mppi dial gm`).
- G1 standup: `script/fetch_g1_model.sh` once, then `CUKERNELSOS_PYTHON=<...>/python ALGOS="ps mppi dial mtp cbo gm" script/run_g1_benchmark.sh runs/g1_benchmark`.
- PushT, walker, double cart-pole, bugtrap, G1 walk (queue driver, 6 methods x 6 seeds, skips runs that already have a `final.json`):

  `TASKS=<pusht|walker|double_cart_pole|bugtrap|g1_walk> CUKS_GPU=<...>/cuKernelSOS_gpu/python PY=$(which python) SYSPY=/usr/bin/python3 RUNS_DIR=$HOME/overnight_runs script/overnight_queue.sh`

  The queue runs the code of the current checkout, and sets `KSOS_GPU_SOLVE` and `CUKERNELSOS_PYTHON` itself. Optional: `SEEDS`, `METHODS`, `MAX_CYCLES`, `BASE_TIMEOUT`, `GM_TIMEOUT`.
- G1 walk also needs the mocap reference data. It is not in the repo (CC-BY-NC-ND 4.0 license). The README says where it comes from (Hugging Face cache).

## Algorithm settings

Same for every task; only the task's own horizon, knots and spline type change. PushT uses the baseline noise values from the senior's document (below).

| Method | Setting |
|---|---|
| PS | 2000 samples x 5 iterations, noise 0.2 |
| MPPI | 2000 x 5, noise 0.2, temperature 0.001 |
| DIAL | 2000 x 5, noise 1.0, temperature 0.01, beta 1.0 / 1.0 |
| MTP | 2000 x 5, Hydrax defaults (m_pts 3, 50 per layer, degree 2, 5 elites, sigma 0.5 / 0.1 / 1.0, temperature 0.1, beta 0.1, alpha 0.5, akima) |
| CBO | 2000 x 5, Hydrax defaults (initial noise 1.0, temperature 0.01, consensus weight 1.0, noise weight 10, step size 0.1) |
| GlobalMPPI | 1000 x 5, noise 0.1, temperature 0.1, CACS with 5 restarts and 1000 KSOS candidates, ADMM-cuda, warm start, GPU solve |

PushT baselines: PS noise 0.4; MPPI noise 1.0, temperature 0.1; DIAL noise 1.0, temperature 0.01, beta 1.0.

| Task | Horizon | Knots | Spline | Cycles |
|---|---|---|---|---|
| cube | 0.25 s | 4 | zero | 100 |
| G1 standup | 0.6 s | 4 | zero | 100 |
| PushT | 1.0 s | 6 | cubic | 100 |
| walker | 0.6 s | 5 | zero | 100 |
| double cart-pole | 1.0 s | 4 | cubic | 100 |
| bugtrap | 2.0 s | 11 | cubic | 100 |
| G1 walk | 0.8 s | 4 | zero | 400 |

MTP and CBO use the untuned Hydrax defaults (senior's decision). The MTP paper used per-task settings, which are not used here.

## What the results show

Short version; each task README has the full table with per-seed values and how it was computed. Medians over 6 seeds unless stated.

- **cube:** GlobalMPPI is best by a wide margin in the final cost (about 0.005, against 0.015 for MTP, 0.031 DIAL, 0.039 CBO, 0.089 MPPI, 0.47 PS). MTP is better than gm on position in 2 of 6 seeds.
- **G1 standup:** gm has the best median and is the most consistent, but it is not the best in every seed. Several baselines end lying or upside down in some seeds.
- **PushT:** gm beats PS, MPPI, DIAL and CBO but does not beat MTP (MTP has the lower median cost, 1.6e-4 against 3.2e-4). gm is about 30 times slower per cycle.
- **walker:** no advantage for gm. PS falls in 3 of 6 seeds, the other methods walk.
- **double cart-pole:** the task does not separate the methods (upright in 1 of 6 seeds for DIAL and CBO, 0 of 6 for the others).
- **bugtrap (2.0 s horizon):** PS and MPPI stay in the cup; DIAL, MTP, CBO and gm leave it. Final distance: MTP 0.014 m, gm 0.016 m, DIAL 0.031 m, CBO 0.160 m, PS and MPPI about 0.25 m. gm is not better than MTP or DIAL here and is 7 to 18 times slower per cycle.
- **G1 walk:** only baselines (8 runs, 1 to 2 seeds per method); the other 28 runs, including all gm runs, are left for a rerun on a free GPU. No gm comparison is possible.

Cost: gm needs roughly 7 to 200 times more time per cycle than the baselines, depending on the task (cube about 13x, G1 standup about 17x, PushT about 30x, walker about 40x, bugtrap 7 to 18x, double cart-pole about 200x). We have not run an equal-wall-clock comparison.

## Known issues and caveats

- **Shared GPU.** Wall times are noisy. On bugtrap the same runs were about 10 to 20 times slower for the baselines and about 4 times slower for gm in the later seeds than in the first seeds, because other users' jobs took the GPU.
- **gm needs free GPU memory.** `cuSOLVER error 7` (an internal error when memory is short) killed gm runs when other jobs left less than about 4 GB free. Start gm with at least 6 GB free. A crashed gm run restarts from cycle 0. bugtrap gm s5 had to be rerun once because of this.
- **Timeouts.** `script/overnight_queue.sh` uses 3600 s per attempt for the five baselines and 14400 s for gm by default; both are overridable with the environment variables `BASE_TIMEOUT` and `GM_TIMEOUT`. A timed-out run is not retried. Before every attempt the queue waits until enough GPU memory is free (6 GB for gm, 2 GB for the baselines; `MIN_FREE_GM_MB`, `MIN_FREE_BASE_MB`), and it keeps the `cycles.csv` of a failed attempt as `cycles.csv.attemptN`. On G1 walk a gm run takes 4 to 5 hours on the shared GPU (about 36 to 43 s per cycle, 400 cycles), so a gm run needs more than the 14400 s default: set `GM_TIMEOUT` (default 14400, environment variable) higher for that task.
- **G1 walk status.** Not completed: 8 of 36 runs done, 28 left for a rerun on a free GPU. gm s0: the first attempt reached cycle 118 and then failed with cuSOLVER error 7 under GPU memory shortage (attempts 2 and 3 failed in cycle 1); a later by-hand run with an 8 h timeout was stopped by hand at cycle 66 of 400 (no crash). ps s0: stopped by hand at 130 of 400 cycles (an earlier attempt reached 398 of 400 and hit the timeout). cbo s1 was stopped by hand. See the README status matrix.
- **No `args.json`** in the PushT, walker, double cart-pole, bugtrap and G1 walk folders. The settings in the READMEs were checked against the code defaults.
