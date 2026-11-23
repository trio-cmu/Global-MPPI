# Double cart-pole (hydrax `double_cart_pole` swing-up): six methods x 6 seeds

**This benchmark does not discriminate between the methods.** After 100 cycles (2.0 s of simulated time) all six methods end with the
tip at 3.5 to 3.7 m, below the target 4.0 m, and only two of the 36 runs (one DIAL, one CBO seed) are inside the "upright" window
(|tip z - 4| < 0.2). The time is too short for a swing-up; the curves show the methods still on the way up, not who swings up best.
The planner costs differ more (CBO and DIAL are clearly worse, PS / MTP / GlobalMPPI are close), but the planner cost is a
prediction, and GlobalMPPI is not better than PS or MTP here while taking about 200 times longer per cycle.

`double_cart_pole_final.png`: planner cost vs cycle (median over 6 seeds, IQR band, log y) and tip z vs cycle (median, IQR band,
target 4). `double_cart_pole_summary.txt`: the table and the final tip z of every seed. `<method>/s<seed>/cycles.csv`: one row per
control cycle 0..99 with `plan_s` (seconds; cycle 0 includes JIT), `best_cost` (the planner's predicted cost of the cycle), `cart_x`,
`pole1_rad`, `pole2_rad`, `tip_z`, `gpu_mem_mb`, `solve_paths` (gm only: cumulative count of KSOS solves by path); row 100 has only
`best_cost` (the last planning call). `final.json`: final cart x, pole angles, tip z, `upright`, mean plan time, `solve_paths`.
No videos, no npy files. Methods: `ps mppi dial mtp cbo gm` (gm = GlobalMPPI).

| method | cost@100 median / mean ± std | tip z at the end (median) | upright | plan_s median |
|---|---|---|---|---|
| PS | 0.65 / 0.669 ± 0.078 | 3.70 | 0/6 | 0.24 s |
| MPPI | 2.29 / 2.35 ± 1.29 | 3.59 | 0/6 | 0.23 s |
| DIAL | 7.76 / 6.93 ± 2.99 | 3.72 | 1/6 | 0.24 s |
| MTP | 1.07 / 1.05 ± 0.274 | 3.62 | 0/6 | 0.24 s |
| CBO | 25 / 24.8 ± 14.3 | 3.49 | 1/6 | 0.24 s |
| GlobalMPPI | 1.13 / 1.15 ± 0.351 | 3.58 | 0/6 | 49.3 s |

## The CACS duplicate fix

GlobalMPPI's CACS sampling clips the local proposals to the control box. With a 1-D control (one cart force) and 4 knots the control
vector has only 4 dimensions, and many proposals land on the same corners of the box, which gives duplicate samples. cuKernelSOS
needs distinct samples and stops with `samples must be distinct`; the first double_cart_pole GlobalMPPI run crashed this way.
Commit `75a1a5f` (`global_mppi/algs/global_mppi.py`, `_solve_cacs`) drops the duplicate rows before the KSOS problem is built. It does
nothing when there are no duplicates, so it does not change the other tasks. It is part of the `benchmarks` branch, so every task uses it; the
runs of the other tasks were made without it. The GlobalMPPI runs here use it.

## Settings

`examples/double_cart_pole.py` on the `benchmarks` branch. Task from hydrax `bd53d43` (`global_mppi/tasks/double_cart_pole.py`, model
`global_mppi/models/double_cart_pole`): swing the tip to z = 4 m over the cart, with velocity and control penalties; the simulation
uses the task's model as it is (timestep 0.01 s, no contacts). Control at 50 Hz, horizon 1.0 s, 4 knots, cubic spline, 5 iterations,
100 cycles per run (2.0 s), seeds 0-5, control dimension 1 x 4 knots = 4. The noise and temperature values are those of
`examples/cube.py`; nothing was tuned.

| method | samples | other |
|---|---|---|
| PS | 2000 | noise 0.2 |
| MPPI | 2000 | noise 0.2, temperature 0.001 |
| DIAL | 2000 | noise 1.0, temperature 0.01, beta 1.0 / 1.0 |
| MTP | 2000 | hydrax defaults (m_pts 3, n_per_layer 50, degree 2, 5 elites, sigma 0.5 / 0.1 / 1.0, temperature 0.1, beta 0.1, alpha 0.5, akima) |
| CBO | 2000 | hydrax defaults (noise 1.0, temperature 0.01, consensus weight 1.0, noise weight 10, step 0.1) |
| GlobalMPPI | 1000 KSOS candidates per restart, then MPPI 1000 x 5 at noise 0.1, temperature 0.1 | CACS (h 0.7), ADMM-cuda, 5 restarts, warm start, `gpu_solve` on |

MTP and CBO use the untuned hyperparameters of hydrax `bd53d43` (`global_mppi/algs/mtp.py`, `cbo.py`).

## cuKernelSOS requirement (GlobalMPPI only)

`ssisyg/cuKernelSOS`, branch `gpu-solve`, commit `2768612` (a WIP, not merged: GPU affine solve, off by default).
`KSOS_GPU_SOLVE=1` switches `gpu_solve` on and `CUKERNELSOS_PYTHON=<cuKernelSOS checkout>/python` selects the build (the `--gpu` flag of
`double_cart_pole.py` only turns on the GPU ADMM). All 6 gm runs have `solve_paths: {'gpu': 505}` (101 planning calls x 5 restarts, no
CPU fallback). The baselines never call cuKernelSOS. `KSOS_GPU_SOLVE=1` is needed for this task; the cube and G1 standup benchmarks do not need it (`examples/cube.py --gpu` and `examples/g1.py --gpu` set `gpu_solve` themselves).

ksos-tools and cuKernelSOS pins (as in the cube and G1 standup READMEs): ksos-tools is a git submodule of this branch, `https://github.com/Zhongqi2/ksos-tools`, branch `gpu-solve-exp`, commit `86e6163` (`git submodule update --init`); cuKernelSOS: `https://github.com/ssisyg/cuKernelSOS`, branch `gpu-solve`, commit `2768612` (on top of `4180908`), built as described in `BENCHMARK_MAP.md` ("Quick start from a fresh clone", step 3) and selected with `CUKERNELSOS_PYTHON`. MuJoCo / MJX 3.3.6; the exact package list of the env is `docs/env_ksos_pip_freeze.txt`. Hydrax is not required (ported copies of Hydrax `bd53d43`).

## Rerun

Environment: the Global-MPPI env with MuJoCo / mujoco-mjx **3.3.6** (not 4.0), the `benchmarks` branch checked out. `script/` is
the queue and plotting tools. The runs of this folder were made with an earlier version of `script/`; `script/` here was later synced to a newer version that also supports the bugtrap and g1_walk tasks (nothing changes for double_cart_pole: the redrawn figure and table of this folder are identical).

```bash
# everything: 6 methods x 6 seeds, retries, logs, figure, tables (gm takes about 1 h per seed: 6-7 h for the batch on our shared GPU)
TASKS=double_cart_pole CUKS_GPU=/path/to/cuKernelSOS_gpu/python PY=$(which python) script/overnight_queue.sh
# output: logs/overnight_runs/<timestamp>/{STATUS.txt, double_cart_pole/batch.log, double_cart_pole/results/, final_figs/}
```

The queue runs the code of the checkout it lives in (the `benchmarks` branch); there is no branch or worktree switching.

One run by hand (seed S, from the repo root):

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
# ps | mppi | dial | mtp | cbo
python examples/double_cart_pole.py --headless --seed-start S --seed-end $((S+1)) --max-cycles 100 --out-dir OUT ps
# GlobalMPPI
KSOS_GPU_SOLVE=1 CUKERNELSOS_PYTHON=/path/to/cuKernelSOS_gpu/python \
python examples/double_cart_pole.py --headless --seed-start S --seed-end $((S+1)) --max-cycles 100 --out-dir OUT --gpu globalmppi
```

Each run writes `OUT/<ps|mppi|dial|mtp|cbo|gm>/s<S>/{cycles.csv, final.json, args.json, *.npy, video}`. Redraw and repackage:

```bash
python script/final_figs.py <queue folder> double_cart_pole <out dir>                                       # from the runs
python script/package_task_results.py <queue folder>/double_cart_pole/results data/double_cart_pole_results # this folder's files
python script/final_figs.py data/double_cart_pole_results double_cart_pole data/double_cart_pole_results   # from this folder
```

## Notes

- Timeouts of `script/overnight_queue.sh`: 3600 s per attempt for ps, mppi, dial, mtp and cbo, 14400 s for gm (`BASE_TIMEOUT` and `GM_TIMEOUT`: the defaults, overridable as environment variables, e.g. `GM_TIMEOUT=28800 script/overnight_queue.sh`; before every attempt the queue also waits for free GPU memory, see the header of the script). A timed-out run is not retried. gm may need more than 14400 s on a busy GPU.
- This folder holds `cycles.csv` and `final.json` per run, no `args.json`: the settings in the table above were checked against the code defaults (`examples/double_cart_pole.py`, `global_mppi/algs/`), not against recorded files.
- The GPU was shared with other users' jobs during these runs, so the plan times (GlobalMPPI 47 to 55 s per cycle on average, 26 to 56 s
  per cycle in earlier checks) are inflated and vary; compare costs, not times. An `nvidia-smi` snapshot was stored before every run
  (not included here).
- 6 seeds per method; IQR and std over 6 values are rough. The start state is fixed, so the seed only changes the controllers' random draws.
- `best_cost` is the planner's predicted cost over the horizon, not the realised cost; the realised behaviour is `tip_z`, `cart_x` and
  the pole angles.
- A longer run (several seconds of simulated time) would be needed to see which method actually swings up and balances. Not done.
- No GCS.
