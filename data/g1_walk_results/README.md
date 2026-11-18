# G1 walk (hydrax `HumanoidMocap`): partial results, 8 finished runs, no GlobalMPPI run yet

Unitree G1 tracking a retargeted motion capture of a walk (`walk1_subject1`), six planners: `ps mppi dial mtp cbo gm` (gm = GlobalMPPI).
Task description, metrics and the run-length argument: `docs/g1_walk.md`. **This folder is incomplete and meant to be extended**
(see the end): only 8 of the 36 planned runs (6 methods x 6 seeds) finished, **1 to 2 seeds per method, and none for gm**. There is
therefore **no comparison with GlobalMPPI yet**, and the numbers below are not a ranking of the methods.

## Contents

- `<method>/s<seed>/cycles.csv`: one row per control cycle 0..399 (50 Hz, 8 s of simulated time) with `plan_s` (seconds; cycle 0 includes JIT),
  `best_cost` (the planner's **predicted** cost of the cycle), base `x y z`, `tilt_deg`, `ref_base_err` (distance of the base to the reference
  base position at the same time, m), `ref_joint_err` (2-norm of the joint error, rad), `track_cost` (the task's running cost at zero control,
  evaluated on the real simulated state, 0 to 1, 0 = on the reference), `gpu_mem_mb`, `solve_paths` (gm only). Row 400 has only `best_cost` (the
  last planning call). `final.json`: final base state and errors, `mean_ref_base_err`, `mean_track_cost`, `min_base_z`, `fell`, mean plan time.
- `g1_walk_final.png`: planner cost (predicted), tracking cost of the real state, base distance to the reference and base height versus
  cycle (median over the seeds that exist, IQR band; with 1 or 2 seeds the "band" is the range); the grey band marks the first step of the
  reference (3.62 to 4.22 s, cycles 181 to 211). `g1_walk_summary.txt`: the table and the per-seed final base error and minimum base height.
- `<method>/s<seed>/PENDING.md` instead of data: the runs that are not done (see the status matrix).
- Not included: videos, per-run logs, `args.json`, the `.npy` histories. **The reference motion is not included either** (see below).
  Because there is no `args.json` here, the settings below were checked against the code defaults (`examples/g1_walk.py`,
  `global_mppi/algs/`), not against recorded files.

## Task and settings

- Model `scene_23dof.xml` (23 actuators), planning step 0.02 s, control at 50 Hz. Horizon 0.8 s, 4 knots, zero-order-hold spline, control
  dimension 23 x 4 = 92. Simulation timestep 0.01 s, 10 solver iterations, 50 line-search iterations. The robot starts at reference frame 0.
  Seeds only change the controllers' random draws (the start state is fixed). 400 cycles = 8 s per run.
- Planner parameters are those of the other tasks of this benchmark (cube, G1 standup, bugtrap): ps, mppi, dial 2000 samples x 5 iterations;
  ps noise 0.2; mppi noise 0.2, temperature 0.001; dial noise 1.0, temperature 0.01, beta 1.0 / 1.0. gm (not run yet): 1000 KSOS candidates x
  5 restarts, CACS (h = 0.7 x vmax = 2.162, vmax 3.09), ADMM-cuda with `gpu_solve`, warm start, then MPPI 1000 x 5 at noise 0.1, temperature 0.1.
- **MTP and CBO use Hydrax defaults** (untuned; the MTP paper used per-task settings): mtp `m_pts` 3, `n_per_layer` 50, `degree` 2, 5 elites,
  sigma 0.5 / 0.1 / 1.0, temperature 0.1, beta 0.1, alpha 0.5, akima; cbo noise 1.0, temperature 0.01, consensus weight 1.0, noise weight 10,
  step 0.1 (the four required values are those of hydrax's bugtrap example). Both 2000 samples x 5 iterations.
- **Why 400 cycles:** the reference stands still for about 3.6 s and its first step is at 3.62 to 4.22 s, so 100 cycles (2 s) would only cover
  the standing phase; 8 s cover the first five steps (details in `docs/g1_walk.md`). 400 cycles is our choice, not a value from the paper.

**Metrics.** Tracking cost of the real state (`track_cost`) and base distance to the reference are the measures of how well the reference is
followed. A run **fell** if the base height was below **0.5 m** in any cycle (standing pelvis about 0.8 m). `best_cost` is the planner's predicted
cost over the horizon, not the realised cost, and should not be compared across methods as a performance measure.

## Reference motion: not included

The task downloads `Lafan1/mocap/UnitreeG1/walk1_subject1.npz` from the Hugging Face dataset
[`robfiras/loco-mujoco-datasets`](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets) on first use, into the Hugging Face cache
(`~/.cache/huggingface` or `$HF_HOME`). It is not part of this repository and must not be added to it. License of the dataset:
**CC-BY-NC-ND 4.0** (non-commercial, no derivatives); check it before any other use. The G1 model is fetched by `script/fetch_g1_model.sh` (not vendored).

## Results of the 8 finished runs

Per run (no averaging; `fell` = base below 0.5 m at some cycle; plan time = mean per cycle without cycle 0, on a GPU shared with other jobs):

| run | tracking cost (mean over 400 cycles) | final base error [m] | min base height [m] | fell | plan time [s] |
|---|---|---|---|---|---|
| ps s1 | 0.108 | 0.02 | 0.78 | no | 4.9 |
| mppi s0 | 0.550 | 1.14 | 0.11 | yes | 8.6 |
| mppi s1 | 0.278 | 0.08 | 0.77 | no | 4.9 |
| dial s0 | 0.314 | 0.03 | 0.75 | no | 5.9 |
| dial s1 | 0.552 | 1.35 | 0.09 | yes | 3.7 |
| mtp s0 | 0.146 | 0.02 | 0.78 | no | 5.6 |
| mtp s1 | 0.121 | 0.02 | 0.78 | no | 3.7 |
| cbo s0 | 0.659 | 1.73 | 0.09 | yes | 5.6 |

Falls: ps 0/1, mppi 1/2, dial 1/2, mtp 0/2, cbo 1/1. The method-level medians and means are in `g1_walk_summary.txt`; with n = 1 to 2 per method
they are only a restatement of the table above. No conclusions about GlobalMPPI can be drawn: it has no finished run.

**Plan times are inflated.** The baselines took 3.7 to 8.6 s per cycle (about 0.6 s in a 20-cycle test on a less loaded GPU) because the GPU was shared with other users' jobs;
compare the behaviour, not the times. The wall time of the finished runs was 26 to 59 minutes.

## Status of the 36 planned runs

| method | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|
| ps | unfinished (398/400, timeout) | done | pending (not run) | pending (not run) | pending (not run) | pending (not run) |
| mppi | done | done | pending (not run) | pending (not run) | pending (not run) | pending (not run) |
| dial | done | done | pending (not run) | pending (not run) | pending (not run) | pending (not run) |
| mtp | done | done | pending (not run) | pending (not run) | pending (not run) | pending (not run) |
| cbo | done | unfinished (stopped) | pending (not run) | pending (not run) | pending (not run) | pending (not run) |
| gm | failed (cuSOLVER error 7) | pending (not run) | pending (not run) | pending (not run) | pending (not run) | pending (not run) |

8 done, 1 failed (gm s0: cuSOLVER error 7 under GPU memory shortage, attempt 1 reached cycle 118), 2 unfinished (ps s0: 398 of 400 cycles, hit the 3600 s timeout; cbo s1: stopped), 25 pending (not run yet). Every folder that is not done holds a `PENDING.md` with the status, the cause and the exact command to produce the run; these folders contain no data. **The figure and the summary currently cover only the 8 finished runs**; they are regenerated with `script/final_figs.py` once more runs exist (see "Extending this folder").

## Unfinished runs

| run | status | cause |
|---|---|---|
| gm s0 | no result | cuSOLVER error 7 (`cusolverDnCreate`) under GPU memory shortage: the GPU was nearly full (28 of 32.6 GB in use at the start of attempts 2 and 3). Attempt 1 reached cycle 118 (about 74 min); attempts 2 and 3 failed in cycle 1 |
| ps s0 | 398 of 400 cycles | hit the 3600 s timeout (about 9 to 10 s per cycle on the loaded GPU); no `final.json`, not retried |
| cbo s1 | stopped | the queue was stopped by hand while it was running |
| seeds 2 to 5, all methods; gm s1 | not run | |

**GlobalMPPI cost on a shared GPU** (measured on the 118 cycles of gm s0 attempt 1, during the walk): about **36 s per cycle** (median 35.6 s,
range 22 to 64 s; cycle 0 with JIT 151 s), i.e. about **100 cycles per hour and about 4 h per 400-cycle run**. gm needs at least **6 GB of free GPU
memory** (its cuSOLVER handles failed with 3 to 4 GB free). These times depend strongly on what else runs on the GPU.

## Rerun

Environment: the Global-MPPI env with MuJoCo / mujoco-mjx **3.3.6** (not 4.0). GlobalMPPI needs the `gpu-solve` build of cuKernelSOS:
`ssisyg/cuKernelSOS`, branch `gpu-solve`, commit `2768612`, selected with `CUKERNELSOS_PYTHON=<checkout>/python`. `KSOS_GPU_SOLVE=1` is **not needed for
this task** (`examples/g1_walk.py --gpu` sets `gpu_solve` itself; the walker, double cart-pole, bugtrap and PushT tasks do need it). The baselines never call cuKernelSOS.

```bash
script/fetch_g1_model.sh                      # once: the G1 model
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
# ps | mppi | dial | mtp | cbo   (seed S; the reference is downloaded on the first run)
python examples/g1_walk.py --headless --seed-start S --seed-end $((S+1)) --max-cycles 400 --out-dir OUT ps
# GlobalMPPI (one run: about 4 h on a shared GPU)
CUKERNELSOS_PYTHON=/path/to/cuKernelSOS_gpu/python \
python examples/g1_walk.py --headless --seed-start S --seed-end $((S+1)) --max-cycles 400 --out-dir OUT --gpu globalmppi
# the whole batch (6 methods x 6 seeds), retries, logs, figure and table
TASKS=g1_walk CUKS_GPU=/path/to/cuKernelSOS_gpu/python PY=$(which python) script/overnight_queue.sh
```

**Timeouts of the queue** (`script/overnight_queue.sh`): 3600 s per attempt for ps, mppi, dial, mtp and cbo, 14400 s for gm (`BASE_TIMEOUT` and
`GM_TIMEOUT`, plain assignments at the top of the script, **not environment variables**: edit the script). A timed-out run is not retried.
For this task both are too tight on a shared GPU: a baseline run needed up to 59 minutes, and a gm run takes about 4 h (14400 s), so raise
`GM_TIMEOUT` (and `BASE_TIMEOUT`) before running the batch on a busy GPU.

## Extending this folder

When more runs of the queue have finished (a queue folder `<queue>/g1_walk/results/<method>/s<seed>/` with `final.json`):

```bash
python script/package_task_results.py <queue>/g1_walk/results data/g1_walk_results   # adds every finished run (cycles.csv + final.json, no videos); unfinished runs are skipped
python script/final_figs.py data/g1_walk_results g1_walk data/g1_walk_results         # redraws g1_walk_final.png and g1_walk_summary.txt
```

Then update the per-run table, the falls line and the "Unfinished runs" section of this README by hand (and remove the statement that there is no gm run
once gm results exist). The packager needs the `*best_cost_history*.npy` of each run in the results folder.
