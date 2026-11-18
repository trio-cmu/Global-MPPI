# Walker (hydrax `walker` task): six methods x 6 seeds

`walker_final.png`: planner cost vs cycle (median over 6 seeds, IQR band, log y); torso height and forward x vs cycle (median, IQR
band). `walker_summary.txt`: the table (cost at cycle 100, last-25-cycle cost, torso height, forward x, falls, plan time) and the
final x / torso height of every seed. `<method>/s<seed>/cycles.csv`: one row per control cycle 0..99 with `plan_s` (seconds; cycle 0
includes JIT), `best_cost` (the planner's predicted cost of the cycle), `torso_h` [m], `x` [m], `torso_angle_deg`, `gpu_mem_mb`,
`solve_paths` (gm only: cumulative count of KSOS solves by path); row 100 has only `best_cost` (the last planning call). `final.json`:
final torso height, x and angle, mean plan time, `solve_paths`. No videos, no npy files. Methods: `ps mppi dial mtp cbo gm`
(gm = GlobalMPPI).

| method | cost@100 median / mean ± std | torso h last-25 [m] | x at the end [m] | fell | plan_s |
|---|---|---|---|---|---|
| PS | 2.92 / 3.93 ± 3.57 | 0.667 | 0.885 | 3/6 | 0.37 s |
| MPPI | 0.157 / 0.256 ± 0.234 | 1.26 | 2.43 | 0/6 | 0.37 s |
| DIAL | 0.131 / 0.653 ± 1.25 | 1.25 | 2.36 | 0/6 | 0.37 s |
| MTP | 0.141 / 0.203 ± 0.194 | 1.18 | 2.65 | 0/6 | 0.37 s |
| CBO | 0.476 / 2.17 ± 3.7 | 1.24 | 2.53 | 0/6 | 0.37 s |
| GlobalMPPI | 0.297 / 0.35 ± 0.251 | 1.26 | 2.28 | 0/6 | 15.1 s |

(medians over the 6 seeds, "fell" = final torso height < 0.6 m.) The task is hydrax's: walk forward at 1.5 m/s with the torso at
1.2 m, upright. Only PS falls; the other five walk with similar speed. GlobalMPPI is not better than the baselines here.

## Settings

`examples/walker.py` on branch `walker`, unchanged. Task from hydrax `bd53d43` (`global_mppi/tasks/walker.py`, model
`global_mppi/models/walker`). Planning model timestep 0.01 s (60 steps over the horizon), control at 50 Hz, simulation timestep 0.005 s
with 50 solver iterations. Horizon 0.6 s, 5 knots, zero-order-hold spline (`spline_type="zero"`), 5 iterations, 100 cycles per run
(2.0 s of simulated time), seeds 0-5. The noise and temperature values are those of `examples/cube.py`; nothing was tuned.

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
`walker.py` only turns on the GPU ADMM). All 6 gm runs have `solve_paths: {'gpu': 505}` (101 planning calls x 5 restarts, no CPU
fallback). The baselines never call cuKernelSOS. `KSOS_GPU_SOLVE=1` is needed for this task; the cube and G1 standup benchmarks do not need it (`examples/cube.py --gpu` and `examples/g1.py --gpu` set `gpu_solve` themselves).

## Rerun

Environment: the Global-MPPI env with MuJoCo / mujoco-mjx **3.3.6** (not 4.0), this branch (`walker`) checked out. `script/` is the
queue and plotting tools. The runs of this folder were made with the version of branch `pusht-bench` at commits `36f33fe` and `1bb853e`; `script/` here was later synced to a newer version that also supports the bugtrap and g1_walk tasks (nothing changes for walker: the redrawn figure and table of this folder are identical).

```bash
# everything: 6 methods x 6 seeds, retries, logs, figure, tables
TASKS=walker CUKS_GPU=/path/to/cuKernelSOS_gpu/python PY=$(which python) script/overnight_queue.sh
# output: logs/overnight_runs/<timestamp>/{STATUS.txt, walker/batch.log, walker/results/, final_figs/walker_final.png}
```

The queue runs the code of branch `walker` through a git worktree in `../overnight_wt/walker` (set `WT` to move it).

One run by hand (seed S, from the repo root):

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
# ps | mppi | dial | mtp | cbo
python examples/walker.py --headless --seed-start S --seed-end $((S+1)) --max-cycles 100 --out-dir OUT ps
# GlobalMPPI
KSOS_GPU_SOLVE=1 CUKERNELSOS_PYTHON=/path/to/cuKernelSOS_gpu/python \
python examples/walker.py --headless --seed-start S --seed-end $((S+1)) --max-cycles 100 --out-dir OUT --gpu globalmppi
```

Each run writes `OUT/<ps|mppi|dial|mtp|cbo|gm>/s<S>/{cycles.csv, final.json, args.json, *.npy, video}`. Redraw and repackage:

```bash
python script/final_figs.py <queue folder> walker <out dir>                              # figure and table from the runs
python script/package_task_results.py <queue folder>/walker/results data/walker_results   # the files of this folder
python script/final_figs.py data/walker_results walker data/walker_results               # figure and table from this folder
```

## Notes

- Timeouts of `script/overnight_queue.sh`: 3600 s per attempt for ps, mppi, dial, mtp and cbo, 14400 s for gm (`BASE_TIMEOUT` and `GM_TIMEOUT`, set at the top of the script; they are plain assignments, not environment variables, so change them in the script). A timed-out run is not retried. gm may need more than 14400 s on a busy GPU.
- This folder holds `cycles.csv` and `final.json` per run, no `args.json`: the settings in the table above were checked against the code defaults (`examples/walker.py`, `global_mppi/algs/`), not against recorded files.
- The GPU was shared with other users' jobs during these runs, so the plan times (GlobalMPPI 15 s per cycle) are inflated and vary
  between runs; compare costs, not times. An `nvidia-smi` snapshot was stored before every run (not included here).
- 6 seeds per method; IQR and std over 6 values are rough. `best_cost` is the planner's predicted cost over the horizon, not the
  realised cost; the realised behaviour is in `torso_h`, `x` and `torso_angle_deg`.
- Each run stops after 100 cycles (2 s): a walking gait over a longer time was not tested.
- No GCS.
