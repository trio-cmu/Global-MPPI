# Bugtrap (hydrax `bugtrap`): six methods x 6 seeds

A point mass starts at the open end of a U-shaped cup (x = -0.15, y = 0) and has to reach a goal at x = 0.25 that lies **behind the closed end** of the
cup: the direct path runs into the cup (a local minimum of the distance cost), the way to the goal is a detour around the outside of the cup walls.
Six planners: `ps mppi dial mtp cbo gm` (gm = GlobalMPPI), 6 seeds each (0-5), 100 control cycles (2.0 s, 50 Hz), all 36 runs finished.

`bugtrap_final.png`: the xy paths of the 6 seeds of every method over the cup (to scale, dots = final positions, goal star with the 2 cm and 5 cm
circles) and the distance to the goal versus cycle (median over the 6 seeds, IQR band). `bugtrap_summary.txt`: the table and the final distance of
every seed. `<method>/s<seed>/cycles.csv`: one row per control cycle 0..99 with `plan_s` (seconds; cycle 0 includes JIT), `best_cost` (the planner's
predicted cost), `x`, `y`, `distance` (to the goal), `inside_cup` (0/1) of the state the cycle planned from, `gpu_mem_mb`, `solve_paths` (gm only); row 100 has only
`best_cost` (the last planning call). `final.json`: the final position and distance (state after the last cycle), `left_cup`, `reached`,
`reached_5cm`, mean plan time, `solve_paths` (gm). No videos, no npy files, no `args.json`: the settings below were checked against the code defaults
(`examples/bugtrap.py`, `global_mppi/algs/`), not against recorded files.

## Metrics

- **final distance** to the goal (m) after the last cycle: the primary measure.
- **left the cup** (`left_cup` in `final.json`): the final position is outside the cup box, i.e. **not** (x in (-0.11, 0.09) and |y| < 0.09). It is only a
  statement about the final position; whether a run passed around the cup walls on the way is visible in the paths of the figure, not in this flag, and
  a run can leave the cup and still end far from the goal (CBO below).
- **reached@2cm** (`reached`): final distance below 0.02 m; **reached@5cm** (`reached_5cm`): below 0.05 m.
- `best_cost` is the planner's predicted cost over the horizon, not the realised cost.

## Results (6 seeds per method, recomputed from `final.json` / `cycles.csv`)

| method | final distance median (IQR) [m] | mean ± std [m] | left the cup | reached@2cm | reached@5cm | plan time per cycle, median [s] |
|---|---|---|---|---|---|---|
| PS | 0.248 (0.246 - 0.249) | 0.248 ± 0.003 | 0/6 | 0/6 | 0/6 | 1.37 |
| MPPI | 0.252 (0.243 - 0.261) | 0.253 ± 0.013 | 0/6 | 0/6 | 0/6 | 1.61 |
| DIAL | 0.031 (0.029 - 0.031) | 0.029 ± 0.004 | 6/6 | 0/6 | 6/6 | 3.47 |
| MTP | 0.014 (0.011 - 0.015) | 0.013 ± 0.003 | 6/6 | 6/6 | 6/6 | 2.23 |
| CBO | 0.160 (0.146 - 0.168) | 0.155 ± 0.019 | 6/6 | 0/6 | 0/6 | 1.76 |
| GlobalMPPI | 0.016 (0.013 - 0.022) | 0.018 ± 0.008 | 6/6 | 4/6 | 6/6 | 25.1 |

(Plan time: median over all cycles of all 6 seeds without cycle 0; cycle times varied strongly with the load of the shared GPU, see the notes.)

Final distance per seed [m]:

| method | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|
| PS | 0.249 | 0.243 | 0.248 | 0.249 | 0.251 | 0.245 |
| MPPI | 0.254 | 0.241 | 0.264 | 0.239 | 0.250 | 0.272 |
| DIAL | 0.030 | 0.031 | 0.028 | 0.031 | 0.022 | 0.032 |
| MTP | 0.010 | 0.015 | 0.013 | 0.017 | 0.015 | 0.011 |
| CBO | 0.175 | 0.143 | 0.155 | 0.125 | 0.165 | 0.169 |
| GlobalMPPI | 0.012 | 0.007 | 0.029 | 0.017 | 0.024 | 0.016 |

Reading, with 6 seeds per method and no significance test: PS and MPPI stay in the cup (the local minimum) in every seed. DIAL, MTP, CBO and GlobalMPPI
all leave the cup around its outside; CBO does not get close to the goal (its paths wander along the far wall). MTP and GlobalMPPI end closest to the goal.
**GlobalMPPI is not better than the best baseline here:** MTP has the lower median final distance (0.014 vs 0.016 m), reached 2 cm in all 6 seeds
(GlobalMPPI in 4 of 6), and GlobalMPPI is closer than MTP in only 2 of the 6 seeds (s1: 0.007 vs 0.015 m; s3: 0.0165 vs 0.0174 m, about equal); its s2 and s4 end at 2.9 and 2.4 cm.
DIAL ends at about 3 cm without reaching 2 cm. The differences between MTP, GlobalMPPI and DIAL are a few millimetres to 2 cm over 6 seeds. GlobalMPPI needs
about 7 to 18 times more time per cycle than the baselines on the same (shared) GPU (median 25 s against 1.4 to 3.5 s), so on this task it buys no advantage over MTP.

## Settings

`examples/bugtrap.py` with the benchmark setting `--horizon 2.0 --num-knots 11 --spline-type cubic`: planning step 0.01 s, horizon 2.0 s (200 steps),
11 knots, cubic spline, control dimension 2 x 11 = 22, 5 iterations, control at 50 Hz, 100 cycles per run (2.0 s of simulated time), seeds 0-5. The task is
hydrax's `bugtrap` (`global_mppi/tasks/bugtrap.py`). Planner parameters are those of the other tasks of this benchmark (cube, G1, walker, ...), nothing tuned:

| method | samples | other |
|---|---|---|
| PS | 2000 | noise 0.2 |
| MPPI | 2000 | noise 0.2, temperature 0.001 |
| DIAL | 2000 | noise 1.0, temperature 0.01, beta 1.0 / 1.0 |
| MTP | 2000 | **Hydrax defaults** (the MTP paper used per-task settings): m_pts 3, n_per_layer 50, degree 2, 5 elites, sigma 0.5 / 0.1 / 1.0, temperature 0.1, beta 0.1, alpha 0.5, akima |
| CBO | 2000 | **Hydrax defaults**: noise 1.0, temperature 0.01, consensus weight 1.0, noise weight 10, step 0.1 (these four values come from hydrax's bugtrap example) |
| GlobalMPPI | 1000 KSOS candidates per restart, then MPPI 1000 x 5 at noise 0.1, temperature 0.1 | CACS (h = 0.7 x vmax = 0.7), ADMM-cuda, 5 restarts, warm start, `gpu_solve` on |

**Why a horizon of 2.0 s.** The default of hydrax's example is 1.0 s. A planner minimises the cost over its horizon only, and at 1.0 s even the best
plan found for that window is to stay in the cup, so a planner that minimises the window cost is driven into the cup. The table is an analysis of the cost over one planning
window from the start state (cubic spline, knot spacing about 0.25 s; global optimum found by random search plus CEM; "stay" = the best plan that never leaves
|y| <= 0.115, "detour" = the best plan that does), not a closed-loop run of the six methods:

| horizon | knots | best plan overall: window cost / final distance | best "stay" plan: cost / final distance | best "detour" plan: cost / final distance |
|---|---|---|---|---|
| 1.0 s | 5 | 0.750 / 0.200 m (stays) | 0.750 / 0.199 m | 0.908 / 0.066 m |
| 1.5 s | 7 | 0.872 / 0.006 m (detour) | 0.936 / 0.200 m | 0.872 / 0.006 m |
| 2.0 s | 9 | 0.873 / 0.004 m (detour) | 1.122 / 0.198 m | 0.873 / 0.002 m |
| 2.0 s | 11 (benchmark) | 0.874 / 0.003 m (detour) | 1.120 / 0.199 m | 0.874 / 0.004 m |

At 1.0 s the cheapest plan in the window stays in the cup and the detour plan is more expensive, so the planners are driven into the cup; from 1.5 s on the detour
is the cheapest plan in the window and the stay plans become clearly worse. 2.0 s with 11 knots (22-D control) was chosen as the benchmark setting. **Closed-loop
runs of the six methods at 1.0 s are not part of this folder** (no such data were kept).

## cuKernelSOS requirement (GlobalMPPI only)

`ssisyg/cuKernelSOS`, branch `gpu-solve`, commit `2768612` (a WIP: GPU affine solve, off by default), selected with
`CUKERNELSOS_PYTHON=<cuKernelSOS checkout>/python`. `KSOS_GPU_SOLVE=1` is **not needed for this task**: `examples/bugtrap.py --gpu` sets `gpu_solve` itself (the walker,
double cart-pole and PushT tasks do need it). All 6 gm runs report `solve_paths: {'gpu': 505}` (101 planning calls x 5 restarts, no CPU fallback). The baselines never
call cuKernelSOS. **gm needs at least 6 GB of free GPU memory**: its cuSOLVER handles fail with an error when the (shared) GPU is nearly full.
**gm s5 was run twice:** its first batch attempts (3) all failed with cuSOLVER error 7 under GPU memory shortage (about 3 to 4 GB free), the run was repeated
once later with 8 GB free and finished normally (attempt 1, 89 minutes); the data of this folder are that repeated run.

ksos-tools and cuKernelSOS pins (as in the cube and G1 standup READMEs): ksos-tools is a git submodule of this branch, `https://github.com/Zhongqi2/ksos-tools`, branch `gpu-solve-exp`, commit `86e6163` (`git submodule update --init`); cuKernelSOS: `https://github.com/ssisyg/cuKernelSOS`, branch `gpu-solve`, commit `2768612` (on top of `4180908`), built as described in `BENCHMARK_MAP.md` ("Quick start from a fresh clone", step 3) and selected with `CUKERNELSOS_PYTHON`. MuJoCo / MJX 3.3.6; the exact package list of the env is `docs/env_ksos_pip_freeze.txt`. Hydrax is not required (ported copies of Hydrax `bd53d43`).

## Rerun

Environment: the Global-MPPI env with MuJoCo / mujoco-mjx **3.3.6** (not 4.0), the `benchmarks` branch checked out.

```bash
# everything: 6 methods x 6 seeds, retries, logs, figure, tables
TASKS=bugtrap CUKS_GPU=/path/to/cuKernelSOS_gpu/python PY=$(which python) script/overnight_queue.sh
# output: logs/overnight_runs/<timestamp>/{STATUS.txt, bugtrap/batch.log, bugtrap/results/, final_figs/bugtrap_final.png}
```

One run by hand (seed S, from the repo root):

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
# ps | mppi | dial | mtp | cbo
python examples/bugtrap.py --headless --seed-start S --seed-end $((S+1)) --max-cycles 100 --horizon 2.0 --num-knots 11 --spline-type cubic --out-dir OUT ps
# GlobalMPPI
CUKERNELSOS_PYTHON=/path/to/cuKernelSOS_gpu/python \
python examples/bugtrap.py --headless --seed-start S --seed-end $((S+1)) --max-cycles 100 --horizon 2.0 --num-knots 11 --spline-type cubic --out-dir OUT --gpu globalmppi
```

The queue passes the benchmark setting (`--horizon 2.0 --num-knots 11 --spline-type cubic`) itself. Redraw and repackage:

```bash
python script/package_task_results.py <queue folder>/bugtrap/results data/bugtrap_results   # the files of this folder
python script/final_figs.py data/bugtrap_results bugtrap data/bugtrap_results              # figure and table from this folder
```

**Timeouts of the queue** (`script/overnight_queue.sh`): 3600 s per attempt for ps, mppi, dial, mtp and cbo, 14400 s for gm (`BASE_TIMEOUT` and `GM_TIMEOUT`: the defaults, overridable as environment variables, e.g. `GM_TIMEOUT=28800 script/overnight_queue.sh`; before every attempt the queue also waits for free GPU memory, see the header of the script). A timed-out run is not retried; a crash is retried up to twice.

## Notes

- **The GPU was shared with other users' jobs during these runs**, so the plan times are inflated and vary strongly between runs: the mean plan time per
  cycle of a gm run was 11 to 16 s for seeds 0 to 2 and 52 to 60 s for seeds 3 to 5, and a baseline run took 1.4 minutes (ps, seed 0) in the quiet early
  part of the batch and up to 52 minutes (mppi, seed 4) later. Compare costs and distances, not times.
- 6 seeds per method; the start state is fixed, so the seed only changes the controllers' random draws. IQR and std over 6 values are rough.
- Each run stops after 100 cycles (2 s); whether more cycles would change the final distances was not tested.
