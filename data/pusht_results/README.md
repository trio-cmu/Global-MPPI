# PushT closed loop: six methods x 6 seeds

`pusht_final.png`: planner cost vs cycle (median over 6 seeds, IQR band, log y); cost at cycle 100 per method (median, IQR bars,
faint dots = the 6 seeds); plan time per cycle (median, IQR band, log y). `pusht_summary.txt`: the table (median / mean ± std /
IQR of the cost at cycle 100, plan time) and the cost of every seed. `<method>/s<seed>/cycles.csv`: `cycle` (0..100), `best_cost`
(the planner's predicted cost of the cycle, as printed in the log "Cycle N: Best cost"), `plan_s` (seconds; empty at cycle 100,
which is only the final cost; cycle 0 includes JIT). No raw logs, no videos. Methods: `ps mppi dial mtp cbo gm` (gm = GlobalMPPI).

| method | cost@100 median (IQR) | mean ± std | plan_s median |
|---|---|---|---|
| PS | 7.9e-4 (5.9e-4 - 2.5e-3) | 0.59 ± 1.44 | 0.57 s |
| MPPI | 0.405 (0.36 - 0.417) | 0.39 ± 0.035 | 0.56 s |
| DIAL | 0.0231 (0.0151 - 0.0243) | 0.020 ± 0.0075 | 0.48 s |
| MTP | 1.55e-4 (9.5e-5 - 2.1e-4) | 2.3e-4 ± 2.7e-4 | 0.49 s |
| CBO | 0.019 (0.0174 - 0.0263) | 0.113 ± 0.227 | 0.51 s |
| GlobalMPPI | 3.15e-4 (2.0e-4 - 4.0e-4) | 3.4e-4 ± 2.3e-4 | 14.67 s |

## Settings

Task and controller settings are those of `examples/pusht.py`, unchanged: scene `models/pusht/scene.xml` (the paper model), start
`qpos = [0.1, 0.1, 1.3, 0, 0]`, cubic spline, 6 knots, horizon 1.0 s, 5 iterations, planning at 50 Hz, simulation at 1000 Hz,
100 cycles per run, seeds 0-5. Nothing was tuned.

| method | samples | other |
|---|---|---|
| PS | 2000 | noise 0.4 |
| MPPI | 2000 | noise 1.0, temperature 0.1 |
| DIAL | 2000 | noise 1.0, temperature 0.01, beta 1.0 / 1.0 |
| MTP | 2000 | hydrax defaults (`global_mppi/algs/mtp.py`: m_pts 3, n_per_layer 50, degree 2, 5 elites, sigma 0.5 / 0.1 / 1.0, temperature 0.1, beta 0.1, alpha 0.5, akima) |
| CBO | 2000 | hydrax defaults (`global_mppi/algs/cbo.py`: noise 1.0, temperature 0.01, consensus weight 1.0, noise weight 10, step 0.1) |
| GlobalMPPI | 1000 KSOS candidates per restart, then MPPI 1000 x 5 at noise 0.1 | CACS sampling, ADMM-cuda, 5 restarts, warm start, `gpu_solve` on |

MTP and CBO are ported from hydrax `bd53d43`; their hyperparameters are the untuned upstream defaults (CBO's four required values are
those of hydrax's bugtrap example).

## cuKernelSOS requirement (GlobalMPPI only)

GlobalMPPI needs the `gpu-solve` build of cuKernelSOS: `ssisyg/cuKernelSOS`, branch `gpu-solve`, commit `2768612`
(a WIP, not merged: GPU affine solve, off by default). `gpu_solve` is switched on with `KSOS_GPU_SOLVE=1` and the build is selected
with `CUKERNELSOS_PYTHON=<cuKernelSOS checkout>/python`. `KSOS_GPU_SOLVE=1` is needed for PushT (`examples/pusht.py` reads it); the cube and G1 benchmarks do not need it (`examples/cube.py --gpu` and `examples/g1.py --gpu` set `gpu_solve` themselves), the other tasks do. All 6 gm runs report `solve_paths: {'gpu': 505}` (101 cycles x 5 restarts,
all on the GPU, no CPU fallback). The baselines never call cuKernelSOS.

## Rerun

Environment: the Global-MPPI env with MuJoCo / mujoco-mjx **3.3.6** (not 4.0), the `benchmarks` branch checked out.

```bash
# everything: 6 methods x 6 seeds, retries, logs, figure, tables (3.5 h on our shared GPU)
CUKS_GPU=/path/to/cuKernelSOS_gpu/python PY=$(which python) script/overnight_queue.sh
# output: logs/overnight_runs/<timestamp>/{STATUS.txt, pusht/batch.log, pusht/summary.txt, final_figs/pusht_final.png}
```

One run by hand (seed S, from the repo root):

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# ps | mppi | dial | mtp | cbo
python examples/pusht.py --no-wandb --headless --seed S --num-samples 2000 ps
# GlobalMPPI
KSOS_GPU_SOLVE=1 CUKERNELSOS_PYTHON=/path/to/cuKernelSOS_gpu/python \
python examples/pusht.py --no-wandb --headless --seed S --ksos-solver ADMM-cuda --ksos-sampling cacs \
    --ksos-restarts 5 --ksos-warm-start --mppi-noise 0.1 globalmppi
```

Logs go to `examples/single_run_test/PushT/closed_loop/seeds/<ps2000|mppi2000|dial2000|mtp2000|cbo2000|globalmppi_cacs5new>_seed<S>.log`
(the queue does this; by hand redirect stdout there). Then, from the logs (python 3.6+, needs numpy and matplotlib):

```bash
python script/pusht_extract_csv.py <seeds log dir> data/pusht_results        # logs -> <method>/s<seed>/cycles.csv
python script/final_figs.py data/pusht_results pusht data/pusht_results      # figure and table from the csv files
```

## Notes

- Timeouts of `script/overnight_queue.sh`: 3600 s per attempt for ps, mppi, dial, mtp and cbo, 14400 s for gm (`BASE_TIMEOUT` and `GM_TIMEOUT`: the defaults, overridable as environment variables, e.g. `GM_TIMEOUT=28800 script/overnight_queue.sh`; before every attempt the queue also waits for free GPU memory, see the header of the script). A timed-out run is not retried. gm may need more than 14400 s on a busy GPU.
- This folder holds `cycles.csv` per run only, no `args.json`: the settings in the table above were checked against the defaults in the code (`examples/pusht.py`, `global_mppi/algs/`), not against recorded files.
- The GPU was shared with other users' jobs during these runs. Plan times are therefore inflated (GlobalMPPI about 14.7 s per cycle
  against about 7.3 s in the senior's reproduction document); compare the costs, not the times. The queue stored an `nvidia-smi`
  snapshot before every run (not included here).
- 6 seeds per method; the seed changes the controller's random draws only (the start state is fixed). IQR and std over 6 values are
  rough: PS has one seed stuck at a cost of 3.5 (it drives the mean and std of PS, not its median).
- `best_cost` is the planner's predicted cost, not the realised cost of the closed-loop trajectory.
- No GCS in these figures or tables.
