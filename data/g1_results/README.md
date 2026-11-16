# G1 standup benchmark (g1_fixed), MuJoCo 3.3.6

Unitree G1 humanoid standing up from a fallen start (50 Hz, horizon 0.6 s, 4 knots, 29 actuators, control dimension 116), 6 seeds (0-5) × {ps, mppi, dial, gm}, 100 control cycles, headless, run on a shared RTX 5090 (2026-11-13).

## How to run

```bash
script/fetch_g1_model.sh                                   # once: downloads the model (35 MB), checks sha256
export CUKERNELSOS_PYTHON=<path>/cuKernelSOS_gpu/python    # gpu-solve build, see docs/cube_benchmark_reproduce.md
script/run_g1_benchmark.sh runs/g1_benchmark               # ps mppi dial gm, seeds 0-5, 100 cycles
python data/g1_results/figures/plot_g1_cost.py runs/g1_benchmark --out g1_fixed_cost.png
```

One run: `python examples/g1.py --headless --seed-start 0 --seed-end 1 --max-cycles 100 --out-dir runs/g1 ps` (also `mppi`, `dial`; `--gpu globalmppi` for gm).

## The task and the one-line fix

`global_mppi/tasks/humanoid_standup.py` is the upstream hydrax task (`hydrax/tasks/humanoid_standup.py`, bd53d43) with one change. Upstream, the orientation term is `sum(square(rotate([0,0,1], torso_quat)))`: the squared norm of a rotated unit vector, which is always 1, whatever the posture. It adds a constant 10 per step to the running cost, i.e. a constant 16.0 (= 10 × (30 steps × 0.02 s + 1)) to the cost of every planning rollout, and gives the planner no gradient towards an upright torso. Here it is `sum(square(rotate([0,0,1], torso_quat)[:2]))`, which is sin² of the tilt (0 when upright). Everything else is unchanged (height term to 0.9 m, nominal joint posture term). Note that it is also 0 for a robot upside down.

The same experiment was also run on the unmodified upstream task (same settings, 48 runs in total with this version); the results are available on request.

## Versions and settings

- MuJoCo / mujoco-mjx 3.3.6; Global-MPPI `db94bc7` (clean tree, as recorded in every `args.json`). The runs were made with a driver script that `examples/g1.py` is the port of (same controller, simulation and initial-state settings, checked with a 10-cycle rerun; reruns of one seed are not bit-identical on the GPU), ksos-tools `86e6163`, cuKernelSOS `2768612` (`gpu-solve`), G1 model from hydrax `bd53d43` (`hydrax/models/g1`, from MuJoCo Menagerie, BSD-3-Clause, Unitree Robotics).
- Parameters are those of the cube benchmark (`data/cube_results_mj336/README.md`): ps / mppi / dial 2000 samples × 5 iterations, noise 0.2 (ps, mppi) / 1.0 (dial), temperature 0.001 (mppi) / 0.01 (dial), beta 1.0 / 1.0; gm `--gpu` with CACS, 5 restarts × 1000 candidates, ADMM-cuda with `gpu_solve`, warm start, final MPPI 1000 samples × 5 iterations, noise 0.1, temperature 0.1. Changed for G1: horizon 0.6 s, and the CACS radius `h = 0.7 × vmax = 2.162` (vmax = 3.089 here, 1.88 for the cube; cube.py's scaling rule).
- Simulated robot: timestep 0.01, stiffer contacts (`o_solimp`), start = "stand" keyframe with base quaternion (0.7, 0, -0.7, 0) (lying on its side). Every gm run has `solve_paths = {'gpu': 505}` (all ADMM solves on the GPU).

## Results (6 seeds, after 100 cycles)

Final torso height (m) / tilt (deg) per seed (`final.json`):

| algo | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|
| ps | 0.11 / 107 | 0.33 / 9 | 0.12 / 111 | 0.10 / 92 | 0.04 / 93 | 0.12 / 109 |
| mppi | 0.59 / 47 | 0.91 / 1 | 0.90 / 3 | 0.89 / 11 | 0.95 / 6 | 0.70 / 6 |
| dial | 0.94 / 14 | 0.99 / 11 | 0.81 / 5 | 0.33 / 175 | 0.67 / 16 | 0.90 / 6 |
| gm | 0.87 / 1 | 0.96 / 1 | 0.87 / 2 | 0.94 / 3 | 0.68 / 3 | 0.74 / 7 |

Median / mean over seeds. Streak = longest run of consecutive cycles with torso height > 0.9 m and tilt < 10° ("standing"); last-25 = mean over the last 25 cycles:

| algo | final torso h (m) | final tilt (deg) | last-25 tilt (deg) | last-25 torso h (m) | longest streak (cycles) | plan time per cycle (s) |
|---|---|---|---|---|---|---|
| ps | 0.11 / 0.14 | 100.0 / 86.9 | 99.2 / 86.9 | 0.09 / 0.12 | 0.0 / 0.0 | 0.53 / 0.59 |
| mppi | 0.89 / 0.82 | 6.3 / 12.4 | 8.2 / 21.6 | 0.81 / 0.81 | 2.0 / 13.0 | 0.53 / 0.54 |
| dial | 0.86 / 0.77 | 12.4 / 37.9 | 9.8 / 36.0 | 0.89 / 0.80 | 17.0 / 14.7 | 0.53 / 0.59 |
| gm | 0.87 / 0.84 | 2.5 / 2.9 | 3.3 / 3.9 | 0.81 / 0.82 | 4.5 / 9.5 | 9.18 / 9.91 |

gm has the smallest and most consistent tilt (every seed 1-7° at the end), but it does not stand longer than mppi or dial by the streak measure, and the height often stays at 0.7-0.9 m. Six seeds are too few for a ranking. Plan times were measured on a GPU shared with other jobs.

## Files

`<algo>/s<seed>/` with `cycles.csv` (cycle, plan_s, best_cost, torso_h, tilt_deg, gpu_mem_mb, solve_paths; cycles 0-99), `final.json` (state after cycle 100, `solve_paths` for gm), `args.json`. `figures/g1_fixed_cost.png` (median and 25-75 % band of `best_cost`, from `cycles.csv`) and `figures/plot_g1_cost.py`. No videos or logs. The `args.json` files were written by the driver script used for the runs (equivalent settings; they contain this machine's paths).
