# Cube (LEAP hand) benchmark results, MuJoCo 3.3.6

6 seeds (0-5) × {ps, mppi, dial, mtp, cbo, gm, gm_noise0.02}, 100 control cycles, headless, run unattended on a shared RTX 5090 (ps, mppi, dial, gm, gm_noise0.02 on 2026-11-12; mtp and cbo on 2026-11-16). How to reproduce: `docs/cube_benchmark_reproduce.md`, driver `script/run_cube_benchmark.sh`, cost plot `script/plot_cube_cost.py`.

Full rerun (the script's default runs only ps mppi dial gm):

```bash
CUKERNELSOS_PYTHON=<path>/cuKernelSOS_gpu/python CONFIGS="ps mppi dial mtp cbo gm gm_noise0.02" script/run_cube_benchmark.sh runs/cube_benchmark
```

MTP and CBO use Hydrax defaults (the MTP paper used per-task settings). gm needs the cuKernelSOS gpu-solve branch (`ssisyg/cuKernelSOS`, branch `gpu-solve`, `2768612`), selected with `CUKERNELSOS_PYTHON`. For the cube task `KSOS_GPU_SOLVE=1` is not needed (`examples/cube.py --gpu` sets `gpu_solve` itself); the other tasks (walker, double cart-pole, bugtrap, G1 walk, PushT) need it.

The driver's timeout is 3600 s per attempt (`RUN_TIMEOUT`); gm may need more on a busy GPU.

## Versions

- MuJoCo / mujoco-mjx **3.3.6** (not 4.0.0, whose contact dynamics differ), Python 3.12, jax 0.5.3
- Global-MPPI `58f9add` (code of the ps, mppi, dial, gm and gm_noise0.02 runs), ksos-tools `86e6163` (`gpu-solve-exp`), cuKernelSOS `2768612` (`gpu-solve`, from `ssisyg/cuKernelSOS`)
- the ps, mppi, dial, gm and gm_noise0.02 `args.json` record `git_commit` 58f9add, `git_dirty` false, mujoco 3.3.6
- the mtp and cbo runs use the code of `6af9f9d` (contained in this branch: `global_mppi/algs/mtp.py`, `cbo.py`, and the `mtp` / `cbo` subcommands of `examples/cube.py`). Their `args.json` record the commit the working tree was on while the batch ran: `b9184da` with uncommitted changes (3 runs), `6af9f9d` with uncommitted changes (1 run), `227f7ca` (8 runs; the bugtrap task commit, contained in this branch; `mtp.py`, `cbo.py` and `cube.py` are identical to it); the files used (`examples/`, `global_mppi/algs`, `alg_base.py`, `simulation`, `utils`) are identical in `6af9f9d` and `227f7ca` and the uncommitted changes were this code before it was committed. mujoco 3.3.6

## Settings (all read back from `args.json`)

- horizon 0.25 s, 4 knots (zero spline), 25 Hz, 100 cycles
- ps / mppi / dial: 2000 samples × 5 iterations; noise 0.2 (ps, mppi), 1.0 (dial); temperature 0.001 (mppi), 0.01 (dial); dial beta_opt_iter = beta_horizon = 1.0
- gm: `--gpu` (CACS, 5 restarts × 1000 candidates, KSOS on ADMM-cuda with `gpu_solve`, warm start), final MPPI 1000 samples × 5 iterations, noise 0.1, temperature 0.1, CACS h = 1.316; every gm run has `solve_paths = {'gpu': 505}`
- gm_noise0.02: the same as gm with `--mppi-noise 0.02`
- **mtp and cbo (ported from hydrax `bd53d43`): upstream default hyperparameters, untuned for this task**, 2000 samples × 5 iterations like ps / mppi / dial. mtp: the defaults of its constructor (`m_pts` 3, `n_per_layer` 50, `degree` 2, `num_elites` 5, `sigma_start` 0.5, `sigma_min` 0.1, `sigma_max` 1.0, `temperature` 0.1, `beta` 0.1, `alpha` 0.5, akima interpolation). cbo has no defaults for four of its arguments; they are taken from the hydrax bugtrap example (`examples/bugtrap.py`), not from a cube-like task: `initial_noise_level` 1.0, `temperature` 0.01, `consensus_weight` 1.0, `noise_weight` 10.0 (`step_size` 0.1 is its default). The controller dict of every run is in its `args.json`.

## Final error after 100 cycles (position mm / angle deg)

| config | s0 | s1 | s2 | s3 | s4 | s5 | mean ± std |
|---|---|---|---|---|---|---|---|
| ps | 17.2 / 74.0 | 20.5 / 80.3 | 21.4 / 66.9 | 18.6 / 20.7 | 30.5 / 68.1 | 25.1 / 70.4 | 22.2 ± 4.9 / 63.4 ± 21.5 |
| mppi | 22.7 / 55.4 | 23.1 / 18.0 | 11.9 / 3.2 | 18.5 / 48.6 | 28.5 / 78.7 | 15.8 / 23.4 | 20.1 ± 5.9 / 37.9 ± 27.9 |
| dial | 24.3 / 31.0 | 8.8 / 13.9 | 19.6 / 20.4 | 8.2 / 14.4 | 25.3 / 9.3 | 21.7 / 19.8 | 18.0 ± 7.6 / 18.1 ± 7.5 |
| mtp | 5.1 / 0.6 | 17.4 / 20.7 | 20.4 / 16.5 | 14.6 / 16.6 | 11.9 / 5.7 | 28.5 / 6.6 | 16.3 ± 7.9 / 11.1 ± 7.9 |
| cbo | 16.7 / 14.8 | 15.2 / 14.7 | 16.1 / 14.9 | 10.5 / 13.9 | 12.9 / 19.8 | 12.8 / 19.8 | 14.0 ± 2.3 / 16.3 ± 2.7 |
| gm | 9.4 / 5.6 | 20.7 / 3.9 | 14.7 / 15.0 | 7.5 / 3.7 | 6.6 / 2.3 | 9.1 / 6.7 | 11.3 ± 5.4 / 6.2 ± 4.6 |
| gm_noise0.02 | 16.4 / 15.3 | 4.5 / 0.6 | 24.7 / 6.3 | 12.2 / 2.1 | 9.9 / 13.9 | 7.5 / 4.0 | 12.5 ± 7.2 / 7.0 ± 6.2 |

Mean ± std over seeds (ddof 1). GPU runs are not bit-identical between reruns. Mean planning time per cycle (without cycle 0): about 1 s for ps / mppi / dial, 2.3 s for mtp, 2.8 s for cbo, 12.9 s for gm, 11.2 s for gm_noise0.02 (GPU shared with other jobs).

## Planner cost at iteration 100 (median over 6 seeds, 25-75 % range)

`best_cost` of cycle 100, the planner's predicted cost (it includes a position term and a control term besides the orientation term):

| gm | mtp | dial | cbo | mppi | ps |
|---|---|---|---|---|---|
| 0.0048 (0.0047-0.0055) | 0.0153 (0.0047-0.0173) | 0.0308 (0.0203-0.0442) | 0.0386 (0.0377-0.0422) | 0.0894 (0.0490-0.1732) | 0.467 (0.362-0.551) |

gm is lowest on cost, position and angle. mtp is second on cost and angle, cbo second on position with the smallest spread (position ± 2.3 mm, angle ± 2.7°). Per seed, mtp beats gm on position at s0 and s1 and on angle at s0 (s5 is a tie, 6.6° vs 6.7°); cbo beats gm on position at s1 only. Six seeds, no significance test.

## Layout

`<config>/<algo>/s<seed>/` (config = algo for ps, mppi, dial, mtp, cbo, gm) with `cycles.csv` (cycle, plan_s, best_cost, pos_err_mm, ang_err_deg, gpu_util, gpu_mem_mb; cycles 0-99), `final.json` (error after cycle 100, `solve_paths` for gm), `args.json`, and `cube_nomppi_<alg>_best_cost_history_seed<k>.npy` (best_cost, cycles 0-100). `batch.log` is the log of the batch (no failed runs, no retries). `figures/`: error curves (`cube_curves.png`), final error (`cube_final.png`), cost convergence (`cube_cost_*.png`), and the six-algorithm cost convergence `cube_cost_six.png` with its tables `cube_six_summary.txt`, made by `plot_cube_six.py` (`python plot_cube_six.py [data dir] [out dir]`, reads this folder; gm_noise0.02 is not drawn).

Not included: videos, per-run logs, qpos histories (`*robot_pos_history*.npy`), the 3 partial seed-0 runs from an earlier windowed attempt. `figures/plot_cube_cost.py` reads fixed locations under `~` (`~/cube_runs_mj336`, `data/logs`); `script/plot_cube_cost.py` takes the paths as arguments. The `args.json` files contain this machine's absolute paths.
