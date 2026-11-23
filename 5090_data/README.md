# RTX 5090 benchmark reruns (benchmarks branch code)

Started 2026-11-19 on this desktop's RTX 5090. Both batches run the `benchmarks` branch at `7d71c56`
(the code of `data/pusht_results` and `data/double_cart_pole_results`), not the `pre-release` checkout.

- `pusht_benchmark/`: PushT, PS / MPPI / DIAL / MTP / CBO / GlobalMPPI x seeds 0-5, 100 cycles. Runner and
  layout copied from `data/pusht_4090_data` (see its README); progress in `pusht_benchmark/STATUS.txt`,
  `batch.log`; figure and table `pusht_final.png`, `pusht_summary.txt` (update after every run).
- `double_cart_pole_benchmark/<timestamp>/`: double cart-pole, same 6 x 6, run by the branch's own
  `script/overnight_queue.sh` (`TASKS=double_cart_pole`); progress in `STATUS.txt` there.
- `pusht_gcs_cost/`: the PushT batch again with the GCS cost (50|p|^2 + 15*2(1-cos th) + 10 x pusher path
  off the T); see its README.
- `g1_benchmark/`: G1 standup, ps / mppi / dial / mtp / cbo / gm x seeds 0-5, 100 cycles, by the branch's
  `script/run_g1_benchmark.sh` (settings as `data/g1_results`); `batch.log`, `logs/`, `<algo>/s<seed>/final.json`.
  Queued by `run_g1.sh` after `run_next.sh` (cart-pole) finishes. G1 model fetched by `script/fetch_g1_model.sh`.
- `gcs_replay_gcs_cost/`, `gcs_open_loop_video/`: exported GCS plans replayed / rendered and scored (no GCS solve).
- Queue: `run_all.sh` started the PushT benchmark; it was replaced by `run_next.sh`, which waits for the
  PushT benchmark to finish, then runs `pusht_gcs_cost/`, then the double cart-pole. Output: `queue.out`.

## Setup (all in `~/.cache/global-mppi/pusht_5090/`, separate from the repo and other envs)

- `Global-MPPI/`: git worktree of `benchmarks` @ `7d71c56`.
- `ksos-tools/`: `Zhongqi2/ksos-tools` `gpu-solve-exp` @ `86e6163`.
- `cuKernelSOS/`: `ssisyg/cuKernelSOS` `gpu-solve` @ `2768612`, built with CUDA 12.9 for sm_120.
- `venv/`: Python 3.12.3, `docs/env_ksos_pip_freeze.txt` installed with `--no-deps` (JAX 0.5.3, MuJoCo/MJX 3.3.6),
  plus matplotlib 3.11.2 for the figures (numpy kept at 2.5.4).

GlobalMPPI runs use `KSOS_GPU_SOLVE=1` and `CUKERNELSOS_PYTHON=<above>/cuKernelSOS/python`; a PushT gm run only
counts as complete with `solve_paths == {'gpu': 505}`.

Check: PS seed 0 reproduces `data/pusht_results` exactly (cost at cycle 100 = 5.5251472e-05).
`systemd-inhibit` is not permitted for this user; the desktop's AC sleep setting is "nothing".

Resume PushT after an interruption (completed runs are skipped):
`~/.cache/global-mppi/pusht_5090/venv/bin/python 5090_data/pusht_benchmark/run_benchmark.py`

## Overnight queue (2026-11-20, `run_overnight.sh`, output in `queue.out`)

G1 -> `g1_benchmark/`, cube -> `cube_benchmark/`, G1 walk -> `g1_walk_benchmark/<timestamp>/`, then the double cart-pole with
CACS h = 0.3 -> `double_cart_pole_h0.3/` (all 6 methods x seeds 0-5). The cart-pole run with h = 0.7 was stopped: in its 4-D control
space (1 actuator x 4 knots) CACS could not place 1000 samples (150-750 after 600-1400 rounds, ~14 s per restart, ~78 s per cycle).
10-cycle GlobalMPPI checks on a free GPU: G1 (h 2.162), cube (h 1.316) and G1 walk (h 2.162) fill 1000 samples in 3-4 rounds,
about 4-8 s per cycle, so their default h is kept; cart-pole h = 0.3 fills 1000 in ~16 rounds, ~9.6 s per cycle. The cart-pole
h = 0.3 results differ in that one setting from `data/double_cart_pole_results` (h = 0.7). Its seed-0 baselines (ps, mppi, dial,
mtp, cbo; h is not used by them) are copied from the stopped h = 0.7 batch. The G1 walk gm runs took 36-43 s per cycle in
`data/g1_walk_results` because that GPU was shared, not because of h.

### Results as they finish

- G1 (`g1_benchmark/`, 00:26-02:05): 36/36, no failed runs, all gm `solve_paths == {'gpu': 505}`, gm 5.0 s per cycle.
  Median cost at cycle 100: PS 11.23, MPPI 1.47, DIAL 2.28, MTP 4.63, CBO 6.28, Global-MPPI 0.84 (lowest).
  Figure and tables: `g1_benchmark/figures/` (made by `data/g1_results/figures/plot_g1_six.py`).
- Cube (`cube_benchmark/`, 02:05-04:04): 36/36, no failed runs, all gm `solve_paths == {'gpu': 505}`, gm 6.8 s per cycle.
  Median cost at cycle 100: PS 0.401, MPPI 0.024, DIAL 0.038, MTP 0.041, CBO 0.045, Global-MPPI 0.021 (lowest; 0.0048 in
  `data/cube_results_mj336`). Figure and tables: `cube_benchmark/figures/` (made by `data/cube_results_mj336/figures/plot_cube_six.py`).
- G1 walk (`g1_walk_benchmark/20261120_040452/`, 04:04-10:36): 36/36, no failed runs, all gm `solve_paths == {'gpu': 2005}`,
  gm 6.0 s per cycle. Falls out of 6: PS 1, MPPI 4, DIAL 2, MTP 0, CBO 6, gm 0. Mean tracking cost: MTP 0.126, PS 0.159,
  gm 0.233, DIAL 0.374, MPPI 0.454, CBO 0.729. Figure and tables: `final_figs/` (the batch log says `g1_walk/figures/`; they
  are in `final_figs/`).
- Double cart-pole h = 0.3 (`double_cart_pole_h0.3/`): stopped by the user at 10:37 on 2026-11-20, during its first gm run (gm s0,
  incomplete, no `final.json`). Only the 5 seed-0 baselines copied from the h = 0.7 batch are there. Rerun `run_overnight.sh`'s
  cart-pole step to resume; completed runs (with `final.json`) are skipped.
- G1 MPPI rerun (2026-11-20 10:44-10:52, user request): `g1_benchmark/mppi/` is the rerun, the first run is `g1_benchmark/mppi_run1/`
  (logs in `logs/mppi_run1/`). Same settings and seeds, but results differ already at cycle 0 (GPU floating-point nondeterminism in
  the MJX rollouts; G1 runs are not bit-reproducible). Rerun: falls (tilt > 90 deg) 2/6 (s1, s4) vs 1/6 (s3) before; median cost at
  cycle 100 3.22 vs 1.47. `figures/` is redrawn with the rerun; the previous figure/table are `figures/*_mppi_run1.*`.
- PushT path cost in GCS's model (`pusht_gcs_model_path_cost/`, 2026-11-20 11:48-13:39): the README commands of
  `4090_data/pusht_gcs_model_path_cost/` at d69dfa4 (worktree `~/.cache/global-mppi/pusht_5090/Global-MPPI-pathcost`), run through
  `run_with_own_cache.py` (/tmp/jax_cache belongs to another user). 36/36, all 101 states. J = L_T + 0.1 L_P + 10 D, mean over 6
  seeds: GCS (4090 plan, `4090_data/gcs_path_pusher_wp0.1/`) 0.170, Global-MPPI 0.204, MTP 0.202, DIAL 0.236, PS 0.236, MPPI 0.308,
  CBO 1.343. With 20 D: gm 0.227, MTP 0.229; with 100 D: gm 0.409, MTP 0.444. Best gm seed 1 (0.190): `best_globalmppi_s1_*.mp4`;
  top views by `render_top_view.py`. My own GCS plan with this cost (`gcs_plan/gcs_path_wp0.1_traj.npz`, 0.197) is not used.

### PushT path cost, pusher motion after the T stops not counted (2026-11-20)
`score_path_cost.py --drop-tail` / `plot_path_cost.py --drop-tail`: L_P leaves out the pusher path after the T's last
motion (the pusher wandered 130-240 mm there in most controller runs; GCS returns home, 129 mm). Applied to every
method and GCS. Mean J (w = 10 / 20 / 100): GCS 0.157; gm 0.185 / 0.208 / 0.390; MTP 0.194 / 0.221 / 0.436;
DIAL 0.220; PS 0.226; MPPI 0.289; CBO 1.163 (its T never moves on 5 seeds, so its whole pusher path is "after").
Figures: `pusht_gcs_model_path_cost/pusht_gcs_model_path_cost{,_wfinal20,_wfinal100}_droptail.png`.

### PS sample sweep, PushT path cost (2026-11-20, 16:00-17:45)
`pusht_gcs_model_path_cost/run_ps_samples.sh`: PS with 2000 / 4000 / 8000 / 16000 / 32000 samples per iteration, 5
iterations, seeds 0-5, otherwise as the main batch (ps_n<N>/). All 30 runs finished. Other users' jobs shared the GPU,
so times are slower than the main batch (2000: 377 vs 171 ms/cycle). Mean J (drop-tail, w = 10) / ms per cycle:
2k 0.226 / 377, 4k 0.208 / 623, 8k 0.213 / 1175, 16k 0.210 / 2174, 32k 0.192 / 4788.
Figure: `pusht_pareto_percycle_pssweep_droptail.png` (`plot_pareto.py --per-cycle --ps-sweep`).
PS 64000 samples (2026-11-21, 10:54-11:33, free GPU): J 0.188 ± 0.028, 3639 ms/cycle; all 6 seeds finished (ps_n64000/).

### Why Global-MPPI (cuKernelSOS) is slow on PushT path cost (2026-11-21, seed 0 rerun on a free GPU)
`pusht_gcs_model_path_cost/gm_profile/` (profile_gm.py wraps the KSOS steps with timers; calls_seed0.csv, console_s0.log).
Same J as before (0.180). 9.1 s per cycle = 5 restarts x ~1.8 s: ADMM solve_ksos 5.7 s (630 iterations x 1.4 ms +
0.18 s setup per restart; 18% of restarts hit the 2 s time limit), build_problem (CPU banded Cholesky) 1.7 s, CACS 1.3 s
(rollouts only 0.28 s of it), cliques 0.1 s, final MPPI 0.2 s. The ADMM solution wins the guard in 25% of restart-0s
and 2-6% of restarts 1-4; restarts 1-4 improve the plan's cost in 34/100 cycles, by 3.8% on average.

### Global-MPPI with single-thread OpenBLAS (gm_blas1, 2026-11-22, free 5090)
Same settings as `gm/`; only `OPENBLAS_NUM_THREADS=1` (script `pusht_gcs_model_path_cost/run_gm_blas1.sh`).
| | per cycle (cycles 1-99) | total solve (99 cycles) | J drop-tail, w=10 |
|---|---|---|---|
| gm (default BLAS) | 9.20 s | 911 ± 72 s | 0.185 ± 0.011 |
| gm_blas1 (1 thread) | 6.47 s | 641 ± 64 s | 0.194 ± 0.016 |
30% faster. J is the same on 4 seeds; s1 (0.168 → 0.191) and s2 (0.195 → 0.223) differ because the ADMM time limit makes runs timing-dependent.

### Global-MPPI final-MPPI variants (run_gm_variants.sh, 1-thread BLAS, 2026-11-22)
J drop-tail w=10, 6 seeds, no ADMM failures (failed seeds from GPU memory exhaustion were rerun; originals in invalid_gpu_oom/).
| | J | per cycle |
|---|---|---|
| MPPI noise 0.1 (gm_blas1) | 0.194 ± 0.016 | 6.47 s (free GPU) |
| no final MPPI (gm_nomppi) | 0.184 ± 0.012 | 8.12 s (mostly shared GPU) |
| MPPI noise 0.2 (gm_noise0.2) | 0.215 ± 0.022 | 9.59 s (mostly shared GPU) |
Plot: pusht_pareto_sweeps_nomtp_gmvariants_droptail.png (`python plot_pareto_sweeps.py --no-mtp --gm-variants`).
