# obstacle_field benchmark: status

Updated 2026-11-24 01:41:46. Layout seed 19 (old layout), terminal goal term 10*|p-goal| (not squared), 100 cycles, horizon 2.0 s, 11 knots, cubic spline, seeds 0 to 5. Reference (grid-search shortest path): 0.530 m; straight line 0.500 m.

Runs: `<method>_s<seed>/` (cycles.csv, final.json, args.json, video). `pending` = not finished yet (may be running).

| method | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|
| PS | finished | finished | finished | finished | finished | finished |
| MPPI | finished | finished | finished | finished | finished | finished |
| DIAL | finished | finished | finished | finished | finished | finished |
| MTP | finished | finished | finished | finished | finished | finished |
| CBO | finished | finished | finished | finished | finished | finished |
| GlobalMPPI | FAILED | finished | finished | finished | finished | finished |

Finished runs only:

| method | runs | mean path length (m) | mean final distance (cm) | reached (< 2 cm) | collision | last 10 cycles all within 5 cm |
|---|---|---|---|---|---|---|
| PS | 6 | 0.657 | 0.3 | 6/6 | 0/6 | 6/6 |
| MPPI | 6 | 0.744 | 1.8 | 4/6 | 0/6 | 6/6 |
| DIAL | 6 | 0.753 | 3.0 | 1/6 | 0/6 | 6/6 |
| MTP | 6 | 0.685 | 0.3 | 6/6 | 0/6 | 6/6 |
| CBO | 6 | 0.879 | 6.3 | 0/6 | 0/6 | 0/6 |
| GlobalMPPI | 5 | 0.689 | 0.7 | 5/5 | 0/5 | 5/5 |

**gm (GlobalMPPI) was run on a shared, busy GPU**: its ADMM stops at its time limit (`status=time_limit` in every call we checked), so the gm results are preliminary. The baselines ran in parallel, five at a time, on the same shared GPU; plan times are not comparable with idle-GPU runs.

**Failures and NaN findings** (recorded only, nothing was changed or fixed):

- `gm_s0` FAILED (log `gm_s0_a2.log`): File "/home/junlong/miniforge3/envs/global-mppi/lib/python3.12/site-packages/jax/_src/xla_bridge.py", line 931, in backends / raise RuntimeError(err_msg) / RuntimeError: Unable to initialize backend 'cuda': INTERNAL: no supported devices found for platform CUDA (you may need to uninstall the failing plugin package,

Files: `metrics.csv` (one row per finished run), `summary.txt` (one row per method), `reference_path.json`, `obstacle_field_trajectories.png` (all finished trajectories over the layout).

