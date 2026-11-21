# obstacle_field benchmark: status

Updated 2026-11-21 01:03:00. Layout seed 19, 100 cycles, horizon 2.0 s, 11 knots, cubic spline, seeds 0 to 5. Reference (grid-search shortest path): 0.530 m; straight line 0.500 m.

Runs: `<method>_s<seed>/` (cycles.csv, final.json, args.json, video). `pending` = not finished yet (may be running).

| method | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|
| PS | finished | finished | finished | finished | finished | finished |
| MPPI | finished | finished | finished | finished | finished | finished |
| DIAL | finished | finished | finished | finished | finished | finished |
| MTP | finished | finished | finished | finished | finished | finished |
| CBO | finished | finished | finished | finished | finished | finished |
| GlobalMPPI | finished | finished | pending | pending | pending | pending |

Finished runs only:

| method | runs | mean path length (m) | reached (< 2 cm) | collision |
|---|---|---|---|---|
| PS | 6 | 0.301 | 0/6 | 0/6 |
| MPPI | 6 | 0.795 | 0/6 | 0/6 |
| DIAL | 6 | 0.765 | 0/6 | 0/6 |
| MTP | 6 | 0.791 | 0/6 | 0/6 |
| CBO | 6 | 0.817 | 0/6 | 0/6 |
| GlobalMPPI | 2 | 0.666 | 0/2 | 0/2 |

**gm (GlobalMPPI) was run on a shared, busy GPU**: its ADMM stops at its time limit (`status=time_limit` in every call we checked), so the gm results are preliminary. The baselines ran in parallel, five at a time, on the same shared GPU; plan times are not comparable with idle-GPU runs.

Files: `metrics.csv` (one row per finished run), `summary.txt` (one row per method), `reference_path.json`, `obstacle_field_trajectories.png` (all finished trajectories over the layout).

Last push: 2026-11-20 22:28:48 pushed d1d2e3b to origin/obstacle-field
