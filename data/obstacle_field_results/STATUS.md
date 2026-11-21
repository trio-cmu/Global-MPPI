# obstacle_field benchmark: status

Updated 2026-11-20 19:28:50. Layout seed 19, 100 cycles, horizon 2.0 s, 11 knots, cubic spline, seeds 0 to 5. Reference (grid-search shortest path): 0.530 m; straight line 0.500 m.

Runs: `<method>_s<seed>/` (cycles.csv, final.json, args.json, video). `pending` = not finished yet (may be running).

| method | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|
| PS | finished | finished | pending | pending | pending | pending |
| MPPI | finished | finished | pending | pending | pending | pending |
| DIAL | finished | finished | pending | pending | pending | pending |
| MTP | finished | finished | pending | pending | pending | pending |
| CBO | finished | finished | pending | pending | pending | pending |
| GlobalMPPI | pending | pending | pending | pending | pending | pending |

Finished runs only:

| method | runs | mean path length (m) | reached (< 2 cm) | collision |
|---|---|---|---|---|
| PS | 2 | 0.275 | 0/2 | 0/2 |
| MPPI | 2 | 0.767 | 0/2 | 0/2 |
| DIAL | 2 | 0.664 | 0/2 | 0/2 |
| MTP | 2 | 0.758 | 0/2 | 0/2 |
| CBO | 2 | 0.791 | 0/2 | 0/2 |
| GlobalMPPI | 0 | | | |

**gm (GlobalMPPI) was run on a shared, busy GPU**: its ADMM stops at its time limit (`status=time_limit` in every call we checked), so the gm results are preliminary. The baselines ran in parallel, five at a time, on the same shared GPU; plan times are not comparable with idle-GPU runs.

Files: `metrics.csv` (one row per finished run), `summary.txt` (one row per method), `reference_path.json`, `obstacle_field_trajectories.png` (all finished trajectories over the layout).

Last push: 2026-11-20 18:58:56 pushed 94a813e to origin/obstacle-field
