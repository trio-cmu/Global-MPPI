# obstacle_field benchmark: status

Updated 2026-11-20 21:02:17. Layout seed 19, 100 cycles, horizon 2.0 s, 11 knots, cubic spline, seeds 0 to 5. Reference (grid-search shortest path): 0.530 m; straight line 0.500 m.

Runs: `<method>_s<seed>/` (cycles.csv, final.json, args.json, video). `pending` = not finished yet (may be running).

| method | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|
| PS | finished | finished | finished | finished | finished | pending |
| MPPI | finished | finished | finished | finished | finished | pending |
| DIAL | finished | finished | finished | finished | finished | pending |
| MTP | finished | finished | finished | finished | finished | pending |
| CBO | finished | finished | finished | finished | finished | pending |
| GlobalMPPI | pending | pending | pending | pending | pending | pending |

Finished runs only:

| method | runs | mean path length (m) | reached (< 2 cm) | collision |
|---|---|---|---|---|
| PS | 5 | 0.290 | 0/5 | 0/5 |
| MPPI | 5 | 0.818 | 0/5 | 0/5 |
| DIAL | 5 | 0.747 | 0/5 | 0/5 |
| MTP | 5 | 0.780 | 0/5 | 0/5 |
| CBO | 5 | 0.813 | 0/5 | 0/5 |
| GlobalMPPI | 0 | | | |

**gm (GlobalMPPI) was run on a shared, busy GPU**: its ADMM stops at its time limit (`status=time_limit` in every call we checked), so the gm results are preliminary. The baselines ran in parallel, five at a time, on the same shared GPU; plan times are not comparable with idle-GPU runs.

Files: `metrics.csv` (one row per finished run), `summary.txt` (one row per method), `reference_path.json`, `obstacle_field_trajectories.png` (all finished trajectories over the layout).

Last push: 2026-11-20 20:31:22 pushed 38a48da to origin/obstacle-field
