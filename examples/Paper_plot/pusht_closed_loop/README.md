# PushT closed loop, 6 seeds (main figure)

`pusht_closed_loop_6seeds_main.png`: PS, MPPI, DIAL (2000 × 5 each), Global-MPPI (CACS ×5, k512) and GCS.
Panels: cost curve (geometric mean ± 1 std over seeds), final cost, and planning time per cycle.

Regenerate:

```bash
~/miniconda3/envs/ksos/bin/python plot_closed_loop_seeds.py --gcs
```

By default the script reads `logs/` and writes `pusht_closed_loop_6seeds_main.png` in this folder.
To write to a different file: `python plot_closed_loop_seeds.py logs other.png --gcs`.

Cost curve only (the left panel; x "Iteration", y "Cost"; no titles):

```bash
~/miniconda3/envs/ksos/bin/python plot_cost_curve.py      # -> pusht_closed_loop_cost_curve.png
```

## Data

- `logs/<method>_seed<k>.log`, seeds 0–5. These are the closed-loop logs from `pusht.py`, copied from
  `examples/single_run_test/PushT/closed_loop/seeds/` through the `six_seeds_main/` links.
  - `ps2000`: Predictive Sampling
  - `mppi2000`: MPPI
  - `dial2000`: DIAL-MPC
  - `globalmppi_cacs5new`: Global-MPPI
- The script parses `Cycle N: Best cost: ...` and `Cycle N: plan time: ...s` from each log.
  - Cycle 0 (JIT) is left out of the time statistics.
  - Final cost is the cost at the last cycle (100).
- GCS has no log. Its 6 per-run planning times are hard-coded as `GCS_TIMES` in the script, and its final cost is
  set to 1e-5. `--gcs` adds these as bars.

Source script: `examples/single_run_test/PushT/closed_loop/plot_closed_loop_seeds.py`. The only change in this copy:
the default input and output paths are relative to this folder.
