# PushT on the RTX 5090 with the GCS cost

The same batch as `../pusht_benchmark/` (benchmarks code @ 7d71c56, MuJoCo 3.3.6, 6 methods x seeds 0-5,
100 cycles, the same controller settings and GlobalMPPI CACS + ADMM-cuda + gpu_solve), with one change:
the PushT cost is the GCS cost (`gcs_cost.py`, patched into the task by `run_one.py`):

    running   50 |p - goal|^2 + 15 * 2 (1 - cos th) + 10 * |pusher velocity| * [pusher off the T]
    terminal  50 |p - goal|^2 + 15 * 2 (1 - cos th)

Summed as dt * running over a rollout, the last term is 10 x the pusher's arc length while it is not
touching the T (gap > 1 mm). This is the cost of `global_mppi/tasks/pusht.py` on `pre-release`; checked
on three states to give identical running and terminal costs. Simulation in MuJoCo (no `--gcs-model`).

The costs here are planner costs under the GCS cost; they are not comparable to `../pusht_benchmark/`
(benchmark cost 50|p|^2 + 15 th^2). Layout, status files and resume: as `../pusht_benchmark/`
(`STATUS.txt`, `batch.log`, `<method>/s<seed>/`, `pusht_final.png`, `pusht_summary.txt`).
Started by `../run_next.sh` after the PushT benchmark finishes.
