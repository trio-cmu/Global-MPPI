"""Package the runs of an example task (walker, double_cart_pole, bugtrap, g1_walk) for the repo: cycles.csv and final.json of every run,
without videos, npy files or args.json (python 3.6 compatible).

    python package_task_results.py RESULTS_DIR OUT_DIR

RESULTS_DIR/<algo>/s<seed>/{cycles.csv, final.json, *best_cost_history*.npy} (algo ps mppi dial mtp cbo gm, seeds 0-5; e.g.
<queue>/walker/results). OUT_DIR gets the same layout; cycles.csv has one more row (cycle 100; 400 for g1_walk), with only best_cost (the
planner's cost of the last planning call, from the npy), when the run has a best-cost npy.
"""
import csv
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np

src, out = Path(sys.argv[1]), Path(sys.argv[2])
for a in ["ps", "mppi", "dial", "mtp", "cbo", "gm"]:
    for s in range(6):
        d, o = src / a / ("s%d" % s), out / a / ("s%d" % s)
        if not (d / "final.json").exists():
            continue   # a partial batch is packaged with what exists
        o.mkdir(parents=True, exist_ok=True)
        rows = list(csv.reader(open(str(d / "cycles.csv"))))
        f = glob.glob(str(d / "*best_cost_history*.npy"))
        if f:   # walker, double_cart_pole, g1_walk: the last row = the final (n+1-th) planning call
            h = np.squeeze(np.load(f[0]))
            n = len(rows) - 1   # cycles
            assert len(h) == n + 1 and abs(float(rows[n][2]) - h[n - 1]) < 6e-5, (a, s)
            assert rows[0][2] == "best_cost"
            rows.append([str(n), "", "%.4f" % h[n]] + [""] * (len(rows[0]) - 3))
        with open(str(o / "cycles.csv"), "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        shutil.copy(str(d / "final.json"), str(o / "final.json"))
