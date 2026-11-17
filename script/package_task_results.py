"""Package the runs of an example task (walker, double_cart_pole, bugtrap) for the repo: cycles.csv and final.json of every run,
without videos, npy files or args.json (python 3.6 compatible).

    python package_task_results.py RESULTS_DIR OUT_DIR

RESULTS_DIR/<algo>/s<seed>/{cycles.csv, final.json, *best_cost_history*.npy} (algo ps mppi dial mtp cbo gm, seeds 0-5; e.g.
<queue>/walker/results). OUT_DIR gets the same layout; cycles.csv has one more row, cycle 100, with only best_cost (the
planner's cost of the last, 101st planning call, from the npy) when the run has a best-cost npy (walker, double_cart_pole);
bugtrap runs are copied as they are.
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
        if f:   # walker, double_cart_pole: row 100 = the 101st planning call
            h = np.squeeze(np.load(f[0]))
            assert len(rows) == 101 and len(h) == 101 and abs(float(rows[100][2]) - h[99]) < 6e-5, (a, s)
            assert rows[0][2] == "best_cost"
            rows.append(["100", "", "%.4f" % h[100]] + [""] * (len(rows[0]) - 3))
        with open(str(o / "cycles.csv"), "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        shutil.copy(str(d / "final.json"), str(o / "final.json"))
