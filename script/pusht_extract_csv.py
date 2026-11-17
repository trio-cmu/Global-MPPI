"""Extract per-run cycles.csv from PushT run logs (python 3.6 compatible).

    python pusht_extract_csv.py SEEDS_LOG_DIR OUT_DIR

Reads <prefix>_seed<k>.log (ps2000 mppi2000 dial2000 mtp2000 cbo2000 globalmppi_cacs5new; "Cycle N: Best cost" and
"Cycle N: plan time" lines) and writes OUT_DIR/<ps|mppi|dial|mtp|cbo|gm>/s<k>/cycles.csv with columns cycle (0..100),
best_cost (the planner's predicted cost after the cycle's planning) and plan_s (seconds, empty at cycle 100).
"""
import csv
import re
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
PFX = {"ps": "ps2000", "mppi": "mppi2000", "dial": "dial2000", "mtp": "mtp2000", "cbo": "cbo2000", "gm": "globalmppi_cacs5new"}
for a, pre in PFX.items():
    for s in range(6):
        t = (src / ("%s_seed%d.log" % (pre, s))).read_text(errors="ignore")
        c = {int(float(k)): float(v) for k, v in re.findall(r"Cycle ([\d.]+): Best cost: \[?([-\d.e+]+)", t)}
        pl = {int(float(k)): float(v) for k, v in re.findall(r"Cycle ([\d.]+): plan time: ([\d.]+)s", t)}
        d = out / a / ("s%d" % s)
        d.mkdir(parents=True, exist_ok=True)
        with open(str(d / "cycles.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cycle", "best_cost", "plan_s"])
            for i in range(max(c) + 1):
                w.writerow([i, repr(c[i]), ("%.4f" % pl[i]) if i in pl else ""])
