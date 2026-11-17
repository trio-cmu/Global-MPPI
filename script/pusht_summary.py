"""PushT six-method summary from the run logs (python 3.6 compatible).

    python pusht_summary.py SEEDS_LOG_DIR OUT_TXT

Reads <prefix>_seed<k>.log for ps2000 mppi2000 dial2000 mtp2000 cbo2000 globalmppi_cacs5new ("Cycle N: Best cost" and
"Cycle N: plan time" lines). No GCS.
"""
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

src, out = Path(sys.argv[1]), sys.argv[2]
M = [("ps2000", "PS"), ("mppi2000", "MPPI"), ("dial2000", "DIAL"), ("mtp2000", "MTP"), ("cbo2000", "CBO"), ("globalmppi_cacs5new", "GlobalMPPI (CACS)")]
L = []


def p(s=""):
    L.append(s)
    print(s)


p("=== PushT closed loop, 100 cycles, per method over the seeds found ===")
p("%-18s %-6s %-30s %-24s %-14s" % ("method", "seeds", "final cost mean ± std", "median (IQR) at cycle 100", "plan_s (no JIT)"))
per_seed = {}
for prefix, label in M:
    runs = []
    for f in sorted(src.glob("%s_seed*.log" % prefix)):
        t = f.read_text(errors="ignore")
        c = {int(float(a)): float(b) for a, b in re.findall(r"Cycle ([\d.]+): Best cost: \[?([-\d.e+]+)", t)}
        pl = {int(float(a)): float(b) for a, b in re.findall(r"Cycle ([\d.]+): plan time: ([\d.]+)s", t)}
        if c:
            runs.append((int(re.search(r"seed(\d+)", f.name).group(1)), c[max(c)], max(c), np.mean([v for k, v in pl.items() if k > 0]) if len(pl) > 1 else float("nan")))
    per_seed[label] = runs
    if not runs:
        p("%-18s %-6s (no logs)" % (label, 0))
        continue
    fin = [r[1] for r in runs]
    q = np.quantile(fin, [0.25, 0.5, 0.75])
    p("%-18s %-6d %-30s %-24s %-14s" % (label, len(runs), "%.3g ± %.3g" % (st.mean(fin), st.stdev(fin) if len(fin) > 1 else 0.0),
                                     "%.3g (%.3g-%.3g)" % (q[1], q[0], q[2]), "%.2f s" % st.mean([r[3] for r in runs])))
p("\nfinal cost per seed:")
for label, runs in per_seed.items():
    p("  %-18s %s" % (label, "  ".join("s%d %.3g (c%d)" % (s, c, n) for s, c, n, _ in runs)))
open(out, "w").write("\n".join(L) + "\n")
