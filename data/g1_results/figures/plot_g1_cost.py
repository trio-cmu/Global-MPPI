"""G1 standup (g1_fixed) cost convergence: median over seeds (line) + 25-75 % band, log y, as plot_cube_cost.py.

    python plot_g1_cost.py DATA_DIR [--out g1_fixed_cost.png]

DATA_DIR holds <algo>/s<seed>/cycles.csv (algo = ps, mppi, dial, gm), e.g. data/g1_results. cost = the best_cost column
(the planner's predicted cost of its plan), cycles 0..99. Prints the median cost at the last cycle.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--out", default="g1_fixed_cost.png")
ap.add_argument("--seeds", type=int, default=6)
args = ap.parse_args()

COLORS = {"Global-MPPI": "tab:blue", "Predictive Sampling": "tab:orange", "MPPI": "tab:green", "DIAL-MPC": "tab:red"}
ALGO = {"Global-MPPI": "gm", "Predictive Sampling": "ps", "MPPI": "mppi", "DIAL-MPC": "dial"}
data = {
    name: np.stack([[float(r["best_cost"]) for r in csv.DictReader(open(Path(args.data) / a / f"s{s}" / "cycles.csv"))] for s in range(args.seeds)])
    for name, a in ALGO.items()
}
assert all(v.min() > 0 for v in data.values()), "non-positive cost on a log axis"
fig, ax = plt.subplots(figsize=(5.8, 6))
for name, c in data.items():
    q25, med, q75 = np.quantile(c, [0.25, 0.5, 0.75], axis=0)
    x = np.arange(c.shape[1])
    ax.plot(x, med, label=name, color=COLORS[name], lw=2)
    ax.fill_between(x, q25, q75, color=COLORS[name], alpha=0.25)
ax.set_title("G1 standup (orientation term on 2 components)")
ax.set_xlabel("Iteration"); ax.set_ylabel("Cost"); ax.set_yscale("log"); ax.set_xlim(0, c.shape[1] - 1)
ax.grid(True, alpha=0.3, lw=0.8); ax.tick_params(axis="both", which="minor", length=1.5, width=0.4)
ax.legend(loc="lower left", frameon=True, fancybox=False, framealpha=0.9, edgecolor="0.3")
plt.tight_layout(); fig.savefig(args.out, dpi=300, bbox_inches="tight")
print("saved", args.out)
for name, c in data.items():
    q = np.quantile(c[:, -1], [0.25, 0.5, 0.75]); print("%-20s last cycle (%d): median %.3f  IQR [%.3f, %.3f]" % (name, c.shape[1] - 1, q[1], q[0], q[2]))
