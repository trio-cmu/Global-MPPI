"""PushT closed-loop cost curves in the style of data/logs/plot_result.py: median over seeds, band = 25%-75%
quantiles, log y axis.

    python plot_paper_style.py [log_dir] [out.png]      (defaults: six_seeds_main, pusht_closed_loop_paper.png)

Reads <method>_seed<k>.log (the pusht.py closed-loop logs) for the methods below; a .pdf is saved next to the .png.
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).resolve().parent
log_dir = here / (sys.argv[1] if len(sys.argv) > 1 else "six_seeds_main")
out_path = here / (sys.argv[2] if len(sys.argv) > 2 else "pusht_closed_loop_paper.png")

METHODS = {  # log prefix -> (name, color), plotted in this order
    "globalmppi_cacs5new": ("Global-MPPI", "tab:blue"),
    "ps2000": ("Predictive Sampling", "tab:orange"),
    "mppi2000": ("MPPI", "tab:green"),
    "dial2000": ("DIAL-MPC", "tab:red"),
}


def load(prefix):
    """(seeds, cycles) array of the best cost per control cycle."""
    runs = []
    for path in sorted(log_dir.glob(f"{prefix}_seed*.log")):
        text = path.read_text(errors="ignore")
        cost = {int(float(c)): float(v) for c, v in re.findall(r"Cycle ([\d.]+): Best cost: \[?([-\d.e+]+)", text)}
        runs.append(np.array([cost[c] for c in sorted(cost)]))
    if not runs:
        raise FileNotFoundError(f"no {prefix}_seed*.log in {log_dir}")
    n = min(len(r) for r in runs)
    return np.stack([r[:n] for r in runs])


fig, ax = plt.subplots(figsize=(6, 5))
for prefix, (name, color) in METHODS.items():
    costs = load(prefix)
    iters = np.arange(costs.shape[1])
    q25, med, q75 = np.quantile(costs, [0.25, 0.5, 0.75], axis=0)
    ax.plot(iters, med, label=name, color=color, linestyle="-", linewidth=2)
    ax.fill_between(iters, q25, q75, color=color, alpha=0.25)
    print(f"{name:20s} {costs.shape[0]} seeds, final median {med[-1]:.2e} (IQR {q25[-1]:.2e} - {q75[-1]:.2e})")

ax.set_title("PushT")
ax.set_xlabel("Iteration")
ax.set_ylabel("Cost")
ax.set_yscale("log")
ax.tick_params(axis="both", which="minor", length=1.5, width=0.4)
ax.grid(True, alpha=0.3, linewidth=0.8)
ax.legend(loc="lower left", frameon=True, fancybox=False, framealpha=0.9, edgecolor="0.3")
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
print(f"Saved: {out_path}")
