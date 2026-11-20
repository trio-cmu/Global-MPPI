"""Double cart-pole: planner cost per control cycle, median over seeds with the 25-75 % band (log y).

    python plot_cycle_cost.py [run dir] [out.png]      (defaults: compare/ and double_cart_pole_cycle_cost.png here)

The run dir holds <method>/s<seed>/: the best_cost_history .npy if present (101 entries), else the
best_cost column of cycles.csv. compare/ links the baselines and the 5-restart run of
../double_cart_pole and the 1-restart run of this folder.
"""

import csv
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

here = Path(__file__).resolve().parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "compare"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "double_cart_pole_cycle_cost.png"

METHODS = {  # folder: (label, color, line style)
    "ps": ("PS", "#52514e", "-"),
    "mppi": ("MPPI", "#1baf7a", "-"),
    "dial": ("DIAL", "#eb6834", "-"),
    "mtp": ("MTP", "#8e6bd1", "-"),
    "cbo": ("CBO", "#a0785a", "-"),
    "gm5": ("cuKernelSOS (5 restarts)", "#d43d51", "-"),
    "gm1": ("cuKernelSOS (1 restart)", "#d43d51", (0, (4, 2))),
}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def load(method):
    """(seeds, cycles) planner cost."""
    runs = []
    for d in sorted((src / method).glob("s*")):
        npy = glob.glob(str(d / "*best_cost_history*.npy"))
        if npy:
            runs.append(np.squeeze(np.load(npy[0])))
        else:
            runs.append(np.array([float(r["best_cost"]) for r in csv.DictReader(open(d / "cycles.csv"))]))
    n = min(len(r) for r in runs)
    return np.stack([r[:n] for r in runs])


fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=150)
ax.grid(True, color=GRID, lw=0.8)
ax.tick_params(colors=INK_2, length=0, labelsize=11)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#c9c8c2")

for method, (label, color, ls) in METHODS.items():
    C = np.maximum(load(method), 1e-12)
    q25, med, q75 = np.quantile(C, [0.25, 0.5, 0.75], axis=0)
    x = np.arange(C.shape[1])
    ax.plot(x, med, color=color, lw=2.2, ls=ls, label=label)
    ax.fill_between(x, q25, q75, color=color, alpha=0.12, lw=0)
    print(f"{label:26s} {len(C)} seeds, final median cost {med[-1]:.3f}")

ax.set_yscale("log")
ax.set_xlim(0, 100)
ax.set_xlabel("Cycle", color=INK, fontsize=13, weight="bold")
ax.set_ylabel("Cost", color=INK, fontsize=13, weight="bold")
ax.legend(frameon=False, fontsize=10, labelcolor=INK_2, loc="lower left")
fig.tight_layout()
fig.savefig(out, facecolor="white")
print(out)
