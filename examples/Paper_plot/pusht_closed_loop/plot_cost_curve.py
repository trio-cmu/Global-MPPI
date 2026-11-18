"""Left panel of pusht_closed_loop_6seeds_main.png on its own: closed-loop cost per cycle,
geometric mean ± 1 std (of log10 cost) over seeds.

    python plot_cost_curve.py [logs dir] [out.png]      (defaults: logs/, pusht_closed_loop_cost_curve.png here)
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

here = Path(__file__).resolve().parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "logs"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "pusht_closed_loop_cost_curve.png"

METHODS = {  # log prefix: (label, color), same as the main figure
    "ps2000": ("PS", "#52514e"),
    "mppi2000": ("MPPI", "#1baf7a"),
    "dial2000": ("DIAL", "#eb6834"),
    "globalmppi_cacs5new": ("cuKernelSOS", "#d43d51"),
}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def load(prefix):
    """(seeds, cycles) array of the best cost per control cycle."""
    runs = []
    for path in sorted(src.glob(f"{prefix}_seed*.log")):
        text = path.read_text(errors="ignore")
        cost = {int(float(c)): float(v) for c, v in re.findall(r"Cycle ([\d.]+): Best cost: \[?([-\d.e+]+)", text)}
        if cost:
            runs.append(np.array([cost[c] for c in sorted(cost)]))
    n = min(len(r) for r in runs)
    return np.stack([r[:n] for r in runs])


fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.grid(True, color=GRID, lw=0.8)
ax.tick_params(colors=INK_2, length=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#c9c8c2")

for prefix, (label, color) in METHODS.items():
    C = load(prefix)
    logC = np.log10(np.maximum(C, 1e-6))
    gm, gs = logC.mean(0), logC.std(0, ddof=1)
    it = np.arange(C.shape[1])
    ax.plot(it, 10 ** gm, color=color, lw=2.2, label=label)
    ax.fill_between(it, 10 ** (gm - gs), 10 ** (gm + gs), color=color, alpha=0.15)
    print(f"{label:16s} {len(C)} seeds, final geometric mean {10 ** gm[-1]:.2e}")

ax.set_yscale("log")
ax.set_xlabel("Iteration", color=INK, fontsize=13, weight="bold")
ax.set_ylabel("Cost", color=INK, fontsize=13, weight="bold")
ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
fig.tight_layout()
fig.savefig(out, facecolor=fig.get_facecolor())
print(out)
