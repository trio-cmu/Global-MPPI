"""Plot the per-cycle GlobalMPPI planning-time breakdown.

Usage: python examples/plot_timing.py <globalmppi_timing_seedN.txt> [out.png]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

PHASES = ["sampling_rollout", "autocalib", "ksos_solve", "mppi"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else path.replace(".txt", ".png")
data = np.atleast_2d(np.loadtxt(path))
cycles, phases, total = data[:, 0], data[:, 2:6], data[:, 6]

fig, (ax0, ax1) = plt.subplots(
    1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [3, 1]}
)
bottom = np.zeros_like(total)
for i, (name, c) in enumerate(zip(PHASES, COLORS)):
    ax0.bar(cycles, phases[:, i], bottom=bottom, color=c, label=name, width=0.9)
    bottom += phases[:, i]
ax0.set_xlabel("cycle")
ax0.set_ylabel("time per cycle (s)")
ax0.set_title("GlobalMPPI planning time per cycle (summed over restarts)")
ax0.legend(loc="upper right")
ax0.margins(x=0.01)

# Exclude cycle 0 (JIT / warm-up) from the summary statistics
steady = phases[1:] if len(phases) > 1 else phases
means = steady.mean(axis=0)
ax1.barh(PHASES, means, color=COLORS, xerr=steady.std(axis=0))
for i, m in enumerate(means):
    ax1.text(m, i, f" {m:.2f}s ({100 * m / means.sum():.0f}%)", va="bottom")
ax1.set_xlabel("mean time per cycle (s), cycle 0 excluded")
ax1.invert_yaxis()

fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"saved {out}")
