"""Cost curve and planning time of closed-loop PushT runs, parsed from pusht.py logs.

    python plot_closed_loop.py name=run.log [name=run.log ...] [--out closed_loop.png]

Reads the "Cycle N: Best cost: ..." and "Cycle N: plan time: ...s" lines.
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

argv = sys.argv[1:]
out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else Path("closed_loop.png")
runs = [a.split("=", 1) for a in argv if "=" in a]
COLORS = ["#6a4fd3", "#1baf7a", "#eb6834", "#2a78d6"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

fig, (ax_c, ax_t) = plt.subplots(1, 2, figsize=(15, 5.6), dpi=150)
fig.patch.set_facecolor(SURFACE)
for ax in (ax_c, ax_t):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")

summary = []
for (name, path), color in zip(runs, COLORS):
    text = Path(path).read_text()
    cost = {int(float(c)): float(v) for c, v in re.findall(r"Cycle ([\d.]+): Best cost: \[?([-\d.e+]+)", text)}
    plan = {int(float(c)): float(v) for c, v in re.findall(r"Cycle ([\d.]+): plan time: ([\d.]+)s", text)}
    cyc = sorted(cost)
    ax_c.plot(cyc, [cost[c] for c in cyc], color=color, lw=2, label=name)
    pc = sorted(plan)
    t = np.array([plan[c] for c in pc])
    ax_t.plot(pc, t, color=color, lw=1.5, marker="o", ms=2.5, label=f"{name} (median {np.median(t[1:]):.2f} s)")
    summary.append((name, cost[cyc[-1]], np.mean([cost[c] for c in cyc]), np.median(t[1:]), t[1:].mean(), t[1:].max(), t[0]))

ax_c.set_yscale("log")
ax_c.set_xlabel("control cycle", color=INK_2)
ax_c.set_ylabel("cost of the planned trajectory (log scale)", color=INK_2)
ax_c.set_title("Cost curve", loc="left", fontsize=11, color=INK, weight="bold")
ax_c.legend(frameon=False, fontsize=9, labelcolor=INK_2)
ax_t.set_yscale("log")
ax_t.set_xlabel("control cycle", color=INK_2)
ax_t.set_ylabel("planning time per cycle (s, log scale)", color=INK_2)
ax_t.set_title("Planning time per cycle (cycle 0 includes JIT compilation)", loc="left", fontsize=11, color=INK,
               weight="bold")
ax_t.legend(frameon=False, fontsize=9, labelcolor=INK_2)
fig.suptitle("PushT closed loop: GlobalMPPI (KSOS ADMM-cuda + MPPI) vs DIAL", x=0.04, ha="left", fontsize=12.5,
             color=INK, weight="bold")
fig.text(0.04, 0.905, "start: block (0.1, 0.1, 1.3), pusher (0, 0); 100 cycles at 50 Hz; seed 0. GlobalMPPI: 5 restarts × "
         "1000 two-level Sobol samples, ADMM-cuda, MPPI 1000 × 5 (σ 0.2). DIAL: pusht.py, 256 × 5",
         fontsize=9, color=INK_2)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(out, facecolor=fig.get_facecolor())
fig.savefig(out.with_suffix(".pdf"), facecolor=fig.get_facecolor())
for s in summary:
    print(f"{s[0]:<28} final cost {s[1]:.4f}  mean cost {s[2]:.3f}  plan time median {s[3]:.3f}s mean {s[4]:.3f}s "
          f"max {s[5]:.3f}s  (cycle 0 {s[6]:.1f}s)")
print(out)
