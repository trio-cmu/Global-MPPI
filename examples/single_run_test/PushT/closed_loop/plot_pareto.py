"""Cost-time trade-off (Pareto front) of the PushT closed-loop methods: x = mean planning time per cycle,
y = mean final cost (cost at cycle 100), both log scale, one marker per method (mean over seeds) and faint dots
for the single seeds. The front joins the methods no other method beats on both time and cost; times within
1% count as equal (PS / MPPI / DIAL all run at 0.36 s).

    python plot_pareto.py [log_dir] [out.png]      (defaults: six_seeds_main, pusht_pareto.png)
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).resolve().parent
log_dir = here / (sys.argv[1] if len(sys.argv) > 1 else "six_seeds_main")
out_path = here / (sys.argv[2] if len(sys.argv) > 2 else "pusht_pareto.png")

# log prefix -> (label, color): ours = the accent (red); every method its own hue
METHODS = {
    "ps2000": ("PS", "#7b52ab"),
    "mppi2000": ("MPPI", "#1b9e77"),
    "dial2000": ("DIAL", "#e69f00"),
    "globalmppi_cacs5new": ("CACS (ours)", "#c8102e"),
}
GCS_TIMES = [119.84, 119.94, 119.91, 119.59, 119.57, 120.35]  # planning time per cycle (s), one per run
EXTERNAL = {"gcs": ("GCS", "#2f6db5", [1e-5] * len(GCS_TIMES), GCS_TIMES)}  # per run: final cost, planning time (s)
SURFACE, INK, INK_2, GRID, SHADE, FRONT = "#ffffff", "#111111", "#4a4a48", "#e4e4e0", "#e8ebf1", "#33333a"
TEXT_COLOR = {"dial2000": "#b37a00"}  # labels take the method colour; amber darkened so the text stays readable on white
plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5})


def load(prefix):
    """Per seed: (final cost, mean planning time over cycles 1..end)."""
    out = []
    for path in sorted(log_dir.glob(f"{prefix}_seed*.log")):
        text = path.read_text(errors="ignore")
        cost = {int(float(c)): float(v) for c, v in re.findall(r"Cycle ([\d.]+): Best cost: \[?([-\d.e+]+)", text)}
        plan = {int(float(c)): float(v) for c, v in re.findall(r"Cycle ([\d.]+): plan time: ([\d.]+)s", text)}
        out.append((cost[max(cost)], np.mean([v for c, v in plan.items() if c > 0])))
    if not out:
        raise FileNotFoundError(f"no {prefix}_seed*.log in {log_dir}")
    return np.array(out)


points = []  # (key, label, color, mean time, mean cost, per-seed array or None)
for prefix, (label, color) in METHODS.items():
    runs = load(prefix)
    points.append((prefix, label, color, runs[:, 1].mean(), runs[:, 0].mean(), runs))
for key, (label, color, costs, times) in EXTERNAL.items():
    runs = np.column_stack([costs, times])
    points.append((key, label, color, runs[:, 1].mean(), runs[:, 0].mean(), runs))


def dominated(p, q, tol=0.01):
    """q beats p: no slower (within tol) and no costlier, better in at least one."""
    no_slower = q[3] <= p[3] * (1 + tol)
    return no_slower and q[4] <= p[4] and (q[4] < p[4] or q[3] < p[3] * (1 - tol))


front = sorted([p for p in points if not any(dominated(p, q) for q in points if q is not p)], key=lambda p: p[3])

fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xscale("log")
ax.set_yscale("log")

# dominated region: everything slower and costlier than some front point (staircase above/right of the front)
xs, ys = [p[3] for p in front], [p[4] for p in front]
x_hi, y_hi = 400, 5
stair_x, stair_y = [], []
for i, (x, y) in enumerate(zip(xs, ys)):
    stair_x += [x, xs[i + 1] if i + 1 < len(xs) else x_hi]
    stair_y += [y, y]
ax.fill_between(stair_x, stair_y, y_hi, color=SHADE, lw=0, zorder=0)
ax.plot(xs, ys, color=FRONT, lw=2.6, zorder=2, solid_capstyle="round")

for key, label, color, t, cost, runs in points:
    if runs is not None:
        ax.scatter(runs[:, 1], runs[:, 0], color=color, s=34, alpha=0.35, lw=0, zorder=3)
    on_front = any(p[0] == key for p in front)
    ax.scatter([t], [cost], color=color, s=220 if on_front else 150, marker="o" if on_front else "X",
               edgecolor=SURFACE, lw=2, zorder=4)

# direct labels, nudged clear of each other and of the front line (PS / MPPI / DIAL share x)
offsets = {"ps2000": (14, 9), "mppi2000": (14, -9), "dial2000": (14, 16), "globalmppi_cacs5new": (14, 16), "gcs": (0, -30)}
for key, label, color, t, cost, runs in points:
    dx, dy = offsets.get(key, (10, 8))
    ours = key == "globalmppi_cacs5new"
    txt = f"{label}  {t:.2f} s, {cost:.1e}" if runs is not None and t < 1 else f"{label}\n{t:.2f} s, {cost:.1e}"
    ax.annotate(txt, (t, cost), xytext=(dx, dy), textcoords="offset points", fontsize=13.5 if ours else 12.5,
                color=TEXT_COLOR.get(key, color), weight="bold" if ours else "normal",
                ha="center" if dx == 0 else ("right" if dx < 0 else "left"), va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85), zorder=5)

ax.set_xlim(0.2, x_hi)
ax.set_ylim(1e-6, y_hi)
ax.set_xlabel("planning time per cycle (s)", color=INK)
ax.set_ylabel("final cost at cycle 100", color=INK)
by = {p[0]: p for p in points}
ours, dial, gcs = by["globalmppi_cacs5new"], by["dial2000"], by["gcs"]
ax.set_title("Pareto plot for PushT", loc="left", fontsize=16, color=INK, weight="bold", pad=30)
ax.text(0, 1.035, f"CACS is on the Pareto front: {dial[4] / ours[4]:.0f}× lower cost than DIAL, "
        f"{gcs[3] / ours[3]:.0f}× faster than GCS", transform=ax.transAxes, fontsize=12.5, color=INK_2, va="bottom")
ax.grid(True, which="major", color=GRID, lw=0.9)
ax.tick_params(colors=INK_2)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#b9b9b4")
ax.legend(handles=[
    plt.Line2D([], [], color=FRONT, lw=2.6, label="Pareto front"),
    plt.Line2D([], [], color=INK_2, marker="o", ls="", ms=11, label="on the front (mean)"),
    plt.Line2D([], [], color=INK_2, marker="X", ls="", ms=10, label="dominated (mean)"),
    plt.Line2D([], [], color=INK_2, marker="o", ls="", ms=6, alpha=0.4, label="single seeds"),
    plt.Rectangle((0, 0), 1, 1, fc=SHADE, ec="#c4c8d0", lw=0.8, label="dominated region"),
], loc="upper right", frameon=True, facecolor=SURFACE, edgecolor="#d4d4cf", framealpha=1, fancybox=False, fontsize=12, labelcolor=INK_2)
fig.text(0.01, 0.005, "Mean over 6 runs per method; both axes log scale. Times within 1% count as equal.",
         fontsize=10.5, color=INK_2)
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
print("front:", " -> ".join(p[1] for p in front))
print(f"Saved: {out_path}")
