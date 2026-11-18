"""Solve time vs. number of samples only (the left panel of pusht_solver_benchmark.png), square, log-log axes.

    python plot_solver_time.py [bench.json ...] [out.png] [time limit s, default 120]

Several JSONs are repeated runs of the same benchmark: each point is then the mean solve time over the runs, with
error bars and a band of ± 1 standard deviation. A solver counts as finished at n only if every run finished; a
hollow marker on the limit line is the first n where a run did not finish (time limit, or out of memory where
labelled). Prints the mean, std and variance of every point.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter  # noqa: E402

import numpy as np  # noqa: E402

args = sys.argv[1:]
srcs = [Path(a) for a in args if a.endswith(".json")] or [Path("pusht_solver_benchmark.json")]
out = next((Path(a) for a in args if a.endswith(".png")), Path("pusht_solver_benchmark_time.png"))
limit = next((float(a) for a in args if a.replace(".", "", 1).isdigit()), 120.0)
runs = [json.loads(src.read_text()) for src in srcs]
FS = next((float(a[5:]) for a in args if a.startswith("--fs=")), 12.0)  # one font size for all text
records = [r for run in runs for r in run]

SOLVERS = {  # name: (label, color, marker), as plot_solver_benchmark.py
    "newton": ("Newton", "#2a78d6", "o"),
    "MOSEK": ("MOSEK", "#f08c00", "D"),
    "ADMM-sparse": ("Ours (CPU)", "#e87ba4", "v"),
    "ADMM-cuda": ("Ours cuKernelSOS (GPU)", "#6a4fd3", "h"),
}
SURFACE, INK, INK_2, GRID, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#c9c8c2"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix"})  # Times for all text, Times-style math (10^2, O(n))
BIG = FS > 14  # large text: bigger figure, thicker marks, legend below the plot, short limit note
K = FS / 12 if BIG else 1.0  # scale of lines and markers
fig, ax = plt.subplots(figsize=(6.4 * FS / 18, 6.4 * FS / 18 * 1.13) if BIG else (6.4, 6.4), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.grid(True, which="major", color="#bdbbb4", lw=0.9)  # only the powers of ten
ax.grid(False, which="minor")
ax.set_axisbelow(True)
for side in ("left", "bottom"):  # dark x and y axis lines
    ax.spines[side].set_color(INK)
    ax.spines[side].set_linewidth(1.5)
for side in ("top", "right"):  # light lines close the square
    ax.spines[side].set_color("#9f9d96")
    ax.spines[side].set_linewidth(1.0)

ns = sorted({r["n"] for r in records})
rel_std = []  # std / mean of every point, for the subtitle
handles = []
first_point = {}  # solver -> (n, mean time) of its first point, to anchor the O(n) / O(n^3) guides
style = dict(lw=2 * K ** 0.6, ms=9 * K ** 0.6, mec=SURFACE, mew=1.5 * K ** 0.5)
for name, (label, color, marker) in SOLVERS.items():
    by_n = {}  # n -> this solver's record in each run
    for run in runs:
        for r in run:
            if r["solver"] == name:
                by_n.setdefault(r["n"], []).append(r)
    ok_n, mean, std, first = [], [], [], None
    for n in sorted(by_n):
        rs = by_n[n]
        if not all(r["converged"] for r in rs):
            first = next(r for r in rs if not r["converged"])
            break
        t = np.array([r["time"] for r in rs])
        ok_n.append(n); mean.append(t.mean()); std.append(t.std(ddof=1) if len(t) > 1 else 0.0)
        rel_std.append(std[-1] / mean[-1])
        print(f"{label:<24} n {n:5d}: {len(t)} runs, mean {t.mean():9.4f} s, std {std[-1]:.4f} s, var {std[-1] ** 2:.2e}")
    mean, std = np.array(mean), np.array(std)
    if ok_n:
        first_point[name] = (ok_n[0], mean[0])
        ax.fill_between(ok_n, np.maximum(mean - std, mean * 0.05), mean + std, color=color, alpha=0.18, lw=0)
        ax.errorbar(ok_n, mean, yerr=[np.minimum(std, mean * 0.95), std], color=color, marker=marker,
                    capsize=3.5, elinewidth=1.2, capthick=1.2, **style)
        handles.append(plt.Line2D([], [], color=color, marker=marker, label=label, **style))
    if first is not None:
        if ok_n:
            ax.plot([ok_n[-1], first["n"]], [mean[-1], limit], color=color, lw=1.5, ls=(0, (3, 3)))
        ax.plot(first["n"], limit, marker=marker, ms=10 * K ** 0.6, mfc=SURFACE, mec=color, mew=2 * K ** 0.5)

# O(n) and O(n^3) guides: gray dotted power laws t = t0 (n / n0)^p, through the first point of the GPU solver (p = 1)
# and of newton (p = 3); on log-log axes they are straight lines of slope p
ylo = min(min(t for _, t in first_point.values()), min(r["time"] for r in records if r["converged"])) / (1.6 if BIG else 8)  # room for the legend inside
yhi = limit * 1.6
xg = np.geomspace(min(ns) / 1.5, max(ns) * 1.5, 60)
for solver, p, tag in (("ADMM-cuda", 1, r"$\mathcal{O}(n)$"), ("newton", 3, r"$\mathcal{O}(n^3)$")):
    if solver not in first_point:
        continue
    n0, t0 = first_point[solver]
    yg = t0 * (xg / n0) ** p
    keep = yg <= limit  # stop at the time-limit line
    ax.plot(xg[keep], yg[keep], color="gray", ls=":", lw=2, zorder=1)
    if p == 1:  # label at the right end, above the line
        ax.text(xg[-1] / 1.08, yg[-1] * 1.15, tag, color="dimgray", fontsize=FS, ha="right", va="bottom")
    else:  # label just right of the line, where it reaches 30 s
        xt = n0 * (30 / t0) ** (1 / 3)
        ax.text(xt * 1.08, 30, tag, color="dimgray", fontsize=FS, ha="left", va="center")

ax.axhline(limit, color=INK_2, lw=0.8, ls=(0, (2, 3)))
ax.text(ns[-1] * 1.4, limit * 1.15, f"{limit:.0f} s limit" + ("" if BIG else " (hollow = did not finish)"),
        ha="right", va="bottom",
        fontsize=FS, color=INK_2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(ylo, yhi)
ax.set_xlim(min(ns) / 1.5, max(ns) * 1.5)
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_locator(LogLocator(base=10))
    axis.set_major_formatter(LogFormatterMathtext())
    axis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    axis.set_minor_formatter(NullFormatter())
ax.tick_params(which="major", length=6 * K ** 0.6, width=1.3 * K ** 0.5, color=INK, labelcolor=INK, labelsize=FS)
ax.tick_params(which="minor", length=3.5, width=1.0, color=INK)
ax.set_xlabel("Number of samples", color=INK, fontsize=FS + 5, weight="bold")
ax.set_ylabel("Time [s]", color=INK, fontsize=FS + 5, weight="bold")
# no title (the paper caption names the figure)
if BIG:
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.115), ncol=2, frameon=False, fontsize=FS,
              labelcolor=INK, columnspacing=1.0, handlelength=1.8, handletextpad=0.4)
else:
    ax.legend(handles=handles, loc="lower right", ncol=2, frameon=True, framealpha=0.95, edgecolor="#bdbbb4",
              facecolor="white", fontsize=FS, labelcolor=INK, columnspacing=1.2, handlelength=2.2)
ax.set_box_aspect(1)  # square plot area
fig.tight_layout()
fig.savefig(out, facecolor=fig.get_facecolor())
print(out)
