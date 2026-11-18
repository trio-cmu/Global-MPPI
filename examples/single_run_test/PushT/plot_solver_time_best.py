"""Solve time vs. number of samples over repeated runs, in the style of a "best + range" timing plot:
the marker is the best (fastest) run at each n, the vertical line spans fastest to slowest run.

    python plot_solver_time_best.py run1.json run2.json run3.json [out.png] [time limit s, default 120]

A solver counts as finished at n only if every run finished; at the first n where a run did not, a hollow marker
sits on the time-limit line (labelled when the cause was memory). Gray dotted lines are O(n) and O(n^3) guides.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

args = sys.argv[1:]
srcs = [Path(a) for a in args if a.endswith(".json")]
out = next((Path(a) for a in args if a.endswith(".png")), Path("pusht_solver_benchmark_time_best.png"))
limit = next((float(a) for a in args if a.replace(".", "", 1).isdigit()), 120.0)
runs = [json.loads(s.read_text()) for s in srcs]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "font.size": 12,
})
SOLVERS = {  # name: (label, color)
    "newton": ("newton", "tab:orange"),
    "MOSEK": ("MOSEK", "tab:purple"),
    "ADMM-sparse": ("Ours (CPU)", "tab:blue"),
    "ADMM-cuda": ("Ours cuKernelSOS (GPU)", "tab:red"),
}

fig, ax = plt.subplots(figsize=(6.0, 6.8), dpi=200)
ns_all = sorted({r["n"] for run in runs for r in run})
for name, (label, color) in SOLVERS.items():
    by_n = {}
    for run in runs:
        for r in run:
            if r["solver"] == name:
                by_n.setdefault(r["n"], []).append(r)
    ns, best, lo, hi, first = [], [], [], [], None
    for n in sorted(by_n):
        rs = by_n[n]
        if not all(r["converged"] for r in rs):
            first = next(r for r in rs if not r["converged"])
            break
        t = np.array([r["time"] for r in rs])
        ns.append(n); best.append(t.min()); lo.append(t.min()); hi.append(t.max())
        print(f"{label:<24} n {n:5d}: best {t.min():9.4f} s, worst {t.max():9.4f} s ({len(t)} runs)")
    if ns:
        ax.vlines(ns, lo, hi, color=color, lw=2.0, zorder=2)
        ax.plot(ns, best, color=color, marker="o", ms=8, lw=1.8, label=label, zorder=3)
    if first is not None:
        if ns:
            ax.plot([ns[-1], first["n"]], [best[-1], limit], color=color, lw=1.5, ls=(0, (3, 2)), zorder=3)
        ax.plot(first["n"], limit, marker="o", ms=8, mfc="white", mec=color, mew=2, zorder=4)
        if "err_space" in first["status"] or "memory" in first["status"].lower():
            ax.annotate("out of memory", (first["n"], limit), xytext=(7, -7), textcoords="offset points",
                        ha="left", va="top", fontsize=10, color=color)

# O(n) and O(n^3) guides, through the GPU solver's and newton's first points
n0, n1 = ns_all[0] / 1.4, ns_all[-1] * 1.4
x = np.geomspace(n0, n1, 50)
for k, (y0, tag) in enumerate([(0.03, r"$\mathcal{O}(n)$"), (0.06, r"$\mathcal{O}(n^3)$")]):
    p = 1 if k == 0 else 3
    y = y0 * (x / ns_all[0]) ** p
    ax.plot(x, y, color="gray", ls=":", lw=2, zorder=1)
    if p == 1:  # label at the right end, above the line
        ax.text(x[-1] / 1.05, y[-1] * 1.1, tag, color="gray", fontsize=11, ha="right", va="bottom")
    else:  # label just right of the line, where it reaches 30 s
        xt = ns_all[0] * (30 / y0) ** (1 / 3)
        ax.text(xt * 1.08, 30, tag, color="gray", fontsize=11, ha="left", va="center")

ax.axhline(limit, color="gray", lw=0.8, ls="--", zorder=1)
ax.text(n1 / 1.05, limit * 1.06, f"{limit:.0f} s limit", ha="right", va="bottom", fontsize=10, color="dimgray")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(n0, n1)
ax.set_ylim(0.015, limit * 2.2)
ax.grid(True, which="major", color="0.75", lw=0.8)
ax.set_axisbelow(True)
ax.set_xlabel("number of samples")
ax.set_ylabel("time [s]")
ax.set_title("PushT", fontsize=17, pad=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=True, fancybox=True, framealpha=0.95,
          edgecolor="0.8", fontsize=11)
ax.set_box_aspect(1)
fig.tight_layout()
fig.savefig(out, facecolor="white")
print(out)
