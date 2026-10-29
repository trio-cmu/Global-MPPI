"""Plot a JSON written by pusht_solver_benchmark.py.

    python plot_solver_benchmark.py [bench.json] [out.png] [time limit s, default 120]

Left: solve time against n, converged runs only; a hollow marker on the limit
line is the first n where a solver did not converge in time. Middle: each
solver's c against the best sample cost. Right: the rollout cost at each
solver's x against the best sample cost (PushT has no known minimum).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pusht_solver_benchmark.json")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".png")
limit = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
records = json.loads(src.read_text())

SOLVERS = {  # name: (label, color, marker)
    "newton": ("newton (Python, log-det barrier, Wendland K)", "#2a78d6", "o"),
    "newton-rs": ("newton-rs (Rust, log-det barrier, Laplacian K)", "#1baf7a", "s"),
    "ADMM-sparse": ("ADMM-sparse (CPU, no log-det)", "#e87ba4", "v"),
    "ADMM-cuda": ("ADMM-cuda (GPU, Anderson on)", "#6a4fd3", "h"),
}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

fig, (ax_t, ax_c, ax_x) = plt.subplots(1, 3, figsize=(20, 5.8), dpi=150,
                                       gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
fig.patch.set_facecolor(SURFACE)
for ax in (ax_t, ax_c, ax_x):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")

ns = sorted({r["n"] for r in records})
style = dict(lw=2, ms=7, mec=SURFACE, mew=1.5)
for name, (label, color, marker) in SOLVERS.items():
    rows = sorted((r for r in records if r["solver"] == name), key=lambda r: r["n"])
    if not rows:
        continue
    ok = [r for r in rows if r["converged"]]
    if ok:
        ax_t.plot([r["n"] for r in ok], [r["time"] for r in ok], color=color, marker=marker, label=label, **style)
        for ax, key in ((ax_c, "c"), (ax_x, "x_cost")):
            ax.plot([r["n"] for r in ok], [r[key] for r in ok], color=color, marker=marker, label=label,
                    lw=1.5, ms=6, mec=SURFACE, mew=1.2, alpha=0.9)
    failed = [r for r in rows if not r["converged"]]
    if failed:
        first = failed[0]
        if ok:
            ax_t.plot([ok[-1]["n"], first["n"]], [ok[-1]["time"], limit], color=color, lw=1.5, ls=(0, (3, 3)))
        else:
            ax_t.plot([], [], color=color, marker=marker, label=label + " (no converged run)", **style)
        ax_t.plot(first["n"], limit, marker=marker, ms=8, mfc=SURFACE, mec=color, mew=2)

ax_t.axhline(limit, color=INK_2, lw=0.8, ls=(0, (2, 3)))
ax_t.text(ns[-1], limit * 1.2, f"{limit:.0f} s time limit (hollow = not converged in time)", ha="right",
          va="bottom", fontsize=8.5, color=INK_2)
ax_t.set_yscale("log")
ax_t.set_ylabel("solve time (s, log scale)", color=INK_2)
ax_t.set_title("Solve time, converged runs only", loc="left", fontsize=11, color=INK, weight="bold")
ax_t.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK_2)

best = {r["n"]: r["best_sample_f"] for r in records}
bn = sorted(best)
for ax in (ax_c, ax_x):
    ax.plot(bn, [best[n] for n in bn], color=INK, lw=1.2, ls=(0, (4, 2)), marker="x", ms=7,
            label="best sample, min f(x_i)")
    ax.set_ylabel("cost", color=INK_2)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_2)
ax_c.set_title("Bound c vs. best sample", loc="left", fontsize=11, color=INK, weight="bold")
ax_x.set_title("Rollout cost at x = Σ αᵢ xᵢ vs. best sample", loc="left", fontsize=11, color=INK, weight="bold")
for ax in (ax_t, ax_c, ax_x):
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n:,}" for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("number of samples n", color=INK_2)

r0 = records[0]
fig.suptitle("PushT (12-D: 6 knots × 2 controls), two-level Sobol samples: n/10 clusters × 10 points, RCM order",
             x=0.04, ha="left", fontsize=12.5, color=INK, weight="bold")
fig.text(0.04, 0.905, f"one start state, seed {r0.get('seed', 0)}; Wendland φ₁₂ (newton-rs: Laplacian, same σ), "
         "λ = 1e-5; one timed run per point, each in a fresh process; ADMM eps 1e-6 / 1e-5; "
         "a solver that fails at one n is skipped for larger n", fontsize=9, color=INK_2)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(out, facecolor=fig.get_facecolor())
fig.savefig(out.with_suffix(".pdf"), facecolor=fig.get_facecolor())
print(out)
