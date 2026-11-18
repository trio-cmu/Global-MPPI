"""Mean and standard deviation over seeds of a JSON written by pusht_method_benchmark.py.

    python plot_method_mean_std.py [bench.json] [out.png] [subtitle]

Left: plan cost, mean with a ±1 std error bar, the seeds as dots, the variance
printed above each bar. Right: planning time, the same way.
"""

import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pusht_method_benchmark_equal_budget.json")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + "_mean_std.png")
note = sys.argv[3] if len(sys.argv) > 3 else ""
records = json.loads(src.read_text())

FIXED = {"PS": "#52514e", "MPPI": "#eb6834", "DIAL": "#1baf7a"}
EXTRA = ["#f2b134", "#e87ba4", "#6a4fd3", "#2a78d6"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def short(m):
    return (m.replace(" restarts", "").replace(" + LSE ", "\n+ LSE ").replace(" + MPPI x5 (noise ", "\n+ MPPI, σ ")
            .replace(")", ""))


names = list(dict.fromkeys(r["method"] for r in records))
PALETTE = ["#52514e", "#eb6834", "#1baf7a", "#2a78d6", "#f2b134", "#e87ba4", "#6a4fd3", "#8c6d31"]
colors = {m: FIXED.get(m) or EXTRA[[n for n in names if n not in FIXED].index(m) % 4] for m in names}
if len(names) > 4:  # one color per method
    colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(names)}

width = max(15, 2.4 * len(names))
fig, (ax_c, ax_t) = plt.subplots(1, 2, figsize=(width, 6.2), dpi=150, gridspec_kw=dict(width_ratios=[1, 1]))
fig.patch.set_facecolor(SURFACE)
for ax in (ax_c, ax_t):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")

for ax, key, scale, unit in ((ax_c, "cost", 1.0, ""), (ax_t, "time", 1e3, " ms")):
    for i, m in enumerate(names):
        v = np.array([r[key] for r in records if r["method"] == m]) * scale
        mean, std = v.mean(), v.std(ddof=1)
        ax.bar(i, mean, width=0.6, color=colors[m], alpha=0.25)
        lower_err = min(std, mean * 0.95) if key == "time" else std  # stays positive on a log axis
        ax.errorbar(i, mean, yerr=[[lower_err], [std]], color=colors[m], capsize=8, lw=2, capthick=2, zorder=3)
        ax.scatter(i + np.linspace(-0.15, 0.15, len(v)), v, color=colors[m], s=22, edgecolor=SURFACE, lw=1,
                   zorder=4)
        top = max(mean + std, v.max())
        fmt = ".2f" if key == "cost" else ".0f"
        ax.text(i, top, f"{mean:{fmt}}{unit}\n± {std:{fmt}}" + (f"\nvar {std ** 2:.2f}" if key == "cost" else ""),
                ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([short(m) for m in names], color=INK_2, fontsize=9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.set_xlim(-0.6, len(names) - 0.4)
ax_c.set_ylabel("rollout cost of the plan", color=INK_2)
ax_c.set_title("Plan cost: mean ± std over seeds (dots = seeds)", loc="left", fontsize=11, color=INK, weight="bold")
ax_t.set_ylabel("time of one planning step (ms)", color=INK_2)
times = [r["time"] for r in records]
if max(times) / min(times) > 30:  # very different scales: log axis
    ax_t.set_yscale("log")
    ax_t.set_ylim(min(times) * 1e3 / 2, max(times) * 1e3 * 8)
    ax_t.set_ylabel("time of one planning step (ms, log scale)", color=INK_2)
ax_t.set_title("Planning time: mean ± std over seeds", loc="left", fontsize=11, color=INK, weight="bold")

seeds = sorted({r["seed"] for r in records})
fig.suptitle("PushT, one planning step from one start state: mean and variance over seeds", x=0.04, ha="left",
             fontsize=12.5, color=INK, weight="bold")
fig.text(0.04, 0.925, textwrap.fill(f"{note + '; ' if note else ''}seeds {seeds[0]}–{seeds[-1]} "
                                    f"(n = {len(seeds)}); std uses ddof = 1", 190),
         fontsize=9, color=INK_2, va="top")
fig.tight_layout(rect=(0, 0, 1, 0.87))
fig.savefig(out, facecolor=fig.get_facecolor())
fig.savefig(out.with_suffix(".pdf"), facecolor=fig.get_facecolor())
print(out)
