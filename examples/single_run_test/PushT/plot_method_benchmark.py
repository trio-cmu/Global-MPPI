"""Plot a JSON written by pusht_method_benchmark.py.

    python plot_method_benchmark.py [bench.json] [out.png]

Left: the plan's rollout cost for every method, one dot per seed, a bar at
the mean (KSOS also marks each seed's best sample). Right: time of the
planning step (mean over seeds); KSOS split into its parts.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pusht_method_benchmark.json")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".png")
records = json.loads(src.read_text())

METHODS = {  # name: (label, color)
    "PS": ("PS", "#52514e"),
    "MPPI": ("MPPI", "#eb6834"),
    "DIAL": ("DIAL", "#1baf7a"),
    "KSOS": ("KSOS\n(ADMM-cuda)", "#6a4fd3"),
    "KSOS, guarded": ("KSOS, guarded\nmin(x, best)", "#2a78d6"),
}
PARTS = {"sampling": "#c9c8c2", "rollout": "#e87ba4", "rcm+kernel": "#f2b134", "admm": "#6a4fd3",
         "guard rollout": "#2a78d6", "x rollout": "#2a78d6", "mppi": "#eb6834"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

fig, (ax_c, ax_t) = plt.subplots(1, 2, figsize=(15, 5.8), dpi=150, gridspec_kw=dict(width_ratios=[1.3, 1]))
fig.patch.set_facecolor(SURFACE)
for ax in (ax_c, ax_t):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")

EXTRA = ["#6a4fd3", "#2a78d6", "#f2b134", "#e87ba4"]
for m in dict.fromkeys(r["method"] for r in records):  # methods without a fixed style
    if m not in METHODS:
        short = m.replace(" restarts", "").replace(" + MPPI x5 (noise ", "\n+ MPPI, σ ").replace(")", "")
        METHODS[m] = (short.replace(" + ", "\n+ ") if "σ" not in short else short, EXTRA[sum(k not in ("PS", "MPPI", "DIAL") for k in METHODS) % 4])
names = [m for m in METHODS if any(r["method"] == m for r in records)]
seeds = sorted({r["seed"] for r in records})
jitter = np.linspace(-0.18, 0.18, len(seeds))
for i, name in enumerate(names):
    label, color = METHODS[name]
    rows = sorted((r for r in records if r["method"] == name), key=lambda r: r["seed"])
    cost = np.array([r["cost"] for r in rows])
    ax_c.bar(i, cost.mean(), width=0.6, color=color, alpha=0.18)
    ax_c.hlines(cost.mean(), i - 0.3, i + 0.3, color=color, lw=2)
    ax_c.scatter(i + jitter[: len(cost)], cost, color=color, s=36, edgecolor=SURFACE, lw=1.2, zorder=3)
    ax_c.text(i, cost.mean(), f"{cost.mean():.2f}", ha="center", va="bottom", fontsize=9, color=INK,
              weight="bold")
    if name.startswith("KSOS") and "MPPI" not in name and all("best_sample" in r for r in rows):
        best = np.array([r["best_sample"] for r in rows])
        ax_c.scatter(i + jitter[: len(best)], best, marker="x", color=INK, s=30, lw=1.2, zorder=4,
                     label="KSOS: best of its n samples (per seed)")
ax_c.set_xticks(range(len(names)))
ax_c.set_xticklabels([METHODS[n][0] for n in names], color=INK_2)
ax_c.set_ylabel("rollout cost of the plan", color=INK_2)
ax_c.set_title("Plan cost (dots = seeds, bar = mean; lower is better)", loc="left", fontsize=11, color=INK,
               weight="bold")
ax_c.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_2)

shown = set()
for i, name in enumerate(names):
    label, color = METHODS[name]
    rows = [r for r in records if r["method"] == name]
    if "time_parts" in rows[0]:
        parts = {k: np.median([r["time_parts"][k] for r in rows]) * 1e3 for k in rows[0]["time_parts"]}
        parts = {k: v for k, v in parts.items() if v > 0}
        if name == "KSOS":
            parts.pop("guard rollout", None)
        bottom = 0.0
        for k, v in parts.items():
            ax_t.bar(i, v, bottom=bottom, width=0.6, color=PARTS[k], label=k if k not in shown else None)
            shown.add(k)
            bottom += v
        total = bottom
    else:
        total = np.median([r["time"] for r in rows]) * 1e3
        ax_t.bar(i, total, width=0.6, color=color, alpha=0.85)
    rollouts = rows[0]["rollouts"]
    worst = max(r["time"] for r in rows) * 1e3
    ax_t.text(i, total, f"{total:.0f} ms\n{rollouts:,} rollouts" + (f"\n(max {worst / 1e3:.1f} s)" if worst > 2 * total else ""), ha="center", va="bottom", fontsize=8.5,
              color=INK_2)
ax_t.set_xticks(range(len(names)))
ax_t.set_xticklabels([METHODS[n][0] for n in names], color=INK_2)
ax_t.set_ylabel("time of one planning step (ms)", color=INK_2)
ax_t.set_yscale("log")
ax_t.set_ylim(50, ax_t.get_ylim()[1] * 3)
ax_t.set_title("Planning time (median over seeds, log scale, JIT excluded)", loc="left", fontsize=11, color=INK,
               weight="bold")
ax_t.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK_2, title="KSOS parts",
            title_fontsize=8.5)

fig.suptitle("PushT, one planning step from one start state (cold start): KSOS vs PS / MPPI / DIAL",
             x=0.04, ha="left", fontsize=12.5, color=INK, weight="bold")
note = sys.argv[3] if len(sys.argv) > 3 else (
    "PS / MPPI / DIAL: examples/pusht.py hyperparameters (256 samples × 5 iterations); KSOS: two-level Sobol "
    "(n/10 clusters × 10), Wendland, λ = 1e-5, ADMM-cuda")
import textwrap  # noqa: E402
fig.text(0.04, 0.925, textwrap.fill(f"{note}; seeds {seeds[0]}–{seeds[-1]}", 190), fontsize=9, color=INK_2, va="top")
fig.tight_layout(rect=(0, 0, 1, 0.87))
fig.savefig(out, facecolor=fig.get_facecolor())
fig.savefig(out.with_suffix(".pdf"), facecolor=fig.get_facecolor())
print(out)
