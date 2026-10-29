"""Closed-loop PushT over several seeds: cost curves, final cost and planning time, mean and variance.

    python plot_closed_loop_seeds.py [seeds dir] [out.png]

Reads seeds/<method>_seed<k>.log written by pusht.py ("Cycle N: Best cost: ..." and
"Cycle N: plan time: ...s"). Cycle 0 (JIT compilation) is left out of the time statistics.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("seeds")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("pusht_closed_loop_seeds.png")
LABELS = {"globalmppi": "GlobalMPPI (KSOS + MPPI σ 0.2)", "dial": "DIAL (256 × 5)",
          "dial2000": "DIAL (2000 × 5)", "ksosonly": "KSOS only (no MPPI)"}
LABELS["globalmppi_s0.1"] = "GlobalMPPI (KSOS + MPPI σ 0.1)"
LABELS["globalmppi_lse"] = "GlobalMPPI (KSOS + LSE + MPPI σ 0.1)"
LABELS["ps2000"] = "PS (2000 × 5)"
LABELS["mppi2000"] = "MPPI (2000 × 5)"
LABELS["globalmppi_n500"] = "GlobalMPPI 500 + 500 (KSOS + MPPI σ 0.1)"
LABELS["globalmppi_s0.05"] = "GlobalMPPI (KSOS + MPPI σ 0.05)"
COLORS = {"globalmppi": "#6a4fd3", "dial": "#1baf7a", "dial2000": "#eb6834", "ksosonly": "#2a78d6", "globalmppi_s0.1": "#e87ba4",
          "globalmppi_s0.05": "#f2b134", "globalmppi_n500": "#8c6d31", "ps2000": "#52514e",
          "mppi2000": "#1baf7a", "globalmppi_lse": "#6a4fd3"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

EXTRA_COLORS = ["#e87ba4", "#f2b134", "#8c6d31", "#52514e"]
runs = defaultdict(dict)  # method -> seed -> (cost (cycles,), plan time (cycles,))
for path in sorted(src.glob("*_seed*.log")):
    method, seed = re.match(r"(.+)_seed(\d+)\.log", path.name).groups()
    text = path.read_text()
    cost = {int(float(c)): float(v) for c, v in re.findall(r"Cycle ([\d.]+): Best cost: \[?([-\d.e+]+)", text)}
    plan = {int(float(c)): float(v) for c, v in re.findall(r"Cycle ([\d.]+): plan time: ([\d.]+)s", text)}
    if not cost:
        continue
    if method not in LABELS:  # e.g. globalmppi_s0.05: KSOS + MPPI at another noise level
        noise = re.match(r"globalmppi_s([\d.]+)", method)
        LABELS[method] = f"GlobalMPPI (KSOS + MPPI σ {noise.group(1)})" if noise else method
        COLORS[method] = EXTRA_COLORS[(len(COLORS) - 4) % len(EXTRA_COLORS)]
    runs[method][int(seed)] = (np.array([cost[c] for c in sorted(cost)]), np.array([plan[c] for c in sorted(plan)]))

fig, (ax_c, ax_f, ax_t) = plt.subplots(1, 3, figsize=(19, 5.8), dpi=150, gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
fig.patch.set_facecolor(SURFACE)
for ax in (ax_c, ax_f, ax_t):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")

methods = [m for m in LABELS if m in runs]
rows = []
for i, m in enumerate(methods):
    seeds = sorted(runs[m])
    n = min(len(runs[m][s][0]) for s in seeds)
    C = np.stack([runs[m][s][0][:n] for s in seeds])  # (seeds, cycles)
    logC = np.log10(np.maximum(C, 1e-6))
    for c in C:
        ax_c.plot(np.arange(n), c, color=COLORS[m], lw=0.6, alpha=0.3)
    gm, gs = logC.mean(0), (logC.std(0, ddof=1) if len(seeds) > 1 else np.zeros(n))
    ax_c.plot(np.arange(n), 10 ** gm, color=COLORS[m], lw=2.2, label=f"{LABELS[m]} ({len(seeds)} seeds)")
    ax_c.fill_between(np.arange(n), 10 ** (gm - gs), 10 ** (gm + gs), color=COLORS[m], alpha=0.15)

    final = C[:, -1]
    mean_cost = C.mean(1)  # each run's mean cost over the cycles
    T = np.concatenate([runs[m][s][1][1:] for s in seeds])  # all cycles but 0
    per_run_t = np.array([runs[m][s][1][1:].mean() for s in seeds])
    rows.append((m, seeds, final, mean_cost, T, per_run_t))
    for ax, v, label in ((ax_f, final, "final"), (ax_t, per_run_t, "time")):
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        ax.bar(i, v.mean(), width=0.6, color=COLORS[m], alpha=0.25)
        if len(v) > 1:
            ax.errorbar(i, v.mean(), yerr=sd, color=COLORS[m], capsize=8, lw=2, capthick=2, zorder=3)
        ax.scatter(i + np.linspace(-0.15, 0.15, len(v)), v, color=COLORS[m], s=24, edgecolor=SURFACE, lw=1, zorder=4)
        fmt = ".4f" if label == "final" else ".2f"
        ax.text(i, max(v.max(), v.mean() + sd), f"{v.mean():{fmt}}" + (" s" if label == "time" else "")
                + (f"\n± {sd:{fmt}}" + (f"\nvar {sd ** 2:.1e}" if label == "final" else " s") if len(v) > 1 else ""),
                ha="center", va="bottom", fontsize=8.5, color=INK)

ax_c.set_yscale("log")
ax_c.set_xlabel("control cycle", color=INK_2)
ax_c.set_ylabel("cost of the planned trajectory (log scale)", color=INK_2)
ax_c.set_title("Cost curve" + (": geometric mean ± 1 std (log), thin = seeds" if max(len(r) for r in runs.values()) > 1 else ""), loc="left", fontsize=11, color=INK,
               weight="bold")
ax_c.legend(frameon=False, fontsize=9, labelcolor=INK_2)
for ax in (ax_f, ax_t):
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([LABELS[m].replace(" (", "\n(").replace(" + ", "\n+ ") for m in methods], color=INK_2,
                       fontsize=8.5)
    ax.set_xlim(-0.6, len(methods) - 0.4)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
finals = np.concatenate([r[2] for r in rows])
if finals.max() / finals.min() > 100:  # very different scales: log axis
    ax_f.set_yscale("log")
    ax_f.set_ylim(finals.min() / 3, finals.max() * 20)
ax_f.set_ylabel("cost at the last cycle" + (" (log scale)" if ax_f.get_yscale() == "log" else ""), color=INK_2)
ax_f.set_title("Final cost" + (": mean ± std over seeds" if max(len(r) for r in runs.values()) > 1 else ""), loc="left", fontsize=11, color=INK, weight="bold")
ax_t.set_ylabel("mean planning time per cycle (s)", color=INK_2)
ax_t.set_title("Planning time per cycle" + (": mean ± std over seeds" if max(len(r) for r in runs.values()) > 1 else ""), loc="left", fontsize=11, color=INK, weight="bold")

fig.suptitle("PushT closed loop: GlobalMPPI (KSOS ADMM-cuda + MPPI) vs PS / MPPI / DIAL", x=0.04, ha="left",
             fontsize=12.5, color=INK, weight="bold")
import textwrap  # noqa: E402
fig.text(0.04, 0.925, textwrap.fill("start: block (0.1, 0.1, 1.3), pusher (0, 0); 100 cycles at 50 Hz. GlobalMPPI: 5 restarts × 1000 "
         "two-level Sobol samples, ADMM-cuda (5,000 rollouts per cycle), then MPPI 1000 × 5 (σ 0.1, ≈ 10,000 rollouts in all; the 500 + 500 run halves both); "
         "PS / MPPI / DIAL: pusht.py settings with 2000 × 5 (≈ 10,000 rollouts). "
         "Time statistics leave out cycle 0 (JIT); std uses ddof = 1", 200), fontsize=9, color=INK_2, va="top")
fig.tight_layout(rect=(0, 0, 1, 0.87))
fig.savefig(out, facecolor=fig.get_facecolor())
fig.savefig(out.with_suffix(".pdf"), facecolor=fig.get_facecolor())

for m, seeds, final, mean_cost, T, per_run_t in rows:
    print(f"{LABELS[m]} seeds {seeds}")
    print("  final cost  " + " / ".join(f"{v:.4f}" for v in final) + (
        f"   mean {final.mean():.4f} ± {final.std(ddof=1):.4f} (var {final.var(ddof=1):.2e})" if len(final) > 1 else ""))
    print("  mean cost   " + " / ".join(f"{v:.3f}" for v in mean_cost))
    print(f"  plan time   per run " + " / ".join(f"{v:.2f}" for v in per_run_t)
          + f" s; all cycles median {np.median(T):.3f} max {T.max():.2f} s")
print(out)
