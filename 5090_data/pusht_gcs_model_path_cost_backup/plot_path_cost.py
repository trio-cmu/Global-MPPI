"""PushT in GCS's quasi-static model, path cost J = L_T + 0.1 L_P + w D_final (score_path_cost.py; w = --w-final, 10).

Left: J of every seed (dots) with the mean ± std per method, GCS's plan as a dashed line.
Right: the mean of each term of J per method (stacked), GCS included.

    python plot_path_cost.py [results dir] [GCS traj .npz] [out.png] [--w-final W]
    (defaults: this folder, ../gcs_path_pusher_wp0.1/gcs_path_pusher_wp0.1_traj.npz, pusht_gcs_model_path_cost.png
    here, or pusht_gcs_model_path_cost_wfinal<W>.png for another W; W = 10, the weight the controllers plan with)

Only finished runs (101 states: cycles 0-100) are drawn; a run still in progress is skipped.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent))
from score_path_cost import score  # noqa: E402

W_FINAL, W_P = 10.0, 0.1
argv = []
for a in sys.argv[1:]:
    if a.startswith("--w-final"):
        W_FINAL = float(a.split("=", 1)[1])
    else:
        argv.append(a)
src = Path(argv[0]) if len(argv) > 0 else here
gcs = Path(argv[1]) if len(argv) > 1 else here.parent / "gcs_path_pusher_wp0.1" / "gcs_path_pusher_wp0.1_traj.npz"
suffix = "" if W_FINAL == 10 else f"_wfinal{W_FINAL:g}"
out = Path(argv[2]) if len(argv) > 2 else here / f"pusht_gcs_model_path_cost{suffix}.png"
N_STATES = 101  # a finished run: the state of cycles 0-100

METHODS = {  # folder: (label, color)
    "ps": ("PS", "#52514e"),
    "mppi": ("MPPI", "#1baf7a"),
    "dial": ("DIAL", "#eb6834"),
    "mtp": ("MTP", "#8e6bd1"),
    "cbo": ("CBO", "#a0785a"),
    "gm": ("cuKernelSOS", "#d43d51"),
}
GCS_COLOR = "#2a78d6"
TERMS = (("L_T_m", 1.0, "T path  $L_T$", "#9bb7d4"),
         ("L_P_m", W_P, "0.1 × pusher path  $L_P$", "#e8b86d"),
         ("D_final_m", W_FINAL, f"{W_FINAL:g} × final distance  $D_{{final}}$", "#c96f6f"))
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

runs = {}
for folder in METHODS:
    files = [f for f in sorted((src / folder).glob("s*/*robot_pos_history*.npy")) if len(np.load(f)) >= N_STATES]
    if files:
        runs[folder] = [score(str(f), W_FINAL, W_P, 0.02) for f in files]
g = score(str(gcs), W_FINAL, W_P, 0.02)

fig, (ax_j, ax_t) = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=150, gridspec_kw=dict(width_ratios=[1, 1.15]))
for ax in (ax_j, ax_t):
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=11, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")

# left: J per seed
rng = np.random.default_rng(0)
for i, (folder, rs) in enumerate(runs.items()):
    label, color = METHODS[folder]
    J = np.array([r["J"] for r in rs])
    ax_j.scatter(i + rng.uniform(-0.12, 0.12, len(J)), J, s=46, color=color, alpha=0.75, edgecolor="white", lw=1.2, zorder=3)
    ax_j.errorbar(i + 0.3, J.mean(), yerr=J.std(ddof=1), fmt="o", ms=8, color=color, capsize=5, lw=2, zorder=4)
    ax_j.text(i + 0.42, J.mean(), f"{J.mean():.3f}", va="center", fontsize=10, color=INK_2)
    print(f"{label:12s} n={len(J)} J {J.mean():.4f} ± {J.std(ddof=1):.4f}")
ax_j.axhline(g["J"], color=GCS_COLOR, ls="--", lw=2, zorder=2)
ax_j.text(len(runs) - 0.35, g["J"] * 1.03, f"GCS  {g['J']:.3f}", color=GCS_COLOR, fontsize=11, weight="bold", va="bottom", ha="right")
ax_j.set_xticks(range(len(runs)))
ax_j.set_xticklabels([METHODS[f][0] for f in runs], fontsize=12, color=INK)
ax_j.set_xlim(-0.5, len(runs) - 0.3)
ax_j.set_yscale("log")  # one method can fail by an order of magnitude (CBO): keep the others readable
J_all = [r["J"] for rs in runs.values() for r in rs] + [g["J"]]
ticks = [t for t in (0.15, 0.2, 0.3, 0.5, 1, 1.5, 2, 3, 5, 10, 20) if min(J_all) * 0.9 <= t <= max(J_all) * 1.1]
ax_j.set_yticks(ticks)
ax_j.set_yticklabels([f"{t:g}" for t in ticks])
ax_j.minorticks_off()
ax_j.set_ylim(min(J_all) * 0.75, max(J_all) * 1.4)
ax_j.set_ylabel(f"J = $L_T$ + 0.1 $L_P$ + {W_FINAL:g} $D_{{final}}$", color=INK, fontsize=12, weight="bold")
ax_j.set_title("Path cost per seed (lower is better)", loc="left", fontsize=12, color=INK, weight="bold")

# right: the terms of J, stacked (mean over seeds)
names = ["GCS"] + [METHODS[f][0] for f in runs]
rows = [[g[k] * w for k, w, _, _ in TERMS]] + [[np.mean([r[k] for r in rs]) * w for k, w, _, _ in TERMS] for rs in runs.values()]
rows = np.array(rows)
# bars much taller than the rest (a failed method) are cut at the top of the axis and labelled
_tot = np.sort(rows.sum(1))
CAP = max(0.36, 1.3 * _tot[-2]) if _tot[-1] > 2 * _tot[-2] else 1.25 * _tot[-1]
bottom = np.zeros(len(names))
for j, (_, _, tlabel, tcolor) in enumerate(TERMS):
    ax_t.bar(names, rows[:, j], bottom=bottom, color=tcolor, width=0.6, edgecolor="white", lw=2, label=tlabel)
    bottom += rows[:, j]
for i, tot in enumerate(bottom):
    if tot > CAP:
        ax_t.text(i, CAP * 0.985, f"{tot:.3f}\n(cut)", ha="center", va="top", fontsize=10, color="white", weight="bold")
    else:
        ax_t.text(i, tot + 0.004, f"{tot:.3f}", ha="center", va="bottom", fontsize=10, color=INK_2)
for lbl, c in zip(ax_t.get_xticklabels(), [GCS_COLOR] + [METHODS[f][1] for f in runs]):
    lbl.set_color(c)
    lbl.set_fontsize(12)
    lbl.set_weight("bold")
ax_t.set_ylabel("Mean contribution to J", color=INK, fontsize=12, weight="bold")
ax_t.set_title("What J is made of", loc="left", fontsize=12, color=INK, weight="bold")
ax_t.legend(frameon=False, fontsize=10, labelcolor=INK_2, loc="upper left")
ax_t.set_ylim(0, min(bottom.max() * 1.25, CAP))

fig.suptitle("PushT in GCS's quasi-static model, path cost (6 seeds per method; GCS: one open-loop plan)",
             x=0.01, ha="left", fontsize=12.5, color=INK, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out, facecolor="white")
print(out)
