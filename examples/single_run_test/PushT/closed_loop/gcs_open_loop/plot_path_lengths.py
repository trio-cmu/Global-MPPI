"""Path lengths of the five runs in videos_gcs_model/all_methods_gcs_model.mp4 (seed 0, GCS's quasi-static model):
T path, pusher path, final distance to the goal, and score = T path + pusher path + W * final distance.

Reads videos_gcs_model/all_methods_gcs_model_path_lengths.json and _path_score.json.

    python plot_path_lengths.py [out.png]
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent / "videos_gcs_model"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "all_methods_gcs_model_paths.png"
paths = json.load(open(HERE / "all_methods_gcs_model_path_lengths.json"))
score = json.load(open(HERE / "all_methods_gcs_model_path_score.json"))
W = score["W_per_m"]
METHODS = [("PS", "PS", "#52514e"), ("MPPI", "MPPI", "#1baf7a"), ("DIAL", "DIAL", "#eb6834"),
           ("CACS (ours)", "CACS\n(ours)", "#d43d51"), ("GCS (new cost)", "GCS\n(new cost)", "#2a78d6")]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

fig, axes = plt.subplots(1, 4, figsize=(19, 5.4), dpi=150)
fig.patch.set_facecolor(SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
x = np.arange(len(METHODS))
names = [m for m, _, _ in METHODS]
colors = [c for _, _, c in METHODS]
ticks = [t for _, t, _ in METHODS]


def label(ax, vals, fmt):
    top = max(vals)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02 * top, fmt(v), ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_ylim(0, top * 1.15)


# 1. T path
ax = axes[0]
v = [paths[n]["block_com_path_m"] for n in names]
ax.bar(x, v, width=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
ax.axhline(0.126, color=INK_2, lw=1.2, ls="--")
ax.text(1, 0.126 + 0.004, "straight line 0.126 m", ha="center", va="bottom", fontsize=8.5, color=INK_2,
        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
label(ax, v, lambda a: f"{a:.3f} m")
ax.set_ylabel("T centre-of-mass path (m)", color=INK_2)
ax.set_title("T path length", loc="left", fontsize=11, color=INK, weight="bold")

# 2. pusher path, split into not touching / touching the T
ax = axes[1]
off = np.array([paths[n]["pusher_until_stop_off_contact_m"] for n in names])
on = np.array([paths[n]["pusher_until_stop_in_contact_m"] for n in names])
ax.bar(x, off, width=0.62, color=colors, edgecolor=SURFACE, linewidth=2, label="not touching the T")
ax.bar(x, on, width=0.62, bottom=off, color=colors, alpha=0.45, edgecolor=SURFACE, linewidth=2, label="touching the T")
label(ax, list(off + on), lambda a: f"{a:.2f} m")
ax.set_ylabel("pusher path until the T stops moving (m)", color=INK_2)
ax.set_title("Pusher path length (until the T stops)", loc="left", fontsize=11, color=INK, weight="bold")

h = [plt.Rectangle((0, 0), 1, 1, color="#8a8984"), plt.Rectangle((0, 0), 1, 1, color="#8a8984", alpha=0.45)]
ax.legend(h, ["not touching the T", "touching the T"], frameon=False, fontsize=9, labelcolor=INK_2, loc="upper center", bbox_to_anchor=(0.4, 1.0))

# 3. final distance to the goal
ax = axes[2]
d = [score["results"][n]["final_corner_dist_mm"] for n in names]
ax.bar(x, d, width=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
label(ax, d, lambda a: f"{a:.1f} mm")
ax.set_ylabel("final distance to goal (mm, mean over the T's 8 corners)", color=INK_2)
ax.set_title("Final distance to the goal", loc="left", fontsize=11, color=INK, weight="bold")

# 4. score, stacked by part
ax = axes[3]
tp = np.array([paths[n]["block_com_path_m"] for n in names])
pp = np.array([paths[n]["pusher_path_until_block_stops_m"] for n in names])
pen = W * np.array(d) / 1000
ax.bar(x, tp, width=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
ax.bar(x, pp, width=0.62, bottom=tp, color=colors, alpha=0.55, edgecolor=SURFACE, linewidth=2)
ax.bar(x, pen, width=0.62, bottom=tp + pp, color=colors, alpha=0.25, edgecolor=SURFACE, linewidth=2, hatch="///")
label(ax, list(tp + pp + pen), lambda a: f"{a:.2f}")
ax.set_ylabel(f"T path + pusher path + {W:g} x final distance", color=INK_2)
ax.set_title("Score (lower is better)", loc="left", fontsize=11, color=INK, weight="bold")
h = [plt.Rectangle((0, 0), 1, 1, color="#8a8984"), plt.Rectangle((0, 0), 1, 1, color="#8a8984", alpha=0.55),
     plt.Rectangle((0, 0), 1, 1, facecolor="#8a8984", alpha=0.25, hatch="///")]
ax.legend(h, ["T path", "pusher path", f"{W:g} x final distance"], frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")

for ax in axes:
    ax.set_xticks(x, ticks, color=INK_2)
axes[1].set_xticks(x, [f"{t.split(chr(10))[0]}\nT stops\nat {paths[n]['t_block_stops_s']:.2f} s" for t, n in zip(ticks, names)],
                   color=INK_2, fontsize=9)
fig.suptitle("PushT in GCS's quasi-static model, seed 0: path lengths and distance to goal", x=0.02, ha="left",
             fontsize=12.5, color=INK, weight="bold")
fig.text(0.02, 0.905, "PS / MPPI / DIAL / CACS: closed loop, 100 cycles = 2 s (run with the previous cost). GCS: one plan with the "
         "new cost, 10.5 s. Start: block (0.1, 0.1, 1.3), pusher (0, 0.1). Pusher path counted until the T stops moving "
         "for good (its last push); the pusher's later moves are not counted.", fontsize=9, color=INK_2, va="top")
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(out, facecolor=fig.get_facecolor())
fig.savefig(out.with_suffix(".pdf"), facecolor=fig.get_facecolor())
print(out)
