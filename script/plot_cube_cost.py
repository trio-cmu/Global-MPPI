"""Cost convergence on the cube task: median over seeds (line) and 25-75 % band, log y axis.

    python script/plot_cube_cost.py RUNS_DIR [--out cube_cost.png] [--old OLD_LOGS_DIR]

RUNS_DIR is the output of script/run_cube_benchmark.sh (<config>/<algo>/s<seed>/*best_cost_history_seed<seed>.npy for the
configs ps, mppi, dial, gm). The cost is best_cost_history: at every control cycle, the summed cost (running + terminal) of the
planned mean trajectory, i.e. the planner's predicted cost, not the realised cost. 101 values per run (cycles 0..100).
With --old, a second panel shows the same plot for a folder of old runs (data/logs: cube_{ps,mppi,dial,lse2_mppiksos}_best_cost_history_seed<k>.npy).
Prints the median / quartiles at iteration 100.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("runs")
ap.add_argument("--out", default="cube_cost.png")
ap.add_argument("--old", default=None)
ap.add_argument("--seeds", type=int, default=6)
args = ap.parse_args()

COLORS = {"Global-MPPI": "tab:blue", "Predictive Sampling": "tab:orange", "MPPI": "tab:green", "DIAL-MPC": "tab:red"}
NEW = {"Global-MPPI": "gm", "Predictive Sampling": "ps", "MPPI": "mppi", "DIAL-MPC": "dial"}   # config = algo folder
OLD = {"Global-MPPI": "lse2_mppiksos", "Predictive Sampling": "ps", "MPPI": "mppi", "DIAL-MPC": "dial"}


def load(pattern):
    runs = [np.squeeze(np.load(glob.glob(pattern.format(s=s))[0])) for s in range(args.seeds)]
    return np.stack(runs)


new = {n: load(os.path.join(args.runs, c, c, "s{s}", "*best_cost_history_seed{s}.npy")) for n, c in NEW.items()}
old = {n: load(os.path.join(args.old, "cube_%s_best_cost_history_seed{s}.npy" % t)) for n, t in OLD.items()} if args.old else None


def panel(ax, data, title):
    for name, c in data.items():
        q25, med, q75 = np.quantile(c, [0.25, 0.5, 0.75], axis=0)
        x = np.arange(c.shape[1])
        ax.plot(x, med, label=name, color=COLORS[name], lw=2)
        ax.fill_between(x, q25, q75, color=COLORS[name], alpha=0.25)
    ax.set_title(title); ax.set_xlabel("Iteration"); ax.set_ylabel("Cost"); ax.set_yscale("log"); ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, lw=0.8)
    ax.legend(loc="lower left", frameon=True, fancybox=False, framealpha=0.9, edgecolor="0.3")


panels = [(new, "Dexterous in-hand manipulation")] + ([(old, "Dexterous in-hand manipulation (old runs)")] if old else [])
fig, axes = plt.subplots(1, len(panels), sharey=True, figsize=(5.5 * len(panels), 6), squeeze=False)
for ax, (d, t) in zip(axes[0], panels):
    panel(ax, d, t)
plt.tight_layout()
fig.savefig(args.out, dpi=300, bbox_inches="tight")
print("saved", args.out)
for label, d in [("runs", new)] + ([("old", old)] if old else []):
    for n, c in d.items():
        q = np.quantile(c[:, 100], [0.25, 0.5, 0.75])
        print("%-4s %-20s iteration 100: median %.4f  IQR [%.4f, %.4f]" % (label, n, q[1], q[0], q[2]))
