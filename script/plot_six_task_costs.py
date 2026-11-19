"""Redraw the six selected benchmark cost plots in equally sized subplots.

    python script/plot_six_task_costs.py [--out data/all_six_cost_vs_cycle.png]

Cube shows all six methods from cube_cost_six.png using native 101-value
NPY histories; G1 standup preserves its four methods and CSV cycles 0–99;
the other tasks use CSV cycles 0–100. Lines are seed medians, bands are IQRs.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("ps", "mppi", "dial", "mtp", "cbo", "gm")
ORIGINAL_FOUR = ("gm", "ps", "mppi", "dial")
LABELS = {"ps": "PS", "mppi": "MPPI", "dial": "DIAL", "mtp": "MTP",
          "cbo": "CBO", "gm": "GlobalMPPI"}
COLORS = {"ps": "tab:orange", "mppi": "tab:green", "dial": "tab:red",
          "mtp": "tab:purple", "cbo": "tab:brown", "gm": "tab:blue"}
TASKS = (
    ("(a) PushT", "pusht_results", METHODS, "csv", "lower left"),
    ("(b) Dexterous in-hand manipulation", "cube_results_mj336", METHODS, "npy", "lower left"),
    ("(c) Double cart-pole", "double_cart_pole_results", METHODS, "csv", "lower left"),
    ("(d) G1 standup", "g1_results", ORIGINAL_FOUR, "csv", "lower left"),
    ("(e) Bugtrap", "bugtrap_results", METHODS, "csv", "lower left"),
    ("(f) Walker", "walker_results", METHODS, "csv", "upper right"),
)


def load_costs(folder, method, source):
    base = ROOT / "data" / folder / method
    if folder == "cube_results_mj336":
        base /= method
    runs = []
    cycle_axis = None
    for seed in range(6):
        run = base / f"s{seed}"
        if source == "npy":
            paths = list(run.glob("*best_cost_history*.npy"))
            if len(paths) != 1:
                raise ValueError(f"Expected one cost history in {run}, found {len(paths)}")
            costs = np.load(paths[0]).reshape(-1)
            cycles = np.arange(len(costs))
        else:
            with (run / "cycles.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            costs = np.array([float(r["best_cost"]) for r in rows])
            cycles = np.array([int(r["cycle"]) for r in rows])
        expected = 100 if folder == "g1_results" else 101
        if len(costs) != expected or not np.array_equal(cycles, np.arange(expected)):
            raise ValueError(f"Unexpected or missing cycles in {run}")
        if not np.all(np.isfinite(costs)) or not np.all(costs > 0):
            raise ValueError(f"Nonpositive or nonfinite costs in {run}")
        if cycle_axis is not None and not np.array_equal(cycle_axis, cycles):
            raise ValueError(f"Seed cycle mismatch in {base}")
        cycle_axis = cycles
        runs.append(costs)
    return cycle_axis, np.stack(runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/all_six_cost_vs_cycle.png")
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14,
                         "axes.labelsize": 12, "legend.fontsize": 10,
                         "axes.spines.top": True, "axes.spines.right": True})
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    # A fixed GridSpec gives every axes rectangle identical width and height.
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.105, top=0.95,
                        wspace=0.25, hspace=0.30)
    for ax, (title, folder, methods, source, legend_location) in zip(axes.flat, TASKS):
        for method in methods:
            cycles, costs = load_costs(folder, method, source)
            low, median, high = np.quantile(costs, [0.25, 0.5, 0.75], axis=0)
            ax.fill_between(cycles, low, high, color=COLORS[method], alpha=0.22, linewidth=0)
            ax.plot(cycles, median, label=LABELS[method], color=COLORS[method], linewidth=2)
        ax.set(title=title, xlabel="Cycle", ylabel="Cost", xlim=(0, 100), yscale="log")
        ax.set_xticks(np.arange(0, 101, 20))
        ax.grid(True, which="major", alpha=0.28, linewidth=0.8)
        ax.tick_params(axis="both", which="minor", length=2, width=0.5)
        ax.legend(loc=legend_location, frameon=True, fancybox=False,
                  framealpha=0.92, edgecolor="0.5")
    fig.text(0.5, 0.025, "Median across 6 seeds; shaded bands show the 25th–75th percentiles.",
             ha="center", color="0.35", fontsize=11)
    fig.canvas.draw()
    sizes = np.array([[ax.bbox.width, ax.bbox.height] for ax in axes.flat])
    if not np.allclose(sizes, sizes[0], rtol=0, atol=1e-8):
        raise ValueError(f"Subplot sizes differ: {sizes}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, facecolor="white")
    print(f"Saved {args.out}")
    print(f"Verified all 6 axes have equal dimensions: {sizes[0].tolist()} display pixels")
    plt.close(fig)


if __name__ == "__main__":
    main()
