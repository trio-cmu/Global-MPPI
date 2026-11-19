"""Plot bugtrap planner cost against cycle: median and IQR across six seeds.

    python script/plot_bugtrap_cost.py [RESULTS_DIR] [--out OUTPUT_STEM]

Reads <method>/s<seed>/cycles.csv and writes PNG and PDF figures.
The plotted best_cost is the predicted horizon cost, as in the benchmark README.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = {
    "ps": ("PS", "tab:orange"),
    "mppi": ("MPPI", "tab:green"),
    "dial": ("DIAL", "tab:red"),
    "mtp": ("MTP", "tab:purple"),
    "cbo": ("CBO", "tab:brown"),
    "gm": ("GlobalMPPI", "tab:blue"),
}


def main():
    default = Path(__file__).resolve().parents[1] / "data/bugtrap_results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", type=Path, default=default)
    parser.add_argument("--out", type=Path, help="Output filename stem (without extension)")
    args = parser.parse_args()
    output = args.out or args.results / "bugtrap_cost_vs_cycle"
    output.parent.mkdir(parents=True, exist_ok=True)

    cycles = np.arange(101)
    data = {}
    for method in METHODS:
        runs = []
        for seed in range(6):
            source = args.results / method / f"s{seed}" / "cycles.csv"
            with source.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            actual = [int(row["cycle"]) for row in rows]
            if actual != cycles.tolist():
                raise ValueError(f"{source}: expected exactly cycles 0–100")
            costs = np.array([float(row["best_cost"]) for row in rows])
            if not np.all(np.isfinite(costs)) or not np.all(costs > 0):
                raise ValueError(f"{source}: log plot requires finite, positive costs")
            runs.append(costs)
        data[method] = np.stack(runs)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for method, (label, color) in METHODS.items():
        low, median, high = np.quantile(data[method], [0.25, 0.5, 0.75], axis=0)
        ax.fill_between(cycles, low, high, color=color, alpha=0.16, linewidth=0)
        ax.plot(cycles, median, label=label, color=color,
                linewidth=2.6 if method == "gm" else 2.0)
        print(f"{label:12s} cost@100: median={median[-1]:.6g}, "
              f"IQR=[{low[-1]:.6g}, {high[-1]:.6g}]")
    ax.set(title="Bugtrap: cost vs. control cycle", xlabel="Control cycle",
           ylabel="Predicted planner cost", xlim=(0, 100), yscale="log")
    ax.set_xticks(np.arange(0, 101, 20))
    ax.grid(True, which="major", alpha=0.25, linewidth=0.7)
    ax.grid(True, which="minor", axis="y", alpha=0.10, linewidth=0.5)
    ax.legend(loc="lower left", ncol=2, framealpha=0.95, edgecolor="0.85")
    fig.text(0.5, 0.025, "Median across 6 seeds; shaded bands show the 25th–75th percentiles.",
             ha="center", fontsize=10, color="0.35")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    for suffix in (".png", ".pdf"):
        destination = output.with_suffix(suffix)
        fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved {destination}")
    plt.close(fig)


if __name__ == "__main__":
    main()
