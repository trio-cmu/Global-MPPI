"""Cost convergence in the style of Global-MPPI/data/logs/plot_result.py (median + IQR, log y), cube task only.

    /usr/bin/python3 plot_cube_cost.py

cost = best_cost_history (planner's predicted cost of its planned mean trajectory at each cycle, sum over the horizon incl.
terminal; NOT the realized cost). Ours: ~/cube_runs_mj336/<cfg>/<algo>/s<k>/*best_cost_history_seed<k>.npy (101 values, cycles 0..100,
== cycles.csv best_cost for 0..99). Old: Global-MPPI/data/logs/cube_{ps,mppi,dial,lse2_mppiksos}_best_cost_history_seed<k>.npy.
gm_noise0.02 not drawn. Writes cube_cost_new.png, cube_cost_old.png, cube_cost_new_vs_old.png.
"""
import glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path.home() / "cube_runs_mj336"; OUT = R / "figures"
OLD = Path.home() / "kernelSOS_project/Global-MPPI/data/logs"
COLORS = {"Global-MPPI": "tab:blue", "Predictive Sampling": "tab:orange", "MPPI": "tab:green", "DIAL-MPC": "tab:red"}
NEW = {"Global-MPPI": ("gm", "gm"), "Predictive Sampling": ("ps", "ps"), "MPPI": ("mppi", "mppi"), "DIAL-MPC": ("dial", "dial")}
OLDTAG = {"Global-MPPI": "lse2_mppiksos", "Predictive Sampling": "ps", "MPPI": "mppi", "DIAL-MPC": "dial"}
new = {n: np.stack([np.squeeze(np.load(glob.glob(f"{R}/{c}/{a}/s{s}/*best_cost_history_seed{s}.npy")[0])) for s in range(6)]) for n, (c, a) in NEW.items()}
old = {n: np.stack([np.squeeze(np.load(OLD / f"cube_{t}_best_cost_history_seed{s}.npy")) for s in range(6)]) for n, t in OLDTAG.items()}
assert all(v.shape == (6, 101) for v in list(new.values()) + list(old.values()))


def panel(ax, data, title):
    x = np.arange(101)
    for name, c in data.items():
        q25, med, q75 = np.quantile(c, [0.25, 0.5, 0.75], axis=0)
        ax.plot(x, med, label=name, color=COLORS[name], lw=2)
        ax.fill_between(x, q25, q75, color=COLORS[name], alpha=0.25)
    ax.set_title(title); ax.set_xlabel("Iteration"); ax.set_ylabel("Cost"); ax.set_yscale("log"); ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, lw=0.8); ax.tick_params(axis="both", which="minor", length=1.5, width=0.4)
    ax.legend(loc="lower left", frameon=True, fancybox=False, framealpha=0.9, edgecolor="0.3")


for fn, data in (("cube_cost_new.png", new), ("cube_cost_old.png", old)):
    f, ax = plt.subplots(figsize=(5.5, 6)); panel(ax, data, "Dexterous in-hand manipulation"); plt.tight_layout(); f.savefig(OUT / fn, dpi=300, bbox_inches="tight")
f, axes = plt.subplots(1, 2, sharey=True, figsize=(11, 6))
panel(axes[0], new, "Dexterous in-hand manipulation (new runs, MuJoCo 3.3.6)")
panel(axes[1], old, "Dexterous in-hand manipulation (old data/logs)")
plt.tight_layout(); f.savefig(OUT / "cube_cost_new_vs_old.png", dpi=300, bbox_inches="tight")
print("median cost at iteration 100 (index 100) / 99; IQR at 100")
for lab, d in (("new", new), ("old", old)):
    for n, c in d.items():
        q = np.quantile(c[:, 100], [.25, .5, .75]); print(f"{lab} {n:20s} it100 {q[1]:.4f} [{q[0]:.4f}, {q[2]:.4f}]  it99 {np.median(c[:, 99]):.4f}  it0 {np.median(c[:, 0]):.3f}")
