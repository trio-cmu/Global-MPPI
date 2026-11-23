"""cube_cost_new.png with MTP and CBO added: same style as plot_cube_cost.py (median + IQR, log y, best_cost_history),
data from this folder (<algo>/<algo>/s<k>/*best_cost_history_seed<k>.npy). gm_noise0.02 not drawn.

    python figures/plot_cube_cost_new6.py      (run from anywhere; writes ../cube_cost_new.png)
"""
import glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent.parent
NEW = {"Global-MPPI": "gm", "Predictive Sampling": "ps", "MPPI": "mppi", "DIAL-MPC": "dial", "MTP": "mtp", "CBO": "cbo"}
COLORS = {"Global-MPPI": "tab:blue", "Predictive Sampling": "tab:orange", "MPPI": "tab:green", "DIAL-MPC": "tab:red",
          "MTP": "tab:purple", "CBO": "tab:brown"}
data = {n: np.stack([np.squeeze(np.load(glob.glob(f"{R}/{a}/{a}/s{s}/*best_cost_history_seed{s}.npy")[0])) for s in range(6)])
        for n, a in NEW.items()}
assert all(v.shape == (6, 101) for v in data.values())
f, ax = plt.subplots(figsize=(5.5, 6))
x = np.arange(101)
for name, c in data.items():
    q25, med, q75 = np.quantile(c, [0.25, 0.5, 0.75], axis=0)
    ax.plot(x, med, label=name, color=COLORS[name], lw=2)
    ax.fill_between(x, q25, q75, color=COLORS[name], alpha=0.25)
ax.set_title("Dexterous in-hand manipulation"); ax.set_xlabel("Iteration"); ax.set_ylabel("Cost"); ax.set_yscale("log"); ax.set_xlim(0, 100)
ax.grid(True, alpha=0.3, lw=0.8); ax.tick_params(axis="both", which="minor", length=1.5, width=0.4)
ax.legend(loc="lower left", frameon=True, fancybox=False, framealpha=0.9, edgecolor="0.3")
plt.tight_layout(); f.savefig(R / "cube_cost_new.png", dpi=300, bbox_inches="tight")
for n, c in data.items():
    q = np.quantile(c[:, 100], [.25, .5, .75]); print(f"{n:20s} it100 {q[1]:.4f} [{q[0]:.4f}, {q[2]:.4f}]")
