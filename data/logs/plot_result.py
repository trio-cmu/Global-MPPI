import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
data_dir = Path(__file__).resolve().parent
out_path = data_dir / "all_cost_convergence_comparison_pusht_cube.png"

num_trials = 6
num_iters = 100

# -----------------------------
# Utilities
# -----------------------------
def load_histories(task_name: str, method_tag: str, num_trials: int):
    """
    Loads arrays from:
      {task}_{method_tag}_best_cost_history_seed{seed}.npy
    Returns: list of 1D arrays (len = num_trials)
    """
    arr_list = []
    missing = []
    for seed in range(num_trials):
        fname = f"{task_name}_{method_tag}_best_cost_history_seed{seed}.npy"
        fpath = data_dir / fname
        if not fpath.exists():
            missing.append(str(fpath))
            continue
        arr = np.load(fpath)
        arr = np.squeeze(arr)
        arr_list.append(arr)

    if missing:
        msg = "\n".join(missing[:20])
        more = "" if len(missing) <= 20 else f"\n... and {len(missing)-20} more"
        raise FileNotFoundError(
            f"Missing {len(missing)} files for task='{task_name}', method_tag='{method_tag}'.\n"
            f"Examples:\n{msg}{more}\n\n"
            f"Check your data_dir or filename pattern."
        )

    return arr_list

def stack_to_T(cost_list, num_iters):
    costs = np.stack(cost_list, axis=0)
    return costs[:, :num_iters]

# -----------------------------
# Load data
# -----------------------------
# Cube
cube_ps   = stack_to_T(load_histories("cube",  "ps",   num_trials), num_iters)
cube_ksos = stack_to_T(load_histories("cube",  "lse2_mppiksos", num_trials), num_iters)
cube_dial = stack_to_T(load_histories("cube",  "dial", num_trials), num_iters)
cube_mppi = stack_to_T(load_histories("cube",  "mppi", num_trials), num_iters)

# PushT
pusht_ps   = stack_to_T(load_histories("pushT", "ps",   num_trials), num_iters)
pusht_ksos = stack_to_T(load_histories("pushT", "mppiksos", num_trials), num_iters)
pusht_dial = stack_to_T(load_histories("pushT", "dial", num_trials), num_iters)
pusht_mppi = stack_to_T(load_histories("pushT", "mppi", num_trials), num_iters)

# -----------------------------
# Plot settings
# -----------------------------
COLORS = {
    "Global-MPPI": "tab:blue",
    "Predictive Sampling": "tab:orange",
    "MPPI": "tab:green",
    "DIAL-MPC": "tab:red",
}
LINESTYLES = {k: "-" for k in COLORS.keys()}

iters = np.arange(num_iters)

def plot_median_iqr(ax, methods, title):
    for name, costs in methods.items():
        q25, med, q75 = np.quantile(costs, [0.25, 0.5, 0.75], axis=0)

        ax.plot(
            iters, med,
            label=name,
            color=COLORS[name],
            linestyle=LINESTYLES[name],
            linewidth=2,
        )
        ax.fill_between(
            iters, q25, q75,
            color=COLORS[name],
            alpha=0.25,
        )

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_yscale("log")
    ax.grid(True)
    ax.tick_params(axis="both", which="minor", length=1.5, width=0.4)

# -----------------------------
# Plot: 1 figure, 2 subplots (PushT left, Cube right)
# -----------------------------
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 6))

methods_pusht = {
    "Global-MPPI": pusht_ksos,
    "Predictive Sampling": pusht_ps,
    "MPPI": pusht_mppi,
    "DIAL-MPC": pusht_dial,
}
plot_median_iqr(axes[0], methods_pusht, "PushT")

methods_cube = {
    "Global-MPPI": cube_ksos,
    "Predictive Sampling": cube_ps,
    "MPPI": cube_mppi,
    "DIAL-MPC": cube_dial,
}
plot_median_iqr(axes[1], methods_cube, "Dexterous in-hand manipulation")

for ax in axes:
    ax.legend(
        loc="lower left",
        frameon=True,
        fancybox=False,
        framealpha=0.9,
        edgecolor="0.3",
    )
    ax.grid(True, alpha=0.3, linewidth=0.8)
plt.tight_layout()

out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")