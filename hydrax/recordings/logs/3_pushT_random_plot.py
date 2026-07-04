import numpy as np
import matplotlib.pyplot as plt
import ipdb
# -----------------------------
# Generate random cost data
# -----------------------------
np.random.seed(0)

from pathlib import Path

data_dir = Path("/home/zhongqi/Documents/workspace/mppi_repo/hydrax/hydrax/recordings/logs/")

ps_cost_list = []
ksos_cost_list1 = []
ksos_cost_list1_nolse = []
mppi_cost_list = []
dial_cost_list = []
task_name = "pushT_random"
num_trials = 6
num_iters = 80

for seed in range(num_trials):
    fname = f"{task_name}_ps_best_cost_history_seed{seed}.npy"
    arr = np.load(data_dir / fname)
    arr = np.squeeze(arr)
    ps_cost_list.append(arr)

for seed in range(num_trials):
    fname = f"{task_name}_mppiksos_best_cost_history_seed{seed}.npy"
    arr = np.load(data_dir / fname)
    arr = np.squeeze(arr)
    ksos_cost_list1.append(arr)

# for seed in range(num_trials):
#     fname = f"{task_name}_nolse_mppiksos_best_cost_history_seed{seed}.npy"
#     arr = np.load(data_dir / fname)
#     arr = np.squeeze(arr)
#     ksos_cost_list1_nolse.append(arr)
      
for seed in range(num_trials):
    fname = f"{task_name}_dial_best_cost_history_seed{seed}.npy"
    arr = np.load(data_dir / fname)
    arr = np.squeeze(arr)
    dial_cost_list.append(arr)
    
for seed in range(num_trials):
    fname = f"{task_name}_mppi_best_cost_history_seed{seed}.npy"
    arr = np.load(data_dir / fname)
    arr = np.squeeze(arr)
    mppi_cost_list.append(arr)
# shape: (6, 100)

# ksos_cost_list1_nolse = [
#     np.pad(
#         np.asarray(x),
#         (0, num_iters - len(x)),
#         constant_values=x[-1],
#     ) if len(x) < num_iters else np.asarray(x)[:num_iters]
#     for x in ksos_cost_list1_nolse
# ]

ipdb.set_trace()
ps_costs = np.stack(ps_cost_list, axis=0)[:, :num_iters]
ksos_costs = np.stack(ksos_cost_list1, axis=0)[:, :num_iters]
# ksos_costs_nolse = np.stack(ksos_cost_list1_nolse, axis=0)[:, :num_iters]
mppi_costs = np.stack(mppi_cost_list, axis=0)[:, :num_iters]
dial_costs = np.stack(dial_cost_list, axis=0)[:, :num_iters]

# print(ps_costs.shape)  # (6, 100)
# print(ksos_costs.shape)  # (6, 100)

ipdb.set_trace()
num_runs = 1

methods = {
    "KernelSOS": ksos_costs,
    # "KernelSOS (No LSE)": ksos_costs_nolse,
    "Predictive Sampling": ps_costs,
    "MPPI": mppi_costs,
    "DIAL-MPC": dial_costs,
}

# -----------------------------
# Plot settings
# -----------------------------
COLORS = {
    "KernelSOS": "tab:blue",
    # "KernelSOS (No LSE)": "tab:cyan",
    "Predictive Sampling": "tab:orange",
    "MPPI": "tab:green",
    "DIAL-MPC": "tab:red",
}
LINESTYLES = {
    "KernelSOS": "-",
    # "KernelSOS (No LSE)": "-",
    "Predictive Sampling": "-",
    "MPPI": "-",
    "DIAL-MPC": "-",
}

fig, ax = plt.subplots(figsize=(6, 4))
iters = np.arange(num_iters)

# -----------------------------
# Plot median + IQR
# -----------------------------
for name, costs in methods.items():
    q25, med, q75 = np.quantile(costs, [0.25, 0.5, 0.75], axis=0)

    ax.plot(
        iters,
        med,
        label=name,
        color=COLORS[name],
        linestyle=LINESTYLES[name],
        linewidth=2,
    )
    ax.fill_between(
        iters,
        q25,
        q75,
        color=COLORS[name],
        alpha=0.25,
    )

# -----------------------------
# Styling
# -----------------------------
ax.set_xlabel("Iteration")
ax.set_ylabel("Cost")
ax.set_title("PushT Random Cost Convergence Comparison")
ax.set_yscale("log")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()
plt.savefig("hydrax/media/pushT_random_cost_convergence_comparison1.png", dpi=300)