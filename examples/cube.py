import argparse

from evosax.algorithms.distribution_based.cma_es import CMA_ES

import mujoco
import ipdb
from global_mppi.algs import CEM, MPPI, Evosax, PredictiveSampling, DIAL, MPPIKSOS
from global_mppi.simulation.deterministic import run_interactive
from global_mppi.tasks.cube import CubeRotation

"""
Run an interactive simulation of the cube rotation task.

Double click on the floating target cube, then change the goal orientation with
[ctrl + left click].
"""

# Define the task (cost and dynamics)
task = CubeRotation()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the cube rotation task."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("cmaes", help="CMA-ES")
subparsers.add_parser("dial", help="DIAL")
subparsers.add_parser("mppiksos", help="MPPI-KSOS")
args = parser.parse_args()

# Set the controller based on command-line arguments
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=128,
        noise_level=0.2,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=4,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=128,
        noise_level=0.2,
        temperature=0.001,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=4,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=128,
        num_elites=5,
        sigma_start=0.5,
        sigma_min=0.5,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=4,
    )
elif args.algorithm == "cmaes":
    print("Running CMA-ES")
    ctrl = Evosax(
        task,
        CMA_ES,
        num_samples=128,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=4,
    )
elif args.algorithm == "dial":
    print("Running DIAL")
    ctrl = DIAL(
        task,
        num_samples=128,
        noise_level=1.0,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=0.01,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=4,
    )
elif args.algorithm == "mppiksos":
    print("Running MPPI-KSOS")
    ctrl = MPPIKSOS(
        task,
        num_samples=128,
        noise_level=0.1,
        temperature=0.1,
        spline_type="zero",
        plan_horizon=0.25,
        num_knots=4,
        iterations=5,
    )
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)

# Run the interactive simulation
start_seed = 0
set_random_pos = True
for trial_idx in range(start_seed, 6):
    mj_model = task.mj_model
    # set large friction for qusial static pushing
    # mj_model.geom_friction[:] = np.array([5.0, 0.005, 0.0001])
    mj_data = mujoco.MjData(mj_model)
    qpos = [
    -0.8, 0., -0.8, -0.8, -0.8,
    0., -0.8, -0.8, -0.8, 0.,
    -0.8, -0.8, -0.8, -0.8, -0.8,
    0., 0.11, 0., 0.1, 0.70710678,
    0., 0., 0.70710678
    ]
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    print(f"\n=== Seed {trial_idx} ===")
    run_interactive(
        ctrl,
        mj_model,
        mj_data,
        frequency=25,
        fixed_camera_id=None,
        show_traces=False,
        max_traces=1,
        trace_color=[1.0, 1.0, 1.0, 1.0],
        max_cycles=100,
        current_task = "cube_nomppi",
        current_seed=trial_idx,
    )
