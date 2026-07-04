import mujoco
import argparse

from evosax.algorithms.distribution_based.cma_es import CMA_ES
from hydrax.algs import PredictiveSampling, MPPI, CEM, Evosax, DIAL, KSOS, MPPIKSOS
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.pusht import PushT
import ipdb
import numpy as np
"""
Run an interactive simulation of the push-T task with predictive sampling.
"""

# Define the task (cost and dynamics)
task = PushT()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the particle tracking task."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("cmaes", help="CMA-ES")
subparsers.add_parser("openes", help="OpenAI-ES")
subparsers.add_parser("sa", help="Simulated Annealing")
subparsers.add_parser("xnes", help="Exponential Natural Evolution Strategy")
subparsers.add_parser("gld", help="Gradient-Less Descent")
subparsers.add_parser("rs", help="Uniform Random Search")
subparsers.add_parser(
    "dial", help="Diffusion-Inspired Annealing for Legged MPC (DIAL)"
)
subparsers.add_parser(
    "ksos", help="Kernel-Smoothed Optimized Sampling (KSOS)"
)
subparsers.add_parser(
    "mppiksos", help="Model Predictive Path Integral Control with KSOS"
)
args = parser.parse_args()

# Set up the controller
# Set the controller based on command-line arguments
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=256,
        noise_level=0.4,
        num_randomizations=1,
        plan_horizon=1.0,
        spline_type="cubic",
        num_knots=6,
    ) 
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=256,
        noise_level=1.0,
        temperature=0.1,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=6,
        iterations = 5,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=256, #256 # more num_samples is better for CEM
        num_elites=3,
        sigma_start=1,
        sigma_min=0.1,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=6,
        explore_fraction = 0.1,
        # iterations = 10,        # more iterations is better for CEM
    )
elif args.algorithm == "cmaes":
    ctrl = Evosax(
        task,
        CMA_ES,
        num_samples=256,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=0.001,
        plan_horizon=1.0,
        spline_type="cubic",
        num_knots=4,
        iterations=5,
    )
elif args.algorithm == "dial":
    print("Running DIAL")
    ctrl = DIAL(
        task,
        num_samples=256,
        noise_level=1.0,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=0.01,
        plan_horizon=1.0,
        spline_type="cubic",
        num_knots=6,
    )
elif args.algorithm == "mppiksos":
    print("Running MPPI-KSOS")
    ctrl = MPPIKSOS(
        task,
        num_samples=256,
        noise_level=0.1,
        temperature=0.1,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=6,
        iterations=5,
    )

# Run the interactive simulation
start_seed = 0
set_random_pos = False
for trial_idx in range(start_seed, 6): 
    # Define the model used for simulation
    mj_model = task.mj_model
    mj_model.opt.timestep = 0.001
    mj_model.opt.iterations = 100
    mj_model.opt.ls_iterations = 50
    # set large friction for qusial static pushing
    # mj_model.geom_friction[:] = np.array([5.0, 0.005, 0.0001])
    mj_data = mujoco.MjData(mj_model)
    # ipdb.set_trace()
    np.random.seed(trial_idx)
    if set_random_pos == True:
        block_initial_pos = np.array([
            np.random.uniform(0.05, 0.1),
            np.random.uniform(0.05, 0.1),
            np.random.uniform(-1.3, 1.3),
        ])
        mj_data.qpos = [block_initial_pos[0], block_initial_pos[1], block_initial_pos[2], 0.0, 0.0]
        # ipdb.set_trace()
    else:
        mj_data.qpos = [0.1, 0.1, 1.3, 0.0, 0.0]
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    print(f"\n=== Seed {trial_idx} ===")
    run_interactive(
        ctrl,
        mj_model,
        mj_data,
        frequency=50,
        show_traces=False,
        record_video=False,
        max_traces=5,
        current_seed=trial_idx,
        max_cycles=100,
        current_task="pushT_ps4",
    )
    # ipdb.set_trace()
