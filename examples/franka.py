import argparse
import os
import sys

# Headless (the default) renders offscreen and records the rollout to an mp4
# under logs/<task>_<algorithm>/<timestamp>/; the offscreen GL backend must be
# selected before importing mujoco. --no-headless opens the interactive viewer.
if "--no-headless" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import jax.numpy as jnp
import mujoco

from global_mppi.algs import DIAL, MPPI, GlobalMPPI
from global_mppi.simulation.deterministic import run_interactive
from global_mppi.tasks.franka import FrankaReach

"""
Move the FR3 arm from the `start` to the `goal` keyframe around the ball
obstacles, and save a video of the rollout.
"""

parser = argparse.ArgumentParser(description="FR3 reach-around-obstacles task.")
parser.add_argument(
    "--ksos-solver",
    choices=["newton", "newton-rs", "ADMM-sparse"],
    default="newton",
    help="KSOS solver used by GlobalMPPI (default: newton)",
)
parser.add_argument(
    "--ksos-restarts",
    type=int,
    default=5,
    help="KSOS restarts per planning cycle for GlobalMPPI (default: 5)",
)
parser.add_argument(
    "--max-cycles",
    type=int,
    default=150,
    help="Number of planning cycles to simulate (default: 150)",
)
parser.add_argument(
    "--plan-horizon",
    type=float,
    default=0.25,
    help="Planning horizon in seconds (default: 0.25, i.e. 25 steps at dt=0.01)",
)
parser.add_argument(
    "--goal-weight",
    type=float,
    default=10.0,
    help="Running-cost weight of the squared joint distance to the goal "
    "(default: 10)",
)
parser.add_argument(
    "--terminal-weight",
    type=float,
    default=1000.0,
    help="Terminal-cost weight of the squared joint distance to the goal "
    "(default: 1000)",
)
parser.add_argument(
    "--headless",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run without a viewer window and record the rollout to a video. "
    "On by default; use --no-headless to open an interactive viewer.",
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser(
    "dial", help="Diffusion-Inspired Annealing for Legged MPC (DIAL)"
)
subparsers.add_parser("globalmppi", help="GlobalMPPI")
args = parser.parse_args()

task = FrankaReach(
    goal_weight=args.goal_weight,
    terminal_weight=args.terminal_weight,
)

if args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=1024,
        noise_level=0.1,
        temperature=0.1,
        normalize_costs=True,
        spline_type="cubic",
        plan_horizon=args.plan_horizon,
        num_knots=6,
        iterations=5,
    )
elif args.algorithm == "dial":
    print("Running DIAL")
    ctrl = DIAL(
        task,
        num_samples=1024,
        noise_level=0.1,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=0.1,
        normalize_costs=True,
        spline_type="cubic",
        plan_horizon=args.plan_horizon,
        num_knots=6,
        iterations=5,
    )
else:
    print(f"Running GlobalMPPI (KSOS solver: {args.ksos_solver})")
    ctrl = GlobalMPPI(
        task,
        num_samples=1024,
        noise_level=0.1,
        temperature=0.1,
        spline_type="cubic",
        plan_horizon=args.plan_horizon,
        num_knots=6,
        iterations=5,
        ksos_solver=args.ksos_solver,
        ksos_num_restart=args.ksos_restarts,
    )

# Start at rest in the `start` keyframe, holding that pose as the initial plan
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)
mujoco.mj_resetDataKeyframe(mj_model, mj_data, mj_model.key("start").id)
initial_knots = jnp.tile(task.ctrl_start, (ctrl.num_knots, 1))

run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=25,
    initial_knots=initial_knots,
    show_traces=False,
    max_cycles=args.max_cycles,
    current_task="franka",
    use_wandb=False,
    headless=args.headless,
)
