import argparse
import json
import os
import sys
import getpass

# On a headless machine (e.g. a remote server with no display) MuJoCo must use
# an offscreen GL backend. Select it BEFORE importing mujoco so the render
# context is created correctly. EGL uses the GPU; fall back to `osmesa` (CPU)
# by setting MUJOCO_GL yourself if EGL is unavailable.
# Headless is the default; only an explicit --no-headless opts out.
if "--no-headless" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
from pathlib import Path

# cuKernelSOS's Python binding (GPU ADMM, --ksos-solver ADMM-cuda): the first
# cuKernelSOS/python in a parent folder
for _parent in Path(__file__).resolve().parents:
    if (_parent / "cuKernelSOS" / "python" / "cukernelsos").is_dir():
        os.environ.setdefault("CUKERNELSOS_PYTHON", str(_parent / "cuKernelSOS" / "python"))
        break

from global_mppi.algs import PredictiveSampling, MPPI, DIAL, GlobalMPPI
from global_mppi.simulation.deterministic import run_interactive
from global_mppi.tasks.pusht import PushT
"""
Run an interactive simulation of the push-T task with predictive sampling.
"""

# Define the task (cost and dynamics)
task = PushT()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the particle tracking task."
)
# Weights & Biases logging options (place before the algorithm, e.g.
# `python pusht.py --wandb ps`)
parser.add_argument(
    "--wandb",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Log best_cost_history to Weights & Biases "
    "(on by default; use --no-wandb to disable)",
)
parser.add_argument(
    "--ksos-solver",
    choices=["newton", "newton-rs", "ADMM-sparse", "ADMM-cuda"],
    default="newton",
    help="KSOS solver used by GlobalMPPI (default: newton). ADMM-cuda uses the "
    "single-step test's setting: 1000 two-level Sobol candidates per restart, "
    "GPU ADMM, then MPPI with 1000 samples x 5 iterations at noise --mppi-noise (0.1)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="seed of the run: the controller's random draws (default: 0)",
)
parser.add_argument(
    "--num-samples",
    type=int,
    default=None,
    help="samples per iteration for ps / mppi / dial (default: 256), and for GlobalMPPI ADMM-cuda's "
    "MPPI step (default: 1000)",
)
parser.add_argument(
    "--ksos-cluster-size",
    type=int,
    default=10,
    help="GlobalMPPI ADMM-cuda: points per two-level cluster = ADMM clique size (default: 10)",
)
parser.add_argument(
    "--ksos-samples",
    type=int,
    default=1000,
    help="GlobalMPPI ADMM-cuda: KSOS candidates per restart (default: 1000)",
)
parser.add_argument(
    "--mppi-noise",
    type=float,
    default=0.1,
    help="GlobalMPPI ADMM-cuda: noise level of the MPPI step after KSOS (default: 0.1)",
)
parser.add_argument(
    "--lse-samples",
    type=int,
    default=0,
    help="GlobalMPPI ADMM-cuda: LSE perturbations per KSOS candidate (0 = no LSE; 100 in the single-step test)",
)
parser.add_argument(
    "--lse-sigmas",
    type=float,
    nargs=2,
    default=[0.2, 0.01],
    metavar=("FIRST", "LAST"),
    help="GlobalMPPI ADMM-cuda: LSE perturbation scale at the first and last restart (default: 0.2 0.01)",
)
parser.add_argument(
    "--lse-lambda",
    type=float,
    default=1.0,
    help="GlobalMPPI ADMM-cuda: LSE temperature (default: 1)",
)
parser.add_argument(
    "--ksos-sampling",
    choices=["two-level", "cacs"],
    default="two-level",
    help="GlobalMPPI ADMM-cuda: KSOS candidates from two-level Sobol in shrinking boxes (default), or budgeted "
    "cost-aware chordal sampling (CACS) in the same shrinking boxes, at every restart",
)
parser.add_argument(
    "--ksos-warm-start",
    action="store_true",
    help="GlobalMPPI ADMM-cuda with --ksos-sampling cacs: start from the previous cycle's plan (it joins the "
    "CACS initial batch and the guard)",
)
parser.add_argument(
    "--cacs",
    type=json.loads,
    default=None,
    help='GlobalMPPI ADMM-cuda with --ksos-sampling cacs: CacsConfig overrides as JSON, e.g. \'{"k": 128}\' '
    "(global_mppi/cacs.py; n is --ksos-samples)",
)
parser.add_argument(
    "--gnc",
    type=float,
    nargs=2,
    default=None,
    metavar=("FIRST", "LAST"),
    help="GlobalMPPI ADMM-cuda: kernel-surrogate GNC of the KSOS costs, bandwidth factor x sigma from FIRST to "
    "LAST over the restarts (e.g. 1.0 0.1; default: off)",
)
parser.add_argument(
    "--no-ksos-mppi",
    action="store_true",
    help="GlobalMPPI ADMM-cuda: no MPPI after the KSOS restarts; execute the KSOS solution",
)
parser.add_argument(
    "--ksos-restarts",
    type=int,
    default=5,
    help="KSOS restarts per planning cycle for GlobalMPPI (default: 5)",
)
parser.add_argument(
    "--wendland-sigma",
    type=float,
    default=1.75,
    help="Wendland kernel radius at restart 0 for ADMM-sparse (default: 1.75)",
)
parser.add_argument(
    "--wandb-project",
    default="global-mppi",
    help="W&B project name (default: global-mppi)",
)
parser.add_argument(
    "--wandb-entity",
    default=None,
    help="W&B entity/team (defaults to your W&B account's default entity)",
)
parser.add_argument(
    "--wandb-key",
    default=None,
    help="W&B API key. Only needed the first time; afterwards it is cached.",
)
parser.add_argument(
    "--headless",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run without a viewer window and record the rollout to a video "
    "(uploaded to W&B when --wandb is set). Use on remote/headless machines. "
    "On by default; use --no-headless to open an interactive viewer.",
)

subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser(
    "dial", help="Diffusion-Inspired Annealing for Legged MPC (DIAL)"
)
subparsers.add_parser(
    "globalmppi", help="GlobalMPPI"
)
args = parser.parse_args()

# Authenticate with Weights & Biases. The first time, prompt for an API key
# (grab yours from https://wandb.ai/authorize); it is cached in ~/.netrc so
# subsequent runs log in automatically.
if args.wandb:
    import wandb

    def _wandb_logged_in() -> bool:
        if os.environ.get("WANDB_API_KEY"):
            return True
        netrc = os.path.expanduser("~/.netrc")
        try:
            with open(netrc) as f:
                return "api.wandb.ai" in f.read()
        except OSError:
            return False

    if args.wandb_key:
        wandb.login(key=args.wandb_key)
    elif _wandb_logged_in():
        wandb.login()
    else:
        key = getpass.getpass(
            "Enter your W&B API key (from https://wandb.ai/authorize): "
        ).strip()
        wandb.login(key=key)

# Set up the controller
# Set the controller based on command-line arguments
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=args.num_samples or 256,
        noise_level=0.4,
        num_randomizations=1,
        plan_horizon=1.0,
        spline_type="cubic",
        num_knots=6,
        iterations = 5,
    ) 
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=args.num_samples or 256,
        noise_level=1.0,
        temperature=0.1,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=6,
        iterations = 5,
    )
elif args.algorithm == "dial":
    print("Running DIAL")
    ctrl = DIAL(
        task,
        num_samples=args.num_samples or 256,
        noise_level=1.0,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=0.01,
        plan_horizon=1.0,
        spline_type="cubic",
        num_knots=6,
        iterations = 5,
    )
elif args.algorithm == "globalmppi" and args.ksos_solver == "ADMM-cuda":
    print(f"Running GlobalMPPI (KSOS solver: ADMM-cuda, two-level sampling, no LSE)")
    ctrl = GlobalMPPI(
        task,
        num_samples=args.num_samples or 1000,
        noise_level=0.2,
        temperature=0.1,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=6,
        iterations=5,
        ksos_solver="ADMM-cuda",
        ksos_num_restart=args.ksos_restarts,
        ksos_sampling=args.ksos_sampling,
        ksos_num_samples=args.ksos_samples,
        mppi_noise=args.mppi_noise,
        seed=args.seed,
        ksos_refine=not args.no_ksos_mppi,
        ksos_lse_samples=args.lse_samples,
        ksos_lse_sigmas=tuple(args.lse_sigmas),
        ksos_lse_lambda=args.lse_lambda,
        ksos_cluster_size=args.ksos_cluster_size,
        ksos_gnc=tuple(args.gnc) if args.gnc else None,
        ksos_warm_start=args.ksos_warm_start,
        ksos_cacs_options=args.cacs,
    )
elif args.algorithm == "globalmppi":
    print(f"Running GlobalMPPI (KSOS solver: {args.ksos_solver})")
    ctrl = GlobalMPPI(
        task,
        num_samples=256,
        noise_level=0.1,
        temperature=0.1,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=6,
        iterations=5,
        ksos_solver=args.ksos_solver,
        ksos_num_restart=args.ksos_restarts,
        wendland_sigma=args.wendland_sigma,
    )

# Run the interactive simulation
start_seed = args.seed
for trial_idx in range(start_seed, start_seed + 1):
    # Define the model used for simulation
    mj_model = task.mj_model
    mj_model.opt.timestep = 0.001
    mj_model.opt.iterations = 100
    mj_model.opt.ls_iterations = 50
    mj_data = mujoco.MjData(mj_model)
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
        current_task="pushT_globalmppi",
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        headless=args.headless,
    )
