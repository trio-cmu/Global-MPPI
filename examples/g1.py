import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

# Headless (no window): MuJoCo needs an offscreen GL backend, selected BEFORE
# mujoco is imported. (Set MUJOCO_GL=osmesa yourself to use the CPU instead.)
if "--headless" in sys.argv:
    os.environ["MUJOCO_GL"] = "egl"

# cuKernelSOS's Python binding (GPU ADMM): the first cuKernelSOS/python in a
# parent folder, unless CUKERNELSOS_PYTHON is already set (use the gpu-solve
# build for --gpu). Must be set before global_mppi is imported (it preloads the
# extension before jax).
for _parent in Path(__file__).resolve().parents:
    if (_parent / "cuKernelSOS" / "python" / "cukernelsos").is_dir():
        os.environ.setdefault("CUKERNELSOS_PYTHON", str(_parent / "cuKernelSOS" / "python"))
        break
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import mujoco
import numpy as np
import global_mppi
from global_mppi.algs import MPPI, PredictiveSampling, DIAL, GlobalMPPI, MTP, CBO
from global_mppi.simulation.deterministic import run_interactive, RUN_TIMESTAMP
from global_mppi.tasks.humanoid_standup import HumanoidStandup

"""
Run a simulation of the Unitree G1 standup task (orientation term with two
components, see global_mppi/tasks/humanoid_standup.py), one folder per run:

    <out-dir>/<ps|mppi|dial|gm>/s<seed>/  cycles.csv, final.json, args.json, ...

Options go before the algorithm, e.g.
    python examples/g1.py --headless --gpu --seed-start 0 --seed-end 1 globalmppi

The model is not in the repo: run script/fetch_g1_model.sh once.
The planner settings are those of examples/cube.py (samples, iterations, noise,
temperature, beta, CACS), except the horizon (0.6 s) and the CACS radius (0.7 * vmax).
"""

REPO = Path(__file__).resolve().parents[1]
TAGS = {"ps": "ps", "mppi": "mppi", "dial": "dial", "globalmppi": "gm", "mtp": "mtp", "cbo": "cbo"}
CONTROL_FREQUENCY_HZ = 50

# Define the task (cost and dynamics)
task = HumanoidStandup()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run a simulation of the G1 standup task. "
    "Options go before the algorithm, e.g. `python g1.py --headless --gpu globalmppi`."
)
parser.add_argument(
    "--headless",
    action="store_true",
    help="no window: render offscreen (MUJOCO_GL=egl) and record a video (default: off)",
)
parser.add_argument("--seed-start", type=int, default=0, help="first seed, inclusive (default: 0)")
parser.add_argument("--seed-end", type=int, default=6, help="last seed, exclusive (default: 6)")
parser.add_argument("--max-cycles", type=int, default=100, help="control cycles per run (default: 100)")
parser.add_argument(
    "--out-dir",
    default=None,
    help="root folder of the outputs; each run goes to <out-dir>/<algo>/s<seed>/ "
    "(default: logs/g1/<timestamp>/ at the repo root)",
)
parser.add_argument(
    "--gpu",
    action="store_true",
    help="GlobalMPPI: CACS sampling + cuKernelSOS ADMM-cuda with this branch's solver settings, 5 restarts, "
    "1000 candidates per restart (needs CUKERNELSOS_PYTHON to point to the cuKernelSOS build)",
)
parser.add_argument(
    "--num-samples",
    type=int,
    default=None,
    help="samples per iteration (default: 2000 for ps / mppi / dial, 1000 for globalmppi)",
)
parser.add_argument("--iterations", type=int, default=5, help="optimization iterations per planning cycle (default: 5)")
parser.add_argument(
    "--noise",
    type=float,
    default=None,
    help="noise_level (default: the cube.py value: ps 0.2, mppi 0.2, dial 1.0, globalmppi 0.1)",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=None,
    help="temperature (default: the cube.py value: mppi 0.001, dial 0.01, globalmppi 0.1)",
)
parser.add_argument("--mppi-noise", type=float, default=0.1, help="globalmppi: noise of the final MPPI step (default: 0.1)")
parser.add_argument(
    "--warm-start",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="globalmppi: restart 0 of CACS also considers the previous cycle's plan (default: on)",
)
parser.add_argument(
    "--cacs-h",
    type=float,
    default=None,
    help="globalmppi: CACS kernel radius h (default: 0.7 * vmax, vmax = largest |actuator ctrlrange|)",
)
subparsers = parser.add_subparsers(dest="algorithm", help="Sampling algorithm (choose one)")
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("dial", help="DIAL")
subparsers.add_parser("mtp", help="Model Tensor Planning (hydrax, upstream defaults)")
subparsers.add_parser("cbo", help="Consensus-Based Optimization (hydrax, upstream example values)")
subparsers.add_parser("globalmppi", help="GlobalMPPI (needs --gpu)")
args = parser.parse_args()
algo = args.algorithm or "ps"
if args.gpu and algo != "globalmppi":
    parser.error("--gpu only applies to the globalmppi algorithm")
if algo == "globalmppi" and not args.gpu:
    parser.error("globalmppi needs --gpu")
# Controller settings: the cube.py values, horizon 0.6 s (the planning step of the
# G1 model is 0.02 s, so 30 steps), 4 knots.
common = dict(plan_horizon=0.6, spline_type="zero", num_knots=4)
# MTP and CBO (ported from hydrax, global_mppi/algs/mtp.py and cbo.py): 2000 samples x 5
# iterations like the other baselines, all other hyperparameters at the hydrax values. For MTP these
# are the defaults of its constructor. CBO has no defaults for four of its arguments; the values are
# those of the only hydrax example that uses it (examples/bugtrap.py at bd53d43).
UPSTREAM_DEFAULTS = {
    "mtp": dict(
        m_pts=3, n_per_layer=50, degree=2, num_elites=5, sigma_start=0.5, sigma_min=0.1,
        sigma_max=1.0, temperature=0.1, beta=0.1, alpha=0.5, mtp_interpolation="akima",
    ),
    "cbo": dict(
        initial_noise_level=1.0, temperature=0.01, consensus_weight=1.0, noise_weight=10.0, step_size=0.1,
    ),
}
default_noise = {"ps": 0.2, "mppi": 0.2, "dial": 1.0, "globalmppi": 0.1}.get(algo)
default_temperature = {"mppi": 0.001, "dial": 0.01, "globalmppi": 0.1}
if args.num_samples is not None:
    num_samples = args.num_samples
else:
    num_samples = 1000 if algo == "globalmppi" else 2000
if algo in UPSTREAM_DEFAULTS:
    cfg = dict(num_samples=num_samples, iterations=args.iterations, **common, **UPSTREAM_DEFAULTS[algo])
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.noise is not None:
        if algo == "cbo":
            cfg["initial_noise_level"] = args.noise
        else:
            parser.error("mtp has no noise_level (its sigma_start / sigma_min / sigma_max are fixed)")
else:
    cfg = dict(
        num_samples=num_samples,
        noise_level=default_noise if args.noise is None else args.noise,
        iterations=args.iterations,
        **common,
    )
    if algo != "ps":
        cfg["temperature"] = default_temperature[algo] if args.temperature is None else args.temperature
    elif args.temperature is not None:
        parser.error("ps has no temperature")

vmax = float(np.abs(task.mj_model.actuator_ctrlrange).max())
cacs_h = None
if algo == "ps":
    print("Running predictive sampling")
    ctrl = PredictiveSampling(task, **cfg)
elif algo == "mppi":
    print("Running MPPI")
    ctrl = MPPI(task, **cfg)
elif algo == "dial":
    print("Running DIAL")
    cfg.update(beta_opt_iter=1.0, beta_horizon=1.0)
    ctrl = DIAL(task, **cfg)
elif algo in ("mtp", "cbo"):
    print(f"Running {algo.upper()} (hydrax hyperparameters): {cfg}")
    ctrl = (MTP if algo == "mtp" else CBO)(task, **cfg)
else:
    print("Running GlobalMPPI (CACS + ADMM-cuda, branch solver settings)")
    # CACS kernel radius: h = 0.7 was tuned for a control range of 1.0, so it is
    # scaled by this model's vmax (as examples/cube.py and examples/closed_loop_test.py).
    cacs_h = 0.7 * vmax if args.cacs_h is None else args.cacs_h
    cfg.update(
        ksos_solver="ADMM-cuda",
        ksos_sampling="cacs",
        ksos_num_restart=5,
        ksos_num_samples=1000,
        ksos_warm_start=args.warm_start,
        mppi_noise=args.mppi_noise,
        ksos_cacs_options={"h": cacs_h},
    )
    ctrl = GlobalMPPI(task, **cfg)

# Record solve paths when the installed cuKernelSOS exposes them.
SOLVE_PATHS = {}
if args.gpu:
    from ksos_tools.solvers import ADMM

    _solve_ksos = ADMM.solve_ksos

    def _solve_ksos_recorded(*a, **kw):
        x, info = _solve_ksos(*a, **kw)
        path = (info.get("admm_stats") or {}).get("solve_path")
        if path is not None:
            SOLVE_PATHS[path] = SOLVE_PATHS.get(path, 0) + 1
        return x, info

    ADMM.solve_ksos = _solve_ksos_recorded

# Where the code comes from: a stale pip install shows up here
provenance = {"global_mppi": global_mppi.__file__}
import importlib.metadata as _md

provenance["mujoco"] = getattr(mujoco, "__version__", None)
try:
    provenance["mujoco_mjx"] = _md.version("mujoco-mjx")
except Exception:  # noqa: BLE001
    provenance["mujoco_mjx"] = None
for _name, _cmd in (
    ("git_commit", ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]),
    ("git_dirty", ["git", "-C", str(REPO), "status", "--porcelain"]),
):
    try:
        _out = subprocess.run(_cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        provenance[_name] = _out if _name == "git_commit" else bool(_out)
    except Exception:  # noqa: BLE001
        provenance[_name] = None
print(f"code: {provenance}", flush=True)


def gpu_state():
    """(utilization %, memory used MiB) of the first GPU; NaN if nvidia-smi fails."""
    try:
        line = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()[0]
        util, mem = (float(x) for x in line.split(","))
        return util, mem
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan")


def process_gpu_mem():
    """GPU memory (MiB) of this process, NaN if it is not listed."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        for line in out.splitlines():
            pid, mem = (x.strip() for x in line.split(","))
            if int(pid) == os.getpid():
                return float(mem)
    except Exception:  # noqa: BLE001
        pass
    return float("nan")


class CycleCsv:
    """One CSV row per control cycle, flushed as soon as it is written.

    Columns: cycle, plan_s (planning time, s), best_cost (the planner's predicted cost of its
    plan), torso_h (height of the imu_in_torso site, m), tilt_deg (angle between the torso z
    axis and the world z axis), gpu_mem_mb (GPU memory of this process), solve_paths
    (globalmppi: cumulative count of ADMM solves per path). Torso height and tilt are of the
    state the cycle planned from.
    """

    HEADER = ["cycle", "plan_s", "best_cost", "torso_h", "tilt_deg", "gpu_mem_mb", "solve_paths"]

    def __init__(self, path, mj_model):
        self.site = mj_model.site("imu_in_torso").id
        self.rows = 0
        self.plan_times = []
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.HEADER)
        self.file.flush()

    def state(self, mj_model, mj_data):
        """(torso height in m, tilt in degrees) of mj_data."""
        mujoco.mj_forward(mj_model, mj_data)
        z = mj_data.site_xmat[self.site].reshape(3, 3)[:, 2]
        return float(mj_data.site_xpos[self.site, 2]), float(np.degrees(np.arccos(np.clip(z[2], -1.0, 1.0))))

    def __call__(self, cycle, plan_time, best_cost, mj_model, mj_data):
        height, tilt = self.state(mj_model, mj_data)
        self.plan_times.append(plan_time)
        cost = float(np.asarray(best_cost).ravel()[0])
        self.writer.writerow(
            [cycle, f"{plan_time:.3f}", f"{cost:.4f}", f"{height:.4f}", f"{tilt:.2f}", process_gpu_mem(),
             json.dumps(SOLVE_PATHS) if args.gpu else ""]
        )
        self.file.flush()
        self.rows += 1

    def close(self):
        self.file.close()


out_root = Path(args.out_dir) if args.out_dir else REPO / "logs" / "g1" / RUN_TIMESTAMP

# Run the simulation. The planner uses the model as it is (planning step 0.02 s);
# the simulated robot gets a smaller step and stiffer contacts. The robot starts
# from the "stand" keyframe, rotated to lie on its side (it has to stand up).
for trial_idx in range(args.seed_start, args.seed_end):
    mj_model = mujoco.MjModel.from_xml_path(str(Path(global_mppi.ROOT) / "models" / "g1" / "scene.xml"))   # a fresh copy for the simulation
    mj_model.opt.timestep = 0.01
    mj_model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
    mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = mj_model.keyframe("stand").qpos
    mj_data.qpos[3:7] = [0.7, 0.0, -0.7, 0.0]

    run_dir = out_root / TAGS[algo] / f"s{trial_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    gpu_util, gpu_mem = gpu_state()
    with open(run_dir / "args.json", "w") as f:
        json.dump(
            {
                "task_version": "g1_fixed",
                "task_file": "global_mppi/tasks/humanoid_standup.py",
                "algorithm": algo,
                "seed": trial_idx,
                "max_cycles": args.max_cycles,
                "control_frequency_hz": CONTROL_FREQUENCY_HZ,
                "command_line_args": vars(args),
                "controller": {"class": type(ctrl).__name__, **cfg},
                "vmax": vmax,
                "cacs_h": cacs_h,
                "nu": int(task.mj_model.nu),
                "control_dim": int(task.mj_model.nu) * cfg["num_knots"],
                "planning_dt": task.dt,
                "sim_model": {"timestep": 0.01, "o_solimp": [0.9, 0.95, 0.001, 0.5, 2], "enableflags": "mjENBL_OVERRIDE"},
                "initial_state": "keyframe stand with qpos[3:7] = [0.7, 0, -0.7, 0]",
                "cukernelsos_python": os.environ.get("CUKERNELSOS_PYTHON"),
                "gpu_before_run": {"util_percent": gpu_util, "mem_used_mib": gpu_mem},
                "code": provenance,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n=== Seed {trial_idx} ===")
    if hasattr(ctrl, "_cacs_rng"):
        # run_interactive resets the policy params (rng key, mean) per seed, but
        # GlobalMPPI keeps the CACS random stream on the controller object: reset
        # it too, so that a seed gives the same result alone or in a sequence.
        ctrl._cacs_rng = np.random.default_rng(trial_idx + 101)
        ctrl._cacs_calls = 0
    SOLVE_PATHS.clear()
    cycle_csv = CycleCsv(str(run_dir / "cycles.csv"), mj_model)
    run_interactive(
        ctrl,
        mj_model,
        mj_data,
        frequency=CONTROL_FREQUENCY_HZ,
        show_traces=False,
        max_cycles=args.max_cycles,
        current_task="g1_fixed",
        current_seed=trial_idx,
        headless=args.headless,
        out_dir=str(run_dir),
        cycle_logger=cycle_csv,
    )
    # State when the run ends (the last cycle of cycles.csv is the state it
    # planned from, one control step earlier)
    height, tilt = cycle_csv.state(mj_model, mj_data)
    plans = cycle_csv.plan_times
    final = {
        "final_torso_h": height,
        "final_tilt_deg": tilt,
        "sim_time": float(mj_data.time),
        "cycles": cycle_csv.rows,
        "mean_plan_s_excl_cycle0": float(np.mean(plans[1:])) if len(plans) > 1 else None,
    }
    if args.gpu:
        final["solve_paths"] = dict(SOLVE_PATHS)
    with open(run_dir / "final.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"== final torso {height:.3f} m, tilt {tilt:.1f} deg ({cycle_csv.rows} cycles)")
    if args.gpu:
        print(f"solve_paths: {dict(SOLVE_PATHS)}")
    cycle_csv.close()
