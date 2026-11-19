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

from copy import deepcopy

import global_mppi  # Preload cuKernelSOS before JAX.
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from global_mppi.algs import MPPI, PredictiveSampling, DIAL, GlobalMPPI, MTP, CBO
from global_mppi.simulation.deterministic import run_interactive, RUN_TIMESTAMP
from global_mppi.tasks.humanoid_mocap import HumanoidMocap

"""
Run a simulation of the G1 walk task (hydrax HumanoidMocap, unchanged): the Unitree G1 (23 actuated joints,
models/g1/scene_23dof.xml) tracks the retargeted motion-capture walk Lafan1/mocap/UnitreeG1/walk1_subject1.npz of the
LocoMuJoCo dataset (downloaded from Hugging Face on first use: robfiras/loco-mujoco-datasets, about 2.4 MB, CC-BY-NC-ND 4.0).
The reference advances with the simulation time (frame = int(time * fps)); the cost is the tracking cost of
global_mppi/tasks/humanoid_mocap.py. One folder per run:

    <out-dir>/<ps|mppi|dial|mtp|cbo|gm>/s<seed>/  cycles.csv, final.json, args.json, ...

Options go before the algorithm, e.g.
    python examples/g1_walk.py --headless --gpu --seed-start 0 --seed-end 1 globalmppi

The model is not in the repo: run script/fetch_g1_model.sh once.
The planner settings are those of examples/cube.py (samples, iterations, noise, temperature, beta, CACS), except
the horizon (0.8 s, 4 knots, zero-order hold, as the hydrax example) and the CACS radius (0.7 * vmax, vmax = the
largest |actuator ctrlrange| of the model = 3.09). As the hydrax example, the simulated robot has timestep 0.01,
10 solver iterations (50 line-search iterations) and soft contacts, starts at reference frame 0 and the first
knots are the first reference joint positions; the planner uses the model's 0.02 s step.

Run length: 400 cycles (8 s). The reference stands still for about 3.6 s, its first step ends at 4.22 s and the steps then
last about 0.75 s each, so 100 cycles would only cover the standing phase (see docs/g1_walk.md).
"""

REPO = Path(__file__).resolve().parents[1]
TAGS = {"ps": "ps", "mppi": "mppi", "dial": "dial", "globalmppi": "gm", "mtp": "mtp", "cbo": "cbo"}
CONTROL_FREQUENCY_HZ = 50
FALL_HEIGHT = 0.5   # m: a run 'fell' if the base height was below this at any cycle (the standing pelvis is at about 0.8 m)

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run a simulation of the G1 walk (mocap tracking) task. "
    "Options go before the algorithm, e.g. `python g1_walk.py --headless --gpu globalmppi`."
)
parser.add_argument(
    "--headless",
    action="store_true",
    help="no window: render offscreen (MUJOCO_GL=egl) and record a video (default: off)",
)
parser.add_argument("--seed-start", type=int, default=0, help="first seed, inclusive (default: 0)")
parser.add_argument("--seed-end", type=int, default=6, help="last seed, exclusive (default: 6)")
parser.add_argument("--max-cycles", type=int, default=400, help="control cycles per run (default: 400 = 8 s at 50 Hz, about 5 steps of the reference)")
parser.add_argument(
    "--reference-filename",
    default="Lafan1/mocap/UnitreeG1/walk1_subject1.npz",
    help="reference motion in robfiras/loco-mujoco-datasets (default: %(default)s, as the hydrax example)",
)
parser.add_argument(
    "--out-dir",
    default=None,
    help="root folder of the outputs; each run goes to <out-dir>/<algo>/s<seed>/ "
    "(default: logs/g1_walk/<timestamp>/ at the repo root)",
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
# Define the task (cost and dynamics); downloads the reference motion on first use
task = HumanoidMocap(reference_filename=args.reference_filename)
# Controller settings: the cube.py values, horizon 0.8 s, 4 knots, zero spline as the hydrax example.
common = dict(plan_horizon=0.8, spline_type="zero", num_knots=4)
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

# vmax as in cube.py: the largest |actuator ctrlrange|. The 23 position actuators take joint angles in
# [-3.09, 2.97] rad, so vmax = 3.09 and CACS h = 0.7 * vmax (as examples/g1.py).
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

    Columns: cycle, plan_s (planning time, s), best_cost (the planner's predicted cost of its plan), x, y, z
    (position of the base, m), tilt_deg (angle between the base z axis and the vertical, degrees; 0 = upright),
    ref_base_err (distance of the base to the reference base position at this time, m), ref_joint_err (2-norm of
    the joint-angle error to the reference at this time, rad), track_cost (the task's tracking cost of the real state,
    i.e. its running cost at zero control, between 0 and 1; 0 = on the reference), gpu_mem_mb (GPU memory of this process),
    solve_paths (globalmppi: cumulative count of ADMM solves per path). The state is the one the cycle planned from.
    """

    HEADER = ["cycle", "plan_s", "best_cost", "x", "y", "z", "tilt_deg", "ref_base_err", "ref_joint_err", "track_cost", "gpu_mem_mb", "solve_paths"]

    def __init__(self, path):
        self.ref = np.asarray(task.reference_qpos)
        self.fps = float(task.reference_fps)
        self.min_z = float("inf")

        @jax.jit
        def track(qpos, qvel, time):
            d = mjx.make_data(task.model).replace(qpos=qpos, qvel=qvel, time=time)
            return task.running_cost(mjx.forward(task.model, d), jnp.zeros(task.model.nu))

        self._track = track
        self.rows = 0
        self.plan_times = []
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.HEADER)
        self.file.flush()

    def state(self, mj_data):
        """(x, y, z, tilt deg, base error to the reference m, joint error rad, tracking cost) of mj_data."""
        q = np.asarray(mj_data.qpos)
        qref = self.ref[int(np.clip(int(mj_data.time * self.fps), 0, len(self.ref) - 1))]
        qw, qx, qy, qz = q[3:7] / np.linalg.norm(q[3:7])
        tilt = float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0))))
        cost = float(self._track(jnp.asarray(q), jnp.asarray(mj_data.qvel), jnp.asarray(mj_data.time)))
        return float(q[0]), float(q[1]), float(q[2]), tilt, float(np.linalg.norm(q[:3] - qref[:3])), float(np.linalg.norm(q[7:] - qref[7:])), cost

    def __call__(self, cycle, plan_time, best_cost, mj_model, mj_data):
        x, y, z, tilt, eb, ej, tc = self.state(mj_data)
        self.min_z = min(self.min_z, z)
        self.plan_times.append(plan_time)
        cost = float(np.asarray(best_cost).ravel()[0])
        self.writer.writerow(
            [cycle, f"{plan_time:.3f}", f"{cost:.4f}", f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", f"{tilt:.2f}", f"{eb:.4f}", f"{ej:.4f}", f"{tc:.5f}",
             process_gpu_mem(), json.dumps(SOLVE_PATHS) if args.gpu else ""]
        )
        self.file.flush()
        self.rows += 1

    def close(self):
        self.file.close()


out_root = Path(args.out_dir) if args.out_dir else REPO / "logs" / "g1_walk" / RUN_TIMESTAMP

# Run the simulation. As the hydrax example (examples/humanoid_mocap.py): the simulated robot has timestep 0.01,
# 10 solver iterations, 50 line-search iterations and the soft-constraint override; it starts at reference frame 0
# and the first knots are the first reference joint positions (the planner keeps the model's 0.02 s).
SIM_MODEL = {"timestep": 0.01, "iterations": 10, "ls_iterations": 50, "o_solimp": [0.9, 0.95, 0.001, 0.5, 2], "enableflags": "mjENBL_OVERRIDE"}
for trial_idx in range(args.seed_start, args.seed_end):
    mj_model = deepcopy(task.mj_model)
    mj_model.opt.timestep = SIM_MODEL["timestep"]
    mj_model.opt.iterations = SIM_MODEL["iterations"]
    mj_model.opt.ls_iterations = SIM_MODEL["ls_iterations"]
    mj_model.opt.o_solimp = SIM_MODEL["o_solimp"]
    mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = np.asarray(task.reference_qpos[0])
    initial_knots = task.reference_qpos[: cfg["num_knots"], 7:]

    run_dir = out_root / TAGS[algo] / f"s{trial_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    gpu_util, gpu_mem = gpu_state()
    with open(run_dir / "args.json", "w") as f:
        json.dump(
            {
                "task": "g1_walk",
                "task_file": "global_mppi/tasks/humanoid_mocap.py",
                "algorithm": algo,
                "seed": trial_idx,
                "max_cycles": args.max_cycles,
                "control_frequency_hz": CONTROL_FREQUENCY_HZ,
                "command_line_args": vars(args),
                "controller": {"class": type(ctrl).__name__, **cfg},
                "vmax": vmax,
                "vmax_rule": "largest |actuator ctrlrange| of the model (joint-angle targets in rad)",
                "cacs_h": cacs_h,
                "nq": int(task.mj_model.nq),
                "nv": int(task.mj_model.nv),
                "nu": int(task.mj_model.nu),
                "control_dim": int(task.mj_model.nu) * cfg["num_knots"],
                "planning_dt": task.dt,
                "sim_model": SIM_MODEL,
                "reference": {
                    "file": args.reference_filename,
                    "source": "https://huggingface.co/datasets/robfiras/loco-mujoco-datasets (CC-BY-NC-ND 4.0)",
                    "fps": float(task.reference_fps),
                    "frames": int(task.reference_qpos.shape[0]),
                },
                "initial_state": "reference frame 0; initial knots = first reference joint positions",
                "fall_height_m": FALL_HEIGHT,
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
    cycle_csv = CycleCsv(str(run_dir / "cycles.csv"))
    run_interactive(
        ctrl,
        mj_model,
        mj_data,
        frequency=CONTROL_FREQUENCY_HZ,
        initial_knots=initial_knots,
        show_traces=False,
        max_traces=1,
        max_cycles=args.max_cycles,
        current_task="g1_walk",
        current_seed=trial_idx,
        headless=args.headless,
        out_dir=str(run_dir),
        cycle_logger=cycle_csv,
    )
    # State when the run ends (the last cycle of cycles.csv is the state it planned from)
    x, y, z, tilt, eb, ej, tc = cycle_csv.state(mj_data)
    plans = cycle_csv.plan_times
    base_errs = [float(r[7]) for r in csv.reader(open(run_dir / "cycles.csv")) if r[0] != "cycle"]
    track_costs = [float(r[9]) for r in csv.reader(open(run_dir / "cycles.csv")) if r[0] != "cycle"]
    min_z = min(cycle_csv.min_z, z)
    final = {
        "final_x": x,
        "final_y": y,
        "final_z": z,
        "final_tilt_deg": tilt,
        "final_ref_base_err": eb,
        "final_ref_joint_err": ej,
        "final_track_cost": tc,
        "mean_ref_base_err": float(np.mean(base_errs)),
        "mean_track_cost": float(np.mean(track_costs)),
        "min_base_z": min_z,
        "fell": bool(min_z < FALL_HEIGHT),
        "sim_time": float(mj_data.time),
        "cycles": cycle_csv.rows,
        "mean_plan_s_excl_cycle0": float(np.mean(plans[1:])) if len(plans) > 1 else None,
    }
    if args.gpu:
        final["solve_paths"] = dict(SOLVE_PATHS)
    with open(run_dir / "final.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"== final base height {z:.3f} m, tilt {tilt:.1f} deg, base error to the reference {eb:.3f} m, min base height {min_z:.3f} m, fell {final['fell']} ({cycle_csv.rows} cycles)")
    if args.gpu:
        print(f"solve_paths: {dict(SOLVE_PATHS)}")
    cycle_csv.close()
