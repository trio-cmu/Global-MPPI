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
# parent folder, unless CUKERNELSOS_PYTHON is already set (the real-robot runs
# set it to cuKernelSOS_gpu/python, which has the gpu_solve option). Must be set
# before global_mppi is imported (it preloads the extension before jax).
for _parent in Path(__file__).resolve().parents:
    if (_parent / "cuKernelSOS" / "python" / "cukernelsos").is_dir():
        os.environ.setdefault("CUKERNELSOS_PYTHON", str(_parent / "cuKernelSOS" / "python"))
        break

import mujoco
import numpy as np
import global_mppi
from global_mppi.algs import MPPI, PredictiveSampling, DIAL, GlobalMPPI
from global_mppi.simulation.deterministic import run_interactive, RUN_TIMESTAMP
from global_mppi.tasks.cube import CubeRotation

"""
Run a simulation of the cube rotation task (LEAP hand), one folder per run:

    <out-dir>/<ps|mppi|dial|gm>/s<seed>/  cycles.csv, final.json, args.json, ...

Options go before the algorithm, e.g.
    python examples/cube.py --headless --gpu --seed-start 0 --seed-end 1 globalmppi

With a window, double click on the floating target cube, then change the goal
orientation with [ctrl + left click].
"""

REPO = Path(__file__).resolve().parents[1]
TAGS = {"ps": "ps", "mppi": "mppi", "dial": "dial", "globalmppi": "gm"}

# Define the task (cost and dynamics)
task = CubeRotation()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run a simulation of the cube rotation task. "
    "Options go before the algorithm, e.g. `python cube.py --headless --gpu globalmppi`."
)
parser.add_argument(
    "--headless",
    action="store_true",
    help="no window: render offscreen (MUJOCO_GL=egl) and record a video (default: off)",
)
parser.add_argument(
    "--seed-start", type=int, default=0, help="first seed, inclusive (default: 0)"
)
parser.add_argument(
    "--seed-end", type=int, default=6, help="last seed, exclusive (default: 6)"
)
parser.add_argument(
    "--max-cycles", type=int, default=100, help="control cycles per run (default: 100)"
)
parser.add_argument(
    "--out-dir",
    default=None,
    help="root folder of the outputs; each run goes to <out-dir>/<algo>/s<seed>/ "
    "(default: logs/cube/<timestamp>/ at the repo root)",
)
parser.add_argument(
    "--gpu",
    action="store_true",
    help="GlobalMPPI: the new algorithm (CACS sampling + cuKernelSOS ADMM-cuda with "
    "gpu_solve, 5 restarts, 1000 candidates per restart), as in the PushT real runs",
)
parser.add_argument(
    "--legacy-newton",
    action="store_true",
    help="GlobalMPPI: the old CPU Newton version of the first paper (128 samples)",
)
parser.add_argument(
    "--num-samples",
    type=int,
    default=None,
    help="samples per iteration (default: 2000 for ps / mppi / dial; globalmppi: "
    "1000 with --gpu, 128 with --legacy-newton)",
)
parser.add_argument(
    "--iterations",
    type=int,
    default=5,
    help="optimization iterations per planning cycle (default: 5)",
)
parser.add_argument(
    "--noise",
    type=float,
    default=None,
    help="noise_level (default: the cube.py value of the algorithm: ps 0.2, "
    "mppi 0.2, dial 1.0, globalmppi 0.1)",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=None,
    help="temperature (default: the cube.py value: mppi 0.001, dial 0.01, "
    "globalmppi 0.1; ps has none)",
)
parser.add_argument(
    "--mppi-noise",
    type=float,
    default=0.1,
    help="globalmppi --gpu: noise of the final MPPI step after the KSOS restarts "
    "(default: 0.1, the simulation benchmark value; the PushT real runs used 0.02)",
)
parser.add_argument(
    "--warm-start",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="globalmppi --gpu: restart 0 of CACS also considers the previous "
    "cycle's plan (default: on, as in the simulation benchmark; the PushT real "
    "runs had it off; --no-warm-start turns it off)",
)
parser.add_argument(
    "--cacs-h",
    type=float,
    default=None,
    help="globalmppi --gpu: CACS kernel radius h (default: 0.7 * vmax, vmax = "
    "largest |actuator ctrlrange| of the model, as examples/closed_loop_test.py)",
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("dial", help="DIAL")
subparsers.add_parser("globalmppi", help="GlobalMPPI")
args = parser.parse_args()
algo = args.algorithm or "ps"
if args.gpu and algo != "globalmppi":
    parser.error("--gpu only applies to the globalmppi algorithm")
if args.legacy_newton and algo != "globalmppi":
    parser.error("--legacy-newton only applies to the globalmppi algorithm")
if args.gpu and args.legacy_newton:
    parser.error("--gpu and --legacy-newton are exclusive")
if algo == "globalmppi" and not (args.gpu or args.legacy_newton):
    parser.error("globalmppi needs --gpu (new algorithm) or --legacy-newton (old one)")


# Controller settings. horizon / knots / spline are the cube.py values; the
# sample counts and iterations follow the PushT setup and can be changed with
# the options above.
common = dict(plan_horizon=0.25, spline_type="zero", num_knots=4)
default_noise = {"ps": 0.2, "mppi": 0.2, "dial": 1.0, "globalmppi": 0.1}[algo]
default_temperature = {"mppi": 0.001, "dial": 0.01, "globalmppi": 0.1}
if args.num_samples is not None:
    num_samples = args.num_samples
elif algo == "globalmppi":
    num_samples = 1000 if args.gpu else 128
else:
    num_samples = 2000
cfg = dict(
    num_samples=num_samples,
    noise_level=default_noise if args.noise is None else args.noise,
    iterations=args.iterations,
    **common,
)
if algo != "ps":
    cfg["temperature"] = (
        default_temperature[algo] if args.temperature is None else args.temperature
    )
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
else:
    if args.gpu:
        print("Running GlobalMPPI (CACS + ADMM-cuda, gpu_solve)")
        # CACS kernel radius: h = 0.7 was tuned for a control range of 1.0, so
        # it is scaled by this model's vmax (as examples/closed_loop_test.py).
        # The trust-region box is the task's u_min / u_max (the actuator
        # ctrlrange) times decay_rate ** restart, so it follows the ranges by
        # itself.
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
    else:
        print("Running GlobalMPPI (legacy CPU Newton version)")
    ctrl = GlobalMPPI(task, **cfg)
    if args.gpu:
        ctrl.admm_cuda_options["cuda_options"]["gpu_solve"] = True
        cfg["gpu_solve"] = True

# solve_path of every cuKernelSOS ADMM solve (as examples/closed_loop_test.py): "gpu", or "cpu (reason)"
# when gpu_solve was requested but not used. Counted per run.
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
try:
    import ksos_tools

    provenance["ksos_tools"] = ksos_tools.__file__
except Exception as exc:  # noqa: BLE001
    provenance["ksos_tools"] = f"import failed: {exc}"
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
    """(utilization %, memory used MiB) of the first GPU; NaN if nvidia-smi fails.

    It includes this process, but it is read right after planning, so it mostly
    shows what other users are running on the shared GPU.
    """
    try:
        line = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()[0]
        util, mem = (float(x) for x in line.split(","))
        return util, mem
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan")


class CycleCsv:
    """One CSV row per control cycle, flushed as soon as it is written.

    Columns: cycle, plan_s (planning time, s), best_cost, pos_err_mm (norm of
    the cube_position sensor: the cube's offset from its target), ang_err_deg
    (rotation angle between the cube_orientation sensor and the goal
    orientation [1, 0, 0, 0]: 2 * arccos |q_w|), gpu_util (%) and gpu_mem_mb
    after planning. The errors are of the state the cycle planned from. They
    are read from a scratch MjData, so the simulation is untouched.
    """

    HEADER = ["cycle", "plan_s", "best_cost", "pos_err_mm", "ang_err_deg", "gpu_util", "gpu_mem_mb"]

    def __init__(self, path, mj_model):
        self.scratch = mujoco.MjData(mj_model)
        sid = lambda n: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, n)
        self.pos_adr = mj_model.sensor_adr[sid("cube_position")]
        self.ori_adr = mj_model.sensor_adr[sid("cube_orientation")]
        self.rows = 0
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.HEADER)
        self.file.flush()

    def errors(self, mj_model, mj_data):
        """(position error in mm, orientation error in degrees) of mj_data."""
        d = self.scratch
        d.qpos[:], d.qvel[:], d.time = mj_data.qpos, mj_data.qvel, mj_data.time
        d.mocap_pos[:], d.mocap_quat[:] = mj_data.mocap_pos, mj_data.mocap_quat
        mujoco.mj_forward(mj_model, d)
        pos_mm = 1000.0 * float(np.linalg.norm(d.sensordata[self.pos_adr : self.pos_adr + 3]))
        quat = np.array(d.sensordata[self.ori_adr : self.ori_adr + 4])
        w = min(abs(quat[0]) / np.linalg.norm(quat), 1.0)
        return pos_mm, float(np.degrees(2.0 * np.arccos(w)))

    def __call__(self, cycle, plan_time, best_cost, mj_model, mj_data):
        pos_mm, ang_deg = self.errors(mj_model, mj_data)
        util, mem = gpu_state()
        cost = float(np.asarray(best_cost).ravel()[0])
        self.writer.writerow(
            [cycle, f"{plan_time:.6f}", f"{cost:.6f}", f"{pos_mm:.4f}", f"{ang_deg:.4f}", util, mem]
        )
        self.file.flush()
        self.rows += 1

    def close(self):
        self.file.close()


out_root = Path(args.out_dir) if args.out_dir else REPO / "logs" / "cube" / RUN_TIMESTAMP

# Run the simulation. The initial state is the model's default (the hand at its
# reference pose, the cube in the palm rotated by 90 deg about z).
for trial_idx in range(args.seed_start, args.seed_end):
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    run_dir = out_root / TAGS[algo] / f"s{trial_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    gpu_util, gpu_mem = gpu_state()
    with open(run_dir / "args.json", "w") as f:
        json.dump(
            {
                "algorithm": algo,
                "seed": trial_idx,
                "command_line_args": vars(args),
                "controller": {"class": type(ctrl).__name__, **cfg},
                "vmax": vmax,
                "cacs_h": cacs_h,
                "control_frequency_hz": 25,
                "max_cycles": args.max_cycles,
                "cukernelsos_python": os.environ.get("CUKERNELSOS_PYTHON"),
                "ksos_gpu_solve_env": os.environ.get("KSOS_GPU_SOLVE"),
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
        frequency=25,
        fixed_camera_id=None,
        show_traces=False,
        max_traces=1,
        trace_color=[1.0, 1.0, 1.0, 1.0],
        max_cycles=args.max_cycles,
        current_task="cube_nomppi",
        current_seed=trial_idx,
        headless=args.headless,
        out_dir=str(run_dir),
        cycle_logger=cycle_csv,
    )
    # State when the run ends (the last cycle of cycles.csv is the state it
    # planned from, one control step earlier)
    pos_mm, ang_deg = cycle_csv.errors(mj_model, mj_data)
    final = {"pos_err_mm": pos_mm, "ang_err_deg": ang_deg, "cycles": cycle_csv.rows}
    if args.gpu:
        final["solve_paths"] = dict(SOLVE_PATHS)
    with open(run_dir / "final.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"== final cube error: {pos_mm:.2f} mm, {ang_deg:.2f} deg ({cycle_csv.rows} cycles)")
    if args.gpu:
        print(f"solve_paths: {dict(SOLVE_PATHS)}")
    cycle_csv.close()
