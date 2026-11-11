import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Headless (no window): MuJoCo needs an offscreen GL backend, selected BEFORE
# mujoco is imported. (Set MUJOCO_GL=osmesa yourself to use the CPU instead.)
if "--headless" in sys.argv:
    os.environ["MUJOCO_GL"] = "egl"

# cuKernelSOS's Python binding (GPU ADMM): the first cuKernelSOS/python in a
# parent folder. Must be set before global_mppi is imported (it preloads the
# extension before jax).
for _parent in Path(__file__).resolve().parents:
    if (_parent / "cuKernelSOS" / "python" / "cukernelsos").is_dir():
        os.environ.setdefault("CUKERNELSOS_PYTHON", str(_parent / "cuKernelSOS" / "python"))
        break

import mujoco
import numpy as np
from global_mppi.algs import MPPI, PredictiveSampling, DIAL, GlobalMPPI
from global_mppi.simulation.deterministic import run_interactive, RUN_TIMESTAMP
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
    description="Run an interactive simulation of the cube rotation task. "
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
    "--out-dir",
    default=None,
    help="folder for the outputs (per-seed CSV, .npy histories, video); "
    "default: logs/cube_nomppi_<algorithm>/<timestamp>/",
)
parser.add_argument(
    "--gpu",
    action="store_true",
    help="GlobalMPPI only: solve KSOS with the GPU ADMM of cuKernelSOS "
    "(ksos_solver=ADMM-cuda, gpu_solve=True, CACS sampling, 5 restarts, 1000 "
    "candidates per restart). Default: off, the CPU Newton solver",
)
parser.add_argument(
    "--num-samples",
    type=int,
    default=None,
    help="samples per iteration (default: 2000 for ps / mppi / dial; globalmppi: "
    "1000 with --gpu, 128 without)",
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
    "(default: 0.1)",
)
parser.add_argument(
    "--warm-start",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="globalmppi --gpu: restart 0 of CACS also considers the previous "
    "cycle's plan (default: on)",
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
if args.gpu and args.algorithm != "globalmppi":
    parser.error("--gpu only applies to the globalmppi algorithm")


# Controller settings. horizon / knots / spline are the cube.py values; the
# sample counts and iterations follow the PushT simulation setup and can be
# changed with the options above.
common = dict(plan_horizon=0.25, spline_type="zero", num_knots=4)
algo = args.algorithm or "ps"
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
    print("Running GlobalMPPI")
    if args.gpu:
        # CACS kernel radius: h = 0.7 was tuned for a control range of 1.0, so
        # it is scaled by this model's vmax (as examples/closed_loop_test.py).
        # The trust-region box is the task's u_min / u_max (the actuator
        # ctrlrange) times decay_rate ** restart, so it follows the ranges by
        # itself.
        cfg.update(
            ksos_solver="ADMM-cuda",
            ksos_sampling="cacs",
            ksos_num_restart=5,
            ksos_num_samples=1000,
            ksos_warm_start=args.warm_start,
            mppi_noise=args.mppi_noise,
            ksos_cacs_options={
                "h": 0.7 * vmax if args.cacs_h is None else args.cacs_h
            },
        )
    ctrl = GlobalMPPI(task, **cfg)
    if args.gpu:
        ctrl.admm_cuda_options["cuda_options"]["gpu_solve"] = True
        cfg["gpu_solve"] = True

# Output folder, and the full settings of this run in it
out_dir = args.out_dir or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs",
    f"cube_nomppi_{ctrl.ctrl_name}",
    RUN_TIMESTAMP,
)
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "args.json"), "w") as f:
    json.dump(
        {
            "algorithm": algo,
            "command_line_args": vars(args),
            "controller": {"class": type(ctrl).__name__, **cfg},
            "vmax": vmax,
            "control_frequency_hz": 25,
            "max_cycles": 100,
            "cukernelsos_python": os.environ.get("CUKERNELSOS_PYTHON"),
        },
        f,
        indent=2,
        default=str,
    )


class CycleCsv:
    """One CSV row per control cycle, flushed as soon as it is written.

    Columns: cycle, plan_time (s), best_cost, pos_err_mm (norm of the
    cube_position sensor, the cube's offset from its target), ori_err_deg
    (rotation angle between the cube_orientation sensor and the goal
    orientation [1, 0, 0, 0]). The errors are of the state the cycle planned
    from. They are read from a scratch MjData, so the simulation is untouched.
    """

    def __init__(self, path, mj_model):
        self.scratch = mujoco.MjData(mj_model)
        sid = lambda n: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, n)
        self.pos_adr = mj_model.sensor_adr[sid("cube_position")]
        self.ori_adr = mj_model.sensor_adr[sid("cube_orientation")]
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["cycle", "plan_time", "best_cost", "pos_err_mm", "ori_err_deg"])
        self.file.flush()

    def __call__(self, cycle, plan_time, best_cost, mj_model, mj_data):
        d = self.scratch
        d.qpos[:], d.qvel[:], d.time = mj_data.qpos, mj_data.qvel, mj_data.time
        d.mocap_pos[:], d.mocap_quat[:] = mj_data.mocap_pos, mj_data.mocap_quat
        mujoco.mj_forward(mj_model, d)
        pos_mm = 1000.0 * float(np.linalg.norm(d.sensordata[self.pos_adr : self.pos_adr + 3]))
        quat = np.array(d.sensordata[self.ori_adr : self.ori_adr + 4])
        w = min(abs(quat[0]) / np.linalg.norm(quat), 1.0)
        ori_deg = float(np.degrees(2.0 * np.arccos(w)))
        cost = float(np.asarray(best_cost).ravel()[0])
        self.writer.writerow([cycle, f"{plan_time:.6f}", f"{cost:.6f}", f"{pos_mm:.4f}", f"{ori_deg:.4f}"])
        self.file.flush()

    def close(self):
        self.file.close()


# Define the model used for simulation
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)

# Run the interactive simulation
for trial_idx in range(args.seed_start, args.seed_end):
    mj_model = task.mj_model
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
    cycle_csv = CycleCsv(
        os.path.join(out_dir, f"cube_nomppi_{ctrl.ctrl_name}_cycles_seed{trial_idx}.csv"),
        mj_model,
    )
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
        headless=args.headless,
        out_dir=out_dir,
        cycle_logger=cycle_csv,
    )
    cycle_csv.close()
