"""Hyperparameter sweep of GlobalMPPI with CACS (the closed-loop configuration) on the current PushT cost.

One planning cycle from pusht.py's start state, run exactly as global_mppi/simulation/deterministic.py runs a
cycle with the controller of
    pusht.py --ksos-solver ADMM-cuda --ksos-sampling cacs --ksos-restarts 5 --ksos-warm-start --mppi-noise 0.1
i.e. for each restart: ksos_rollout (CACS sampling + rollouts, kernel radius h * 0.8^restart), solve_ksos
(GPU ADMM, guard); then MPPI (1000 x 5) around the result. The score is the cost of the resulting plan (the
"Best cost" the closed loop logs) and the planning time. One setting changes at a time from the defaults
(CacsConfig, mppi_noise 0.1, 5 restarts); every setting runs the same seeds. The first seed of each setting is
run twice (JIT compiles on the first), timings come from the later runs.

    python pusht_cacs_param_sweep.py --seeds 0 1 2 3 4 --json pusht_cacs_param_sweep.json
    python pusht_cacs_param_sweep.py --only h k --seeds 0 1      # a subset of the parameters
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _parent in Path(__file__).resolve().parents:  # cuKernelSOS python binding, as pusht.py
    if (_parent / "cuKernelSOS" / "python" / "cukernelsos").is_dir():
        os.environ.setdefault("CUKERNELSOS_PYTHON", str(_parent / "cuKernelSOS" / "python"))
        break

import jax
import mujoco
import numpy as np
from mujoco import mjx

from global_mppi.algs import GlobalMPPI
from global_mppi.tasks.pusht import PushT

parser = argparse.ArgumentParser(description="CACS hyperparameter sweep on PushT (one planning cycle)")
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
parser.add_argument("--only", nargs="*", default=None, help="sweep only these parameters (and the defaults)")
parser.add_argument("--json", default="pusht_cacs_param_sweep.json")
args = parser.parse_args()

# one at a time around the closed-loop defaults (CacsConfig: n 1000, initial 200, reuse 40, h 0.7,
# max_clique 20, batch 768, k 512, global_every 125; GlobalMPPI: 5 restarts, MPPI noise 0.1)
SWEEP = {
    "h": [0.5, 0.9, 1.2],
    "k": [256, 768],
    "batch": [512, 1536],
    "initial": [100, 400],
    "reuse": [20, 80],
    "max_clique": [10, 30],
    "n": [500, 1500],
    "global_every": [60, 250],
    "mppi_noise": [0.05, 0.2],
    "restarts": [3, 8],
}
if args.only:
    SWEEP = {k: v for k, v in SWEEP.items() if k in args.only}
SETTINGS = [("default", None)] + [(p, v) for p, vals in SWEEP.items() for v in vals]

task = PushT()
mj_model = task.mj_model
mj_model.opt.timestep = 0.001
mj_model.opt.iterations = 100
mj_model.opt.ls_iterations = 50
mj_data = mujoco.MjData(mj_model)
mj_data.qpos = [0.1, 0.1, 1.3, 0.0, 0.0]
mj_data.qvel[:] = 0.0
mujoco.mj_forward(mj_model, mj_data)
state = mjx.put_data(mj_model, mj_data).replace(mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat)


class Quiet:
    def __enter__(self):
        self.out, sys.stdout = sys.stdout, open(os.devnull, "w")

    def __exit__(self, *exc):
        sys.stdout.close()
        sys.stdout = self.out


def controller(param, value):
    """pusht.py's ADMM-cuda GlobalMPPI with --ksos-sampling cacs --ksos-restarts 5 --ksos-warm-start --mppi-noise 0.1."""
    cacs, mppi_noise, restarts, n = {}, 0.1, 5, 1000
    if param == "mppi_noise":
        mppi_noise = value
    elif param == "restarts":
        restarts = value
    elif param == "n":
        n = value
    elif param != "default":
        cacs[param] = value
    return GlobalMPPI(
        task, num_samples=1000, noise_level=0.2, temperature=0.1, spline_type="cubic", plan_horizon=1.0,
        num_knots=6, iterations=5, ksos_solver="ADMM-cuda", ksos_num_restart=restarts, ksos_sampling="cacs",
        ksos_num_samples=n, mppi_noise=mppi_noise, seed=0, ksos_refine=True, ksos_lse_samples=0,
        ksos_lse_sigmas=(0.2, 0.01), ksos_lse_lambda=1.0, ksos_cluster_size=10, ksos_gnc=None,
        ksos_warm_start=True, ksos_cacs_options=cacs,
    )


def plan_cycle(ctrl, rollout, optimize, seed):
    """One closed-loop planning cycle (deterministic.py) from the start state, with the controller's draws as if
    constructed with this seed. Returns (plan cost, KSOS cost before MPPI, time)."""
    ctrl._cacs_rng, ctrl._cacs_calls = np.random.default_rng(seed + 101), 0
    params = ctrl.init_params(seed=seed)
    t0 = time.perf_counter()
    for it in range(ctrl.ksos_num_restart):
        params = jax.block_until_ready(rollout(state, params.replace(iter=it)))
        with Quiet():
            params = ctrl.solve_ksos(params, state)
    ksos = float(params.ksos_result_cost)
    params, _, best = optimize(state, params)
    jax.block_until_ready(best)
    return float(np.sum(best.costs)), ksos, time.perf_counter() - t0


results = []
for param, value in SETTINGS:
    ctrl = controller(param, value)
    rollout, optimize = jax.jit(ctrl.ksos_rollout), jax.jit(ctrl.optimize)
    plan_cycle(ctrl, rollout, optimize, args.seeds[0])  # compile
    rows = [plan_cycle(ctrl, rollout, optimize, s) for s in args.seeds]
    cost, ksos, t = (np.array(x) for x in zip(*rows))
    results.append(dict(param=param, value=value, seeds=args.seeds, cost=cost.tolist(), ksos_cost=ksos.tolist(),
                        time=t.tolist()))
    print(f"RESULT {param:>12s} = {str(value):>6s}: plan cost {cost.mean():8.4f} ± {cost.std(ddof=1):7.4f} "
          f"(median {np.median(cost):.4f})  KSOS-only {ksos.mean():8.4f}  time {t.mean():6.2f} s   "
          f"per seed " + " / ".join(f"{c:.3f}" for c in cost), flush=True)
    Path(args.json).write_text(json.dumps(results, indent=1))
print("saved", args.json)
