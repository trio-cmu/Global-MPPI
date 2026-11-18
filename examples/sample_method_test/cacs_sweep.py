"""Hyperparameter sweep of CACS sampling alone (global_mppi/cacs.py, no KSOS) on PushT.

Each run is one CACS pass as GlobalMPPI._solve_cacs makes it at restart 0 (the whole box
[-1, 1]^12, kernel radius h, the controller's rollout), from the start state of pusht.py.
One parameter changes at a time from the CacsConfig defaults (--stage oat), or a list of
full configurations is run (--configs '[{"k": 256, "replace": 80}, ...]').
Per run: sampling time (JIT compiles done before), rollouts, best sample cost, and the
mean of the 20 cheapest samples (is there a good basin, not one lucky point).
Seeds: cacs_sample(seed=s, rng=default_rng(s + 101)), so the initial Sobol batch differs
per seed (in the closed loop it is the call counter, the same for every seed).

Example:
    python cacs_sweep.py --stage oat --seeds 0 1 2 3 4 --json cacs_sweep_oat.json
"""

import argparse
import json
import time
from dataclasses import asdict, replace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from global_mppi.algs import GlobalMPPI
from global_mppi.cacs import CacsConfig, cacs_sample
from global_mppi.tasks.pusht import PushT

parser = argparse.ArgumentParser(description="sweep of CACS sampling hyperparameters on PushT")
parser.add_argument("--stage", choices=["oat", "configs"], default="oat")
parser.add_argument("--configs", default="[]", help="JSON list of CacsConfig overrides (--stage configs)")
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
parser.add_argument("--json", default="cacs_sweep.json")
args = parser.parse_args()

OAT = {  # one at a time, around the defaults
    "k": [64, 256, 512],
    "n": [250, 500],
    "batch": [256, 1536],
    "initial": [100, 400],
    "replace": [0, 80, 160],
    "h": [0.8, 1.6],
    "max_clique": [10, 15, 30],
    "global_every": [60, 250],
}

# ---------------------------------------------------------------- PushT and the closed loop's rollout
task = PushT()
mj_model = task.mj_model
mj_model.opt.timestep = 0.001
mj_model.opt.iterations = 100
mj_model.opt.ls_iterations = 50
mj_data = mujoco.MjData(mj_model)
mj_data.qpos = [0.1, 0.1, 1.3, 0.0, 0.0]
mj_data.qvel[:] = 0.0
mujoco.mj_forward(mj_model, mj_data)
state = mjx.put_data(mj_model, mj_data)
state = state.replace(mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat)
ctrl = GlobalMPPI(task, num_samples=1000, noise_level=0.2, temperature=0.1, spline_type="cubic",
                  plan_horizon=1.0, num_knots=6, iterations=5, ksos_solver="ADMM-cuda",
                  ksos_num_restart=1, ksos_sampling="cacs", ksos_num_samples=1000, mppi_noise=0.1)
params = ctrl.init_params(seed=0)
shape = (ctrl.num_knots, task.model.nu)
lo = np.tile(np.asarray(task.u_min, np.float64), ctrl.num_knots)
hi = np.tile(np.asarray(task.u_max, np.float64), ctrl.num_knots)
compiled = set()


def roll(points, pad_to=None):
    """As GlobalMPPI._solve_cacs's roll (clip, pad to a fixed shape)."""
    points = np.clip(np.atleast_2d(points), lo, hi)
    m = len(points)
    if pad_to is not None and m < pad_to:
        points = np.vstack([points, np.zeros((pad_to - m, len(lo)))])
    c = ctrl._jit_costs(state, params.tk, jnp.asarray(points.reshape(-1, *shape), jnp.float32))
    return np.asarray(c, dtype=np.float64)[:m]


def warm(cfg):
    """Compile every rollout shape this configuration uses, before timing."""
    for s in (cfg.initial, cfg.k, cfg.replace_batch, 1):
        if s not in compiled:
            jax.block_until_ready(roll(np.zeros((s, len(lo)))))
            compiled.add(s)


def run(cfg, seed):
    warm(cfg)
    t0 = time.perf_counter()
    X, f, st = cacs_sample(roll, lo, hi, cfg, seed=seed, rng=np.random.default_rng(seed + 101))
    t = time.perf_counter() - t0
    return dict(seed=seed, time=t, rollouts=int(st["rollouts"]), kept=len(X), best=float(f.min()),
                top20=float(np.sort(f)[:20].mean()), rounds=int(st["rounds"]), replaced=int(st["replaced"]))


if args.stage == "oat":
    variants = [("default", {})] + [(f"{p}={v}", {p: v}) for p, vals in OAT.items() for v in vals]
else:
    variants = [(" ".join(f"{k}={v}" for k, v in c.items()) or "default", c) for c in json.loads(args.configs)]

out = []
print(f"{'config':<22} {'time (s)':>14} {'rollouts':>8} {'best':>14} {'top-20 mean':>14} {'rounds':>6}")
for name, over in variants:
    over = dict(over)
    if "replace" in over and "replace_batch" not in over:  # replacements in one batch, as the default
        over["replace_batch"] = max(1, over["replace"])
    cfg = replace(CacsConfig(), **over)
    runs = [run(cfg, s) for s in args.seeds]
    T, B, Q = (np.array([r[k] for r in runs]) for k in ("time", "best", "top20"))
    out.append(dict(name=name, config=asdict(cfg), runs=runs))
    print(f"{name:<22} {T.mean():7.3f} ± {T.std(ddof=1):.3f} {np.mean([r['rollouts'] for r in runs]):>8.0f} "
          f"{B.mean():7.3f} ± {B.std(ddof=1):.2f} {Q.mean():7.3f} ± {Q.std(ddof=1):.2f} "
          f"{np.mean([r['rounds'] for r in runs]):>6.1f}", flush=True)

with open(args.json, "w") as fh:
    json.dump(dict(seeds=args.seeds, results=out), fh, indent=1)
print("RESULT saved", args.json)
