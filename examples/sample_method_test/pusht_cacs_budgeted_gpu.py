"""Single-step test of budgeted global cost-aware chordal sampling (CACS) on PushT, run
through the SAME code as the closed loop (examples/pusht.py ... --ksos-solver ADMM-cuda
--ksos-sampling cacs globalmppi).

The controller is built with pusht.py's ADMM-cuda settings and one planning step is run the
way global_mppi/simulation/deterministic.py runs cycle 0, restart 0:
    params = ctrl.init_params(seed=seed); params = ctrl.ksos_rollout(state, params)
    params = ctrl.solve_ksos(params, state)
so CACS is global_mppi/cacs.py (CacsConfig defaults: n 1000, 200 initial Sobol rollouts,
40 seeds, h 0.7, cliques < 20, 768 proposals and k = 512 picks per round, no replacements) and KSOS is GlobalMPPI._solve_cacs (Wendland kernel of radius h, reverse
insertion order, cuKernelSOS GPU ADMM with the controller's options, time limit 2 s), then
the guard (the cheaper of x and the best sample). The script only wraps cacs_sample and
ADMM.solve_ksos to record what they return and how long they take.

Baselines with the same number of rollouts as CACS:
  two-level  GlobalMPPI's two-level Sobol path (ksos_rollout + _solve_admm_cuda), clusters
             of --cluster-size, RCM order, same ADMM options and guard
  Sobol      the best of the same number of plain scrambled Sobol points (no KSOS)
Every x is judged by a fresh rollout. Each method is run once untimed first (JIT compiles),
then reset to the same seed and timed.

Example:
    python pusht_cacs_budgeted_gpu.py --seed 0
    python pusht_cacs_budgeted_gpu.py --seeds 0 1 2 --json cacs_single_step.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# cuKernelSOS's Python binding (GPU ADMM): the first cuKernelSOS/python in a parent folder, as pusht.py
for _parent in Path(__file__).resolve().parents:
    if (_parent / "cuKernelSOS" / "python" / "cukernelsos").is_dir():
        os.environ.setdefault("CUKERNELSOS_PYTHON", str(_parent / "cuKernelSOS" / "python"))
        break

import jax
import mujoco
import numpy as np
from mujoco import mjx
from scipy.stats import qmc

import global_mppi.algs.global_mppi as gm
from global_mppi.algs import GlobalMPPI
from global_mppi.tasks.pusht import PushT

parser = argparse.ArgumentParser(description="single-step CACS vs two-level on PushT, closed-loop code")
parser.add_argument("--pusher", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Y"))
parser.add_argument("--block", type=float, nargs=3, default=[0.1, 0.1, 1.3], metavar=("X", "Y", "THETA"))
parser.add_argument("--seeds", "--seed", type=int, nargs="+", default=[0],
                    help="controller seeds (GlobalMPPI seed=..., init_params(seed=...)), as pusht.py --seed")
parser.add_argument("--n", type=int, default=1000, help="CACS samples kept for KSOS (pusht.py --ksos-samples)")
parser.add_argument("--cluster-size", type=int, default=10, help="two-level cluster size (pusht.py --ksos-cluster-size)")
parser.add_argument("--two-level-n", type=int, default=None,
                    help="two-level samples (default: the CACS rollout count, rounded to the cluster size)")
parser.add_argument("--warm-start", action="store_true",
                    help="pusht.py --ksos-warm-start (at cycle 0 the warm start is the zero plan)")
parser.add_argument("--json", default=None, help="write the results here")
args = parser.parse_args()

# ---------------------------------------------------------------- PushT, as pusht.py
task = PushT()
mj_model = task.mj_model
mj_model.opt.timestep = 0.001
mj_model.opt.iterations = 100
mj_model.opt.ls_iterations = 50
mj_data = mujoco.MjData(mj_model)
mj_data.qpos = [*args.block, *args.pusher]  # [block x, y, theta, pusher x, y]
mj_data.qvel[:] = 0.0
mujoco.mj_forward(mj_model, mj_data)
state = mjx.put_data(mj_model, mj_data)
state = state.replace(mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat)


def controller(sampling, n, seed):
    """GlobalMPPI exactly as pusht.py's ADMM-cuda branch (defaults of its flags), one restart."""
    return GlobalMPPI(
        task, num_samples=1000, noise_level=0.2, temperature=0.1, spline_type="cubic",
        plan_horizon=1.0, num_knots=6, iterations=5, ksos_solver="ADMM-cuda", ksos_num_restart=1,
        ksos_sampling=sampling, ksos_num_samples=n, mppi_noise=0.1, seed=seed, ksos_refine=True,
        ksos_lse_samples=0, ksos_lse_sigmas=(0.2, 0.01), ksos_lse_lambda=1.0,
        ksos_cluster_size=args.cluster_size, ksos_gnc=None, ksos_warm_start=args.warm_start,
    )


# ---------------------------------------------------------------- record, do not change, the library calls
REC = {}
_cacs_sample, _solve_ksos = gm.cacs_sample, gm.ADMM.solve_ksos


def cacs_sample_rec(*a, **k):
    t0 = time.perf_counter()
    out = _cacs_sample(*a, **k)
    REC["cacs_time"], REC["cacs"] = time.perf_counter() - t0, out
    return out


def solve_ksos_rec(*a, **k):
    t0 = time.perf_counter()
    x, info = _solve_ksos(*a, **k)
    REC["admm_time"], REC["x"], REC["info"] = time.perf_counter() - t0, np.asarray(x), info
    return x, info


gm.cacs_sample = cacs_sample_rec
gm.ADMM = type("ADMMRecorder", (), {"solve_ksos": staticmethod(solve_ksos_rec)})


class Quiet:
    """Silence the controller's prints (the closed-loop log lines) during a call."""

    def __enter__(self):
        self.out, sys.stdout = sys.stdout, open(os.devnull, "w")

    def __exit__(self, *exc):
        sys.stdout.close()
        sys.stdout = self.out


def x_cost(ctrl, params):
    """Fresh rollout of the ADMM solution x, clipped into the box (as the guard does)."""
    x = np.clip(REC["x"].reshape(-1), np.tile(task.u_min, ctrl.num_knots), np.tile(task.u_max, ctrl.num_knots))
    return float(ctrl._jit_costs(state, params.tk, x.reshape(1, ctrl.num_knots, -1).astype(np.float32))[0])


def admm_row(name, rollouts, n_ksos, best, xc, solution, t_sample, t_total):
    info = REC["info"]
    return dict(method=name, rollouts=int(rollouts), to_ksos=int(n_ksos),
                max_clique=int(info["admm_stats"]["max_clique"]), best_sample=float(best), x=xc,
                solution=float(solution), status=info["status"], iters=int(info["iterations"]),
                admm_setup=float(info["setup_time"]), admm_iterations=float(info["solve_time"]),
                admm_total=float(REC["admm_time"]), sampling=float(t_sample), total=float(t_total))


def run_cacs(seed):
    ctrl = controller("cacs", args.n, seed)
    rollout = jax.jit(ctrl.ksos_rollout)
    for timed in (False, True):  # first call compiles every rollout shape; then the same seed again, timed
        ctrl._cacs_rng, ctrl._cacs_calls = np.random.default_rng(seed + 101), 0  # as constructed
        params = rollout(state, ctrl.init_params(seed=seed))
        jax.block_until_ready(params)
        t0 = time.perf_counter()
        with Quiet():
            out = ctrl.solve_ksos(params, state)
        t_total = time.perf_counter() - t0
    samples, f, stats = REC["cacs"]
    row = admm_row("CACS", stats["rollouts"], len(samples), f.min(), x_cost(ctrl, params),
                   out.ksos_result_cost, REC["cacs_time"], t_total)
    row.update(rounds=stats["rounds"], seeds=stats["seeds"], replaced=stats["replaced"],
               initial_best=stats["initial_best"])
    return row


def run_two_level(seed, n):
    ctrl = controller("two-level", n, seed)
    rollout = jax.jit(ctrl.ksos_rollout)
    for timed in (False, True):
        t0 = time.perf_counter()
        params = rollout(state, ctrl.init_params(seed=seed))  # two-level sampling + rollouts (GPU)
        jax.block_until_ready(params)
        t_sample = time.perf_counter() - t0
        t1 = time.perf_counter()
        with Quiet():
            out = ctrl.solve_ksos(params, state)
        t_solve = time.perf_counter() - t1
    f = np.asarray(params.lse_cost)
    return admm_row("two-level", n, n, f.min(), x_cost(ctrl, params), out.ksos_result_cost,
                    t_sample, t_sample + t_solve)


def best_sobol(seed, n):
    ctrl = controller("cacs", 10, seed)  # only for its rollout
    params = ctrl.init_params(seed=seed)
    lo, hi = np.tile(task.u_min, ctrl.num_knots), np.tile(task.u_max, ctrl.num_knots)
    X = qmc.Sobol(len(lo), scramble=True, seed=seed).random(n) * (hi - lo) + lo
    return float(np.min(ctrl._jit_costs(state, params.tk, X.reshape(n, ctrl.num_knots, -1).astype(np.float32))))


results = []
for seed in args.seeds:
    cacs = run_cacs(seed)
    n2 = args.two_level_n or int(round(cacs["rollouts"] / args.cluster_size)) * args.cluster_size
    rows = [cacs, run_two_level(seed, n2)]
    sobol = best_sobol(seed, cacs["rollouts"])
    results.append(dict(seed=seed, rows=rows, sobol_best=sobol))

    print(f"\nseed {seed}: CACS {cacs['rounds']} rounds, {cacs['seeds']} graph seeds, {cacs['replaced']}/{gm.CacsConfig().replace} "
          f"replacements, initial Sobol best {cacs['initial_best']:.3f}")
    print(f"{'method':<10} {'rollouts':>8} {'to KSOS':>7} {'clique':>6} {'best':>8} {'x':>8} {'solution':>8} "
          f"{'status':<15} {'iters':>6} {'sampling':>9} {'admm':>7} {'total':>7}")
    for r in rows:
        print(f"{r['method']:<10} {r['rollouts']:>8} {r['to_ksos']:>7} {r['max_clique']:>6} {r['best_sample']:>8.4f} "
              f"{r['x']:>8.4f} {r['solution']:>8.4f} {r['status']:<15} {r['iters']:>6} {r['sampling']:>8.3f}s "
              f"{r['admm_total']:>6.3f}s {r['total']:>6.3f}s")
    print(f"{'Sobol':<10} {cacs['rollouts']:>8}  best sample {sobol:.4f} (no KSOS)")
    print(f"RESULT seed={seed}: CACS solution {rows[0]['solution']:.4f} sampling {rows[0]['sampling']:.3f}s "
          f"admm {rows[0]['admm_total']:.3f}s | two-level solution {rows[1]['solution']:.4f} "
          f"sampling {rows[1]['sampling']:.3f}s admm {rows[1]['admm_total']:.3f}s | Sobol best {sobol:.4f}")

if args.json:
    Path(args.json).write_text(json.dumps(dict(args=vars(args), results=results), indent=1))
    print("RESULT saved", args.json)
