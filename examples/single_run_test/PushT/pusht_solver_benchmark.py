"""Solve time of the KSOS solvers on PushT samples (one state, outside the control loop).

For each n: two-level GPU Sobol samples (n/10 clusters of 10), one rollout
each, RCM order, Wendland kernel -> Phi, then every solver on the same data:
  newton       ksos-tools' damped Newton on the log-det barrier problem (Python, same Wendland K)
  newton-rs    the Rust newton_sos solver, log-det barrier; it only has native kernels, so
               it uses the Laplacian kernel with the same sigma (not the same K)
  MOSEK        ksos-tools' CVXPY + MOSEK solver (ksos_tools.solvers.external.solve_primal): the log-det
               problem, the same as newton, with one dense n x n PSD variable B and a dense Phi; no chordal
               decomposition. The time includes CVXPY's model building; the process is capped at 20 GB
  COSMO-dual   COSMO.jl on the no-log-det dual (the same problem as ADMM) with COSMO's chordal decomposition
               of the PSD constraint, in one warm Julia worker (ksos_tools.benchmark.runner: CosmoWorker,
               run_cosmo); one untimed warm-up solve first, so Julia's compile time is not counted
  ADMM-sparse  no-log-det dual, ADMM iterations on the CPU
  ADMM-cuda    the same ADMM with its iterations on the GPU (cuKernelSOS, Anderson on)
Each solve runs in a fresh spawned process with a time limit; a solver that
fails at one n is skipped for larger n. Every x is judged by a fresh rollout.
Records are appended to --json, one per (n, solver).

Example:
    python pusht_solver_benchmark.py --n 100 500 --json bench.json
    python plot_solver_benchmark.py bench.json
"""

import json
import multiprocessing
import os
import threading
import time
from pathlib import Path

import numpy as np

SOLVERS = ("newton", "newton-rs", "MOSEK", "COSMO-dual", "ADMM-sparse", "ADMM-cuda")


def _newton_rs(conn, prob, limit, options):
    """Child process: newton_sos (Rust) with the Laplacian kernel; killed at the limit."""

    def expire():
        conn.send(("timeout", f"killed after {limit:.0f} s"))
        os._exit(0)

    threading.Timer(limit, expire).start()
    try:
        import newton_sos

        n = prob.n
        start = time.perf_counter()
        rs = newton_sos.Problem(prob.lambd, prob.lambd / n, prob.samples.astype(np.float64), prob.f.astype(np.float64))
        rs.initialize_native_kernel("laplacian", prob.sigma)
        result = newton_sos.solve(rs, max_iter=options["newton_iters"], verbose=False, method="partial_piv_lu")
        elapsed = time.perf_counter() - start
        alpha = np.asarray(result.alpha, dtype=float).reshape(-1)
        conn.send(("ok", {"converged": bool(result.converged), "status": str(result.status), "time": elapsed,
                          "iterations": int(result.iterations), "c": float(result.cost), "fTa": float(prob.f @ alpha),
                          "x": (alpha @ prob.samples / alpha.sum()).tolist()}))
    except Exception as error:  # noqa: BLE001 -- forwarded to the parent
        conn.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        conn.close()
        os._exit(0)


def _mosek_cvxpy(conn, prob, limit, options):
    """Child process: ksos_tools.solvers.external.solve_primal (CVXPY + MOSEK, log-det problem, as newton);
    killed at the limit, address space capped so a huge CVXPY model fails instead of exhausting memory."""
    import resource

    def expire():
        conn.send(("timeout", f"killed after {limit:.0f} s"))
        os._exit(0)

    threading.Timer(limit, expire).start()
    cap = int(options.get("mosek_mem_gb", 20) * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    try:
        from ksos_tools.solvers import external
        from ksos_tools.solvers.problem import Problem

        n = prob.n
        problem = Problem(lambd=prob.lambd, t=prob.lambd / n)  # the same t as newton (runner._solve_newton)
        problem.register_fixed_samples(prob.samples, None, prob.f)
        problem.Phi = prob.Phi.toarray()
        start = time.perf_counter()
        _, info = external.solve_primal(problem, solver="MOSEK")
        elapsed = time.perf_counter() - start
        ok = bool(info["success"]) and info.get("alpha") is not None
        rec = {"converged": ok, "status": str(info["status"]), "time": elapsed}
        if ok:
            alpha = np.asarray(info["alpha"], dtype=float).reshape(-1)
            rec.update(iterations=None, c=float(np.asarray(info["cost"]).reshape(-1)[0]), fTa=float(prob.f @ alpha),
                       x=(alpha @ prob.samples / alpha.sum()).tolist())
        conn.send(("ok", rec))
    except Exception as error:  # noqa: BLE001 -- forwarded to the parent
        conn.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        conn.close()
        os._exit(0)


def run_cosmo_dual(prob, limit, options, state):
    """COSMO-dual with chordal decomposition in the persistent Julia worker; a record dict."""
    import shutil
    import tempfile

    from ksos_tools.benchmark.runner import CosmoWorker, run_cosmo, write_cosmo_input

    opts = {"cosmo_eps_abs": options["admm_eps_abs"], "cosmo_eps_rel": options["admm_eps_rel"],
            "decompose": True, "limit": limit}
    if state.get("cosmo") is None:
        print("  (starting the warm COSMO worker ...)", flush=True)
        state["cosmo"] = CosmoWorker()
        state["tmp"] = tempfile.mkdtemp(prefix="cosmo_dual_")
    worker = state["cosmo"]
    directory = Path(state["tmp"]) / f"n{prob.n}"
    write_cosmo_input(prob, directory, opts)
    try:
        if not state.get("warm"):  # untimed warm-up solve (Julia compilation)
            run_cosmo(worker, prob, directory, limit, opts)
            state["warm"] = True
        return run_cosmo(worker, prob, directory, limit, opts)
    except (TimeoutError, RuntimeError) as error:
        worker.stop()
        state["cosmo"] = None
        return {"converged": False, "status": str(error)}
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def run_solver(solver, prob, limit, options):
    """One solve in a fresh spawned process; a record dict."""
    from ksos_tools.benchmark.runner import _python_child

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    own = {"newton-rs": _newton_rs, "MOSEK": _mosek_cvxpy}
    target = own.get(solver, _python_child)
    args = (child_conn, prob, limit, options) if solver in own else (child_conn, solver, prob, limit, options)
    process = ctx.Process(target=target, args=args)
    process.start()
    child_conn.close()
    if not parent_conn.poll(limit + 600):
        process.kill()
        process.join()
        return {"converged": False, "status": "no answer (killed)"}
    kind, payload = parent_conn.recv()
    process.join(30)
    if process.is_alive():
        process.kill()
    if kind == "ok":
        return payload
    return {"converged": False, "status": payload}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="KSOS solver time on PushT samples")
    parser.add_argument("--pusher", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Y"))
    parser.add_argument("--block", type=float, nargs=3, default=[0.1, 0.1, 1.3], metavar=("X", "Y", "THETA"))
    parser.add_argument("--n", type=int, nargs="+", default=[100, 500, 1000, 3000])
    parser.add_argument("--cluster-size", type=int, default=10)
    parser.add_argument("--solvers", nargs="+", default=list(SOLVERS), choices=SOLVERS)
    parser.add_argument("--lambd", type=float, default=1e-5)
    parser.add_argument("--timeout", type=float, default=120.0, help="time limit per solve (s)")
    parser.add_argument("--newton-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--admm-eps-abs", type=float, default=1e-6, help="ADMM absolute residual tolerance")
    parser.add_argument("--admm-eps-rel", type=float, default=1e-5, help="ADMM relative residual tolerance")
    parser.add_argument("--tol", type=float, default=None,
                        help="one tolerance for all solvers: newton decrement, MOSEK gap / feasibility, ADMM eps_abs and "
                             "eps_rel (see bench_tol/sitecustomize.py)")
    parser.add_argument("--json", default="pusht_solver_benchmark.json", help="append the records here")
    args = parser.parse_args()
    if args.tol is not None:  # spawned solver processes load bench_tol/sitecustomize.py, which applies it
        args.admm_eps_abs = args.admm_eps_rel = args.tol
        os.environ["KSOS_BENCH_TOL"] = repr(args.tol)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(Path(__file__).resolve().parent / "bench_tol")] + [p for p in [os.environ.get("PYTHONPATH")] if p])
    # cuKernelSOS's Python binding (GPU ADMM): the first cuKernelSOS/python in a parent folder
    for parent in Path(__file__).resolve().parents:
        if (parent / "cuKernelSOS" / "python" / "cukernelsos").is_dir():
            os.environ.setdefault("CUKERNELSOS_PYTHON", str(parent / "cuKernelSOS" / "python"))
            break

    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    from global_mppi.algs import GlobalMPPI
    from global_mppi.sampling import TwoLevelSampler
    from global_mppi.tasks.pusht import PushT
    from ksos_tools.problems.features import build_problem
    from ksos_tools.problems.sampling import rcm_order

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
    ctrl = GlobalMPPI(task, num_samples=256, noise_level=0.1, temperature=0.1,
                      spline_type="cubic", plan_horizon=1.0, num_knots=6)
    nu = task.model.nu
    dim = ctrl.num_knots * nu
    tk = jnp.linspace(0.0, ctrl.plan_horizon, ctrl.num_knots) + state.time
    u_min = np.tile(np.asarray(task.u_min), ctrl.num_knots)
    u_max = np.tile(np.asarray(task.u_max), ctrl.num_knots)

    @jax.jit
    def _costs(knots):
        return ctrl.rollout(state, tk, knots).costs.sum(axis=-1)

    def rollout(points):
        points = np.clip(np.atleast_2d(points), u_min, u_max)
        return np.asarray(_costs(jnp.asarray(points.reshape(-1, ctrl.num_knots, nu), jnp.float32)), np.float64)

    options = {"newton_iters": args.newton_iters, "admm_eps_abs": args.admm_eps_abs, "admm_eps_rel": args.admm_eps_rel,
               "admm_threads": 1}
    out = Path(args.json)
    records = json.loads(out.read_text()) if out.exists() else []
    skipped = {r["solver"] for r in records if not r["converged"]}
    lower, upper = jnp.asarray(u_min, jnp.float32), jnp.asarray(u_max, jnp.float32)
    cosmo_state = {}
    for n in args.n:
        sampler = TwoLevelSampler(n, dim, cluster_size=args.cluster_size, seed=args.seed)
        points, sigma, _ = jax.jit(sampler.sample)(jax.random.PRNGKey(args.seed), lower, upper)
        X, sigma = np.asarray(points, np.float64), float(sigma)
        f = rollout(X)
        order = rcm_order(X, sigma)
        prob = build_problem(X[order], sigma, lambda _: f[order], lambd=args.lambd)
        print(f"=== n = {n}: sigma {sigma:.3f}, best sample {f.min():.4f}, bandwidth {prob.bandwidth}")
        for solver in args.solvers:
            if solver in skipped:
                print(f"{solver:<12} skipped (failed at a smaller n)")
                continue
            if solver == "COSMO-dual":
                r = run_cosmo_dual(prob, args.timeout, options, cosmo_state)
            else:
                r = run_solver(solver, prob, args.timeout, options)
            rec = dict(n=n, solver=solver, seed=args.seed, sigma=sigma, best_sample_f=float(f.min()), **r)
            if args.tol is not None:
                rec["tol"] = args.tol
            if solver.startswith("ADMM"):
                rec.update(admm_eps_abs=args.admm_eps_abs, admm_eps_rel=args.admm_eps_rel)
            if r["converged"]:
                rec["x_cost"] = float(rollout(np.asarray(r["x"]))[0])
                print(f"{solver:<12} {r['time']:8.3f} s  iters {r.get('iterations')}  c {r['c']:.4f}  "
                      + (f"max clique {r['max_clique']}  " if r.get("max_clique") else "")
                      + f"x cost {rec['x_cost']:.4f}  {r['status'][:40]}")
            else:
                skipped.add(solver)
                print(f"{solver:<12} NOT CONVERGED: {r['status'][:80]}"
                      + (f" after {r['time']:.1f} s" if r.get("time") is not None else ""))
            records = [q for q in records if not (q["n"] == n and q["solver"] == solver)] + [rec]
            out.write_text(json.dumps(records, indent=1))
    if cosmo_state.get("cosmo") is not None:
        cosmo_state["cosmo"].stop()
    print("RESULT written", out)


if __name__ == "__main__":
    main()
