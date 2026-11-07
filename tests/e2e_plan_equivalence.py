"""End-to-end equivalence of the planner (branch speedup-postprocess): same seed -> same plans.

Runs examples/closed_loop_test.py --sim --algo globalmppi in this process, with two patches that change no
behaviour of the code under test:
  * after GlobalMPPI.__init__, admm_cuda_options is UPDATED (not replaced) with time_limit=None and max_iter=N,
    so the ADMM stops after a fixed number of iterations instead of 2 s of wall clock (otherwise the iteration
    count, and so the plan, depends on the GPU load). Some solves converge before N, some stop at N (status
    max_iterations): both branches of `ok = info["success"]` in _solve_cacs are exercised.
  * GlobalMPPI._solve_cacs and ADMM.solve_ksos are wrapped to record, per restart, the new mean, its guard cost,
    and the ADMM status / iterations / success / x.

  python tests/e2e_plan_equivalence.py run --out A.npz            # on the unchanged code
  python tests/e2e_plan_equivalence.py run --out B.npz            # again: the run-to-run noise floor
  python tests/e2e_plan_equivalence.py compare A.npz B.npz        # exit code 1 on any difference > --tol
  python tests/e2e_plan_equivalence.py run --out C.npz && python tests/e2e_plan_equivalence.py compare A.npz C.npz

Needs PYTHONPATH with Global-MPPI, Global-MPPI/ksos-tools, cuKernelSOS/python, XLA_PYTHON_CLIENT_PREALLOCATE=false on a
shared GPU, and a GPU. The simulated closed loop feeds each plan into the next cycle, so a difference anywhere
shows up in all later cycles.
"""
import argparse
import atexit
import os
import sys
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(a):
    sys.path.insert(0, os.path.join(ROOT, "examples"))
    sys.path.insert(0, ROOT)
    from ksos_tools.solvers import ADMM
    import global_mppi.algs.global_mppi as gm

    rec = []          # per restart
    solves = []       # per ADMM solve
    orig_init, orig_cacs, orig_solve = gm.GlobalMPPI.__init__, gm.GlobalMPPI._solve_cacs, ADMM.solve_ksos

    def init(self, *args, **kw):
        orig_init(self, *args, **kw)
        self.admm_cuda_options.update(time_limit=None, max_iter=a.max_iter)

    def solve_ksos(*args, **kw):
        x, info = orig_solve(*args, **kw)
        solves.append(dict(status=info["status"], iterations=int(info["iterations"]), success=bool(info["success"]),
                           x=np.full(12, np.nan) if x is None else np.asarray(x, float).reshape(-1)))
        return x, info

    def solve_cacs(self, params, state):
        n0 = len(solves)
        out = orig_cacs(self, params, state)
        s = solves[n0] if len(solves) > n0 else dict(status="none", iterations=0, success=False, x=np.full(12, np.nan))
        rec.append(dict(mean=np.asarray(out.mean, float).reshape(-1), cost=float(out.ksos_result_cost), **s))
        return out

    gm.GlobalMPPI.__init__, gm.GlobalMPPI._solve_cacs, ADMM.solve_ksos = init, solve_cacs, solve_ksos

    def dump():
        d = {"n_restarts": np.array(len(rec))}
        for i, r in enumerate(rec):
            d[f"r{i}/mean"], d[f"r{i}/cost"] = r["mean"], np.array(r["cost"])
            d[f"r{i}/status"], d[f"r{i}/iterations"] = np.array(r["status"]), np.array(r["iterations"])
            d[f"r{i}/success"], d[f"r{i}/x"] = np.array(r["success"]), r["x"]
        np.savez(a.out, **d)
        st = [r["status"] for r in rec]
        print(f"\n== e2e: {len(rec)} restarts recorded; statuses {dict((s, st.count(s)) for s in set(st))}; saved {a.out}")

    atexit.register(dump)
    outdir = tempfile.mkdtemp(prefix="e2e_run_")
    sys.argv = ["closed_loop_test.py", "--sim", "--algo", "globalmppi", "--ksos-solver", "ADMM-cuda", "--ksos-sampling", "cacs",
                "--max-cycles", str(a.cycles), "--max-time", "3600", "--seed", str(a.seed), "--out", outdir]
    import closed_loop_test
    closed_loop_test.main()


def compare(a):
    A, B = np.load(a.a, allow_pickle=False), np.load(a.b, allow_pickle=False)
    bad, worst, ignored = 0, 0.0, 0
    na, nb = int(A["n_restarts"]), int(B["n_restarts"])
    n = min(na, nb)
    if n < 10 or (na != nb and not a.prefix):
        print(f"FAIL: restarts {na} vs {nb} (use --prefix to compare the common prefix; at least 10 needed)")
        return 1
    if na != nb:   # plan i depends only on plans before it, so the common prefix is a valid comparison
        print(f"NOTE: {na} vs {nb} restarts (the run length depends on the wall clock); comparing the first {n}")
    for k in sorted(A.files):
        if k == "n_restarts" or int(k.split("/")[0][1:]) >= n:
            continue
        x, y = A[k], B[k]
        if k.endswith("/x") and not (bool(A[k[:-1] + "success"]) and bool(B[k[:-1] + "success"])):
            ignored += 1   # _solve_cacs discards x unless the solve converged: it cannot reach the plan
            continue
        if x.dtype.kind in "US" or k.endswith(("/iterations", "/success")):
            ok = bool(np.all(x == y))
            msg = f"{x} vs {y}"
        else:
            both_nan = np.isnan(x) & np.isnan(y)
            d = np.where(both_nan, 0.0, np.abs(x - y))
            d = float(np.max(d)) if d.size else 0.0
            ok, msg = d <= a.tol, f"max|diff| {d:.2e}"
            worst = max(worst, d)
        if not ok:
            bad += 1
            print(f"  FAIL  {k}: {msg}")
    print(f"{'PASS' if bad == 0 else 'FAIL'}: {n} restarts, {bad} differences, worst max|diff| {worst:.2e}, tol {a.tol:g} "
          f"(x of {ignored} non-converged solves not compared: the planner discards it)")
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--cycles", type=int, default=3)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--max-iter", type=int, default=150)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--tol", type=float, default=1e-6)
    c.add_argument("--prefix", action="store_true", help="compare the common prefix when the runs have different lengths")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        sys.exit(compare(a))


if __name__ == "__main__":
    main()
