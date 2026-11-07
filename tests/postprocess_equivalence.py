"""Equivalence check for the post-processing / GPU-ADMM speed-ups (branch speedup-postprocess).

The speed-ups must not change the algorithm. This script runs the KSOS solve on fixed samples
(fixed CACS seed, fixed iteration counts, no time limit) and compares the outcome against a baseline
saved from the code before the change.

  # 1) on the unchanged code: save the baseline (and, once, the run-to-run noise floor)
  python tests/postprocess_equivalence.py save    --out BASE.npz
  python tests/postprocess_equivalence.py save    --out BASE2.npz
  python tests/postprocess_equivalence.py compare --baseline BASE.npz --candidate BASE2.npz   # noise floor
  # 2) after each change: run the candidate and compare (exit code 1 on any failure)
  python tests/postprocess_equivalence.py compare --baseline BASE.npz
  # 3) repair bisection only: the returned end must be the feasible one
  python tests/postprocess_equivalence.py repair

Needs PYTHONPATH to hold Global-MPPI, Global-MPPI/ksos-tools and cuKernelSOS/python, and a GPU.
No JAX / MuJoCo: the samples come from global_mppi.cacs.cacs_sample on a cheap numpy cost.

Compared per scenario (all must agree within --tol, default 1e-6): status (exactly), iterations (exactly),
alpha, alpha_raw, x, the objective f@alpha and the dual estimate c. eigmin / eigmin_raw are compared only
if both runs report them (a change may drop the eigenvalue diagnostics). The repair multiplier
(alpha_t = (1 - t) alpha + t / N, returned in repair_t) is compared too.
"""
import argparse
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

DIM = 12                    # 6 knots x 2 controls, as the PushT planner
HALF = 0.1                  # vmax
H = 0.07                    # CACS kernel radius used in the real run
N_SAMPLES = 500             # smaller than the real 1000 so the baseline runs in seconds; same structure
LAMBD = 1e-5

# name -> options of the solve. eps 0 never satisfies the stopping test: exactly max_iter iterations.
SCENARIOS = {
    "fixed_30": dict(max_iter=30, eps_abs=0.0, eps_rel=0.0),
    "fixed_100": dict(max_iter=100, eps_abs=0.0, eps_rel=0.0),
    "fixed_300": dict(max_iter=300, eps_abs=0.0, eps_rel=0.0),
    "converge_default_eps": dict(max_iter=5000, eps_abs=1e-6, eps_rel=1e-5),
}


def make_problem(seed):
    from dataclasses import replace  # noqa: F401
    from global_mppi.cacs import CacsConfig, cacs_sample
    from ksos_tools.problems.features import build_problem

    rng = np.random.default_rng(seed)
    target = rng.uniform(-0.05, 0.05, DIM)

    def roll(points, pad_to=None):
        p = np.atleast_2d(points)
        return 100.0 * np.sum((p - target) ** 2, axis=1) + 0.3 * np.sin(30 * p).sum(axis=1) + 3.0

    cfg = CacsConfig(n=N_SAMPLES, h=H, max_clique=20, initial=200, reuse=40, k=512, batch=768, replace=0)
    lo, hi = -HALF * np.ones(DIM), HALF * np.ones(DIM)
    samples, f, _ = cacs_sample(roll, lo, hi, cfg, seed=seed, rng=np.random.default_rng(seed))
    order = np.arange(len(samples))[::-1]  # reverse insertion order, as _solve_cacs
    return build_problem(samples[order], cfg.h, lambda _: f[order], lambd=LAMBD)


def run_scenario(problem, opts):
    from ksos_tools.solvers.ADMM import solve_dual

    admm = dict(time_limit=None, record_history=False, backend="cuda", cuda_options={"anderson_reg": 1e-10})
    admm.update(opts)
    # solve_dual (not solve_ksos): the AdmmResult also carries alpha_raw. The options are those of
    # GlobalMPPI.admm_cuda_options except for the time limit.
    r = solve_dual(problem.Phi, problem.f, problem.lambd, samples=problem.samples, **admm)
    info = dict(status=r.status, iterations=r.iterations, alpha=r.alpha, alpha_raw=r.alpha_raw, objective=r.objective,
                objective_raw=r.objective_raw, c=r.c, repair_t=r.repair_t, eigmin=r.eigmin, eigmin_raw=r.eigmin_raw)
    return r.x, info


PLANNER_OPTS = dict(postprocess="converged", repair_tol=1e-6, diagnostics=False)   # what GlobalMPPI.admm_cuda_options adds


def store(out, name, x, info):
    out[f"{name}/status"] = np.array(info["status"])
    out[f"{name}/iterations"] = np.array(int(info["iterations"]))
    out[f"{name}/x"] = np.asarray(x, dtype=float).reshape(-1)
    out[f"{name}/alpha"] = np.asarray(info["alpha"], dtype=float)
    out[f"{name}/alpha_raw"] = np.asarray(info["alpha_raw"], dtype=float)
    out[f"{name}/objective"] = np.array(float(info["objective"]))
    out[f"{name}/objective_raw"] = np.array(float(info["objective_raw"]))
    out[f"{name}/c"] = np.array(float(info["c"]))
    out[f"{name}/repair_t"] = np.array(float(info["repair_t"]))
    for key in ("eigmin", "eigmin_raw"):
        if info.get(key) is not None and np.isfinite(info[key]):
            out[f"{name}/{key}"] = np.array(float(info[key]))


def collect(seed):
    out = {}
    problem = make_problem(seed)
    out["meta/n"] = np.array(problem.n)
    out["meta/bandwidth"] = np.array(problem.bandwidth)
    for name, opts in SCENARIOS.items():
        x, info = run_scenario(problem, opts)
        out[f"{name}/status"] = np.array(info["status"])
        out[f"{name}/iterations"] = np.array(int(info["iterations"]))
        out[f"{name}/x"] = np.asarray(x, dtype=float).reshape(-1)
        out[f"{name}/alpha"] = np.asarray(info["alpha"], dtype=float)
        out[f"{name}/alpha_raw"] = np.asarray(info["alpha_raw"], dtype=float)
        out[f"{name}/objective"] = np.array(float(info["objective"]))
        out[f"{name}/objective_raw"] = np.array(float(info["objective_raw"]))
        out[f"{name}/c"] = np.array(float(info["c"]))
        out[f"{name}/repair_t"] = np.array(float(info["repair_t"]))
        for key in ("eigmin", "eigmin_raw"):
            if key in info and info[key] is not None and np.isfinite(info[key]):
                out[f"{name}/{key}"] = np.array(float(info[key]))
        print(f"  {name:22s} status={info['status']:15s} iters={info['iterations']:5d} "
              f"repair_t={info['repair_t']:.3e} obj={info['objective']:.8f}", flush=True)
        try:   # the planner's options (only the candidate code knows them; the baseline run has none)
            xp, ip = run_scenario(problem, {**opts, **PLANNER_OPTS})
        except TypeError:
            continue
        store(out, f"planner:{name}", xp, ip)
    return out


def cmd_save(a):
    out = {}
    for seed in a.seeds:
        print(f"seed {seed}: n_samples {N_SAMPLES}")
        for k, v in collect(seed).items():
            out[f"seed{seed}/{k}"] = v
    np.savez(a.out, **out)
    print("saved", a.out)
    return 0


def compare_arrays(name, a, b, tol):
    if a.dtype.kind in "US" or b.dtype.kind in "US":
        return str(a) == str(b), f"{a} vs {b}"
    if name.endswith("/iterations"):
        return int(a) == int(b), f"{int(a)} vs {int(b)}"
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    d = float(np.max(np.abs(a - b))) if a.size else 0.0
    return d <= tol, f"max|diff| {d:.2e}"


def cmd_compare(a):
    base = np.load(a.baseline, allow_pickle=False)
    if a.candidate:
        cand = dict(np.load(a.candidate, allow_pickle=False))
    else:
        cand = {}
        for seed in a.seeds:
            print(f"seed {seed}: running the candidate")
            for k, v in collect(seed).items():
                cand[f"seed{seed}/{k}"] = v
    bad, worst = 0, 0.0
    optional = ("/eigmin", "/eigmin_raw")
    for key in sorted(base.files):
        if key not in cand:
            if key.endswith(optional):
                print(f"  SKIP  {key}: not reported by the candidate")
                continue
            print(f"  FAIL  {key}: missing in the candidate")
            bad += 1
            continue
        ok, msg = compare_arrays(key, base[key], np.asarray(cand[key]), a.tol)
        if not ok:
            bad += 1
        if "max|diff|" in msg:
            worst = max(worst, float(msg.split()[1]))
        if not ok or a.verbose:
            print(f"  {'ok  ' if ok else 'FAIL'}  {key}: {msg}")
    # planner options vs the baseline's default options. A solve that converged: everything must agree. One that did
    # not: the planner discards x (success False), so only what is computed before the post-processing must agree.
    n_planner = 0
    for key in sorted(k for k in cand if "/planner:" in k and k.endswith("/status")):
        seed_pref, rest = key.split("/planner:")
        name = rest.rsplit("/", 1)[0]
        pk, bk = f"{seed_pref}/planner:{name}", f"{seed_pref}/{name}"
        converged = str(base[f"{bk}/status"]) == "converged"
        fields = ["status", "iterations", "alpha_raw", "objective_raw", "c"]
        if converged:
            fields += ["x", "alpha", "objective", "repair_t"]
        n_planner += 1
        for fld in fields:
            ok, msg = compare_arrays(f"{bk}/{fld}", base[f"{bk}/{fld}"], np.asarray(cand[f"{pk}/{fld}"]), a.tol)
            if "max|diff|" in msg:
                worst = max(worst, float(msg.split()[1]))
            if not ok:
                bad += 1
            if not ok or a.verbose:
                print(f"  {'ok  ' if ok else 'FAIL'}  planner options {bk}/{fld} ({'converged' if converged else 'not converged'}): {msg}")
    if not n_planner and not a.candidate:
        print("  NOTE  the candidate reported no planner-option runs (code without postprocess options)")
    else:
        print(f"  planner options: {n_planner} scenario runs checked")
    print(f"\n{'PASS' if bad == 0 else 'FAIL'}: {len(base.files)} entries, {bad} failures, "
          f"worst max|diff| {worst:.2e}, tol {a.tol:g}")
    return 0 if bad == 0 else 1


# ---- repair bisection: the returned end of the bracket must be the feasible one -------------------------

def reference_repair(structure, ops, alpha, margin=1e-12):
    """Independent reference: the original 60-step / 1e-14 bisection, kept here so that it does not move with the code."""
    from ksos_tools.solvers.ADMM.operators import assemble_matrix, is_positive_definite

    n = len(alpha)
    uniform = np.full(n, 1.0 / n)
    shift = margin * float(np.max(np.abs(ops.M_coords(uniform)[structure.is_diag])))
    M1 = ops.M_coords(uniform) - shift * structure.is_diag
    M0 = ops.M_coords(alpha) - shift * structure.is_diag
    pd = lambda t: is_positive_definite(assemble_matrix(structure, (1 - t) * M0 + t * M1))  # noqa: E731
    if pd(0.0):
        return 0.0, pd
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if pd(mid):
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-14 * max(hi, 1e-300):
            break
    return hi, pd


def cmd_repair(a):
    from ksos_tools.solvers.ADMM import build_operators, chordal_decomposition
    from ksos_tools.solvers.ADMM.operators import assemble_matrix, feasibility_repair, is_positive_definite

    bad = 0
    import inspect
    a.repair_kwargs = {"rel_tol": 1e-6} if "rel_tol" in inspect.signature(feasibility_repair).parameters else {}
    print("repair tolerance:", a.repair_kwargs or "(code without rel_tol)")
    cases = []
    for seed in a.seeds:
        problem = make_problem(seed)
        structure = chordal_decomposition(problem.Phi)
        cases.append((seed, problem, structure, build_operators(problem.Phi, structure, problem.lambd)))
    n = cases[0][1].n
    for (seed, problem, structure, ops), iters in [(c, i) for c in cases for i in (30, 100, 300)]:  # early stops: M(alpha_raw) is not PSD, so the repair has work to do
        _, info = run_scenario(problem, dict(max_iter=iters, eps_abs=0.0, eps_rel=0.0, repair=False))
        alpha_raw = np.asarray(info["alpha_raw"], dtype=float)
        t_ref, pd_ref = reference_repair(structure, ops, alpha_raw)
        alpha_new, t_new = feasibility_repair(structure, ops, alpha_raw, **a.repair_kwargs)
        uniform = np.full(n, 1.0 / n)
        shift = 1e-12 * float(np.max(np.abs(ops.M_coords(uniform)[structure.is_diag])))
        M_new = ops.M_coords(alpha_new) - shift * structure.is_diag
        feasible = is_positive_definite(assemble_matrix(structure, M_new))
        # the returned end must be feasible (hi), and not further from the exact boundary than the tolerance
        gap = t_new - t_ref
        ok = feasible and gap >= -1e-14 and abs(gap) <= a.tol
        d_alpha = float(np.max(np.abs(alpha_new - ((1 - t_ref) * alpha_raw + t_ref * uniform))))
        ok = ok and d_alpha <= a.tol
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  seed={seed} max_iter={iters:4d} repair needed={t_ref > 0}  t_ref={t_ref:.9e} "
              f"t_new={t_new:.9e} gap={gap:+.2e} feasible_at_returned={feasible} max|d alpha|={d_alpha:.2e}")
        if t_ref == 0.0:
            print("        (alpha_raw already feasible: this case does not exercise the bisection)")
    # the returned point must also be feasible when the test is run on an alpha that needs t close to 1
    print(f"\n{'PASS' if bad == 0 else 'FAIL'}: {bad} failures, tol {a.tol:g}")
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("save", "compare", "repair"):
        p = sub.add_parser(name)
        p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        p.add_argument("--tol", type=float, default=1e-6)
        if name == "save":
            p.add_argument("--out", required=True)
        if name == "compare":
            p.add_argument("--baseline", required=True)
            p.add_argument("--candidate", help="a saved run to compare instead of running the code (noise floor)")
            p.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    sys.exit({"save": cmd_save, "compare": cmd_compare, "repair": cmd_repair}[a.cmd](a))


if __name__ == "__main__":
    main()
