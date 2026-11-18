"""Common solver tolerance for pusht_solver_benchmark.py --tol (loaded at start-up of each spawned solver process).

pusht_solver_benchmark.py puts this folder on PYTHONPATH and sets KSOS_BENCH_TOL; Python imports sitecustomize at
start-up, so every solver child gets:
  newton  stop when the squared Newton decrement < tol   (ksos-tools uses problem.epsilon = t * n = lambda)
  MOSEK   interior-point relative gap, primal and dual feasibility tolerances = tol   (CVXPY default 1e-8)
ADMM takes its tolerances from the benchmark options (eps_abs = eps_rel = tol); nothing to do here.
"""
import os

_tol = os.environ.get("KSOS_BENCH_TOL")
if _tol:
    _tol = float(_tol)
    from ksos_tools.solvers.problem import Problem

    Problem.epsilon = property(lambda self: _tol)

    import cvxpy as _cp

    _solve = _cp.Problem.solve

    def _solve_tol(self, *args, **kw):
        if (args and args[0] == "MOSEK") or kw.get("solver") == "MOSEK":
            kw["mosek_params"] = {**{"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": _tol, "MSK_DPAR_INTPNT_CO_TOL_PFEAS": _tol,
                                     "MSK_DPAR_INTPNT_CO_TOL_DFEAS": _tol}, **kw.get("mosek_params", {})}
        return _solve(self, *args, **kw)

    _cp.Problem.solve = _solve_tol
