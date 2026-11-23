"""Run examples/pusht.py (Global-MPPI) with timers around the pieces of _solve_cacs, without changing the code.

Wraps, in global_mppi.algs.global_mppi's namespace: cacs_sample, build_problem, _elimination_cliques,
chordal_decomposition, ADMM.solve_ksos (also records its info: setup_time, solve_time, iterations, status, cliques),
and _jit_costs (rollout calls). One CSV row per call: gm_profile/calls_seed<seed>.csv.
Usage (as run_with_own_cache.py): python profile_gm.py examples/pusht.py <args>
"""
import csv, os, runpy, sys, time
# cuKernelSOS's extension must load before jax (see global_mppi/__init__.py), which this wrapper imports first
_ck = os.environ.get("CUKERNELSOS_PYTHON")
if _ck:
    sys.path.insert(0, _ck)
    try:
        import cukernelsos._cukernelsos  # noqa: F401
    except ImportError:
        pass
import jax
_orig = jax.config.update
def _update(name, value):
    if name == "jax_compilation_cache_dir":
        value = os.path.expanduser("~/.cache/global-mppi/jax_cache")
    return _orig(name, value)
jax.config.update = _update

import global_mppi.algs.global_mppi as G  # noqa: E402

seed = sys.argv[sys.argv.index("--seed") + 1] if "--seed" in sys.argv else "0"
out = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"calls_seed{seed}.csv"), "w", newline="")
w = csv.writer(out); w.writerow(["t", "what", "seconds", "extra"])
T0 = time.perf_counter()

def timed(name, fn, extra=None):
    def wrapper(*a, **k):
        t = time.perf_counter(); r = fn(*a, **k); dt = time.perf_counter() - t
        if name == "jit_costs":
            try: r.block_until_ready()
            except Exception: pass
            dt = time.perf_counter() - t
        w.writerow([f"{t - T0:.4f}", name, f"{dt:.6f}", extra(r, a, k) if extra else ""]); out.flush()
        return r
    return wrapper

def admm_extra(r, a, k):
    info = r[1]
    st = info.get("admm_stats", {}) or {}
    return (f"setup={info.get('setup_time', '')} solve={info.get('solve_time', '')} iters={info.get('iterations')} "
            f"status={info.get('status')} max_clique={st.get('max_clique', '')} num_cliques={st.get('num_cliques', '')} "
            f"n={len(k.get('samples', [])) if k.get('samples') is not None else ''}")

G.cacs_sample = timed("cacs_sample", G.cacs_sample, lambda r, a, k: f"n={len(r[0])} rollouts={r[2].get('rollouts')} rounds={r[2].get('rounds')}")
import pickle
_bp = G.build_problem
_dumped = [0]
def _build_dump(samples, sigma, f, *a, **k):
    if os.environ.get("DUMP_BUILD") and _dumped[0] < 30:
        fv = f(samples)
        pickle.dump(dict(samples=np.asarray(samples), sigma=sigma, f=np.asarray(fv), lambd=k.get("lambd")),
                    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"build_{_dumped[0]:02d}.pkl"), "wb"))
        _dumped[0] += 1
    return _bp(samples, sigma, f, *a, **k)
import numpy as np
G.build_problem = timed("build_problem", _build_dump)
G._elimination_cliques = timed("elimination_cliques", G._elimination_cliques)
G.chordal_decomposition = timed("chordal_decomposition", G.chordal_decomposition)
G.ADMM.solve_ksos = timed("admm_solve_ksos", G.ADMM.solve_ksos, admm_extra)
_init = G.GlobalMPPI.__init__
def init(self, *a, **k):
    _init(self, *a, **k)
    self._jit_costs = timed("jit_costs", self._jit_costs)
G.GlobalMPPI.__init__ = init

sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
