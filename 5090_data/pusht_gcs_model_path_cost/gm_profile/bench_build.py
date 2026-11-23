"""Benchmark build_problem (banded Cholesky, ksos-tools) against a dense-LAPACK Cholesky that gives the same Phi.
Inputs: build_XX.pkl dumped by profile_gm.py (DUMP_BUILD=1)."""
import glob, pickle, time
import numpy as np, scipy.sparse as sp, scipy.linalg as sla
from scipy.spatial import cKDTree
from ksos_tools.problems.features import build_problem, wendland

def dense_phi(samples, sigma):
    n, dim = samples.shape
    pairs = cKDTree(samples).query_pairs(sigma, output_type="ndarray"); i, j = pairs[:, 0], pairs[:, 1]
    v = wendland(np.linalg.norm(samples[i] - samples[j], axis=1) / sigma, max(dim, 3))
    K = np.eye(n); K[i, j] = v; K[j, i] = v
    return sp.csc_matrix(sla.cholesky(K, lower=False, overwrite_a=True, check_finite=False))  # K = U^T U, the same U

P = [pickle.load(open(f, "rb")) for f in sorted(glob.glob("build_*.pkl"))]
def med(fn, rep=5):
    ts = []
    for _ in range(rep):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return np.median(ts) * 1e3
a = [med(lambda: build_problem(p["samples"], p["sigma"], lambda _: p["f"], lambd=1e-5)) for p in P]
b = [med(lambda: dense_phi(p["samples"], p["sigma"])) for p in P]
err = max(abs(build_problem(p["samples"], p["sigma"], lambda _: p["f"]).Phi - dense_phi(p["samples"], p["sigma"])).max() for p in P)
print(f"{len(P)} problems: banded build_problem median {np.median(a):.1f} ms (mean {np.mean(a):.1f}); dense Cholesky median {np.median(b):.1f} ms (mean {np.mean(b):.1f}); max |Phi diff| {err:.1e}")
