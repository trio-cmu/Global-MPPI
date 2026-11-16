"""G1 standup (g1_fixed): six-algorithm cost convergence (median over 6 seeds, 25-75 % band, log y) and the final tables.

    python plot_g1_six.py [DATA_DIR] [OUT_DIR]      (defaults: the folder above this script, and this folder)

DATA_DIR = data/g1_results with <algo>/s<seed>/{final.json, cycles.csv, *best_cost_history*.npy} for ps mppi dial mtp cbo gm.
Writes g1_cost_six.png and g1_six_summary.txt. Streak = longest run of cycles with torso height > 0.9 m and tilt < 10 deg.
"""
import csv, glob, json, statistics as st, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
task = "g1"
HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE
ALGOS = ["ps", "mppi", "dial", "mtp", "cbo", "gm"]
NAME = {"gm": "Global-MPPI", "ps": "Predictive Sampling", "mppi": "MPPI", "dial": "DIAL-MPC", "mtp": "MTP", "cbo": "CBO"}
COL = {"gm": "tab:blue", "ps": "tab:orange", "mppi": "tab:green", "dial": "tab:red", "mtp": "tab:purple", "cbo": "tab:brown"}


def rundir(a, s):
    return DATA / (f"{a}/{a}/s{s}" if task == "cube" else f"{a}/s{s}")


SEEDS = range(6)
hist = {a: np.stack([np.squeeze(np.load(glob.glob(str(rundir(a, s) / "*best_cost_history*.npy"))[0])) for s in SEEDS]) for a in ALGOS}
assert all(v.shape == (6, 101) and v.min() > 0 for v in hist.values())
fig, ax = plt.subplots(figsize=(6.4, 6))
for a in ALGOS:
    q25, med, q75 = np.quantile(hist[a], [0.25, 0.5, 0.75], axis=0); x = np.arange(101)
    ax.plot(x, med, label=NAME[a], color=COL[a], lw=2); ax.fill_between(x, q25, q75, color=COL[a], alpha=0.22)
ax.set_title("Dexterous in-hand manipulation" if task == "cube" else "G1 standup (g1_fixed)")
ax.set_xlabel("Iteration"); ax.set_ylabel("Cost"); ax.set_yscale("log"); ax.set_xlim(0, 100); ax.grid(True, alpha=0.3, lw=0.8)
ax.legend(loc="lower left", frameon=True, fancybox=False, framealpha=0.9, edgecolor="0.3")
plt.tight_layout(); fig.savefig(OUT / f"{task}_cost_six.png", dpi=300, bbox_inches="tight")
L = []
def p(s=""): L.append(s); print(s)
p(f"=== {task}: median cost at cycle 100 (IQR) ===")
for a in ALGOS:
    q = np.quantile(hist[a][:, 100], [.25, .5, .75]); p("%-12s %.4f (%.4f-%.4f)" % (NAME[a], q[1], q[0], q[2]))
if task == "cube":
    F = {a: [json.load(open(rundir(a, s) / "final.json")) for s in SEEDS] for a in ALGOS}
    p("\n=== final error after 100 cycles, pos mm / ang deg ===")
    p("%-6s" % "algo" + "".join("   s%d       " % s for s in SEEDS) + "  mean ± std (ddof 1)")
    for a in ALGOS:
        pp = [f["pos_err_mm"] for f in F[a]]; aa = [f["ang_err_deg"] for f in F[a]]
        p("%-6s" % a + "".join(" %5.1f/%5.1f " % (x, y) for x, y in zip(pp, aa)) + "  %.1f ± %.1f / %.1f ± %.1f" % (st.mean(pp), st.stdev(pp), st.mean(aa), st.stdev(aa)))
    p("\nplan time per cycle (mean over seeds, without cycle 0):")
    for a in ALGOS:
        v = [st.mean(float(r["plan_s"]) for r in list(csv.DictReader(open(rundir(a, s) / "cycles.csv")))[1:]) for s in SEEDS]; p("  %-6s %.2f s" % (a, st.mean(v)))
else:
    def met(a, s):
        r = list(csv.DictReader(open(rundir(a, s) / "cycles.csv"))); h = [float(x["torso_h"]) for x in r]; t = [float(x["tilt_deg"]) for x in r]
        ok = [x > 0.9 and y < 10 for x, y in zip(h, t)]; best = cur = 0
        for o in ok: cur = cur + 1 if o else 0; best = max(best, cur)
        f = json.load(open(rundir(a, s) / "final.json"))
        return dict(cost=float(r[-1]["best_cost"]), c25=st.median([float(x["best_cost"]) for x in r[-25:]]), fh=f["final_torso_h"], ft=f["final_tilt_deg"], h25=st.mean(h[-25:]), t25=st.mean(t[-25:]), streak=best,
                    frac=sum(ok) / len(ok), plan=st.mean(float(x["plan_s"]) for x in r[1:]))
    M = {(a, s): met(a, s) for a in ALGOS for s in SEEDS}
    cols = [("cost", "last best_cost", "%.3f"), ("c25", "cost med last25", "%.3f"), ("fh", "final h (m)", "%.2f"), ("ft", "final tilt", "%.1f"), ("h25", "h last25", "%.3f"), ("t25", "tilt last25", "%.1f"),
            ("streak", "streak", "%.1f"), ("frac", "frac ok", "%.2f"), ("plan", "plan_s", "%.2f")]
    p("\n=== g1_fixed per algo (median / mean over 6 seeds); streak = longest run of cycles with torso_h > 0.9 m and tilt < 10 deg ===")
    p("%-6s" % "algo" + "".join("%-20s" % c[1] for c in cols))
    for a in ALGOS:
        p("%-6s" % a + "".join("%-20s" % ((c[2] % st.median([M[a, s][c[0]] for s in SEEDS])) + " / " + (c[2] % st.mean([M[a, s][c[0]] for s in SEEDS]))) for c in cols))
    for key, nm, fmt in (("t25", "last-25 mean tilt (deg)", "%6.1f"), ("h25", "last-25 mean torso h (m)", "%6.3f"), ("streak", "longest streak (cycles)", "%6d")):
        p(f"\n=== per-seed: {nm} ===\n%-6s" % "algo" + "".join("   s%d    " % s for s in SEEDS))
        for a in ALGOS: p("%-6s" % a + "".join((" " + fmt + "  ") % M[a, s][key] for s in SEEDS))
    p("\n=== per-seed final torso h (m) / tilt (deg) ===")
    for a in ALGOS: p("%-6s" % a + "".join(" %4.2f/%4.0f " % (M[a, s]["fh"], M[a, s]["ft"]) for s in SEEDS))
(OUT / f"{task}_six_summary.txt").write_text("\n".join(L) + "\n")
