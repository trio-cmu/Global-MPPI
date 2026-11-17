"""Six-algorithm figure and tables for a walker / double_cart_pole batch (python 3.6 compatible).

    python plot_task.py {walker|double_cart_pole} RESULTS_DIR OUT_DIR

RESULTS_DIR/<algo>/s<seed>/{cycles.csv, final.json, *best_cost_history*.npy}, algo in ps mppi dial mtp cbo gm.
Writes OUT_DIR/<task>_cost_six.png (median over seeds + 25-75 % band, log y, cost = best_cost, cycles 0..100) and
OUT_DIR/<task>_six_summary.txt (median cost at cycle 100, final-state tables, plan time). Seeds missing are skipped.
"""
import csv
import glob
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

task, res, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
ALGOS = ["ps", "mppi", "dial", "mtp", "cbo", "gm"]
NAME = {"gm": "Global-MPPI", "ps": "Predictive Sampling", "mppi": "MPPI", "dial": "DIAL-MPC", "mtp": "MTP", "cbo": "CBO"}
COL = {"gm": "tab:blue", "ps": "tab:orange", "mppi": "tab:green", "dial": "tab:red", "mtp": "tab:purple", "cbo": "tab:brown"}
SEEDS = range(6)
L = []


def p(s=""):
    L.append(s)
    print(s)


def run_dirs(a):
    return [(s, os.path.join(res, a, "s%d" % s)) for s in SEEDS if os.path.exists(os.path.join(res, a, "s%d" % s, "final.json"))]


hist = {}
for a in ALGOS:
    h = []
    for s, d in run_dirs(a):
        f = glob.glob(os.path.join(d, "*best_cost_history*.npy"))
        if f:
            h.append(np.squeeze(np.load(f[0])))
    if h:
        n = min(len(x) for x in h)
        hist[a] = np.stack([x[:n] for x in h])
p("=== %s: %s ===" % (task, ", ".join("%s %d seeds" % (a, len(run_dirs(a))) for a in ALGOS)))
if hist:
    fig, ax = plt.subplots(figsize=(6.4, 6))
    for a in ALGOS:
        if a not in hist:
            continue
        c = hist[a]
        pos = np.maximum(c, 1e-12)
        q25, med, q75 = np.quantile(pos, [0.25, 0.5, 0.75], axis=0)
        x = np.arange(c.shape[1])
        ax.plot(x, med, label=NAME[a], color=COL[a], lw=2)
        ax.fill_between(x, q25, q75, color=COL[a], alpha=0.22)
    ax.set_title(task.replace("_", " "))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, lw=0.8)
    ax.legend(loc="lower left", frameon=True, fancybox=False, framealpha=0.9, edgecolor="0.3")
    plt.tight_layout()
    fig.savefig(os.path.join(out, "%s_cost_six.png" % task), dpi=300, bbox_inches="tight")
    p("\nmedian cost at the last iteration (IQR):")
    for a in ALGOS:
        if a in hist:
            q = np.quantile(hist[a][:, -1], [0.25, 0.5, 0.75])
            p("  %-12s %.4f (%.4f-%.4f)" % (NAME[a], q[1], q[0], q[2]))


def rows(a, s, d):
    return list(csv.DictReader(open(os.path.join(d, "cycles.csv"))))


def mean(v):
    return st.mean(v) if v else float("nan")


def sd(v):
    return st.stdev(v) if len(v) > 1 else 0.0


p("\nplan time per cycle (mean over seeds, without cycle 0):")
for a in ALGOS:
    v = [mean([float(r["plan_s"]) for r in rows(a, s, d)[1:]]) for s, d in run_dirs(a)]
    if v:
        p("  %-6s %.3f s" % (a, mean(v)))

if task == "walker":
    p("\n=== final state after the last cycle (per seed: forward x m / torso height m / torso angle deg); fell = final height < 0.6 m ===")
    p("%-6s" % "algo" + "".join("  s%d                " % s for s in SEEDS) + "  mean x ± std | fell")
    for a in ALGOS:
        fs = {s: json.load(open(os.path.join(d, "final.json"))) for s, d in run_dirs(a)}
        if not fs:
            continue
        cells = ["%6.2f/%5.2f/%6.1f" % (fs[s]["final_x"], fs[s]["final_torso_h"], fs[s]["final_torso_angle_deg"]) if s in fs else "      -          " for s in SEEDS]
        xs = [f["final_x"] for f in fs.values()]
        fell = sum(1 for f in fs.values() if f["final_torso_h"] < 0.6)
        p("%-6s " % a + "  ".join(cells) + "  %.2f ± %.2f | %d/%d" % (mean(xs), sd(xs), fell, len(fs)))
    p("\nlast-25-cycle means (median / mean over seeds): torso height m, |torso angle| deg, mean forward speed m/s (sim time = cycles / 50 Hz)")
    for a in ALGOS:
        hs, an, sp = [], [], []
        for s, d in run_dirs(a):
            r = rows(a, s, d)
            hs.append(mean([float(x["torso_h"]) for x in r[-25:]]))
            an.append(mean([abs(float(x["torso_angle_deg"])) for x in r[-25:]]))
            sp.append((float(r[-1]["x"]) - float(r[0]["x"])) / (0.02 * max(len(r) - 1, 1)))
        if hs:
            p("  %-6s h %.2f / %.2f   |angle| %.1f / %.1f   speed %.2f / %.2f" % (a, st.median(hs), mean(hs), st.median(an), mean(an), st.median(sp), mean(sp)))
else:
    p("\n=== final state (per seed: tip z m / cart x m); upright = |tip z - 4| < 0.2 ===")
    p("%-6s" % "algo" + "".join("  s%d          " % s for s in SEEDS) + "  upright")
    for a in ALGOS:
        fs = {s: json.load(open(os.path.join(d, "final.json"))) for s, d in run_dirs(a)}
        if not fs:
            continue
        cells = ["%5.2f/%5.2f" % (fs[s]["final_tip_z"], fs[s]["final_cart_x"]) if s in fs else "     -     " for s in SEEDS]
        p("%-6s " % a + "  ".join(cells) + "  %d/%d" % (sum(1 for f in fs.values() if f["upright"]), len(fs)))
    p("\nlast-25-cycle mean tip z (median / mean over seeds), and the share of the last 25 cycles with |tip z - 4| < 0.2")
    for a in ALGOS:
        tz, fr = [], []
        for s, d in run_dirs(a):
            r = rows(a, s, d)[-25:]
            z = [float(x["tip_z"]) for x in r]
            tz.append(mean(z))
            fr.append(sum(1 for v in z if abs(v - 4.0) < 0.2) / float(len(z)))
        if tz:
            p("  %-6s tip z %.2f / %.2f   upright share %.2f / %.2f" % (a, st.median(tz), mean(tz), st.median(fr), mean(fr)))
gms = [json.load(open(os.path.join(d, "final.json"))).get("solve_paths") for s, d in run_dirs("gm")]
p("\ngm solve_paths per run: %s" % ("all gpu" if gms and all(g and set(g) == {"gpu"} for g in gms) else gms))
open(os.path.join(out, "%s_six_summary.txt" % task), "w").write("\n".join(L) + "\n")
