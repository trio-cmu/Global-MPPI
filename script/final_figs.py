"""Final six-method figures + tables for PushT / walker / double_cart_pole (read-only on the run data).

    python final_figs.py RUN_DIR [TASKS] [OUT_DIR]

RUN_DIR  a queue folder (script/overnight_queue.sh: pusht/six_seeds_main/*.log, <task>/results/<algo>/s<seed>/...) or, for PushT,
         the packaged data/pusht_results (<algo>/s<seed>/cycles.csv, written by script/pusht_extract_csv.py)
TASKS    comma list of pusht,walker,double_cart_pole (default: those found in RUN_DIR)
OUT_DIR  default RUN_DIR/final_figs.  Writes <task>_final.png and <task>_summary.txt (python 3.6 compatible).
"""
import csv, glob, json, re, statistics as st, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(sys.argv[1])
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else R / "final_figs"
OUT.mkdir(parents=True, exist_ok=True)
ALGOS = ["ps", "mppi", "dial", "mtp", "cbo", "gm"]
NAME = {"gm": "GlobalMPPI (gm)", "ps": "PS", "mppi": "MPPI", "dial": "DIAL", "mtp": "MTP", "cbo": "CBO"}
COL = {"gm": "tab:blue", "ps": "tab:orange", "mppi": "tab:green", "dial": "tab:red", "mtp": "tab:purple", "cbo": "tab:brown"}
SEEDS = range(6)
PFX = {"ps": "ps2000", "mppi": "mppi2000", "dial": "dial2000", "mtp": "mtp2000", "cbo": "cbo2000", "gm": "globalmppi_cacs5new"}


def band(ax, a, y, x=None):
    q25, med, q75 = np.quantile(y, [.25, .5, .75], axis=0)
    x = np.arange(y.shape[1]) if x is None else x
    ax.plot(x, med, label=NAME[a], color=COL[a], lw=2); ax.fill_between(x, q25, q75, color=COL[a], alpha=0.22)


def style(ax, title, ylab, log=False):
    ax.set_title(title); ax.set_xlabel("Cycle"); ax.set_ylabel(ylab); ax.set_xlim(0, 100); ax.grid(True, alpha=.3, lw=.8)
    if log: ax.set_yscale("log")


def leg(ax, loc):
    ax.legend(loc=loc, frameon=True, fancybox=False, framealpha=.9, edgecolor="0.3")


def writer(name):
    L = []
    def p(s=""): L.append(s); print(s)
    return L, p


def fmt(v):
    return "%.3g" % v


# ---------------- PushT ----------------
def pusht():
    src = R / "pusht" / "six_seeds_main"; cost, plan, found = {}, {}, {}
    for a in ALGOS:
        cs, ps_, ids = [], [], []
        for s in SEEDS:
            # a partial batch (SEEDS / METHODS set in overnight_queue.sh) is plotted with what exists
            if not ((src / f"{PFX[a]}_seed{s}.log").exists() if src.is_dir() else (R / a / f"s{s}" / "cycles.csv").exists()):
                continue
            if src.is_dir():   # queue folder: parse the run logs
                t = (src / f"{PFX[a]}_seed{s}.log").read_text(errors="ignore")
                c = {int(float(k)): float(v) for k, v in re.findall(r"Cycle ([\d.]+): Best cost: \[?([-\d.e+]+)", t)}
                pl = {int(float(k)): float(v) for k, v in re.findall(r"Cycle ([\d.]+): plan time: ([\d.]+)s", t)}
            else:              # packaged data: <algo>/s<seed>/cycles.csv with cycle, best_cost, plan_s (empty at cycle 100)
                rows = list(csv.DictReader(open(R / a / f"s{s}" / "cycles.csv")))
                c = {int(r["cycle"]): float(r["best_cost"]) for r in rows}
                pl = {int(r["cycle"]): float(r["plan_s"]) for r in rows if r["plan_s"] != ""}
            assert sorted(c) == list(range(101)), (a, s, len(c))
            assert sorted(pl) == list(range(100)), (a, s, len(pl))
            cs.append([c[i] for i in range(101)]); ps_.append([pl[i] for i in range(100)]); ids.append(s)
        if cs:
            cost[a], plan[a], found[a] = np.array(cs), np.array(ps_), ids
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6))
    ax = axs[0]
    for a in cost: band(ax, a, np.maximum(cost[a], 1e-12))
    style(ax, "PushT: planner cost", "Cost", True); leg(ax, "lower left")
    ax = axs[1]
    for i, a in enumerate(cost):
        f = cost[a][:, 100]; q25, med, q75 = np.quantile(f, [.25, .5, .75])
        ax.scatter(np.full(len(f), i) + np.linspace(-.15, .15, len(f)), f, s=12, color=COL[a], alpha=.35, zorder=2)
        ax.errorbar(i, med, yerr=[[med - q25], [q75 - med]], fmt="o", color=COL[a], ms=8, lw=2.5, capsize=7, capthick=2.5, zorder=4)
    ax.set_xticks(range(len(cost))); ax.set_xticklabels([NAME[a] for a in cost], rotation=20); ax.set_yscale("log"); ax.set_ylabel("Cost at cycle 100")
    ax.set_title("Cost at cycle 100\n(marker = median over the seeds, bars = IQR, faint dots = seeds)"); ax.grid(True, alpha=.3, lw=.8, axis="y")
    ax = axs[2]
    for a in cost: band(ax, a, plan[a][:, 1:], np.arange(1, 100))
    style(ax, "Plan time per cycle (cycle 0 = JIT, omitted)", "plan time [s]", True); leg(ax, "center right")
    plt.tight_layout(); fig.savefig(OUT / "pusht_final.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    L, p = writer("pusht")
    p("=== PushT closed loop, cost = planner best_cost, cycle 0..100; seeds per method: %s ===" % ", ".join("%s %d" % (NAME[a], len(cost[a])) for a in cost))
    p("%-16s %-13s %-12s %-12s %-26s %-18s %-12s" % ("method", "median@100", "mean@100", "std@100", "IQR@100", "plan_s median", "plan_s mean"))
    for a in cost:
        f = cost[a][:, 100]; q = np.quantile(f, [.25, .75]); pl = plan[a][:, 1:]
        p("%-16s %-13s %-12s %-12s %-26s %-18s %-12s" % (NAME[a], fmt(np.median(f)), fmt(f.mean()), fmt(f.std(ddof=1) if len(f) > 1 else 0.0), "%s - %s" % (fmt(q[0]), fmt(q[1])), "%.2f s" % np.median(pl), "%.2f s" % pl.mean()))
    p("\nper-seed cost at cycle 100:")
    for a in cost: p("  %-16s %s" % (NAME[a], "  ".join("s%d %s" % (s, fmt(c)) for s, c in zip(found[a], cost[a][:, 100]))))
    (OUT / "pusht_summary.txt").write_text("\n".join(L) + "\n")


# ---------------- walker / double_cart_pole ----------------
def load(task):
    D = {}
    for a in ALGOS:
        h, rows, fin = [], [], []
        for s in SEEDS:
            d = R / task / "results" / a / f"s{s}"
            h.append(np.squeeze(np.load(glob.glob(str(d / "*best_cost_history*.npy"))[0])))
            rows.append(list(csv.DictReader(open(d / "cycles.csv")))); fin.append(json.load(open(d / "final.json")))
        assert all(len(x) == 101 for x in h) and all(len(r) == 100 for r in rows), a
        D[a] = (np.stack(h), rows, fin)
    return D


def col(rows, k): return np.array([[float(r[k]) for r in rr] for rr in rows])


def ms(f): return "%s / %s ± %s" % (fmt(np.median(f)), fmt(np.mean(f)), fmt(np.std(f, ddof=1)))


def walker():
    D = load("walker"); fig, axs = plt.subplots(1, 3, figsize=(18, 5.6)); ax = axs[0]
    for a in ALGOS: band(ax, a, np.maximum(D[a][0], 1e-12))
    style(ax, "Walker: planner cost", "Cost", True); leg(ax, "upper right")
    for ax, k, t, yl in ((axs[1], "torso_h", "Walker: torso height", "torso height [m]"), (axs[2], "x", "Walker: forward x", "forward x [m]")):
        for a in ALGOS: band(ax, a, col(D[a][1], k), np.arange(100))
        style(ax, t, yl)
    axs[1].axhline(0.6, color="0.4", ls="--", lw=1); axs[1].text(2, 0.62, "fell < 0.6 m", fontsize=8, color="0.3"); leg(axs[2], "upper left")
    plt.tight_layout(); fig.savefig(OUT / "walker_final.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    L, p = writer("walker")
    p("=== walker, 6 seeds. cell = median / mean ± std (ddof 1) over seeds; last-25: cost = median over cycles 76..100 (per seed), states = mean over cycles 75..99 (per seed); fell = final torso h < 0.6 m ===")
    hd = ("method", "cost@100", "cost last-25 median", "torso h last-25 [m]", "x at last cycle [m]", "x last-25 mean [m]", "fell", "plan_s median")
    p("%-16s" % hd[0] + "".join("%-30s" % h for h in hd[1:]))
    for a in ALGOS:
        h, rows, fin = D[a]; c25 = np.array([np.median(x[76:101]) for x in h]); th = col(rows, "torso_h")[:, -25:].mean(1); x = col(rows, "x")
        fell = sum(f["final_torso_h"] < 0.6 for f in fin); pl = col(rows, "plan_s")[:, 1:]
        p("%-16s" % NAME[a] + "".join("%-30s" % c for c in (ms(h[:, 100]), ms(c25), ms(th), ms(x[:, -1]), ms(x[:, -25:].mean(1)), "%d/6" % fell, "%.2f s" % np.median(pl))))
    p("\nper-seed final x / torso h:")
    for a in ALGOS: p("  %-16s %s" % (NAME[a], "  ".join("s%d %.2f/%.2f" % (s, f["final_x"], f["final_torso_h"]) for s, f in enumerate(D[a][2]))))
    (OUT / "walker_summary.txt").write_text("\n".join(L) + "\n")


def dcp():
    D = load("double_cart_pole"); fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.6)); ax = axs[0]
    for a in ALGOS: band(ax, a, np.maximum(D[a][0], 1e-12))
    style(ax, "Double cart-pole: planner cost", "Cost", True); leg(ax, "lower left")
    ax = axs[1]
    for a in ALGOS: band(ax, a, col(D[a][1], "tip_z"), np.arange(100))
    ax.axhline(4.0, color="k", ls="--", lw=1); ax.text(2, 4.03, "target 4", fontsize=8)
    style(ax, "Double cart-pole: tip z", "tip z [m]"); leg(ax, "lower right")
    plt.tight_layout(); fig.savefig(OUT / "double_cart_pole_final.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    L, p = writer("dcp")
    p("=== double_cart_pole, 6 seeds. cell = median / mean ± std (ddof 1); upright = final |tip z - 4| < 0.2 (as in final.json); last-25: cost = median over cycles 76..100, tip z = mean over cycles 75..99 (per seed) ===")
    hd = ("method", "cost@100", "cost last-25 median", "tip z@end", "tip z last-25 mean", "upright", "plan_s median")
    p("%-16s" % hd[0] + "".join("%-30s" % h for h in hd[1:]))
    for a in ALGOS:
        h, rows, fin = D[a]; c25 = np.array([np.median(x[76:101]) for x in h]); tz = col(rows, "tip_z"); pl = col(rows, "plan_s")[:, 1:]
        up = sum(bool(f["upright"]) for f in fin)
        p("%-16s" % NAME[a] + "".join("%-30s" % c for c in (ms(h[:, 100]), ms(c25), ms(tz[:, -1]), ms(tz[:, -25:].mean(1)), "%d/6" % up, "%.2f s" % np.median(pl))))
    p("\nper-seed final tip z (upright?):")
    for a in ALGOS: p("  %-16s %s" % (NAME[a], "  ".join("s%d %.2f%s" % (s, f["final_tip_z"], "*" if f["upright"] else "") for s, f in enumerate(D[a][2]))))
    (OUT / "double_cart_pole_summary.txt").write_text("\n".join(L) + "\n")


ALL = {"pusht": pusht, "walker": walker, "double_cart_pole": dcp}
want = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else [t for t in ALL if (R / t).is_dir() or (t == "pusht" and (R / "ps" / "s0" / "cycles.csv").exists())]
for t in want:
    ALL[t](); print()
