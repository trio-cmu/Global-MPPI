"""Final six-method figures + tables for PushT / walker / double_cart_pole / bugtrap / g1_walk (read-only on the run data).

    python final_figs.py RUN_DIR [TASKS] [OUT_DIR]

RUN_DIR  a queue folder (script/overnight_queue.sh: pusht/six_seeds_main/*.log, <task>/results/<algo>/s<seed>/...) or packaged data:
         data/pusht_results (script/pusht_extract_csv.py) or data/walker_results, data/bugtrap_results, data/g1_walk_results (script/package_task_results.py)
TASKS    comma list of pusht,walker,double_cart_pole,bugtrap,g1_walk (default: those found in RUN_DIR)
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
    """<RUN>/<task>/results/<algo>/s<seed>/ (queue folder) or <RUN>/<algo>/s<seed>/ (packaged data, script/package_task_results.py).
    Cost history: the *best_cost_history*.npy if present, else the best_cost column of cycles.csv (101 rows: cycle 100 has only
    best_cost). States and plan times: the first 100 rows of cycles.csv."""
    base = R / task / "results" if (R / task / "results").is_dir() else R
    D = {}
    for a in ALGOS:
        h, rows, fin, ids = [], [], [], []
        for s in SEEDS:
            d = base / a / f"s{s}"
            if not (d / "cycles.csv").exists() or not (d / "final.json").exists():
                continue   # a partial batch (SEEDS / METHODS set in overnight_queue.sh) is plotted with what exists
            rr = list(csv.DictReader(open(d / "cycles.csv")))
            f = glob.glob(str(d / "*best_cost_history*.npy"))
            h.append(np.squeeze(np.load(f[0])) if f else np.array([float(r["best_cost"]) for r in rr]))
            rows.append(rr[:100]); fin.append(json.load(open(d / "final.json"))); ids.append(s)
        if h:
            assert all(len(x) == 101 for x in h) and all(len(r) == 100 for r in rows), a
            D[a] = (np.stack(h), rows, fin, ids)
    return D


def col(rows, k): return np.array([[float(r[k]) for r in rr] for rr in rows])


def ms(f): return "%s / %s ± %s" % (fmt(np.median(f)), fmt(np.mean(f)), fmt(np.std(f, ddof=1) if len(f) > 1 else 0.0))


def walker():
    D = load("walker"); fig, axs = plt.subplots(1, 3, figsize=(18, 5.6)); ax = axs[0]
    for a in D: band(ax, a, np.maximum(D[a][0], 1e-12))
    style(ax, "Walker: planner cost", "Cost", True); leg(ax, "upper right")
    for ax, k, t, yl in ((axs[1], "torso_h", "Walker: torso height", "torso height [m]"), (axs[2], "x", "Walker: forward x", "forward x [m]")):
        for a in D: band(ax, a, col(D[a][1], k), np.arange(100))
        style(ax, t, yl)
    axs[1].axhline(0.6, color="0.4", ls="--", lw=1); axs[1].text(2, 0.62, "fell < 0.6 m", fontsize=8, color="0.3"); leg(axs[2], "upper left")
    plt.tight_layout(); fig.savefig(OUT / "walker_final.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    L, p = writer("walker")
    p("=== walker, seeds per method: %s. cell = median / mean ± std (ddof 1) over seeds; last-25: cost = median over cycles 76..100 (per seed), states = mean over cycles 75..99 (per seed); fell = final torso h < 0.6 m ===" % ", ".join("%s %d" % (NAME[a], len(D[a][2])) for a in D))
    hd = ("method", "cost@100", "cost last-25 median", "torso h last-25 [m]", "x at last cycle [m]", "x last-25 mean [m]", "fell", "plan_s median")
    p("%-16s" % hd[0] + "".join("%-30s" % h for h in hd[1:]))
    for a in D:
        h, rows, fin, ids = D[a]; c25 = np.array([np.median(x[76:101]) for x in h]); th = col(rows, "torso_h")[:, -25:].mean(1); x = col(rows, "x")
        fell = sum(f["final_torso_h"] < 0.6 for f in fin); pl = col(rows, "plan_s")[:, 1:]
        p("%-16s" % NAME[a] + "".join("%-30s" % c for c in (ms(h[:, 100]), ms(c25), ms(th), ms(x[:, -1]), ms(x[:, -25:].mean(1)), "%d/%d" % (fell, len(fin)), "%.2f s" % np.median(pl))))
    p("\nper-seed final x / torso h:")
    for a in D: p("  %-16s %s" % (NAME[a], "  ".join("s%d %.2f/%.2f" % (s, f["final_x"], f["final_torso_h"]) for s, f in zip(D[a][3], D[a][2]))))
    (OUT / "walker_summary.txt").write_text("\n".join(L) + "\n")


def dcp():
    D = load("double_cart_pole"); fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.6)); ax = axs[0]
    for a in D: band(ax, a, np.maximum(D[a][0], 1e-12))
    style(ax, "Double cart-pole: planner cost", "Cost", True); leg(ax, "lower left")
    ax = axs[1]
    for a in D: band(ax, a, col(D[a][1], "tip_z"), np.arange(100))
    ax.axhline(4.0, color="k", ls="--", lw=1); ax.text(2, 4.03, "target 4", fontsize=8)
    style(ax, "Double cart-pole: tip z", "tip z [m]"); leg(ax, "lower right")
    plt.tight_layout(); fig.savefig(OUT / "double_cart_pole_final.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    L, p = writer("dcp")
    p("=== double_cart_pole, seeds per method: %s. cell = median / mean ± std (ddof 1); upright = final |tip z - 4| < 0.2 (as in final.json); last-25: cost = median over cycles 76..100, tip z = mean over cycles 75..99 (per seed) ===" % ", ".join("%s %d" % (NAME[a], len(D[a][2])) for a in D))
    hd = ("method", "cost@100", "cost last-25 median", "tip z@end", "tip z last-25 mean", "upright", "plan_s median")
    p("%-16s" % hd[0] + "".join("%-30s" % h for h in hd[1:]))
    for a in D:
        h, rows, fin, ids = D[a]; c25 = np.array([np.median(x[76:101]) for x in h]); tz = col(rows, "tip_z"); pl = col(rows, "plan_s")[:, 1:]
        up = sum(bool(f["upright"]) for f in fin)
        p("%-16s" % NAME[a] + "".join("%-30s" % c for c in (ms(h[:, 100]), ms(c25), ms(tz[:, -1]), ms(tz[:, -25:].mean(1)), "%d/%d" % (up, len(fin)), "%.2f s" % np.median(pl))))
    p("\nper-seed final tip z (upright?):")
    for a in D: p("  %-16s %s" % (NAME[a], "  ".join("s%d %.2f%s" % (s, f["final_tip_z"], "*" if f["upright"] else "") for s, f in zip(D[a][3], D[a][2]))))
    (OUT / "double_cart_pole_summary.txt").write_text("\n".join(L) + "\n")


def g1_walk():
    D = load("g1_walk"); fig, axs = plt.subplots(2, 2, figsize=(13, 10.5)); axs = axs.ravel()
    for a in D: band(axs[0], a, np.maximum(D[a][0], 1e-12))
    style(axs[0], "G1 walk: planner cost", "Cost", True); leg(axs[0], "upper right")
    for ax, k, t, yl in ((axs[1], "ref_base_err", "G1 walk: base distance to the reference", "base error [m]"), (axs[2], "z", "G1 walk: base height", "base height z [m]"), (axs[3], "tilt_deg", "G1 walk: base tilt", "tilt [deg]")):
        for a in D: band(ax, a, col(D[a][1], k), np.arange(100))
        style(ax, t, yl)
    axs[2].axhline(0.5, color="0.4", ls="--", lw=1); axs[2].text(2, 0.52, "fell < 0.5 m", fontsize=8, color="0.3")
    plt.tight_layout(); fig.savefig(OUT / "g1_walk_final.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    L, p = writer("g1_walk")
    p("=== g1_walk, seeds per method: %s. cell = median / mean ± std (ddof 1) over seeds; last-25: cost = median over cycles 76..100 (per seed), states = mean over cycles 75..99 (per seed); fell = final base height < 0.5 m ===" % ", ".join("%s %d" % (NAME[a], len(D[a][2])) for a in D))
    hd = ("method", "cost@100", "cost last-25 median", "base error mean over run [m]", "base error last-25 [m]", "base height last-25 [m]", "tilt last-25 [deg]", "fell", "plan_s median")
    p("%-16s" % hd[0] + "".join("%-30s" % h for h in hd[1:]))
    for a in D:
        h, rows, fin, ids = D[a]; c25 = np.array([np.median(x[76:101]) for x in h]); pl = col(rows, "plan_s")[:, 1:]
        p("%-16s" % NAME[a] + "".join("%-30s" % c for c in (ms(h[:, 100]), ms(c25), ms(col(rows, "ref_base_err").mean(1)), ms(col(rows, "ref_base_err")[:, -25:].mean(1)), ms(col(rows, "z")[:, -25:].mean(1)),
                                                              ms(col(rows, "tilt_deg")[:, -25:].mean(1)), "%d/%d" % (sum(f["fell"] for f in fin), len(fin)), "%.2f s" % np.median(pl))))
    p("\nper-seed final base error [m] / height [m] / tilt [deg]  (* = fell):")
    for a in D: p("  %-16s %s" % (NAME[a], "  ".join("s%d %.2f/%.2f/%.0f%s" % (s, f["final_ref_base_err"], f["final_z"], f["final_tilt_deg"], "*" if f["fell"] else "") for s, f in zip(D[a][3], D[a][2]))))
    (OUT / "g1_walk_summary.txt").write_text("\n".join(L) + "\n")


def bugtrap():
    """xy paths over the cup (to scale, 6 seeds per method) and distance to the goal vs cycle (median, IQR)."""
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Circle, Rectangle
    base = R / "bugtrap" / "results" if (R / "bugtrap" / "results").is_dir() else R
    D = {}
    for a in ALGOS:
        rr, fin, ids = [], [], []
        for s in SEEDS:
            d = base / a / f"s{s}"
            if not (d / "cycles.csv").exists() or not (d / "final.json").exists():
                continue   # a partial batch is plotted with what exists
            rows = [r for r in csv.DictReader(open(d / "cycles.csv")) if r["x"] != ""]; f = json.load(open(d / "final.json"))   # the packaged row 100 has no state
            assert len(rows) > 0, (a, s)
            rr.append(rows); fin.append(f); ids.append(s)
        if rr:
            xy = np.array([[(float(r["x"]), float(r["y"])) for r in rows] + [(f["final_x"], f["final_y"])] for rows, f in zip(rr, fin)])   # (n, 101, 2)
            dist = np.array([[float(r["distance"]) for r in rows] + [f["final_distance"]] for rows, f in zip(rr, fin)])                       # (n, 101)
            plan = np.array([[float(r["plan_s"]) for r in rows[1:]] for rows in rr])
            D[a] = dict(xy=xy, dist=dist, plan=plan, fin=fin, ids=ids)
    walls = [((-.31, 0), (.02, .3)), ((.31, 0), (.02, .3)), ((0, -.31), (.3, .02)), ((0, .31), (.3, .02))]   # outer arena (centre, half size), models/particle_navigation
    inner = [((.1, 0), (.01, .11)), ((0, -.1), (.11, .01)), ((0, .1), (.11, .01))]                           # the U (cup): back wall at x = 0.1, open to the left
    fig = plt.figure(figsize=(17, 7.2)); gs = GridSpec(2, 4, width_ratios=[1, 1, 1, 1.35], figure=fig)
    for k, a in enumerate(ALGOS):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        for c, h in walls: ax.add_patch(Rectangle((c[0] - h[0], c[1] - h[1]), 2 * h[0], 2 * h[1], color="0.75", zorder=1))
        for c, h in inner: ax.add_patch(Rectangle((c[0] - h[0], c[1] - h[1]), 2 * h[0], 2 * h[1], color="0.35", zorder=2))
        ax.add_patch(Circle((0.25, 0), 0.05, fill=False, ls="--", lw=.8, color="k", zorder=2)); ax.add_patch(Circle((0.25, 0), 0.02, fill=False, lw=.8, color="k", zorder=2))
        ax.plot([-0.15], [0], "k^", ms=6, zorder=5); ax.plot([0.25], [0], "k*", ms=9, zorder=5)
        if a in D:
            for p_, f in zip(D[a]["xy"], D[a]["fin"]):
                ax.plot(p_[:, 0], p_[:, 1], color=COL[a], lw=1.4, alpha=.8, zorder=4); ax.plot(p_[-1, 0], p_[-1, 1], "o", color=COL[a], ms=4, zorder=6)
            ax.set_title("%s  (%d seeds, left the cup %d)" % (NAME[a], len(D[a]["ids"]), sum(f["left_cup"] for f in D[a]["fin"])), fontsize=10)
        else:
            ax.set_title("%s (no data)" % NAME[a], fontsize=10)
        ax.set_xlim(-0.2, 0.33); ax.set_ylim(-0.33, 0.33); ax.set_aspect("equal"); ax.grid(True, alpha=.3, lw=.6)
        ax.set_xlabel("x [m]" if k >= 3 else ""); ax.set_ylabel("y [m]" if k % 3 == 0 else "")
    ax = fig.add_subplot(gs[:, 3])
    for a in D: band(ax, a, D[a]["dist"])
    ax.axhline(0.02, color="0.4", ls="--", lw=1); ax.axhline(0.05, color="0.4", ls=":", lw=1); ax.text(2, 0.022, "2 cm", fontsize=8, color="0.3"); ax.text(2, 0.052, "5 cm", fontsize=8, color="0.3")
    style(ax, "Distance to the goal", "distance [m]"); leg(ax, "upper right")
    fig.suptitle("bugtrap: the point mass starts at the open end of the cup (triangle), the goal (star; circles 2 / 5 cm) is behind its closed end. Paths to scale, dots = final positions", fontsize=10)
    plt.tight_layout(); fig.savefig(OUT / "bugtrap_final.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    L, p = writer("bugtrap")
    p("=== bugtrap, seeds per method: %s. left_cup = the final position is outside the cup; reached = final distance below 2 cm / 5 cm; distance in m ===" % ", ".join("%s %d" % (NAME[a], len(D[a]["ids"])) for a in D))
    hd = ("method", "final distance median", "mean ± std", "left the cup", "reached 2 cm", "reached 5 cm", "plan_s median")
    p("%-16s" % hd[0] + "".join("%-24s" % h for h in hd[1:]))
    for a in D:
        f = D[a]["dist"][:, -1]; n = len(f); fin = D[a]["fin"]
        p("%-16s" % NAME[a] + "".join("%-24s" % c for c in (fmt(np.median(f)), "%s ± %s" % (fmt(np.mean(f)), fmt(np.std(f, ddof=1) if n > 1 else 0.0)), "%d/%d" % (sum(x["left_cup"] for x in fin), n),
                                                              "%d/%d" % (int((f < 0.02).sum()), n), "%d/%d" % (int((f < 0.05).sum()), n), "%.2f s" % np.median(D[a]["plan"]))))
    p("\nper-seed final distance:")
    for a in D: p("  %-16s %s" % (NAME[a], "  ".join("s%d %.3f%s" % (s, f["final_distance"], "" if f["left_cup"] else "*") for s, f in zip(D[a]["ids"], D[a]["fin"]))))
    p("(* = final position inside the cup)")
    (OUT / "bugtrap_summary.txt").write_text("\n".join(L) + "\n")


ALL = {"pusht": pusht, "walker": walker, "double_cart_pole": dcp, "bugtrap": bugtrap, "g1_walk": g1_walk}
want = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else [t for t in ALL if (R / t).is_dir() or (t == "pusht" and (R / "ps" / "s0" / "cycles.csv").exists())]
for t in want:
    try:
        ALL[t](); print()
    except FileNotFoundError as e:   # a task without (complete) data does not stop the others
        print("%s: skipped, missing file: %s" % (t, e))
