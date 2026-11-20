"""Metrics and figures of the obstacle_field task.

    python script/obstacle_field_metrics.py RESULTS_DIR [--out-dir DIR]      per-run metrics + summary + trajectory figure
    python script/obstacle_field_metrics.py --plot-layout OUT.png [--layout-seed 19]   layout, start, goal, shortest and second-best route
    python script/obstacle_field_metrics.py --plot-candidates OUT.png --seeds 11 19 22   several layouts side by side

RESULTS_DIR holds <method>/s<seed>/{cycles.csv, final.json, args.json} (what examples/obstacle_field.py writes; the
queue uses <queue>/obstacle_field/results). Written to --out-dir (default RESULTS_DIR/../figures): metrics.csv (one
row per run), summary.txt (one row per method), reference_path.json, obstacle_field_trajectories.png.

Per run, recomputed from cycles.csv (one position per cycle = every 0.02 s; the polyline through them, closed with the
final position of final.json, is the path):
  path_length    length of the polyline (m); path_ratio = path_length / reference length
  reached        final distance to the goal < 0.02 m
  collision      min clearance < 0; clearance = distance from the particle surface to the nearest circle surface
                 (|p - c| - r - 0.01 m), taken on the polyline segments, not only at the samples
  min_clearance  the minimum of that (m)
  final_distance m;  plan_s: mean plan time per cycle without cycle 0 (the JIT compile), s
Reference: the shortest collision-free path of the particle centre on a 2.5 mm grid (Dijkstra with 16 neighbours,
then line-of-sight shortening), keeping 5 mm free gap to every circle and wall (global_mppi/tasks/obstacle_field_layout.py).
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("obstacle_field_layout", ROOT / "global_mppi/tasks/obstacle_field_layout.py")
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)  # numpy only: no jax, no GPU

METHODS = {  # tag: (label, colour)
    "ps": ("PS", "tab:orange"),
    "mppi": ("MPPI", "tab:green"),
    "dial": ("DIAL", "tab:red"),
    "mtp": ("MTP", "tab:purple"),
    "cbo": ("CBO", "tab:brown"),
    "gm": ("GlobalMPPI", "tab:blue"),
}
GOAL_RADIUS = 0.02


def segment_clearance(xy, centers, radii):
    """Minimum over the polyline xy (N,2) of the particle-surface clearance to the circles (exact, per segment)."""
    best = np.inf
    for p, q in zip(xy[:-1], xy[1:]) if len(xy) > 1 else [(xy[0], xy[0])]:
        d = q - p
        denom = float(d @ d)
        for c, r in zip(centers, radii):
            t = 0.0 if denom == 0 else float(np.clip((c - p) @ d / denom, 0.0, 1.0))
            best = min(best, float(np.linalg.norm(p + t * d - c)) - r - L.PARTICLE_RADIUS)
    return best


def layout_of(run_dir, layout_seed):
    a = run_dir / "args.json"
    if a.exists():
        info = json.load(open(a))
        if "obstacle_centers" in info:
            return np.array(info["obstacle_centers"]), np.array(info["obstacle_radii"]), info.get("layout_seed")
    c, r = L.make_layout(layout_seed)
    return c, r, layout_seed


def run_metrics(run_dir, ref_len, layout_seed):
    rows = list(csv.DictReader(open(run_dir / "cycles.csv")))
    if not rows:
        return None
    xy = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    plan = np.array([float(r["plan_s"]) for r in rows])
    fin = json.load(open(run_dir / "final.json")) if (run_dir / "final.json").exists() else None
    if fin is not None:
        xy = np.vstack([xy, [fin["final_x"], fin["final_y"]]])
    centers, radii, _ = layout_of(run_dir, layout_seed)
    length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    clear = segment_clearance(xy, centers, radii)
    dist = float(np.linalg.norm(xy[-1] - L.GOAL))
    return dict(
        method=run_dir.parent.name, seed=int(run_dir.name[1:]), finished=fin is not None, cycles=len(rows),
        path_length=length, path_ratio=length / ref_len, reached=dist < GOAL_RADIUS, collision=clear < 0.0,
        min_clearance=clear, final_distance=dist, plan_s=float(plan[1:].mean()) if len(plan) > 1 else float("nan"), xy=xy,
    )


def draw_layout(ax, centers, radii, ref_path=None):
    ax.add_patch(plt.Rectangle((-L.HALF, -L.HALF), 2 * L.HALF, 2 * L.HALF, fill=False, ec="0.35", lw=2))
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc="0.75", ec="0.3"))
    ax.plot(*L.START, "o", c="tab:blue", ms=8, zorder=5, label="start")
    ax.plot(*L.GOAL, "*", c="tab:green", ms=14, zorder=5, label="goal")
    if ref_path is not None:
        ax.plot(ref_path[:, 0], ref_path[:, 1], "-", c="k", lw=1.8, zorder=4, label="grid-search shortest path")
    ax.set_xlim(-0.31, 0.31)
    ax.set_ylim(-0.31, 0.31)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def draw_candidate(ax, seed, centers, radii, info):
    draw_layout(ax, centers, radii, info["path"])
    ax.plot(info["second_path"][:, 0], info["second_path"][:, 1], "--", c="tab:red", lw=1.8, zorder=4, label="second-best route")
    ax.add_patch(plt.Circle(L.START, L.PARTICLE_RADIUS, fc="tab:blue", ec="k", zorder=6))  # the particle, to scale (r = 1 cm)
    ax.set_title(f"layout_seed={seed}: shortest {info['shortest']:.3f} m (x{info['shortest'] / info['straight']:.3f}), "
                 f"second {info['second']:.3f} m (+{(info['second'] - info['shortest']) * 100:.1f} cm)", fontsize=8)


def plot_layout(out, layout_seed):
    """Layout, start (with the particle to scale), goal, shortest and second-best route of one seed."""
    plot_candidates(out, [layout_seed])


def plot_candidates(out, seeds):
    fig, axes = plt.subplots(1, len(seeds), figsize=(5.2 * len(seeds), 5.6), squeeze=False)
    for ax, seed in zip(axes[0], seeds):
        k, centers, radii, info = L.find_layout(seed)  # full search (no cached hint), a few seconds
        draw_candidate(ax, seed, centers, radii, info)
        print(f"layout_seed {seed}: candidate #{k}, straight {info['straight']:.4f} m, shortest {info['shortest']:.4f} m, "
              f"second-best route {info['second']:.4f} m")
    axes[0][0].legend(loc="lower center", ncol=2, fontsize=6)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--layout-seed", type=int, default=19, help="layout of runs without args.json, and of --plot-layout")
    ap.add_argument("--plot-layout", type=Path, metavar="OUT.png")
    ap.add_argument("--plot-candidates", type=Path, metavar="OUT.png", help="side by side with --seeds")
    ap.add_argument("--seeds", type=int, nargs="+", default=[19])
    a = ap.parse_args()
    if a.plot_candidates:
        plot_candidates(a.plot_candidates, a.seeds)
        return
    if a.plot_layout:
        plot_layout(a.plot_layout, a.layout_seed)
        return
    if a.results is None:
        ap.error("RESULTS_DIR or --plot-layout is needed")
    out = a.out_dir or a.results.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    runs = sorted(p.parent for p in a.results.glob("*/s*/cycles.csv"))
    if not runs:
        raise SystemExit(f"no <method>/s<seed>/cycles.csv under {a.results}")
    centers, radii, seed = layout_of(runs[0], a.layout_seed)
    path, ref_len = L.shortest_path(centers, radii)
    json.dump({"layout_seed": seed, "reference_length_m": ref_len, "reference_path": path.tolist()}, open(out / "reference_path.json", "w"), indent=1)
    res = [m for m in (run_metrics(r, ref_len, a.layout_seed) for r in runs) if m]
    cols = ["method", "seed", "finished", "cycles", "path_length", "path_ratio", "reached", "collision", "min_clearance", "final_distance", "plan_s"]
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for m in sorted(res, key=lambda m: (list(METHODS).index(m["method"]) if m["method"] in METHODS else 99, m["seed"])):
            w.writerow([f"{m[c]:.5f}" if isinstance(m[c], float) else m[c] for c in cols])
    lines = [f"obstacle_field layout_seed={seed}; reference (grid-search shortest path) {ref_len:.4f} m; straight line 0.5000 m",
             f"{'method':<11}{'runs':>5}{'reached':>9}{'collided':>9}{'path (m)':>16}{'path/ref':>10}{'final dist (m)':>16}{'min clear (m)':>15}{'plan s/cycle':>14}"]
    for tag in [t for t in METHODS if any(m["method"] == t for m in res)]:
        g = [m for m in res if m["method"] == tag]
        ms = lambda k: f"{np.mean([m[k] for m in g]):.3f}+-{np.std([m[k] for m in g]):.3f}"  # noqa: E731
        lines.append(f"{METHODS[tag][0]:<11}{len(g):>5}{sum(m['reached'] for m in g):>7}/{len(g):<1}{sum(m['collision'] for m in g):>7}/{len(g):<1}"
                     f"{ms('path_length'):>16}{np.mean([m['path_ratio'] for m in g]):>10.3f}{ms('final_distance'):>16}{np.mean([m['min_clearance'] for m in g]):>15.4f}{np.nanmean([m['plan_s'] for m in g]):>14.2f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_layout(ax, centers, radii, path)
    for m in res:
        lab, col = METHODS.get(m["method"], (m["method"], "k"))
        ax.plot(m["xy"][:, 0], m["xy"][:, 1], c=col, lw=1, alpha=0.7, label=lab if m["seed"] == min(x["seed"] for x in res if x["method"] == m["method"]) else None)
    ax.legend(fontsize=8, ncol=2, loc="lower center")
    ax.set_title(f"obstacle_field trajectories, layout_seed={seed}")
    fig.tight_layout()
    fig.savefig(out / "obstacle_field_trajectories.png", dpi=150)


if __name__ == "__main__":
    main()
