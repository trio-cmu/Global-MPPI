#!/usr/bin/env python3
"""Table of every tuning run under runs/tune_mppi/<id>/seed<N>/ (result.json + qpos_history.npy; numpy only).

  python summarize_tune.py [--root runs/tune_mppi] [--out runs/tune_mppi/table.md]

first <10mm/<3deg = first time the T pose is inside BOTH tolerances (not necessarily held 1 s); max overshoot =
largest position error after that moment (how far it is pushed back out); '-' = never inside / not applicable.
Runs shorter than 60 s of sim time (skipped / cut) are marked with their sim time.
"""
import argparse
import glob
import json
import os

import numpy as np


def row(d):
    r = json.load(open(os.path.join(d, "result.json")))
    h = np.load(os.path.join(d, "qpos_history.npy"))
    e = (r.get("args") or {}).get("entry") or {}
    k = r.get("config_algo", {})
    p = dict(e.get("params") or e.get("mppi") or {})
    noise = p.get("noise_level", (p.get("noise_level_vmax", k.get("noise_level_vmax", 0)) or 0) * 0.1 if "noise_level" not in k else k["noise_level"])
    if "noise_level" in p:
        noise = p["noise_level"]
    pos = np.linalg.norm(h[:, 1:3], axis=1) * 1000
    ang = np.abs(np.degrees((h[:, 3] + np.pi) % (2 * np.pi) - np.pi))
    inside = np.where((pos < 10.0) & (ang < 3.0))[0]
    enter = f"{h[inside[0], 0]:.1f}" if len(inside) else "-"
    over = f"{pos[inside[0]:].max():.0f}" if len(inside) else "-"
    start = "left (-0.335,-0.078)" if e.get("pusher_start") else "standard (-0.133,-0.010)"
    return {
        "id": e.get("id", os.path.basename(os.path.dirname(d))), "algo": r["algo"], "seed": r["seed"],
        "params": f"N={k.get('num_samples')} noise={noise:g} T={k.get('temperature')} H={k.get('plan_horizon')} "
                  f"it={k.get('iterations', 1)} K={k.get('num_knots')}"
                  + (f" beta={k.get('beta_opt_iter')}/{k.get('beta_horizon')}" if r["algo"] == "dial" else "")
                  + f" appr={e.get('w_approach', 0)}",
        "start": start, "sim": f"{r['sim_time']:.1f}" + ("*" if r["sim_time"] < 59.5 else ""),
        "final": f"{r['final_pos_mm']:.1f} mm / {r['final_ang_deg']:.2f}°", "min": f"{r['min_pos_mm']:.1f} mm / {r['min_ang_deg']:.2f}°",
        "enter": enter, "over": over, "plan": f"{r['plan_mean_s']:.3f} / {r['plan_p95_s']:.3f}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/tune_mppi")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(a.root, "*", "seed*")), key=os.path.getmtime)
    rows = [row(d) for d in dirs if os.path.exists(os.path.join(d, "result.json"))]
    lines = ["| config | algo | key parameters | pusher start | seed | sim time (s) | final pos / ang | min pos / ang | first <10mm&<3° (s) | max pos after entering (mm) | plan mean / p95 (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['algo']} | {r['params']} | {r['start']} | {r['seed']} | {r['sim']} | {r['final']} | {r['min']} | "
                     f"{r['enter']} | {r['over']} | {r['plan']} |")
    lines.append("\n`*` = the run ended before 60 s of sim time (skipped / cut); its 'final' is the error when it stopped.")
    text = "\n".join(lines)
    print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
