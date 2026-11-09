"""PushT: final cost and planning time of PS / MPPI / DIAL / CACS (closed loop, 6 seeds) and GCS
(plan once, executed open loop in MuJoCo by run_gcs_open_loop.py). Every final cost is the same
number: the PushT cost of the current plan over the next 1 s from the state MuJoCo reached, as
pusht.py logs it ("Best cost").

    python plot_gcs_comparison.py [out.png]                 # default MuJoCo PushT
    python plot_gcs_comparison.py --gcs-model [out.png]    # everything in GCS's own quasi-static model

Inputs:
    ../six_seeds_main/<method>_seed<k>.log      sampling planners (final cost = cycle 100)
    ../gcs_model_seeds/<method>_seed<k>.log     the same in GCS's model, global_mppi/tasks/pusht_gcs.py (--gcs-model)
    gcs_open_loop_<demo|matched|gcs_model>.json  GCS replays (final cost = the cycle after the plan ends)
    gcs_solve_times.jsonl                       6 timed GCS solves, demo parameters (time_gcs_solves.py)
    gcs_matched_timing.json                     the GCS solve with MuJoCo's friction and mass
GCS planning time = relaxed GCS + binary flows + feasible solve (formulation left out), as before.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

here = Path(__file__).resolve().parent
QS = "--gcs-model" in sys.argv
argv = [a for a in sys.argv if a != "--gcs-model"]
out = Path(argv[1]) if len(argv) > 1 else here / f"pusht_gcs_comparison{'_gcs_model' if QS else ''}.png"
SEEDS = here.parent / ("gcs_model_seeds" if QS else "six_seeds_main")
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SAMPLERS = [("ps2000", "PS\n(2000 × 5)", "#52514e"), ("mppi2000", "MPPI\n(2000 × 5)", "#1baf7a"),
            ("dial2000", "DIAL\n(2000 × 5)", "#eb6834"), ("globalmppi_cacs5new", "CACS (ours)", "#d43d51")]
GCS_BLUE = "#2a78d6"

rows = []  # (label, color, hatch, final costs, time per solve, total planning time per run, note)
curves = []  # (legend label, color, line style, cost per cycle (runs, cycles))
for key, label, color in SAMPLERS:
    finals, per_solve, total, C = [], [], [], []
    for path in sorted(SEEDS.glob(f"{key}_seed*.log")):
        text = path.read_text()
        cost = {int(float(c)): float(v) for c, v in re.findall(r"Cycle ([\d.]+): Best cost: \[?([-\d.e+]+)", text)}
        plan = {int(float(c)): float(v) for c, v in re.findall(r"Cycle ([\d.]+): plan time: ([\d.]+)s", text)}
        if max(cost, default=0) < 100:  # run not finished
            continue
        finals.append(cost[max(cost)])
        C.append([cost[c] for c in range(101)])
        t = np.array([plan[c] for c in sorted(plan) if c > 0])  # cycle 0 = JIT compilation
        per_solve.append(t.mean())
        total.append(t.sum())
    if len(finals) < 6:  # fewer seeds than the caption says
        label += f"\n[{len(finals)} seed{'s' if len(finals) != 1 else ''} only]"
    rows.append((label, color, None, np.array(finals), np.array(per_solve), np.array(total),
                 f"{len(finals)} seeds, 100 cycles"))
    if C:
        curves.append((f"{label.splitlines()[0]} ({len(C)} seed{'s' if len(C) != 1 else ''})", color, "-", np.array(C)))

gcs_demo_t = np.array([r["relaxed_gcs_time"] + r["binary_flows_time"] + r["feasible_time"]
                       for r in map(json.loads, (here / "gcs_solve_times.jsonl").read_text().splitlines())])
gcs_times = {"demo": gcs_demo_t}
matched_timing = here / "gcs_matched_timing.json"
if matched_timing.exists():
    m = json.loads(matched_timing.read_text())
    gcs_times["matched"] = np.array([m["relaxed_gcs_time"] + m["binary_flows_time"] + m["feasible_time"]])
GCS_NAMES = {"demo": "GCS\ndemo params", "matched": "GCS\nmatched params", "gcs_model": "GCS\nexecuted"}
gcs_times["gcs_model"] = gcs_demo_t  # the same plan as "demo"; the model is GCS's own
FLOOR = None  # where a final cost of exactly 0 is drawn on the log axis
if QS:
    # Two GCS plans, both in GCS's model: with GCS's own (demo) cost, and re-solved with our task cost and
    # our pusher speed limit (planning-through-contact/pusht_test/run_gcs_task_cost.py). Shown as planned:
    # the pose each plan ends at (the goal, cost 0) and its solve time.
    t_task = json.loads((here / "gcs_task_cost_vlim_timing.json").read_text())
    t_task = np.array([t_task["relaxed_gcs_time"] + t_task["binary_flows_time"] + t_task["feasible_time"]])
    for tag, name, color, t_gcs in (("gcs_model", "GCS\n(its cost)", "#8fb8ea", gcs_demo_t),
                                    ("task_cost_vlim_gcs_model", "GCS\n(our cost)", GCS_BLUE, t_task)):
        d = json.loads((here / f"gcs_open_loop_{tag}.json").read_text())
        rows.append((name, color, "..", np.array([d["planned_final_cost"]]), t_gcs, t_gcs,
                     "plan ends\nat the goal"))
for tag in (() if QS else ("demo", "matched")):
    path = here / f"gcs_open_loop_{tag}.json"
    if not path.exists():
        continue
    runs = json.loads(path.read_text())["runs"]
    for s, hatch in (("1", None), ("10.875", "//")):
        if s not in runs:
            continue
        r = runs[s]
        when = f"{r['duration']:.1f} s plan" if s == "1" else f"{r['duration']:.1f} s (×{float(s):.3g})"
        rows.append((f"{GCS_NAMES[tag]}\n{when}", GCS_BLUE if tag == "matched" else "#8fb8ea", hatch,
                     np.array([r["final_cost"]]), gcs_times[tag], gcs_times[tag],
                     f"off {100 * r['final_pos_err']:.1f} cm\n{np.degrees(r['final_ang_err']):.0f}°"))
        if s == "10.875":  # the plan played in the same 2 s / 100 cycles
            name = GCS_NAMES[tag].replace("\n", " ")
            curves.append((f"{name} open loop, plan played in 2 s", GCS_BLUE if tag == "matched" else "#8fb8ea", "--",
                           np.array([r["costs"][:101]])))

fig, axes = plt.subplots(1, 3, figsize=(24, 6.8), dpi=150, gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
fig.patch.set_facecolor(SURFACE)
ax_c = axes[0]  # cost curve, as plot_closed_loop_seeds.py: geometric mean +- 1 std (log) over seeds
ax_c.set_facecolor(SURFACE)
ax_c.grid(True, color=GRID, lw=0.8)
ax_c.tick_params(colors=INK_2, length=0)
for side in ("top", "right"):
    ax_c.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax_c.spines[side].set_color("#c9c8c2")
for name, color, ls, C in curves:
    logC = np.log10(np.maximum(C, 1e-6))
    gm = logC.mean(0)
    ax_c.plot(np.arange(C.shape[1]), 10 ** gm, color=color, lw=2.2, ls=ls, label=name)
    if len(C) > 1:
        gs = logC.std(0, ddof=1)
        ax_c.fill_between(np.arange(C.shape[1]), 10 ** (gm - gs), 10 ** (gm + gs), color=color, alpha=0.15)
ax_c.set_yscale("log")
ax_c.set_xlim(0, 100)
ax_c.set_xlabel("control cycle (20 ms each)", color=INK_2)
ax_c.set_ylabel("cost of the current plan over the next 1 s (log scale)", color=INK_2)
ax_c.set_title("Cost curve: geometric mean ± 1 std (log) over seeds", loc="left", fontsize=11, color=INK, weight="bold")
ax_c.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower left")
panels = [(axes[1], 3, "Final cost (log scale)", "cost of the plan over the next 1 s at the end"),
          (axes[2], 4, "Planning time per solve (log scale)", "s: one MPC cycle (ours), one full solve (GCS)")]
fmt = lambda v, i: (f"{v:.1e}" if v < 1e-2 else f"{v:.3g}") if i == 3 else f"{v:.3g} s"
for ax, i, title, ylabel in panels:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    lo, hi = min(r[i].min() for r in rows if r[i].max() > 0), max(r[i].max() for r in rows)
    FLOOR = lo / 4
    for k, row in enumerate(rows):
        v = row[i]
        if i == 3 and v.max() <= 0:  # exactly 0: no bar on a log axis; mark it at the bottom
            ax.text(k, FLOOR * 1.5, "0\n" + row[6], ha="center", va="bottom", fontsize=8.5, color=INK)
            ax.scatter([k], [FLOOR], color=row[1], s=40, marker="v", zorder=4, clip_on=False)
            continue
        ax.bar(k, v.mean(), width=0.62, color=row[1], alpha=0.3, hatch=row[2], edgecolor=row[1], lw=0)
        if len(v) > 1:
            low = max(v.mean() - v.std(ddof=1), v.min())
            ax.errorbar(k, v.mean(), yerr=[[v.mean() - low], [v.std(ddof=1)]], color=row[1], capsize=6, lw=1.8, zorder=3)
        ax.scatter(k + np.linspace(-0.14, 0.14, len(v)) if len(v) > 1 else [k], v, color=row[1], s=22,
                   edgecolor=SURFACE, lw=1, zorder=4)
        note = f"\n{row[6]}" if i == 3 and row[2] is not False and k >= len(SAMPLERS) else ""
        ax.text(k, v.max() * 1.5, fmt(v.mean(), i) + note, ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, hi * (40 if i == 3 else 12))
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=7.8, color=INK_2)
    ax.axvline(len(SAMPLERS) - 0.5, color="#c9c8c2", lw=1, ls="--")
    ax.set_title(title, loc="left", fontsize=11, color=INK, weight="bold")
    ax.set_ylabel(ylabel, color=INK_2)

if QS:
    title = "PushT in GCS's own dynamics: sampling MPC in closed loop vs GCS planned once"
    caption = ("Every method plans and runs in GCS's quasi-static pusher-slider model (Hogan & Rodriguez; "
               "global_mppi/tasks/pusht_gcs.py) with GCS's parameters: ellipsoidal limit surface about the COM "
               "(c = 0.3), pusher friction 0.1, the block moves only while pushed, kinematic pusher (|v| ≤ 1 m/s "
               "per axis). Same start (block 0.1, 0.1, 1.3 rad; pusher at the origin), goal and cost. PS / MPPI / "
               "DIAL / CACS replan every 20 ms for 100 cycles = 2 s (6 seeds, mean ± std, dots = seeds; time "
               "leaves out cycle 0, JIT). GCS 'its cost' = GCS's default demo objective (arc length, velocity and force "
               "regularization); 'our cost' = re-solved with our task cost (50 |pos|² + 15·2(1 − cos θ), summed over "
               "the plan) and our pusher speed limit (1 m/s), GCS's terms off. Both keep the goal as a constraint; "
               "GCS final cost = the pose its plan ends at (the goal, 0). "
               "GCS time = relaxed GCS + rounding (its cost: 6 solves).")
else:
    title = "PushT: sampling MPC in closed loop vs GCS planned once and executed open loop, all scored in MuJoCo"
    caption = ("Same start (block 0.1, 0.1, 1.3 rad; pusher at the origin), same goal, same MuJoCo model (1 kHz) "
               "and the same cost. PS / MPPI / DIAL / CACS replan every 20 ms for 100 cycles (6 seeds, mean ± std, "
               "dots = seeds; time leaves out cycle 0, JIT). GCS plans once with its quasi-static model; the pusher "
               "tracks the plan (position control), with no feedback on the block. 'demo params' = GCS's demo "
               "friction and mass (pusher 0.1, table 0.5); 'matched params' = re-solved with MuJoCo's friction 1.0 "
               "and mass 0.056 kg. Hatched = plan played 10.9× faster to fit the same 2 s. GCS time = relaxed GCS + "
               "rounding (6 solves for demo params).")
fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK, weight="bold")
fig.text(0.01, 0.915, caption, fontsize=8.6, color=INK_2, va="top", wrap=True)
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(out, facecolor=SURFACE)
fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
for r in rows:
    print(f"{r[0].replace(chr(10), ' '):45s} final {r[3].mean():.4g}  per solve {r[4].mean():.3g} s  total {r[5].mean():.3g} s  {r[6]}")
print("RESULT saved", out)
