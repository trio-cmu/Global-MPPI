"""GCS on PushT as in the paper: plan once, then execute the plan in MuJoCo (no replanning).

    python run_gcs_open_loop.py                  # GCS demo parameters; speed-ups 1, 2, 5 and 10.875 (= 2 s)
    python run_gcs_open_loop.py --plan gcs_matched_traj.npz --tag matched   # MuJoCo's friction and mass
    python run_gcs_open_loop.py --gcs-model --tag gcs_model --speedups 1 10.875  # in GCS's own quasi-static model

The plan is the rounded GCS solution from planning-through-contact (pusht_test/export_gcs_traj.py,
copied here as gcs_global_mppi_traj.npz): the same start as pusht.py (block qpos [0.1, 0.1, 1.3],
pusher at the origin), 21.75 s at GCS's own timing, sampled every 0.01 s. GCS plans with its
quasi-static model; MuJoCo moves the block with full second-order dynamics.

Execution, the same as pusht.py's closed loop except that the plan never changes:
  * simulation at 1 kHz (100 solver iterations), one control cycle = 20 ms (50 Hz);
  * the pusher tracks the planned pusher path, ctrl = v_ref + KP (p_ref - p), clipped to the
    actuator range (a position-controlled robot, as on hardware). There is no feedback on the block;
  * --speedups s plays the plan s times faster (21.75 s / s).
Each cycle logs, in pusht.py's format, "Best cost" = the PushT cost of following the rest of the plan
for the next 1 s from the current state, rolled out in the planning model (dt 0.01) with the same
sum dt * running_cost + terminal_cost as Global-MPPI, so it is the number our methods log.
The run continues until the plan has ended plus one cycle; "final cost" is the last cycle's cost.

Outputs (next to this file): gcs_open_loop_<tag>_x<s>.log and gcs_open_loop_<tag>.json (cost curves, final cost,
cost at cycle 100, final block position / angle error, pusher tracking error).
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from global_mppi import ROOT as GM_ROOT
from global_mppi.tasks.pusht import PushT

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Open-loop GCS plan executed in MuJoCo PushT")
parser.add_argument("--plan", type=Path, default=here / "gcs_global_mppi_traj.npz")
parser.add_argument("--speedups", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.875])
parser.add_argument("--tag", default="demo", help="name of the plan in the output files")
parser.add_argument("--gcs-model", action="store_true",
                    help="execute in GCS's own quasi-static model (global_mppi/tasks/pusht_gcs.py) instead of MuJoCo")
parser.add_argument("--kp", type=float, default=100.0, help="pusher position gain (1/s); 100 keeps the pusher within ~0.1 mm of the plan")
args = parser.parse_args()

FREQUENCY = 50.0
PUSHER_BODY_POS = np.array([0.0, 0.1])  # pusher qpos is relative to this
plan = np.load(args.plan)
T_PLAN, P_PLAN = plan["t"], plan["pusher"]
V_PLAN = np.gradient(P_PLAN, T_PLAN, axis=0)
V_PLAN[-1] = 0.0

if args.gcs_model:
    from global_mppi.tasks.pusht_gcs import PushTGCS
    task = PushTGCS()
else:
    task = PushT()
H = int(round(1.0 / task.dt))  # 100 planning steps = 1 s, as pusht.py
sim_model = mujoco.MjModel.from_xml_path(GM_ROOT + "/models/pusht/scene.xml")
BLOCK = mujoco.mj_name2id(sim_model, mujoco.mjtObj.mjOBJ_BODY, "block")
sim_model.opt.timestep = 0.001
sim_model.opt.iterations = 100
sim_model.opt.ls_iterations = 50
sim_steps = max(int(1.0 / FREQUENCY / sim_model.opt.timestep), 1)
u_min, u_max = np.asarray(task.u_min), np.asarray(task.u_max)


def reference(t, s):
    """Pusher position and velocity at execution time(s) t for speed-up s; holds the end pose."""
    tp = np.clip(np.asarray(t) * s, 0.0, T_PLAN[-1])
    p = np.stack([np.interp(tp, T_PLAN, P_PLAN[:, k]) for k in range(2)], -1)
    v = s * np.stack([np.interp(tp, T_PLAN, V_PLAN[:, k]) for k in range(2)], -1)
    return p, np.where((np.asarray(t) * s >= T_PLAN[-1])[..., None], 0.0, v)


@jax.jit
def horizon_cost(state, p_ref, v_ref):
    """Global-MPPI's cost of following (p_ref, v_ref) for H planning steps from state."""
    def step(x, ref):
        u = jnp.clip(ref[1] + args.kp * (ref[0] - (x.qpos[3:5] + PUSHER_BODY_POS)), task.u_min, task.u_max)
        x = task.step(task.model, x.replace(ctrl=u))
        return x, task.dt * task.running_cost(x, u)
    final, costs = jax.lax.scan(step, state, (p_ref, v_ref))
    return costs.sum() + task.terminal_cost(final)


def block_error(data):
    """Block position error (m) and angle error (rad) w.r.t. the goal pose (0, 0, 0)."""
    (x, y), th = data.xpos[BLOCK][:2], data.qpos[2]
    return float(np.hypot(x, y)), float(abs((th + np.pi) % (2 * np.pi) - np.pi))


results = {}
for s in args.speedups:
    data = mujoco.MjData(sim_model)
    data.qpos[:] = [0.1, 0.1, 1.3, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(sim_model, data)
    duration = T_PLAN[-1] / s
    cycles = int(np.ceil(duration * FREQUENCY)) + 1
    costs, track_err, saturated = [], 0.0, 0
    log = open(here / f"gcs_open_loop_{args.tag}_x{s:g}.log", "w")
    for cycle in range(cycles):
        state = mjx.put_data(sim_model, data).replace(mocap_pos=data.mocap_pos, mocap_quat=data.mocap_quat)
        p_h, v_h = reference(data.time + task.dt * np.arange(H), s)
        cost = float(horizon_cost(state, jnp.asarray(p_h, jnp.float32), jnp.asarray(v_h, jnp.float32)))
        costs.append(cost)
        log.write(f"Cycle {cycle}: Best cost: {cost}\nCycle {cycle}: plan time: 0.0s\n")
        for _ in range(sim_steps):
            p_ref, v_ref = reference(data.time, s)
            pusher = data.qpos[3:5] + PUSHER_BODY_POS
            u = v_ref + args.kp * (p_ref - pusher)
            saturated += int(np.any(np.abs(u) > u_max))
            data.ctrl[:] = np.clip(u, u_min, u_max)
            task.sim_step(sim_model, data) if args.gcs_model else mujoco.mj_step(sim_model, data)
            track_err = max(track_err, float(np.linalg.norm(p_ref - (data.qpos[3:5] + PUSHER_BODY_POS))))
    log.close()
    pos_err, ang_err = block_error(data)
    results[f"{s:g}"] = dict(
        speedup=s, duration=duration, cycles=cycles, costs=costs, final_cost=costs[-1],
        cost_at_cycle_100=costs[min(100, len(costs) - 1)], final_pos_err=pos_err, final_ang_err=ang_err,
        final_block_pose=[*data.xpos[BLOCK][:2].tolist(), float(data.qpos[2])], max_pusher_tracking_err=track_err,
        saturated_fraction=saturated / (cycles * sim_steps))
    print(f"x{s:g}: plan {duration:.2f} s ({cycles} cycles), final cost {costs[-1]:.4g}, "
          f"cost at cycle 100 {results[f'{s:g}']['cost_at_cycle_100']:.4g}, block error {100 * pos_err:.2f} cm / "
          f"{np.degrees(ang_err):.2f} deg, max pusher tracking error {100 * track_err:.2f} cm, "
          f"saturated {100 * saturated / (cycles * sim_steps):.1f}% of steps", flush=True)

out = here / f"gcs_open_loop_{args.tag}.json"
# GCS's own claim: the cost (as above) of holding the plan's last pose, the pose GCS planned to reach
b = plan["block_body"][-1]
planned = float(2 * (50 * np.sum(b[:2] ** 2) + 15 * ((b[2] + np.pi) % (2 * np.pi) - np.pi) ** 2))
out.write_text(json.dumps(dict(plan=str(args.plan.name), kp=args.kp, gcs_model=args.gcs_model,
                               planned_final_cost=planned, planned_final_pose=b.tolist(), runs=results), indent=1))
print("RESULT saved", out)
