"""GCS (planning-through-contact) plan for the Global-MPPI PushT task, with the user's cost.

Cost (added to every contact and non-contact mode of the GCS problem; no other cost terms):

    sum over knot points  h * [ W_POS |p_COM - p_COM,goal|^2 + W_ANG * 2 (1 - cos th) + W_CLOSE |p_block - p_pusher|^2 ]
    + W_PATH * pusher arc length in non-contact modes            (upstream `pusher_arc_length`; 0 = off)

h = time of the mode / number of its knot points. p_block is the block body origin (pusht.xml), as
_close_to_block_err in global_mppi/tasks/pusht.py. The goal pose is a hard constraint (GCS).

Problem = Global-MPPI PushT (as global_mppi/tasks/pusht_gcs.py): the T of pusht.xml (crossbar 0.06 x 0.02 on an
exposed 0.02 x 0.06 stem) with its frame at the COM, 0.012857 above the body origin; pusher radius 0.01;
slider-pusher friction 0.1; limit-surface integration constant 0.3; start block body (0.1, 0.1, 1.3), pusher at
(0, 0.1) (pusher qpos 0); goal block body (0, 0, 0), pusher back at (0, 0.1).
Mode timing as upstream get_default_plan_config(use_case="demo"): 1.5 s / 4 knots per contact mode,
0.75 s / 3 knots per non-contact mode.

Upstream planning-through-contact (a3998f2) is used unmodified: the cost terms are added by wrapping its
_define_costs / add_cost_to_vertex methods here.

    MOSEKLM_LICENSE_FILE=~/mosek/mosek.lic python gcs_plan.py --tag NAME [--w-pos 50 --w-ang 15 --w-close 10 --w-path 0]
    -> gcs_<NAME>_traj.npz (Global-MPPI format: t, qpos, block_body, block_com, pusher, com_in_body, dt), gcs_<NAME>_info.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from pydrake.solvers import Binding, MathematicalProgram, QuadraticCost
from pydrake.symbolic import Expression

# Drake 1.52 compatibility (upstream targets drake ~1.32): AddLinearConstraint(expression, lb, ub) no longer
# accepts 1-element arrays as bounds (band_sparse_semidefinite_relaxation.py passes evaluator().lower_bound()).
_add_linear_constraint = MathematicalProgram.AddLinearConstraint


def _add_linear_constraint_compat(self, *a, **kw):
    if len(a) == 3 and isinstance(a[0], Expression) and np.ndim(a[1]) == 1 and np.size(a[1]) == 1:
        a = (a[0], float(np.asarray(a[1])[0]), float(np.asarray(a[2])[0]))
    return _add_linear_constraint(self, *a, **kw)


MathematicalProgram.AddLinearConstraint = _add_linear_constraint_compat

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.non_collision import NonCollisionMode
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    ContactConfig, ContactCost, NonCollisionCost, PlanarPlanConfig, PlanarPushingStartAndGoal,
    PlanarSolverParams, SliderPusherSystemConfig,
)
from planning_through_contact.planning.planar.planar_pushing_planner import PlanarPushingPlanner

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--w-pos", type=float, default=50.0)
ap.add_argument("--w-ang", type=float, default=15.0)
ap.add_argument("--w-close", type=float, default=10.0)
ap.add_argument("--w-path", type=float, default=0.0)
ap.add_argument("--time-contact", type=float, default=1.5)
ap.add_argument("--time-non-collision", type=float, default=0.75)
ap.add_argument("--rounding-steps", type=int, default=20)
args = ap.parse_args()
here = Path(__file__).resolve().parent

COM_IN_BODY = np.array([0.0, 0.06 * 0.02 * 0.03 / (0.02 * 0.08 + 0.06 * 0.02)])  # 0.012857, as pusht_gcs.py
BODY_IN_COM = -COM_IN_BODY  # block body origin in the COM frame (rotation invariant offset along y)
PUSHER_HOME = np.array([0.0, 0.1])


class PushTGeometry(TPusher2d):
    """TPusher2d with the pusht.xml T; upstream hard-codes the COM offset of its own default T."""

    @property
    def com_offset(self):
        # frame origin relative to the crossbar center: the crossbar center is 0.03 above the body origin
        return np.array([0.0, COM_IN_BODY[1] - 0.03]).reshape((-1, 1))


def com_pose(x, y, th):
    c, s = np.cos(th), np.sin(th)
    return np.array([x, y]) + np.array([[c, -s], [s, c]]) @ COM_IN_BODY


GOAL_COM = com_pose(0.0, 0.0, 0.0)


def task_cost_expr(p_WB, cos_th, sin_th, p_BP):
    """W_POS |p_COM - goal|^2 + W_ANG 2 (1 - cos th) + W_CLOSE |p_BP - body origin|^2 (all in drake symbolic).

    2 (1 - cos th) is written as (1 - cos th)^2 + sin^2 th, equal on cos^2 + sin^2 = 1 (which GCS enforces): a sum of
    squares, so GCS accepts it as a non-negative edge/vertex cost (a linear term in cos th is rejected)."""
    d = p_WB.flatten() - GOAL_COM
    e = p_BP.flatten() - BODY_IN_COM
    return (args.w_pos * (d @ d) + args.w_ang * ((1.0 - cos_th) ** 2 + sin_th ** 2)
            + args.w_close * (e @ e))


# --- contact modes: one quadratic cost per knot point, relaxed together with the rest of the mode
_face_define_costs = FaceContactMode._define_costs


def face_define_costs(self):
    _face_define_costs(self)
    v, n = self.variables, self.num_knot_points
    h = self.time_in_mode / n
    self.costs["task"] = []
    for k in range(n):
        expr = h * task_cost_expr(v.p_WBs[k], v.cos_ths[k], v.sin_ths[k], v.p_BPs[k])
        self.costs["task"].append(self.prog_wrapper.add_quadratic_cost(k, k, expr))


FaceContactMode._define_costs = face_define_costs

# --- non-contact modes: the block is still; one convex quadratic cost on the mode's variables
_nc_define_cost = NonCollisionMode._define_cost
_nc_add_cost = NonCollisionMode.add_cost_to_vertex


def nc_define_cost(self):
    _nc_define_cost(self)
    self.task_cost = None
    if self.num_knot_points == 1:  # source / target vertices: no duration
        return
    v, n = self.variables, self.num_knot_points
    h = self.time_in_mode / n
    expr = sum(h * task_cost_expr(v.p_WB, v.cos_th, v.sin_th, v.p_BPs[k]) for k in range(n))
    cost = self.prog.AddQuadraticCost(expr, is_convex=True)
    Q = cost.evaluator().Q() + np.eye(cost.evaluator().Q().shape[0]) * 1e-10  # as upstream: Cholesky-safe
    cost.evaluator().UpdateCoefficients(Q, cost.evaluator().b(), cost.evaluator().c())
    self.task_cost = cost
    self.costs["task"] = [cost]


def nc_add_cost(self, vertex):
    _nc_add_cost(self, vertex)
    if getattr(self, "task_cost", None) is not None:
        idx = self.get_variable_indices_in_gcs_vertex(self.task_cost.variables())
        vertex.AddCost(Binding[QuadraticCost](self.task_cost.evaluator(), vertex.x()[idx]))


NonCollisionMode._define_cost = nc_define_cost
NonCollisionMode.add_cost_to_vertex = nc_add_cost

# --- problem
geometry = PushTGeometry(box_1=Box2d(0.06, 0.02), box_2=Box2d(0.02, 0.06))
slider = RigidBody("t_pusher", geometry, 0.1)
config = PlanarPlanConfig(
    dynamics_config=SliderPusherSystemConfig(
        slider=slider, pusher_radius=0.01, friction_coeff_slider_pusher=0.1,
        friction_coeff_table_slider=0.5, integration_constant=0.3,
    ),
    num_knot_points_contact=4,
    num_knot_points_non_collision=3,
    use_band_sparsity=True,
    contact_config=ContactConfig(cost=ContactCost(), lam_min=0.0, lam_max=1.0),
    non_collision_cost=NonCollisionCost(pusher_arc_length=args.w_path if args.w_path > 0 else None),
    continuity_on_pusher_velocity=True,
    allow_teleportation=False,
    time_in_contact=args.time_contact,
    time_non_collision=args.time_non_collision,
)
start_com = com_pose(0.1, 0.1, 1.3)
config.start_and_goal = PlanarPushingStartAndGoal(
    slider_initial_pose=PlanarPose(start_com[0], start_com[1], 1.3),
    slider_target_pose=PlanarPose(GOAL_COM[0], GOAL_COM[1], 0.0),
    pusher_initial_pose=PlanarPose(PUSHER_HOME[0], PUSHER_HOME[1], 0.0),
    pusher_target_pose=PlanarPose(PUSHER_HOME[0], PUSHER_HOME[1], 0.0),
)
solver_params = PlanarSolverParams(
    rounding_steps=args.rounding_steps, solver="mosek", measure_solve_time=True,
    assert_result=False, assert_nan_values=True, print_cost=True, print_path=True,
    nonl_round_major_feas_tol=1e-5, nonl_round_minor_feas_tol=1e-5, nonl_round_opt_tol=1e-5,
)

t0 = time.perf_counter()
planner = PlanarPushingPlanner(config)
planner.config.start_and_goal = config.start_and_goal
planner.formulate_problem()
t1 = time.perf_counter()
path = planner.plan_path(solver_params)
t2 = time.perf_counter()
if path is None:
    raise SystemExit("GCS found no feasible path")

traj = path.to_traj(rounded=True)
relaxed_cost = float(path.result.get_optimal_cost())
rounded_cost = float(path.rounded_result.get_optimal_cost())
print(f"path: {path.get_path_names()}")
print(f"relaxed cost {relaxed_cost:.4f}, rounded cost {rounded_cost:.4f}, formulate {t1 - t0:.1f} s, solve+round {t2 - t1:.1f} s")

# --- export in the Global-MPPI plan format (0.01 s samples)
dt = 0.01
ts = np.arange(0.0, traj.end_time + 1e-9, dt)
ts[-1] = min(ts[-1], traj.end_time)
com = np.array([traj.get_p_WB(t).flatten() for t in ts])
R = [traj.get_R_WB(t) for t in ts]
th = np.unwrap([np.arctan2(r[1, 0], r[0, 0]) for r in R])
body = com - np.array([r @ COM_IN_BODY for r in R])
pusher = np.array([traj.get_p_WP(t).flatten() for t in ts])
qpos = np.column_stack([body, th, pusher - PUSHER_HOME])
np.savez(here / f"gcs_{args.tag}_traj.npz", t=ts, qpos=qpos, block_body=np.column_stack([body, th]),
         block_com=np.column_stack([com, th]), pusher=pusher, com_in_body=COM_IN_BODY, dt=dt)
info = dict(tag=args.tag, weights=dict(w_pos=args.w_pos, w_ang=args.w_ang, w_close=args.w_close, w_path=args.w_path),
            time_contact=args.time_contact, time_non_collision=args.time_non_collision,
            path=path.get_path_names(), relaxed_cost=relaxed_cost, rounded_cost=rounded_cost,
            formulate_s=t1 - t0, solve_and_round_s=t2 - t1, duration_s=float(traj.end_time),
            start_qpos=qpos[0].tolist(), final_qpos=qpos[-1].tolist())
(here / f"gcs_{args.tag}_info.json").write_text(json.dumps(info, indent=1))
print("saved", here / f"gcs_{args.tag}_traj.npz")
