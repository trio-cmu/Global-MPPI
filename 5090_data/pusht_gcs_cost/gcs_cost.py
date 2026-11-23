"""The GCS PushT cost, patched into the benchmarks branch's PushT task (everything else unchanged).

    running  l(x, u) = 50 |p - goal|^2 + 15 * 2 (1 - cos th) + 10 * |pusher velocity| * [pusher off the T]
    terminal l_T(x)  = 50 |p - goal|^2 + 15 * 2 (1 - cos th)

Summed as dt * l over a rollout, the last term is 10 x the distance the pusher travels while not touching
the T (GCS's pusher_arc_length). Same as global_mppi/tasks/pusht.py on pre-release (pusher_gap, CONTACT_TOL
and the cost copied from there); position and orientation errors are the benchmark task's own sensors.
"""
import jax
import jax.numpy as jnp

from global_mppi.tasks.pusht import PushT

W_POS, W_ANG, W_PATH = 50.0, 15.0, 10.0
# The T as two boxes in the block body frame (pusht.xml): (center, half size)
T_BOXES = ((jnp.array([0.0, 0.03]), jnp.array([0.03, 0.01])),   # crossbar
           (jnp.array([0.0, 0.0]), jnp.array([0.01, 0.04])))    # stem
PUSHER_HOME = jnp.array([0.0, 0.1])  # pusher qpos is relative to this
PUSHER_RADIUS = 0.01
CONTACT_TOL = 1e-3  # the pusher counts as touching the T when its gap is at most this (m)


def pusher_gap(qpos):
    """Distance from the pusher surface to the T (negative when penetrating), from qpos."""
    c, s = jnp.cos(qpos[2]), jnp.sin(qpos[2])
    d = qpos[3:5] + PUSHER_HOME - qpos[:2]
    p = jnp.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])  # pusher center in the block frame

    def box_sd(center, half):
        q = jnp.abs(p - center) - half
        return jnp.linalg.norm(jnp.maximum(q, 0.0)) + jnp.minimum(jnp.max(q), 0.0)

    return jnp.minimum(*[box_sd(*b) for b in T_BOXES]) - PUSHER_RADIUS


def pose_cost(self, state):
    position_err = self._get_position_err(state)
    orientation_err = self._get_orientation_err(state)
    return (W_POS * jnp.sum(jnp.square(position_err))
            + W_ANG * 2.0 * (1.0 - jnp.cos(jnp.linalg.norm(orientation_err))))  # |orientation_err| = |th|


def running_cost(self, state, control):
    off_contact = pusher_gap(state.qpos) > CONTACT_TOL
    pusher_speed = jnp.sqrt(jnp.sum(jnp.square(state.qvel[3:5])) + 1e-12)
    return pose_cost(self, state) + W_PATH * off_contact * pusher_speed


def terminal_cost(self, state):
    return pose_cost(self, state)


def apply():
    PushT.pose_cost = pose_cost
    PushT.running_cost = running_cost
    PushT.terminal_cost = terminal_cost
