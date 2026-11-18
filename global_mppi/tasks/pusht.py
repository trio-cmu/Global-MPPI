from typing import Dict

import jax
import jax.numpy as jnp
import jaxlie
import mujoco
from mujoco import mjx

from global_mppi import ROOT
from global_mppi.task_base import Task

# The T as two boxes in the block body frame (pusht.xml): (center, half size)
T_BOXES = ((jnp.array([0.0, 0.03]), jnp.array([0.03, 0.01])),   # crossbar
           (jnp.array([0.0, 0.0]), jnp.array([0.01, 0.04])))    # stem
PUSHER_HOME = jnp.array([0.0, 0.1])  # pusher qpos is relative to this
PUSHER_RADIUS = 0.01
CONTACT_TOL = 1e-3  # the pusher counts as touching the T when its gap is at most this (m)


def pusher_gap(qpos: jax.Array) -> jax.Array:
    """Distance from the pusher surface to the T (negative when penetrating), from qpos."""
    c, s = jnp.cos(qpos[2]), jnp.sin(qpos[2])
    d = qpos[3:5] + PUSHER_HOME - qpos[:2]
    p = jnp.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])  # pusher center in the block frame

    def box_sd(center, half):
        q = jnp.abs(p - center) - half
        return jnp.linalg.norm(jnp.maximum(q, 0.0)) + jnp.minimum(jnp.max(q), 0.0)

    return jnp.minimum(*[box_sd(*b) for b in T_BOXES]) - PUSHER_RADIUS


class PushT(Task):
    """Push a T-shaped block to a desired pose."""

    def __init__(self, close_to_block_weight: float = 0.0, pusher_path_weight: float = 10.0) -> None:
        """Load the MuJoCo model and set task parameters.

        The cost is the one GCS plans with in pusht_test/run_gcs_task_cost.py (--angle cos
        --path-weight 0 --pusher-path-weight 10): pose cost 50 |p_COM - goal COM|^2 + 15 * 2 (1 - cos θ)
        plus, while the pusher is off the T, pusher_path_weight * pusher speed (summed over time: the
        pusher's path length, GCS's pusher_arc_length).

        Args:
            close_to_block_weight: weight of the squared pusher-to-block distance
                in the running cost. 0 (the default) keeps the original task:
                then every rollout that never touches the block costs the same.
            pusher_path_weight: weight of the pusher path length off contact (GCS default 10; 0 = off).
        """
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/pusht/scene.xml"
        )
        super().__init__(mj_model, trace_sites=["pusher"])
        self.close_to_block_weight = close_to_block_weight
        self.pusher_path_weight = pusher_path_weight

        # Get sensor ids
        self.block_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "position"
        )
        self.block_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation"
        )

    def _get_position_err(self, state: mjx.Data) -> jax.Array:
        """Position of the block relative to the target position."""
        sensor_adr = self.model.sensor_adr[self.block_position_sensor]
        return state.sensordata[sensor_adr : sensor_adr + 3]

    def _get_orientation_err(self, state: mjx.Data) -> jax.Array:
        """Orientation of the block relative to the target orientation."""
        sensor_adr = self.model.sensor_adr[self.block_orientation_sensor]
        block_quat = state.sensordata[sensor_adr : sensor_adr + 4]
        goal_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        return mjx._src.math.quat_sub(block_quat, goal_quat)
    
    def _get_se3_err(self, state: mjx.Data) -> jax.Array:
        sensor_adr = self.model.sensor_adr[self.block_orientation_sensor]
        block_quat = state.sensordata[sensor_adr : sensor_adr + 4]
        goal_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        R = jaxlie.SO3(block_quat)
        Rg = jaxlie.SO3(goal_quat)

        sensor_adr = self.model.sensor_adr[self.block_position_sensor]
        block_pos = state.sensordata[sensor_adr : sensor_adr + 3]
        goal_pos = jnp.array([0.0, 0.0, 0.0])
        T  = jaxlie.SE3.from_rotation_and_translation(R,  block_pos)
        Tg = jaxlie.SE3.from_rotation_and_translation(Rg, goal_pos)

        err = (T @ Tg.inverse()).log()      # shape (6,)
        return err

    def _close_to_block_err(self, state: mjx.Data) -> jax.Array:
        """Position of the pusher block relative to the block."""
        block_pos = state.qpos[:2]
        pusher_pos = state.qpos[3:] + jnp.array([0.0, 0.1])  # y bias
        return block_pos - pusher_pos

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) = pose cost + pusher_path_weight * |pusher velocity| off contact.

        Summed as dt * ℓ over a rollout, the second term is pusher_path_weight * (distance the pusher
        travels while not touching the T), GCS's pusher_arc_length."""
        off_contact = pusher_gap(state.qpos) > CONTACT_TOL
        pusher_speed = jnp.sqrt(jnp.sum(jnp.square(state.qvel[3:5])) + 1e-12)
        return self.pose_cost(state) + self.pusher_path_weight * off_contact * pusher_speed

    def pose_cost(self, state: mjx.Data) -> jax.Array:
        """50 |p_COM - p_COM_goal|^2 + 15 * 2 (1 - cos θ) (+ the close-to-block term, 0 by default).

        2 (1 - cos θ) = (1 - cos θ)^2 + sin^2 θ ≈ θ^2 near the goal: the angle cost GCS can represent."""
        position_err = self._get_position_err(state)
        orientation_err = self._get_orientation_err(state)
        close_to_block_err = self._close_to_block_err(state)

        position_cost = jnp.sum(jnp.square(position_err))
        orientation_cost = 2.0 * (1.0 - jnp.cos(jnp.linalg.norm(orientation_err)))  # |orientation_err| = |θ|
        close_to_block_cost = jnp.sum(jnp.square(close_to_block_err))

        return (
            50 * position_cost
            + 15 * orientation_cost
            + self.close_to_block_weight * close_to_block_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ℓ_T(x_T): the pose cost (GCS's goal is a constraint, with no pusher term)."""
        return self.pose_cost(state)

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the level of friction."""
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=1.0, maxval=1.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}
