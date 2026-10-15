from typing import Dict

import jax
import jax.numpy as jnp
import jaxlie
import mujoco
import numpy as np
from mujoco import mjx

from global_mppi import ROOT
from global_mppi.task_base import Task

class PushT(Task):
    """Push a T-shaped block to a desired pose."""

    def __init__(
        self,
        xml_path: str | None = None,
        w_pos: float = 50.0,
        w_ori: float = 15.0,
        w_approach: float = 0.0,
    ) -> None:
        """Load the MuJoCo model and set task parameters.

        Args:
            xml_path: Scene to load (default models/pusht/scene.xml: the 12 cm T / 5 mm rod
                of the real setup; the original paper model is the git tag pusht-paper-v0).
            w_pos: Weight on |block position error|^2 in the running cost.
            w_ori: Weight on |block orientation error|^2 in the running cost.
            w_approach: Weight on |block - pusher|^2 (pusher-to-T centre distance) in the running cost.
                0 (default) keeps the paper cost, which has no incentive to approach the block.
        """
        mj_model = mujoco.MjModel.from_xml_path(
            xml_path or ROOT + "/models/pusht/scene.xml"
        )
        super().__init__(mj_model, trace_sites=["pusher"])
        self.w_pos = w_pos
        self.w_ori = w_ori
        self.w_approach = w_approach

        # Get sensor ids
        self.block_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "position"
        )
        self.block_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation"
        )

    def start_qpos(self, name: str = "start_2") -> np.ndarray:
        """qpos [T_x, T_y, T_theta, root_x, root_y] of the keyframe `name` defined in the xml."""
        key = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key < 0:
            raise KeyError(f"no keyframe {name!r} in the pusht xml")
        return np.array(self.mj_model.key_qpos[key])

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
        """The running cost ℓ(xₜ, uₜ)."""
        position_err = self._get_position_err(state)
        orientation_err = self._get_orientation_err(state)
        close_to_block_err = self._close_to_block_err(state)

        position_cost = jnp.sum(jnp.square(position_err))
        orientation_cost = jnp.sum(jnp.square(orientation_err))
        close_to_block_cost = jnp.sum(jnp.square(close_to_block_err))

        return (
            self.w_pos * position_cost
            + self.w_ori * orientation_cost
            + self.w_approach * close_to_block_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ℓ_T(x_T)."""
        return self.running_cost(state, jnp.zeros(self.model.nu))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the level of friction."""
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=1.0, maxval=1.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}
