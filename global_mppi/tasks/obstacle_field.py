"""Velocity-controlled point mass crossing a field of 9 circular obstacles.

Same particle, actuators, outer walls and cost weights as ``bugtrap`` (hydrax / MJPC particle task), but there are no
inner U walls: the start is on the left, the goal on the right, and 9 circles (``make_layout(layout_seed)``,
radius 0.025 to 0.045 m) stand in between. The wall-SDF cost term of bugtrap is replaced by the same formula on the
signed distance to the nearest circle, ``|p - c| - r``.
"""

from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from global_mppi import ROOT
from global_mppi.task_base import Task
from global_mppi.tasks.obstacle_field_layout import GOAL, START, make_layout


def build_model(layout_seed: int = 19) -> mujoco.MjModel:
    """MuJoCo model: models/obstacle_field/scene.xml with the circles of ``make_layout(layout_seed)`` as cylinders."""
    centers, radii = make_layout(layout_seed)
    geoms = "\n".join(
        f'    <geom name="obstacle_{i}" class="wall" type="cylinder" pos="{c[0]:.6f} {c[1]:.6f} .02" size="{r:.6f} .02" rgba="0.85 0.4 0.3 1"/>'
        for i, (c, r) in enumerate(zip(centers, radii))
    )
    xml = open(ROOT + "/models/obstacle_field/scene.xml").read()
    assert "<!-- OBSTACLES -->" in xml
    return mujoco.MjModel.from_xml_string(xml.replace("<!-- OBSTACLES -->", geoms))


class ObstacleField(Task):
    """A planar point mass that must cross a field of circular obstacles."""

    def __init__(
        self,
        layout_seed: int = 19,
        wall_weight: float = 5.0,
        wall_sharpness: float = 50.0,
        position_weight: float = 5.0,
        velocity_weight: float = 0.1,
        control_weight: float = 0.1,
    ) -> None:
        """Load the MuJoCo model and set task parameters.

        Args:
            layout_seed: seed of ``make_layout`` (default 19: the fixed benchmark layout).
            wall_weight: Multiplier on the exponential obstacle proximity cost (as bugtrap's wall_weight).
            wall_sharpness: Decay rate in the ``exp(-sharpness * dist)`` obstacle cost.
            position_weight: Weight on the squared position tracking error.
            velocity_weight: Weight on the squared velocity penalty.
            control_weight: Weight on the squared control penalty.
        """
        self.layout_seed = layout_seed
        self.centers, self.radii = make_layout(layout_seed)
        mj_model = build_model(layout_seed)
        super().__init__(mj_model, trace_sites=["pointmass"])

        self.pointmass_id = mj_model.site("pointmass").id
        self.start = START
        self.goal = GOAL

        self.wall_weight = wall_weight
        self.wall_sharpness = wall_sharpness
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.control_weight = control_weight

        self._obs_pos = jnp.array(self.centers)
        self._obs_radius = jnp.array(self.radii)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ): obstacle SDF + tracking + control."""
        pos = state.site_xpos[self.pointmass_id][:2]
        # Signed distance to each circle, nearest one (negative inside a circle).
        dist = (jnp.linalg.norm(pos[None, :] - self._obs_pos, axis=-1) - self._obs_radius).min()
        wall_cost = self.wall_weight * jnp.exp(-self.wall_sharpness * dist)
        control_cost = jnp.sum(jnp.square(control))
        return (
            wall_cost
            + self.terminal_cost(state)
            + self.control_weight * control_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T): position tracking + velocity reg."""
        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.pointmass_id] - state.mocap_pos[0])
        )
        velocity_cost = jnp.sum(jnp.square(state.qvel))
        return (
            self.position_weight * position_cost
            + self.velocity_weight * velocity_cost
        )

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomly perturb the actuator gains by ±10%."""
        multiplier = jax.random.uniform(
            rng,
            self.model.actuator_gainprm[:, 0].shape,
            minval=0.9,
            maxval=1.1,
        )
        new_gains = self.model.actuator_gainprm[:, 0] * multiplier
        new_gains = self.model.actuator_gainprm.at[:, 0].set(new_gains)
        return {"actuator_gainprm": new_gains}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly shift the measured particle position."""
        shift = jax.random.uniform(rng, (2,), minval=-0.01, maxval=0.01)
        return {"qpos": data.qpos + shift}
