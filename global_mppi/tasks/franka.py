from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from scipy.spatial import ConvexHull

from global_mppi import ROOT
from global_mppi.task_base import Task


def _sample_hull_surface(
    verts: np.ndarray, num_points: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample points uniformly on the convex hull of a mesh (what MuJoCo collides).

    Args:
        verts: (V, 3) mesh vertices.
        num_points: number of surface samples.
        rng: random number generator.

    Returns:
        (V' + num_points, 3) hull vertices followed by the surface samples.
    """
    hull = ConvexHull(verts)
    tri = verts[hull.simplices]  # (F, 3, 3)
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    face = rng.choice(len(tri), size=num_points, p=area / area.sum())
    u, v = rng.random((2, num_points))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[face]
    samples = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    return np.concatenate([verts[hull.vertices], samples])


def _fit_spheres(
    points: np.ndarray, num_spheres: int, iterations: int = 50
) -> tuple:
    """Cover a point cloud with spheres from a k-means clustering.

    Args:
        points: (P, 3) points, in the geom frame.
        num_spheres: number of clusters (spheres).
        iterations: number of k-means iterations.

    Returns:
        (K, 3) sphere centers in the geom frame and (K,) radii, K <= num_spheres.
        Every point lies inside the sphere of its cluster.
    """
    # Deterministic farthest-point initialization
    centers = [points.mean(axis=0)]
    for _ in range(num_spheres - 1):
        dist = np.min(
            np.linalg.norm(points[:, None] - np.array(centers)[None], axis=-1), axis=1
        )
        centers.append(points[dist.argmax()])
    centers = np.array(centers)

    for _ in range(iterations):
        label = np.linalg.norm(
            points[:, None] - centers[None], axis=-1
        ).argmin(axis=1)
        centers = np.array([
            points[label == k].mean(axis=0) if np.any(label == k) else centers[k]
            for k in range(num_spheres)
        ])

    keep = [k for k in range(num_spheres) if np.any(label == k)]
    radii = np.array([
        np.linalg.norm(points[label == k] - centers[k], axis=1).max() for k in keep
    ])
    return centers[keep], radii


class FrankaReach(Task):
    """Move the FR3 arm to a goal configuration along a short path.

    The scene (models/franka/fr3_balls_scene.xml) has static spherical obstacles
    placed so that the straight joint-space line from the `start` keyframe to the
    `goal` keyframe collides; the arm has to retract its elbow to get around them.

    Obstacles are physical, not a cost: the balls and the table are rigid
    collision geoms, both in the MJX rollouts and in simulation. The arm
    collides through a few spheres per link, fit to its collision meshes (MJX
    mesh collisions are slow); the original mesh collision geoms are disabled.
    Contacts are limited to arm<->balls and arm<->table (links 1 and 2 sit on
    the base and are exempt from the table); there is no self-collision.
    """

    def __init__(
        self,
        goal_weight: float = 10.0,
        path_length_weight: float = 0.5,
        terminal_weight: float = 1000.0,
        spheres_per_link: int = 3,
    ) -> None:
        """Load the MuJoCo model and set task parameters.

        Args:
            goal_weight: weight of the squared joint-space distance to the goal
                in the running cost (pulls the arm toward the goal).
            path_length_weight: weight of the joint-speed norm ||q̇||, whose sum
                over time is the joint-space path length.
            terminal_weight: weight of the squared joint error to the goal and
                of the squared joint speed at the end of the horizon.
            spheres_per_link: number of collision spheres fit to each link mesh.
        """
        # The gripper is not used: drop its actuator so the 7 arm joints are
        # the only controls.
        spec = mujoco.MjSpec.from_file(ROOT + "/models/franka/fr3_balls_scene.xml")
        spec.delete(spec.actuator("gripper"))
        for key in spec.keys:
            key.ctrl = list(key.qpos)[:7]

        # Fit collision spheres to the moving arm's collision meshes (in the
        # compiled model, where mesh geom frames are known), then add them to
        # the spec as geoms on the same bodies. The base link (fr3_link0)
        # never moves and is skipped.
        mesh_model = spec.compile()
        base_body = mesh_model.body("fr3_link0").id
        low_links = [mesh_model.body(f"fr3_link{i}").id for i in (1, 2)]
        rng = np.random.default_rng(0)
        spheres = []  # (body name, center in body frame, radius)
        for g in range(mesh_model.ngeom):
            body = mesh_model.geom_bodyid[g]
            if (
                mesh_model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
                or mesh_model.geom_contype[g] == 0
                or body in (0, base_body)
            ):
                continue
            mesh = mesh_model.geom_dataid[g]
            adr, num = mesh_model.mesh_vertadr[mesh], mesh_model.mesh_vertnum[mesh]
            is_finger = "finger" in mesh_model.body(body).name
            centers, radii = _fit_spheres(
                _sample_hull_surface(mesh_model.mesh_vert[adr : adr + num], 2000, rng),
                1 if is_finger else spheres_per_link,
            )
            for c, r in zip(centers, radii):
                c_body = np.zeros(3)
                mujoco.mju_rotVecQuat(c_body, c, mesh_model.geom_quat[g])
                c_body += mesh_model.geom_pos[g]
                spheres.append((mesh_model.body(body).name, c_body, r, body in low_links))

        # Collision filtering with contype/conaffinity bits: arm spheres
        # (contype 1|2, or 1 on links 1-2) hit balls (conaffinity 1) and the
        # table (conaffinity 2); nothing else collides.
        stiff = dict(solref=[0.02, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2.0])
        for geom in spec.geoms:
            if geom.name.startswith("ball_"):
                geom.contype, geom.conaffinity = 0, 1
            elif geom.name == "table":
                geom.contype, geom.conaffinity = 0, 2
            else:
                geom.contype, geom.conaffinity = 0, 0
                continue
            geom.solref, geom.solimp = stiff["solref"], stiff["solimp"]
        for i, (body, c, r, is_low) in enumerate(spheres):
            spec.body(body).add_geom(
                name=f"arm_sphere_{i}",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[r, 0, 0],
                pos=c,
                contype=1 if is_low else 3,
                conaffinity=0,
                group=3,
                mass=0,
                density=0,
                rgba=[0.2, 0.6, 1.0, 0.3],
                **stiff,
            )
        mj_model = spec.compile()

        # Re-center the position actuators so that ctrl is an offset from the
        # middle of each joint's range: target = mid + ctrl, ctrl in [-h, h].
        # The samplers (in particular GlobalMPPI's KSOS box) assume control
        # bounds that are symmetric about zero, which raw joint targets are not.
        lo, hi = mj_model.actuator_ctrlrange[:, 0].copy(), mj_model.actuator_ctrlrange[:, 1].copy()
        self.ctrl_mid = jnp.array((lo + hi) / 2)
        mj_model.actuator_biasprm[:, 0] = mj_model.actuator_gainprm[:, 0] * (lo + hi) / 2
        mj_model.actuator_ctrlrange[:, 0] = -(hi - lo) / 2
        mj_model.actuator_ctrlrange[:, 1] = (hi - lo) / 2
        mj_model.key_ctrl[:] = mj_model.key_qpos[:, :7] - (lo + hi) / 2

        # Sphere-sphere / sphere-box contacts only, so a few solver iterations
        # suffice (MJX cost scales with these).
        mj_model.opt.iterations = 4
        mj_model.opt.ls_iterations = 10
        super().__init__(mj_model, trace_sites=["pusher_site"])

        self.goal_weight = goal_weight
        self.path_length_weight = path_length_weight
        self.terminal_weight = terminal_weight

        # Start/goal from the scene keyframes (7 arm joints + 2 fingers);
        # ctrl_start holds the start pose, in the re-centered control space.
        self.q_start = jnp.array(mj_model.key("start").qpos)
        self.q_goal = jnp.array(mj_model.key("goal").qpos[:7])
        self.ctrl_start = jnp.array(mj_model.key("start").ctrl)

        # Obstacle balls, table top height and arm spheres, for `clearance`
        ball_ids = [
            i for i in range(mj_model.ngeom)
            if mj_model.geom(i).name.startswith("ball_")
        ]
        self.ball_pos = jnp.array(mj_model.geom_pos[ball_ids])
        self.ball_radius = jnp.array(mj_model.geom_size[ball_ids, 0])
        table = mj_model.geom("table")
        self.table_z = float(table.pos[2] + table.size[2])
        self.sphere_ids = jnp.array(
            [mj_model.geom(f"arm_sphere_{i}").id for i in range(len(spheres))]
        )
        self.sphere_radius = jnp.array(mj_model.geom_size[self.sphere_ids, 0])
        self.sphere_check_table = jnp.array([not low for *_, low in spheres])

    def clearance(self, state: mjx.Data) -> jax.Array:
        """Smallest signed distance (m) from the arm spheres to any ball or the
        table top. Diagnostic only; not part of the cost."""
        centers = state.geom_xpos[self.sphere_ids]
        ball_dist = (
            jnp.linalg.norm(centers[:, None] - self.ball_pos[None], axis=-1)
            - self.sphere_radius[:, None]
            - self.ball_radius[None]
        )
        table_dist = jnp.where(
            self.sphere_check_table,
            centers[:, 2] - self.sphere_radius - self.table_z,
            jnp.inf,
        )
        return jnp.minimum(ball_dist.min(), table_dist.min())

    def _goal_err(self, state: mjx.Data) -> jax.Array:
        """Joint-space error of the arm to the goal configuration."""
        return state.qpos[:7] - self.q_goal

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        goal_cost = jnp.sum(jnp.square(self._goal_err(state)))
        # ||q̇|| summed over time is the joint-space path length; the small
        # epsilon keeps the gradient finite at rest.
        speed = jnp.sqrt(jnp.sum(jnp.square(state.qvel[:7])) + 1e-6)
        return self.goal_weight * goal_cost + self.path_length_weight * speed

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        goal_cost = jnp.sum(jnp.square(self._goal_err(state)))
        velocity_cost = jnp.sum(jnp.square(state.qvel[:7]))
        return self.terminal_weight * (goal_cost + 0.1 * velocity_cost)

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly shift the measured arm configuration."""
        shift = 0.005 * jax.random.normal(rng, (self.model.nq,))
        return {"qpos": data.qpos + shift}
