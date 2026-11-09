"""PushT in GCS's own dynamics: the quasi-static pusher-slider model of Graesdal et al. (RSS 2024),
from Hogan & Rodriguez (planning_through_contact/.../slider_pusher_system.py), written in JAX.

The model GCS plans with, with GCS's PushT parameters (pusht_test/run_gcs_global_mppi.py):
  * the T of pusht.xml (crossbar .06 x .02 over an exposed .02 x .06 stem), frame at its COM
    (0, 0.01285714) in the MuJoCo block frame;
  * ellipsoidal limit surface about the COM: block twist V = D J^T f, D = diag(1, 1, 1 / (c r)^2)
    (GCS's diag(f_max^-2, f_max^-2, tau_max^-2) up to scale), c = 0.3, r = 0.08544 (GCS's
    max_contact_radius); the block moves only while it is pushed;
  * a round pusher of radius 0.01, pusher-block Coulomb friction mu = 0.1 (stick or slide);
  * the pusher is kinematic: its velocity is the control (a position-controlled robot).
GCS writes the dynamics for the one face in contact. Here the pusher may touch any face, so each
step solves the contact problem for the two boxes of the T (at most two contacts, at the inner
corners): every combination of {separate, stick, slide +, slide -} is solved as a small linear
system and the first one that satisfies the contact conditions is used (velocity-level time
stepping: the gap after the step stays >= 0).

State: qpos of pusht.xml = [block body x, y, theta, pusher x, y - 0.1], so the start state, goal and
cost are those of the MuJoCo task.
"""

import itertools

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from global_mppi.tasks.pusht import PushT

COM = jnp.array([0.0, 0.06 * 0.02 * 0.03 / (0.02 * 0.08 + 0.06 * 0.02)])  # in the block body frame
PUSHER_HOME = jnp.array([0.0, 0.1])  # pusher qpos is relative to this
PUSHER_RADIUS = 0.01
MU = 0.1  # GCS demo friction_coeff_slider_pusher
C_LS, R_MAX = 0.3, 0.08544003745317531  # GCS integration_constant, max_contact_radius
D_LS = jnp.diag(jnp.array([1.0, 1.0, 1.0 / (C_LS * R_MAX) ** 2]))
# the T as two boxes in the COM frame: (center, half size). The stem is only its exposed part.
BOXES = (
    (jnp.array([0.0, 0.03]) - COM, jnp.array([0.03, 0.01])),  # crossbar, y in [0.02, 0.04]
    (jnp.array([0.0, -0.01]) - COM, jnp.array([0.01, 0.03])),  # stem, y in [-0.04, 0.02]
)
MODES = jnp.array(list(itertools.product(range(4), repeat=2)))  # 0 separate, 1 stick, 2 slide +, 3 slide -
TOL = 1e-7


def _rot(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s], [s, c]])


def _box_contact(p, center, half):
    """Outward normal, nearest point on the box and gap (distance - pusher radius) for pusher center p."""
    q = p - center
    nearest = jnp.clip(q, -half, half)
    d = q - nearest
    dist = jnp.linalg.norm(d)
    outside = dist > 1e-9
    depth = half - jnp.abs(q)  # inside: distance to each pair of faces
    ax = jnp.argmin(depth)
    sgn = jnp.where(q[ax] >= 0, 1.0, -1.0)
    n_in = jnp.zeros(2).at[ax].set(sgn)
    c_in = q.at[ax].set(sgn * half[ax])
    n = jnp.where(outside, d / jnp.maximum(dist, 1e-12), n_in)
    c = jnp.where(outside, nearest, c_in) + center
    gap = jnp.where(outside, dist, -depth[ax]) - PUSHER_RADIUS
    return n, c, gap


def block_twist(p_b, v_b, dt):
    """Block twist (vx, vy, omega) at the COM in the block frame, for the pusher at p_b moving at v_b
    (both in the block frame, relative to the COM), over one step dt."""
    n, c, gap = jax.tree.map(lambda *a: jnp.stack(a), *[_box_contact(p_b, *b) for b in BOXES])
    t = jnp.stack([-n[:, 1], n[:, 0]], -1)
    J = jax.vmap(lambda ci: jnp.array([[1.0, 0.0, -ci[1]], [0.0, 1.0, ci[0]]]))(c)  # (2, 2, 3)
    # unknowns z = (lam_n0, lam_t0, lam_n1, lam_t1); force on the block f_i = -lam_n n_i + lam_t t_i
    G = jnp.stack([D_LS @ J[0].T @ -n[0], D_LS @ J[0].T @ t[0],
                   D_LS @ J[1].T @ -n[1], D_LS @ J[1].T @ t[1]], -1)  # V = G z
    An = jnp.einsum("ij,ijk,kl->il", n, J, G)  # normal velocity of the block at contact i = An_i . z
    At = jnp.einsum("ij,ijk,kl->il", t, J, G)
    vn, vt = n @ v_b, t @ v_b  # pusher velocity along each normal / tangent
    eye = jnp.eye(4)

    def solve(mode):
        rows, rhs = [], []
        for i in range(2):
            m, en, et = mode[i], eye[2 * i], eye[2 * i + 1]
            rows.append(jnp.where(m == 0, en, An[i]))  # separate: lam_n = 0; else gap after the step = 0
            rhs.append(jnp.where(m == 0, 0.0, vn[i] + gap[i] / dt))
            slide = jnp.where(m == 2, 1.0, -1.0) * MU
            rows.append(jnp.where(m == 0, et, jnp.where(m == 1, At[i], et - slide * en)))
            rhs.append(jnp.where(m == 1, vt[i], 0.0))
        z = jnp.linalg.solve(jnp.stack(rows) + 1e-12 * eye, jnp.stack(rhs))
        lam_n, lam_t = z[0::2], z[1::2]
        gap_next = gap + dt * (vn - An @ z)  # pusher relative motion along n
        slip = vt - At @ z  # pusher relative motion along t
        ok = jnp.where(mode == 0, gap_next >= -TOL * dt,
                       (lam_n >= -TOL) & jnp.where(mode == 1, jnp.abs(lam_t) <= MU * lam_n + TOL,
                                                   jnp.where(mode == 2, slip >= -TOL, slip <= TOL)))
        return z, jnp.all(ok) & jnp.all(jnp.isfinite(z))

    zs, valid = jax.vmap(solve)(MODES)
    z = jnp.where(jnp.any(valid), zs[jnp.argmax(valid)], 0.0)
    return G @ z


def gcs_step(qpos, u, dt):
    """One step of the quasi-static model: qpos (5,) of pusht.xml, pusher velocity u (2,)."""
    theta = qpos[2]
    R = _rot(theta)
    com = qpos[:2] + R @ COM
    p_w = qpos[3:5] + PUSHER_HOME
    V = block_twist(R.T @ (p_w - com), R.T @ u, dt)
    theta_next = theta + dt * V[2]
    com_next = com + dt * (R @ V[:2])
    body_next = com_next - _rot(theta_next) @ COM
    return jnp.concatenate([body_next, theta_next[None], qpos[3:5] + dt * u])


def wrap(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


class PushTGCS(PushT):
    """PushT with GCS's quasi-static dynamics (gcs_step) in place of MuJoCo; same state, goal and cost.

    The planners keep an mjx.Data state; only qpos, ctrl and time are used. The cost is pusht.py's,
    computed from qpos: 50 |block position|^2 + 15 (block angle)^2 (the goal is the origin).
    """

    def __init__(self, close_to_block_weight: float = 0.0) -> None:
        super().__init__(close_to_block_weight=close_to_block_weight)
        self._sim_step = jax.jit(gcs_step)

    def step(self, model: mjx.Model, data: mjx.Data) -> mjx.Data:
        u = jnp.clip(data.ctrl, self.u_min, self.u_max)
        return data.replace(qpos=gcs_step(data.qpos, u, self.dt), time=data.time + self.dt)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        pos = state.qpos[:2]
        pusher_err = state.qpos[:2] - (state.qpos[3:5] + PUSHER_HOME)
        return (50 * jnp.sum(jnp.square(pos)) + 15 * jnp.square(wrap(state.qpos[2]))
                + self.close_to_block_weight * jnp.sum(jnp.square(pusher_err)))

    def get_trace_sites(self, state: mjx.Data) -> jax.Array:
        return jnp.concatenate([state.qpos[3:5] + PUSHER_HOME, jnp.array([0.011])])[None]

    def sim_step(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> None:
        """The closed-loop 'real' system: one step of gcs_step at the simulation timestep."""
        u = np.clip(mj_data.ctrl, np.asarray(self.u_min), np.asarray(self.u_max))
        mj_data.qpos[:] = np.asarray(self._sim_step(jnp.asarray(mj_data.qpos), jnp.asarray(u), mj_model.opt.timestep))
        mj_data.qvel[:] = 0.0
        mj_data.time += mj_model.opt.timestep
        mujoco.mj_kinematics(mj_model, mj_data)
