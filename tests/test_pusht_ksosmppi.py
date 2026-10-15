import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from global_mppi.algs import GlobalMPPI
from global_mppi.tasks.pusht import PushT


def test_ksos_rollout_and_solve() -> None:
    """Run a few KSOS rollout + solve iterations on the push-T task and check
    that the resulting mean is finite and no worse than the initial guess."""
    task = PushT()
    opt = GlobalMPPI(
        task,
        num_samples=256,
        noise_level=0.01,  # 0.1 x the pusher speed range (+-0.1 m/s), the same ratio as noise 0.1 for the old +-1 range
        temperature=0.1,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=6,
    )
    # Start from the T of the start_2 keyframe (block away from the goal, as in examples/pusht.py), but with the
    # pusher 1 cm from the T on the far side from the goal. From the keyframe's own pusher position (9 cm away) a
    # 1 s horizon at 0.1 m/s cannot change the cost, and the KSOS elite filter only keeps rollouts whose cost starts
    # to fall within the first 0.2 s, i.e. the pusher must reach the T quickly.
    q = np.array(task.start_qpos("start_2"))
    mj = task.mj_model
    anchor = np.asarray(mj.body_pos[mj.body("pusher").id][:2])
    u = -q[:2] / np.linalg.norm(q[:2])                    # T -> goal direction (the goal is the sim origin)
    c, s_ = np.cos(q[2]), np.sin(q[2])
    R = np.array([[c, -s_], [s_, c]])

    def dist_to_T(p):                                      # distance from a world point to the T (stem + crossbar boxes)
        d = np.inf
        for g in ("stem", "crossbar"):
            gid = mj.geom(g).id
            centre, half = mj.geom_pos[gid][:2], mj.geom_size[gid][:2]
            local = R.T @ (p - q[:2]) - centre
            out = np.maximum(np.abs(local) - half, 0.0)
            d = min(d, float(np.linalg.norm(out)))
        return d

    lo, hi = 0.0, 0.4                                      # along -u from the T centre: find the point 1 cm outside
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if dist_to_T(q[:2] - mid * u) < 0.01 else (lo, mid)
    q[3:5] = (q[:2] - hi * u) - anchor
    state = mjx.make_data(task.model)
    state = state.replace(qpos=jnp.array(q), qvel=jnp.zeros_like(state.qvel))
    state = jax.jit(mjx.forward)(task.model, state)
    params = opt.init_params()

    initial_cost, _ = opt.get_cost_with_best_samples(state, params)

    jit_ksos_rollout = jax.jit(opt.ksos_rollout)
    for iter in range(opt.ksos_num_restart):
        params = params.replace(iter=iter)
        params = jit_ksos_rollout(state, params)
        params = opt.solve_ksos(params)

    updated_cost, _ = opt.get_cost_with_best_samples(
        state, params.replace(mean=params.ksos_mean)
    )

    assert jnp.isfinite(updated_cost).all()
    assert jnp.all(updated_cost < initial_cost)


if __name__ == "__main__":
    test_ksos_rollout_and_solve()
