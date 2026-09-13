import jax
import jax.numpy as jnp
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
        noise_level=0.1,
        temperature=0.1,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=6,
    )
    # Start with the block away from the goal, as in examples/pusht.py, so
    # there is meaningful cost to reduce.
    state = mjx.make_data(task.model)
    state = state.replace(
        qpos=jnp.array([0.1, 0.1, 1.3, 0.0, 0.0]), qvel=jnp.zeros_like(state.qvel)
    )
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
