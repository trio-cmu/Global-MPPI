import jax
import jax.numpy as jnp
from mujoco import mjx

from global_mppi.alg_base import Trajectory
from global_mppi.algs.predictive_sampling import PredictiveSampling
from global_mppi.tasks.pusht import PushT


def _make_ps(**overrides) -> PredictiveSampling:
    task = PushT()
    kwargs = dict(
        num_samples=5,
        noise_level=0.2,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=3,
    )
    kwargs.update(overrides)
    return PredictiveSampling(task, **kwargs)


def test_sample_knots_includes_the_current_mean_as_a_sample() -> None:
    """The first sampled control tape should be the unperturbed mean, so
    predictive sampling never does worse than simply repeating its last
    plan."""
    opt = _make_ps()
    params = opt.init_params()
    params = params.replace(mean=jnp.full((opt.num_knots, opt.task.model.nu), 0.3))

    controls, _ = opt.sample_knots(params)

    assert controls.shape == (opt.num_samples, opt.num_knots, opt.task.model.nu)
    assert jnp.allclose(controls[0], params.mean)


def test_update_params_picks_the_lowest_cost_rollout() -> None:
    """update_params should set the mean to the knots of whichever rollout
    had the lowest total cost, not a weighted average of all of them."""
    opt = _make_ps()
    params = opt.init_params()

    knots = jnp.stack(
        [
            jnp.full((opt.num_knots, opt.task.model.nu), 0.0),
            jnp.full((opt.num_knots, opt.task.model.nu), 1.0),
            jnp.full((opt.num_knots, opt.task.model.nu), 2.0),
        ]
    )
    costs = jnp.array([[2.0], [0.1], [5.0]])  # rollout 1 is cheapest
    rollouts = Trajectory(
        controls=jnp.zeros((3, 1, opt.task.model.nu)),
        knots=knots,
        costs=costs,
        trace_sites=jnp.zeros((3, 1, 3)),
    )

    new_params = opt.update_params(params, rollouts)

    assert jnp.allclose(new_params.mean, knots[1])


def test_optimize_does_not_increase_the_best_cost() -> None:
    """Running predictive sampling for a few iterations on the push-T task
    should not make the best rollout worse than the initial guess."""
    opt = _make_ps(num_samples=128, noise_level=0.3)
    jit_optimize = jax.jit(opt.optimize)

    state = mjx.make_data(opt.task.model)
    state = state.replace(
        qpos=jnp.array([0.1, 0.1, 1.3, 0.0, 0.0]), qvel=jnp.zeros_like(state.qvel)
    )
    state = jax.jit(mjx.forward)(opt.task.model, state)
    params = opt.init_params()

    initial_cost, _ = opt.get_cost_with_best_samples(state, params)

    for _ in range(5):
        params, _, _ = jit_optimize(state, params)

    final_cost, _ = opt.get_cost_with_best_samples(state, params)
    assert jnp.isfinite(final_cost).all()
    assert final_cost <= initial_cost
