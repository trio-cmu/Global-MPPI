import jax
import jax.numpy as jnp
from mujoco import mjx

from global_mppi.alg_base import Trajectory
from global_mppi.algs.mppi import MPPI
from global_mppi.tasks.pusht import PushT


def _make_mppi(**overrides) -> MPPI:
    task = PushT()
    kwargs = dict(
        num_samples=8,
        noise_level=0.1,
        temperature=1.0,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=4,
    )
    kwargs.update(overrides)
    return MPPI(task, **kwargs)


def test_sample_knots_shape_and_rng_advances() -> None:
    """Sampled knots should have shape (num_samples, num_knots, nu), and the
    rng key in the returned params should have advanced."""
    opt = _make_mppi(num_samples=5, num_knots=3)
    params = opt.init_params()

    controls, new_params = opt.sample_knots(params)

    assert controls.shape == (5, 3, opt.task.model.nu)
    assert not jnp.array_equal(new_params.rng, params.rng)


def test_update_params_is_a_softmax_weighted_average_of_knots() -> None:
    """update_params should reproduce a hand-computed softmax-weighted mean
    over knots, favoring lower-cost rollouts."""
    opt = _make_mppi(temperature=2.0, num_knots=1)
    params = opt.init_params()

    # Three candidate rollouts with distinct total costs and knots.
    knots = jnp.array([[[0.0, 0.0]], [[1.0, 1.0]], [[2.0, 2.0]]])
    costs = jnp.array([[1.0], [0.5], [3.0]])  # summed over time -> [1.0, 0.5, 3.0]
    rollouts = Trajectory(
        controls=jnp.zeros((3, 1, 2)),
        knots=knots,
        costs=costs,
        trace_sites=jnp.zeros((3, 1, 3)),
    )

    new_params = opt.update_params(params, rollouts)

    total_costs = jnp.sum(costs, axis=1)
    weights = jax.nn.softmax(-total_costs / opt.temperature, axis=0)
    expected_mean = jnp.sum(weights[:, None, None] * knots, axis=0)
    assert jnp.allclose(new_params.mean, expected_mean)
    # Lower-cost knots should be weighted more heavily than higher-cost ones.
    assert weights[1] > weights[0] > weights[2]


def test_optimize_does_not_increase_the_best_cost() -> None:
    """Running MPPI for a few iterations on the push-T task should not make
    the best (mean) rollout worse than the initial guess."""
    opt = _make_mppi(num_samples=128, noise_level=0.2, temperature=0.1)
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
