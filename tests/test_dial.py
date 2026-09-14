import jax
import jax.numpy as jnp
import pytest
from mujoco import mjx

from global_mppi.algs.dial import DIAL
from global_mppi.tasks.pusht import PushT


def _make_dial(**overrides) -> DIAL:
    task = PushT()
    kwargs = dict(
        num_samples=5,
        noise_level=1.0,
        beta_opt_iter=2.0,
        beta_horizon=2.0,
        temperature=0.1,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=4,
        iterations=5,
    )
    kwargs.update(overrides)
    return DIAL(task, **kwargs)


def test_nonpositive_beta_opt_iter_raises() -> None:
    """beta_opt_iter must be strictly positive."""
    with pytest.raises(AssertionError, match="beta_opt_iter"):
        _make_dial(beta_opt_iter=0.0)


def test_nonpositive_beta_horizon_raises() -> None:
    """beta_horizon must be strictly positive."""
    with pytest.raises(AssertionError, match="beta_horizon"):
        _make_dial(beta_horizon=-1.0)


def test_noise_anneals_across_horizon_and_opt_iterations(monkeypatch) -> None:
    """Sampled noise scale should match
    sigma[i,h] = noise_level * exp(-i/(beta_opt_iter*N) - (H-h)/(beta_horizon*H)),
    and the opt_iteration counter should advance (wrapping modulo
    `iterations`)."""
    opt = _make_dial()
    params = opt.init_params()
    old_iter = params.opt_iteration  # 0, from init_params

    # Replace Gaussian noise with all ones so sampled controls equal
    # mean + noise_level(i, h) exactly, letting us check the formula directly.
    monkeypatch.setattr(
        jax.random,
        "normal",
        lambda *_a, **_kw: jnp.ones(
            (opt.num_samples, opt.num_knots, opt.task.model.nu)
        ),
    )

    controls, new_params = opt.sample_knots(params)

    h = jnp.arange(opt.num_knots)
    expected_sigma = opt.noise_level * jnp.exp(
        -old_iter / (opt.beta_opt_iter * opt.iterations)
        - (opt.num_knots - 1 - h) / (opt.beta_horizon * opt.num_knots)
    )
    actual_sigma = controls[0, :, 0] - params.mean[:, 0]
    assert jnp.allclose(actual_sigma, expected_sigma)
    # Noise should shrink monotonically along the horizon (more annealed near
    # the start of the plan, less annealed near the end).
    assert jnp.all(jnp.diff(expected_sigma) >= 0)

    assert new_params.opt_iteration == (old_iter + 1) % opt.iterations


def test_optimize_does_not_increase_the_best_cost() -> None:
    """Running DIAL for a few iterations on the push-T task should not make
    the best (mean) rollout worse than the initial guess."""
    opt = _make_dial(num_samples=128, iterations=5)
    jit_optimize = jax.jit(opt.optimize)

    state = mjx.make_data(opt.task.model)
    state = state.replace(
        qpos=jnp.array([0.1, 0.1, 1.3, 0.0, 0.0]), qvel=jnp.zeros_like(state.qvel)
    )
    state = jax.jit(mjx.forward)(opt.task.model, state)
    params = opt.init_params()

    initial_cost, _ = opt.get_cost_with_best_samples(state, params)

    params, _, _ = jit_optimize(state, params)

    final_cost, _ = opt.get_cost_with_best_samples(state, params)
    assert jnp.isfinite(final_cost).all()
    assert final_cost <= initial_cost
