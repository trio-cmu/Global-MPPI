import jax.numpy as jnp
import pytest

from global_mppi.algs.mppi import MPPI
from global_mppi.tasks.pusht import PushT


def _make_mppi(**overrides) -> MPPI:
    """Build a small MPPI controller (any concrete SamplingBasedController
    works, since these tests exercise base-class behavior)."""
    task = PushT()
    kwargs = dict(
        num_samples=4,
        noise_level=0.1,
        temperature=1.0,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=4,
    )
    kwargs.update(overrides)
    return MPPI(task, **kwargs)


def test_zero_iterations_raises() -> None:
    """The base controller should reject a non-positive number of iterations."""
    with pytest.raises(ValueError, match="iterations must be greater than 0"):
        _make_mppi(iterations=0)


def test_init_params_defaults() -> None:
    """init_params should zero-initialize the mean and report an infinite
    best cost until a rollout has actually been evaluated."""
    opt = _make_mppi(num_knots=5)
    params = opt.init_params(seed=7)

    assert params.mean.shape == (5, opt.task.model.nu)
    assert jnp.allclose(params.mean, 0.0)
    assert params.tk.shape == (5,)
    assert jnp.isinf(params.best_cost)
    assert params.best_trace is None


def test_init_params_rejects_mismatched_initial_knots_shape() -> None:
    """Passing initial_knots of the wrong shape should fail loudly rather
    than silently broadcasting or truncating."""
    opt = _make_mppi(num_knots=4)
    bad_knots = jnp.zeros((3, opt.task.model.nu))  # wrong num_knots

    with pytest.raises(AssertionError):
        opt.init_params(initial_knots=bad_knots)


def test_get_action_reads_off_the_zero_order_spline() -> None:
    """get_action should return the held knot value for a zero-order spline,
    tracking the most recent knot as time advances."""
    opt = _make_mppi(spline_type="zero", num_knots=3, plan_horizon=1.0)
    params = opt.init_params()
    knots = jnp.array([[0.1, -0.1], [0.2, -0.2], [0.3, -0.3]])
    params = params.replace(mean=knots)

    action_at_start = opt.get_action(params, jnp.array(0.0))
    action_at_end = opt.get_action(params, jnp.array(1.0))

    assert jnp.allclose(action_at_start, knots[0])
    assert jnp.allclose(action_at_end, knots[-1])


def test_domain_randomization_broadcasts_the_model() -> None:
    """With num_randomizations > 1, the internal model should carry a
    leading randomization axis for any parameter the task randomizes, and
    randomized_axes should mark that parameter (and only that parameter)
    for vmapping."""
    opt = _make_mppi(num_randomizations=4, seed=0)

    assert opt.model.geom_friction.shape[0] == 4
    assert opt.randomized_axes.geom_friction == 0
    # Unrandomized fields keep their original (unbatched) shape.
    assert opt.model.geom_friction.shape[1:] == opt.task.model.geom_friction.shape


def test_no_domain_randomization_by_default() -> None:
    """With the default num_randomizations=1, the controller should reuse
    the task's model directly (no batching, no randomized axes)."""
    opt = _make_mppi(num_randomizations=1)

    assert opt.model is opt.task.model
    assert opt.randomized_axes is None
