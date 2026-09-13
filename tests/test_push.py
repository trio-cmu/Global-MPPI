import jax
import jax.numpy as jnp
from mujoco import mjx

from global_mppi.algs.cem import CEM
from global_mppi.tasks.pusht import PushT


def test_cem_open_loop() -> None:
    """Run CEM open-loop on the push-T task and check the cost converges."""
    task = PushT()
    opt = CEM(
        task,
        num_samples=1000,
        num_elites=4,
        sigma_start=1.0,
        sigma_min=0.1,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )
    jit_opt = jax.jit(opt.optimize)

    state = mjx.make_data(task.model)
    params = opt.init_params()

    for _ in range(300):
        params, rollouts, rollouts_best = jit_opt(state, params)

    # Roll out the solution and check that it's good enough.
    knots = params.mean[None]
    tk = jnp.linspace(0.0, opt.plan_horizon, opt.num_knots)
    tq = jnp.linspace(0.0, opt.plan_horizon - opt.dt, opt.ctrl_steps)
    controls = opt.interp_func(tq, tk, knots)
    _, final_rollout = jax.jit(opt.eval_rollouts)(
        task.model, state, controls, knots
    )

    total_cost = jnp.sum(final_rollout.costs[0])
    assert total_cost <= 9.0
    assert jnp.all(params.cov >= opt.sigma_min)


def test_explore_fraction() -> None:
    """Unit test for sampling controls with different explore_fraction values.

    Verifies that:
      - The overall controls array shape is correct.
      - The split between main and exploration samples is as expected.
    """
    task = PushT()
    num_samples = 10
    num_elites = 2
    sigma_start = 1.0
    sigma_min = 0.1

    # Test different fractions: no exploration, partial exploration, full exploration.
    for explore_fraction in [0.0, 0.3, 0.5, 0.75, 1.0]:
        opt = CEM(
            task=task,
            num_samples=num_samples,
            num_elites=num_elites,
            sigma_start=sigma_start,
            sigma_min=sigma_min,
            explore_fraction=explore_fraction,
            plan_horizon=1.0,
            spline_type="zero",
            num_knots=11,
        )
        params = opt.init_params(seed=42)
        controls, _ = opt.sample_knots(params)

        # Check the overall shape of the controls array.
        expected_shape = (num_samples, opt.num_knots, task.model.nu)
        assert controls.shape == expected_shape, (
            f"Expected shape {expected_shape} but got {controls.shape} "
            f"for explore_fraction = {explore_fraction}"
        )

        # Calculate expected number of exploration samples.
        num_explore = int(explore_fraction * num_samples)
        num_main = num_samples - num_explore

        # The implementation concatenates main samples first and exploration samples later.
        main_controls = controls[:num_main]
        explore_controls = controls[num_main:]

        # Verify that the main and exploration segments have the correct shapes.
        expected_main_shape = (num_main, opt.num_knots, task.model.nu)
        expected_explore_shape = (
            num_explore,
            opt.num_knots,
            task.model.nu,
        )
        assert main_controls.shape == expected_main_shape, (
            f"Expected main controls shape {expected_main_shape} but got {main_controls.shape} "
            f"for explore_fraction = {explore_fraction}"
        )
        assert explore_controls.shape == expected_explore_shape, (
            f"Expected explore controls shape {expected_explore_shape} but got {explore_controls.shape} "
            f"for explore_fraction = {explore_fraction}"
        )


if __name__ == "__main__":
    test_cem_open_loop()
    test_explore_fraction()
