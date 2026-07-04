from typing import Literal, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax.struct import dataclass
# jax.config.update("jax_disable_jit", True)
from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task
import ipdb
from mujoco import mjx
from jax.experimental import io_callback
from ksos_tools.solvers import newton
@dataclass
class MPPIKSOSParams(SamplingParams):
    """Policy parameters for model-predictive path integral control.

    Same as SamplingParams, but with a different name for clarity.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
    """
    iter: jax.Array
    states: mjx.Data

class MPPIKSOS(SamplingBasedController):
    """Model-predictive path integral control.

    Implements "MPPI-generic" as described in https://arxiv.org/abs/2409.07563.
    Unlike the original MPPI derivation, this does not assume stochastic,
    control-affine dynamics or a separable cost function that is quadratic in
    control.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        temperature: float,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            noise_level: The scale of Gaussian noise to add to sampled controls.
            temperature: The temperature parameter λ. Higher values take a more
                         even average over the samples.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
        """
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
        )
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.temperature = temperature
        self.ksos_lambda = 1e-3
        self.ksos_sigma = 0.1   # 0.1
        self.ksos_epsilon = 1e-3
        self.ksos_kernel = "Gauss"
        self.ksos_iterations = 500
        self.ksos_linesearch = True

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> MPPIKSOSParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        iter = 0
        return MPPIKSOSParams(tk=_params.tk, mean=_params.mean, rng=_params.rng, iter=iter, states=None)

    def sample_knots(self, params: MPPIKSOSParams) -> Tuple[jax.Array, MPPIKSOSParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.num_knots,
                self.task.model.nu,
            ),
        )
        controls = params.mean + self.noise_level * noise
        # ipdb.set_trace()
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: MPPIKSOSParams, rollouts: Trajectory
    ) -> MPPIKSOSParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
        
        # run ksos every 10 iterations
        new_iter = jnp.array(params.iter) + 1
        predicate = jnp.equal(jnp.mod(new_iter, 10), 0)

        if new_iter % 10 == 0:
            jax.debug.print("KSOS iteration! {}", new_iter)
            # perform KSOS update
            rng, sample_rng = jax.random.split(params.rng)
            half_range = 0.6 
            controls = jax.random.uniform(
                sample_rng,            
                (
                    30,
                    self.num_knots,
                    self.task.model.nu,
                ),           
                minval=self.task.u_min,
                maxval=self.task.u_max
                # minval=params.mean - half_range,
                # maxval=params.mean + half_range
            )            
            ksos_knots = jnp.clip(
                controls, self.task.u_min, self.task.u_max
            )
            ksos_rollouts = self.rollout_with_randomizations(
                params.states, params.tk, ksos_knots, sample_rng
            )
            ksos_costs = jnp.sum(ksos_rollouts.costs, axis=1)
            def _ksos_solver_callback(knots_np, costs_np):
                knots_np = np.asarray(knots_np)
                costs_np = np.asarray(costs_np)
                n_samples = costs_np.shape[0]
                problem = newton.Problem(
                    lambd=self.ksos_lambda,
                    t=self.ksos_epsilon / max(n_samples, 1),
                    use_K=False,
                )
                problem.register_fixed_samples(
                    knots_np.reshape(n_samples, -1),
                    f_samples=costs_np,
                )
                # ipdb.set_trace()
                success = problem.initialize_kernel(self.ksos_sigma, self.ksos_kernel)
                if not success:
                    print("Kernel initialization failed")
                    return params.mean

                try:
                    updated_knots, _ = newton.damped_newton_advanced(
                        problem,
                        iterations=self.ksos_iterations,
                        verbose=False,
                        linesearch=self.ksos_linesearch,
                        return_B=False,
                    )
                    updated_knots = updated_knots.reshape(
                        self.num_knots, self.task.model.nu
                    ).astype(np.float32)
                    if not np.isfinite(updated_knots).all():
                        print("damped_newton_advanced failed")
                        return params.mean
                    return updated_knots
                except Exception:
                    print("damped_newton_advanced failed")
                    return params.mean

            updated_mean = io_callback(  # type: ignore[arg-type]
                _ksos_solver_callback,
                jax.ShapeDtypeStruct(   
                    (self.num_knots, self.task.model.nu), ksos_knots.dtype
                ),
                ksos_knots,
                ksos_costs
            )
            
            
        # # operand 传入 new_iter 和 mean，分支直接接收
        # def _do_ksos(ops):
        #     new_iter, mean = ops
        #     jax.debug.print("self.new_iter! {}", new_iter)
        #     # perform KSOS update
        #     rng, sample_rng = jax.random.split(params.rng)
        #     half_range = 0.6 
        #     controls = jax.random.uniform(
        #         sample_rng,            
        #         (
        #             100,
        #             self.num_knots,
        #             self.task.model.nu,
        #         ),           
        #         minval=self.task.u_min,
        #         maxval=self.task.u_max
        #         # minval=params.mean - half_range,
        #         # maxval=params.mean + half_range
        #     )            
        #     ksos_knots = jnp.clip(
        #         controls, self.task.u_min, self.task.u_max
        #     )
        #     ksos_rollouts = self.rollout_with_randomizations(
        #         params.states, params.tk, ksos_knots, sample_rng
        #     )
        #     ksos_costs = jnp.sum(ksos_rollouts.costs, axis=1)
        #     def _ksos_solver_callback(knots_np, costs_np):
        #         knots_np = np.asarray(knots_np)
        #         costs_np = np.asarray(costs_np)
        #         n_samples = costs_np.shape[0]
        #         problem = newton.Problem(
        #             lambd=self.ksos_lambda,
        #             t=self.ksos_epsilon / max(n_samples, 1),
        #             use_K=False,
        #         )
        #         problem.register_fixed_samples(
        #             knots_np.reshape(n_samples, -1),
        #             f_samples=costs_np,
        #         )
        #         # ipdb.set_trace()
        #         success = problem.initialize_kernel(self.ksos_sigma, self.ksos_kernel)
        #         if not success:
        #             print("Kernel initialization failed")
        #             return params.mean

        #         try:
        #             updated_knots, _ = newton.damped_newton_advanced(
        #                 problem,
        #                 iterations=self.ksos_iterations,
        #                 verbose=False,
        #                 linesearch=self.ksos_linesearch,
        #                 return_B=False,
        #             )
        #             updated_knots = updated_knots.reshape(
        #                 self.num_knots, self.task.model.nu
        #             ).astype(np.float32)
        #             if not np.isfinite(updated_knots).all():
        #                 print("damped_newton_advanced failed")
        #                 return params.mean
        #             return updated_knots
        #         except Exception:
        #             print("damped_newton_advanced failed")
        #             return params.mean

        #     updated_mean = io_callback(  # type: ignore[arg-type]
        #         _ksos_solver_callback,
        #         jax.ShapeDtypeStruct(
        #             (self.num_knots, self.task.model.nu), ksos_knots.dtype
        #         ),
        #         ksos_knots,
        #         ksos_costs
        #     )
            
        #     # compute the control sequence from the knots and compare best costs
        #     ksos_best_costs = self.get_cost_with_best_samples(
        #         params.states, params, params.tk, updated_mean
        #     )
        #     mppi_best_costs = self.get_cost_with_best_samples(
        #         params.states, params, params.tk, mean
        #     )

        #     # reduce to scalar best costs (min over rollouts)
        #     ksos_best = jnp.min(ksos_best_costs)
        #     mppi_best = jnp.min(mppi_best_costs)

        #     # Branch using jax.lax.cond so behavior is correct under JIT/VMAP.
        #     def _accept(_):
        #         jax.debug.print("KSOS accepted ({} < {})", ksos_best, mppi_best)
        #         return updated_mean

        #     def _reject(_):
        #         jax.debug.print("KSOS rejected ({} >= {})", ksos_best, mppi_best)
        #         return mean

        #     new_mean = jax.lax.cond(jnp.less(ksos_best, mppi_best), _accept, _reject, operand=None)

        #     return new_mean

        # def _keep(ops):
        #     _, mean = ops
        #     return mean
        # # operand 是一个 pytree，这里用 (new_iter, mean)
        # update_mean = jax.lax.cond(predicate, _do_ksos, _keep, operand=(new_iter, mean))

        # jax.debug.print("update_mean (every10) {}", update_mean)

        # use update_mean (may be equal to mean when predicate is false)
        return params.replace(mean=mean, iter=new_iter)
