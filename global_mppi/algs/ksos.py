from typing import Literal, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import io_callback
from flax.struct import dataclass

from global_mppi.alg_base import SamplingBasedController, SamplingParams, Trajectory
from global_mppi.risk import RiskStrategy
from global_mppi.task_base import Task

import ipdb
from ksos_tools.solvers import newton

@dataclass
class KSOSParams(SamplingParams):
    """Policy parameters for model-predictive path integral control.

    Same as SamplingParams, but with a different name for clarity.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
    """
    iter: jax.Array

class KSOS(SamplingBasedController):
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
    ) -> KSOSParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        iter = 0
        return KSOSParams(tk=_params.tk, mean=_params.mean,iter=iter, rng=_params.rng)

    def sample_knots(self, params: KSOSParams) -> Tuple[jax.Array, KSOSParams]:
        """Sample a control sequence."""
        rng, _ = jax.random.split(params.rng)
        
        # sample uniform for exploration
        decay = jnp.power(0.9, params.iter).astype(params.mean.dtype)
        half_range = 0.6 * decay
        # half_range = 1.0 - params.iter * 0.15
        min_controls = params.mean - half_range
        max_controls = params.mean + half_range
        fractions = jnp.linspace(
            0.0, 1.0, self.num_samples, dtype=params.mean.dtype
        )[:, None, None]
        controls = min_controls + fractions * (max_controls - min_controls)

        # rng, sample_rng = jax.random.split(params.rng)
        
        # # sample uniform for exploration
        # decay = jnp.power(0.9, params.iter).astype(params.mean.dtype)
        # half_range = 0.6 * decay
        # # half_range = 1.0 - params.iter * 0.15
        # controls = jax.random.uniform(
        #     sample_rng,            
        #     (
        #         self.num_samples,
        #         self.num_knots,
        #         self.task.model.nu,
        #     ),
        #     minval=params.mean - half_range,
        #     maxval=params.mean + half_range
        #     # minval=self.task.u_min,
        #     # maxval=self.task.u_max
        # )
        
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: KSOSParams, rollouts: Trajectory
    ) -> KSOSParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)

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
                print("Run KSOS solver succeeded")
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
                (self.num_knots, self.task.model.nu), rollouts.knots.dtype
            ),
            rollouts.knots,
            costs
        )
        # jax.debug.print("run updated_mean{}",updated_mean)
        # ipdb.set_trace()
        # minimal cost
        # indices = jnp.argsort(costs)
        # elites = indices[:1]
        # updated_mean = rollouts.knots[elites[0]]
        
        return params.replace(mean=updated_mean)
