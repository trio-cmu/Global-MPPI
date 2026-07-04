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
from ksos_tools.solvers import newton
from typing import Any, Literal, Tuple
from scipy.stats import qmc
from jax import lax

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
    lse_cost: jax.Array
    cost: jax.Array
    knots: jax.Array
    ksos_mean: jax.Array
    ksos_result_cost: jax.Array
    ksos_trace: jax.Array

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
        ctrl_name : str = "mppiksos",
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
            ctrl_name=ctrl_name,
        )
        # mppi parameters
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.temperature = temperature
        
        # lse parameters
        self.lse_num_samples = 100
  
        self.lse_smoothing_sigma = jnp.array([0.3, 0.2, 0.1, 0.05, 0.001])
        # self.lse_smoothing_sigma = jnp.array([0.4, 0.2, 0.1, 0.05, 0.0025])
        self.lse_lambda = 0.1
        
        # ksos parameters
        self.ksos_num_samples = 256 # 256
        self.ksos_lambda = 1e-5 # 1e-3
        self.ksos_sigma = 0.01 # 0.1
        self.ksos_epsilon = 1e-3
        self.ksos_kernel = "Laplace" 
        # self.ksos_kernel = "Gauss"
        self.ksos_iterations = 100
        self.ksos_linesearch = True
        self.ksos_num_restart = 5
        # self.ksos_sampling = "linspace"
        self.ksos_sampling = "uniform" # for pushT
        self.decay_rate = 0.85  #  for pushT
        
        self.debug = False
        self.is_lse_smoothing = True

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> MPPIKSOSParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        iter = 0
        lse_cost = jnp.array(0.0)
        cost = jnp.array(0.0)
        ksos_mean = jnp.zeros_like(_params.mean)
        ksos_result_cost = jnp.array(0.0)
        return MPPIKSOSParams(tk=_params.tk, mean=_params.mean, rng=_params.rng, iter=iter, lse_cost=lse_cost, cost = cost, ksos_mean=ksos_mean, ksos_result_cost = ksos_result_cost, knots=None, best_cost= _params.best_cost, ksos_trace=None,best_trace=_params.best_trace)

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
        sigma = self.lse_smoothing_sigma[params.iter]
        controls = params.mean + sigma * noise
        jax.debug.print("sigma: {}", sigma)
        # controls = params.mean + self.noise_level * noise

        return controls, params.replace(rng=rng)

    def sample_ksos_knots(self, params: MPPIKSOSParams) -> Tuple[jax.Array, MPPIKSOSParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)

        center = lax.cond(
            jnp.equal(params.iter, 0),
            lambda _: jnp.zeros_like(params.ksos_mean),
            lambda _: params.ksos_mean,
            operand=None,
        )
        # cov = lax.cond(
        #     jnp.equal(params.iter, 0),
        #     lambda _: jnp.full_like(params.mean, self.sigma_start),
        #     lambda _: params.cov,
        #     operand=None,
        # )
        # center = params.ksos_mean
        # jax.debug.print("center!!!! {}", center)
        radius = (
            jnp.power(self.decay_rate, params.iter).astype(params.mean.dtype)
            * self.task.u_max
        )
        # ipdb.set_trace()
        if self.ksos_sampling == "linspace":
            ksos_knots_old = jnp.linspace(
                center - radius,
                center + radius,
                self.ksos_num_samples,
            )                           
            # Shape: [ksos_num_samples, num_knots, nu]
            # perm = np.random.permutation(6)   # e.g. [3, 1, 5, 0, 2, 4]
            # ksos_knots = ksos_knots_old[:, perm, :]
            ksos_knots_old = ksos_knots_old.reshape(-1, 2)     # shape (24, 2)
            np.random.shuffle(ksos_knots_old)
            ksos_knots_old = ksos_knots_old.reshape(30, 6, 2)
            # ipdb.set_trace()
        elif self.ksos_sampling == "uniform":
            ksos_knots = jax.random.uniform(
                sample_rng,
                (
                    self.ksos_num_samples,
                    self.num_knots,
                    self.task.model.nu,
                ),                          
                minval=center - radius,
                maxval=center + radius,
            )
        elif self.ksos_sampling == "gaussian":
            noise = jax.random.normal(
                sample_rng,
                (
                    self.ksos_num_samples,
                    self.num_knots,
                    self.task.model.nu,
                ),
            )
            ksos_knots = center + radius * noise
        elif self.ksos_sampling == "Sobol":
            dim = int(center.size)
            u = self.sobol_unit  # (N, dim) in [0,1)
            shift = jax.random.uniform(sample_rng, (dim,), dtype=u.dtype)
            u = jnp.mod(u + shift, 1.0)

            lower = (center - radius).reshape(-1)
            upper = (center + radius).reshape(-1)

            samples = lower + (upper - lower) * u   # (N, dim)

            ksos_knots = samples.astype(params.mean.dtype).reshape(
                self.ksos_num_samples, self.num_knots, self.task.model.nu
            )
        else:
            raise ValueError(f"Unknown ksos_sampling mode: {self.ksos_sampling}")
        
        return ksos_knots, params.replace(rng=rng)
    
    def sample_lse_knots(self, ksos_knots: jax.Array, params: MPPIKSOSParams) -> Tuple[jax.Array, MPPIKSOSParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        num_ksos = ksos_knots.shape[0]
        noise = jax.random.normal(
            sample_rng,
            (
                num_ksos,
                self.lse_num_samples,
                self.num_knots,
                self.task.model.nu,
            ),
            dtype=params.mean.dtype,
        )

        sigma = self.lse_smoothing_sigma[params.iter]
        lse_knots = ksos_knots[:, None, :, :] + sigma * noise       #Shape: [ksos_num_samples, lse_num_samples, num_knots, nu] same as [batch_size, num_samples, N]
        
        return lse_knots, params.replace(rng=rng)

    def general_rollout(
        self,
        state: mjx.Data,
        tk: jax.Array,
        knots: jax.Array,
        params: MPPIKSOSParams,
    ) -> MPPIKSOSParams:
        """Compute rollouts without applying domain randomization."""
        
        tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
        controls = self.interp_func(tq, tk, knots)  # (num_rollouts, H, nu)
        _, rollouts = self.eval_rollouts(self.model, state, controls, knots)
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        
        # filter out bad rollouts 
        # first_20 = rollouts.costs[:, :20]        # (256, 20)
        # diff = first_20[:, 1:] - first_20[:, :-1]   # (256, 19)
        # rollout_index = (diff < 0).any(axis=1).astype(int)   # (256,)
        # idx_ones = jnp.where(rollout_index == 1)[0]

        # costs = costs[idx_ones]
        # knots = knots[idx_ones]
        # trace_sites = rollouts.trace_sites[idx_ones]
        # costs = jnp.where(rollout_index == 1, costs, jnp.inf)
        
        return params.replace(knots= knots, lse_cost=costs, cost = rollouts.costs, ksos_trace=rollouts.trace_sites)
    
    def lse_smoothing_rollout(
        self,
        state: mjx.Data,
        tk: jax.Array,
        knots: jax.Array,
        params: MPPIKSOSParams,
    ) -> MPPIKSOSParams:
        """Compute rollouts without applying domain randomization."""
        
        # generate lse samples
        lse_knots, params = self.sample_lse_knots(knots, params)
        
        tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
        num_rollouts, num_lse, num_knots, nu = lse_knots.shape
        if self.debug:
            controls = self.interp_func(tq, tk, knots)  # (num_rollouts, H, nu)
            _, rollouts = self.eval_rollouts(self.model, state, controls, knots)
            params = params.replace(ksos_trace=rollouts.trace_sites)

        flat_knots = lse_knots.reshape(num_rollouts * num_lse, num_knots, nu)
        flat_controls = self.interp_func(tq, tk, flat_knots)  # (num_rollouts, H, nu)
        _, flat_rollouts = self.eval_rollouts(self.model, state, flat_controls, flat_knots)
        total_costs = flat_rollouts.costs.reshape(num_rollouts, num_lse, flat_rollouts.costs.shape[1])
        # trace_sites = flat_rollouts.trace_sites.reshape(num_rollouts, num_lse, flat_rollouts.trace_sites.shape[1], flat_rollouts.trace_sites.shape[2], flat_rollouts.trace_sites.shape[3])
        # ipdb.set_trace()
        # sum cost over time steps
        sum_costs = jnp.sum(total_costs, axis=-1)  # sum over time steps
        min_costs = jnp.min(sum_costs, axis=1, keepdims=True)
        samples_exp = jnp.exp( - (sum_costs - min_costs) / self.lse_lambda )   # Shape: [num_rollouts, num_lse]
        sum = jnp.mean(samples_exp, axis=1, keepdims=True)
        lse_cost = - jnp.log(sum) * self.lse_lambda + min_costs

        return params.replace(knots= knots, cost = total_costs, lse_cost=lse_cost.squeeze())
    
    def ksos_rollout(self, state: mjx.Data, params: Any) -> Any:
        """Perform an optimization step to update the policy parameters.

        Args:
            state: The initial state x₀.
            params: The current policy parameters, U ~ π(params).

        Returns:
            Updated policy parameters
            Rollouts used to update the parameters
        """
        # update tk with sim time
        # best_cost = jnp.atleast_1d(params.best_cost)[0]
        # plan_horizon = lax.cond(
        #     best_cost <= 0.003,
        #     lambda _: jnp.array(0.5, dtype=best_cost.dtype),
        #     lambda _: jnp.array(self.plan_horizon, dtype=best_cost.dtype),
        #     operand=None,
        # )
        
        tk = params.tk
        new_tk = (jnp.linspace(0.0, self.plan_horizon, self.num_knots) + state.time)
        new_mean = self.interp_func(new_tk, tk, params.mean[None, ...])[0]
        params = params.replace(tk=new_tk, mean=new_mean)

        # sample knots
        knots, params = self.sample_ksos_knots(params)
        knots = jnp.clip(knots, self.task.u_min, self.task.u_max)

        rng, dr_rng = jax.random.split(params.rng)
        
        # compute lse smoothed costs
        if self.is_lse_smoothing:
            params = self.lse_smoothing_rollout(state, tk, knots, params)
        else:
            params = self.general_rollout(state, tk, knots, params)

        params = params.replace(rng=rng)
        # downsample to get elites (top k samples)
        # indices = jnp.argsort(params.lse_cost)
        # elites = indices[: 30]

        # lse_costs = params.lse_cost[elites]
        # knots_elites = params.knots[elites]
        
        # add the best cost for last (not good!!!!)
        # best_cost = jnp.atleast_1d(params.best_cost)
        # lse_costs = jnp.concatenate([lse_costs, best_cost], axis=0)
        # knots_elites = jnp.concatenate([knots_elites, params.mean[None, ...]], axis=0)
        
        # params = params.replace(lse_cost=params.lse_cost, knots=params.knots)

        return params
    
    def solve_ksos(
        self,
        params: MPPIKSOSParams,
    ) -> MPPIKSOSParams:
        """Compute rollouts without applying domain randomization."""
        
        indices = jnp.argsort(params.lse_cost)
        # elites = indices[: 60]
        elites = indices[: 80]
        
        # ipdb.set_trace()
        if self.is_lse_smoothing == False:
            costs = params.cost[elites]
        else:
            costs = params.cost.mean(axis=1)[elites]
        lse_costs = params.lse_cost[elites]
        knots_elites = params.knots[elites]

        # filter out bad rollouts
        first_20 = costs[:, :20]        # (256, 20)
        diff = first_20[:, 1:] - first_20[:, :-1]   # (256, 19)
        rollout_index = (diff < 0).any(axis=1).astype(int)   # (256,)
        idx_ones = jnp.where(rollout_index == 1)[0]

        if idx_ones.shape[0] == 0:
            print("All elites are bad rollouts !!!")
            return params
        
        costs = costs[idx_ones]
        lse_costs = lse_costs[idx_ones]
        knots_elites = knots_elites[idx_ones]

        # ipdb.set_trace()
        n_samples = lse_costs.shape[0]
        problem = newton.Problem(
            lambd=self.ksos_lambda,
            t=self.ksos_epsilon / max(n_samples, 1),
            use_K=False,
        )
        problem.register_fixed_samples(
            knots_elites.reshape(n_samples, -1),
            f_samples=lse_costs,
        )
        # ipdb.set_trace()
        success = problem.initialize_kernel(self.ksos_sigma, self.ksos_kernel)
        if not success:
            raise RuntimeError("Failed to initialize kernel")
        try:
            updated_knots, info = newton.damped_newton_advanced(
                problem,
                iterations=self.ksos_iterations,
                verbose=False,
                linesearch=self.ksos_linesearch,
                return_B=False,
            )
            updated_knots = updated_knots.reshape(self.num_knots, self.task.model.nu).astype(np.float32)

        except Exception:
            print("damped_newton_advanced failed !!!")
            return params.mean

        # ps sampling
        # print("lse_cost",params.lse_cost)
        # jax.debug.print("lse_cost: {}", params.lse_cost)
        # updated_knots = params.knots[0] 
        # ipdb.set_trace()
        return params.replace(ksos_mean=updated_knots,ksos_result_cost=info['cost'], lse_cost = lse_costs)

    def update_params(
        self, params: MPPIKSOSParams, rollouts: Trajectory
    ) -> MPPIKSOSParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        mppi_mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
        
        return params.replace(mean=mppi_mean)