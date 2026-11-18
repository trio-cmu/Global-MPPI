import os
import time
from dataclasses import replace
from typing import Any, Literal, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from global_mppi.alg_base import SamplingBasedController, SamplingParams, Trajectory
from global_mppi.risk import RiskStrategy
from global_mppi.task_base import Task
from mujoco import mjx
from ksos_tools.solvers import ksos, ADMM
from ksos_tools.problems.features import build_problem
from ksos_tools.problems.sampling import rcm_order
from jax import lax
from scipy.stats import qmc
from global_mppi.sampling import TwoLevelSampler
from global_mppi.cacs import CacsConfig, cacs_sample

@dataclass
class GlobalMPPIParams(SamplingParams):
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
    ksos_sigma: jax.Array

class GlobalMPPI(SamplingBasedController):
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
        ctrl_name : str = "globalmppi",
        ksos_solver: str = "newton",
        ksos_num_restart: int = 5,
        wendland_sigma: float = 1.75,
        ksos_sampling: str = "Sobol",
        ksos_num_samples: int = 256,
        mppi_noise: float = None,
        ksos_refine: bool = True,
        ksos_lse_samples: int = 0,
        ksos_lse_sigmas: Tuple[float, float] = (0.2, 0.01),
        ksos_lse_lambda: float = 1.0,
        ksos_lse_chunk: int = 10000,
        ksos_cluster_size: int = 10,
        ksos_gnc: Tuple[float, float] = None,
        ksos_warm_start: bool = False,
        ksos_cacs_options: dict = None,
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
            ksos_solver: "newton", "newton-rs" or "ADMM-sparse".
            ksos_num_restart: The number of KSOS restarts per planning cycle.
            wendland_sigma: Wendland kernel radius at restart 0 (ADMM-sparse
                            only); it shrinks with `decay_rate` per restart.
            ksos_sampling: KSOS candidate design, "Sobol" (plain, in the
                           trust region), "two-level" (two-level Sobol
                           clusters in the trust region; the sampler also
                           sets the kernel radius) or "cacs" (budgeted
                           cost-aware chordal sampling over the whole box,
                           global_mppi/cacs.py; ADMM-cuda only, it rolls
                           out its own candidates in `solve_ksos`).
            ksos_num_samples: KSOS candidates per restart.
            mppi_noise: Fixed noise level of the MPPI step after the
                        restarts; None keeps `lse_smoothing_sigma[iter]`.
            ksos_refine: Run MPPI after the KSOS restarts. False executes
                         the KSOS solution as it is.
            ksos_lse_samples: ADMM-cuda path: LSE smoothing of the KSOS
                              costs with this many Gaussian perturbations
                              per candidate (0 = off).
            ksos_lse_sigmas: LSE perturbation scale at the first and last
                             restart (geometric in between).
            ksos_lse_lambda: LSE temperature.
            ksos_lse_chunk: Perturbed rollouts per sequential GPU batch.
            ksos_cluster_size: Points per cluster of the two-level design,
                               which is the clique size of the ADMM problem.
            ksos_gnc: ADMM-cuda path: kernel-surrogate GNC. Each candidate's
                      cost becomes the Gaussian-weighted mean of the
                      restart's costs, bandwidth = factor x the sampler's
                      kernel radius, the factor geometric from the first to
                      the last value over the restarts (no extra rollouts).
                      None keeps the plain costs.
            ksos_warm_start: CACS path: at restart 0, the previous cycle's
                             plan (shifted to the current time) joins the
                             CACS initial batch and is a guard candidate,
                             so the KSOS result is never worse than it.
            ksos_cacs_options: CACS path: overrides of the CacsConfig defaults
                               (global_mppi/cacs.py), e.g. {"k": 128}; n is
                               ksos_num_samples.
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
        self.lse_lambda = 0.1

        # ksos parameters
        self.ksos_num_samples = ksos_num_samples
        self.elite_num = 130
        self.ksos_lambda = 1e-5
        self.ksos_sigma = 0.01
        self.ksos_epsilon = 1e-2
        self.ksos_kernel = "Laplace"
        self.ksos_iterations = 100
        self.ksos_linesearch = True
        self.ksos_num_restart = ksos_num_restart
        self.ksos_solver = ksos_solver
        # ADMM-sparse: compactly supported Wendland kernel (dimension-correct
        # form chosen by ksos-tools' build_problem), RCM-ordered samples
        self.wendland_sigma = wendland_sigma
        self.admm_options = dict(
            eps_abs=1e-6, eps_rel=1e-5, time_limit=2.0, record_history=False
        )
        self.ksos_sampling = ksos_sampling
        self.mppi_noise = mppi_noise
        self.ksos_refine = ksos_refine
        self.ksos_lse_samples = ksos_lse_samples
        self.ksos_lse_sigmas = jnp.asarray(
            np.geomspace(*ksos_lse_sigmas, max(ksos_num_restart, 1)), dtype=jnp.float32
        )
        self.ksos_lse_lambda = ksos_lse_lambda
        self.ksos_lse_chunk = ksos_lse_chunk
        self.ksos_warm_start = ksos_warm_start
        self.ksos_gnc_factors = (
            np.geomspace(*ksos_gnc, max(ksos_num_restart, 1)) if ksos_gnc else None
        )
        # ADMM-cuda: cuKernelSOS's GPU ADMM, the single-step test's settings
        self.admm_cuda_options = dict(
            eps_abs=1e-6, eps_rel=1e-5, time_limit=2.0, record_history=False,
            backend="cuda", cuda_options={"anderson_reg": 1e-10},
            # post-processing that does not change the plan: _solve_cacs / _solve_admm_cuda use x only if the solve
            # converged, so skip the eigenvalue check + repair of the others; the repair bisection to 1e-6 relative
            # (returns the feasible end of the bracket); no eigmin diagnostic of the repaired alpha
            postprocess="converged", repair_tol=1e-6, diagnostics=False,
        )
        if os.environ.get("KSOS_GPU_SOLVE") == "1":   # run_exp.sh: affine solve on the GPU (cuKernelSOS gpu-solve build)
            self.admm_cuda_options["cuda_options"]["gpu_solve"] = True
        self.cacs_config = CacsConfig(**{**(ksos_cacs_options or {}), "n": ksos_num_samples})
        self._cacs_rng = np.random.default_rng(seed + 101)
        self._cacs_calls = 0
        if ksos_sampling == "two-level":
            self.two_level = TwoLevelSampler(
                ksos_num_samples, self.num_knots * self.task.model.nu, cluster_size=ksos_cluster_size, seed=seed
            )
        # total rollout cost of knot sequences (m, num_knots, nu), for the
        # ADMM-cuda guard (x and the best sample)
        self._jit_costs = jax.jit(
            lambda state, tk, knots: self.rollout(state, tk, knots).costs.sum(axis=-1)
        )
        self.decay_rate = 0.80
        self.is_lse_smoothing = False

        # Scrambled Sobol points in the unit cube, (ksos_num_samples, dim),
        # generated once; `sample_ksos_knots` applies a random shift each
        # call and maps them onto the trust region.
        self.sobol_unit = jnp.asarray(
            qmc.Sobol(
                d=self.num_knots * self.task.model.nu, scramble=True, seed=seed
            ).random(self.ksos_num_samples),
            dtype=jnp.float32,
        )

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> GlobalMPPIParams:
        """Initialize the policy parameters, extending the base params with the
        extra fields GlobalMPPI needs to track the KSOS/LSE optimization state
        (iteration counter, LSE-smoothed costs, KSOS mean and trace)."""
        _params = super().init_params(initial_knots, seed)
        iter = 0
        lse_cost = jnp.array(0.0)
        cost = jnp.array(0.0)
        ksos_mean = jnp.zeros_like(_params.mean)
        ksos_result_cost = jnp.array(0.0)
        ksos_sigma = jnp.array(0.0, dtype=jnp.float32)
        return GlobalMPPIParams(tk=_params.tk, mean=_params.mean, rng=_params.rng, iter=iter, lse_cost=lse_cost, cost = cost, ksos_mean=ksos_mean, ksos_result_cost = ksos_result_cost, knots=None, best_cost= _params.best_cost, ksos_trace=None,best_trace=_params.best_trace, ksos_sigma=ksos_sigma)

    def sample_knots(self, params: GlobalMPPIParams) -> Tuple[jax.Array, GlobalMPPIParams]:
        """Sample MPPI candidate knots as Gaussian noise around the current
        mean, with the noise scale taken from `lse_smoothing_sigma` at the
        current iteration (used by the base class's `optimize`/MPPI update,
        as opposed to the KSOS candidate sampling in `sample_ksos_knots`)."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.num_knots,
                self.task.model.nu,
            ),
        )
        sigma = (
            self.lse_smoothing_sigma[params.iter]
            if self.mppi_noise is None
            else self.mppi_noise
        )
        controls = params.mean + sigma * noise

        return controls, params.replace(rng=rng)

    def sample_ksos_knots(self, params: GlobalMPPIParams) -> Tuple[jax.Array, GlobalMPPIParams]:
        """Sample `ksos_num_samples` candidate knot sequences for the KSOS
        solver. Candidates are drawn from a trust region centered at the
        previous restart's KSOS solution, `params.mean` (or zero on the first
        iteration), whose radius shrinks geometrically with `decay_rate` each
        iteration.
        The sampling distribution is controlled by `self.ksos_sampling`
        ("linspace", "uniform", "gaussian", or "Sobol")."""
        rng, sample_rng = jax.random.split(params.rng)

        center = lax.cond(
            jnp.equal(params.iter, 0),
            lambda _: jnp.zeros_like(params.mean),
            lambda _: params.mean,
            operand=None,
        )
        radius = (
            jnp.power(self.decay_rate, params.iter).astype(params.mean.dtype)
            * self.task.u_max
        )
        if self.ksos_sampling == "linspace":
            ksos_knots_old = jnp.linspace(
                center - radius,
                center + radius,
                self.ksos_num_samples,
            )
            # Shape: [ksos_num_samples, num_knots, nu]
            ksos_knots_old = ksos_knots_old.reshape(-1, 2)     # shape (24, 2)
            np.random.shuffle(ksos_knots_old)
            ksos_knots_old = ksos_knots_old.reshape(30, 6, 2)
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
        elif self.ksos_sampling == "two-level":
            # the trust region clipped into the control box; the sampler
            # returns its own kernel radius for this design
            u_min = jnp.broadcast_to(self.task.u_min, center.shape)
            u_max = jnp.broadcast_to(self.task.u_max, center.shape)
            lower = jnp.maximum(center - radius, u_min).reshape(-1)
            upper = jnp.minimum(center + radius, u_max).reshape(-1)
            samples, sigma, _ = self.two_level.sample(sample_rng, lower, upper)
            ksos_knots = samples.astype(params.mean.dtype).reshape(
                self.ksos_num_samples, self.num_knots, self.task.model.nu
            )
            return ksos_knots, params.replace(rng=rng, ksos_sigma=sigma)
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
    
    def sample_lse_knots(self, ksos_knots: jax.Array, params: GlobalMPPIParams) -> Tuple[jax.Array, GlobalMPPIParams]:
        """For each KSOS candidate knot sequence, sample `lse_num_samples`
        nearby perturbations (Gaussian noise scaled by the current
        `lse_smoothing_sigma`) used to estimate a smoothed cost for that
        candidate via log-sum-exp in `lse_smoothing_rollout`."""
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
        params: GlobalMPPIParams,
    ) -> GlobalMPPIParams:
        """Roll out each KSOS candidate knot sequence once (no domain
        randomization, no LSE smoothing) and record its total cost. This is
        the cheaper alternative to `lse_smoothing_rollout`, used when
        `self.is_lse_smoothing` is False."""

        tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
        controls = self.interp_func(tq, tk, knots)  # (num_rollouts, H, nu)
        _, rollouts = self.eval_rollouts(self.model, state, controls, knots)
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps

        return params.replace(knots= knots, lse_cost=costs, cost = rollouts.costs, ksos_trace=rollouts.trace_sites)
    
    def lse_smoothing_rollout(
        self,
        state: mjx.Data,
        tk: jax.Array,
        knots: jax.Array,
        params: GlobalMPPIParams,
    ) -> GlobalMPPIParams:
        """Estimate a smoothed cost for each KSOS candidate knot sequence by
        rolling out `lse_num_samples` nearby perturbations (from
        `sample_lse_knots`) and combining their costs with a log-sum-exp soft
        minimum, tempered by `lse_lambda`. This trades extra rollouts for a
        cost estimate that is less sensitive to a single unlucky sample than
        `general_rollout`."""

        # generate lse samples
        lse_knots, params = self.sample_lse_knots(knots, params)
        
        tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
        num_rollouts, num_lse, num_knots, nu = lse_knots.shape

        flat_knots = lse_knots.reshape(num_rollouts * num_lse, num_knots, nu)
        flat_controls = self.interp_func(tq, tk, flat_knots)  # (num_rollouts, H, nu)
        _, flat_rollouts = self.eval_rollouts(self.model, state, flat_controls, flat_knots)
        total_costs = flat_rollouts.costs.reshape(num_rollouts, num_lse, flat_rollouts.costs.shape[1])
        
        # sum cost over time steps
        sum_costs = jnp.sum(total_costs, axis=-1)  # sum over time steps
        min_costs = jnp.min(sum_costs, axis=1, keepdims=True)
        samples_exp = jnp.exp( - (sum_costs - min_costs) / self.lse_lambda )   # Shape: [num_rollouts, num_lse]
        sum = jnp.mean(samples_exp, axis=1, keepdims=True)
        lse_cost = - jnp.log(sum) * self.lse_lambda + min_costs

        return params.replace(knots= knots, cost = total_costs, lse_cost=lse_cost.squeeze())
    
    def chunked_lse_rollout(
        self,
        state: mjx.Data,
        tk: jax.Array,
        knots: jax.Array,
        params: GlobalMPPIParams,
    ) -> GlobalMPPIParams:
        """LSE-smoothed cost of each candidate, as `lse_smoothing_rollout`
        but for many candidates: `ksos_lse_samples` Gaussian perturbations of
        scale `ksos_lse_sigmas[iter]` per candidate (clipped into the control
        box), rolled out `ksos_lse_chunk` at a time with `lax.map` so the
        GPU memory stays bounded; only the total costs are kept. The smoothed
        cost is -lambda log mean exp(-J / lambda), the minimum subtracted."""
        rng, sample_rng = jax.random.split(params.rng)
        n, L = knots.shape[0], self.ksos_lse_samples
        sigma = self.ksos_lse_sigmas[params.iter]
        noise = jax.random.normal(sample_rng, (n, L) + knots.shape[1:], dtype=knots.dtype)
        perturbed = jnp.clip(knots[:, None] + sigma * noise, self.task.u_min, self.task.u_max)
        flat = perturbed.reshape((n * L,) + knots.shape[1:])
        chunk = self.ksos_lse_chunk
        pad = (-flat.shape[0]) % chunk
        flat = jnp.concatenate([flat, jnp.repeat(flat[:1], pad, axis=0)])
        costs = lax.map(
            lambda k: self.rollout(state, tk, k).costs.sum(axis=-1),
            flat.reshape((-1, chunk) + knots.shape[1:]),
        ).reshape(-1)[: n * L].reshape(n, L)
        m = jnp.min(costs, axis=1, keepdims=True)
        lam = self.ksos_lse_lambda
        lse = m - lam * jnp.log(jnp.mean(jnp.exp(-(costs - m) / lam), axis=1, keepdims=True))
        return params.replace(
            knots=knots, lse_cost=lse.squeeze(-1), cost=jnp.zeros((n, 1), costs.dtype), rng=rng
        )

    def ksos_rollout(self, state: mjx.Data, params: Any) -> Any:
        """Warm-start the spline knot times to the current sim time, sample a
        batch of KSOS candidate knots, and roll them all out to get a cost
        for each candidate (via `lse_smoothing_rollout` or `general_rollout`,
        depending on `self.is_lse_smoothing`). The resulting per-candidate
        costs are consumed by `solve_ksos` to fit an updated mean.

        Args:
            state: The initial state x₀.
            params: The current policy parameters, U ~ π(params).

        Returns:
            Updated policy parameters, with `knots`/`lse_cost`/`cost` set to
            the sampled candidates and their evaluated costs.
        """
        # update tk with sim time
        tk = params.tk
        new_tk = (jnp.linspace(0.0, self.plan_horizon, self.num_knots) + state.time)
        new_mean = self.interp_func(new_tk, tk, params.mean[None, ...])[0]
        params = params.replace(tk=new_tk, mean=new_mean)
        if self.ksos_sampling == "cacs":  # CACS samples and rolls out on the host, in solve_ksos
            return params

        # sample knots
        knots, params = self.sample_ksos_knots(params)
        knots = jnp.clip(knots, self.task.u_min, self.task.u_max)

        rng, dr_rng = jax.random.split(params.rng)
        
        # compute lse smoothed costs
        if self.ksos_lse_samples:
            params = self.chunked_lse_rollout(state, tk, knots, params)
        elif self.is_lse_smoothing:
            params = self.lse_smoothing_rollout(state, tk, knots, params)
        else:
            params = self.general_rollout(state, tk, knots, params)

        params = params.replace(rng=rng)

        return params
    
    def solve_ksos(
        self,
        params: GlobalMPPIParams,
        state: mjx.Data = None,
    ) -> GlobalMPPIParams:
        """Fit an updated mean from the KSOS candidates rolled out in
        `ksos_rollout`. Takes the `self.elite_num` lowest-cost candidates,
        drops any whose per-step cost trace never decreases (a heuristic filter for rollouts
        that never make progress), and hands the survivors to the KernelSOS
        solver (`ksos.solve`, using `self.ksos_solver`) to produce a new
        `ksos_mean`. Falls back to returning `params` unchanged if every
        elite is filtered out or the solver fails."""

        if self.ksos_solver == "ADMM-cuda" and self.ksos_sampling == "cacs":
            return self._solve_cacs(params, state)
        if self.ksos_solver == "ADMM-cuda":
            return self._solve_admm_cuda(params, state)

        # Elite selection/filtering runs on the host in numpy: the filter
        # below has a data-dependent output shape, which in jnp would trigger
        # a fresh XLA compile almost every cycle.
        lse_cost = np.asarray(params.lse_cost)
        indices = np.argsort(lse_cost)
        elites = indices[: self.elite_num]

        if self.is_lse_smoothing == False:
            costs = np.asarray(params.cost)[elites]
        else:
            costs = np.asarray(params.cost).mean(axis=1)[elites]
        lse_costs = lse_cost[elites]
        knots_elites = np.asarray(params.knots)[elites]

        # filter out bad rollouts
        first_20 = costs[:, :20]
        diff = first_20[:, 1:] - first_20[:, :-1]
        rollout_index = (diff < 0).any(axis=1).astype(int)
        idx_ones = np.where(rollout_index == 1)[0]

        if idx_ones.shape[0] == 0:
            print("All elites are bad rollouts !!!")
            return params
        
        costs = costs[idx_ones]
        lse_costs = lse_costs[idx_ones]
        knots_elites = knots_elites[idx_ones]

        n_samples = lse_costs.shape[0]
        samples = np.asarray(
            knots_elites.reshape(n_samples, -1),
            dtype=np.float64,
        )
        sample_costs = np.asarray(lse_costs, dtype=np.float64).reshape(-1)
        try:
            if self.ksos_solver == "ADMM-sparse":
                updated_knots, info = self._solve_admm(
                    samples, sample_costs, int(params.iter)
                )
            else:
                updated_knots, info = ksos.solve(
                    f=None,
                    samples=samples,
                    f_samples=sample_costs,
                    lambd=self.ksos_lambda,
                    sigma=float(self.ksos_sigma),
                    epsilon=self.ksos_epsilon,
                    kernel=self.ksos_kernel,
                    solver=self.ksos_solver,
                )
            if updated_knots is None:
                raise RuntimeError(
                    info.get("status", "KSOS solver returned no solution")
                )
            updated_knots = np.asarray(
                updated_knots,
                dtype=np.float32,
            ).reshape(
                self.num_knots,
                self.task.model.nu,
            )
        except Exception as error:
            print(f"KSOS solver failed: {error}")
            return params

        # The new KSOS solution becomes the mean, so the next restart's trust
        # region (and MPPI after the last restart) is centered on it. On any
        # failure above, `mean` keeps the previous solution.
        return params.replace(
            mean=updated_knots,
            ksos_mean=updated_knots,
            # keep params' array shapes fixed so jitted calls don't re-trace
            ksos_result_cost=jnp.asarray(info["cost"], dtype=jnp.float32),
        )

    def _solve_admm(self, samples: np.ndarray, f: np.ndarray, it: int):
        """ADMM-sparse KSOS on the elites, with a Wendland kernel of radius
        `wendland_sigma * decay_rate**it` (shrinking with the trust region).
        The samples are RCM-ordered so K's band, and so the cliques, stay
        small; the solution is an alpha-weighted mean of the samples, so the
        order does not need undoing."""
        sigma = self.wendland_sigma * self.decay_rate**it
        perm = rcm_order(samples, sigma)
        problem = build_problem(
            samples[perm], sigma, lambda _: f[perm], lambd=self.ksos_lambda
        )
        x, info = ADMM.solve_ksos(
            problem.Phi,
            problem.f,
            problem.lambd,
            samples=problem.samples,
            **self.admm_options,
        )
        print(
            f"ADMM-sparse: n={problem.n} sigma={sigma:.3f} "
            f"band={problem.bandwidth} "
            f"max_clique={info['admm_stats']['max_clique']} "
            f"mean_clique={info['admm_stats']['mean_clique']:.1f} "
            f"num_cliques={info['admm_stats']['num_cliques']} "
            f"status={info['status']} "
            f"iters={info['iterations']} cost={info['cost']:.4f} "
            f"time={info['setup_time'] + info['solve_time']:.3f}s"
        )
        if not info["success"]:
            return None, info
        return x, info

    def _solve_cacs(self, params: GlobalMPPIParams, state: mjx.Data) -> GlobalMPPIParams:
        """CACS in this restart's trust region (it rolls out its own candidates):
        the box of half-width decay_rate**iter around the previous restart's
        solution (the whole control box at restart 0), with the kernel radius h
        shrunk by the same factor,
        then KSOS with cuKernelSOS's GPU ADMM on the CACS samples in reverse
        insertion order (a perfect elimination order of the CACS certificate),
        Wendland kernel of the CACS radius h, then the guard: the cheapest (by a
        fresh rollout) of x, the best sample and, after restart 0, the previous
        restart's solution."""
        shape = (self.num_knots, self.task.model.nu)
        lo = np.tile(np.asarray(self.task.u_min, dtype=np.float64), self.num_knots)
        hi = np.tile(np.asarray(self.task.u_max, dtype=np.float64), self.num_knots)
        it = int(params.iter)
        scale = self.decay_rate ** it
        center = np.asarray(params.mean, dtype=np.float64).reshape(-1) if it > 0 else np.zeros_like(lo)
        box_lo = np.maximum(center - scale * hi, lo)
        box_hi = np.minimum(center + scale * hi, hi)
        cfg = replace(self.cacs_config, h=self.cacs_config.h * scale)

        def roll(points, pad_to=None):
            points = np.clip(np.atleast_2d(points), lo, hi)
            m = len(points)
            if pad_to is not None and m < pad_to:  # fixed batch shape: no recompile per round
                points = np.vstack([points, np.zeros((pad_to - m, len(lo)))])
            c = self._jit_costs(state, params.tk, jnp.asarray(points.reshape(-1, *shape), jnp.float32))
            return np.asarray(c, dtype=np.float64)[:m]

        t0 = time.perf_counter()
        # warm start: the previous cycle's plan, already shifted to this cycle's knot times
        warm = np.asarray(params.mean, dtype=np.float64).reshape(1, -1) if self.ksos_warm_start and it == 0 else None
        samples, f, stats = cacs_sample(
            roll, box_lo, box_hi, cfg, seed=self._cacs_calls, rng=self._cacs_rng, extra=warm
        )
        self._cacs_calls += 1
        t1 = time.perf_counter()
        order = np.arange(len(samples))[::-1]
        problem = build_problem(samples[order], cfg.h, lambda _: f[order], lambd=self.ksos_lambda)
        try:
            x, info = ADMM.solve_ksos(
                problem.Phi, problem.f, problem.lambd, samples=problem.samples,
                **self.admm_cuda_options,
            )
            ok = bool(info["success"]) and np.all(np.isfinite(x))
        except Exception as error:
            print(f"KSOS solver failed: {error}")
            x, info, ok = None, {"status": "error", "iterations": 0}, False
        t2 = time.perf_counter()
        candidates = [(samples[np.argmin(f)], float(f.min()))]
        if warm is not None:
            candidates.append((np.clip(warm[0], lo, hi), float(stats["extra_costs"][0])))
        x_cost = None
        if ok:
            x = np.clip(np.asarray(x).reshape(-1), lo, hi)
            x_cost = float(roll(x)[0])
            candidates.append((x, x_cost))
        if int(params.iter) > 0:
            candidates.append((np.asarray(params.mean).reshape(-1), float(params.ksos_result_cost)))
        knots, cost = min(candidates, key=lambda c: c[1])
        t3 = time.perf_counter()
        print(f"KSOS timing: cacs {t1 - t0:.4f} admm {t2 - t1:.4f} guard {t3 - t2:.4f}")
        print(
            f"ADMM-cuda: restart {it} CACS n={len(samples)} h={cfg.h:.3f} rollouts={stats['rollouts']} "
            f"rounds={stats['rounds']} replaced={stats['replaced']} status={info['status']} "
            f"iters={info['iterations']} best sample {f.min():.4f}"
            + (f" warm {float(stats['extra_costs'][0]):.4f}" if warm is not None else "")
            + (f" x {x_cost:.4f}" if x_cost is not None else "")
            + f" -> solution {cost:.4f}"
        )
        knots = jnp.asarray(np.asarray(knots, dtype=np.float32).reshape(shape))
        return params.replace(
            mean=knots,
            ksos_mean=knots,
            ksos_result_cost=jnp.asarray(cost, dtype=jnp.float32),
        )

    def _solve_admm_cuda(
        self, params: GlobalMPPIParams, state: mjx.Data
    ) -> GlobalMPPIParams:
        """KSOS on all candidates of this restart with cuKernelSOS's GPU ADMM:
        RCM order, Wendland kernel of the sampler's radius, then a guard: the
        restart's solution is the cheapest (by a fresh rollout) of x, the
        best candidate and the previous restart's solution. On restart 0
        there is no previous solution."""
        t0 = time.perf_counter()
        samples = np.asarray(params.knots, dtype=np.float64).reshape(self.ksos_num_samples, -1)
        f = np.asarray(params.lse_cost, dtype=np.float64).reshape(-1)
        sigma = float(params.ksos_sigma)
        f_true = f
        if self.ksos_gnc_factors is not None:
            h = self.ksos_gnc_factors[int(params.iter)] * sigma
            d2 = np.sum((samples[:, None] - samples[None]) ** 2, axis=-1)
            w = np.exp(-0.5 * d2 / h ** 2)
            f = (w @ f_true) / w.sum(axis=1)
        t1 = time.perf_counter()
        perm = rcm_order(samples, sigma)
        problem = build_problem(samples[perm], sigma, lambda _: f[perm], lambd=self.ksos_lambda)
        t2 = time.perf_counter()
        try:
            x, info = ADMM.solve_ksos(
                problem.Phi, problem.f, problem.lambd, samples=problem.samples,
                **self.admm_cuda_options,
            )
            ok = bool(info["success"]) and np.all(np.isfinite(x))
        except Exception as error:
            print(f"KSOS solver failed: {error}")
            x, info, ok = None, {"status": "error", "iterations": 0}, False
        t3 = time.perf_counter()

        shape = (self.num_knots, self.task.model.nu)
        best = samples[np.argmin(f)]
        if self.ksos_lse_samples:  # f is smoothed: judge the best candidate by its own rollout
            best_cost = float(self._jit_costs(state, params.tk, jnp.asarray(best.reshape(1, self.num_knots, -1), jnp.float32))[0])
        else:  # with GNC, f is smoothed: the candidate's own cost is already known
            best_cost = float(f_true[np.argmin(f)])
        candidates = [(best, best_cost)]
        if ok:
            x = np.clip(x.reshape(shape), self.task.u_min, self.task.u_max).reshape(-1)
            x_cost = float(self._jit_costs(state, params.tk, jnp.asarray(x.reshape(1, *shape), jnp.float32))[0])
            candidates.append((x, x_cost))
        if int(params.iter) > 0:
            candidates.append((np.asarray(params.mean).reshape(-1), float(params.ksos_result_cost)))
        knots, cost = min(candidates, key=lambda c: c[1])
        t4 = time.perf_counter()
        print(
            f"KSOS timing: to host {t1 - t0:.4f} rcm+kernel {t2 - t1:.4f} admm {t3 - t2:.4f} "
            f"guard {t4 - t3:.4f}"
        )
        print(
            f"ADMM-cuda: restart {int(params.iter)} n={problem.n} sigma={sigma:.3f} "
            f"status={info['status']} iters={info['iterations']} "
            f"best sample {best_cost:.4f}"
            + (f" (LSE {f.min():.4f})" if self.ksos_lse_samples else "")
            + (f" (GNC {f.min():.4f})" if self.ksos_gnc_factors is not None else "")
            + (f" x {x_cost:.4f}" if ok else "")
            + f" -> solution {cost:.4f}"
        )
        knots = jnp.asarray(np.asarray(knots, dtype=np.float32).reshape(shape))
        return params.replace(
            mean=knots,
            ksos_mean=knots,
            ksos_result_cost=jnp.asarray(cost, dtype=jnp.float32),
        )

    def plan_rollout(self, state: mjx.Data, params: GlobalMPPIParams) -> Trajectory:
        """Rollout of the current mean alone, shaped like `optimize`'s
        `rollouts_best`; used in place of MPPI when `ksos_refine` is False."""
        return self.rollout(state, params.tk, params.mean[None, ...])

    def update_params(
        self, params: GlobalMPPIParams, rollouts: Trajectory
    ) -> GlobalMPPIParams:
        """MPPI update: set the mean to a softmax-weighted average of the
        rollout knots, favoring lower-cost samples (temperature-scaled by
        `self.temperature`). This is the local-search step the base class's
        `optimize` runs after `solve_ksos` seeds the mean with a KSOS
        solution."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        mppi_mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
        
        return params.replace(mean=mppi_mean)
