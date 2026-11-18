"""Planner: build the controller of an algorithm from the config and run one planning step.

ps / mppi / dial: params = jit(optimize)(state, params).
globalmppi: the KSOS global step (as in simulation/deterministic.py) before the MPPI refinement. It needs the
ksos-tools package (not installed on every machine); NOT tested on this machine.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from global_mppi.algs import DIAL, MPPI, GlobalMPPI, PredictiveSampling
from global_mppi.closed_loop.config import algo_kwargs

ALGOS = {"ps": PredictiveSampling, "mppi": MPPI, "dial": DIAL, "globalmppi": GlobalMPPI}


class Planner:
    def __init__(self, scene, algo: str, cfg: dict):
        if algo not in ALGOS:
            raise ValueError(f"unknown algorithm {algo!r}; choose from {list(ALGOS)}")
        self.scene, self.algo = scene, algo
        self.kwargs = algo_kwargs(cfg, algo)
        # noise is configured as a multiple of the control range (vmax), so changing vmax needs no re-tuning
        if "noise_level" in self.kwargs:                  # an explicit absolute noise_level wins over the multiple
            self.kwargs.pop("noise_level_vmax", None)
        if "noise_level_vmax" in self.kwargs:
            self.kwargs["noise_level"] = float(self.kwargs.pop("noise_level_vmax")) * scene.vmax
        self.ctrl = ALGOS[algo](scene.task, **self.kwargs)
        self.jit_opt = jax.jit(self.ctrl.optimize)
        self.jit_interp = jax.jit(self.ctrl.interp_func)
        if algo == "globalmppi":
            self.jit_ksos_rollout = jax.jit(self.ctrl.ksos_rollout)
        self.mjx_data = mjx.put_data(scene.task.mj_model, scene.new_data())

    def init_params(self, seed: int):
        return self.ctrl.init_params(seed=seed)

    def sync(self, data):
        return self.mjx_data.replace(
            qpos=jnp.array(data.qpos), qvel=jnp.array(data.qvel), mocap_pos=jnp.array(data.mocap_pos),
            mocap_quat=jnp.array(data.mocap_quat), time=data.time)

    def warmup(self, data) -> float:
        """Compile (twice, like run_interactive). Returns the wall time."""
        t0 = time.time()
        p = self.init_params(0)
        s = self.sync(data)
        for _ in range(2):
            p = self.plan(s, p)
        jax.block_until_ready(p.mean)
        tq = jnp.arange(0, 2) * 0.001
        self.jit_interp(tq, p.tk, p.mean[None, ...])
        return time.time() - t0

    def plan(self, state, params):
        if self.algo == "globalmppi":
            params = self._ksos(state, params)
        params, _, _ = self.jit_opt(state, params)
        return params

    def _ksos(self, state, params):
        from global_mppi.simulation.deterministic import neg_lml  # GP length-scale auto-calibration, as in run_interactive
        from scipy.optimize import minimize

        c = self.ctrl
        for it in range(c.ksos_num_restart):
            params = params.replace(iter=it)
            params = jax.block_until_ready(self.jit_ksos_rollout(state, params))
            n = params.lse_cost.shape[0]
            X = np.asarray(params.knots.reshape(n, -1))
            y = np.asarray(params.lse_cost)
            Xn = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
            yn = y - y.mean()
            grid = np.log(np.linspace(0.01, 30.0, 10))
            ls0 = grid[int(np.argmin([neg_lml(g, Xn, yn) for g in grid]))]
            res = minimize(neg_lml, x0=np.array([ls0]), args=(Xn, yn), method="L-BFGS-B",
                           bounds=[(np.log(1e-2), np.log(30.0))])
            c.ksos_sigma = float(np.exp(res.x[0]))
            params = c.solve_ksos(params)
            params = params.replace(mean=params.ksos_mean)
            params, _, _ = self.jit_opt(state, params)
        return params

    def controls(self, params, t0: float, n_sub: int, dt: float) -> np.ndarray:
        tq = jnp.arange(0, n_sub) * dt + t0
        return np.asarray(self.jit_interp(tq, params.tk, params.mean[None, ...]))[0]
