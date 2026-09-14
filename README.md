# Global-MPPI

Global Sampling-Based Trajectory Optimization for Contact-Rich Manipulation
via KernelSOS.

Contact-rich manipulation costs are highly non-convex, so purely local
sampling-based MPC (MPPI, predictive sampling, ...) easily gets stuck in bad
local minima. GlobalMPPI adds a global search step: candidate control-spline
knots are evaluated, smoothed with a log-sum-exp estimate, and handed to a
[KernelSOS](https://github.com/Simple-Robotics/ksos-tools) solver (vendored
here as the `ksos-tools` submodule) to fit a new, better-informed mean before
the usual MPPI local refinement runs on top.

## Algorithms

All algorithms share the same `SamplingBasedController` interface
(`global_mppi/alg_base.py`) and control-spline knot representation
(`global_mppi/utils/spline.py`), so they can be swapped in for the same task
with the same simulation loop.

| Algorithm | Flag | Description |
| --- | --- | --- |
| Predictive Sampling | `ps` | Samples control tapes around the current mean and keeps the lowest-cost one ([Howell et al., 2022](https://arxiv.org/abs/2212.00541)). |
| MPPI | `mppi` | Model-predictive path integral control: a softmax-weighted average over sampled rollouts ([Vlahov et al., 2024](https://arxiv.org/abs/2409.07563)). |
| DIAL-MPC | `dial` | MPPI with noise annealed across both optimization iterations and the planning horizon ([Xue et al., 2024](https://arxiv.org/abs/2409.15610)). |
| GlobalMPPI | `globalmppi` | This work: a KernelSOS global-search step seeds the mean, followed by an MPPI local-search refinement. |

`CEM` and `Evosax`/CMA-ES controllers are also implemented in
`global_mppi/algs/` but are not wired into the example scripts.

## Tasks

- **PushT** (`global_mppi/tasks/pusht.py`) — push a T-shaped block to a target
  pose with a point pusher.
- **CubeRotation** (`global_mppi/tasks/cube.py`) — reorient a cube to a target
  orientation using a LEAP hand.

Both tasks define their dynamics via a MuJoCo model (`global_mppi/models/`)
and their running/terminal costs via the abstract `Task` interface in
`global_mppi/task_base.py`.

## Requirements

- Python **>= 3.12**
- An NVIDIA GPU with CUDA 12 (for `jax[cuda12]` / `mujoco-mjx`)

## Installation

The repository uses a git submodule (`ksos-tools`) for the KernelSOS solver, so
clone recursively (or initialize the submodule after cloning):

```bash
# Fresh clone
git clone --recurse-submodules <repo-url>
cd Global-MPPI

# Already cloned without --recurse-submodules?
git submodule update --init --recursive
```

Create an environment (Python 3.12) and install the package plus the submodule.
Using [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .            # main package + all Python dependencies
uv pip install -e ./ksos-tools # KernelSOS solver (git submodule, not on PyPI)
```

Or with plain `pip` inside a Python 3.12 environment:

```bash
pip install -e .
pip install -e ./ksos-tools
```

All Python dependencies (jax, mujoco, jaxlie, cvxpy, eigenpy, ipdb,
imageio-ffmpeg, warp, ...) are declared in `pyproject.toml` and installed
automatically. The `ksos-tools` submodule is the only piece that must be
installed separately, because it is a git submodule rather than a PyPI package.

## Project layout

```
global_mppi/
├── algs/          # Controllers: PS, MPPI, DIAL, GlobalMPPI, CEM, CMA-ES
├── tasks/         # Task definitions (dynamics + costs): PushT, CubeRotation
├── models/        # MuJoCo XML models for each task
├── simulation/    # Interactive/async simulation loops, W&B + video logging
├── utils/         # Control-spline interpolation utilities
├── alg_base.py    # Abstract SamplingBasedController base class
├── task_base.py   # Abstract Task interface
└── risk.py        # Domain-randomization risk strategies (average, worst-case, CVaR, ...)
examples/          # Runnable scripts: pusht.py, cube.py
tests/             # pytest test suite
data/logs/         # Example best-cost-history logs + plotting script
ksos-tools/        # KernelSOS solver (git submodule)
```

## Running the examples

The example scripts live in `examples/`. Each takes an algorithm as a
positional argument (`ps`, `mppi`, `dial`, `globalmppi`).

```bash
source .venv/bin/activate

# Push-T task with Predictive Sampling
python examples/pusht.py ps

# Cube rotation task with GlobalMPPI
python examples/cube.py globalmppi
```

`examples/cube.py` always opens an interactive MuJoCo viewer; the
`--headless`/`--wandb` flags below apply only to `examples/pusht.py`.

On a **headless machine** (remote server / dev desktop with no display), add
`--headless` so MuJoCo uses the offscreen GL backend and records a video:

```bash
python examples/pusht.py --headless ps
```

Video recording (and W&B video upload) needs an `ffmpeg` binary on `PATH`. The
`imageio-ffmpeg` dependency bundles one; expose it with:

```bash
ln -sf "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" ~/.local/bin/ffmpeg
export PATH="$HOME/.local/bin:$PATH"
```

### Weights & Biases logging (optional)

```bash
python examples/pusht.py --headless --wandb --wandb-entity <your-entity> ps
```

Pass `--wandb-entity` (or set `export WANDB_ENTITY=<your-entity>`) if your W&B
account has no default entity, otherwise `wandb.init` errors with
`entity not specified`. The first run prompts for your API key (from
https://wandb.ai/authorize) and caches it in `~/.netrc`.

## Running the tests

Install the `dev` extra (adds `pytest`) and run the suite from the repo root:

```bash
uv pip install -e ".[dev]"   # or: pip install -e ".[dev]"
pytest tests/
```

The tests instantiate real MuJoCo/MJX models and JIT-compile rollouts, so
expect the first run of each test file to take a while.

## Citation

If you use this codebase, please cite:

```bibtex
@article{wei2026global,
  title={Global Sampling-Based Trajectory Optimization for Contact-Rich Manipulation via KernelSOS},
  author={Wei, Zhongqi and Duembgen, Frederike},
  journal={arXiv preprint arXiv:2604.27175},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

This codebase builds on [hydrax](https://github.com/vincekurtz/hydrax), a
library for sampling-based MPC in MuJoCo MJX.
