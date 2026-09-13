# Global-MPPI

Global Sampling-Based Trajectory Optimization for Contact-Rich Manipulation
via KernelSOS.

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

## Running the examples

The example scripts live in `examples/`. Each takes an algorithm as a
positional argument (`ps`, `mppi`, `cem`, `cmaes`, `dial`,
`globalmppi`, ...).

```bash
source .venv/bin/activate

# Push-T task with Predictive Sampling
python examples/pusht.py ps
```

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

## Citation

If you use this codebase, please cite:

```bibtex
@article{wei2026global,
  title={Global Sampling-Based Trajectory Optimization for Contact-Rich Manipulation via {K}ernel{SOS}},
  author={Wei, Zhongqi and D{\"u}mbgen, Frederike},
  journal={arXiv preprint arXiv:2604.27175},
  year={2026}
}
```

## Acknowledgements

This codebase builds on [hydrax](https://github.com/vincekurtz/hydrax), a
library for sampling-based MPC in MuJoCo MJX.
