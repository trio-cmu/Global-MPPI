import os
import sys
from pathlib import Path


def _preload_cukernelsos():
    """Load cuKernelSOS's extension (GPU ADMM) BEFORE jax, when CUKERNELSOS_PYTHON is set.

    cuKernelSOS is built against the system CUDA 13 toolkit; jax brings pip CUDA 12 libraries. Both cuSPARSE
    builds have the soname libcusparse.so.12, so when jax is loaded first the extension binds to jax's CUDA 12.9
    cuSPARSE (12.5) instead of its own (12.7) and runs it with the CUDA 13 runtime: this corrupted host memory
    every few hundred ADMM calls (segfaults at random places, e.g. scipy's cKDTree, in long closed-loop runs).
    Loaded first, the extension resolves its own cuSPARSE and jax still loads its copy by full path.
    """
    path = os.environ.get("CUKERNELSOS_PYTHON")
    if not path or "jax" in sys.modules:
        return
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        import cukernelsos._cukernelsos  # noqa: F401
    except ImportError:
        pass  # no GPU ADMM build; ADMM-cuda reports it when used


_preload_cukernelsos()

import jax  # noqa: E402  (after the cuKernelSOS preload)

# package root
ROOT = str(Path(__file__).parent.absolute())

# Set XLA flags for better performance
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true "

# Enable persistent compilation cache
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
