"""Run a Global-MPPI example with the JAX compilation cache in ~/.cache/global-mppi/jax_cache instead of the
package's hard-coded /tmp/jax_cache (owned by another user on this machine). Nothing else changes.

    python run_with_own_cache.py examples/pusht.py ARGS...
"""
import os
import runpy
import sys

import jax

CACHE = os.path.expanduser("~/.cache/global-mppi/jax_cache")
os.makedirs(CACHE, exist_ok=True)
_update = jax.config.update


def update(name, value):
    if name == "jax_compilation_cache_dir":
        value = CACHE
    return _update(name, value)


jax.config.update = update
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
