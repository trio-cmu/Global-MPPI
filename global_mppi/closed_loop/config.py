"""Config loading: yaml file + "section.key=value" overrides (value parsed as yaml)."""
import copy
import os

import yaml

from global_mppi import ROOT

REPO_ROOT = os.path.dirname(ROOT)


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, val = item.partition("=")
        if not _:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def resolve_path(p: str) -> str:
    """Relative paths are relative to the repo root."""
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def algo_kwargs(cfg: dict, algo: str) -> dict:
    return copy.deepcopy(cfg["algos"][algo])
