"""Configuration loading.

Doc 11 §5 rule 3 generalised: no threshold, date, CRS or path is hard-coded in
``src/``. Everything comes from ``config/default.yaml``, so an experiment is
reproducible from a config file alone.

Usage::

    from hydrocascade.config import load_config
    cfg = load_config()
    cfg.crs.analysis                 # attribute access
    cfg["floods"]["decluster_days"]  # or mapping access
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from . import paths


class Node(Mapping):
    """Read-only nested mapping with attribute access and helpful errors."""

    __slots__ = ("_d", "_trail")

    def __init__(self, d: dict, trail: str = "config"):
        self._d = d
        self._trail = trail

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(str(exc)) from None

    def __getitem__(self, key: str) -> Any:
        if key not in self._d:
            known = ", ".join(sorted(self._d)) or "<empty>"
            raise KeyError(f"{self._trail}.{key} is not set. Known keys: {known}")
        val = self._d[key]
        if isinstance(val, dict):
            return Node(val, f"{self._trail}.{key}")
        return val

    def __iter__(self):
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self._d else default

    def to_dict(self) -> dict:
        import copy
        return copy.deepcopy(self._d)

    def __repr__(self) -> str:
        return f"<{self._trail}: {', '.join(sorted(self._d))}>"


def load_config(path: str | Path | None = None) -> Node:
    """Load the project config. Defaults to ``config/default.yaml``."""
    p = Path(path) if path else paths.CONFIG / "default.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping at top level: {p}")
    return Node(data)


def seed_everything(cfg: Node) -> int:
    """Set every RNG we use to the configured seed (NFR-08: determinism)."""
    import random

    import numpy as np

    seed = int(cfg.project.seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def bbox(cfg: Node) -> tuple[float, float, float, float]:
    """Seed bounding box as ``(minlon, minlat, maxlon, maxlat)`` in EPSG:4326."""
    b = list(cfg.region.seed_bbox)
    if len(b) != 4:
        raise ValueError("region.seed_bbox must be [minlon, minlat, maxlon, maxlat]")
    minlon, minlat, maxlon, maxlat = (float(x) for x in b)
    if not (minlon < maxlon and minlat < maxlat):
        raise ValueError(f"region.seed_bbox is not ordered correctly: {b}")
    return minlon, minlat, maxlon, maxlat
