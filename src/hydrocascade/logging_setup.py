"""Uniform logging for CLI stages."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup(level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger("hydrocascade")
    if not _CONFIGURED:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
                                         datefmt="%H:%M:%S"))
        root.addHandler(h)
        root.propagate = False
        _CONFIGURED = True
    root.setLevel(level)
    return root


def get(name: str) -> logging.Logger:
    setup()
    return logging.getLogger(f"hydrocascade.{name}")
