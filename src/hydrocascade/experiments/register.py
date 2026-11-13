"""``experiments/register.csv`` — Doc 08 §1.

    *Every experiment is recorded with: id, description, model config, data
    split, seed, git commit, wall-clock time, and results file. **No number
    appears in the paper unless it has a register entry.***

The register is append-only. An experiment that was run and then superseded
stays in the file — deleting it would hide that the earlier answer existed, and
"we ran it twice and reported the better one" is exactly what a register is for
catching.
"""

from __future__ import annotations

import csv
from datetime import datetime

from .. import paths
from ..logging_setup import get

log = get("experiments.register")

REGISTER_PATH = paths.EXPERIMENTS / "register.csv"

COLUMNS = ("experiment_id", "timestamp", "description", "model_config",
           "data_split", "seed", "git_commit", "wall_clock_s",
           "log_likelihood", "aic", "branching_ratio", "results_file", "notes")


def _next_id() -> str:
    n = 0
    if REGISTER_PATH.is_file():
        with open(REGISTER_PATH, newline="", encoding="utf-8") as fh:
            n = sum(1 for _ in csv.reader(fh)) - 1
    return f"E-{max(n, 0) + 1:04d}"


def append(description: str, *, model_config: str, data_split: str,
           seed: int, wall_clock_s: float = 0.0,
           log_likelihood: float = float("nan"), aic: float = float("nan"),
           branching_ratio: float = float("nan"), results_file: str = "",
           notes: str = "") -> str:
    """Add one row and return its experiment id."""
    from ..cascade.fit import git_commit

    REGISTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not REGISTER_PATH.is_file()
    eid = _next_id()
    row = {
        "experiment_id": eid,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "description": description,
        "model_config": model_config,
        "data_split": data_split,
        "seed": seed,
        "git_commit": git_commit(),
        "wall_clock_s": round(float(wall_clock_s), 2),
        "log_likelihood": log_likelihood,
        "aic": aic,
        "branching_ratio": branching_ratio,
        "results_file": results_file,
        "notes": notes,
    }
    with open(REGISTER_PATH, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        if new:
            w.writeheader()
        w.writerow(row)
    log.info("register %s — %s", eid, description)
    return eid


def read():
    import pandas as pd
    if not REGISTER_PATH.is_file():
        return pd.DataFrame(columns=list(COLUMNS))
    return pd.read_csv(REGISTER_PATH)
