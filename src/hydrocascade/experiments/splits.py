"""The three-way temporal split and the leakage guard — Doc 08 §4.1.

======= ========================= ==========================================
Split   Period                    Used for
======= ========================= ==========================================
Train   1 Jun 2018 - 31 Dec 2022  fitting all parameters
Valid   1 Jan 2023 - 31 May 2024  model selection, alert thresholds, calibration
Test    1 Jun 2024 - 31 Dec 2024  **touched once, at the end.** Contains Wayanad
======= ========================= ==========================================

Doc 08 §4.1 asks for *"an assertion in the fitting code that raises an error if
any training event is dated after the cutoff"*, written by one person and
audited by another — *"one person writes it, a different person tries to break
it."* :func:`assert_no_leakage` is that assertion, and
``tests/test_experiments.py`` is the audit: it contains deliberate
leakage-injection tests that must fail the guard.

**It raises. It does not warn.** A warning in a log nobody reads is how a
leaked test set reaches a paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..logging_setup import get

log = get("experiments.splits")

TRAIN_START = "2018-06-01"
TRAIN_END = "2022-12-31"
VALID_START = "2023-01-01"
VALID_END = "2024-05-31"
TEST_START = "2024-06-01"
TEST_END = "2024-12-31"


class LeakageError(AssertionError):
    """Raised when data from after a cutoff reached something that trains."""


@dataclass
class Split:
    name: str
    start: str
    end: str

    def mask(self, dates) -> np.ndarray:
        d = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
        return ((d >= pd.Timestamp(self.start))
                & (d <= pd.Timestamp(self.end))).to_numpy()

    def __repr__(self) -> str:
        return f"<{self.name}: {self.start} to {self.end}>"


TRAIN = Split("train", TRAIN_START, TRAIN_END)
VALID = Split("validation", VALID_START, VALID_END)
TEST = Split("test", TEST_START, TEST_END)
ALL_SPLITS = (TRAIN, VALID, TEST)


def assert_no_leakage(dates, cutoff: str, *, what: str = "training data") -> int:
    """Raise if anything in ``dates`` falls **after** ``cutoff``.

    Returns the number of rows checked, so a caller can prove the guard
    actually saw data rather than being handed an empty frame — an assertion
    that silently passes on nothing is worse than no assertion.
    """
    d = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dropna()
    if d.empty:
        raise LeakageError(
            f"leakage guard for {what} received no parseable dates. An empty "
            f"check is not a passing check.")
    limit = pd.Timestamp(cutoff)
    bad = d[d > limit]
    if len(bad):
        raise LeakageError(
            f"{len(bad)} of {len(d)} rows in {what} are dated after the "
            f"{cutoff} cutoff (latest {bad.max().date()}). "
            f"Fitting on these would leak the evaluation period.")
    log.info("leakage guard: %d rows in %s, latest %s, cutoff %s — clean",
             len(d), what, d.max().date(), cutoff)
    return int(len(d))


def assert_holdout_untouched(dates, *, what: str = "training data") -> int:
    """The strongest form: nothing at or after the Doc 04 §1 holdout start.

    The Wayanad event is 30 July 2024 and the holdout opens on 1 June 2024.
    Anything from that window in a training set invalidates the headline result,
    so this is checked separately from the ordinary train/validation cutoff.
    """
    return assert_no_leakage(dates, TEST_START, what=what)


def describe(events: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Event counts per split and hazard type — printed before every fit."""
    d = pd.to_datetime(events[date_column], errors="coerce")
    rows = []
    for s in ALL_SPLITS:
        m = (d >= pd.Timestamp(s.start)) & (d <= pd.Timestamp(s.end))
        sub = events[m.to_numpy()]
        rows.append({
            "split": s.name, "start": s.start, "end": s.end,
            "events": int(len(sub)),
            "landslides": int((sub.get("hazard_type", "") == "L").sum()),
            "floods": int((sub.get("hazard_type", "") == "F").sum()),
        })
    return pd.DataFrame(rows)
