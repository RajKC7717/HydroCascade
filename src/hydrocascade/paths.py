"""Canonical project paths.

Layout follows Doc 10 §8 exactly::

    data/
      raw/<source_code>/<YYYY-MM-DD_downloaded>/   # never edited, ever
      interim/                                     # cleaned, still recognisable
      processed/                                   # model-ready
      catalog/                                     # events.csv, cascade_pairs.csv
      reference/                                   # documented events used to score, never to fit

Nothing in ``src/`` may construct a path by string concatenation. Ask here.
"""

from __future__ import annotations

from pathlib import Path

# src/hydrocascade/paths.py -> src/hydrocascade -> src -> repo root
ROOT = Path(__file__).resolve().parents[2]

CONFIG = ROOT / "config"
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
CATALOG = DATA / "catalog"
REFERENCE = DATA / "reference"     # independently documented events, for corroboration
SPATIAL = PROCESSED / "spatial"
ROOTCAUSE = PROCESSED / "rootcause"

FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
EXPERIMENTS = ROOT / "experiments"
DOCS = ROOT / "docs"

_ALL_DIRS = [
    DATA, RAW, INTERIM, PROCESSED, CATALOG, REFERENCE, SPATIAL, ROOTCAUSE,
    FIGURES, REPORTS, MODELS, EXPERIMENTS,
]


def ensure_dirs() -> None:
    """Create the standard tree. Safe to call repeatedly."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def raw_dir(source_code: str, downloaded: str) -> Path:
    """``data/raw/<SOURCE>/<YYYY-MM-DD_downloaded>/`` — one folder per download.

    The folder is where ``PROVENANCE.md`` lives; see :mod:`hydrocascade.provenance`.
    """
    d = RAW / source_code / downloaded
    d.mkdir(parents=True, exist_ok=True)
    return d


def latest_raw(source_code: str) -> Path | None:
    """Most recent download folder for a source, or None if never downloaded."""
    base = RAW / source_code
    if not base.is_dir():
        return None
    subs = sorted((p for p in base.iterdir() if p.is_dir()), reverse=True)
    return subs[0] if subs else None
