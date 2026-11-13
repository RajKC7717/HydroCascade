"""The Explorer actually renders.

A Streamlit page fails at *render* time, not at import time, so "it imports" is
not evidence that it works. ``AppTest`` executes the script the way the server
does and collects anything that would have shown the user a red traceback.

The test skips on a clean clone: it needs the built artefacts, and a missing
``data/`` directory is not a code defect.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hydrocascade import paths

APP = paths.ROOT / "src" / "hydrocascade" / "dashboard" / "app.py"

pytest.importorskip("streamlit")

_REQUIRED = (
    paths.CATALOG / "events.csv",
    paths.SPATIAL / "channel_network.gpkg",
    paths.SPATIAL / "slope_units.gpkg",
    paths.SPATIAL / "travel_time.npz",
)
_MISSING = [p.name for p in _REQUIRED if not p.is_file()]

pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=f"pipeline artefacts not built: {', '.join(_MISSING)}")


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    return at


def test_the_page_renders_without_an_exception(app):
    assert not app.exception, [e.value for e in app.exception]


def test_every_tab_is_present(app):
    """Nine tabs: cascade path, catalog, spatial framework, rainfall, drivers,
    the model, backtest and alerts, gates and reports, how to reproduce.

    A missing one means a section raised before its tab was created — which
    Streamlit renders as a silently shorter tab strip rather than an error.
    """
    assert len(app.tabs) == 9


def test_the_headline_metrics_are_populated(app):
    """An em dash in a headline metric means a loader silently returned None."""
    headline = {m.label: m.value for m in app.metric}
    for label in ("events", "slope units", "river reaches", "basin area"):
        assert label in headline, f"{label} metric missing"
        assert headline[label] != "—", f"{label} did not load"


def test_the_cascade_path_reports_an_injection_reach(app):
    """Doc 05 §9 gate 2 in the UI: the Mundakkai crown resolves to a reach."""
    headline = {m.label: m.value for m in app.metric}
    assert headline.get("injection reach", "").startswith("R-")


def test_synthetic_data_is_announced_in_red(app):
    """Doc 10 §7: a stand-in must never be able to pass for the real thing.

    If the sidebar banner ever stops rendering, someone will screenshot a
    synthetic catalog into a slide.
    """
    from hydrocascade.config import load_config
    from hydrocascade.ingest import adapters

    if not adapters.synthetic_sources(load_config()):
        pytest.skip("no synthetic sources configured")
    assert any("Synthetic stand-ins active" in e.value for e in app.error)
