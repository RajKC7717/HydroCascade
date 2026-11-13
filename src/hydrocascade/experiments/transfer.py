"""Spatial transfer to a second region — Doc 08 §4.3.

    *Fit on Kerala. Apply, **without refitting**, to the Maharashtra Western
    Ghats: Malin (2014), Taliye (2021), Irshalwadi (2023). Report the
    degradation.*

    *"Our model fitted on Kerala also behaves sensibly in Maharashtra" is a far
    stronger sentence than "we added three more rows to the catalog."*

**Status: machinery built, region not built.** Running the second region means
running Phases 1-2 again for it — new DEM tiles, new basin, new network, new
slope units, roughly 1.5-2.5 GB of disk. It is item 2 on Doc 09 §8's own
de-scope ladder. What exists here instead is:

* :func:`apply_fitted` — evaluate a fitted parameter set on any assembled
  region, with an explicit refusal to refit;
* :func:`region_config` — produce the config for a second region;
* :func:`degradation` — the comparison table the report needs.

So the transfer test is one config file and one pipeline run away, not one
research project away. The quality-gate report records it as BLOCKED with that
distinction spelled out, rather than as done.

**The parameters must not be refitted.** That is the entire content of the test:
a model that has to be refitted per region has not transferred, it has merely
been fitted twice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import paths
from ..config import Node
from ..logging_setup import get

log = get("experiments.transfer")

TRANSFER_CONFIG = paths.CONFIG / "region_maharashtra.yaml"

#: Doc 08 §4.3's three events. Coordinates are deliberately absent: none has
#: been sourced, and inventing them would put unsourced rows into a validation
#: set, which is exactly what `data/reference/README.md` forbids.
TRANSFER_EVENTS = [
    {"name": "Malin", "year": 2014, "district": "Pune",
     "source": "NEEDS_SOURCING"},
    {"name": "Taliye", "year": 2021, "district": "Raigad",
     "source": "NEEDS_SOURCING"},
    {"name": "Irshalwadi", "year": 2023, "district": "Raigad",
     "source": "NEEDS_SOURCING"},
]


class RefitAttempted(RuntimeError):
    """Raised if transfer code tries to optimise. It must not."""


@dataclass
class TransferResult:
    region: str
    log_likelihood: float
    n_events: int
    per_event_ll: float
    branching: float
    status: str = "ok"
    note: str = ""


def apply_fitted(cfg: Node, params: dict, spec, *, region_name: str
                 ) -> TransferResult:
    """Evaluate fitted parameters on an assembled region. **Never optimises.**

    Returns log-likelihood *per event* as well as the total, because the two
    regions have different catalog sizes and a raw total would compare the
    number of events rather than the quality of the fit.
    """
    import torch

    from ..cascade import model

    data = model.assemble(cfg)
    obj = model.make_objective(data, spec)
    q = {k: torch.tensor(float(v), dtype=torch.float64) for k, v in params.items()}
    with torch.no_grad():
        ll = float(obj(q))

    n = max(data.n_events, 1)
    r = TransferResult(region=region_name, log_likelihood=ll, n_events=n,
                       per_event_ll=ll / n,
                       branching=model.branching(params, data, spec))
    log.info("transfer to %s: logL %.2f over %d events (%.4f per event)",
             region_name, ll, n, r.per_event_ll)
    return r


def degradation(home: TransferResult, away: TransferResult) -> dict:
    """How much worse the model does off its home region."""
    d = away.per_event_ll - home.per_event_ll
    return {
        "home": home.region, "away": away.region,
        "home_ll_per_event": home.per_event_ll,
        "away_ll_per_event": away.per_event_ll,
        "delta_per_event": d,
        "relative_pct": (100.0 * d / abs(home.per_event_ll)
                         if home.per_event_ll else float("nan")),
        "verdict": ("transfers — held-out fit per event is comparable"
                    if abs(d) < 0.1 * abs(home.per_event_ll)
                    else "degrades — the parameters are region-specific"),
    }


def write_region_config(cfg: Node, *, seed_lat: float = 19.15,
                        seed_lon: float = 73.65,
                        bbox=(72.90, 18.30, 74.10, 19.70)) -> "Path":
    """Emit ``config/region_maharashtra.yaml``, ready to run.

    The seed point is in the northern Western Ghats around the Malin / Taliye /
    Irshalwadi cluster; the pipeline derives the actual basin hydrologically
    from it, exactly as it did for the Chaliyar. **Nothing else changes** —
    Doc 08 §4.3's whole point is that scaling to a second region is a config
    edit, and this file is the proof of that claim rather than an assertion of
    it.
    """
    import yaml

    d = cfg.to_dict()
    d["region"] = dict(d["region"])
    d["region"].update(
        id="maharashtra_transfer",
        description=("Northern Western Ghats transfer region for the Doc 08 "
                     "§4.3 spatial transfer test. Derived hydrologically from "
                     "the seed point, exactly as the Chaliyar pilot is. "
                     "Coordinates for Malin, Taliye and Irshalwadi are NOT "
                     "sourced yet — see experiments/transfer.py."),
        seed_bbox=list(bbox),
        seed_point={"lat": seed_lat, "lon": seed_lon,
                    "name": "Northern Western Ghats transfer seed"})
    d["landmarks"] = {"transfer_seed": {
        "lat": seed_lat, "lon": seed_lon,
        "note": "placeholder; replace with sourced event coordinates"}}
    d.pop("wayanad_event", None)

    TRANSFER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Doc 08 §4.3 spatial transfer region — generated by\n"
        "# hydrocascade.experiments.transfer.write_region_config.\n"
        "#\n"
        "# To run the transfer test:\n"
        "#   hydrocascade aoi dem flow network units projection traveltime \\\n"
        "#       rainfall catalog --config config/region_maharashtra.yaml\n"
        "#   hydrocascade transfer --config config/region_maharashtra.yaml\n"
        "#\n"
        "# The fitted parameters are NOT refitted. That is the test.\n"
        "# Needs roughly 1.5-2.5 GB of free disk for the second DEM.\n\n")
    TRANSFER_CONFIG.write_text(header + yaml.safe_dump(d, sort_keys=False),
                               encoding="utf-8")
    log.info("wrote %s", TRANSFER_CONFIG.relative_to(paths.ROOT))
    return TRANSFER_CONFIG


def status() -> dict:
    """Is the transfer region actually built?"""
    built = TRANSFER_CONFIG.is_file()
    sourced = [e for e in TRANSFER_EVENTS if e["source"] != "NEEDS_SOURCING"]
    return {
        "config_written": built,
        "config_path": str(TRANSFER_CONFIG.relative_to(paths.ROOT)),
        "region_built": False,
        "events_sourced": len(sourced),
        "events_total": len(TRANSFER_EVENTS),
        "blocked_because": (
            "the second region's DEM, network and slope units are not built "
            "(roughly 1.5-2.5 GB of disk), and none of the three Maharashtra "
            "events has sourced coordinates. Both are one afternoon of work; "
            "neither is a code change."),
    }
