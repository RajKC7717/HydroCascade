"""HydroCascade command line.

Every pipeline stage is a subcommand, every stage is idempotent, and every
stage reads its knobs from ``config/default.yaml``. Running the stages in
listed order reproduces Phases 1-2 from an empty ``data/`` directory::

    hydrocascade aoi          # study boundary from HydroBASINS
    hydrocascade dem          # Copernicus GLO-30, mosaicked and reprojected
    hydrocascade flow         # sink fill, D8, flow accumulation
    hydrocascade network      # channel network, 500 m reaches, directed graph
    hydrocascade units        # slope units (the hillslope domain H)
    hydrocascade projection   # the operator pi: H -> N
    hydrocascade traveltime   # downstream reachability + reference travel time
    hydrocascade rainfall     # CHIRPS download and per-unit daily windows
    hydrocascade catalog      # the event catalog (landslides + floods)
    hydrocascade covariates   # static terrain covariates
    hydrocascade rootcause    # Phase 3: drivers, susceptibility, baselines
    hydrocascade cascade      # Phase 4: the cross-hazard model, S1-S5
    hydrocascade experiments  # Phase 5: ablations, backtest, money figure
    hydrocascade realtime     # Phase 6: alert thresholds and architecture
    hydrocascade figures      # every figure named in Docs 04 and 05
    hydrocascade report       # the named prose deliverables
    hydrocascade check        # quality gates
    hydrocascade app          # launch the Explorer
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from . import paths
from .config import load_config, seed_everything
from .logging_setup import setup


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="path to a config YAML (default: config/default.yaml)")
    p.add_argument("--force", action="store_true", help="re-download / recompute even if cached")
    p.add_argument("-v", "--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hydrocascade", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    stages = {
        "aoi": "derive the study boundary from HydroBASINS (Doc 04 §1)",
        "dem": "fetch, mosaic and reproject the Copernicus DEM (Doc 05 §2.2)",
        "flow": "fill sinks, compute D8 direction and flow accumulation",
        "network": "build the channel network N and its directed graph (Doc 05 §3)",
        "units": "build the slope units, the hillslope domain H (Doc 05 §2)",
        "projection": "compute the projection operator pi: H -> N (Doc 05 §4)",
        "traveltime": "downstream reachability, reference travel time, area ratios (Doc 05 §5)",
        "rainfall": "fetch CHIRPS and build per-unit rainfall windows + API (Doc 05 §7)",
        "catalog": "build data/catalog/events.csv (Doc 04)",
        "covariates": "static terrain covariates per unit and reach (Doc 05 §6)",
        "rootcause": "Phase 3: drivers, SHAP, calibrated susceptibility, baselines B1-B3 (Doc 06)",
        "cascade": "Phase 4: fit the cross-hazard cascade model, stages S1-S5 (Doc 07)",
        "experiments": "Phase 5: ladder, ablations, LOCO, Wayanad backtest, sensitivity (Doc 08)",
        "realtime": "Phase 6: alert thresholds, the trade-off curve, the architecture report (Doc 09)",
        "figures": "regenerate every figure named in Docs 04 and 05",
        "report": "write the Doc 04 §8 prose deliverables: the flood definition, its sensitivity, the blind spot and the corroboration status",
        "check": "run the Phase 1 and Phase 2 quality gates",
        "app": "launch the Streamlit Explorer",
    }
    for name, help_ in stages.items():
        p = sub.add_parser(name, help=help_)
        _add_common(p)

    sub.choices["catalog"].add_argument(
        "--skip-floods", action="store_true",
        help="landslides only; skip the discharge-based flood detection")
    sub.choices["check"].add_argument(
        "--phase", choices=["1", "2", "3", "4", "5", "6", "all"], default="all")
    sub.choices["report"].add_argument(
        "--which", choices=["flood-definition", "all"], default="all")
    sub.choices["cascade"].add_argument(
        "--verify", action="store_true",
        help="only run the S1/S2 parameter-recovery gate, no real-data fit")
    sub.choices["cascade"].add_argument(
        "--stages", default="S3,S4,S5",
        help="comma-separated stages to fit (default S3,S4,S5)")
    sub.choices["cascade"].add_argument("--restarts", type=int, default=8)
    sub.choices["cascade"].add_argument("--steps", type=int, default=400)
    sub.choices["experiments"].add_argument(
        "--quick", action="store_true",
        help="fewer restarts and steps: proves the pipeline runs, NOT a result")
    sub.choices["experiments"].add_argument("--restarts", type=int, default=4)
    sub.choices["experiments"].add_argument("--steps", type=int, default=250)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup(logging.DEBUG if args.verbose else logging.INFO)
    log = logging.getLogger("hydrocascade.cli")

    paths.ensure_dirs()
    cfg = load_config(args.config)
    seed_everything(cfg)

    t0 = time.perf_counter()
    log.info("stage '%s'  region=%s  crs=%s", args.cmd, cfg.region.id, cfg.crs.analysis)

    try:
        _dispatch(args, cfg)
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        log.error("stage '%s' failed: %s", args.cmd, exc)
        if args.verbose:
            raise
        return 1

    log.info("stage '%s' done in %.1f s", args.cmd, time.perf_counter() - t0)
    return 0


def _dispatch(args, cfg) -> None:
    cmd = args.cmd
    force = args.force

    if cmd == "aoi":
        from .ingest import hydrosheds
        hydrosheds.build_aoi(cfg, force=force)

    elif cmd == "dem":
        from .ingest import dem
        dem.prepare(cfg, force=force)

    elif cmd == "flow":
        from .spatial import terrain
        terrain.build(cfg, force=force)

    elif cmd == "network":
        from .spatial import network
        network.build(cfg, force=force)

    elif cmd == "units":
        from .spatial import slope_units
        slope_units.build(cfg, force=force)

    elif cmd == "projection":
        from .spatial import projection
        projection.build(cfg, force=force)

    elif cmd == "traveltime":
        from .spatial import travel_time
        travel_time.build(cfg, force=force)

    elif cmd == "rainfall":
        from .features import rainfall_windows
        rainfall_windows.build(cfg, force=force)

    elif cmd == "catalog":
        from .catalog import build as catalog_build
        catalog_build.run(cfg, skip_floods=args.skip_floods, force=force)

    elif cmd == "covariates":
        from .features import terrain_covariates
        terrain_covariates.build(cfg, force=force)

    elif cmd == "rootcause":
        from .rootcause import run as rootcause_run
        rootcause_run.run(cfg, force=force)

    elif cmd == "cascade":
        from .cascade import stages, verify
        verify.run(seed=int(cfg.project.seed))
        if not args.verify:
            stages.run(cfg, stages=tuple(args.stages.split(",")),
                       n_restarts=args.restarts, steps=args.steps)

    elif cmd == "experiments":
        from .experiments import run as experiments_run
        experiments_run.run(cfg, n_restarts=args.restarts, steps=args.steps,
                            quick=args.quick)

    elif cmd == "realtime":
        from .realtime import run as realtime_run
        realtime_run.run(cfg)

    elif cmd == "figures":
        from .viz import figures
        figures.render_all(cfg)

    elif cmd == "report":
        from .qa import flood_definition
        if args.which in ("flood-definition", "all"):
            flood_definition.build(cfg)

    elif cmd == "check":
        from .qa import gates
        ok = gates.run(cfg, phase=args.phase)
        if not ok:
            raise SystemExit("quality gates failed")

    elif cmd == "app":
        import subprocess
        app = paths.ROOT / "src" / "hydrocascade" / "dashboard" / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], check=False)

    else:  # pragma: no cover — argparse enforces the choices
        raise ValueError(f"unknown stage {cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
