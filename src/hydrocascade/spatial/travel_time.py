"""Downstream reachability, reference travel time and dilution (Doc 05 §5).

This is the computational backbone of the Phase 4 cascade model. Doc 02 §10.2
optimisation 4 calls the precomputed travel-time table *"the single biggest
saving and why Phase 2 builds it early"*.

The key design decision is Doc 05 §5.1. Channel celerity is
``v(s) = v0 * S(s)^0.3`` with ``v0`` a *fitted* parameter, so a table of travel
times would have to be rebuilt at every optimiser step. Instead we store the
path integral, which does not depend on ``v0`` at all::

    T_ref(p -> s) = sum over reaches on the path of ( length / S^0.3 )    [metres]
    tau(p -> s)   = T_ref(p -> s) / v0                                    [seconds]

So the optimiser divides, and the table is built exactly once.

Because each reach has exactly one downstream neighbour, ``Down(p)`` is a chain
rather than a tree, and the whole structure is small enough to build in seconds
rather than the four hours Doc 05 §5.4 budgets for.

Output: ``travel_time.npz``, a CSR structure keyed by source reach.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import paths
from ..config import Node
from ..logging_setup import get
from .network import Network

log = get("spatial.travel_time")

TT_PATH = paths.SPATIAL / "travel_time.npz"


def build(cfg: Node, *, force: bool = False) -> Path:
    if TT_PATH.is_file() and not force:
        log.info("cached  %s", TT_PATH.name)
        return TT_PATH

    net = Network()
    tt_cfg = cfg.spatial.travel_time
    exponent = float(tt_cfg.celerity_exponent)
    min_slope = float(tt_cfg.min_slope)
    v0_ref = float(tt_cfg.v0_reference_ms)
    v0_min = float(tt_cfg.v0_min_ms)
    max_hours = float(tt_cfg.max_reference_hours)

    n = len(net)
    length = net.gdf["length_m"].to_numpy(dtype=np.float64)
    slope = np.maximum(net.gdf["slope"].to_numpy(dtype=np.float64), min_slope)
    area = net.gdf["area_dn_km2"].to_numpy(dtype=np.float64)
    downstream = net.downstream

    # Per-reach contribution to the path integral, in metres.
    t_ref_reach = length / np.power(slope, exponent)

    # Truncate at 48 h assuming the SLOWEST plausible v0, so the cached table
    # remains valid for every v0 the Phase 4 fit might choose.
    t_ref_max = max_hours * 3600.0 * v0_min
    log.info("celerity v = v0 * S^%.2f; reference v0 = %.1f", exponent, v0_ref)
    v_ref = v0_ref * np.power(slope, exponent)
    v_med = float(np.median(v_ref))
    log.info("implied reach celerity at reference v0: median %.2f m/s, "
             "5th-95th %.2f-%.2f m/s", v_med,
             float(np.percentile(v_ref, 5)), float(np.percentile(v_ref, 95)))

    # Doc 05 §5.1: the reference v0 must be "chosen so speeds fall in the
    # physical range 1-5 m/s". v0 is a coefficient, not a speed, so it is easy
    # to set it to a plausible-looking number and get an implausible celerity.
    lo, hi = (float(x) for x in tt_cfg.celerity_physical_range_ms)
    if not (lo <= v_med <= hi):
        suggested = v0_ref * (0.5 * (lo + hi)) / max(v_med, 1e-9)
        log.warning("median celerity %.2f m/s is outside the physical range "
                    "%.1f-%.1f m/s. Set spatial.travel_time.v0_reference_ms to "
                    "about %.1f to centre it.", v_med, lo, hi, suggested)
    log.info("truncation: %.0f h at v0=%.1f m/s  ->  T_ref <= %.0f km",
             max_hours, v0_min, t_ref_max / 1000)

    ds = downstream.tolist()
    tref = t_ref_reach.tolist()

    indptr = np.zeros(n + 1, dtype=np.int64)
    targets: list[int] = []
    t_values: list[float] = []
    ratios: list[float] = []

    for p in range(n):
        acc = 0.0
        s = ds[p]
        # The source reach itself is at T_ref = 0; excitation there is handled
        # by the unshifted part of the kernel, so the chain starts downstream.
        while s >= 0:
            acc += tref[s]
            if acc > t_ref_max:
                break
            targets.append(s)
            t_values.append(acc)
            ratios.append(area[p] / max(area[s], 1e-9))
            s = ds[s]
        indptr[p + 1] = len(targets)

    indices = np.asarray(targets, dtype=np.int32)
    t_ref = np.asarray(t_values, dtype=np.float32)
    area_ratio = np.asarray(ratios, dtype=np.float32)

    counts = np.diff(indptr)
    log.info("reachability: %d pairs, %.1f downstream reaches per source "
             "(median %d, max %d)", indices.size, counts.mean(),
             int(np.median(counts)), int(counts.max()))
    hours_ref = t_ref / v0_ref / 3600.0
    log.info("travel time at reference v0: median %.1f h, max %.1f h",
             float(np.median(hours_ref)), float(hours_ref.max()))
    log.info("dilution A(p)/A(s): median %.4f, min %.6f",
             float(np.median(area_ratio)), float(area_ratio.min()))

    np.savez_compressed(
        TT_PATH,
        indptr=indptr, indices=indices, t_ref_m=t_ref, area_ratio=area_ratio,
        celerity_exponent=np.float64(exponent), v0_reference_ms=np.float64(v0_ref),
        v0_min_ms=np.float64(v0_min), max_reference_hours=np.float64(max_hours),
        reach_slope=slope.astype(np.float32), reach_length_m=length.astype(np.float32),
        reach_area_km2=area.astype(np.float32),
    )
    log.info("wrote   %s (%.1f MB)", TT_PATH.name, TT_PATH.stat().st_size / 1e6)
    return TT_PATH


class TravelTime:
    """CSR downstream reachability with reference travel time and dilution."""

    def __init__(self):
        if not TT_PATH.is_file():
            raise FileNotFoundError(f"{TT_PATH} missing — run `hydrocascade traveltime`")
        z = np.load(TT_PATH, allow_pickle=False)
        self.indptr = z["indptr"]
        self.indices = z["indices"].astype(np.int64)
        self.t_ref_m = z["t_ref_m"].astype(np.float64)
        self.area_ratio = z["area_ratio"].astype(np.float64)
        self.v0_reference_ms = float(z["v0_reference_ms"])
        self.celerity_exponent = float(z["celerity_exponent"])
        self.reach_slope = z["reach_slope"].astype(np.float64)
        self.reach_length_m = z["reach_length_m"].astype(np.float64)
        self.reach_area_km2 = z["reach_area_km2"].astype(np.float64)

    def downstream_of(self, reach_id: int):
        """``(reach_ids, T_ref_metres, area_ratio)`` for one source reach."""
        a, b = self.indptr[reach_id], self.indptr[reach_id + 1]
        return self.indices[a:b], self.t_ref_m[a:b], self.area_ratio[a:b]

    def hours(self, t_ref_m: np.ndarray, v0_ms: float | None = None) -> np.ndarray:
        """Convert the stored path integral into hours for a given ``v0``."""
        v0 = self.v0_reference_ms if v0_ms is None else v0_ms
        return np.asarray(t_ref_m) / v0 / 3600.0


def spot_check(cfg: Node, n_show: int = 25) -> str:
    """The Doc 05 §5.4 check, as printable text.

    *"Pick the Mundakkai injection point and list its downstream reaches with
    their reference travel times. Verify that Chooralmala appears within a few
    kilometres and the Chaliyar at Nilambur appears further along, in the right
    order. Print this list and put it in the Phase 2 report — it is a
    convincing, concrete demonstration that the spatial framework works."*
    """
    import geopandas as gpd
    from shapely.geometry import Point

    from .terrain import Routing
    from .projection import cell_level

    rt = Routing()
    net = Network()
    tt = TravelTime()
    proj = cell_level(rt, net)

    lm = cfg.landmarks
    crown_cell = rt.index_of_lonlat(float(lm.mundakkai_crown["lon"]),
                                    float(lm.mundakkai_crown["lat"]))
    inj_reach = int(proj["injection_reach"][crown_cell])
    d_slope_km = float(proj["d_slope_m"][crown_cell]) / 1000.0

    ids, t_ref, ratio = tt.downstream_of(inj_reach)
    hours = tt.hours(t_ref)

    # Where do the downstream landmarks sit on that chain?
    pts = gpd.GeoDataFrame(
        {"name": ["chooralmala_bridge", "nilambur"]},
        geometry=[Point(float(lm[k]["lon"]), float(lm[k]["lat"]))
                  for k in ("chooralmala_bridge", "nilambur")],
        crs="EPSG:4326").to_crs(net.gdf.crs)

    marks: dict[int, str] = {}
    snapped: dict[str, tuple[int, float]] = {}
    for _, row in pts.iterrows():
        rid, dist = net.snap(row.geometry)
        marks[rid] = row["name"]
        snapped[row["name"]] = (rid, dist)

    out = [
        "Mundakkai crown -> injection point",
        f"  crown cell           : {crown_cell}",
        f"  injection reach      : R-{inj_reach:05d}",
        f"  d_slope (crown->river): {d_slope_km:.2f} km",
        f"  downstream reaches within the truncation radius: {len(ids)}",
        "",
        "  landmarks snapped to the main stem (largest catchment within 750 m):",
        *[f"    {name:<20} R-{rid:05d} at {dist:.0f} m, "
          f"catchment {net.gdf.iloc[rid]['area_dn_km2']:.0f} km2, "
          f"Strahler {int(net.gdf.iloc[rid]['strahler'])}"
          for name, (rid, dist) in snapped.items()],
        "",
        f"{'reach':>10}  {'T_ref km':>9}  {'hours@v0':>9}  {'A(p)/A(s)':>10}  "
        f"{'area km2':>9}  note",
    ]
    show = list(range(min(n_show, len(ids))))
    for k in show:
        rid = int(ids[k])
        note = marks.pop(rid, "")
        out.append(f"  R-{rid:05d}  {t_ref[k]/1000:9.1f}  {hours[k]:9.2f}  "
                   f"{ratio[k]:10.4f}  {tt.reach_area_km2[rid]:9.1f}  {note}")
    # make sure the landmark reaches are shown even if they fall past n_show
    for k in range(len(ids)):
        rid = int(ids[k])
        if rid in marks:
            note = marks.pop(rid)
            out.append(f"  ...")
            out.append(f"  R-{rid:05d}  {t_ref[k]/1000:9.1f}  {hours[k]:9.2f}  "
                       f"{ratio[k]:10.4f}  {tt.reach_area_km2[rid]:9.1f}  {note}")
    for rid, name in marks.items():
        out.append(f"  R-{rid:05d}  {'NOT DOWNSTREAM OF THE INJECTION POINT':>40}  {name}")
    return "\n".join(out)
