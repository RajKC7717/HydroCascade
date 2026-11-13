"""Stages S3-S5 on the real catalog — Doc 07 §4-§6, Doc 02 §5-§6.

The intensity, in words before symbols. For a slope unit ``u`` or a reach ``s``
on day ``t``, the rate of new events is a **background** — how prone this place
is, times how wet it is today — plus a **triggering** term summing the influence
of everything that has already happened.

    lambda_L(u, t) = mu_L(u, t)
                   + sum over past L  alpha_LL * phi(dt) * N(u - u_k)
                   + sum over past F  alpha_FL * phi(dt) * corridor(u)

    lambda_F(s, t) = mu_F(s, t)
                   + sum over past F  alpha_FF * phi(dt) * [s downstream of s_k]
                   + sum over past L  alpha_LF * phi(dt - tau) * gate * dilution

``alpha_LF`` is the number the project exists to estimate.

**The quadrature grid** is ``(slope unit x day)`` for landslides and
``(reach x day)`` for floods, exactly as Doc 07 §4 specifies, so every integral
is a finite sum and nothing is approximated by sampling.

**Why the compensator is cheap.** Written naively the background integral is
17,138 units x 2,406 days = 41 million terms, recomputed at every optimiser
step. But it factorises: the static part depends only on the unit and the
dynamic part only on the rainfall pixel, and there are 63 pixels. So the inner
sum is 63 x 2,406 and the outer is 17,138 — about 170,000 terms instead of 41
million, with no approximation at all. Doc 02 §10 optimisation 4 is the same
trick applied to the travel-time table.

**No Python loop is inside the likelihood** (Doc 07 §5). Every source-target
pair that survives truncation is assembled once, up front, into flat index
arrays; each optimiser step is then a handful of vectorised tensor operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import paths
from ..config import Node
from ..logging_setup import get
from . import kernels

log = get("cascade.model")

L, F = 0, 1                       # hazard type codes, used everywhere below

# Defined once in `kernels`, re-exported here. See the note there on why they
# must never be restated.
CORRIDOR_M = kernels.CORRIDOR_M
LL_SIGMA_M = kernels.LL_SIGMA_M


# ------------------------------------------------------------------- data ---

@dataclass
class CascadeData:
    """Everything the likelihood needs, precomputed and flat."""
    # --- events, sorted by time
    t: np.ndarray                 # days since period start
    kind: np.ndarray              # L or F
    unit: np.ndarray              # slope-unit id for L, reach id for F
    mark: np.ndarray              # centred mark
    inj_reach: np.ndarray         # injection reach (L) / own reach (F)
    d_slope_m: np.ndarray         # crown-to-channel distance (L only, else 0)
    T: float                      # observation length in days

    # --- triggering pairs, (target event i <- source event j)
    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_tref_m: np.ndarray       # path integral, 0 where not a network pair
    pair_area_src: np.ndarray
    pair_area_tgt: np.ndarray
    pair_kind: np.ndarray         # 0 LL, 1 LF, 2 FL, 3 FF
    pair_dist_m: np.ndarray       # straight-line distance, 0 for network pairs

    # --- compensator support
    unit_static: np.ndarray       # (n_units, n_drivers_static)
    unit_pixel: np.ndarray        # rainfall pixel per slope unit
    unit_S: np.ndarray            # calibrated susceptibility per unit
    reach_S: np.ndarray
    reach_pixel_w: np.ndarray     # (n_reach, n_pixel) catchment weights
    rain: np.ndarray              # (n_days, n_pixel, n_dyn_drivers), standardised
    n_units: int
    n_reach: int

    # --- per-source compensator masses over the network
    src_event: np.ndarray         # event index of each (source, downstream) row
    src_tref_m: np.ndarray
    src_area_src: np.ndarray
    src_area_tgt: np.ndarray
    src_kind: np.ndarray          # 1 = L->F, 3 = F->F

    driver_names: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # --- the isotropic flood-side kernel (ablation A2 / baseline B5)
    iso_i: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    iso_j: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    iso_w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    iso_kind: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    iso_has_reach: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    @property
    def n_events(self) -> int:
        return int(self.t.size)


def _standardise(a: np.ndarray, stats: dict | None = None):
    m = float(np.nanmean(a)) if stats is None else stats["mean"]
    s = float(np.nanstd(a)) if stats is None else stats["std"]
    s = s if s > 1e-12 else 1.0
    return (np.nan_to_num(a, nan=m) - m) / s, {"mean": m, "std": s}


def assemble(cfg: Node, *, max_days: float = 60.0,
             spatial_radius_m: float = 5000.0,
             end: str | None = None,
             events: pd.DataFrame | None = None) -> CascadeData:
    """Build :class:`CascadeData` from the catalog and the Phase 2/3 artefacts.

    ``max_days`` is the temporal truncation: Doc 07 §5 drops terms older than
    about five time constants, and the longest fitted timescale here is 14 days
    for the slow mechanism, so 60 days is comfortably beyond any kernel that
    matters while cutting the pair list by orders of magnitude.

    ``end`` truncates the record, which is how the Doc 08 §4.1 three-way split
    and the Wayanad backtest refit are done — the leakage assertion in
    :mod:`hydrocascade.experiments.splits` checks that nothing later leaked in.

    ``events`` substitutes an in-memory catalog for the one on disk. The Doc 08
    §8 sensitivity analyses use it to rebuild the flood events under a different
    threshold or declustering window **without writing over** the real
    ``events.csv`` — a sensitivity run that mutates the catalog it is testing is
    how a variant silently becomes the headline.
    """
    import json

    from ..catalog import schema
    from ..catalog.build import EVENTS_PATH
    from ..features import terrain_covariates as tc
    from ..spatial.network import Network
    from ..spatial.travel_time import TravelTime

    start = pd.Timestamp(str(cfg.period.start))
    stop = pd.Timestamp(end) if end else pd.Timestamp(str(cfg.period.end))

    ev = schema.read(EVENTS_PATH) if events is None else events.copy()
    ev["_d"] = pd.to_datetime(ev["date"], errors="coerce")
    ev = ev[(ev["_d"] >= start) & (ev["_d"] <= stop)].copy()
    ev["_t"] = (ev["_d"] - start).dt.days.astype(float)
    ev["_u"] = pd.to_numeric(ev["spatial_unit_id"], errors="coerce")
    ev = ev.dropna(subset=["_u"])
    ev["_u"] = ev["_u"].astype(int)
    ev["_k"] = np.where(ev["hazard_type"] == "L", L, F)

    # Marks: log area for landslides, log(Qpeak/Q2) for floods, each centred
    # within its own type so eta = 0 is exactly "size does not matter".
    mark = pd.to_numeric(ev["mark_value"], errors="coerce").to_numpy()
    for k in (L, F):
        m = ev["_k"].to_numpy() == k
        if m.any():
            mark[m] = np.nan_to_num(mark[m], nan=np.nanmean(mark[m]))
            mark[m] -= np.nanmean(mark[m])
    ev["_m"] = np.nan_to_num(mark)

    ev = ev.sort_values("_t").reset_index(drop=True)
    T = float((stop - start).days + 1)

    net = Network()
    tt = TravelTime()
    units = tc.load()
    n_units = len(units)
    n_reach = len(net)

    # --- injection reach and d_slope per event ------------------------------
    import geopandas as gpd
    su = gpd.read_file(paths.SPATIAL / "slope_units.gpkg",
                       columns=["unit_id", "injection_reach_id", "d_slope_km"],
                       ignore_geometry=True).set_index("unit_id")
    inj_of_unit = su["injection_reach_id"].to_dict()
    dsl_of_unit = (su["d_slope_km"] * 1000.0).to_dict()

    kind = ev["_k"].to_numpy()
    unit = ev["_u"].to_numpy()
    inj = np.where(kind == L,
                   [inj_of_unit.get(int(u), -1) for u in unit],
                   unit)
    dsl = np.where(kind == L,
                   [dsl_of_unit.get(int(u), 0.0) for u in unit], 0.0)

    area = np.asarray(tt.reach_area_km2, dtype=np.float64)

    # --- per-source downstream rows (compensator + pair candidates) ---------
    src_event, src_tref, src_as, src_at, src_kind = [], [], [], [], []
    down_of_event: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, (k, r) in enumerate(zip(kind, inj)):
        if r < 0 or r >= n_reach:
            continue
        ids, tref, ratio = tt.downstream_of(int(r))
        if ids.size == 0:
            continue
        down_of_event[i] = (ids, tref, ratio)
        src_event.append(np.full(ids.size, i))
        src_tref.append(tref)
        src_as.append(np.full(ids.size, area[int(r)]))
        src_at.append(area[ids])
        src_kind.append(np.full(ids.size, 1 if k == L else 3))

    def _cat(xs, dtype=np.float64):
        return (np.concatenate(xs).astype(dtype) if xs
                else np.zeros(0, dtype=dtype))

    # --- triggering pairs ---------------------------------------------------
    t_arr = ev["_t"].to_numpy()
    pair_i, pair_j, pair_tref, pair_as, pair_at, pair_kind = [], [], [], [], [], []
    pair_dist: list[float] = []

    # reach -> the event indices whose downstream set contains it, as a dict of
    # lists built once. This is the transpose of the travel-time CSR restricted
    # to reaches that actually carry an event.
    reach_sources: dict[int, list[int]] = {}
    for i, (ids, tref, ratio) in down_of_event.items():
        for pos, s in enumerate(ids):
            reach_sources.setdefault(int(s), []).append((i, pos))

    for i in range(len(ev)):
        if kind[i] != F:
            continue
        s = int(unit[i])
        for j, pos in reach_sources.get(s, ()):
            if j >= i:
                continue
            dt = t_arr[i] - t_arr[j]
            if dt < kernels.DAILY_LAG_DAYS or dt > max_days:
                continue
            ids, tref, ratio = down_of_event[j]
            pair_i.append(i)
            pair_j.append(j)
            pair_tref.append(float(tref[pos]))
            pair_as.append(float(area[int(inj[j])]))
            pair_at.append(float(area[s]))
            pair_kind.append(1 if kind[j] == L else 3)
            pair_dist.append(0.0)

    # L targets: nearby prior L events, and prior F events inside the corridor
    from scipy.spatial import cKDTree

    rep = gpd.read_file(paths.SPATIAL / "slope_units.gpkg",
                        columns=["unit_id", "rep_x", "rep_y"],
                        ignore_geometry=True).set_index("unit_id")
    ux = rep["rep_x"].to_dict()
    uy = rep["rep_y"].to_dict()
    is_l = kind == L
    l_idx = np.flatnonzero(is_l)
    l_xy = np.full((len(ev), 2), np.nan)
    if l_idx.size:
        xy = np.array([[ux.get(int(unit[i]), np.nan), uy.get(int(unit[i]), np.nan)]
                       for i in l_idx])
        l_xy[l_idx] = xy
        ok = np.isfinite(xy).all(axis=1)
        tree = cKDTree(xy[ok])
        l_ok = l_idx[ok]
        for a, i in enumerate(l_ok):
            for b in tree.query_ball_point(xy[ok][a], r=spatial_radius_m):
                j = int(l_ok[b])
                if j >= i:
                    continue
                dt = t_arr[i] - t_arr[j]
                if dt < kernels.DAILY_LAG_DAYS or dt > max_days:
                    continue
                pair_i.append(i)
                pair_j.append(j)
                pair_tref.append(0.0)
                pair_as.append(1.0)
                pair_at.append(1.0)
                pair_kind.append(0)
                pair_dist.append(float(np.hypot(*(xy[ok][a] - xy[ok][b]))))

    # --- flood -> landslide, inside the bank corridor ------------------------
    #
    # Without these pairs `alpha_FL` appears only in the compensator, never in
    # the log-intensity term. The optimiser then sees a parameter that can only
    # ever cost likelihood and drives it to zero — which looks exactly like
    # "floods do not trigger landslides" and is actually "we never gave it the
    # chance to say otherwise". Half of the cross-hazard claim was silently
    # untestable until these were added.
    #
    # The corridor is the physical restriction (Doc 02 §5, Doc 07 §5 item 4): a
    # flood undercuts the bank it touches, and a hillslope kilometres away does
    # not care that the river rose.
    f_idx = np.flatnonzero(~is_l)
    if l_idx.size and f_idx.size:
        end_of_reach = net.gdf.set_index("reach_id").geometry.interpolate(
            1.0, normalized=True)
        f_xy = np.array([
            [end_of_reach.loc[int(unit[i])].x, end_of_reach.loc[int(unit[i])].y]
            if int(unit[i]) in end_of_reach.index else [np.nan, np.nan]
            for i in f_idx])
        f_ok = np.isfinite(f_xy).all(axis=1)
        if f_ok.any():
            f_tree = cKDTree(f_xy[f_ok])
            f_valid = f_idx[f_ok]
            for i in l_idx:
                if not np.isfinite(l_xy[i]).all():
                    continue
                for b in f_tree.query_ball_point(l_xy[i], r=CORRIDOR_M):
                    j = int(f_valid[b])
                    if j >= i:
                        continue
                    dt = t_arr[i] - t_arr[j]
                    if dt < kernels.DAILY_LAG_DAYS or dt > max_days:
                        continue
                    pair_i.append(i)
                    pair_j.append(j)
                    pair_tref.append(0.0)
                    pair_as.append(1.0)
                    pair_at.append(1.0)
                    pair_kind.append(2)
                    pair_dist.append(float(np.hypot(*(l_xy[i] - f_xy[f_ok][b]))))

    # --- the isotropic alternative: ablation A2, baseline B5 -----------------
    #
    # Until this existed, `ModelSpec.network` was read nowhere, so "remove the
    # network kernel" changed nothing and ablation A2 reproduced the full model
    # to the last decimal. Doc 08 §2 calls network-versus-isotropic the single
    # most important comparison in the paper, and it was comparing the model to
    # itself.
    #
    # Here every flood-side source spreads its excitation over the reaches within
    # ISO_RADIUS_M by straight-line distance, Gaussian-weighted and normalised so
    # each source's weights sum to 1 over the reaches it can reach. That makes
    # the compensator exactly `alpha x (temporal mass)` per source, with no
    # approximation.
    ends_all = net.gdf.set_index("reach_id").geometry.interpolate(1.0, normalized=True)
    reach_ids_all = ends_all.index.to_numpy().astype(int)
    reach_xy_all = np.column_stack([ends_all.x.to_numpy(), ends_all.y.to_numpy()])
    reach_row = {int(r): k for k, r in enumerate(reach_ids_all)}

    src_xy = np.full((len(ev), 2), np.nan)
    for i in range(len(ev)):
        if kind[i] == L:
            src_xy[i] = (ux.get(int(unit[i]), np.nan), uy.get(int(unit[i]), np.nan))
        else:
            k = reach_row.get(int(unit[i]))
            if k is not None:
                src_xy[i] = reach_xy_all[k]

    s2 = kernels.ISO_SIGMA_M ** 2
    radius = kernels.ISO_RADIUS_M
    ok_src = np.isfinite(src_xy).all(axis=1)
    idx_ok = np.flatnonzero(ok_src)
    Z = np.zeros(len(ev))
    if idx_ok.size:
        reach_tree = cKDTree(reach_xy_all)
        for a, nb in enumerate(reach_tree.query_ball_point(src_xy[idx_ok], r=radius)):
            if nb:
                i = idx_ok[a]
                d2 = ((reach_xy_all[nb] - src_xy[i]) ** 2).sum(axis=1)
                Z[i] = float(np.exp(-d2 / (2.0 * s2)).sum())

    iso_i, iso_j, iso_w, iso_kind = [], [], [], []
    f_targets = np.flatnonzero((kind == F) & ok_src)
    if f_targets.size and idx_ok.size:
        ev_tree = cKDTree(src_xy[idx_ok])
        for a, near in enumerate(ev_tree.query_ball_point(src_xy[f_targets], r=radius)):
            i = int(f_targets[a])
            for b in near:
                j = int(idx_ok[b])
                if j >= i or Z[j] <= 0.0:
                    continue
                dt = t_arr[i] - t_arr[j]
                if dt < kernels.DAILY_LAG_DAYS or dt > max_days:
                    continue
                d2 = float(((src_xy[i] - src_xy[j]) ** 2).sum())
                iso_i.append(i)
                iso_j.append(j)
                iso_w.append(float(np.exp(-d2 / (2.0 * s2)) / Z[j]))
                iso_kind.append(1 if kind[j] == L else 3)

    log.info("assembled %d events (%d L, %d F) over %.0f days; %d triggering "
             "pairs, %d downstream compensator rows",
             len(ev), int((kind == L).sum()), int((kind == F).sum()), T,
             len(pair_i), sum(len(x) for x in src_event))

    # --- background support -------------------------------------------------
    drivers = json.loads((paths.PROCESSED / "top_drivers.json").read_text(
        encoding="utf-8"))["drivers"]
    static_names = [d for d in drivers if d in units.columns]
    dyn_names = [d for d in drivers if d not in units.columns]

    static = np.column_stack([
        _standardise(units[c].to_numpy(dtype=np.float64))[0] for c in static_names
    ]) if static_names else np.zeros((n_units, 0))

    lut = pd.read_parquet(paths.SPATIAL / "unit_rain_pixel.parquet").set_index("unit_id")
    n_lon = int(lut["lon_index"].max()) + 1
    unit_pixel = (lut["lat_index"] * n_lon + lut["lon_index"]).reindex(
        units["unit_id"]).fillna(0).to_numpy().astype(int)

    import xarray as xr
    ds = xr.open_dataset(paths.SPATIAL / "rainfall_pixels.nc").load()
    days = pd.DatetimeIndex(ds.time.to_numpy())
    keep = (days >= start) & (days <= stop)
    rain_list = []
    for c in dyn_names:
        a = ds[c].transpose("time", "latitude", "longitude").to_numpy()[keep]
        a = a.reshape(a.shape[0], -1)
        a, _ = _standardise(a)
        rain_list.append(a)
    ds.close()
    rain = (np.stack(rain_list, axis=-1) if rain_list
            else np.zeros((int(keep.sum()), 63, 0)))

    S_l = pd.read_parquet(paths.PROCESSED / "S_landslide.parquet").set_index("unit_id")
    S_f = pd.read_parquet(paths.PROCESSED / "S_flood.parquet").set_index("unit_id")
    unit_S = S_l["S"].reindex(units["unit_id"]).fillna(S_l["S"].median()).to_numpy()
    reach_S = S_f["S"].reindex(range(n_reach)).fillna(S_f["S"].median()).to_numpy()

    z = np.load(paths.SPATIAL / "reach_rain_weights.npz", allow_pickle=False)
    reach_w = z["weights"].astype(np.float64)

    return CascadeData(
        t=t_arr, kind=kind, unit=unit, mark=ev["_m"].to_numpy(),
        inj_reach=inj.astype(int), d_slope_m=dsl.astype(float), T=T,
        pair_i=np.asarray(pair_i, dtype=np.int64),
        pair_j=np.asarray(pair_j, dtype=np.int64),
        pair_tref_m=np.asarray(pair_tref, dtype=np.float64),
        pair_area_src=np.asarray(pair_as, dtype=np.float64),
        pair_area_tgt=np.asarray(pair_at, dtype=np.float64),
        pair_kind=np.asarray(pair_kind, dtype=np.int64),
        pair_dist_m=np.asarray(pair_dist, dtype=np.float64),
        unit_static=static, unit_pixel=unit_pixel, unit_S=unit_S,
        reach_S=reach_S, reach_pixel_w=reach_w, rain=rain,
        n_units=n_units, n_reach=n_reach,
        src_event=_cat(src_event, np.int64), src_tref_m=_cat(src_tref),
        src_area_src=_cat(src_as), src_area_tgt=_cat(src_at),
        src_kind=_cat(src_kind, np.int64),
        iso_i=np.asarray(iso_i, dtype=np.int64),
        iso_j=np.asarray(iso_j, dtype=np.int64),
        iso_w=np.asarray(iso_w, dtype=np.float64),
        iso_kind=np.asarray(iso_kind, dtype=np.int64),
        iso_has_reach=Z > 0.0,
        driver_names=static_names + dyn_names,
        meta={"static": static_names, "dynamic": dyn_names,
              "start": str(start.date()), "end": str(stop.date()),
              "max_days": max_days, "spatial_radius_m": spatial_radius_m})


# ------------------------------------------------------------- the model ---

@dataclass
class ModelSpec:
    """Which components are switched on. Every ablation is a spec, not a branch."""
    stage: str = "S5"
    self_excite: bool = True      # alpha_LL, alpha_FF; off only for M0-M2
    cross: bool = True            # A1  alpha_LF, alpha_FL
    network: bool = True          # A2  downstream gate instead of isotropic
    travel_time: bool = True      # A3  the wave
    runout_gate: bool = True      # A4
    dilution: bool = True         # A5
    background: bool = True       # A6  S_i and drivers, else constant in space
    rainfall: bool = True         # A7
    marks: bool = True            # A8
    mixture_version: int = 2      # A9  1 = single exponential

    def label(self) -> str:
        off = [n for n, v in (("self-excitation", self.self_excite),
                              ("cross", self.cross), ("network", self.network),
                              ("travel_time", self.travel_time),
                              ("runout", self.runout_gate),
                              ("dilution", self.dilution),
                              ("background", self.background),
                              ("rainfall", self.rainfall), ("marks", self.marks))
               if not v]
        if self.mixture_version == 1:
            off.append("mixture")
        return f"{self.stage}[{'full' if not off else 'no ' + ', no '.join(off)}]"


def initial_params(spec: ModelSpec, data: CascadeData, rng) -> dict[str, float]:
    # Draw the excitation parameters *inside* their stability bounds. Sampling
    # outside and letting `optimise` clamp would collapse every restart onto the
    # same pinned starting value, which quietly turns a 20-restart multi-start
    # into one start repeated 20 times.
    bnd = bounds_for(spec, data)

    def within(key: str, lo_frac: float = 0.05, hi_frac: float = 0.7) -> float:
        lo, hi = bnd.get(key, (1e-3, 0.4))
        return float(np.exp(rng.uniform(np.log(max(lo, 1e-12) + (hi - lo) * lo_frac),
                                        np.log(lo + (hi - lo) * hi_frac))))

    # No `sigma_m` here, deliberately. The landslide -> landslide kernel width is
    # the fixed constant `kernels.LL_SIGMA_M`, shared by the likelihood and the
    # real-time predictor so both evaluate the same expression. A `sigma_m` was
    # drawn here for a while and read by nothing: it survived into three fit
    # records as an apparently fitted 1.2-2.1 km length scale that the
    # likelihood had never seen. A parameter the objective does not use is not
    # free, it is a fabricated number, and `test_no_parameter_is_created_that_
    # the_objective_ignores` now fails if one reappears. Fitting the bandwidth
    # is defensible future work, but it needs the 5 km pair-truncation radius
    # handled in the compensator first, or the estimate is biased low.
    p = {
        "base_L": float(rng.uniform(1e-5, 1e-3)),
        "base_F": float(rng.uniform(1e-4, 1e-2)),
    }
    # A parameter the spec switches off is left out entirely rather than
    # carried at zero: an unused parameter still counts in AIC, and that would
    # penalise exactly the reduced models the ablations exist to compare.
    #
    # The timescales are part of the triggering kernel, so they exist only if
    # something triggers. M0-M2 have no excitation at all, and carried a theta1
    # (and, at mixture_version 2, a theta2 and two mixture weights) that the
    # objective never read — inflating their AIC by exactly the amount this
    # comment forbids, and putting the M2 -> M3 likelihood-ratio test one
    # degree of freedom out.
    excites = spec.self_excite or spec.cross
    if excites:
        p["theta1"] = float(rng.uniform(0.1, 0.9))
    if spec.self_excite:
        p["a_LL"] = within("a_LL")
        p["a_FF"] = within("a_FF")
    if spec.cross:
        p["a_LF"] = within("a_LF")
        p["a_FL"] = within("a_FL")
    if excites and spec.mixture_version >= 2:
        p["theta2"] = float(rng.uniform(2.0, 8.0))
        p["w_logit1"] = float(rng.normal(0.5, 0.3))
        p["w_logit2"] = float(rng.normal(0.0, 0.3))
    if excites and spec.mixture_version >= 3:
        p["theta3"] = float(rng.uniform(20.0, 90.0))
        p["gamma_shape"] = float(rng.uniform(1.5, 4.0))
        p["w_logit3"] = float(rng.normal(-0.5, 0.3))
    # Travel time and dilution are properties of a network path. Without the
    # network there is no path to travel along or catchment to dilute into.
    if spec.travel_time and spec.network:
        p["v0"] = float(rng.uniform(6.0, 16.0))
    # The runout gate multiplies landslide -> flood terms and nothing else
    # (`gate * is_lf + (1 - is_lf)` in the objective), so without the cross
    # terms it has nothing to act on. A1 removes exactly those cross terms, and
    # was carrying two invisible parameters: +4 AIC charged to the one ablation
    # that tests "landslides raise flood risk", biasing that comparison towards
    # the project's own hypothesis.
    if spec.runout_gate and spec.cross:
        p["gate_c"] = float(rng.uniform(300.0, 2000.0))
        p["gate_gamma"] = float(rng.uniform(0.05, 0.6))
    if spec.dilution and spec.network:
        p["psi"] = float(rng.uniform(0.1, 1.2))
    if spec.marks and (spec.self_excite or spec.cross):
        p["eta_L"] = float(rng.uniform(0.05, 0.6))
        p["eta_F"] = float(rng.uniform(0.05, 0.6))
    # Rainfall coefficients belong to `rainfall`, not to `background`. Ablation
    # A6 switches the *spatial* background off (mu constant in space) while
    # rainfall still forces the rate in time; removing rainfall is A7's job.
    # Gating these on `background` made the objective ask for a beta_dyn that
    # was never created, and A6 died with KeyError: 'beta_dyn0'.
    if spec.rainfall:
        for i, _ in enumerate(data.meta["dynamic"]):
            p[f"beta_dyn{i}"] = float(rng.uniform(0.1, 0.8))
    if spec.background:
        for i, _ in enumerate(data.meta["static"]):
            p[f"beta_sta{i}"] = float(rng.uniform(-0.3, 0.6))
    return p


#: Largest value any single entry of the mean-offspring matrix may take. The
#: spectral radius of a 2x2 matrix is at most the largest row sum, so capping
#: every entry at 0.45 guarantees a branching ratio below 0.9 — comfortably
#: stationary — whatever combination the optimiser lands on.
MAX_OFFSPRING_ENTRY = 0.45


def bounds_for(spec: ModelSpec, data: "CascadeData | None" = None
               ) -> dict[str, tuple[float, float]]:
    """Parameter bounds, including the **stability constraint** on alpha.

    Doc 07 §4 lists "branching ratio >= 1" as an expected failure with a named
    cause: *alpha unconstrained*. Doc 07 §7 rule 7 then makes such a fit invalid
    regardless of its likelihood. Left free, the optimiser will happily walk
    there, because an explosive process fits the observed clustering slightly
    better right up until it stops being a probability model.

    The bound is **data-dependent**, and that is the whole point. ``alpha_LF``
    is not the number of offspring — it is the excitation applied to *each*
    reach downstream, and a landslide in this basin reaches a few hundred of
    them. So the quantity that must stay below 1 is ``alpha_LF x (mean
    downstream set size)``, and the bound on ``alpha_LF`` scales down
    accordingly. Bounding ``alpha_LF`` itself at some fixed number would be
    either meaningless or, at S4 where the network switches on, far too
    generous — which is exactly how the S4 fit first came back at 1.569.
    """
    b = {"theta1": kernels.TIMESCALE_BOUNDS["theta1"]}
    if spec.mixture_version >= 2:
        b["theta2"] = kernels.TIMESCALE_BOUNDS["theta2"]
    if spec.mixture_version >= 3:
        b["theta3"] = kernels.TIMESCALE_BOUNDS["theta3"]
        b["gamma_shape"] = (1.05, 8.0)
    if spec.travel_time:
        b["v0"] = kernels.V0_BOUNDS
    if spec.dilution:
        b["psi"] = (0.0, 3.0)
    if spec.runout_gate and spec.cross:          # see initial_params
        b["gate_c"] = (50.0, 20000.0)
        b["gate_gamma"] = (0.0, 2.0)

    if data is not None:
        cap = MAX_OFFSPRING_ENTRY
        n_l = max(int((data.kind == L).sum()), 1)
        n_f = max(int((data.kind == F).sum()), 1)
        # Mean number of targets each source actually reaches, which is what
        # turns an alpha into a count of offspring.
        if spec.network:
            mean_lf = max(float((data.src_kind == 1).sum()) / n_l, 1.0)
            mean_ff = max(float((data.src_kind == 3).sum()) / n_f, 1.0)
        else:
            # isotropic weights are normalised to sum to 1 per source
            mean_lf = mean_ff = 1.0
        b["a_LL"] = (1e-8, cap)
        b["a_FL"] = (1e-8, cap)
        b["a_LF"] = (1e-10, cap / mean_lf)
        b["a_FF"] = (1e-10, cap / mean_ff)
    return b


def make_objective(data: CascadeData, spec: ModelSpec):
    """Return ``objective(params) -> log-likelihood tensor``.

    Everything below is vectorised. The only Python control flow depends on the
    :class:`ModelSpec`, which is fixed before fitting starts, so no branch is
    ever evaluated per event or per step.
    """
    import torch

    d = lambda a, dt=torch.float64: torch.as_tensor(a, dtype=dt)   # noqa: E731

    t = d(data.t)
    kind = d(data.kind, torch.long)
    mark = d(data.mark)
    n_days = int(data.T)

    pi, pj = d(data.pair_i, torch.long), d(data.pair_j, torch.long)
    p_tref = d(data.pair_tref_m)
    p_as, p_at = d(data.pair_area_src), d(data.pair_area_tgt)
    p_kind = d(data.pair_kind, torch.long)
    p_dist = d(data.pair_dist_m)
    p_dt = t[pi] - t[pj]
    p_dsl = d(data.d_slope_m)[pj]
    p_mark = mark[pj]

    s_ev = d(data.src_event, torch.long)
    s_tref, s_as, s_at = d(data.src_tref_m), d(data.src_area_src), d(data.src_area_tgt)
    s_kind = d(data.src_kind, torch.long)
    s_dsl, s_mark = d(data.d_slope_m)[s_ev], mark[s_ev]
    s_t = t[s_ev]

    unit_S = d(data.unit_S)
    reach_S = d(data.reach_S)
    static = d(data.unit_static)
    rain = d(data.rain)                                  # (days, pixel, dyn)
    upix = d(data.unit_pixel, torch.long)

    # Catchment weights normalised to sum to 1 per reach, so applying them is a
    # weighted *average* and not a sum that scales with catchment size. Reaches
    # whose weights are all zero fall back to a uniform average rather than
    # dividing by zero.
    reach_w = d(data.reach_pixel_w)
    _row = reach_w.sum(dim=1, keepdim=True)
    reach_w_norm = torch.where(_row > 1e-12, reach_w / _row.clamp_min(1e-12),
                               torch.full_like(reach_w, 1.0 / reach_w.shape[1]))
    ev_unit = d(data.unit, torch.long)
    ev_day = d(np.clip(data.t.astype(int), 0, max(n_days - 1, 0)), torch.long)

    n_sta = static.shape[1]
    n_dyn = rain.shape[2] if rain.ndim == 3 else 0
    is_l_event = kind == L
    ev_is_l = is_l_event.to(torch.float64)

    # isotropic flood-side pairs, used only when spec.network is False
    iso_i = d(data.iso_i, torch.long)
    iso_j = d(data.iso_j, torch.long)
    iso_w = d(data.iso_w)
    iso_kind = d(data.iso_kind, torch.long)
    iso_dt = t[iso_i] - t[iso_j] if iso_i.numel() else torch.zeros(0, dtype=torch.float64)
    iso_dsl = d(data.d_slope_m)[iso_j] if iso_j.numel() else torch.zeros(0, dtype=torch.float64)
    iso_mark = mark[iso_j] if iso_j.numel() else torch.zeros(0, dtype=torch.float64)
    has_reach = d(data.iso_has_reach.astype(np.float64)) if data.iso_has_reach.size \
        else torch.zeros(data.n_events, dtype=torch.float64)
    ev_dsl = d(data.d_slope_m)

    def kernel_mass(dt_days, q):
        """Integral of the time kernel from ``dt`` to the end of the record."""
        if spec.mixture_version == 1:
            return 1.0 - torch.exp(-dt_days.clamp_min(0.0) / q["theta1"])
        w = torch.softmax(torch.stack(
            [q["w_logit1"], q["w_logit2"]]
            + ([q["w_logit3"]] if spec.mixture_version >= 3 else [])), 0)
        m = w[0] * (1.0 - torch.exp(-dt_days.clamp_min(0.0) / q["theta1"]))
        m = m + w[1] * (1.0 - torch.exp(-dt_days.clamp_min(0.0) / q["theta2"]))
        if spec.mixture_version >= 3:
            m = m + w[2] * torch.ones_like(dt_days)      # gamma mass ~ 1 over 60 d
        return m

    def objective(q):
        # ---------------------------------------------------- background ----
        if spec.background:
            lin_u = torch.zeros(data.n_units, dtype=torch.float64)
            for i in range(n_sta):
                lin_u = lin_u + q[f"beta_sta{i}"] * static[:, i]
            unit_base = unit_S * torch.exp(lin_u)
            reach_base = reach_S
        else:
            unit_base = torch.ones(data.n_units, dtype=torch.float64)
            reach_base = torch.ones(data.n_reach, dtype=torch.float64)

        if spec.rainfall and n_dyn:
            lin_r = torch.zeros(rain.shape[0], rain.shape[1], dtype=torch.float64)
            for i in range(n_dyn):
                lin_r = lin_r + q[f"beta_dyn{i}"] * rain[:, :, i]
            wet = torch.exp(lin_r.clamp(-20.0, 20.0))    # (days, pixel)
        else:
            wet = torch.ones(max(n_days, 1), reach_w.shape[1], dtype=torch.float64)

        # The factorisation that makes this tractable: sum over days once per
        # pixel (63 x 2406), then weight by unit. Not 41 million terms.
        wet_sum_pixel = wet.sum(dim=0)                    # (pixel,)
        comp_L = q["base_L"] * (unit_base * wet_sum_pixel[upix]).sum()
        reach_wet_sum = reach_w_norm @ wet_sum_pixel      # (reach,)
        comp_F = q["base_F"] * (reach_base * reach_wet_sum).sum()

        # The log-intensity term must use the SAME definition of wetness as the
        # compensator above, or `base_F` absorbs the mismatch and every rainfall
        # coefficient is biased. For a reach that means the catchment-weighted
        # average — "rain on the hills is what floods the river" (Doc 05 §7) —
        # not the rainfall at some single pixel.
        #
        # Computed per event rather than as a (days x reach) grid: 6,300 flood
        # events by 63 pixels is 400k numbers, where the full grid would be
        # 15 million and 120 MB rebuilt at every optimiser step.
        unit_ix = ev_unit.clamp(max=data.n_units - 1)
        reach_ix = ev_unit.clamp(max=data.n_reach - 1)
        wet_unit = wet[ev_day, upix[unit_ix]]
        wet_reach = (wet[ev_day] * reach_w_norm[reach_ix]).sum(dim=1)

        mu_ev = torch.where(
            is_l_event,
            q["base_L"] * unit_base[unit_ix] * wet_unit,
            q["base_F"] * reach_base[reach_ix] * wet_reach)

        # ---------------------------------------------------- triggering ----
        if not (spec.self_excite or spec.cross):
            # M0-M2 are inhomogeneous Poisson: no triggering at all. Return
            # here rather than evaluating the kernel and multiplying it by a
            # zero alpha — with nothing to trigger there are no timescales
            # either, so `q` legitimately has no theta to evaluate with.
            return (torch.log(mu_ev.clamp_min(1e-300)).sum() - comp_L - comp_F)

        zero = torch.zeros((), dtype=torch.float64)
        alpha = {0: q["a_LL"] if spec.self_excite else zero,
                 3: q["a_FF"] if spec.self_excite else zero,
                 1: q["a_LF"] if spec.cross else zero,
                 2: q["a_FL"] if spec.cross else zero}
        use_network = bool(spec.network)
        use_tt = bool(spec.travel_time) and use_network
        use_dil = bool(spec.dilution) and use_network
        # The gate acts on landslide -> flood terms only, so it exists only
        # when those terms do. See `initial_params`.
        use_gate = bool(spec.runout_gate) and bool(spec.cross)

        def mark_boost(mk, src_is_l):
            """``exp(eta_i m_k)`` with ``eta`` chosen by the **source's** type.

            Every boost used to apply eta_L, flood sources included, so eta_F
            was fitted but never used — a free parameter that only inflated the
            parameter count and the AIC.
            """
            if not spec.marks or "eta_L" not in q:
                return torch.ones_like(mk)
            eta = q["eta_L"] * src_is_l + q["eta_F"] * (1.0 - src_is_l)
            return torch.exp(eta * mk)

        def factors(dt_days, tref, a_src, a_tgt, dsl, mk, kd):
            shift = (kernels.arrival_shift(tref, q["v0"])
                     if use_tt else torch.zeros_like(dt_days))
            eff = dt_days - shift - kernels.DAILY_LAG_DAYS
            gate = (kernels.runout_gate(dsl, mk, q["gate_c"], q["gate_gamma"])
                    if use_gate else torch.ones_like(dt_days))
            dil = (kernels.dilution(a_src, a_tgt, q["psi"])
                   if use_dil else torch.ones_like(dt_days))
            src_is_l = ((kd == 0) | (kd == 1)).to(dt_days.dtype)
            boost = mark_boost(mk, src_is_l)
            # the runout gate applies only where a landslide feeds a flood
            is_lf = (kd == 1).to(dt_days.dtype)
            gate = gate * is_lf + (1.0 - is_lf)
            return eff, gate * dil * boost

        eff, w = factors(p_dt, p_tref, p_as, p_at, p_dsl, p_mark, p_kind)
        a_vec = torch.stack([alpha[0], alpha[1], alpha[2], alpha[3]])[p_kind]
        phi = kernels.mechanism_mixture(eff, q, version=spec.mixture_version)

        # Spatial density for the hillslope-side kernels, per km^2, so that the
        # log-intensity term integrates to the same total the compensator
        # charges. The compensator gives each source exactly alpha * (kernel
        # mass in time), which assumes the spatial kernel integrates to 1 over
        # space -- so the intensity term must carry a *normalised* density, not
        # a bare alpha. Getting this wrong makes the two halves of the
        # likelihood describe different models.
        space = torch.ones_like(phi)
        is_ll = (p_kind == 0).to(phi.dtype)
        is_fl = (p_kind == 2).to(phi.dtype)
        if bool(is_ll.any()):
            g = kernels.ll_spatial_density(p_dist ** 2)
            space = space * (1.0 - is_ll) + g * is_ll
        if bool(is_fl.any()):
            u = torch.full_like(phi, kernels.corridor_density())
            space = space * (1.0 - is_fl) + u * is_fl

        contrib = a_vec * w * phi * space

        # Flood-side pairs come from one of two kernels, never both. With the
        # network off, the downstream-gated pairs are dropped and the isotropic
        # ones take their place.
        if not use_network:
            is_net_flood = ((p_kind == 1) | (p_kind == 3)).to(phi.dtype)
            contrib = contrib * (1.0 - is_net_flood)

        excite = torch.zeros(data.n_events, dtype=torch.float64)
        excite = excite.index_add(0, pi, contrib)

        if not use_network and iso_i.numel():
            iso_src_l = (iso_kind == 1).to(torch.float64)
            iso_gate = (kernels.runout_gate(iso_dsl, iso_mark, q["gate_c"], q["gate_gamma"])
                        if use_gate else torch.ones_like(iso_w))
            iso_gate = iso_gate * iso_src_l + (1.0 - iso_src_l)
            iso_phi = kernels.mechanism_mixture(iso_dt - kernels.DAILY_LAG_DAYS, q,
                                                version=spec.mixture_version)
            iso_alpha = alpha[1] * iso_src_l + alpha[3] * (1.0 - iso_src_l)
            iso_contrib = (iso_alpha * iso_gate * mark_boost(iso_mark, iso_src_l)
                           * iso_phi * iso_w)
            excite = excite.index_add(0, iso_i, iso_contrib)

        intensity = mu_ev + excite

        # ------------------------------------------- triggering compensator --
        rem = (data.T - t - kernels.DAILY_LAG_DAYS).clamp_min(0.0)
        ev_boost = mark_boost(mark, ev_is_l)

        if use_network:
            s_shift = (kernels.arrival_shift(s_tref, q["v0"])
                       if use_tt else torch.zeros_like(s_tref))
            s_remaining = (data.T - s_t - s_shift - kernels.DAILY_LAG_DAYS).clamp_min(0.0)
            s_gate = (kernels.runout_gate(s_dsl, s_mark, q["gate_c"], q["gate_gamma"])
                      if use_gate else torch.ones_like(s_tref))
            s_dil = (kernels.dilution(s_as, s_at, q["psi"])
                     if use_dil else torch.ones_like(s_tref))
            is_lf = (s_kind == 1).to(s_tref.dtype)
            s_boost = mark_boost(s_mark, is_lf)
            s_alpha = alpha[1] * is_lf + alpha[3] * (1.0 - is_lf)
            s_w = (s_gate * is_lf + (1.0 - is_lf)) * s_dil * s_boost
            comp_trig = (s_alpha * s_w * kernel_mass(s_remaining, q)).sum()
        else:
            # Isotropic weights sum to exactly 1 per source over the reaches it
            # can reach, so each source charges alpha x boost x temporal mass.
            gate_ev = (kernels.runout_gate(ev_dsl, mark, q["gate_c"], q["gate_gamma"])
                       if use_gate else torch.ones_like(rem))
            gate_ev = gate_ev * ev_is_l + (1.0 - ev_is_l)
            a_ev = alpha[1] * ev_is_l + alpha[3] * (1.0 - ev_is_l)
            comp_trig = (a_ev * gate_ev * ev_boost * has_reach
                         * kernel_mass(rem, q)).sum()

        # L->L and F->L compensators. The spatial kernels are normalised, so
        # each source charges alpha x (temporal mass) — times the SAME mark
        # boost its intensity contributions carry. Leaving the boost out here
        # while the intensity term includes it was one more case of the two
        # halves of the likelihood describing different models.
        comp_ll = (alpha[0] * ev_boost * kernel_mass(rem, q) * ev_is_l).sum()
        comp_fl = (alpha[2] * ev_boost * kernel_mass(rem, q) * (1.0 - ev_is_l)).sum()

        total = (torch.log(intensity.clamp_min(1e-300)).sum()
                 - comp_L - comp_F - comp_trig - comp_ll - comp_fl)
        return total

    return objective


def branching(params: dict, data: CascadeData, spec: ModelSpec) -> float:
    """Spectral radius of the mean-offspring matrix.

    Each entry is ``alpha_ij`` times the mean number of targets the kernel can
    actually reach — for the network terms that is the average downstream set
    size, not the whole basin, which is exactly why the network formulation is
    *more* stable than an isotropic one at the same alpha.
    """
    n_l = max(int((data.kind == L).sum()), 1)
    n_f = max(int((data.kind == F).sum()), 1)
    if spec.network:
        mean_lf = int((data.src_kind == 1).sum()) / n_l
        mean_ff = int((data.src_kind == 3).sum()) / n_f
    else:
        # The isotropic kernel's weights sum to 1 per source, so alpha is
        # already a mean offspring count.
        mean_lf = mean_ff = 1.0

    a_ll = float(params.get("a_LL", 0.0)) if spec.self_excite else 0.0
    a_ff = float(params.get("a_FF", 0.0)) * mean_ff if spec.self_excite else 0.0
    a_lf = float(params.get("a_LF", 0.0)) * mean_lf if spec.cross else 0.0
    a_fl = float(params.get("a_FL", 0.0)) if spec.cross else 0.0

    M = np.array([[a_ll, a_lf], [a_fl, a_ff]], dtype=np.float64)
    return float(np.max(np.abs(np.linalg.eigvals(M))))
