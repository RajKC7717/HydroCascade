"""Evaluate a fitted model forward in time — Doc 02 §1.1, Doc 09 §1.4.

The fitting code in :mod:`hydrocascade.cascade.model` computes one scalar, the
log-likelihood of a whole catalog. Replay and the dashboard need the opposite:
the intensity at **one place, on each of a sequence of days**, using only
information available on that day. This module is that view of the same model.

Two conversions matter and are easy to confuse:

``lambda`` — the conditional intensity. A **rate**: events per unit per day.
It is not a probability and can exceed 1.

``P`` — the occurrence probability over a horizon, by the void formula
``P = 1 - exp(-integral lambda)``. Bounded in [0, 1]. This is what a dashboard
may show; ``lambda`` is not.

**Causality is enforced structurally, not by care.** :meth:`Predictor.at` takes
the history as an explicit argument and filters it to ``t' < t`` itself, so a
caller cannot accidentally hand it the future. That is the same discipline the
leakage guard applies to fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import paths
from ..config import Node
from ..logging_setup import get
from ..cascade import kernels, model

log = get("realtime.predict")


@dataclass
class Event:
    """One past event, as the predictor needs it."""
    t: float                  # days since the period start
    kind: int                 # model.L or model.F
    unit: int                 # slope unit (L) or reach (F)
    inj_reach: int
    d_slope_m: float
    mark: float


class Predictor:
    """A fitted model, evaluated forward."""

    def __init__(self, cfg: Node, params: dict, spec: model.ModelSpec):
        import json

        import geopandas as gpd
        import xarray as xr

        from ..features import terrain_covariates as tc
        from ..spatial.network import Network
        from ..spatial.travel_time import TravelTime

        self.cfg = cfg
        self.p = {k: float(v) for k, v in params.items()}
        self.spec = spec
        self.start = pd.Timestamp(str(cfg.period.start))

        self.net = Network()
        self.tt = TravelTime()
        self.units = tc.load()
        self.n_units = len(self.units)

        su = gpd.read_file(paths.SPATIAL / "slope_units.gpkg",
                           columns=["unit_id", "injection_reach_id", "d_slope_km"],
                           ignore_geometry=True).set_index("unit_id")
        self.inj_of_unit = su["injection_reach_id"].to_dict()
        self.dsl_of_unit = (su["d_slope_km"] * 1000.0).to_dict()

        drivers = json.loads(
            (paths.PROCESSED / "top_drivers.json").read_text(encoding="utf-8"))["drivers"]
        self.static_names = [d for d in drivers if d in self.units.columns]
        self.dyn_names = [d for d in drivers if d not in self.units.columns]

        self._static = {}
        for c in self.static_names:
            v = self.units[c].to_numpy(dtype=np.float64)
            m, s = float(np.nanmean(v)), float(np.nanstd(v)) or 1.0
            self._static[c] = (np.nan_to_num(v, nan=m) - m) / s

        lut = pd.read_parquet(paths.SPATIAL / "unit_rain_pixel.parquet").set_index("unit_id")
        self.n_lon = int(lut["lon_index"].max()) + 1
        self.unit_pixel = (lut["lat_index"] * self.n_lon + lut["lon_index"]).to_dict()

        ds = xr.open_dataset(paths.SPATIAL / "rainfall_pixels.nc").load()
        self.rain_days = pd.DatetimeIndex(ds.time.to_numpy()).normalize()
        self._rain = {}
        for c in self.dyn_names:
            a = ds[c].transpose("time", "latitude", "longitude").to_numpy()
            a = a.reshape(a.shape[0], -1)
            m, s = float(np.nanmean(a)), float(np.nanstd(a)) or 1.0
            self._rain[c] = (np.nan_to_num(a, nan=m) - m) / s
        ds.close()
        self._day_index = {d: i for i, d in enumerate(self.rain_days)}

        S_l = pd.read_parquet(paths.PROCESSED / "S_landslide.parquet").set_index("unit_id")
        S_f = pd.read_parquet(paths.PROCESSED / "S_flood.parquet").set_index("unit_id")
        self.S_unit = S_l["S"].to_dict()
        self.S_reach = S_f["S"].to_dict()
        self._S_unit_med = float(S_l["S"].median())
        self._S_reach_med = float(S_f["S"].median())

        self.reach_w = np.load(paths.SPATIAL / "reach_rain_weights.npz",
                               allow_pickle=False)["weights"].astype(np.float64)
        self.area = np.asarray(self.tt.reach_area_km2, dtype=np.float64)

        # Projected positions, so the forward predictor can apply the same
        # spatial kernels the likelihood does.
        rep = gpd.read_file(paths.SPATIAL / "slope_units.gpkg",
                            columns=["unit_id", "rep_x", "rep_y"],
                            ignore_geometry=True)
        self._unit_pos = {int(r.unit_id): (float(r.rep_x), float(r.rep_y))
                          for r in rep.itertuples()
                          if np.isfinite(r.rep_x) and np.isfinite(r.rep_y)}
        ends = self.net.gdf.set_index("reach_id").geometry.interpolate(
            1.0, normalized=True)
        self._reach_pos = {int(i): (float(g.x), float(g.y))
                           for i, g in ends.items()}
        self._near_cache: dict = {}

    # ------------------------------------------------------------ helpers ---

    def day_offset(self, when) -> float:
        return float((pd.Timestamp(when).normalize() - self.start).days)

    def _wet(self, unit_id: int, when) -> float:
        """``exp(sum beta_j z_j)`` — the rainfall multiplier at a unit on a day."""
        if not (self.spec.rainfall and self.dyn_names):
            return 1.0
        i = self._day_index.get(pd.Timestamp(when).normalize())
        if i is None:
            return 1.0
        pix = int(self.unit_pixel.get(int(unit_id), 0))
        lin = 0.0
        for j, c in enumerate(self.dyn_names):
            lin += self.p.get(f"beta_dyn{j}", 0.0) * self._rain[c][i, pix]
        return float(np.exp(np.clip(lin, -20.0, 20.0)))

    def _wet_reach(self, reach_id: int, when) -> float:
        if not (self.spec.rainfall and self.dyn_names):
            return 1.0
        i = self._day_index.get(pd.Timestamp(when).normalize())
        if i is None:
            return 1.0
        w = self.reach_w[int(reach_id)]
        lin = 0.0
        for j, c in enumerate(self.dyn_names):
            row = self._rain[c][i]
            den = w.sum()
            lin += self.p.get(f"beta_dyn{j}", 0.0) * (
                float(w @ row) / den if den > 1e-12 else 0.0)
        return float(np.exp(np.clip(lin, -20.0, 20.0)))

    def _static_term(self, unit_id: int) -> float:
        if not self.spec.background:
            return 1.0
        idx = int(unit_id)
        lin = 0.0
        for j, c in enumerate(self.static_names):
            arr = self._static[c]
            if 0 <= idx < arr.size:
                lin += self.p.get(f"beta_sta{j}", 0.0) * arr[idx]
        return float(np.exp(np.clip(lin, -20.0, 20.0)))

    def _phi(self, dt_days: np.ndarray) -> np.ndarray:
        """The time kernel, in numpy (no autograd needed at prediction time).

        Shifted by the same one-day minimum lag the likelihood uses. A
        predictor that let a same-day event excite would be evaluating a
        model that was never fitted.
        """
        dt = np.asarray(dt_days, dtype=np.float64) - kernels.DAILY_LAG_DAYS
        th1 = self.p.get("theta1", 0.5)
        k1 = np.where(dt >= 0, np.exp(-np.clip(dt, 0, None) / th1) / th1, 0.0)
        if self.spec.mixture_version == 1:
            return k1
        th2 = self.p.get("theta2", 4.0)
        k2 = np.where(dt >= 0, np.exp(-np.clip(dt, 0, None) / th2) / th2, 0.0)
        logits = np.array([self.p.get("w_logit1", 0.0), self.p.get("w_logit2", 0.0)])
        w = np.exp(logits - logits.max())
        w = w / w.sum()
        return w[0] * k1 + w[1] * k2

    # -------------------------------------------------------- the two views ---

    def landslide_intensity(self, unit_id: int, when, history: list[Event]
                            ) -> float:
        """``lambda_L(u, t)`` using only events strictly before ``t``.

        **The spatial kernel is not optional here.** The likelihood restricts
        landslide-to-landslide excitation to a neighbourhood and normalises the
        kernel; a forward predictor that instead adds `alpha` for every past
        event anywhere in the basin computes a completely different quantity.
        With thousands of events in history that difference is five orders of
        magnitude — enough to saturate `P = 1 - exp(-lambda)` at 1.0 on every
        single day, which is exactly what it did before this was fixed.
        """
        t = self.day_offset(when)
        base = self.p.get("base_L", 1e-4)
        S = self.S_unit.get(int(unit_id), self._S_unit_med) if self.spec.background else 1.0
        lam = base * S * self._static_term(unit_id) * self._wet(unit_id, when)

        here = self._unit_xy(unit_id)
        if here is None:
            return float(lam)

        # Only events within the kernel's reach of this unit can ever contribute,
        # and for a fixed unit that set does not change from day to day. Filter
        # it once and reuse it: a backtest baseline is ~2,000 days, and scanning
        # all ~7,000 past events on every one of them is 14 million Python-level
        # iterations to discard all but a handful.
        # The cache holds the history list itself alongside the result. Keying on
        # id() alone would be unsafe: CPython reuses an id as soon as the object
        # is freed, so a new history list built in a loop (LOCO builds one per
        # event) could inherit an old id and silently receive the wrong
        # neighbour set. Keeping the reference means the id cannot be recycled
        # while the entry exists, and the identity check below confirms it.
        key = (int(unit_id), id(history))
        hit = self._near_cache.get(key)
        near = hit[1] if hit is not None and hit[0] is history else None
        if near is None:
            r_ll = (3.0 * model.LL_SIGMA_M) ** 2
            r_fl = model.CORRIDOR_M ** 2
            near = []
            for e in history:
                there = (self._unit_xy(e.unit) if e.kind == model.L
                         else self._reach_xy(e.inj_reach))
                if there is None:
                    continue
                d2 = (here[0] - there[0]) ** 2 + (here[1] - there[1]) ** 2
                if d2 <= (r_ll if e.kind == model.L else r_fl):
                    near.append((e, d2))
            self._near_cache[key] = (history, near)

        past = [(e, d2) for e, d2 in near if e.t < t]
        if not past:
            return float(lam)

        dt = np.array([t - e.t for e, _ in past])
        phi = self._phi(dt)
        # The same functions `model.make_objective` calls — not a restatement of
        # them. The likelihood fits alpha against these normalised kernels, so a
        # predictor using any other kernel is evaluating a model never fitted.
        corridor = kernels.corridor_density()
        for (e, d2), ph in zip(past, phi):
            if ph <= 0:
                continue
            if e.kind == model.L:
                a = self.p.get("a_LL", 0.0)
                space = float(kernels.ll_spatial_density(d2))
            else:
                a = self.p.get("a_FL", 0.0) if self.spec.cross else 0.0
                space = corridor
            boost = np.exp(self.p.get("eta_L", 0.0) * e.mark) if self.spec.marks else 1.0
            lam += a * ph * space * boost
        return float(lam)

    def _unit_xy(self, unit_id: int):
        return self._unit_pos.get(int(unit_id))

    def _reach_xy(self, reach_id: int):
        return self._reach_pos.get(int(reach_id))

    def flood_intensity(self, reach_id: int, when, history: list[Event]
                        ) -> tuple[float, float]:
        """``(lambda_F(s, t), contribution from landslides)``.

        The second value is the attribution Doc 02 §9 asks the dashboard to
        show: how much of today's flood risk at this reach is *because of* an
        upstream landslide, rather than because it is raining.
        """
        t = self.day_offset(when)
        base = self.p.get("base_F", 1e-3)
        S = self.S_reach.get(int(reach_id), self._S_reach_med) if self.spec.background else 1.0
        lam = base * S * self._wet_reach(reach_id, when)
        from_landslide = 0.0

        for e in history:
            if e.t >= t or e.inj_reach < 0:
                continue
            ids, tref, _ = self.tt.downstream_of(int(e.inj_reach))
            hit = np.flatnonzero(ids == int(reach_id))
            if not hit.size:
                continue
            k = int(hit[0])

            shift = (kernels.arrival_shift(tref[k], self.p.get("v0", 9.0))
                     if self.spec.travel_time else 0.0)
            ph = float(self._phi(np.array([t - e.t - float(shift)]))[0])
            if ph <= 0:
                continue

            w = 1.0
            if e.kind == model.L:
                a = self.p.get("a_LF", 0.0) if self.spec.cross else 0.0
                if self.spec.runout_gate:
                    reach_m = (self.p.get("gate_c", 1000.0)
                               * np.exp(self.p.get("gate_gamma", 0.0) * e.mark))
                    w *= 1.0 if e.d_slope_m <= reach_m else 0.0
                if self.spec.marks:
                    w *= np.exp(self.p.get("eta_L", 0.0) * e.mark)
            else:
                a = self.p.get("a_FF", 0.0)

            if self.spec.dilution:
                a_src = self.area[int(e.inj_reach)]
                a_tgt = max(self.area[int(reach_id)], 1e-9)
                w *= float(np.clip(a_src / a_tgt, 1e-12, 1.0) ** self.p.get("psi", 0.5))

            term = a * ph * w
            lam += term
            if e.kind == model.L:
                from_landslide += term
        return float(lam), float(from_landslide)

    def arrival_window(self, inj_reach: int, reach_id: int,
                       v0_lo: float = None, v0_hi: float = None
                       ) -> tuple[float, float]:
        """Predicted arrival, in **hours**, as a range over plausible ``v0``.

        Doc 08 §5 wants this shaded on the money figure: it is a *timing*
        prediction, checkable against a hydrograph, and it is what distinguishes
        the figure from anything else in the literature. The width is honest
        uncertainty in the celerity coefficient, not a confidence band on the
        fit.
        """
        ids, tref, _ = self.tt.downstream_of(int(inj_reach))
        hit = np.flatnonzero(ids == int(reach_id))
        if not hit.size:
            return float("nan"), float("nan")
        t_ref = float(tref[int(hit[0])])
        v0 = self.p.get("v0", 9.0)
        lo = v0_lo if v0_lo is not None else v0 * 0.7
        hi = v0_hi if v0_hi is not None else v0 * 1.4
        return (t_ref / hi / 3600.0, t_ref / lo / 3600.0)

    @staticmethod
    def probability(lam_per_day: np.ndarray, horizon_days: float = 1.0) -> np.ndarray:
        """Void probability ``1 - exp(-integral lambda)`` — Doc 02 §1.1(b)."""
        x = np.asarray(lam_per_day, dtype=np.float64) * float(horizon_days)
        return 1.0 - np.exp(-np.clip(x, 0.0, 50.0))

    def event_from_row(self, row) -> Event:
        unit = int(pd.to_numeric(row["spatial_unit_id"], errors="coerce"))
        kind = model.L if row["hazard_type"] == "L" else model.F
        inj = (int(self.inj_of_unit.get(unit, -1)) if kind == model.L else unit)
        return Event(
            t=self.day_offset(row["date"]), kind=kind, unit=unit, inj_reach=inj,
            d_slope_m=float(self.dsl_of_unit.get(unit, 0.0)) if kind == model.L else 0.0,
            mark=float(pd.to_numeric(row.get("mark_value", 0.0), errors="coerce") or 0.0))
