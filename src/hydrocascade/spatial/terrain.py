"""Stage: run flow routing over the prepared DEM and cache the result.

Doc 05 §2.2 steps 3-4. The products here are consumed by every later stage —
the channel network, the slope units, the projection operator and the
travel-time table — so they are computed once, written to disk, and never
recomputed inside a loop.

Outputs, all in ``data/processed/spatial``:

===================  ==========================================================
``routing.npz``      receiver / step_len / direction / accumulation, flat arrays
``dem_filled.tif``   depression-filled surface, for inspection in QGIS
``flow_acc.tif``     contributing area in km2
``slope_deg.tif``    terrain slope in degrees
===================  ==========================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from .. import paths
from ..config import Node
from ..logging_setup import get
from ..ingest import dem as dem_mod
from . import flow

log = get("spatial.terrain")

ROUTING_PATH = paths.SPATIAL / "routing.npz"
FILLED_PATH = paths.SPATIAL / "dem_filled.tif"
ACC_PATH = paths.SPATIAL / "flow_acc.tif"
SLOPE_PATH = paths.SPATIAL / "slope_deg.tif"


def _write_tif(path: Path, array: np.ndarray, profile: dict) -> None:
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=np.nan, compress="deflate",
             predictor=2, tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(array.astype("float32"), 1)


def build(cfg: Node, *, force: bool = False) -> dict[str, Path]:
    if ROUTING_PATH.is_file() and not force:
        log.info("cached  %s", ROUTING_PATH.name)
        return {"routing": ROUTING_PATH}

    elev, profile = dem_mod.load()
    valid = np.isfinite(elev)
    cellsize = abs(profile["transform"].a)
    cell_area_km2 = (cellsize * cellsize) / 1e6
    rows, cols = elev.shape
    log.info("DEM %d x %d at %.1f m, %d valid cells", rows, cols, cellsize, int(valid.sum()))

    log.info("filling depressions ...")
    filled = flow.fill_depressions(elev, valid)
    raised = np.where(valid, filled - elev, 0.0)
    n_raised = int((raised > 1e-6).sum())
    log.info("        %d cells raised (%.2f%%), max fill depth %.1f m",
             n_raised, 100 * n_raised / max(int(valid.sum()), 1), float(np.nanmax(raised)))

    log.info("resolving flats ...")
    graded = flow.resolve_flats(filled, valid)
    n_graded = int((np.abs(graded - filled) > 0).sum())
    log.info("        %d flat cells given a drainage gradient", n_graded)

    log.info("D8 routing ...")
    receiver, step_len, direction = flow.d8_receivers(graded, valid, cellsize)

    # Every valid cell should now drain somewhere, except genuine outlets on the
    # grid edge or against nodata. Doc 05 §4: units without an injection reach
    # must be investigated, never discarded silently — so count them here.
    interior = valid & ~flow.border_mask(elev.shape)
    from scipy import ndimage
    interior &= ~ndimage.binary_dilation(~valid, np.ones((3, 3), bool))
    stalled = int((receiver.reshape(rows, cols)[interior] == -1).sum())
    if stalled:
        log.warning("%d interior cells still have no drainage direction "
                    "(%.4f%% of valid cells)", stalled, 100 * stalled / int(valid.sum()))
    else:
        log.info("        every interior cell drains; no pits remain")

    log.info("flow accumulation ...")
    order_desc = flow.topological_order(graded, valid, descending=True)
    acc = flow.flow_accumulation(receiver, order_desc)
    area_km2 = (acc * cell_area_km2).reshape(rows, cols)
    area_km2[~valid] = np.nan
    log.info("        max contributing area %.0f km2", float(np.nanmax(area_km2)))

    log.info("terrain slope ...")
    slope = flow.slope_degrees(filled, cellsize)

    paths.SPATIAL.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ROUTING_PATH,
        receiver=receiver.astype(np.int32),
        step_len=step_len.astype(np.float32),
        direction=direction.astype(np.int8),
        acc=acc.astype(np.float32),
        order_desc=order_desc.astype(np.int32),
        shape=np.array([rows, cols], dtype=np.int32),
        cellsize=np.float64(cellsize),
        transform=np.array(profile["transform"], dtype=np.float64)[:6],
        crs=str(profile["crs"]),
        n_stalled=np.int64(stalled),
    )
    _write_tif(FILLED_PATH, np.where(valid, filled, np.nan), profile)
    _write_tif(ACC_PATH, area_km2, profile)
    _write_tif(SLOPE_PATH, slope, profile)

    log.info("wrote   %s (%.0f MB)", ROUTING_PATH.name, ROUTING_PATH.stat().st_size / 1e6)
    return {"routing": ROUTING_PATH, "filled": FILLED_PATH,
            "acc": ACC_PATH, "slope": SLOPE_PATH}


class Routing:
    """Loaded flow-routing products, with the grid geometry attached."""

    def __init__(self, path: Path | None = None):
        p = Path(path) if path else ROUTING_PATH
        if not p.is_file():
            raise FileNotFoundError(f"{p} missing — run `hydrocascade flow` first")
        z = np.load(p, allow_pickle=False)
        self.receiver = z["receiver"].astype(np.int64)
        self.step_len = z["step_len"].astype(np.float64)
        self.direction = z["direction"]
        self.acc = z["acc"].astype(np.float64)
        self.order_desc = z["order_desc"].astype(np.int64)
        self.rows, self.cols = (int(v) for v in z["shape"])
        self.cellsize = float(z["cellsize"])
        self.crs = str(z["crs"])
        from affine import Affine
        self.transform = Affine(*z["transform"])
        self.n_stalled = int(z["n_stalled"])

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    @property
    def cell_area_km2(self) -> float:
        return (self.cellsize ** 2) / 1e6

    @property
    def order_asc(self) -> np.ndarray:
        return self.order_desc[::-1]

    def contributing_area_km2(self) -> np.ndarray:
        return self.acc * self.cell_area_km2

    def rowcol(self, flat: np.ndarray | int):
        flat = np.asarray(flat)
        return flat // self.cols, flat % self.cols

    def flat_index(self, row, col):
        return np.asarray(row) * self.cols + np.asarray(col)

    def xy(self, flat: np.ndarray | int):
        """Projected coordinates of the centre of each cell."""
        r, c = self.rowcol(flat)
        x, y = self.transform @ (np.asarray(c) + 0.5, np.asarray(r) + 0.5)
        return np.asarray(x), np.asarray(y)

    def index_of_lonlat(self, lon: float, lat: float) -> int:
        """Flat index of the cell containing a WGS84 point."""
        import rasterio.transform
        import rasterio.warp
        xs, ys = rasterio.warp.transform("EPSG:4326", self.crs, [lon], [lat])
        # `*` on an Affine is deprecated in favour of `@` for the matrix
        # product; rasterio's own helper avoids picking a side.
        row, col = rasterio.transform.rowcol(self.transform, xs[0], ys[0])
        r, c = int(row), int(col)
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(f"({lat}, {lon}) falls outside the study grid")
        return r * self.cols + c

    def downstream_path(self, start: int, max_steps: int = 1_000_000) -> np.ndarray:
        """Follow D8 downhill from ``start`` until the flow leaves the grid."""
        path = [int(start)]
        i = int(start)
        for _ in range(max_steps):
            nxt = int(self.receiver[i])
            if nxt < 0:
                break
            path.append(nxt)
            i = nxt
        return np.asarray(path, dtype=np.int64)
