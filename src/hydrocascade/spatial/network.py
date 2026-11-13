"""The channel network N — a 1D directed metric graph (Doc 02 §2.2, Doc 05 §3).

Flood events live here. Cells whose contributing area exceeds the configured
threshold are channel cells; they form a tree under the D8 receiver relation,
which we cut at junctions into *links* and then into *reaches* of nominal
500 m.

Building the network from the receiver forest rather than by vectorising a
raster mask means the topology is exact by construction: a reach's downstream
neighbour is whatever its last cell drains into, so the graph cannot disagree
with the flow routing that Phase 4 relies on.

Outputs
-------
``channel_network.gpkg``   one LineString per reach, with attributes
``reach_index.npz``        cell -> reach id, and the per-reach cell lists
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from .. import paths
from ..config import Node
from ..logging_setup import get
from .terrain import Routing

log = get("spatial.network")

NETWORK_PATH = paths.SPATIAL / "channel_network.gpkg"
REACH_INDEX_PATH = paths.SPATIAL / "reach_index.npz"


def channel_mask(rt: Routing, threshold_km2: float) -> np.ndarray:
    """Flat boolean mask of channel cells."""
    return rt.contributing_area_km2() >= threshold_km2


def _channel_topology(rt: Routing, is_channel: np.ndarray):
    """In-degree of each channel cell, counting only channel donors."""
    n = rt.receiver.shape[0]
    donors = np.flatnonzero(is_channel)
    recv = rt.receiver[donors]
    ok = (recv >= 0) & is_channel[np.clip(recv, 0, n - 1)]
    indeg = np.bincount(recv[ok], minlength=n)
    return indeg


def _build_links(rt: Routing, is_channel: np.ndarray, indeg: np.ndarray) -> list[np.ndarray]:
    """Cut the channel tree into maximal junction-to-junction cell paths.

    A link starts at a source (no channel donor) or at a confluence (two or
    more), and runs downstream until the cell *before* the next confluence, or
    until the flow leaves the channel network.
    """
    n = rt.receiver.shape[0]
    chan = is_channel
    recv = rt.receiver
    starts = np.flatnonzero(chan & ((indeg == 0) | (indeg >= 2)))
    log.info("channel cells %d, sources %d, confluences %d",
             int(chan.sum()), int((chan & (indeg == 0)).sum()),
             int((chan & (indeg >= 2)).sum()))

    links: list[np.ndarray] = []
    visited = np.zeros(n, dtype=bool)
    recv_l = recv.tolist()
    chan_l = chan.tolist()
    indeg_l = indeg.tolist()

    for s in starts.tolist():
        cells = [s]
        visited[s] = True
        i = s
        while True:
            nxt = recv_l[i]
            if nxt < 0 or not chan_l[nxt]:
                break                      # leaves the network: outlet
            if indeg_l[nxt] >= 2:
                cells.append(nxt)          # touch the junction, then stop
                break
            cells.append(nxt)
            visited[nxt] = True
            i = nxt
        links.append(np.asarray(cells, dtype=np.int64))

    # Any channel cell not covered belongs to a cycle-free remnant we missed;
    # there should be none, but check rather than assume.
    covered = np.zeros(n, dtype=bool)
    for cells in links:
        covered[cells] = True
    missed = int((chan & ~covered).sum())
    if missed:
        log.warning("%d channel cells were not assigned to a link", missed)
    return links


def _split_into_reaches(cells: np.ndarray, rt: Routing, target_m: float) -> list[np.ndarray]:
    """Cut one link into pieces of roughly ``target_m`` along-channel metres."""
    if cells.size < 2:
        return [cells]
    steps = rt.step_len[cells[:-1]]
    total = float(steps.sum())
    if total <= target_m * 1.5:
        return [cells]

    n_pieces = max(int(round(total / target_m)), 1)
    edge = total / n_pieces
    out: list[np.ndarray] = []
    start = 0
    run = 0.0
    for k in range(cells.size - 1):
        run += steps[k]
        if run >= edge and len(out) < n_pieces - 1:
            out.append(cells[start:k + 2])   # share the junction cell
            start = k + 1
            run = 0.0
    out.append(cells[start:])
    return [p for p in out if p.size >= 1]


def build(cfg: Node, *, force: bool = False) -> dict[str, Path]:
    if NETWORK_PATH.is_file() and REACH_INDEX_PATH.is_file() and not force:
        log.info("cached  %s", NETWORK_PATH.name)
        return {"network": NETWORK_PATH, "index": REACH_INDEX_PATH}

    rt = Routing()
    thr = float(cfg.spatial.flow.channel_threshold_km2)
    target = float(cfg.spatial.network.reach_length_m)
    min_slope = float(cfg.spatial.travel_time.min_slope)

    is_channel = channel_mask(rt, thr)
    if not is_channel.any():
        raise RuntimeError(f"no cell reaches the {thr} km2 channel threshold")

    indeg = _channel_topology(rt, is_channel)
    links = _build_links(rt, is_channel, indeg)
    log.info("links   %d", len(links))

    # Elevation for bed slope: use the filled surface, so a reach can never
    # come out with a negative drop.
    import rasterio
    with rasterio.open(paths.SPATIAL / "dem_filled.tif") as src:
        z = src.read(1).ravel().astype(np.float64)

    area = rt.contributing_area_km2()

    reaches: list[np.ndarray] = []
    for cells in links:
        reaches.extend(_split_into_reaches(cells, rt, target))
    log.info("reaches %d (nominal %.0f m)", len(reaches), target)

    # ------------------------------------------------------------------ rows
    cell_to_reach = np.full(rt.receiver.shape[0], -1, dtype=np.int32)
    records = []
    geoms = []
    last_cell = np.empty(len(reaches), dtype=np.int64)
    first_cell = np.empty(len(reaches), dtype=np.int64)

    for rid, cells in enumerate(reaches):
        # A shared junction cell belongs to the upstream reach that ends on it;
        # assign interior cells first, then let the owning reach claim it.
        cell_to_reach[cells] = rid
        first_cell[rid] = cells[0]
        last_cell[rid] = cells[-1]

        length = float(rt.step_len[cells[:-1]].sum()) if cells.size > 1 else rt.cellsize
        z_up, z_dn = float(z[cells[0]]), float(z[cells[-1]])
        slope = max((z_up - z_dn) / max(length, 1e-9), min_slope)

        x, y = rt.xy(cells)
        geoms.append(LineString(np.column_stack([x, y])) if cells.size > 1
                     else LineString([(x[0], y[0]), (x[0] + 1e-3, y[0])]))

        records.append({
            "reach_id": rid,
            "n_cells": int(cells.size),
            "length_m": length,
            "z_up_m": z_up,
            "z_dn_m": z_dn,
            "slope": slope,
            "area_up_km2": float(area[cells[0]]),
            "area_dn_km2": float(area[cells[-1]]),
        })

    gdf = gpd.GeoDataFrame(pd.DataFrame(records), geometry=geoms, crs=rt.crs)

    # ------------------------------------------------------- downstream link
    # The reach a given reach flows into: whatever its last cell drains to.
    recv_last = rt.receiver[last_cell]
    valid_recv = recv_last >= 0
    ds = np.full(len(reaches), -1, dtype=np.int64)
    ds[valid_recv] = cell_to_reach[recv_last[valid_recv]]
    ds[ds == np.arange(len(reaches))] = -1          # never point at itself
    gdf["downstream_reach_id"] = ds

    n_outlets = int((ds < 0).sum())
    log.info("outlets %d", n_outlets)

    gdf["strahler"] = _strahler(ds, len(reaches))
    log.info("Strahler order: max %d", int(gdf["strahler"].max()))

    paths.SPATIAL.mkdir(parents=True, exist_ok=True)
    gdf.to_file(NETWORK_PATH, driver="GPKG", layer="reaches")

    flat_cells = np.concatenate(reaches)
    offsets = np.cumsum([0] + [c.size for c in reaches])
    np.savez_compressed(
        REACH_INDEX_PATH,
        cell_to_reach=cell_to_reach,
        reach_cells=flat_cells.astype(np.int32),
        reach_offsets=offsets.astype(np.int64),
        first_cell=first_cell.astype(np.int32),
        last_cell=last_cell.astype(np.int32),
        downstream=ds.astype(np.int32),
        is_channel=np.packbits(is_channel),
        n_cells_grid=np.int64(is_channel.size),
    )

    total_km = gdf["length_m"].sum() / 1000
    aoi_km2 = float(area.max())
    log.info("network %.0f km of channel, drainage density %.2f km/km2",
             total_km, total_km / max(aoi_km2, 1))
    return {"network": NETWORK_PATH, "index": REACH_INDEX_PATH}


def _strahler(downstream: np.ndarray, n: int) -> np.ndarray:
    """Strahler stream order over the reach graph."""
    order = np.zeros(n, dtype=np.int32)
    indeg = np.bincount(downstream[downstream >= 0], minlength=n)
    frontier = [i for i in range(n) if indeg[i] == 0]
    order[frontier] = 1
    pending = {i: [] for i in range(n)}
    remaining = indeg.copy()

    stack = list(frontier)
    while stack:
        i = stack.pop()
        d = int(downstream[i])
        if d < 0:
            continue
        pending[d].append(int(order[i]))
        remaining[d] -= 1
        if remaining[d] == 0:
            ups = pending[d]
            top = max(ups)
            order[d] = top + 1 if ups.count(top) > 1 else top
            stack.append(d)
    return order


class Network:
    """Loaded channel network with its cell index."""

    def __init__(self):
        if not NETWORK_PATH.is_file():
            raise FileNotFoundError(f"{NETWORK_PATH} missing — run `hydrocascade network`")
        self.gdf = gpd.read_file(NETWORK_PATH, layer="reaches")
        z = np.load(REACH_INDEX_PATH, allow_pickle=False)
        self.cell_to_reach = z["cell_to_reach"].astype(np.int64)
        self.reach_cells = z["reach_cells"].astype(np.int64)
        self.reach_offsets = z["reach_offsets"]
        self.first_cell = z["first_cell"].astype(np.int64)
        self.last_cell = z["last_cell"].astype(np.int64)
        self.downstream = z["downstream"].astype(np.int64)
        n_grid = int(z["n_cells_grid"])
        self.is_channel = np.unpackbits(z["is_channel"])[:n_grid].astype(bool)

    def __len__(self) -> int:
        return len(self.gdf)

    def cells_of(self, reach_id: int) -> np.ndarray:
        a, b = self.reach_offsets[reach_id], self.reach_offsets[reach_id + 1]
        return self.reach_cells[a:b]

    def snap(self, point, max_dist_m: float = 750.0,
             min_area_km2: float | None = None) -> tuple[int, float]:
        """Snap a projected point to a reach, preferring the main stem.

        Returns ``(reach_id, distance_m)``.

        Nearest-line snapping is the wrong tool for locating a named river at a
        named town: at Nilambur it picks a Strahler-1 roadside stream 528 m away
        in preference to the Chaliyar itself 588 m away. So among the reaches
        within ``max_dist_m`` we take the one with the **largest contributing
        area** — the main stem — which is what "the Chaliyar at Nilambur" means.

        The radius has to stay tight. Widen it to 1.5 km and the rule flips the
        other way at Chooralmala, preferring a 40 km2 river 1,492 m away over
        the Punnapuzha running under the bridge itself. 750 m is enough to let
        the main stem beat an adjacent ditch, and small enough that only
        genuinely co-located reaches compete.
        """
        d = self.gdf.geometry.distance(point)
        near = d[d <= max_dist_m]
        if near.empty:                       # nothing close: fall back to nearest
            k = int(d.idxmin())
            return int(self.gdf.iloc[k]["reach_id"]), float(d.iloc[k])

        cand = self.gdf.loc[near.index]
        if min_area_km2 is not None:
            bigger = cand[cand["area_dn_km2"] >= min_area_km2]
            if not bigger.empty:
                cand = bigger
        k = cand["area_dn_km2"].idxmax()
        return int(self.gdf.loc[k, "reach_id"]), float(d.loc[k])

    def downstream_chain(self, reach_id: int, max_steps: int = 100_000) -> list[int]:
        """Reach ids from ``reach_id`` to the outlet, in order."""
        out = [int(reach_id)]
        i = int(reach_id)
        for _ in range(max_steps):
            nxt = int(self.downstream[i])
            if nxt < 0 or nxt in out[-2:]:
                break
            out.append(nxt)
            i = nxt
        return out
