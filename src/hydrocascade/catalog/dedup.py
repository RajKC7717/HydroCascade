"""Deduplication — Doc 04 §3.3, the rule written down before you start.

    Two landslide records are the **same event** if they are within **500 m**
    and within **1 day** of each other.

    When merging: keep the record with the higher ``confidence``; if tied, keep
    the one with a mapped polygon over a point; if still tied, keep the
    earlier-published source. Record the discarded ``event_id`` in
    ``duplicate_of``.

Doc 04 §3.3 explains why this is written down rather than decided ad hoc: two
inventories will list the same landslide with coordinates a few hundred metres
apart, and without a rule four people merge differently and the catalog becomes
inconsistent.

The "within 500 m and within 1 day" relation is *pairwise*, so groups are its
transitive closure — three records in a 400 m chain are one event even if the
end points are 800 m apart. That is a deliberate choice and it is what
connected components computes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_setup import get

log = get("catalog.dedup")

CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1, "": 0}
GEOMETRY_RANK = {"polygon_crown": 3, "polygon_centroid": 2, "point": 1, "reach": 1, "": 0}


def _to_metric(lat: np.ndarray, lon: np.ndarray, crs: str):
    """Project to the metric analysis CRS; distances cannot be computed in degrees."""
    import pyproj
    tf = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    # pyproj takes the scalar code path for a length-1 array and warns about it,
    # so hand it plain lists and rebuild the arrays ourselves.
    x, y = tf.transform(np.asarray(lon, dtype=float).ravel().tolist(),
                        np.asarray(lat, dtype=float).ravel().tolist())
    return np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))


def find_duplicate_groups(df: pd.DataFrame, *, distance_m: float, days: int,
                          crs: str) -> np.ndarray:
    """Group label per row; rows sharing a label are the same physical event.

    Records of different hazard types are never merged with each other.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(df)
    labels = np.arange(n)
    if n == 0:
        return labels

    dates = pd.to_datetime(df["date"], errors="coerce")
    day_num = (dates - pd.Timestamp("1970-01-01")).dt.days.to_numpy(dtype=float)

    offset = 0
    for htype, idx in df.groupby("hazard_type").groups.items():
        rows = np.asarray(idx)
        sub = df.loc[rows]
        lat = pd.to_numeric(sub["lat"], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(sub["lon"], errors="coerce").to_numpy(dtype=float)

        # A row with no coordinates cannot be matched against anything, so it
        # keeps its own group. Doc 04 §5 deliberately emits documented events
        # whose coordinates are not yet sourced rather than inventing them, so
        # this is a normal state, not an error.
        locatable = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(day_num[rows])
        if not locatable.any():
            labels[rows] = np.arange(rows.size) + offset
            offset += rows.size
            continue

        rows_ok = rows[locatable]
        x, y = _to_metric(lat[locatable], lon[locatable], crs)
        d = day_num[rows_ok]

        # Scale the time axis so a single radius query implements both rules:
        # a pair is a match only if it is within distance_m AND within `days`.
        # Querying the 2D tree first and filtering on time is simpler and exact.
        tree = cKDTree(np.column_stack([x, y]))
        pairs = tree.query_pairs(distance_m, output_type="ndarray")
        if pairs.size:
            keep = np.abs(d[pairs[:, 0]] - d[pairs[:, 1]]) <= days
            pairs = pairs[keep]

        m = len(rows_ok)
        if pairs.size:
            g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(m, m))
            _, comp = connected_components(g, directed=False)
        else:
            comp = np.arange(m)
        labels[rows_ok] = comp + offset
        offset += m

        # unlocatable rows each get a group of their own
        rows_bad = rows[~locatable]
        labels[rows_bad] = np.arange(rows_bad.size) + offset
        offset += rows_bad.size

    return labels


def _preference_key(df: pd.DataFrame) -> pd.DataFrame:
    """Rank rows by the Doc 04 §3.3 tie-break chain, best first."""
    out = pd.DataFrame(index=df.index)
    out["confidence_rank"] = df["confidence"].fillna("").map(
        lambda v: CONFIDENCE_RANK.get(str(v).strip().lower(), 0))
    out["geometry_rank"] = df["geometry_type"].fillna("").map(
        lambda v: GEOMETRY_RANK.get(str(v).strip().lower(), 0))
    # "earlier-published source" is approximated by the earliest access date on
    # record for that source, which is the only publication proxy the schema
    # carries. Ties fall through to a stable sort on event_id.
    acc = pd.to_datetime(df["source_date_accessed"], errors="coerce")
    out["access_order"] = acc.rank(method="min", na_option="bottom")
    return out


def deduplicate(df: pd.DataFrame, *, distance_m: float = 500.0, days: int = 1,
                crs: str = "EPSG:32643",
                hazard_types: tuple[str, ...] = ("L",)
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the merge rule to the given hazard types.

    ``hazard_types`` defaults to landslides only. See the config comment on
    ``catalog.dedup.hazard_types``: applying a 500 m rule to per-reach flood
    events merges adjacent reaches that flooded on the same day, which destroys
    the downstream propagation signal. Floods are de-duplicated by declustering
    instead (Doc 02 §3 step 3).

    Returns ``(kept, discarded)``. Discarded rows keep their original fields and
    carry ``duplicate_of`` pointing at the survivor, so nothing is lost — Doc 04
    §8 wants the merge auditable from ``events_CHANGELOG.md``.
    """
    if df.empty:
        return df.copy(), df.iloc[0:0].copy()

    work = df.reset_index(drop=True).copy()
    in_scope = work["hazard_type"].isin(hazard_types)
    if not in_scope.any():
        return work, work.iloc[0:0].copy()

    scoped = work[in_scope].reset_index(drop=True)
    labels = find_duplicate_groups(scoped, distance_m=distance_m, days=days, crs=crs)

    # rows outside the scope each keep their own group and are never merged
    full = np.empty(len(work), dtype=np.int64)
    full[np.flatnonzero(in_scope.to_numpy())] = labels
    out_idx = np.flatnonzero(~in_scope.to_numpy())
    full[out_idx] = labels.max() + 1 + np.arange(out_idx.size) if labels.size \
        else np.arange(out_idx.size)
    work["_group"] = full

    pref = _preference_key(work)
    work["_conf"] = pref["confidence_rank"]
    work["_geom"] = pref["geometry_rank"]
    work["_acc"] = pref["access_order"]

    work = work.sort_values(
        ["_group", "_conf", "_geom", "_acc", "event_id"],
        ascending=[True, False, False, True, True], kind="stable")

    first = ~work.duplicated("_group", keep="first")
    kept = work[first].copy()
    discarded = work[~first].copy()

    survivor = kept.set_index("_group")["event_id"]
    discarded["duplicate_of"] = discarded["_group"].map(survivor)

    n_groups = work["_group"].nunique()
    log.info("dedup: %d records -> %d events (%d merged away) at %.0f m / %d day, "
             "applied to hazard type(s) %s", len(work), n_groups, len(discarded),
             distance_m, days, "/".join(hazard_types))
    if len(discarded):
        by_source = discarded.groupby("source").size().sort_values(ascending=False)
        log.info("       merged away by source: %s",
                 ", ".join(f"{s}={c}" for s, c in by_source.head(5).items()))

    drop = ["_group", "_conf", "_geom", "_acc"]
    return (kept.drop(columns=drop).reset_index(drop=True),
            discarded.drop(columns=drop).reset_index(drop=True))


def assert_no_duplicates(df: pd.DataFrame, *, distance_m: float = 500.0, days: int = 1,
                         crs: str = "EPSG:32643") -> int:
    """Doc 04 §6 gate 5 / FR-03: zero remaining duplicates. Returns the count."""
    if df.empty:
        return 0
    labels = find_duplicate_groups(df.reset_index(drop=True), distance_m=distance_m,
                                   days=days, crs=crs)
    n_extra = len(labels) - len(np.unique(labels))
    if n_extra:
        log.error("FR-03: %d records remain within %.0f m and %d day of another",
                  n_extra, distance_m, days)
    return n_extra
