"""Flow routing on a DEM: sink filling, flat resolution, D8, accumulation.

Doc 11 §3 offers ``pysheds`` / ``richdem`` / ``WhiteboxTools``. We implement the
routing here instead, for three reasons:

1. The projection operator (Doc 05 §4) needs flow-path *tracing to the first
   channel cell*, which none of those libraries expose directly — we would be
   writing this traversal anyway.
2. It removes a fragile compiled dependency on Windows.
3. Pure functions over plain arrays can be unit-tested on synthetic DEMs whose
   correct answer is known by hand, which is what ``tests/test_flow.py`` does.

Everything here is deliberately free of I/O, config and CRS handling; see
:mod:`hydrocascade.spatial.terrain` for the stage that wires it up.

Conventions
-----------
* Neighbours are indexed 0-7 clockwise from north: N, NE, E, SE, S, SW, W, NW.
* ``receiver[i]`` is the flat index of the cell that ``i`` drains to, or ``-1``
  for an outlet, a pit or an invalid cell. This forest representation makes
  accumulation and tracing single ordered scans.
* Elevation surfaces are ``float64``. The flat-resolution increment is 1e-6 m,
  which float32 could not represent against a 2,600 m relief.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# clockwise from north
D8_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
)
FLAT_EPS = 1e-6


def _structure() -> np.ndarray:
    return np.ones((3, 3), dtype=bool)


def border_mask(shape: tuple[int, int]) -> np.ndarray:
    """True on the outermost ring of the grid."""
    m = np.zeros(shape, dtype=bool)
    m[0, :] = m[-1, :] = True
    m[:, 0] = m[:, -1] = True
    return m


def fill_depressions(dem: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Fill closed depressions to their spill elevation (Soille-Ansoult).

    Grayscale reconstruction by erosion: seed high everywhere except at outlets
    (grid border and any cell touching nodata), then let the surface relax down
    onto the DEM. Depressions come back filled to the level at which they spill.

    Doc 05 §2.2 step 3: *"Skipping this step causes flow paths to terminate
    mid-hillside and will silently break the projection operator."*
    """
    from skimage.morphology import reconstruction

    dem = np.asarray(dem, dtype=np.float64)
    if valid is None:
        valid = np.isfinite(dem)

    # Nodata acts as an outlet: water leaving the data area has left the model.
    outlet = border_mask(dem.shape) | (ndimage.binary_dilation(~valid, _structure()) & valid)

    lo = float(np.min(dem[valid])) - 1.0 if valid.any() else 0.0
    work = np.where(valid, dem, lo)
    seed = np.full(dem.shape, float(np.max(work)) + 1.0, dtype=np.float64)
    seed[outlet] = work[outlet]

    filled = reconstruction(seed, work, method="erosion")
    filled = np.asarray(filled, dtype=np.float64)
    filled[~valid] = np.nan
    return filled


def resolve_flats(filled: np.ndarray, valid: np.ndarray | None = None,
                  eps: float = FLAT_EPS) -> np.ndarray:
    """Impose a drainage gradient across flat surfaces (Barnes et al. 2014).

    A filled depression is perfectly flat, so D8 has no defined direction on it
    and flow paths stall. We raise the stalled cells by a tiny amount that grows
    with their distance from the flat's outlet, so each one ends up strictly
    above the cell it should drain to.

    A *stalled* cell has no lower neighbour but does have an equal one. Two
    stalled cells that touch must be at the same elevation — if they differed,
    the higher one would have a lower neighbour and would not be stalled — so
    connected components of stalled cells are single-elevation regions and a
    BFS inside them cannot leak between flats at different heights.

    The outlet of such a component is not a stalled cell: it is an adjacent cell
    at the *same* elevation that does have somewhere lower to go. Distance is
    measured from the stalled cells that touch one, and the increment is
    ``d_low + 1`` so that every stalled cell sits strictly above the outlet it
    drains through.

    A second, much smaller term pushes flow away from cells that touch higher
    ground, which is what makes the resulting drainage pattern look natural
    rather than striped. It is scaled to stay below one unit of the primary
    gradient, so it breaks ties without ever reordering them.
    """
    filled = np.asarray(filled, dtype=np.float64)
    if valid is None:
        valid = np.isfinite(filled)

    has_lower = np.zeros(filled.shape, dtype=bool)
    has_higher = np.zeros(filled.shape, dtype=bool)
    has_equal = np.zeros(filled.shape, dtype=bool)

    big = np.where(valid, filled, np.inf)
    for dr, dc in D8_OFFSETS:
        nb = _shift(big, dr, dc, fill=np.inf)
        nb_valid = np.isfinite(nb)
        has_lower |= nb_valid & (nb < big)
        has_higher |= nb_valid & (nb > big)
        has_equal |= nb_valid & (nb == big)

    stalled = valid & ~has_lower & has_equal
    if not stalled.any():
        return filled.copy()

    # An outlet of a flat: same elevation as the stalled cell beside it, but it
    # has somewhere lower to go. Note this is a *neighbour* test, not a global
    # one — the surrounding ridge also "has a lower neighbour" and must not
    # count as an outlet.
    drains_out = valid & has_lower
    low_edge = np.zeros(filled.shape, dtype=bool)
    for dr, dc in D8_OFFSETS:
        nb_elev = _shift(big, dr, dc, fill=np.inf)
        nb_drains = _shift(drains_out, dr, dc, fill=False)
        low_edge |= stalled & nb_drains & (nb_elev == big)

    # A flat can also escape by leaving the model entirely: off the grid edge,
    # or into nodata. Water bodies make this the common case, not the rare one
    # — Copernicus DEM flattens the sea and every reservoir to a constant
    # elevation, so they arrive here as huge flats whose only way out is the
    # coastline or the clip boundary. Without this, the whole sea surface reads
    # as an unresolvable depression.
    escapes_model = valid & (border_mask(filled.shape) |
                             (ndimage.binary_dilation(~valid, _structure()) & valid))
    low_edge |= stalled & escapes_model

    high_edge = stalled & has_higher

    d_low = _bfs_within(low_edge, stalled)
    d_high = _bfs_within(high_edge, stalled)

    known_low = d_low >= 0
    known_high = d_high >= 0
    max_high = int(d_high.max()) if known_high.any() else 0
    minor = eps / (2.0 * (max_high + 2))     # strictly below one primary step

    inc = np.zeros(filled.shape, dtype=np.float64)
    sel = stalled & known_low
    inc[sel] = eps * (d_low[sel] + 1)

    # A component with no reachable outlet at all — a closed lake whose rim the
    # DEM never breaches. Nothing can drain it, but we can at least give it a
    # gradient running *away* from the surrounding high ground so flow gathers
    # at one interior point instead of stalling on every cell of the surface.
    orphan = stalled & ~known_low & known_high
    if orphan.any():
        inc[orphan] = eps * (max_high - d_high[orphan] + 1)

    tie = stalled & known_high & known_low
    inc[tie] += minor * (max_high - d_high[tie])

    out = filled.copy()
    out[stalled] = filled[stalled] + inc[stalled]
    return out


def _bfs_within(seeds: np.ndarray, region: np.ndarray, max_iter: int = 100_000) -> np.ndarray:
    """Multi-source BFS hop count inside ``region``; -1 where unreachable."""
    dist = np.full(region.shape, -1, dtype=np.int32)
    cur = seeds & region
    dist[cur] = 0
    d = 0
    while cur.any() and d < max_iter:
        d += 1
        nxt = ndimage.binary_dilation(cur, _structure()) & region & (dist < 0)
        if not nxt.any():
            break
        dist[nxt] = d
        cur = nxt
    return dist


def _shift(a: np.ndarray, dr: int, dc: int, fill: float) -> np.ndarray:
    """``a`` shifted so that result[r, c] == a[r + dr, c + dc], padded with ``fill``."""
    out = np.full_like(a, fill)
    rows, cols = a.shape
    r_src = slice(max(dr, 0), rows + min(dr, 0))
    c_src = slice(max(dc, 0), cols + min(dc, 0))
    r_dst = slice(max(-dr, 0), rows + min(-dr, 0))
    c_dst = slice(max(-dc, 0), cols + min(-dc, 0))
    out[r_dst, c_dst] = a[r_src, c_src]
    return out


def d8_receivers(elev: np.ndarray, valid: np.ndarray | None, cellsize: float
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steepest-descent D8 routing.

    Returns ``(receiver, step_len, direction)`` as flat arrays of length
    ``rows*cols``:

    * ``receiver`` — flat index drained to, ``-1`` for outlet/pit/invalid
    * ``step_len`` — distance to that receiver in metres (``0`` for outlets)
    * ``direction`` — neighbour index 0-7, ``-1`` for outlets

    Steepest descent is by *slope*, i.e. drop divided by distance, so a diagonal
    neighbour must be sqrt(2) times lower to win. Using raw drop instead biases
    flow onto the diagonals and visibly distorts the drainage pattern.
    """
    elev = np.asarray(elev, dtype=np.float64)
    if valid is None:
        valid = np.isfinite(elev)
    rows, cols = elev.shape
    n = rows * cols

    flat_idx = np.arange(n, dtype=np.int64).reshape(rows, cols)
    big = np.where(valid, elev, np.nan)

    best_slope = np.zeros(elev.shape, dtype=np.float64)
    receiver = np.full(elev.shape, -1, dtype=np.int64)
    step_len = np.zeros(elev.shape, dtype=np.float64)
    direction = np.full(elev.shape, -1, dtype=np.int8)

    for k, (dr, dc) in enumerate(D8_OFFSETS):
        dist = cellsize * (np.sqrt(2.0) if dr and dc else 1.0)
        nb_elev = _shift(big, dr, dc, fill=np.nan)
        nb_idx = _shift(flat_idx.astype(np.float64), dr, dc, fill=-1.0)

        drop = big - nb_elev
        with np.errstate(invalid="ignore"):
            slope = drop / dist
            better = valid & np.isfinite(nb_elev) & (slope > best_slope)
        best_slope = np.where(better, slope, best_slope)
        receiver = np.where(better, nb_idx.astype(np.int64), receiver)
        step_len = np.where(better, dist, step_len)
        direction = np.where(better, np.int8(k), direction)

    receiver[~valid] = -1
    step_len[~valid] = 0.0
    direction[~valid] = -1
    return receiver.ravel(), step_len.ravel(), direction.ravel()


def topological_order(elev: np.ndarray, valid: np.ndarray | None = None,
                      descending: bool = True) -> np.ndarray:
    """Flat indices of valid cells ordered by elevation.

    Because every receiver is strictly lower than its donor, elevation order is
    a topological order of the drainage forest: descending visits every donor
    before its receiver, ascending visits every receiver before its donors.
    """
    elev = np.asarray(elev, dtype=np.float64)
    if valid is None:
        valid = np.isfinite(elev)
    flat_valid = valid.ravel()
    idx = np.flatnonzero(flat_valid)
    vals = elev.ravel()[idx]
    order = idx[np.argsort(vals, kind="stable")]
    return order[::-1].copy() if descending else order


def flow_accumulation(receiver: np.ndarray, order_desc: np.ndarray,
                      weights: np.ndarray | None = None) -> np.ndarray:
    """Number of upslope cells draining through each cell (self included).

    Single scan from the highest cell downward: by the time a cell is visited,
    every cell that drains into it has already contributed.
    """
    n = receiver.shape[0]
    acc = np.zeros(n, dtype=np.float64)
    if weights is None:
        acc[order_desc] = 1.0
    else:
        acc[order_desc] = weights[order_desc]

    # Plain lists beat numpy scalar indexing by roughly 3x in this tight loop.
    rec = receiver.tolist()
    a = acc.tolist()
    for i in order_desc.tolist():
        r = rec[i]
        if r >= 0:
            a[r] += a[i]
    return np.asarray(a, dtype=np.float64)


def trace_to_channel(receiver: np.ndarray, step_len: np.ndarray,
                     is_channel: np.ndarray, order_asc: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """The projection operator, evaluated for every cell at once.

    Returns ``(injection, d_slope)``:

    * ``injection[i]`` — flat index of the first channel cell on the D8 flow
      path leaving ``i``; ``-1`` if the path leaves the grid without meeting a
      channel. This is ``pi(x)`` of Doc 02 §2.2.
    * ``d_slope[i]`` — length of that path in metres, ``d_slope(x)`` of Doc 02 §2.3.

    Doing this as one ascending-elevation scan rather than a per-unit walk turns
    an O(units x path length) traversal into a single O(cells) pass: a receiver
    is always lower than its donor, so it is already resolved when the donor is
    visited.
    """
    n = receiver.shape[0]
    injection = np.full(n, -1, dtype=np.int64)
    d_slope = np.full(n, np.nan, dtype=np.float64)

    rec = receiver.tolist()
    slen = step_len.tolist()
    chan = is_channel.tolist()
    inj = injection.tolist()
    dsl = [float("nan")] * n

    for i in order_asc.tolist():
        if chan[i]:
            inj[i] = i
            dsl[i] = 0.0
            continue
        r = rec[i]
        if r < 0:
            continue          # drains off the grid without reaching a channel
        ri = inj[r]
        if ri < 0:
            continue
        inj[i] = ri
        dsl[i] = dsl[r] + slen[i]

    return np.asarray(inj, dtype=np.int64), np.asarray(dsl, dtype=np.float64)


def slope_degrees(elev: np.ndarray, cellsize: float) -> np.ndarray:
    """Terrain slope in degrees by the Horn (1981) 3x3 operator."""
    elev = np.asarray(elev, dtype=np.float64)
    filled = np.where(np.isfinite(elev), elev, np.nan)
    # Horn kernels
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / (8 * cellsize)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64) / (8 * cellsize)
    # Fill NaN with local mean so edges do not poison the convolution, then
    # re-mask afterwards.
    m = np.isfinite(filled)
    tmp = np.where(m, filled, 0.0)
    dzdx = ndimage.convolve(tmp, kx, mode="nearest")
    dzdy = ndimage.convolve(tmp, ky, mode="nearest")
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    slope[~m] = np.nan
    return slope
