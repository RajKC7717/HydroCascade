"""Flow-routing correctness on synthetic DEMs whose answer is known by hand.

Doc 05 §2.2 step 3 warns that a bad sink fill breaks the projection operator
*silently*. These tests are the guard: each one would fail if the corresponding
piece of routing were wrong, and none of them needs a download.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrocascade.spatial import flow

CELL = 30.0


def _idx(shape, r, c):
    return r * shape[1] + c


# --------------------------------------------------------------- receivers ---

def test_inclined_plane_drains_downslope():
    """Every cell on a uniform south-facing plane drains due south."""
    rows, cols = 6, 4
    dem = np.tile(np.arange(rows, dtype=float)[::-1, None] * 10.0, (1, cols))
    receiver, step_len, direction = flow.d8_receivers(dem, None, CELL)

    for r in range(rows - 1):
        for c in range(cols):
            i = _idx(dem.shape, r, c)
            assert receiver[i] == _idx(dem.shape, r + 1, c)
            assert direction[i] == 4                      # index 4 == south
            assert step_len[i] == pytest.approx(CELL)

    # the bottom row has nothing lower: outlets
    for c in range(cols):
        assert receiver[_idx(dem.shape, rows - 1, c)] == -1


def test_steepest_descent_uses_slope_not_raw_drop():
    """A diagonal must be sqrt(2) times lower to beat a cardinal neighbour.

    Ranking by raw drop instead of slope biases drainage onto the diagonals and
    visibly distorts the derived network, so this is worth pinning down.
    """
    dem = np.array([
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0],
        [5.0, 4.0, 3.7],
    ])
    # south:     drop 1.0  over 30 m      -> slope 0.0333
    # southeast: drop 1.3  over 42.4 m    -> slope 0.0306   (bigger drop, gentler slope)
    receiver, _, direction = flow.d8_receivers(dem, None, CELL)
    i = _idx(dem.shape, 1, 1)
    assert receiver[i] == _idx(dem.shape, 2, 1)
    assert direction[i] == 4


# ------------------------------------------------------------------ filling ---

def test_pit_fills_to_its_spill_level():
    """A closed depression rises to the lowest point on its rim, not higher."""
    dem = np.array([
        [10.0, 10.0, 9.0, 10.0, 10.0],
        [10.0, 8.0, 8.0, 8.0, 10.0],
        [10.0, 8.0, 5.0, 8.0, 10.0],
        [10.0, 8.0, 8.0, 8.0, 10.0],
        [10.0, 10.0, 10.0, 10.0, 10.0],
    ])
    filled = flow.fill_depressions(dem)

    # the only way out of the basin is over the 9 on the top edge
    assert filled[2, 2] == pytest.approx(9.0)
    assert filled[1, 1] == pytest.approx(9.0)
    # the rim itself is untouched
    assert filled[0, 0] == pytest.approx(10.0)
    assert filled[0, 2] == pytest.approx(9.0)
    # filling only ever raises terrain
    assert np.all(filled >= dem - 1e-9)


def test_filling_leaves_a_drained_dem_alone():
    rows, cols = 8, 5
    dem = np.tile(np.arange(rows, dtype=float)[::-1, None] * 3.0, (1, cols))
    filled = flow.fill_depressions(dem)
    assert np.allclose(filled, dem)


def test_nodata_acts_as_an_outlet():
    """Cells beside nodata drain into it rather than ponding against it."""
    dem = np.array([
        [np.nan, np.nan, np.nan],
        [np.nan, 5.0, 8.0],
        [np.nan, 8.0, 8.0],
    ])
    filled = flow.fill_depressions(dem)
    assert filled[1, 1] == pytest.approx(5.0)   # not raised: nodata is an escape
    assert np.isnan(filled[0, 0])


# --------------------------------------------------------------------- flats ---

def test_flat_resolution_leaves_no_stalled_cell():
    """After fill + flat resolution, no interior cell is left without a receiver.

    This is the property the projection operator depends on: a cell with no
    receiver ends a flow path mid-hillside, and pi(u) then silently returns
    nothing for every unit upslope of it.
    """
    dem = np.array([
        [10.0, 10.0, 10.0, 10.0, 10.0],
        [10.0, 7.0, 7.0, 7.0, 10.0],
        [10.0, 7.0, 7.0, 7.0, 10.0],
        [10.0, 7.0, 7.0, 7.0, 10.0],
        [10.0, 10.0, 10.0, 5.0, 10.0],
    ])
    filled = flow.fill_depressions(dem)
    graded = flow.resolve_flats(filled)
    receiver, _, _ = flow.d8_receivers(graded, None, CELL)

    stalled = [
        (r, c) for r in range(1, 4) for c in range(1, 4)
        if receiver[_idx(dem.shape, r, c)] == -1
    ]
    assert stalled == [], f"cells with no drainage direction: {stalled}"

    # the increment must be small enough not to reorder real terrain
    assert np.nanmax(np.abs(graded - filled)) < 1e-3


def test_flat_resolution_routes_towards_the_real_outlet():
    """Every cell on the flat must actually reach the notch in the rim.

    The regression this guards: seeding the outlet BFS from any cell that has a
    lower neighbour makes the surrounding ridge look like an outlet, so the flat
    gets a gradient but not one that points anywhere useful.
    """
    dem = np.array([
        [10.0, 10.0, 10.0, 10.0, 10.0],
        [10.0, 7.0, 7.0, 7.0, 10.0],
        [10.0, 7.0, 7.0, 7.0, 10.0],
        [10.0, 7.0, 7.0, 7.0, 10.0],
        [10.0, 10.0, 10.0, 5.0, 10.0],
    ])
    graded = flow.resolve_flats(flow.fill_depressions(dem))
    receiver, _, _ = flow.d8_receivers(graded, None, CELL)
    notch = _idx(dem.shape, 4, 3)

    for r in range(1, 4):
        for c in range(1, 4):
            i = _idx(dem.shape, r, c)
            for _ in range(dem.size):          # walk downstream, bounded
                if i == notch or receiver[i] == -1:
                    break
                i = receiver[i]
            assert i == notch, f"cell ({r},{c}) drained to {i}, not the notch"


def test_flat_draining_off_the_grid_edge_is_resolved():
    """A flat whose only escape is the grid boundary must still drain.

    Copernicus DEM flattens the sea and every reservoir to a constant
    elevation, so on real data this is the common case: a quarter-million cells
    at exactly 0.0 m whose only way out of the model is the coastline. Treating
    "has a lower neighbour" as the only kind of outlet leaves all of them
    stalled.
    """
    dem = np.array([
        [7.0, 7.0, 7.0, 9.0, 9.0],
        [7.0, 7.0, 7.0, 7.0, 9.0],
        [9.0, 7.0, 7.0, 7.0, 9.0],
        [9.0, 9.0, 9.0, 9.0, 9.0],
        [9.0, 9.0, 9.0, 9.0, 9.0],
    ])
    graded = flow.resolve_flats(flow.fill_depressions(dem))
    receiver, _, _ = flow.d8_receivers(graded, None, CELL)

    interior_flat = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    for r, c in interior_flat:
        i = _idx(dem.shape, r, c)
        for _ in range(dem.size):
            if receiver[i] == -1:
                break
            i = receiver[i]
        rr, cc = divmod(i, dem.shape[1])
        assert rr == 0 or cc == 0, f"({r},{c}) stopped at ({rr},{cc}), not the boundary"


def test_flat_draining_into_nodata_is_resolved():
    """Nodata is an escape: water that leaves the data area has left the model."""
    dem = np.array([
        [np.nan, 7.0, 7.0, 9.0],
        [9.0, 7.0, 7.0, 9.0],
        [9.0, 7.0, 7.0, 9.0],
        [9.0, 9.0, 9.0, 9.0],
    ])
    graded = flow.resolve_flats(flow.fill_depressions(dem))
    receiver, _, _ = flow.d8_receivers(graded, None, CELL)
    i = _idx(dem.shape, 2, 2)
    for _ in range(dem.size):
        if receiver[i] == -1:
            break
        i = receiver[i]
    assert receiver[i] == -1, "flow never reached an outlet"

    # It must stop somewhere it can genuinely leave the model: on the boundary,
    # or touching the nodata cell. Which of those it picks is not specified.
    rr, cc = divmod(i, dem.shape[1])
    on_boundary = rr in (0, dem.shape[0] - 1) or cc in (0, dem.shape[1] - 1)
    touches_nodata = np.isnan(dem[max(rr - 1, 0):rr + 2, max(cc - 1, 0):cc + 2]).any()
    assert on_boundary or touches_nodata, f"stopped at ({rr},{cc}), which is not an outlet"


def test_flat_resolution_is_a_no_op_without_flats():
    rows, cols = 6, 6
    dem = np.tile(np.arange(rows, dtype=float)[::-1, None] * 5.0, (1, cols))
    dem = dem + np.arange(cols) * 0.5           # break the row-wise ties
    filled = flow.fill_depressions(dem)
    assert np.allclose(flow.resolve_flats(filled), filled)


# ------------------------------------------------------------ accumulation ---

def test_accumulation_counts_every_upslope_cell():
    """On a plane, the outlet of a column has accumulated the whole column."""
    rows, cols = 6, 4
    dem = np.tile(np.arange(rows, dtype=float)[::-1, None] * 10.0, (1, cols))
    receiver, _, _ = flow.d8_receivers(dem, None, CELL)
    order = flow.topological_order(dem, descending=True)
    acc = flow.flow_accumulation(receiver, order)

    for c in range(cols):
        assert acc[_idx(dem.shape, rows - 1, c)] == pytest.approx(rows)
        assert acc[_idx(dem.shape, 0, c)] == pytest.approx(1.0)


def test_accumulation_conserves_mass_into_a_single_outlet():
    """A converging valley routes every cell through one outlet."""
    rows, cols = 7, 7
    cc = (cols - 1) / 2
    col_profile = np.abs(np.arange(cols) - cc)          # V cross-section
    dem = col_profile[None, :] + np.arange(rows)[::-1, None] * 5.0
    receiver, _, _ = flow.d8_receivers(dem, None, CELL)
    order = flow.topological_order(dem, descending=True)
    acc = flow.flow_accumulation(receiver, order)

    outlets = np.flatnonzero(receiver == -1)
    assert acc[outlets].sum() == pytest.approx(dem.size)


# ------------------------------------------------- the projection operator ---

def test_trace_to_channel_finds_the_first_channel_cell():
    """pi(x) is the first channel cell met, and d_slope is the path length."""
    rows, cols = 5, 3
    dem = np.tile(np.arange(rows, dtype=float)[::-1, None] * 10.0, (1, cols))
    receiver, step_len, _ = flow.d8_receivers(dem, None, CELL)
    order_asc = flow.topological_order(dem, descending=False)

    is_channel = np.zeros(dem.size, dtype=bool)
    is_channel[_idx(dem.shape, rows - 1, 0):] = False
    for c in range(cols):                                # bottom row is channel
        is_channel[_idx(dem.shape, rows - 1, c)] = True

    inj, dslope = flow.trace_to_channel(receiver, step_len, is_channel, order_asc)

    for c in range(cols):
        outlet = _idx(dem.shape, rows - 1, c)
        for r in range(rows):
            i = _idx(dem.shape, r, c)
            assert inj[i] == outlet
            assert dslope[i] == pytest.approx((rows - 1 - r) * CELL)


def test_trace_stops_at_the_nearest_channel_not_the_furthest():
    """A cell above two channel rows injects into the upper one."""
    rows, cols = 5, 2
    dem = np.tile(np.arange(rows, dtype=float)[::-1, None] * 10.0, (1, cols))
    receiver, step_len, _ = flow.d8_receivers(dem, None, CELL)
    order_asc = flow.topological_order(dem, descending=False)

    is_channel = np.zeros(dem.size, dtype=bool)
    is_channel[_idx(dem.shape, 2, 0)] = True             # mid-slope channel
    is_channel[_idx(dem.shape, 4, 0)] = True             # and one lower down

    inj, dslope = flow.trace_to_channel(receiver, step_len, is_channel, order_asc)
    top = _idx(dem.shape, 0, 0)
    assert inj[top] == _idx(dem.shape, 2, 0)
    assert dslope[top] == pytest.approx(2 * CELL)


def test_channel_cells_project_onto_themselves():
    dem = np.tile(np.arange(4, dtype=float)[::-1, None] * 10.0, (1, 2))
    receiver, step_len, _ = flow.d8_receivers(dem, None, CELL)
    order_asc = flow.topological_order(dem, descending=False)
    is_channel = np.zeros(dem.size, dtype=bool)
    is_channel[_idx(dem.shape, 3, 1)] = True

    inj, dslope = flow.trace_to_channel(receiver, step_len, is_channel, order_asc)
    i = _idx(dem.shape, 3, 1)
    assert inj[i] == i
    assert dslope[i] == 0.0


# ------------------------------------------------------------------- slope ---

def test_slope_of_a_known_gradient():
    """A 100% grade must come back as 45 degrees."""
    rows, cols = 5, 5
    dem = np.tile((np.arange(rows, dtype=float)[::-1, None]) * CELL, (1, cols))
    slope = flow.slope_degrees(dem, CELL)
    assert slope[2, 2] == pytest.approx(45.0, abs=1e-6)


def test_slope_of_flat_ground_is_zero():
    dem = np.full((5, 5), 123.0)
    assert flow.slope_degrees(dem, CELL)[2, 2] == pytest.approx(0.0)
