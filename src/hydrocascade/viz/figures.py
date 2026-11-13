"""Figures for Phases 1 and 2.

Named deliverables (Doc 04 §8, Doc 05 §11):

* ``phase2_wayanad_flowpath.png`` — the Mundakkai crown, its D8 path, the
  injection point, and the downstream reaches coloured by travel time.
  Doc 05 §11: *"This is your best Review 1 slide. It makes the whole novel idea
  visible in one picture."*
* ``phase2_network.png`` — the derived channel network over the study basin
* ``phase2_slope_units.png`` — the hillslope domain H
* ``phase1_events_per_year.png``, ``phase1_event_map.png``,
  ``phase1_size_frequency.png`` — added with the catalog stage
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
from matplotlib.colors import LightSource, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .. import paths              # noqa: E402
from ..config import Node         # noqa: E402
from ..logging_setup import get   # noqa: E402

log = get("viz.figures")

DPI = 170
WATER = "#3b7ea1"
LAND = "#f4f1ea"


def _save(fig, name: str) -> Path:
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    out = paths.FIGURES / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("wrote   figures/%s", name)
    return out


def _hillshade(ax, elev, extent, cellsize):
    """Soft hillshade so the terrain reads without competing with the data."""
    ls = LightSource(azdeg=315, altdeg=45)
    z = np.where(np.isfinite(elev), elev, np.nan)
    filled = np.nan_to_num(z, nan=float(np.nanmin(z)) if np.isfinite(z).any() else 0.0)
    shade = ls.hillshade(filled, vert_exag=2.0, dx=cellsize, dy=cellsize)
    shade = np.where(np.isfinite(z), shade, np.nan)
    ax.imshow(shade, cmap="gray", extent=extent, origin="upper",
              alpha=0.55, zorder=1, interpolation="bilinear")


def wayanad_flowpath(cfg: Node) -> Path:
    """The money slide for Review 1."""
    import geopandas as gpd
    import rasterio
    from shapely.geometry import Point

    from ..spatial.network import Network
    from ..spatial.projection import cell_level
    from ..spatial.terrain import Routing
    from ..spatial.travel_time import TravelTime

    rt = Routing()
    net = Network()
    tt = TravelTime()
    proj = cell_level(rt, net)
    lm = cfg.landmarks

    crown_cell = rt.index_of_lonlat(float(lm.mundakkai_crown["lon"]),
                                    float(lm.mundakkai_crown["lat"]))
    inj_reach = int(proj["injection_reach"][crown_cell])
    inj_cell = int(proj["injection_cell"][crown_cell])
    d_slope_km = float(proj["d_slope_m"][crown_cell]) / 1000.0

    ids, t_ref, ratio = tt.downstream_of(inj_reach)
    hours = tt.hours(t_ref)

    # the hillslope flow path from the crown down to the river
    hill_path = []
    i = crown_cell
    for _ in range(100000):
        hill_path.append(i)
        if i == inj_cell or rt.receiver[i] < 0:
            break
        i = int(rt.receiver[i])
    hill_path = np.asarray(hill_path)
    hx, hy = rt.xy(hill_path)

    # ---------------------------------------------------------------- extent
    chain = net.gdf[net.gdf["reach_id"].isin(ids)]
    pad = 6000.0
    minx, miny, maxx, maxy = chain.total_bounds
    minx = min(minx, hx.min()) - pad
    maxx = max(maxx, hx.max()) + pad
    miny = min(miny, hy.min()) - pad
    maxy = max(maxy, hy.max()) + pad

    with rasterio.open(paths.SPATIAL / "dem_filled.tif") as src:
        win = rasterio.windows.from_bounds(minx, miny, maxx, maxy, src.transform)
        elev = src.read(1, window=win)
        wt = src.window_transform(win)
        crown_z = float(next(src.sample([(cx0, cy0) for cx0, cy0 in
                                         zip(*rt.xy(np.array([crown_cell])))]))[0])
    extent = (wt.c, wt.c + elev.shape[1] * wt.a,
              wt.f + elev.shape[0] * wt.e, wt.f)

    # Size the canvas to the data so the map does not float in white space.
    aspect = (maxx - minx) / (maxy - miny)
    height = 9.0
    fig, ax = plt.subplots(figsize=(height * aspect + 2.6, height))
    ax.set_facecolor(LAND)
    _hillshade(ax, elev, extent, rt.cellsize)

    # background network, faint
    in_view = net.gdf.cx[minx:maxx, miny:maxy]
    in_view.plot(ax=ax, color=WATER, linewidth=0.35, alpha=0.5, zorder=2)

    # the downstream reaches, coloured by travel time — the whole point
    norm = Normalize(vmin=0, vmax=float(hours.max()))
    cmap = plt.get_cmap("plasma_r")
    order = np.argsort(hours)
    chain_sorted = chain.set_index("reach_id").loc[ids[order]]
    widths = 1.2 + 2.6 * np.sqrt(
        chain_sorted["area_dn_km2"].to_numpy() / max(chain_sorted["area_dn_km2"].max(), 1))
    chain_sorted.plot(ax=ax, color=cmap(norm(hours[order])), linewidth=widths, zorder=4)

    # the hillslope path: crown -> river
    ax.plot(hx, hy, color="#d1341f", linewidth=2.6, zorder=6,
            solid_capstyle="round")

    cx, cy = rt.xy(np.array([crown_cell]))
    ix, iy = rt.xy(np.array([inj_cell]))
    ax.scatter(cx, cy, s=210, marker="*", color="#d1341f", edgecolor="white",
               linewidth=1.1, zorder=8)
    ax.scatter(ix, iy, s=85, marker="o", color="white", edgecolor="#d1341f",
               linewidth=2.0, zorder=8)

    # Annotations are placed on opposite sides of their anchors: the crown, the
    # injection point and Chooralmala sit within ~5 km of each other, so shared
    # offsets would stack the three boxes on top of one another.
    ax.annotate(f"Mundakkai crown\n11.4654°N 76.1348°E · {crown_z:.0f} m",
                xy=(cx[0], cy[0]), xytext=(-150, 30), textcoords="offset points",
                fontsize=9, fontweight="bold", color="#7a1c10", zorder=9,
                arrowprops=dict(arrowstyle="->", color="#d1341f", lw=1.3),
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#d1341f", alpha=0.95))
    ax.annotate(f"injection point π(u)\nd_slope = {d_slope_km:.2f} km",
                xy=(ix[0], iy[0]), xytext=(-160, -48), textcoords="offset points",
                fontsize=9, color="#7a1c10", zorder=9,
                arrowprops=dict(arrowstyle="->", color="#d1341f", lw=1.3),
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#d1341f", alpha=0.95))

    # downstream landmarks, offset to the right to clear the crown cluster
    for key, label, off in (("chooralmala_bridge", "Chooralmala", (24, 34)),
                            ("nilambur", "Nilambur", (26, -34))):
        pt = gpd.GeoSeries([Point(float(lm[key]["lon"]), float(lm[key]["lat"]))],
                           crs="EPSG:4326").to_crs(net.gdf.crs).iloc[0]
        rid, _ = net.snap(pt)
        k = np.flatnonzero(ids == rid)
        txt = label
        if k.size:
            txt = f"{label}\n{hours[k[0]]:.1f} h · {tt.reach_area_km2[rid]:,.0f} km²"
        ax.scatter([pt.x], [pt.y], s=72, marker="v", color="#1b1b1b",
                   edgecolor="white", linewidth=1.1, zorder=8)
        ax.annotate(txt, xy=(pt.x, pt.y), xytext=off, textcoords="offset points",
                    fontsize=8.5, zorder=9,
                    arrowprops=dict(arrowstyle="-", color="#666", lw=0.9),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888", alpha=0.95))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.031, pad=0.015)
    cb.set_label("travel time from the injection point (hours, at reference $v_0$)", fontsize=9)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#bbb")

    ax.set_title("Excitation arrives as a wave, not a field\n"
                 "Mundakkai 30 July 2024 — the flow path, the injection point, "
                 "and when each downstream reach feels it",
                 fontsize=12.5, fontweight="bold", loc="left", pad=12)

    handles = [
        Line2D([], [], color="#d1341f", lw=2.6,
               label=f"D8 flow path, crown → river ({d_slope_km:.2f} km)"),
        Line2D([], [], color="#d1341f", marker="*", ls="", ms=13, label="landslide crown"),
        Line2D([], [], color="#d1341f", marker="o", ls="", ms=8, mfc="white",
               label="injection point π(u)"),
        Line2D([], [], color=WATER, lw=1.0, alpha=0.6, label="channel network N"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, framealpha=0.94)

    _scalebar(ax, corner="lower right")
    v_med = float(np.median(tt.v0_reference_ms * tt.reach_slope ** 0.3))
    ax.set_xlabel(
        f"{len(ids)} reaches downstream of the injection point within the 48 h\n"
        f"truncation radius. Line width ∝ √catchment area. Excitation reaches\n"
        f"Chooralmala in under an hour, and Nilambur about 12 h later.\n"
        f"Reference $v_0$ = {tt.v0_reference_ms:.0f}, $v = v_0 S^{{0.3}}$, median celerity "
        f"{v_med:.1f} m s⁻¹.\n"
        f"Travel times rescale as $τ = T_{{ref}}/v_0$ once $v_0$ is fitted in Phase 4.",
        fontsize=8, color="#555", labelpad=10, loc="left")
    return _save(fig, "phase2_wayanad_flowpath.png")


def _scalebar(ax, frac: float = 0.18, corner: str = "lower left"):
    x0lim, x1lim = ax.get_xlim()
    y0lim, y1lim = ax.get_ylim()
    span = x1lim - x0lim
    nice = np.array([1, 2, 5, 10, 20, 50, 100]) * 1000.0
    length = nice[np.argmin(np.abs(nice - span * frac))]

    if "right" in corner:
        x0 = x1lim - 0.05 * span - length
    else:
        x0 = x0lim + 0.05 * span
    y0 = y0lim + 0.045 * (y1lim - y0lim)

    ax.plot([x0, x0 + length], [y0, y0], color="#222", lw=3.0, zorder=10,
            solid_capstyle="butt")
    ax.text(x0 + length / 2, y0 + 0.013 * (y1lim - y0lim), f"{length/1000:.0f} km",
            ha="center", fontsize=8.5, color="#222", zorder=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))


def network_overview(cfg: Node) -> Path:
    """The channel network N over the study basin, coloured by Strahler order."""
    import rasterio

    from ..ingest.hydrosheds import load_aoi
    from ..spatial.network import Network
    from ..spatial.terrain import Routing

    rt = Routing()
    net = Network()
    aoi = load_aoi()

    with rasterio.open(paths.SPATIAL / "dem_filled.tif") as src:
        elev = src.read(1, out_shape=(src.height // 2, src.width // 2))
        t = src.transform @ src.transform.scale(2, 2)
    extent = (t.c, t.c + elev.shape[1] * t.a, t.f + elev.shape[0] * t.e, t.f)

    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    ax.set_facecolor(LAND)
    _hillshade(ax, elev, extent, rt.cellsize * 2)
    aoi.boundary.plot(ax=ax, color="#333", linewidth=1.3, zorder=3, linestyle="--")

    for order in sorted(net.gdf["strahler"].unique()):
        sel = net.gdf[net.gdf["strahler"] == order]
        sel.plot(ax=ax, color=WATER, linewidth=0.25 + 0.42 * order,
                 alpha=0.85, zorder=4)

    total_km = net.gdf["length_m"].sum() / 1000
    area = float(aoi["area_km2"].iloc[0])
    ax.set_title(f"Channel network N — {len(net):,} reaches, {total_km:,.0f} km\n"
                 f"Chaliyar basin, {area:,.0f} km², derived from Copernicus DEM 30 m",
                 fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_edgecolor("#bbb")
    handles = [Line2D([], [], color=WATER, lw=0.25 + 0.42 * o, label=f"Strahler {o}")
               for o in sorted(net.gdf["strahler"].unique())]
    handles.append(Line2D([], [], color="#333", ls="--", lw=1.3, label="basin boundary"))
    ax.legend(handles=handles, loc="upper right", fontsize=8.5, framealpha=0.92)
    _scalebar(ax, corner="lower left")
    return _save(fig, "phase2_network.png")


def slope_units_overview(cfg: Node) -> Path:
    """The hillslope domain H, with the size distribution against its target."""
    from ..ingest.hydrosheds import load_aoi
    from ..spatial.slope_units import SlopeUnits

    units = SlopeUnits()
    aoi = load_aoi()
    gdf = units.gdf

    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.16)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    rng = np.random.default_rng(int(cfg.project.seed))
    shade = rng.random(len(gdf))
    gdf.assign(_c=shade).plot(ax=ax, column="_c", cmap="terrain", linewidth=0.05,
                              edgecolor="#ffffff", alpha=0.95)
    aoi.boundary.plot(ax=ax, color="#222", linewidth=1.4, linestyle="--")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(f"Hillslope domain H — {len(gdf):,} slope units",
                 fontsize=12, fontweight="bold", loc="left")
    for s in ax.spines.values():
        s.set_edgecolor("#bbb")

    a = gdf["area_km2"].to_numpy()
    target = float(cfg.spatial.slope_units.target_median_area_km2)
    ax2.hist(a, bins=np.logspace(np.log10(max(a.min(), 1e-3)), np.log10(a.max()), 45),
             color="#6b8f9c", edgecolor="white", linewidth=0.5)
    ax2.set_xscale("log")
    ax2.axvline(np.median(a), color="#d1341f", lw=2,
                label=f"median {np.median(a):.3f} km²")
    ax2.axvline(target, color="#222", lw=1.6, ls="--",
                label=f"Doc 05 target {target:.2f} km²")
    ax2.set_xlabel("slope unit area (km², log scale)")
    ax2.set_ylabel("units")
    ax2.set_title("Size distribution", fontsize=11, fontweight="bold", loc="left")
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    return _save(fig, "phase2_slope_units.png")


def _load_catalog():
    from ..catalog import schema
    from ..catalog.build import EVENTS_PATH
    if not EVENTS_PATH.is_file():
        raise FileNotFoundError("no catalog — run `hydrocascade catalog` first")
    df = schema.read(EVENTS_PATH)
    df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _synthetic_banner(ax, df) -> None:
    """Never let a synthetic figure be mistaken for a real one."""
    if not (df["source"] == "SYNTHETIC").any():
        return
    n = int((df["source"] == "SYNTHETIC").sum())
    ax.text(0.5, 0.5, "SYNTHETIC", transform=ax.transAxes, fontsize=54,
            color="#d1341f", alpha=0.13, ha="center", va="center",
            rotation=24, fontweight="bold", zorder=20)
    ax.text(0.99, 0.01, f"{n:,} of {len(df):,} records are synthetic stand-ins",
            transform=ax.transAxes, fontsize=7.5, color="#d1341f",
            ha="right", va="bottom", zorder=21)


def events_per_year(cfg: Node) -> Path:
    """Doc 04 §8: the events-per-year bar chart."""
    df = _load_catalog()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

    counts = (df.assign(year=df["_date"].dt.year)
                .pivot_table(index="year", columns="hazard_type", aggfunc="size",
                             fill_value=0))
    counts.plot(kind="bar", ax=axes[0], color={"L": "#a4552f", "F": WATER},
                edgecolor="white", linewidth=0.6, width=0.82)
    axes[0].set_xlabel("year"); axes[0].set_ylabel("events")
    axes[0].set_title("Events per year", fontsize=11, fontweight="bold", loc="left")
    axes[0].legend(["flood (F)", "landslide (L)"] if "F" in counts.columns else ["landslide (L)"],
                   fontsize=9)
    axes[0].tick_params(axis="x", rotation=0)

    months = (df.assign(month=df["_date"].dt.month)
                .pivot_table(index="month", columns="hazard_type", aggfunc="size",
                             fill_value=0).reindex(range(1, 13), fill_value=0))
    months.plot(kind="bar", ax=axes[1], color={"L": "#a4552f", "F": WATER},
                edgecolor="white", linewidth=0.6, width=0.82)
    axes[1].axvspan(4.5, 8.5, color="#6b8f9c", alpha=0.12, zorder=0)
    axes[1].text(6.5, axes[1].get_ylim()[1] * 0.94, "SW monsoon", ha="center",
                 fontsize=9, color="#37596b")
    axes[1].set_xlabel("month"); axes[1].set_ylabel("events")
    axes[1].set_title("Seasonality — Doc 04 §6 gate 3", fontsize=11,
                      fontweight="bold", loc="left")
    axes[1].legend(fontsize=9)
    axes[1].tick_params(axis="x", rotation=0)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    _synthetic_banner(axes[0], df)
    fig.suptitle(f"Event catalog — {len(df):,} events, "
                 f"{cfg.period.start} to {cfg.period.end}",
                 fontsize=12.5, fontweight="bold", x=0.005, ha="left")
    return _save(fig, "phase1_events_per_year.png")


def event_map(cfg: Node) -> Path:
    """Doc 04 §8: the map with landslides and flood reaches."""
    import geopandas as gpd
    import rasterio

    from ..ingest.hydrosheds import load_aoi
    from ..spatial.network import Network
    from ..spatial.terrain import Routing

    df = _load_catalog()
    rt = Routing()
    net = Network()
    aoi = load_aoi()

    with rasterio.open(paths.SPATIAL / "dem_filled.tif") as src:
        elev = src.read(1, out_shape=(src.height // 2, src.width // 2))
        t = src.transform @ src.transform.scale(2, 2)
    extent = (t.c, t.c + elev.shape[1] * t.a, t.f + elev.shape[0] * t.e, t.f)

    fig, ax = plt.subplots(figsize=(9.8, 9.8))
    ax.set_facecolor(LAND)
    _hillshade(ax, elev, extent, rt.cellsize * 2)
    net.gdf.plot(ax=ax, color=WATER, linewidth=0.3, alpha=0.55, zorder=3)
    aoi.boundary.plot(ax=ax, color="#333", linewidth=1.3, linestyle="--", zorder=4)

    pts = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    ).to_crs(net.gdf.crs)

    fl = pts[pts["hazard_type"] == "F"]
    ls = pts[pts["hazard_type"] == "L"]
    if not fl.empty:
        fl.plot(ax=ax, color=WATER, markersize=5, alpha=0.30, zorder=5,
                edgecolor="none")
    if not ls.empty:
        size = 6 + 26 * np.sqrt(
            np.nan_to_num(ls["area_m2"].to_numpy(dtype=float), nan=1e3) / 1e5)
        ls.plot(ax=ax, color="#a4552f", markersize=np.clip(size, 5, 90),
                alpha=0.72, zorder=6, edgecolor="white", linewidth=0.25)

    lm = cfg.landmarks
    for key, label in (("mundakkai_crown", "Mundakkai"),
                       ("nilambur", "Nilambur")):
        pt = gpd.GeoSeries([__import__("shapely.geometry", fromlist=["Point"]).Point(
            float(lm[key]["lon"]), float(lm[key]["lat"]))],
            crs="EPSG:4326").to_crs(net.gdf.crs).iloc[0]
        ax.scatter([pt.x], [pt.y], s=150, marker="*", color="#d1341f",
                   edgecolor="white", linewidth=1.0, zorder=8)
        ax.annotate(label, xy=(pt.x, pt.y), xytext=(11, 8),
                    textcoords="offset points", fontsize=9, fontweight="bold",
                    zorder=9, color="#7a1c10")

    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_edgecolor("#bbb")
    ax.set_title(f"Event catalog — {len(ls):,} landslides, {len(fl):,} flood events\n"
                 f"{cfg.region.id}, {cfg.period.start} to {cfg.period.end}",
                 fontsize=12, fontweight="bold", loc="left", pad=10)
    handles = [
        Line2D([], [], color="#a4552f", marker="o", ls="", ms=7, label="landslide (size ∝ √area)"),
        Line2D([], [], color=WATER, marker="o", ls="", ms=5, label="flood event (reach)"),
        Line2D([], [], color="#d1341f", marker="*", ls="", ms=13, label="landmark"),
        Line2D([], [], color="#333", ls="--", lw=1.3, label="basin boundary"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8.5, framealpha=0.93)
    _synthetic_banner(ax, df)
    _scalebar(ax, corner="lower left")
    return _save(fig, "phase1_event_map.png")


def size_frequency(cfg: Node) -> Path:
    """Doc 04 §8 / §3.4: log-log size-frequency with A_min marked."""
    from ..catalog import completeness

    df = _load_catalog()
    areas = pd.to_numeric(df.loc[df["hazard_type"] == "L", "area_m2"],
                          errors="coerce").dropna().to_numpy()
    comp = completeness.find_rollover(areas, n_bins=int(cfg.catalog.completeness.n_bins))

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ok = comp.density > 0
    ax.loglog(comp.bin_centres[ok], comp.density[ok], "o", color="#37596b",
              markersize=5.5, label="frequency density $dN/dA$")

    above = comp.bin_centres >= comp.a_min_m2
    x = comp.bin_centres[above & ok]
    if x.size > 1:
        y0 = comp.density[np.argmax(comp.bin_centres >= comp.a_min_m2)]
        ax.loglog(x, y0 * (x / x[0]) ** (-comp.alpha), "-", color="#d1341f", lw=2,
                  label=f"power law, α = {comp.alpha:.2f} ± {comp.alpha_stderr:.2f}")

    ax.axvline(comp.a_min_m2, color="#222", ls="--", lw=1.6,
               label=f"rollover $A_{{min}}$ = {comp.a_min_m2:,.0f} m²")
    # Only shade the incomplete tail if there is one. When the rollover lands on
    # the smallest bin there is nothing below it to label, and the annotation
    # would sit on top of the data.
    if comp.a_min_m2 > comp.bin_centres.min() * 1.5:
        ax.axvspan(comp.bin_centres.min(), comp.a_min_m2, color="#888", alpha=0.11)
        ax.text(np.sqrt(comp.bin_centres.min() * comp.a_min_m2),
                comp.density[ok].max() * 0.5, "systematically\nincomplete",
                ha="center", fontsize=9, color="#555")

    ax.set_xlabel("mapped landslide area (m²)")
    ax.set_ylabel("frequency density $dN/dA$")
    ax.set_title("Inventory completeness — Doc 04 §3.4\n" + comp.summary(),
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.25, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    _synthetic_banner(ax, df)
    return _save(fig, "phase1_size_frequency.png")


def render_all(cfg: Node) -> list[Path]:
    out = []
    for fn in (wayanad_flowpath, network_overview, slope_units_overview,
               events_per_year, event_map, size_frequency):
        try:
            out.append(fn(cfg))
        except FileNotFoundError as exc:
            log.warning("skipping %s: %s", fn.__name__, exc)
        except Exception as exc:  # noqa: BLE001
            log.error("%s failed: %s: %s", fn.__name__, type(exc).__name__, exc)
    return out
