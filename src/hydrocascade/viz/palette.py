"""The project's figure palette — validated, not chosen by eye.

Every categorical colour below was run through the six checks (lightness band,
chroma floor, colour-vision-deficiency separation, normal-vision separation,
contrast against the surface) rather than picked because it looked right. The
result of the check is recorded here so the next person does not have to redo
it, and so a casual "let's make that one greener" is visibly a decision that
invalidates a test.

Result for ``CATEGORICAL`` on a light surface::

    lightness band      PASS  all 3 inside L 0.43-0.77
    chroma floor        PASS  all 3 >= 0.1
    CVD separation      PASS  worst adjacent dE 21.2 (tritan), 21.3 (protan)
    normal-vision floor PASS  worst adjacent dE 27.0
    contrast vs surface PASS  all 3 >= 3:1

Two earlier candidates failed and are recorded so they are not retried:

* ``#3b7ea1`` — the blue used in the Phase 1-2 figures. Chroma 0.087, below the
  0.1 floor: it reads as grey rather than as a colour. Replaced by ``#1668b0``.
* ``#8a5bc4`` as a third hue — dE 3.3 against the blue under protanopia and
  14.4 even with normal colour vision. Two of the three series would have been
  indistinguishable for roughly one man in twelve.

**Identity is never carried by colour alone.** Any figure with two or more
series carries a legend, and four or fewer are also labelled directly.
"""

from __future__ import annotations

# Categorical — assign in this fixed order, never cycled. A fourth series means
# faceting or an "other" bucket, not a fourth hue invented on the spot.
CATEGORICAL = ("#d1341f", "#1668b0", "#a8710f")

ACCENT = CATEGORICAL[0]      # landslides, the event, the thing being predicted
WATER = CATEGORICAL[1]       # floods, rivers, discharge
THIRD = CATEGORICAL[2]       # a third series when one is genuinely needed

# Sequential — one hue, light to dark, for magnitude. Built from the water blue
# so a susceptibility map and a discharge series read as the same family.
SEQUENTIAL = ("#eef4fa", "#cfe0ef", "#a8c6e0", "#7aa6cd", "#4b83ba", "#1668b0",
              "#0f4d84")

# Diverging — two poles and a neutral grey midpoint, for signed quantities such
# as a SHAP contribution or a residual. Never a hue at the midpoint.
DIVERGING = ("#1668b0", "#8fb4d6", "#d9d9d6", "#e39b8c", "#d1341f")

# Ink. Text wears text tokens, never the series colour.
INK = "#1a1a1a"
INK_SECONDARY = "#4a4a4a"
INK_MUTED = "#767670"
GRID = "#e0e0dc"
SURFACE = "#fcfcfb"

# Status — reserved, never reused as "series 4", and always shipped with a word
# rather than relied on as colour alone.
STATUS = {"good": "#1a7f37", "warning": "#a8710f",
          "serious": "#c4471f", "critical": "#a01b1b"}


def sequential_cmap(name: str = "hydro_seq"):
    """The sequential ramp as a matplotlib colormap."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, list(SEQUENTIAL))


def diverging_cmap(name: str = "hydro_div"):
    """The diverging ramp as a matplotlib colormap."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, list(DIVERGING))


def apply_style() -> None:
    """Recessive grid and axes, ink-coloured text, no chartjunk."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "text.color": INK,
        "legend.frameon": False,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
