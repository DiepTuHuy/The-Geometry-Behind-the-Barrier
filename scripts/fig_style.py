#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_style.py
============
Shared figure style for the paper

    The Geometry Behind the Barrier:
    Fisher Geometry and Linear Mode Connectivity in Wide Neural Networks.

Design rules implemented here (from the figure-design literature):

  R1  One figure, one message; the takeaway is annotated on the plot,
      not hidden in the caption (Rougier et al. 2014, Rules 1-3).
  R2  Every series is keyed in a TABLE drawn inside the panel: one row per
      series, columns [line sample | name | fitted exponent].  This replaces
      the direct end-of-line labelling the figures used previously.

      The trade is deliberate and worth stating, because the earlier choice
      was not arbitrary.  Direct labelling (Cleveland & McGill 1984; Tufte
      1983, "data-ink ratio") removes the eye's round trip between a detached
      key and the curve, and it is the better default when each series carries
      one short name.  Here each series carries a name AND a fitted exponent,
      and once a third column exists the end-of-line form stops being a label
      and becomes a ragged block of text hanging off the right margin -- it
      forced a wide right-hand x-margin in every panel, it could not be
      aligned across series, and in figp4 it ran off the saved page entirely.
      A table puts the exponents in a column, which is the one arrangement
      that lets a reader compare them, and it hands the margin back to the
      data.  The cost is the round trip, and it is paid down by keeping the
      table inside the panel next to the curves rather than outside it.

      `label_at` is kept below: figD1-figD5 still use it, and it remains the
      right tool for a one-off pointer that names no quantity.
  R3  Colorblind-safe Okabe-Ito palette, with a SEMANTIC mapping that is
      IDENTICAL across every figure of the paper:
          NTK-lazy  -> blue       (#0072B2)
          Standard  -> sky blue   (#56B4E9)
          muP       -> vermillion (#D55E00)
      (Okabe & Ito 2008; Wong, Nature Methods Points of View 2011.)
  R4  A full four-sided frame around each panel, with horizontal-only
      gridlines on a very light neutral panel tint (white gridlines) for
      clean figure-ground separation (Tufte 1983; Few 2011 chartjunk debate).
      Tufte would call the closing two spines chartjunk; the counter-argument
      that decides it here is that these panels carry a tinted plotting
      region, and a tint bounded on two sides only reads as an unfinished
      shape rather than as a panel.  Closing the frame also gives the
      in-panel key (R2) an edge to sit against.
  R5  Fonts match the LaTeX Times text: STIX serif at true physical size
      (8 pt) because every figure is saved at exactly the width it is
      \\includegraphics'd at (Rougier et al. 2014, Rule 8).
  R6  Log-log scaling-law plots with decade ticks; power-law fits shown
      as thin dashed reference lines. Vector PDF with editable text.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend; safe on clusters / CI
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Physical page geometry: ICLR text block is 5.5 in wide.
# ----------------------------------------------------------------------
LINEWIDTH_IN = 5.5  # \linewidth at 100% -- figures saved at natural size

# ----------------------------------------------------------------------
# Semantic colour mapping (Okabe-Ito, colorblind-safe)
# ----------------------------------------------------------------------
C = {
    "NTK": "#0072B2",       # blue
    "Standard": "#56B4E9",  # sky blue
    "muP": "#D55E00",       # vermillion
    "ink": "#262626",       # near-black text / neutral curves
    "ref": "#8a8a8a",       # reference lines (ratio = 1, y = x)
    "band": "#000000",      # confidence bands (used with low alpha)
    # Structural accent for SCHEMATICS only (pipeline boxes, proof-dependency
    # nodes).  Deliberately Okabe-Ito bluish green, i.e. a colour that is NOT in
    # the regime mapping above, so a highlighted box can never be misread as
    # "NTK-lazy" / "Standard" / "muP".
    "accent": "#009E73",
    "accent_fill": "#E8F5F0",
    # Palette for the four TERMS of R(w_0) in App. D.2.  Okabe-Ito again, and
    # chosen disjoint from the three regime colours so a term can never be
    # misread as a parameterisation.
    "term_grad": "#E69F00",     # orange     -- endpoint gradient
    "term_fisher": "#009E73",   # bluish green -- Fisher length
    "term_resid": "#CC79A7",    # reddish purple -- fit residual
    "term_cubic": "#000000",    # black      -- third-order remainder
    "panel": "#F2F2F2",     # the shared panel tint, exposed for schematics
}
REGIMES = ["NTK", "Standard", "muP"]
REGIME_LABEL = {"NTK": "NTK-lazy", "Standard": "Standard", "muP": r"$\mu$P"}

ARCH_MARKERS = {"MLP": "o", "CNN": "s", "TS": "^"}

# Architecture palette.  Colour normally belongs to the PARAMETERISATION (R3),
# but a figure with no parameterisation dimension -- every curve in one regime --
# leaves colour free, and three hues read far better than three dashes of one
# hue.  The rule that must not be broken there is a narrower one: do not reuse
# the regime triple, or a curve is read as "NTK-lazy" / "Standard" / "muP".
# Such a figure also states its regime on the plot, so colour cannot be
# mistaken for one.
#
# No black, and nothing very dark.  Black is perfectly legitimate in general --
# it is one of the eight Okabe-Ito colours, and it is the usual choice for a
# theory or reference curve -- but here the three architectures are peers.  A
# pure-black line beside two mid-saturation ones carries far more visual weight
# and invents a hierarchy the data does not have (Cleveland & McGill on
# unintended salience).  All three are therefore mid-tone.
ARCH_COLORS = {"MLP": "#009E73",   # bluish green   (Okabe-Ito)
               "TS": "#AA3377",    # purple         (Tol bright)
               "CNN": "#E69F00"}   # amber          (Okabe-Ito)
ARCH_LABEL = {"MLP": "MLP", "CNN": "CNN", "TS": "Teacher–student"}

WIDTHS = np.array([64, 128, 256, 512, 1024, 2048, 4096])
WIDTH_TICKS = ["64", "128", "256", "512", "1k", "2k", "4k"]


# ----------------------------------------------------------------------
# Global style
# ----------------------------------------------------------------------
def apply_style() -> None:
    plt.rcParams.update({
        # R5: serif fonts that blend with the LaTeX Times text
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        # R1/R5: true physical sizes (figures are included at natural size)
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        # R4: thin, quiet axes
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4,
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "grid.color": "#ffffff",   # white gridlines pop on the tinted panel
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
        # Figure-ground separation (Few 2011 "chartjunk debate"; ggplot /
        # Economist convention): a very light neutral panel tint lifts the
        # plotting region off the page without adding real chartjunk.
        "axes.facecolor": "#F2F2F2",
        "figure.facecolor": "white",
        "legend.frameon": False,
        # R6: vector output, editable text
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def despine(ax, keep_left: bool = True) -> None:
    """R4: close the frame on all four sides, in one quiet grey.

    The name is now a misnomer -- it removed the top/right spines when the
    figures followed the open-axes convention -- but every figure script calls
    it, and renaming it would touch a dozen files to say the same thing.  The
    `keep_left` argument is likewise retained so no existing call breaks; no
    script passes it, and a four-sided frame has no use for it.
    """
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#4d4d4d")
        ax.spines[side].set_linewidth(0.7)
    # R4: horizontal-only grid
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)


def panel_letter(ax, letter: str, dx: float = -0.02, dy: float = 1.04) -> None:
    """Bold panel letter, e.g. '(a)', in axes-fraction coordinates."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="right",
            color="black")


def log_width_axis(ax, widths=WIDTHS, ticks=WIDTH_TICKS) -> None:
    """Shared x axis for scaling plots: log width with 64...4k ticks."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(widths))
    ax.set_xticklabels(ticks)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)


def powerlaw(x, amp, exponent):
    """amp * x**exponent evaluated elementwise (x array-like)."""
    return amp * np.asarray(x, dtype=float) ** exponent


def fit_line_loglog(x, y):
    """Least-squares power-law fit in log-log space -> (x, y_hat, alpha)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    alpha, log_amp = np.polyfit(np.log(x), np.log(y), 1)
    return x, np.exp(log_amp) * x ** alpha, alpha


def label_at(ax, x, y, text, color, dx=5.0, dy=0.0, fontsize=7.5,
             ha="left", weight="bold"):
    """R2: direct label placed just right of a curve's endpoint.

    `dx`/`dy` are offsets in points from the data point, so the label
    always tracks the curve regardless of axis scaling."""
    ax.annotate(text, xy=(x, y), xytext=(dx, dy),
                textcoords="offset points", xycoords="data",
                fontsize=fontsize, color=color, ha=ha, va="center",
                weight=weight, annotation_clip=False)


# ----------------------------------------------------------------------
# R2: the in-panel key, drawn as a table
# ----------------------------------------------------------------------
_TABLE_CORNER = {"upper right": (1.0, 1.0), "upper left": (0.0, 1.0),
                 "lower right": (1.0, 0.0), "lower left": (0.0, 0.0)}


def _axes_text_size(ax, s, fontsize, weight="normal"):
    """Size of `s` as drawn on `ax`, in axes-fraction units.

    Measured rather than estimated, because the third column holds mathtext
    exponents whose width no character count predicts."""
    renderer = ax.figure.canvas.get_renderer()
    probe = ax.text(0.0, 0.0, s, transform=ax.transAxes, fontsize=fontsize,
                    weight=weight)
    bb = probe.get_window_extent(renderer=renderer)
    probe.remove()
    (x0, y0), (x1, y1) = ax.transAxes.inverted().transform(
        [(0.0, 0.0), (bb.width, bb.height)])
    return x1 - x0, y1 - y0


def legend_table(ax, rows, loc="upper right", title=None, fontsize=6.6,
                 margin=0.030, pad=0.028, gap=0.038, sample=0.10,
                 facecolor="white", alpha=0.90, zorder=20):
    """R2: draw this panel's key as a table -- sample | name | value.

    `rows` is a sequence of ``(style, name, value)``:

        style   a dict of Line2D properties (color, ls, marker, lw, ms).  Pass
                the same dict the curve was drawn with, so the key cannot
                drift away from the plot it describes.
        name    the series name; mathtext allowed.
        value   the fitted quantity for that series, or None to leave the
                third column empty for that row.

    Column widths come from the rendered text (`_axes_text_size`), so the
    three columns line up whatever the labels say and however long an exponent
    prints.  Everything is placed in axes fractions, so the table travels with
    the panel and not with the data -- rescaling an axis cannot strand it, and
    a caller may therefore set `ylim` after calling this.

    Returns the table's ``(x0, y0, w, h)`` in axes fractions, so a caller that
    needs to keep a curve clear of it can query where it landed."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    ax.figure.canvas.draw()   # a renderer must exist before text can be measured

    names = [r[1] for r in rows]
    values = [r[2] for r in rows]
    has_value = any(v is not None for v in values)

    name_w = max(_axes_text_size(ax, n, fontsize, "bold")[0] for n in names)
    line_h = max(_axes_text_size(ax, n, fontsize, "bold")[1] for n in names)
    val_w = max([_axes_text_size(ax, v, fontsize)[0]
                 for v in values if v is not None], default=0.0)
    row_h = 1.42 * line_h

    inner_w = sample + gap + name_w + (gap + val_w if has_value else 0.0)
    title_h = 0.0
    if title is not None:
        title_w, th = _axes_text_size(ax, title, fontsize, "bold")
        inner_w = max(inner_w, title_w)
        title_h = 1.62 * th

    w = inner_w + 2 * pad
    h = len(rows) * row_h + title_h + 2 * pad

    corner_x, corner_y = _TABLE_CORNER[loc]
    x0 = margin if corner_x == 0.0 else 1.0 - margin - w
    y0 = margin if corner_y == 0.0 else 1.0 - margin - h

    ax.add_artist(Rectangle((x0, y0), w, h, transform=ax.transAxes,
                            facecolor=facecolor, alpha=alpha,
                            edgecolor="#c8c8c8", linewidth=0.6,
                            zorder=zorder, clip_on=False))

    cursor = y0 + h - pad
    if title is not None:
        ax.text(x0 + pad, cursor - 0.5 * title_h, title,
                transform=ax.transAxes, fontsize=fontsize, weight="bold",
                color=C["ink"], ha="left", va="center", zorder=zorder + 2)
        cursor -= title_h
        ax.add_artist(Line2D([x0 + pad, x0 + w - pad], [cursor, cursor],
                             transform=ax.transAxes, color="#c8c8c8",
                             linewidth=0.6, zorder=zorder + 1, clip_on=False))

    x_sample = x0 + pad
    x_name = x_sample + sample + gap
    x_value = x0 + w - pad
    for i, (style, name, value) in enumerate(rows):
        yc = cursor - (i + 0.5) * row_h
        spec = dict(style)
        colour = spec.pop("color", C["ink"])
        marker = spec.pop("marker", None)
        ax.add_artist(Line2D([x_sample, x_sample + 0.5 * sample,
                              x_sample + sample], [yc] * 3,
                             transform=ax.transAxes, color=colour,
                             marker=marker, markevery=[1],
                             markerfacecolor=colour, markeredgecolor="white",
                             markeredgewidth=0.5, zorder=zorder + 2,
                             clip_on=False, **spec))
        ax.text(x_name, yc, name, transform=ax.transAxes, fontsize=fontsize,
                color=colour, ha="left", va="center", weight="bold",
                zorder=zorder + 2)
        if value is not None:
            ax.text(x_value, yc, value, transform=ax.transAxes,
                    fontsize=fontsize, color=colour, ha="right", va="center",
                    zorder=zorder + 2)
    return x0, y0, w, h


def headroom_for(lo, hi, table_h, margin=0.030, gap=0.035):
    """Top `ylim` that leaves a key of height `table_h` clear of the data.

    On a log axis the data's top sits at a fixed fraction of the panel; this
    returns the `ymax` that puts that fraction just under the table's bottom
    edge.  `table_h` is the third element of what `legend_table` returned, so
    the room made is the room actually needed rather than a guessed constant.
    """
    frac = 1.0 - margin - table_h - gap
    lo_l, hi_l = np.log10(lo), np.log10(hi)
    return 10.0 ** (lo_l + (hi_l - lo_l) / frac)



# Hairline kept between the drawn content and the page edge, so a stroke on the
# outermost artist is never shaved off by the crop.
_PAGE_PAD_IN = 0.02


def _fit_width(fig, target_w_in, rounds=4):
    """Widen (or narrow) the axes column so the drawn content spans the canvas.

    Matplotlib's default subplot margins leave roughly 6-15% of the canvas
    empty on each side.  Combined with saving at a fixed width -- which R5
    requires -- that empty margin is simply a smaller figure on the page for no
    benefit.  Reclaiming it is iterative, because moving the axes also moves the
    tick labels and the axis label that are being measured, so a few rounds are
    run and the loop stops as soon as it converges."""
    renderer = fig.canvas.get_renderer()
    for _ in range(rounds):
        fig.canvas.draw()
        tb = fig.get_tightbbox(renderer)
        left_gap = (tb.x0 - _PAGE_PAD_IN) / target_w_in
        right_gap = (target_w_in - _PAGE_PAD_IN - tb.x1) / target_w_in
        if abs(left_gap) < 0.004 and abs(right_gap) < 0.004:
            return
        sp = fig.subplotpars
        new_left = min(max(sp.left - left_gap, 0.0), 0.9)
        new_right = min(max(sp.right + right_gap, new_left + 0.05), 1.0)
        fig.subplots_adjust(left=new_left, right=new_right)


def settle(fig):
    """Run save()'s axes-widening pass early and redraw.

    Any artist whose orientation is measured off the axes -- a label rotated to
    lie along a data line, say -- has to be positioned against the FINAL axes
    box.  `save` widens the axes as its first step, which moves that box, so a
    rotation computed before then is wrong by however much the widening moved
    things.  Call this first, place the artist, then call `save`: by then the
    content already fills the canvas, so `save` leaves the geometry alone."""
    _fit_width(fig, fig.get_size_inches()[0])
    fig.canvas.draw()


def data_angle(ax, p0, p1):
    """Screen angle in degrees of the data segment p0 -> p1.

    On a log--log panel whose two axes span different numbers of decades, the
    line y = x is NOT at 45 degrees on the page, and a label rotated by a
    hand-written angle drifts away from the line it annotates.  This measures
    the angle the reader actually sees.  Valid only once the axes box is
    final -- see `settle`."""
    (x0, y0), (x1, y1) = ax.transData.transform([p0, p1])
    return float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))


def save(fig, out_dir, name):
    """Write vector PDF (for LaTeX) + PNG preview, at EXACTLY `figsize`.

    R5 requires each figure to be saved at the width it is \\includegraphics'd
    at, so that a font declared at 8 pt prints at 8 pt.  Plain
    ``bbox_inches="tight"`` breaks that: it crops the declared margins away, so
    every figure lands at a different width and LaTeX then rescales each one by
    a different factor.  Measured over this paper's thirteen figures before the
    fix, a declared 8 pt rendered anywhere from 7.8 pt to 9.6 pt -- a 23% spread
    that reads as inconsistency even when a reader cannot name it.

    So: measure the tight bounding box (which may legitimately extend past the
    canvas, since direct labels are drawn with ``annotation_clip=False``), then
    re-centre it and pad it back out to exactly the declared figure size.  The
    saved page is then always `figsize`, the scale factor in LaTeX is always
    1.000, and no content is lost.  If the content genuinely does not fit, say
    so rather than silently cropping or silently rescaling."""
    from matplotlib.transforms import Bbox

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"

    W = fig.get_size_inches()[0]
    fig.canvas.draw()
    if fig.get_tightbbox(fig.canvas.get_renderer()).width < 0.97 * W:
        _fit_width(fig, W)

    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.008)
    tw = tight.width

    # Only the WIDTH matters: \\includegraphics[width=...] scales by width and
    # lets the height follow.  So pad the width out to exactly the declared
    # figsize width and leave the height at whatever the content needs.
    if tw > W + 0.04:      # sub-0.04 in overhang is stroke width, not content
        print(f"[warn] {name}: content is {tw:.2f} in wide but figsize declares "
              f"{W:.2f} in -- pull the overflowing artist inside; saving tight, "
              f"so LaTeX WILL rescale this one")
        bbox = tight
    else:
        cx = 0.5 * (tight.x0 + tight.x1)
        bbox = Bbox.from_extents(cx - W / 2, tight.y0, cx + W / 2, tight.y1)
        fill = tw / W
        if fill < 0.955:
            print(f"[thin] {name}: content fills only {100*fill:.0f}% of the "
                  f"{W:.2f} in canvas -- widen the axes (subplots_adjust) so "
                  f"the figure is not needlessly small on the page")

    fig.savefig(pdf_path, bbox_inches=bbox, pad_inches=0.0)
    fig.savefig(png_path, bbox_inches=bbox, pad_inches=0.0, dpi=220)
    plt.close(fig)
    print(f"[ok] wrote {pdf_path}  ({bbox.width:.2f} x {bbox.height:.2f} in)")

