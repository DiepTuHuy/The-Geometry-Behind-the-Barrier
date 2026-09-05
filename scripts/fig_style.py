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
  R2  Direct labelling at line ends instead of detached legends
      (Cleveland & McGill 1984; Tufte 1983 - "data-ink ratio").
  R3  Colorblind-safe Okabe-Ito palette, with a SEMANTIC mapping that is
      IDENTICAL across every figure of the paper:
          NTK-lazy  -> blue       (#0072B2)
          Standard  -> sky blue   (#56B4E9)
          muP       -> vermillion (#D55E00)
      (Okabe & Ito 2008; Wong, Nature Methods Points of View 2011.)
  R4  No chartjunk: top/right spines removed, horizontal-only gridlines
      on a very light neutral panel tint (white gridlines) for clean
      figure-ground separation (Tufte 1983; Few 2011 chartjunk debate).
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
    """R4: remove top/right spines; keep quiet left/bottom spines."""
    ax.spines[["top", "right"]].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
    ax.spines["left"].set_color("#4d4d4d")
    ax.spines["bottom"].set_color("#4d4d4d")
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

