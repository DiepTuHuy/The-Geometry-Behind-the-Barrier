#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_schematic.py
================
Shared primitives for the *schematic* figures of the appendix (figE, figR, figC).

Everything here works in an abstract 0-100 x 0-100 canvas, so layout arithmetic
stays readable; the figure itself is still saved at a true physical size, so the
font sizes passed in are real points on the printed page (rule R5).

Colour discipline (rule R3, survey conclusion K4): schematics must NOT use the
regime palette (NTK-lazy / Standard / muP), or a highlighted box reads as a
parameterisation.  Use C["accent"] / C["accent_fill"] instead.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from fig_style import C

INK = C["ink"]
ACCENT = C["accent"]


def new_canvas(width_in: float, height_in: float):
    """A borderless 0-100 square canvas of the given physical size."""
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_position((0.0, 0.0, 1.0, 1.0))
    fig.canvas.draw()                 # a renderer is needed to measure text
    return fig, ax


def text_width(ax, s: str, fs: float) -> float:
    """Width of `s` at font size `fs`, in canvas units. Requires a drawn canvas."""
    renderer = ax.figure.canvas.get_renderer()
    probe = ax.text(0, -500, s, fontsize=fs)
    px = probe.get_window_extent(renderer=renderer).width
    probe.remove()
    inv = ax.transData.inverted()
    return inv.transform((px, 0))[0] - inv.transform((0, 0))[0]


def box(ax, x0, x1, yc, h, lines, *, fc="white", ec=INK, lw=0.9, fs=7.0,
        title=None, title_fs=7.4, title_col=None, title_drop=3.2,
        body_drop=8.8, rounding=1.6, aspect=0.55):
    """Rounded box spanning [x0, x1], centred vertically on yc.

    With `title`, the title is set in bold at the top and `lines` below it;
    without, `lines` are centred in the box."""
    ax.add_patch(FancyBboxPatch(
        (x0, yc - h / 2), x1 - x0, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        facecolor=fc, edgecolor=ec, linewidth=lw, mutation_aspect=aspect,
        zorder=3, clip_on=False))
    xc = 0.5 * (x0 + x1)
    if title is None:
        ax.text(xc, yc, "\n".join(lines), ha="center", va="center",
                fontsize=fs, color=INK, zorder=4, linespacing=1.5)
        return
    ax.text(xc, yc + h / 2 - title_drop, title, ha="center", va="top",
            fontsize=title_fs, color=title_col or ec, weight="bold", zorder=4)
    if lines:
        ax.text(xc, yc + h / 2 - body_drop, "\n".join(lines), ha="center",
                va="top", fontsize=fs, color=INK, zorder=4, linespacing=1.5)


def arrow(ax, x0, y0, x1, y1, *, color=INK, lw=1.0, dash=None, scale=8,
          zorder=2, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=scale,
        color=color, linewidth=lw, shrinkA=0, shrinkB=0, zorder=zorder,
        linestyle=dash if dash else "solid",
        connectionstyle=f"arc3,rad={rad}"))


def chain(ax, x0, x1, y, segments, ops, *, fs_seg=7.3, fs_op=6.6,
          col_seg=INK, col_op=ACCENT, op_drop=4.6):
    """Lay `segments` out left-to-right in [x0, x1] with labelled arrows between.

    Positions come from MEASURED text extents, so an operation label always sits
    under its own arrow whatever fonts the rendering machine happens to have."""
    seg_w = [text_width(ax, s, fs_seg) for s in segments]
    gap = ((x1 - x0) - sum(seg_w)) / len(ops)
    cursor = x0
    for i, (s, w) in enumerate(zip(segments, seg_w)):
        ax.text(cursor + w / 2, y, s, ha="center", va="center",
                fontsize=fs_seg, color=col_seg, zorder=4)
        cursor += w
        if i < len(ops):
            a0, a1 = cursor + 0.20 * gap, cursor + 0.80 * gap
            arrow(ax, a0, y, a1, y, color=col_op, lw=0.9, scale=6, zorder=4)
            ax.text(0.5 * (a0 + a1), y - op_drop, ops[i], ha="center",
                    va="center", fontsize=fs_op, color=col_op, zorder=4)
            cursor += gap
