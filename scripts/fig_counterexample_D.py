#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_counterexample_D.py -- Appendix D.1 counterexample figure (restyled).

Analytic construction (no experimental data):
    F_eps(w) = M_F * P_perp + eps * Delta_hat Delta_hat^T   (constant in w)
=>  partial_z F = 0  =>  Gamma = 0, Sec = 0  (flattening holds trivially),
while Q_F(w; Delta) = eps ||Delta||^2 sweeps [0, M_F ||Delta||^2].

Restyle notes (see figures/SURVEY.md):
  * legend moved BELOW panel (a) -- it no longer covers the needle-ellipse,
    the panel's main object (R1/R4);
  * boxed annotations removed; objects are labelled directly (R2);
  * STIX serif fonts matching the LaTeX text, true physical size (R5).

Usage:  python3 scripts/fig_counterexample_D.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend; safe on clusters / CI
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from fig_style import apply_style, despine, panel_letter, save, C

OUT = Path(__file__).resolve().parent.parent / "figures" / "appendix"

# Paper constants (keep in sync with appendix D.1)
M_F = 1.0            # operator-norm cap (Assumption ass:fisher_bound)
EPS_DRAW = 0.04      # eps for panel (a); true 1e-8 needle invisible in print
NORM_DELTA = 1.0     # ||Delta||_2 used to normalise panel (b)

C_CONTOUR = "#7fa8d9"   # level sets of the metric
C_NEEDLE = C["muP"]     # image of unit sphere under F^{1/2}
C_UNIT = C["ref"]       # Euclidean unit circle (reference)
C_DELTA = "#111111"     # displacement vector
C_FLAT = "#009E73"      # flattening quantities (Gamma, Sec)


def metric_quadratic_form(xx, yy, eps, m_f=M_F):
    """w^T F_eps w in the basis (Delta_hat, u_perp): diag(eps, M_F)."""
    return eps * xx**2 + m_f * yy**2


def draw_panel_a(ax, eps=EPS_DRAW):
    t = np.linspace(0.0, 2.0 * np.pi, 400)
    ct, st = np.cos(t), np.sin(t)

    # level sets of the quadratic form (light blue family)
    levels = eps * np.array([0.15, 0.35, 0.65, 1.0, 1.5, 2.2, 3.2, 4.8])
    XX, YY = np.meshgrid(np.linspace(-1.35, 1.35, 400),
                         np.linspace(-1.35, 1.35, 400))
    ax.contour(XX, YY, metric_quadratic_form(XX, YY, eps), levels=levels,
               colors=[C_CONTOUR], linewidths=0.7, alpha=0.75)

    # Euclidean unit sphere and its image under F_eps^{1/2}
    ax.plot(ct, st, ls="--", lw=1.0, color=C_UNIT, zorder=3)
    ax.plot(np.sqrt(eps) * ct, np.sqrt(M_F) * st, lw=2.0, color=C_NEEDLE,
            zorder=4)

    # Delta along the near-null direction
    ax.add_patch(FancyArrowPatch((0, 0), (NORM_DELTA, 0),
                                 arrowstyle="-|>", mutation_scale=9,
                                 lw=1.6, color=C_DELTA, zorder=6))
    ax.text(0.98, -0.13, r"$\widehat{\Delta}$", fontsize=9,
            color=C_DELTA, ha="right")

    # R2: direct labels next to the objects they describe
    ax.text(0.35, 0.93, "Euclidean unit sphere", fontsize=6.8,
            color=C_UNIT, rotation=-36)
    ax.text(-1.28, -1.13,
            r"$F_\varepsilon^{1/2}$(unit sphere):"
            "\nsemi-axes $\\sqrt{\\varepsilon},\\ \\sqrt{M_F}$",
            fontsize=6.8, color=C_NEEDLE)
    ax.text(-1.28, 1.16,
            r"$\partial_z F \equiv 0 \Rightarrow \Gamma \equiv 0,"
            r"\ \mathrm{Sec} \equiv 0$", fontsize=6.8, color=C_FLAT)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.set_xlabel(r"direction of $\widehat{\Delta}$")
    ax.set_ylabel(r"direction of $u_\perp$")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.grid(False)
    panel_letter(ax, "a", dx=-0.02)
    despine(ax)
    ax.grid(False)  # equal-aspect geometry panel: no grid


def draw_panel_b(ax, norm_delta=NORM_DELTA, m_f=M_F):
    eps_grid = np.linspace(0.0, m_f, 400)

    # Fisher length sweeps [0, ||Delta||^2] as eps varies.
    q_f = eps_grid * norm_delta**2
    ax.plot(eps_grid, q_f, lw=1.8, color=C_DELTA,
            label=r"Fisher length $Q_F = \varepsilon\|\Delta\|_2^2$")

    # Flattening quantities are identically zero for EVERY eps.
    ax.plot(eps_grid, np.zeros_like(eps_grid), ls="--", lw=1.8,
            color=C_FLAT,
            label=(r"$\|\partial_z F\|_{\mathrm{op}} = 0 \Rightarrow "
                   r"\Gamma \equiv 0,\ \mathrm{Sec} \equiv 0$"))

    # Instances named in the text.
    ax.scatter([0.0], [0.0], marker="*", s=140, color=C_NEEDLE, zorder=6,
               label=r"$\varepsilon = 0$: Prop. D.1 ($Q_F = 0$)")
    ax.scatter([1e-8], [1e-8 * norm_delta**2], marker="o", s=30,
               facecolor="white", edgecolor=C_NEEDLE, linewidth=1.2,
               zorder=6, label=r"$\varepsilon = 10^{-8}$: $P{=}2$ example")

    ax.annotate("no positive lower bound:\nany value in "
                r"$[0,\,M_F\|\Delta\|_2^2]$ is attainable",
                xy=(0.45, 0.45), xytext=(0.03, 0.78),
                fontsize=7.5, color=C_NEEDLE,
                arrowprops=dict(arrowstyle="->", color=C_NEEDLE, lw=0.9))

    ax.set_xlim(-0.03, m_f)
    ax.set_ylim(-0.08 * m_f * norm_delta**2, 1.10 * m_f * norm_delta**2)
    ax.set_xlabel(r"spectral value $\varepsilon$ along $\widehat{\Delta}$")
    ax.set_ylabel("quantity value (normalised)")
    # R2/R4: frameless legend in the empty lower-right triangle (below the
    # diagonal, above the zero line); short labels keep it clear of both.
    ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.115), fontsize=6.2,
              handlelength=1.6, borderaxespad=0.2)
    panel_letter(ax, "b", dx=-0.02)
    despine(ax)


def main():
    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(5.5, 2.7),
        gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.30})

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)

    # NOTE: tight_layout is incompatible with the equal-aspect panel (a);
    # spacing handled by wspace + bbox_inches="tight" at save time.
    save(fig, OUT, "fig_counterexample_D")
    print(f"     check: Q_F(eps=0) = {0.0 * NORM_DELTA**2:.3e}"
          "  (= 0 exactly, Prop. D.1)")
    print(f"     check: Q_F(eps=1e-8) = {1e-8 * NORM_DELTA**2:.3e}"
          "  (P=2 illustration)")


if __name__ == "__main__":
    main()

