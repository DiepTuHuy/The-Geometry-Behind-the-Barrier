#!/usr/bin/env python3
r"""fig4_rho_gate.py -- Figure 2 cua paper: du luong Fisher-Hessian la CAI CONG.

CAU CHUYEN CUA HINH:
    "Xap xi bac hai B ~ (1/8) Delta^T F Delta dung CHINH XAC o nhung o co rho*
     nho, va sai ca bac o nhung o co rho* lon. Nguong khong phai la che do tham
     so hoa -- ma la du luong Fisher-Hessian tai diem lay (He qua 2.6)."

R = B / [(1/8) Delta^T F Delta].  Vi flen = (1/2) Delta^T F Delta trong CSV nen
R = 4B/flen (xem C.R_FROM_FLEN). He so 1/8 la du bao bac hai cua Singh et al.
tai trung diem: B ~ [alpha(1-alpha)/2] Delta^T Hess Delta voi alpha = 1/2.

CHAY:  python3 fig4_rho_gate.py
File nay CHI DOC CSV.
"""
import numpy as np
import matplotlib.pyplot as plt

import style
import common as C

RHO_MAX = np.sqrt(2.0)


def main():
    T = C.cell_table()
    fig = plt.figure(figsize=(style.W_FULL, 2.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])
    ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # ============================================ (a) cong rho*
    ax0.axhspan(0.5, 2.0, color=style.HAIR, alpha=0.45, lw=0, zorder=0)
    ax0.axhline(1.0, color=style.MUTED, lw=0.7, zorder=1)
    ax0.axvline(0.3, color=style.HAIR, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for _, r in T.iterrows():
        ax0.plot(r.rho_mid, r.R_mid, marker=style.MARKER_MODE[r["mode"]],
                 color=style.COLOR_REGIME[r.regime], ms=3.2, mew=0.45,
                 markeredgecolor="white", linestyle="none", alpha=0.9, zorder=3)
    ax0.set_yscale("log")
    lo = T[T.rho_mid < 0.3].R_mid.dropna()
    hi = T[T.rho_mid > 0.5].R_mid.dropna()
    ax0.text(0.03, 0.055, f"$\\rho^\\ast\\!<\\!0.3$:  median $R={lo.median():.2f}$"
             f"   ({len(lo)} cells)", transform=ax0.transAxes,
             fontsize=6.6, color=style.INK)
    ax0.text(0.52, 0.94, f"$\\rho^\\ast\\!>\\!0.5$:  median $R={hi.median():.2f}$"
             f"   ({len(hi)} cells)", transform=ax0.transAxes,
             fontsize=6.6, color=style.INK, va="top")
    # nhan dat TRONG dai xam -- dai xam la vung du bao DUNG, khong phai vung sai
    ax0.text(0.985, 0.62, r"within $2\times$ of the"  "\n" r"quadratic prediction",
             transform=ax0.get_yaxis_transform(), fontsize=6.2,
             color=style.MUTED, ha="right", va="center")
    ax0.set_xlabel(r"$\rho^{\ast}$ at the midpoint   (scale $[0,\sqrt{2}\,]$)")
    ax0.set_ylabel(r"$R=B\,/\,[\frac{1}{8}\Delta^{\top}\!F(w_{1/2})\Delta]$")
    C.hgrid(ax0, alpha=0.35); C.despine(ax0, offset=3); C.log_decades(ax0, "y")
    C.panel(ax0, "a", dx=-0.015, dy=1.01)

    # ============================================ (b) diem neo dung cho tung che do
    ax1.axhspan(0.5, 2.0, color=style.HAIR, alpha=0.45, lw=0, zorder=0)
    ax1.axhline(1.0, color=style.MUTED, lw=0.7, zorder=1)
    rng = np.random.default_rng(3)
    pos, labels = [], []
    for i, reg in enumerate(style.ORDER_REGIME):
        s = T[T.regime == reg]
        for k, (col, anchor) in enumerate((("R_end", "end"), ("R_mid", "mid"))):
            x = i * 2.6 + k                      # hai cot canh nhau moi che do
            v = s[col].dropna()
            ax1.plot(x + rng.uniform(-.20, .20, len(v)), v, ".", ms=2.2,
                     color=style.MUTED, alpha=0.35, zorder=2)
            ax1.plot([x - .34, x + .34], [v.median()] * 2, "-", lw=1.8,
                     color=style.COLOR_REGIME[reg], zorder=4,
                     solid_capstyle="butt")
            ax1.annotate(f"{v.median():.2f}", (x, v.median()), fontsize=6.2,
                         color=style.COLOR_REGIME[reg], ha="center", va="bottom",
                         xytext=(0, 3), textcoords="offset points", zorder=5)
            pos.append(x); labels.append(anchor)
        ax1.annotate(style.LABEL_REGIME[reg], (i * 2.6 + 0.5, 1.0),
                     xycoords=("data", "axes fraction"), fontsize=7.2,
                     color=style.COLOR_REGIME[reg], ha="center", va="bottom")
    ax1.set_yscale("log")
    ax1.set_xticks(pos); ax1.set_xticklabels(labels, fontsize=6.4)
    ax1.set_xlabel("anchor point where $F$ is probed", labelpad=1)
    ax1.set_ylabel(r"$R$")
    ax1.set_xlim(-0.8, 2 * 2.6 + 1.8)
    C.hgrid(ax1, alpha=0.35); C.despine(ax1, offset=3); C.log_decades(ax1, "y")
    C.panel(ax1, "b", dx=-0.04, dy=1.01)

    fig.legend(handles=C.regime_handles() + C.mode_handles(),
               loc="outside lower center", ncol=6,
               fontsize=6.8, handletextpad=0.35, columnspacing=1.1)

    style.save(fig, "fig4_rho_gate")
    plt.close(fig)
    print(r"  \caption{The Fisher--Hessian residual is the gate. (a) One point per "
          r"cell (216 cells); the quadratic prediction is accurate (grey band = "
          r"within $2\times$) exactly where $\rho^{\ast}$ is small. (b) The correct "
          r"anchor is set by $\rho^{\ast}$, not by the parameterization: in NTK the "
          r"endpoints are the valid probe, in feature learning the midpoint is. "
          r"Bars are medians.}")
    print("     R_mid by regime:", T.groupby("regime").R_mid.median().round(2).to_dict())
    print("     R_end by regime:", T.groupby("regime").R_end.median().round(2).to_dict())


if __name__ == "__main__":
    main()
