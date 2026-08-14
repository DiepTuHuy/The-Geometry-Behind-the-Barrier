#!/usr/bin/env python3
r"""fig3_rho_star.py -- Figure 4 cua paper (cau hoi gating).

CAU CHUYEN CUA HINH:
    "rho*(t) = E||p_w(x) - e_y||_2 do do 'chua khop' cua mo hinh tai trung diem
     duong noi suy. rho* nho => Fisher xap xi Hessian o do; rho* lon => khong.
     O NTK, rho* tai trung diem TANG theo width va tien toi tran sqrt(2): xap xi
     Fisher~Hessian HONG han o che do nay. O muP, rho* sap ba bac thap phan."

Vi sao ve nhu the nay:
  * Ba panel = ba kien truc, DUNG CHUNG truc doc -> ket luan lap lai duoc o ca
    ba, nhin mot phat la thay.
  * Ba cum regime tach nhau ba bac nen dan nhan THEO CUM (chu xam) thay vi
    legend: mat di thang tu duong den ten.
  * Vach ngang sqrt(2) la TRAN LY THUYET cua rho*, ve mo -- neo lai y nghia
    tuyet doi cua truc doc.

CHAY:  python3 fig3_rho_star.py
File nay CHI DOC CSV.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

import style
import common as C

RHO_MAX = np.sqrt(2.0)


def main():
    fig = plt.figure(figsize=(style.W_FULL, 2.05))
    axes, _ = C.grid_with_label_gutter(fig, 1, 3, gutter=0.30,
                                       sharex=False, sharey=True)
    axes = axes[0]

    for j, mode in enumerate(C.MODES):
        ax = axes[j]
        d = C.load_pair_final(mode)
        widths = np.array(sorted(d.width.dropna().unique()), float)
        st = C.med_iqr(d, ["regime", "act", "width"], "rho_mid")

        ax.axhline(RHO_MAX, color=style.HAIR, lw=0.7, ls=(0, (2, 2)), zorder=0)
        if j == 0:
            ax.annotate(r"$\sqrt{2}$  (upper bound on $\rho^{\ast}$)",
                        (widths[0], RHO_MAX), fontsize=6.2, color=style.MUTED,
                        va="bottom", ha="left")

        group_end = []
        for reg in style.ORDER_REGIME:
            g = st[st.regime == reg]
            for a in sorted(set(g.act)):
                s = g[g.act == a].set_index("width").reindex(widths)
                C.curve(ax, widths, s.med.values, color=style.COLOR_REGIME[reg],
                        marker=style.MARKER_MODE[mode], linestyle="-",
                        lw=0.8, ms=2.1, alpha=0.55)
            gm = g.groupby("width").med.median().reindex(widths)
            C.curve(ax, widths, gm.values, color=style.COLOR_REGIME[reg],
                    marker=style.MARKER_MODE[mode],
                    linestyle=style.LINESTYLE_REGIME[reg], lw=1.9, ms=3.4)
            med = g.groupby("width").med.median().reindex(widths)
            if med.notna().any():
                group_end.append((med.dropna().iloc[-1], style.LABEL_REGIME[reg],
                                  style.COLOR_REGIME[reg]))

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(widths))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: C.wlabel(v)))
        ax.set_xlabel("width multiplier" if mode == "cnn" else "width")
        ax.set_title(style.LABEL_MODE[mode], pad=4)
        C.hgrid(ax, alpha=0.4)
        C.despine(ax, offset=3)
        C.log_decades(ax, "y")
        C.panel(ax, "abc"[j], dx=-0.015, dy=1.0)
        if j == 0:
            ax.set_ylabel(r"$\rho^{\ast}$ at the midpoint")
        if j == 2:
            C.end_labels(ax, group_end, min_gap=0.10, fontsize=7.0)

    fig.legend(handles=C.regime_handles(dashed=True), loc="outside lower center", ncol=3,
               fontsize=7.2, handletextpad=0.4, columnspacing=1.4)

    style.save(fig, "fig3_rho_star")
    plt.close(fig)
    print(r"  \caption{Softmax residual $\rho^{\ast}=\mathbb{E}\|p_w(x)-e_y\|_2$ "
          r"at the midpoint of the matched linear interpolation, versus width. "
          r"Thin curves are per-activation medians over 10 pairs; grey labels mark "
          r"the three parameterizations. The Fisher--Hessian identification is "
          r"accurate only where $\rho^{\ast}$ is small, which holds for $\mu$P and "
          r"fails for NTK.}")


if __name__ == "__main__":
    main()
