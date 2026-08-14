#!/usr/bin/env python3
r"""fig6_deviation.py -- Muc 5.1: do lech trac dia, tu bieu dien Green den so do.

CAU CHUYEN CUA HINH:
    "Bo de 4.10 noi xi(t) = tich phan Green cua Gamma(delta,delta), nen no bang 0
     o hai dau va phinh o giua -- panel (a) cho thay dang do tren mot mo hinh do
     choi giai duoc. Panel (b) la SO DO THAT: dev_rel co theo width o ca ba kien
     truc trong NTK, va co cang nhanh khi kien truc cang phi tuyen."

Panel (a) chuyen tu fig0 sang: no minh hoa Bo de 4.10, ma bo de do nam o Muc 5.1,
cach cho mo dau muoi trang. Dat canh so do that thi cai buou do choi moi co cho
dung -- neu dung mot minh no chi noi duoc "co mot cai buou".

CHAY:  python3 fig6_deviation.py
Panel (a) tu chua (mo hinh do choi). Panel (b) CHI DOC CSV.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

import style
import common as C
from fig0_geometry_schematic import (A, B, loss, solve_geodesic,
                                     match_by_projection, C_GEO)

SMOOTH = ["gelu", "tanh", "swish", "softplus"]


def main():
    fig = plt.figure(figsize=(style.W_FULL, 2.20))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.86, 1.56])
    ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # ================================================ (a) dang xi(t) tren mo hinh do choi
    geo = solve_geodesic()
    ts = np.linspace(0, 1, 321)
    lin = A + np.outer(ts, B - A)
    xi = np.linalg.norm(match_by_projection(geo, ts) - lin, axis=1)
    i_xi = int(np.argmax(xi))

    ax0.fill_between(ts, 0, xi, color=C_GEO, alpha=0.11, lw=0)
    ax0.plot(ts, xi, color=C_GEO, lw=1.5, zorder=3)
    ax0.plot([ts[i_xi]], [xi[i_xi]], "o", ms=3.2, color=C_GEO,
             mec="white", mew=0.6, zorder=4)
    ax0.annotate(r"$\sup_t \|\xi(t)\|$", (ts[i_xi], xi[i_xi]),
                 xytext=(0, 5), textcoords="offset points",
                 fontsize=7.5, ha="center", color=style.INK)
    ax0.annotate(r"$\xi(t)=\int_0^1 G(t,s)\,\Gamma(\delta,\delta)\,\mathrm{d}s$",
                 (0.5, 0.06), xycoords="axes fraction", fontsize=7.0,
                 ha="center", va="bottom", color=style.MUTED)
    ax0.set_xlabel(r"$t$")
    ax0.set_ylabel(r"$\|\gamma_g(t)-\gamma_{\mathrm{lin}}(t)\|$", labelpad=1)
    ax0.set_xlim(0, 1); ax0.set_ylim(0, xi.max() * 1.34)
    ax0.set_xticks([0, 0.5, 1]); ax0.set_yticks([])
    ax0.spines["left"].set_visible(False)
    C.despine(ax0, offset=2, which=("bottom",))
    C.panel(ax0, "a", dx=-0.03, dy=1.01)
    ax0.set_title("Green representation (toy model)", fontsize=6.8,
                  color=style.MUTED, pad=3)

    # ================================================ (b) dev_rel DO THAT, che do NTK
    for mode in C.MODES:
        g = C.load_cell_geo(mode)
        g = g[(g.regime == "ntk") & (g.act.isin(SMOOTH))]
        widths = np.array(sorted(g.width.unique()), float)
        # CNN dung boi so 1..8 con MLP/TS dung 64..4096 -> chuan hoa ve width nho
        # nhat de ba kien truc chong len duoc. So mu BAT BIEN duoi phep doi thang nay.
        xrel = widths / widths[0]
        slopes = []
        for a in SMOOTH:
            s = g[g.act == a].set_index("width").reindex(widths)
            ax1.plot(xrel, s.dev_rel_med.values, "-",
                     marker=style.MARKER_MODE[mode], ms=2.8, lw=0.95,
                     color=style.COLOR_ACT[a], mec="white", mew=0.45, zorder=3)
            al, _, _ = C.fit_alpha(widths, s.dev_rel_med.values)
            if np.isfinite(al):
                slopes.append(-al)
        med = g.groupby("width").dev_rel_med.median().reindex(widths)
        ax1.annotate(f"{style.LABEL_MODE[mode].split(' /')[0]}  "
                     f"$n^{{{np.mean(slopes):+.2f}}}$",
                     (xrel[-1], med.iloc[-1]), xytext=(5, 0),
                     textcoords="offset points", fontsize=6.8,
                     color=style.MUTED, va="center")

    ax1.set_xscale("log", base=2); ax1.set_yscale("log")
    ticks = np.array([1, 2, 4, 8, 16, 32, 64], float)
    ax1.xaxis.set_major_locator(FixedLocator(ticks))
    ax1.xaxis.set_minor_locator(NullLocator())
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax1.set_xlim(0.78, 190)
    ax1.set_xlabel(r"width $/$ smallest width")
    ax1.set_ylabel(r"$\mathrm{dev}_{\mathrm{rel}}=\sup_t\|\xi(t)\|/\|\Delta\|$",
                   labelpad=1)
    ax1.set_title(r"measured, NTK regime", fontsize=6.8, color=style.MUTED, pad=3)
    C.hgrid(ax1, alpha=0.4); C.despine(ax1, offset=3); C.log_decades(ax1, "y")
    C.panel(ax1, "b", dx=-0.02, dy=1.01)

    fig.legend(handles=C.act_handles(SMOOTH), loc="outside lower center",
               ncol=4, fontsize=6.8, handletextpad=0.35, columnspacing=1.2)

    style.save(fig, "fig6_deviation")
    plt.close(fig)
    print(r"  \caption{Geodesic deviation. (a) On a solvable two-well toy model the "
          r"Green representation of Lemma 4.10 gives a bump that vanishes at both "
          r"endpoints. (b) The measured relative deviation in the NTK regime shrinks "
          r"with width for all four smooth activations and all three architectures; "
          r"marker shape encodes architecture.}")


if __name__ == "__main__":
    main()
