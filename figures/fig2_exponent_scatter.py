#!/usr/bin/env python3
r"""fig2_exponent_scatter.py -- Figure 1 cua paper.

CAU CHUYEN CUA HINH:
    "Ba dai luong hinh hoc dau -- phang hoa metric, do lech trac dia, he so
     Rayleigh -- KHONG du bao so mu rao can. Do dai Fisher thi CO, he so goc ~1."

Tai lap dung so cua paper: R^2 = 0.20 / 0.11 / 0.02 / 0.91, he so goc 1.12.
Nguon: alpha_B tu combined_p*.csv (DA KIEM la file sinh ra so cua paper);
alpha_dev tu cot dev_rel (chia ||Delta||, dung Muc 6.2) -- KHONG phai dev_rel2.

CHAY:  python3 fig2_exponent_scatter.py
File nay CHI DOC CSV.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import style
import common as C

PANELS = [
    ("alpha_dF",  r"$\alpha_{\partial F}$",                     "metric flattening"),
    ("alpha_dev", r"$\alpha_{\mathrm{dev}_{\mathrm{rel}}}$",    "geodesic deviation"),
    ("alpha_ray", r"$\alpha_{\hat\Delta^{\top}\!F\hat\Delta}$", "Rayleigh (direction)"),
    ("alpha_len", r"$\alpha_{\Delta^{\top}\!F\Delta}$",         "Fisher length"),
]


def main():
    T = C.exponent_table().dropna(subset=["alpha_B", "alpha_dF"])

    fig = plt.figure(figsize=(style.W_FULL, 1.95))
    gs = fig.add_gridspec(1, 4)
    axes = [fig.add_subplot(gs[0, j]) for j in range(4)]
    for ax in axes[1:]:
        ax.sharey(axes[0]); ax.tick_params(labelleft=False)

    for j, (col, sym, what) in enumerate(PANELS):
        ax, s = axes[j], T.dropna(subset=[col])
        ax.axhline(0, color=style.HAIR, lw=0.6, zorder=0)

        b, a = np.polyfit(s[col], s.alpha_B, 1)
        r = float(np.corrcoef(s[col], s.alpha_B)[0, 1])
        xs = np.array([s[col].min(), s[col].max()])
        ax.plot(xs, a + b * xs, color=style.MUTED, lw=0.8, ls=(0, (3.5, 2)), zorder=1)

        for _, row in s.iterrows():
            ax.plot(row[col], row.alpha_B, marker=style.MARKER_MODE[row["mode"]],
                    color=style.COLOR_REGIME[row.regime], ms=3.8, mew=0.5,
                    markeredgecolor="white", linestyle="none", zorder=3)

        # DAN NHAN TRUC TIEP cho ba cum che do, chi lam o panel cuoi -- noi ba cum
        # nam tach bach doc duong cheo. Thu tu theo truc doc GIONG NHAU o ca bon
        # panel nen mot lan dan nhan la doc duoc het.
        # Chu co vien trang (path effect) -> dat de len du lieu van ro.
        if j == 3:
            # NTK nam sat goc trai-duoi nen dan nhan sang PHAI-DUOI; hai cum kia
            # con cho o phia TREN-TRAI.
            place = {"ntk": (+0.07, -0.24, "left"),
                     "sp":  (-0.09, +0.13, "right"),
                     "mup": (-0.09, +0.13, "right")}
            for reg in style.ORDER_REGIME:
                q = s[s.regime == reg]
                if not len(q):
                    continue
                dx, dy, ha = place[reg]
                ax.text(q[col].mean() + dx, q.alpha_B.mean() + dy,
                        style.LABEL_REGIME[reg], fontsize=7.4, ha=ha,
                        va="bottom", color=style.COLOR_REGIME[reg], zorder=6,
                        path_effects=[pe.withStroke(linewidth=2.2,
                                                    foreground="white")])

        strong = r * r > 0.5
        ax.text(0.05, 0.95, rf"$R^2\!=\!{r * r:.2f}$", transform=ax.transAxes,
                fontsize=9.0, va="top", zorder=4,
                color=style.INK if strong else style.MUTED)
        if strong:
            ax.text(0.05, 0.80, f"slope {b:.2f}", transform=ax.transAxes,
                    fontsize=6.4, va="top", color=style.MUTED)

        ax.set_xlabel(sym, labelpad=1)
        ax.set_title(what, fontsize=6.8, color=style.MUTED, pad=3)
        C.hgrid(ax, alpha=0.4)
        C.despine(ax, offset=3)
        C.panel(ax, "abcd"[j], dx=-0.02, dy=1.11)
        if j == 0:
            ax.set_ylabel(r"$\alpha_B$  (barrier; $>0$ = collapses)")

    # Chu thich chi con MARKER = kien truc: ten che do da duoc dan thang len cum.
    fig.legend(handles=C.mode_handles(), loc="outside lower center", ncol=3,
               fontsize=7.0, handletextpad=0.35, columnspacing=1.6)

    style.save(fig, "fig2_exponent_scatter")
    plt.close(fig)
    print(r"  \caption{Barrier exponent versus the exponents of local Fisher "
          r"quantities (36 cells $=$ 3 architectures $\times$ 3 parameterizations "
          r"$\times$ 4 smooth activations). Colour encodes parameterization, "
          r"marker encodes architecture. The first three quantities do not predict "
          r"the barrier; the Fisher length does, with slope $\approx 1$.}")
    for col, sym, _ in PANELS:
        s = T.dropna(subset=[col]); r = float(np.corrcoef(s[col], s.alpha_B)[0, 1])
        print(f"     {sym:38s} n={len(s):3d}  R2={r * r:.2f}")


if __name__ == "__main__":
    main()
