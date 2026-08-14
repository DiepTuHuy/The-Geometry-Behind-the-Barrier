#!/usr/bin/env python3
r"""fig1_barrier_dF.py -- Figure 2 cua paper (hinh so lieu chinh).

BON THAY DOI so voi ban cu, de so sanh truc tiep:
  1. MAU = CHE DO thay vi activation. Che do moi la bien mang ket qua cua bai;
     activation chi la bien kiem chung tinh ben. Ban cu tieu kenh manh nhat
     (mau) vao 5 duong gan nhu chong len nhau.
  2. GOP ba che do vao MOT truc thay vi ba panel -> khoang cach 3 bac giua NTK
     va muP thanh mot hinh anh duy nhat, va panel rong gap doi (2.6in thay vi 1.55in).
  3. VE DU LIEU THO: 10 cap moi o (barrier) / 5 seed moi o (dF) hien ra duoi dang
     cham mo. Ban cu bop chung thanh dai IQR alpha=0.13 -- gan nhu vo hinh khi in.
  4. KHUNG DONG KIN, tick huong vao trong (loi tap chi vat ly) thay vi truc tach roi.

CHAY:  python3 fig1_v2_barrier_dF.py
File nay CHI DOC CSV.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

import style
import common as C

ROWS = [("barrier", "LMC barrier  $B$"),
        ("dF_op",   r"$\|\partial F\|_{\mathrm{op}}$")]


def draw(mode):
    pairs = C.load_pairs(mode)
    nets = C.load_nets(mode).dropna(subset=["dF_op"])
    raw = {"barrier": pairs, "dF_op": nets}
    widths = np.array(sorted(pairs.width.dropna().unique()), float)
    rng = np.random.default_rng(11)

    fig = plt.figure(figsize=(style.W_FULL, 2.32))
    gs = fig.add_gridspec(1, 2, wspace=0.06)
    axes = [fig.add_subplot(gs[0, j]) for j in range(2)]

    for j, (key, ylab) in enumerate(ROWS):
        ax, d = axes[j], raw[key]
        entries = []
        for reg in style.ORDER_REGIME:
            col = style.COLOR_REGIME[reg]
            sub = d[d.regime == reg]

            # (i) du lieu tho: moi cap / moi seed la mot cham
            xs = sub.width.values * np.exp(rng.uniform(-.035, .035, len(sub)))
            ax.plot(xs, sub[key].values, ".", ms=1.9, color=col,
                    alpha=0.22, mew=0, zorder=2)

            # (ii) mot duong manh cho moi activation
            for a in sorted(set(sub.act)):
                m = (sub[sub.act == a].groupby("width")[key].median()
                     .reindex(widths))
                ax.plot(widths, m.values, "-", lw=0.7, color=col,
                        alpha=0.55, zorder=3)

            # (iii) duong day = trung vi gop tren moi activation
            med = sub.groupby("width")[key].median().reindex(widths)
            ax.plot(widths, med.values, marker="o", lw=2.0, ms=3.6, color=col,
                    linestyle=style.LINESTYLE_REGIME[reg],
                    mfc=col, mec="white", mew=0.7, zorder=5)

            # PHAI dung CUNG uoc luong voi fig2 va voi paper: fit alpha cho TUNG
            # activation roi lay trung vi.
            per = [C.fit_alpha(widths,
                               sub[sub.act == a].groupby("width")[key].median()
                               .reindex(widths).values)[0]
                   for a in sorted(set(sub.act))]
            per = [x for x in per if np.isfinite(x)]
            al = float(np.median(per)) if per else np.nan
            entries.append((reg, al))

        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(widths))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: C.wlabel(v)))
        ax.set_xlabel("width multiplier" if mode == "cnn" else "width")
        ax.set_ylabel(ylab)
        ax.set_xlim(widths[0] / 1.22, widths[-1] * 1.22)   # khong chua le du tru nua

        # Hop ket qua dat TRONG panel, o goc trong ma chinh du lieu tao ra.
        # loc="best" de matplotlib tu do vung it bi che nhat -> khong bao gio
        # chong len duong, va khong ton mot khoang trong nao ca.
        # (Ky thuat lay tu cac bai scaling-law: tham so fit nam trong hop nho
        #  o goc, thay vi rai nhan doc theo tung duong.)
        hs = [Line2D([], [], color=style.COLOR_REGIME[r], lw=1.8,
                     linestyle=style.LINESTYLE_REGIME[r], marker="o", ms=3.2,
                     mec="white", mew=0.6,
                     label=rf"{style.LABEL_REGIME[r]}   $\alpha={a:+.2f}$")
              for r, a in entries if np.isfinite(a)]
        ax.legend(handles=hs, loc="best", fontsize=6.9, frameon=False,
                  handlelength=2.2, handletextpad=0.6, labelspacing=0.35,
                  borderaxespad=0.8)
        C.boxed(ax)
        ax.xaxis.set_minor_locator(NullLocator())
        C.panel(ax, "ab"[j], dx=-0.01, dy=1.012)

    style.save(fig, f"fig1_barrier_dF_{mode}")
    plt.close(fig)


if __name__ == "__main__":
    for m in (sys.argv[1:] or list(C.MODES)):
        draw(m)
