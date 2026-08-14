#!/usr/bin/env python3
r"""fig5_path_profile.py -- Do cong doc duong noi suy (kieu hinh MEP/AutoNEB).

CAU CHUYEN CUA HINH:
    "Do cong Fisher doc duong di HAI CHE DO TRAI NGUOC NHAU. O feature-learning
     (standard/muP), do cong DANG len o giua duong -- dung dang 'entropic
     barrier' quen thuoc. O NTK-lazy no SAP xuong ~10 lan tai chinh trung diem,
     noi rao can cao nhat. Fisher khong 'thay' rao can vi tai do tien doan gan
     nhu deu (rho* -> 0.78) nen khong con do nhay de do (Muc 5.3)."

KHAC voi hinh MEP cua Draxler / Entropic-Confinement: duong o day la duong NOI
SUY TUYEN TINH gamma_lin (sau can chinh hoan vi) -- chinh la doi tuong cua paper
nay -- chu khong phai duong MEP tim boi AutoNEB. Truc hoanh vi vay la t, khong
phai chi so pivot chuan hoa.

BA MOC t: du lieu luu san chi co t in {0, 0.5, 1} (cot flen_A / flen_mid /
flen_B). Bo may do DA tinh Gamma tren luoi 9 diem va L tren luoi 41 diem nhung
chi ghi ra gia tri tom tat; muon duong lien tuc thi chi can ghi them cac moc do
ra CSV, khong phai do lai.

CHAY:  python3 fig5_path_profile.py           # ca 3 kien truc
File nay CHI DOC CSV.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

import style
import common as C

TS = np.array([0.0, 0.5, 1.0])


def draw(mode):
    g = C.load_geo_pairs(mode)
    cells = (g.groupby(["regime", "act", "width"])[["flen_A", "flen_mid", "flen_B"]]
             .median().reset_index())
    cells["flen_end"] = 0.5 * (cells.flen_A + cells.flen_B)
    for c in ("flen_A", "flen_mid", "flen_B"):
        cells[c + "_n"] = cells[c] / cells.flen_end
    widths = np.array(sorted(cells.width.unique()), float)
    wmax = widths[-1]
    acts = [a for a in style.ORDER_ACT if a in set(cells.act)]

    # hai hang co Y NGHIA TRUC X KHAC NHAU (t doc duong vs width) nen KHONG
    # duoc chia se truc x giua hai hang; chi chia se trong tung hang.
    fig = plt.figure(figsize=(style.W_FULL, 3.15))
    axes, _ = C.grid_with_label_gutter(fig, 2, 3, gutter=0.31,
                                       sharex=False, sharey=False)
    for j in (1, 2):
        for i in (0, 1):
            axes[i, j].sharey(axes[i, 0]); axes[i, j].tick_params(labelleft=False)
            axes[i, j].sharex(axes[i, 0])

    for j, reg in enumerate(style.ORDER_REGIME):
        # ---------- hang 1: profile do cong tai 3 moc, o width lon nhat ----------
        ax = axes[0, j]
        ax.axhline(1.0, color=style.HAIR, lw=0.7, zorder=0)
        ends = []
        for a in acts:
            s = cells[(cells.regime == reg) & (cells.act == a) & (cells.width == wmax)]
            if not len(s):
                continue
            y = s[["flen_A_n", "flen_mid_n", "flen_B_n"]].values[0]
            C.curve(ax, TS, y, color=style.COLOR_REGIME[reg],
                    marker=style.MARKER_MODE[mode], linestyle="-", ms=3.4)
        ax.set_yscale("log")
        ax.set_xlim(-0.08, 1.08); ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["$\\theta_A$", "mid", "$\\theta_B$"])
        ax.set_title(style.LABEL_REGIME[reg], pad=4)
        C.hgrid(ax, alpha=0.4); C.despine(ax, offset=3); C.log_decades(ax, "y")
        C.panel(ax, "abc"[j], dx=-0.015, dy=1.0)
        if j == 0:
            ax.set_ylabel("Fisher curvature\n"
                          r"$\Delta^{\top}\!F(\gamma(t))\Delta$ / endpoints")
        # Ghi ti so tai WIDTH LON NHAT, khong phai trung vi tren moi width:
        # o CNN/NTK ti so bat qua 1 khi width tang (1.76 -> 0.79) nen trung vi
        # ~0.98 se bi doc nham thanh "khong doi".
        q = cells[(cells.regime == reg) & (cells.width == wmax)]
        rat = float((q.flen_mid / q.flen_end).median())
        arrow = r"$\downarrow$" if rat < 1 else r"$\uparrow$"
        ax.text(0.04, 0.95, rf"mid/end $=${rat:.2f} {arrow}  @ max width",
                transform=ax.transAxes, fontsize=6.8, color=style.INK, va="top")

        # ---------- hang 2: ti so mid/end thay doi the nao theo width ----------
        ax = axes[1, j]
        ax.axhline(1.0, color=style.HAIR, lw=0.7, zorder=0)
        for a in acts:
            s = (cells[(cells.regime == reg) & (cells.act == a)]
                 .set_index("width").reindex(widths))
            C.curve(ax, widths, (s.flen_mid / s.flen_end).values,
                    color=style.COLOR_REGIME[reg],
                    marker=style.MARKER_MODE[mode], linestyle="-", ms=2.6)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(widths))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: C.wlabel(v)))
        ax.set_xlabel("width multiplier" if mode == "cnn" else "width")
        C.hgrid(ax, alpha=0.4); C.despine(ax, offset=3); C.log_decades(ax, "y")
        C.panel(ax, "def"[j], dx=-0.015, dy=1.0)
        if j == 0:
            ax.set_ylabel("curvature ratio\nmid / endpoints")
        if j == 2:
            pass

    fig.legend(handles=C.regime_handles(), loc="outside lower center", ncol=3,
               fontsize=7.2, handletextpad=0.4, columnspacing=1.4)
    style.save(fig, f"fig5_path_profile_{mode}")
    plt.close(fig)
    r = (cells.flen_mid / cells.flen_end).groupby(cells.regime).median()
    print(f"     [{mode}] curvature ratio mid/end: " +
          ", ".join(f"{k}={v:.2f}" for k, v in r.items()))


if __name__ == "__main__":
    print(r"  \caption{Fisher curvature along the matched linear interpolation, "
          r"probed at the two endpoints and the midpoint. Top: profile at the "
          r"largest width, normalised by the endpoint average. Bottom: how the "
          r"midpoint-to-endpoint ratio evolves with width. In feature learning the "
          r"curvature rises in the interior; under NTK it collapses by an order of "
          r"magnitude at exactly the point where the barrier peaks.}")
    for m in (sys.argv[1:] or list(C.MODES)):
        draw(m)
