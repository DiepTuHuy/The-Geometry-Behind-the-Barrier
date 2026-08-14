#!/usr/bin/env python3
r"""figA2_by_activation.py -- PHU LUC: boc tach theo tung activation.

Hinh CHINH (fig1) ma hoa mau theo CHE DO. Hinh nay giu ban boc tach theo
ACTIVATION vi paper co phat bieu o muc activation: Nhan xet 4.5 liet ke 12 so mu
va Bang 1 liet ke 4 hang -- nguoi phan bien kiem hai cho do phai tim thay chung
o dau. Day chinh la cho do.

(Ban goc cua Figure 2)

CAU CHUYEN CUA HINH:
    "Khi width tang, do bien thien cua metric Fisher ||dF||_op TAT DAN theo luat
     luy thua o CA BA che do tham so hoa. Nhung LMC barrier thi khong: NTK TANG,
     Standard gan nhu dung yen, muP SAP ve 0. Hai dai luong nay khong di cung
     nhau -- do la tien de cho ket qua phu dinh o Figure 3."

Vi sao ve nhu the nay:
  * CA HAI hang deu log-log va DUNG CHUNG truc y trong moi hang. Ba cot vi the
    so sanh duoc truc tiep bang mat: khoang cach doc giua NTK va muP la 3 bac
    thap phan, va no hien ra ngay lap tuc.
  * Nhan dat TRUC TIEP o cuoi duong, khong dung legend hop -> mat khong phai
    nhay qua lai giua chu thich va duong.
  * So mu alpha (do doc log-log) in ngay trong o -- con so quan trong nhat cua
    panel nam trong panel.

CHAY:  python3 fig1_barrier_dF.py            # ca 3 kien truc
       python3 fig1_barrier_dF.py mlp        # rieng MLP
File nay CHI DOC CSV.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

import style
import common as C

ROWS = [
    dict(key="barrier", ylabel="LMC barrier  $B$"),
    dict(key="dF_op",   ylabel=r"$\|\partial F\|_{\mathrm{op}}$"),
]


def draw(mode):
    pairs = C.load_pairs(mode)
    nets = C.load_nets(mode).dropna(subset=["dF_op"])
    widths = np.array(sorted(pairs.width.dropna().unique()), float)
    acts = [a for a in style.ORDER_ACT if a in set(pairs.act.astype(str))]

    stats = {
        "barrier": C.med_iqr(pairs, ["regime", "act", "width"], "barrier"),
        "dF_op":   C.med_iqr(nets, ["regime", "act", "width"], "dF_op"),
    }

    fig = plt.figure(figsize=(style.W_FULL, 3.05))
    axes, _ = C.grid_with_label_gutter(fig, 2, 3, gutter=0.31)

    for i, row in enumerate(ROWS):
        st = stats[row["key"]]
        for j, reg in enumerate(style.ORDER_REGIME):
            ax = axes[i, j]
            ends, alphas = [], []
            for a in acts:
                s = (st[(st.regime == reg) & (st.act == a)]
                     .set_index("width").reindex(widths))
                col = C.curve(ax, widths, s.med.values, s.lo.values, s.hi.values,
                              act=a, regime=reg, linestyle="-")
                al, r2, n = C.fit_alpha(widths, s.med.values)
                if np.isfinite(al):
                    alphas.append(al)
                last = s.med.dropna()
                if len(last):
                    ends.append((last.iloc[-1], a, col))

            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            C.hgrid(ax, alpha=0.45)
            C.despine(ax, offset=3)
            C.log_decades(ax, "y")
            C.panel(ax, "abcdef"[i * 3 + j], dx=-0.015, dy=1.0)

            if alphas:   # so mu trung vi cua o -- dau duong = giam theo width
                med_a = float(np.median(alphas))
                C.slope_tag(ax, rf"$\alpha={med_a:+.2f}$", (0.045, 0.055),
                            color=style.MUTED, fontsize=6.8)
            if i == 0:
                ax.set_title(style.LABEL_REGIME[reg], pad=4)
            if j == 0:
                ax.set_ylabel(row["ylabel"])
            if j == 2:
                C.end_labels(ax, ends, min_gap=0.088)
            if i == 1:
                ax.set_xlabel("width multiplier" if mode == "cnn" else "width")

    for ax in axes[1]:
        ax.xaxis.set_major_locator(FixedLocator(widths))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: C.wlabel(v)))

    style.save(fig, f"figA2_by_activation_{mode}")
    plt.close(fig)
    # caption di kem, de dan thang vao LaTeX -- so lieu va hinh khong bao gio lech nhau
    print(r"  \caption{%s. Top: LMC barrier after weight matching. Bottom: "
          r"$\|\partial F\|_{\mathrm{op}}$. Curves are medians over 5 seeds "
          r"(10 pairs) per cell; bands are inter-quartile ranges. Both rows share "
          r"a log $y$-axis across regimes. $\alpha$ is the median log--log slope "
          r"over activations (positive = decays with width).}" % style.LABEL_MODE_TEX[mode])


if __name__ == "__main__":
    for m in (sys.argv[1:] or list(C.MODES)):
        draw(m)
