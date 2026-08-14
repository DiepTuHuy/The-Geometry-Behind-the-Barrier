#!/usr/bin/env python3
r"""figA1_decompose.py -- Phu luc: kiem chung Menh de 4.4 bang phep do.

CAU CHUYEN CUA HINH:
    "Menh de 4.4 chi chung minh alpha = 1/2 cho THANH PHAN Gauss-Newton dong bang
     Ftilde. Voi Fisher THAT con them so hang van chuyen softmax, bi chan boi
     3 C_J^3 rho_S -- KHONG dam bao co theo width, chi co theo do TU TIN cua tien
     doan. Do that: sp n^-0.61 va muP n^-0.46 (co, vi mang rong fit chac hon),
     nhung NTK n^+0.07 (TANG -- rong hon lai kem tu tin hon vi dau ra nhan
     1/sqrt(n) nen logit nho, softmax gan deu). Phep
     do cho thay so hang KHONG duoc dinh ly phu do lai chiem phan lon ||dF||, va
     o NTK thi rho_S con TANG theo width. Do la ly do Gia thiet 4.1 cho F la gia
     thiet co NOI DUNG THUC NGHIEM, phai kiem bang do truc tiep (Nhan xet 4.5)."

Y NGHIA BON COT -- DA XAC MINH tu code-1/decompose_dF.py:

    _dFz_apply(..., pr0=None)  -> (d_z F)v        Fisher THAT
    _dFz_apply(..., pr0=pr0)   -> (d_z Ftilde)v   softmax DONG BANG tai checkpoint

    dF_op        = max_z ||d_z F||_op
    gn_op        = max_z ||d_z Ftilde||_op          <- GAUSS-NEWTON (S dong bang)
    transport_op = max_z ||d_z F - d_z Ftilde||_op  <- VAN CHUYEN SOFTMAX
                   (tai w0 co S = S0 nen hieu nay dung bang E[J^T (d_zS) J])
    rho_S        = E_x ||S(p_w(x))||_op,  S = diag(p) - p p^T
    tr_frac      = transport_op / dF_op

Cau hinh phep do: z NGAU NHIEN (randn chuan hoa), NZ=3 huong, PI_ITERS=8 vong
power-iteration cho op-norm, ACTS=[gelu, tanh], NSEEDS=3
-> 3 che do x 2 ham x 7 width x 3 seed = 126 dong. Chi co MLP.

CHAY:  python3 figA1_decompose.py
File nay CHI DOC CSV.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

import style
import common as C


def main():
    d = C.load_decompose()
    widths = np.array(sorted(d.width.unique()), float)
    acts = sorted(set(d.act))

    fig = plt.figure(figsize=(style.W_FULL, 2.05))
    gs = fig.add_gridspec(1, 3)
    ax0, ax1, ax2 = (fig.add_subplot(gs[0, j]) for j in range(3))

    def med(col, reg, act=None):
        s = d[d.regime == reg]
        if act is not None:
            s = s[s.act == act]
        return s.groupby("width")[col].median().reindex(widths).values

    def band(ax, col, logy=False):
        """Duong day = trung vi gop; duong manh = tung activation."""
        for reg in style.ORDER_REGIME:
            c = style.COLOR_REGIME[reg]
            for a in acts:
                ax.plot(widths, med(col, reg, a), "-", lw=0.7, color=c,
                        alpha=0.5, zorder=2)
            ax.plot(widths, med(col, reg), marker="o", ms=3.2, lw=1.8, color=c,
                    linestyle=style.LINESTYLE_REGIME[reg], mfc=c, mec="white",
                    mew=0.6, zorder=4)

    # ---------------------------------- (a) ti trong cua so hang VAN CHUYEN
    ax0.axhline(1.0, color=style.HAIR, lw=0.7, zorder=0)
    band(ax0, "tr_frac")
    ax0.set_ylim(0, 1.12)
    ax0.set_ylabel(r"transport share of $\|\partial_z F\|$", labelpad=1)
    ax0.set_title("the term the theorem does not cover", fontsize=6.6,
                  color=style.MUTED, pad=3)
    C.panel(ax0, "a", dx=-0.02, dy=1.08)

    # ---------------------------------- (b) rho_S: khong co theo width
    ax1.axhline(0.5, color=style.HAIR, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax1.annotate(r"upper bound $\frac{1}{2}$", (widths[0], 0.5),
                 xytext=(1, 3), textcoords="offset points", fontsize=6.2,
                 color=style.MUTED, va="bottom", ha="left")
    band(ax1, "rho_S")
    ax1.set_yscale("log"); ax1.set_ylim(top=2.2)   # cho chua nhan tren vach 1/2
    C.log_decades(ax1, "y")
    ax1.set_ylabel(r"$\rho_S=\mathbb{E}_x\|S(p_w(x))\|_{\mathrm{op}}$", labelpad=1)
    ax1.set_title(r"$\rho_S$ falls only where the fit gets confident",
                  fontsize=6.6, color=style.MUTED, pad=3)
    a_ntk, _, _ = C.fit_alpha(widths, med("rho_S", "ntk"))
    ax1.annotate(rf"NTK: $n^{{{-a_ntk:+.2f}}}$, grows", (0.05, 0.62),
                 xycoords="axes fraction", fontsize=6.6,
                 color=style.COLOR_REGIME["ntk"])
    C.panel(ax1, "b", dx=-0.02, dy=1.08)

    # ---------------------------------- (c) hai thanh phan, ve rieng
    # Kieu net da danh cho che do, nen phan biet hai DAI LUONG bang MARKER DAC
    # (van chuyen) va MARKER RONG (Gauss-Newton).
    for reg in style.ORDER_REGIME:
        c = style.COLOR_REGIME[reg]
        ax2.plot(widths, med("transport_op", reg), marker="o", ms=3.2, lw=1.6,
                 color=c, linestyle=style.LINESTYLE_REGIME[reg],
                 mfc=c, mec="white", mew=0.6, zorder=4)
        ax2.plot(widths, med("gn_op", reg), marker="o", ms=3.2, lw=1.0,
                 color=c, linestyle=style.LINESTYLE_REGIME[reg],
                 mfc="white", mec=c, mew=0.9, alpha=0.85, zorder=3)
    ax2.set_yscale("log"); C.log_decades(ax2, "y")
    ax2.set_ylabel(r"$\|\cdot\|_{\mathrm{op}}$", labelpad=1)
    ax2.set_title("both components, separately", fontsize=6.6,
                  color=style.MUTED, pad=3)
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    for lab, col, dy in (("transport", "transport_op", 8),
                         ("Gauss–Newton", "gn_op", -12)):
        ax2.annotate(lab, (widths[1], med(col, "sp")[1]), xytext=(3, dy),
                     textcoords="offset points", fontsize=6.6,
                     color=style.COLOR_REGIME["sp"], zorder=6, path_effects=halo)
    C.panel(ax2, "c", dx=-0.02, dy=1.08)

    for ax in (ax0, ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(widths))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: C.wlabel(v)))
        ax.set_xlabel("width")
        C.hgrid(ax, alpha=0.4); C.despine(ax, offset=3)

    hs = C.regime_handles(dashed=True) + [
        Line2D([], [], color=style.MUTED, marker="o", ms=3.4, linestyle="none",
               mfc=style.MUTED, mec="white", mew=0.6, label="transport"),
        Line2D([], [], color=style.MUTED, marker="o", ms=3.4, linestyle="none",
               mfc="white", mec=style.MUTED, mew=0.9, label="Gauss–Newton")]
    fig.legend(handles=hs, loc="outside lower center", ncol=5, fontsize=6.8,
               handletextpad=0.4, columnspacing=1.2)

    style.save(fig, "figA1_decompose")
    plt.close(fig)
    print(r"  \caption{Proposition 4.4 proves the $n^{-1/2}$ rate only for the "
          r"frozen-softmax Gauss--Newton part $\widetilde F$; the measurement "
          r"separates it from the remainder (MLP/MNIST, GELU and TANH, 3 seeds, "
          r"3 random probe directions, power iteration for each operator norm). "
          r"(a) Outside NTK the softmax transport term carries almost all of "
          r"$\partial F$. (b) Its bounding factor $\rho_S$ shrinks only through "
          r"prediction confidence, not through width as such: it falls under "
          r"standard and $\mu$P but \emph{grows} under NTK. (c) The two "
          r"components separately. "
          r"Assumption 4.1 for the true $F$ therefore carries empirical content "
          r"and has to be measured, as Remark 4.5 states.}")
    for reg in style.ORDER_REGIME:
        print(f"     {reg:4s} tr_frac={np.median(med('tr_frac', reg)):.2f}   "
              f"rho_S {med('rho_S', reg)[0]:.2e} -> {med('rho_S', reg)[-1]:.2e}")


if __name__ == "__main__":
    main()
