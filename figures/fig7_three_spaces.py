#!/usr/bin/env python3
r"""fig7_three_spaces.py -- Cong thuc (4): ba khong gian va hai anh xa.

CAU CHUYEN CUA HINH:
    "CUNG MOT CAP DUONG, ve ba lan. Hinh khong doi -- CAI THUOC doi. O Euclid,
     day cung ngan nhat. Doi thuoc sang G_F thi trac dia moi ngan nhat, du hai
     duong van y nguyen. Sang khong gian phan phoi, trac dia thanh cung tron lon."

        (R^P, ||.||_2)  --G_F=F+lam I-->  (R^P, G_F)  --w|->p_w-->  P_K

KY THUAT VE MUON TU tomgoldstein/loss-landscape (plot_2D.py):

    CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    plt.clabel(CS, inline=1, fontsize=8)

  -> Ho KHONG to nen. Ho ve DUONG DONG MUC CO MAU tren nen trang. Nho vay muc
     danh cho du lieu (duong di, diem, qua cau don vi) khong phai tranh cho voi
     nen. Ban truoc cua toi to nen day mau nen moi thu khac bi nuot.

  -> KHONG lay clabel (nhan so noi tuyen). Ho ghi gia tri mat mat THAT; hinh nay
     la mo hinh do choi nen moi con so deu vo nghia -- in ra se thanh gia mao du
     lieu. Muon ky thuat, khong muon con so.

  -> 'summer' di tu xanh luc dam (muc THAP) sang vang nhat (muc CAO), nen long
     chao va thung lung tu duoc vien dam -- dung cho can nhin. Cat bot dau vang
     nhat (0..0.78) de duong muc cao khong bien mat tren nen trang.

BA QUYET DINH THIET KE (hoc tu Tan et al., Geodesic Mode Connectivity):

 1. Day la SO DO, khong phai ba bieu do canh nhau. Toan bo nam tren MOT axes
    (tat truc), nen vat the / mui ten / nhan deu dat bang toa do tuyet doi.
    Mat cong o vat the 3 duoc CHIEU TRUC GIAO bang tay (project) roi ve nhu
    duong 2-D -> cung mot he voi hai vat the kia.

 2. HAI VAT THE DAU DUNG CHUNG MOT NEN. Chung la CUNG MOT TAP HOP R^P; chi cai
    thuoc doi. Neu ve nen khac nhau thi nguoi doc tuong la hai khong gian khac.
    Cai duy nhat khac: qua cau don vi -- TRON o Euclid, ELLIPSE o G_F.

 3. Mui ten dat theo BOUNDING BOX DO DUOC cua vat the, khong phai theo hop danh
    nghia. Manh mat cong giu ti le nen KHONG lap day hop theo chieu ngang -- neo
    mui ten vao canh hop se cho ho hai ben lech nhau (0.16 vs 0.36). Ham
    span_of() tra ve be ngang that, mui ten cang deu tu do.

MOI THU DUOC TINH: metric G = (1+beta L)(I + kappa gradL gradL^T); trac dia =
cuc tieu nang luong duong roi rac; va voi K=3, don hinh xac suat mang metric
Fisher-Rao DANG CU voi 1/8 mat cau ban kinh 2 qua p |-> 2sqrt(p), nen trac dia
la CUNG TRON LON -- chinh xac.

CHAY:  python3 fig7_three_spaces.py
Hinh Y NIEM, khong doc CSV.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon

import style
import common as C
from fig0_geometry_schematic import (A, B, loss, metric, solve_geodesic,
                                     match_by_projection, CMAP, NBAND,
                                     C_LIN, C_GEO, C_MET)
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, to_rgb
import matplotlib.patheffects as pe

INK, HAIR = style.INK, style.MUTED

# Ban NHAT cua bang mau fig0: giu DUNG chieu (tham = day gieng = mat mat thap)
# de hai hinh y niem khong mau thuan nhau, nhung pha trang 52% -- ban 2-D la
# ban do phang, muc cua duong va diem phai noi len tren nen, khong bi nuot.
_WASH = 0.52
CMAP2 = LinearSegmentedColormap.from_list("loss2", [
    tuple(np.array(to_rgb(c)) * (1 - _WASH) + _WASH for c in [c])[0]
    for c in ["#0a3a63", "#12558a", "#1c74a8", "#2f93b4", "#54b0b3",
              "#8bc9a8", "#bfe0a4", "#e8f3c4", "#f7fbe6"]])
HALO = [pe.withStroke(linewidth=2.4, foreground="white")]
MODEL_X, MODEL_Y = (-1.62, 1.62), (-0.88, 1.12)
P_A = np.array([0.850, 0.075, 0.075])
P_B = np.array([0.075, 0.850, 0.075])
P_M = np.array([0.408, 0.262, 0.330])
phi = lambda p: 2.0 * np.sqrt(np.asarray(p, float))

# ------- bo cuc so do (toa do tuyet doi, aspect = equal) -------
W, H = 3.55, 2.30                       # kich thuoc moi vat the
GAP = 1.30
X0 = [0.15, 0.15 + W + GAP, 0.15 + 2 * (W + GAP)]
YB, YT = 0.42, 0.42 + H
ARROW_Y = YB + 0.5 * H                  # tam that cua ca ba vat the (da do lai)
LABEL_Y = YB - 0.30


def box(i):
    return (X0[i], X0[i] + W, YB, YT)


def place(P, b):
    x0, x1, y0, y1 = b
    u = (np.asarray(P, float)[:, 0] - MODEL_X[0]) / (MODEL_X[1] - MODEL_X[0])
    v = (np.asarray(P, float)[:, 1] - MODEL_Y[0]) / (MODEL_Y[1] - MODEL_Y[0])
    return np.stack([x0 + u * (x1 - x0), y0 + v * (y1 - y0)], 1)


def project(P3, elev=20.0, azim=44.0):
    e, a = np.radians(elev), np.radians(azim)
    P3 = np.atleast_2d(np.asarray(P3, float))
    return np.stack([-np.sin(a) * P3[:, 0] + np.cos(a) * P3[:, 1],
                     (-np.sin(e) * np.cos(a) * P3[:, 0]
                      - np.sin(e) * np.sin(a) * P3[:, 1]
                      + np.cos(e) * P3[:, 2])], 1)


def fr_geodesic(p, q, n=240):
    u, v = phi(p) / 2, phi(q) / 2
    a = np.arccos(np.clip(u @ v, -1, 1)); s = np.linspace(0, 1, n)[:, None]
    return 2 * (np.sin((1 - s) * a) * u + np.sin(s * a) * v) / np.sin(a)


def chord_image(p, q, m, n=260):
    Pa, Pb, Pm = phi(p), phi(q), phi(m)
    t = np.linspace(0, 1, n)[:, None]
    X = ((1 - t) ** 2 * Pa + 2 * (1 - t) * t * ((4 * Pm - Pa - Pb) / 2)
         + t ** 2 * Pb)
    return 2 * X / np.linalg.norm(X, axis=1, keepdims=True)


def arclen(X):
    return float(np.sum(np.linalg.norm(np.diff(X, axis=0), axis=1)))


SUMMER = LinearSegmentedColormap.from_list(
    "summer_cut", plt.get_cmap("summer")(np.linspace(0.0, 0.78, 64)))


def landscape(ax, b, X, Y, Z):
    """Nen dung chung cho vat the 1 va 2: cung mot tap hop, cung mot ham mat mat.

    Loi ve cua loss-landscape: chi DUONG DONG MUC CO MAU tren nen trang.
    """
    Xb = b[0] + (X - MODEL_X[0]) / (MODEL_X[1] - MODEL_X[0]) * (b[1] - b[0])
    Yb = b[2] + (Y - MODEL_Y[0]) / (MODEL_Y[1] - MODEL_Y[0]) * (b[3] - b[2])
    ax.add_patch(Polygon([[b[0], b[2]], [b[1], b[2]], [b[1], b[3]], [b[0], b[3]]],
                         closed=True, facecolor="white", edgecolor="none", zorder=0))
    ax.contour(Xb, Yb, Z, levels=np.linspace(Z.min(), Z.max(), 11),
               cmap=SUMMER, linewidths=0.9, zorder=1)
    ax.add_patch(Polygon([[b[0], b[2]], [b[1], b[2]], [b[1], b[3]], [b[0], b[3]]],
                         closed=True, fill=False, edgecolor=INK, lw=0.9, zorder=9))


def unit_ball(ax, b, tt, kind):
    th = np.linspace(0, 2 * np.pi, 160)
    p = A + tt * (B - A)
    if kind == "round":
        e = p[None] + 0.30 * np.stack([np.cos(th), np.sin(th)], 1)
    else:
        w, V = np.linalg.eigh(metric(p)[0])
        semi = 1.0 / np.sqrt(w); semi = 0.30 * semi / semi.max()
        e = (p[:, None] + V @ np.stack([semi[0] * np.cos(th),
                                        semi[1] * np.sin(th)])).T
    E = place(e, b)
    ax.add_patch(Polygon(E, closed=True, facecolor="white", edgecolor="none",
                         alpha=0.88, zorder=3.9))          # lot trang
    ax.add_patch(Polygon(E, closed=True, facecolor=C_MET, alpha=0.26,
                         edgecolor=C_MET, lw=1.1, zorder=4))


def draw_paths(ax, lin2, geo2, names, dy=-0.17):
    ax.plot(*lin2.T, color=C_LIN, lw=1.9, ls=(0, (4.2, 1.9)), zorder=6,
            dash_capstyle="round")
    ax.plot(*geo2.T, color=C_GEO, lw=2.2, zorder=7, solid_capstyle="round")
    for P, nm in zip((lin2[0], lin2[-1]), names):
        ax.plot(*P, "o", ms=5.0, mfc="white", mec=INK, mew=1.2, zorder=10)
        ax.text(P[0], P[1] + dy, nm, fontsize=8.5, ha="center", va="top",
                color=INK, zorder=10, path_effects=HALO)


def main():
    geo = solve_geodesic()
    ts = np.linspace(0, 1, 301)
    lin = A + np.outer(ts, B - A)
    geo_t = match_by_projection(geo, ts)
    gx = np.linspace(*MODEL_X, 260); gy = np.linspace(*MODEL_Y, 190)
    Xm, Ym = np.meshgrid(gx, gy)
    Zm = loss(np.stack([Xm.ravel(), Ym.ravel()], 1)).reshape(Xm.shape)

    XLIM = (0.02, X0[2] + W + 0.62)
    YLIM = (LABEL_Y - 0.36, YT + 0.20)
    fig = plt.figure(figsize=(style.W_FULL,
                              style.W_FULL * (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0])))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect("equal"); ax.axis("off")

    names = (r"$\boldsymbol{\theta}_A$", r"$\boldsymbol{\theta}_B$")
    for i, (kind, tag_lin, tag_geo, sub) in enumerate((
            ("round", "shortest", None, r"$(\mathbb{R}^{P},\ \|\cdot\|_2)$"),
            ("ellipse", None, "shortest", r"$(\mathbb{R}^{P},\ G_F)$"))):
        b = box(i)
        landscape(ax, b, Xm, Ym, Zm)                 # NEN GIONG HET NHAU
        for tt in (0.24, 0.50, 0.76):
            unit_ball(ax, b, tt, kind)               # chi CAI THUOC la khac
        l2, g2 = place(lin, b), place(geo_t, b)
        draw_paths(ax, l2, g2, names)
        if tag_lin:
            ax.text(*(l2[150] + [0, -0.46]), tag_lin, fontsize=8.0, color=C_LIN,
                    ha="center", va="top", zorder=11, path_effects=HALO)
        if tag_geo:
            ax.text(*(g2[150] + [0, 0.11]), tag_geo, fontsize=8.0, color=C_GEO,
                    ha="center", va="bottom", zorder=11, path_effects=HALO)
        ax.text(0.5 * (b[0] + b[1]), LABEL_Y, sub, fontsize=9.0, ha="center",
                va="top", color=INK)

    # ------------------------------------------- vat the 3: manh mat cong P_K
    b = box(2)
    aa = np.linspace(0.55, 1.50, 28); bb = np.linspace(0.08, 1.49, 28)
    S = lambda U, V: np.stack([2 * np.sin(U) * np.cos(V), 2 * np.sin(U) * np.sin(V),
                               2 * np.cos(U)], -1)
    quad = np.vstack([S(aa, np.full_like(aa, bb[0])),
                      S(np.full_like(bb, aa[-1]), bb),
                      S(aa[::-1], np.full_like(aa, bb[-1])),
                      S(np.full_like(bb, aa[0]), bb[::-1])])
    Gd, Ch = fr_geodesic(P_A, P_B), chord_image(P_A, P_B, P_M)
    ref = project(np.vstack([quad, Gd, Ch]))
    mn, mx = ref.min(0), ref.max(0)
    sc = min((b[1] - b[0]) / (mx[0] - mn[0]), (b[3] - b[2]) / (mx[1] - mn[1]))
    aff = lambda P3: ((project(P3) - 0.5 * (mn + mx)) * sc
                      + [0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3])])
    ax.add_patch(Polygon(aff(quad), closed=True, facecolor="#F3F7FA",
                         edgecolor=INK, lw=0.9, zorder=1))
    for c in np.linspace(0.08, 1.49, 7):
        ax.plot(*aff(S(aa, np.full_like(aa, c))).T, color="#C6D4DF", lw=0.75, zorder=2)
    for c in np.linspace(0.55, 1.50, 6):
        ax.plot(*aff(S(np.full_like(bb, c), bb)).T, color="#C6D4DF", lw=0.75, zorder=2)
    g2, c2 = aff(Gd), aff(Ch)
    draw_paths(ax, c2, g2, (r"$\mathbf{p}_A$", r"$\mathbf{p}_B$"))
    ax.text(*(g2[120] + [0, -0.13]), "shortest", fontsize=8.0, color=C_GEO,
            ha="center", va="top", zorder=11)
    ax.text(0.5 * (b[0] + b[1]), LABEL_Y, r"$\mathcal{P}_K$", fontsize=9.0,
            ha="center", va="top", color=INK)

    # ------------------------------------------- hai anh xa
    # Be ngang THAT cua tung vat the: hai cai dau lap day hop, cai thu ba thi
    # khong (manh mat cong giu ti le). Lay so do duoc, khong lay so danh nghia.
    spans = [(X0[0], X0[0] + W), (X0[1], X0[1] + W),
             (aff(quad)[:, 0].min(), aff(quad)[:, 0].max())]
    for i, lab in ((0, r"$G_F=F+\lambda I$"), (1, r"$w\mapsto p_w$")):
        x0, x1 = spans[i][1] + 0.20, spans[i + 1][0] - 0.20
        # Matplotlib's FancyArrowPatch distorts the arrowhead when given a dashed linestyle.
        # We fix this by drawing a dashed line for the shaft and a very short solid arrow for the head.
        # Shorten the line just enough (x1 - 0.03) so the flat cap doesn't stick out past the sharp arrow tip.
        ax.plot([x0, x1 - 0.03], [ARROW_Y, ARROW_Y], color=INK, lw=1.1, linestyle=(0, (3.4, 2.1)), zorder=12)
        ax.add_patch(FancyArrowPatch((x1 - 0.03, ARROW_Y), (x1, ARROW_Y),
                                     arrowstyle="-|>", mutation_scale=10, lw=1.1,
                                     color=INK, zorder=13))
        ax.text(0.5 * (x0 + x1), ARROW_Y + 0.13, lab, fontsize=7.6, ha="center",
                va="bottom", color=INK, zorder=12)

    style.save(fig, "fig7_three_spaces")
    plt.close(fig)
    print(r"  \caption{\textbf{Schematic.} The three spaces of Eq.~(4), with the "
          r"same pair of paths drawn in each. The first two panels show the same "
          r"set $\mathbb{R}^P$ and the same loss field; only the ruler changes, "
          r"shown by the unit balls of the metric. Left: round unit balls, and the "
          r"chord is the shortest path. Middle: under $G_F=F+\lambda I$ the unit "
          r"balls stretch along the level sets and the geodesic becomes the "
          r"shortest path, although neither curve has moved. Unit balls share a "
          r"common major axis, so their shape is meaningful but their absolute "
          r"size is not. Right: the induced picture on the statistical manifold, "
          r"exact for $K=3$, where Fisher--Rao makes $\mathcal{P}_3$ isometric to "
          r"an octant of the radius-$2$ sphere under $p\mapsto2\sqrt{p}$ and "
          r"geodesics are great circles; there the chord's image is %.0f\%% longer. "
          r"The correspondence between $G_F$- and Fisher--Rao geodesics is exact "
          r"only as $\lambda\to0$.}" % (100 * (arclen(Ch) / arclen(Gd) - 1)))


if __name__ == "__main__":
    main()
