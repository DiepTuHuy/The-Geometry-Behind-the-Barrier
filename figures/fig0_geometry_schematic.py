#!/usr/bin/env python3
r"""fig0_geometry_schematic.py -- Figure 1 mo dau cua paper (hinh y niem).

HAI PANEL: (a) da tap 3D voi day cung vs trac dia, (b) mat mat doc hai duong.
Panel ||xi(t)|| cu nam o fig6_deviation.py (thuoc Muc 5.1).

CAU CHUYEN CUA HINH:
    "Duong noi suy TUYEN TINH giua hai nghiem treo qua song nui cua ham mat mat.
     Duong TRAC DIA cua metric Fisher vong theo thung lung. Do lech giua hai
     duong, xi(t), bang 0 o hai dau va phinh ra o giua -- dung dang bieu dien
     Green cua Bo de 4.10."

KHONG PHAI HINH VE TAY. Truong mat mat, metric va duong trac dia deu duoc TINH:
  * mat mat L : thung lung cong noi hai gieng + hai gieng o hai dau
  * metric kieu Fisher:
        G = (1 + beta*Lt) * (I + kappa * gradL gradL^T)
    -> di DOC duong muc thi re, di NGUOC gradient thi dat; o cho mat mat cao
       thi moi huong deu dat. Dung tinh chat cua Fisher: huong nao lam doi dau
       ra nhieu thi huong do "dai".
  * trac dia : cuc tieu hoa nang luong duong roi rac  E = sum dp^T G(mid) dp
  * ellipse  : qua cau don vi cua G tai diem do. Truc DAI = huong re,
               truc NGAN = huong dat.

CHAY:  python3 fig0_geometry_schematic.py
Khong doc CSV, khong doc checkpoint -- hinh y niem, tu chua.
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import style
import common as C

# ---------------------------------------------------------------- truong mat mat
A = np.array([-1.35, 0.0])
B = np.array([1.35, 0.0])
_XV, _HV, _WELL = 1.35, 0.92, 0.42
_CT, _ST = 1.00, 0.55      # do cao va be rong cua ranh ngang


def valley(x):
    """Day thung lung: cung cong noi A va B."""
    return _HV * (1.0 - (x / _XV) ** 2)


def loss(p):
    """Ranh ngang BAO HOA (khong phai parabol) + hai gieng.

    Dung 1 - exp(-d^2) thay cho d^2: ham BI CHAN nen mat mat khong vot len o goc
    mien ve. Nho vay khong can cat cung (np.minimum) -- va chinh phep cat cung la
    thu tao ra cao nguyen phang co canh sac, tuc cho "gap khuc" tren mat 3D.
    """
    p = np.atleast_2d(np.asarray(p, float))
    x, y = p[:, 0], p[:, 1]
    d = y - valley(x)
    trans = _CT * (1.0 - np.exp(-d ** 2 / (2 * _ST ** 2)))
    bowl = 0.05 * (x ** 2 + y ** 2)
    wa = np.exp(-(((x - A[0]) ** 2 + (y - A[1]) ** 2) / (2 * _WELL ** 2)))
    wb = np.exp(-(((x - B[0]) ** 2 + (y - B[1]) ** 2) / (2 * _WELL ** 2)))
    return (trans + bowl - 0.50 * wa - 0.47 * wb).squeeze()


def grad_loss(p, h=1e-4):
    p = np.atleast_2d(np.asarray(p, float))
    gx = (loss(p + [h, 0]) - loss(p - [h, 0])) / (2 * h)
    gy = (loss(p + [0, h]) - loss(p - [0, h])) / (2 * h)
    return np.stack([np.atleast_1d(gx), np.atleast_1d(gy)], axis=-1)


# ---------------------------------------------------------------- metric kieu Fisher
KAPPA, BETA = 3.0, 6.0
_L0 = float(loss(np.array([A])))


def metric(p):
    """G(p), mang (...,2,2)."""
    p = np.atleast_2d(np.asarray(p, float))
    g = grad_loss(p)
    Lt = np.clip(np.atleast_1d(loss(p)) - _L0, 0.0, None)
    G = np.eye(2)[None] + KAPPA * g[:, :, None] * g[:, None, :]
    return (1.0 + BETA * Lt)[:, None, None] * G


def path_energy(interior, n):
    pts = np.vstack([A, interior.reshape(n, 2), B])
    dp = np.diff(pts, axis=0)
    mid = 0.5 * (pts[:-1] + pts[1:])
    return float(np.einsum("ni,nij,nj->", dp, metric(mid), dp))


def solve_geodesic(n=45):
    t = np.linspace(0, 1, n + 2)[1:-1]
    init = A + np.outer(t, B - A)
    init[:, 1] += 0.9 * valley(init[:, 0])          # xuat phat gan thung lung
    res = minimize(path_energy, init.ravel(), args=(n,), method="L-BFGS-B",
                   options=dict(maxiter=5000, ftol=1e-13, gtol=1e-11))
    return np.vstack([A, res.x.reshape(n, 2), B])


def match_by_projection(geo, ts):
    """Diem tren trac dia co HINH CHIEU len day cung bang t -> so o CUNG mot t."""
    d = B - A
    s = ((geo - A) @ d) / (d @ d)
    s, idx = np.unique(s, return_index=True)
    return np.stack([np.interp(ts, s, geo[idx, 0]),
                     np.interp(ts, s, geo[idx, 1])], axis=1)


# ---------------------------------------------------------------- bang mau
# Chieu mau theo lo�i cua cac bai tham chieu: SAU = xanh tham = day gieng
# (mat mat thap), SANG = kem = vanh cao. Truc quan nhu ban do do sau nuoc.
CMAP = LinearSegmentedColormap.from_list("loss", [
    "#0a3a63", "#12558a", "#1c74a8", "#2f93b4", "#54b0b3",
    "#8bc9a8", "#bfe0a4", "#e8f3c4", "#f7fbe6"])
NBAND = 14                                  # chia dai -> mat cong trong nhu duoc dieu khac
# Hinh y niem dung BANG MAU RIENG, khong trung voi mau cua cac hinh du lieu.
# Ly do: fig0 dung TRUOC trong bai. Neu duong trac dia mau cam (#D55E00) thi
# nguoi doc hoc "cam = trac dia", roi sang fig1-fig5 lai phai hoc lai "cam = muP"
# (#D8741A, gan nhu cung mau). Mau than chi + do tham khong xuat hien o dau khac,
# nen khong the nham voi bat ky chuoi du lieu nao.
C_LIN, C_GEO, C_MET = "#2B2B2B", "#C1121F", "#6A4C93"
XLIM, YLIM, ZMAX = (-2.05, 2.05), (-1.45, 1.12), 1.40


def main():
    geo = solve_geodesic()
    ts = np.linspace(0, 1, 321)
    lin = A + np.outer(ts, B - A)
    geo_t = match_by_projection(geo, ts)
    xi = np.linalg.norm(geo_t - lin, axis=1)
    L_lin, L_geo = loss(lin), loss(geo_t)
    chord = 0.5 * (L_lin[0] + L_lin[-1])
    i_star, i_xi = int(np.argmax(L_lin)), int(np.argmax(xi))

    # Chi can panel (a), bieu dien mat 3D.
    fig = plt.figure(figsize=(style.W_FULL, 4.0)) # Enlarge figure height
    # computed_zorder=False: tu quyet dinh thu tu ve, neu khong matplotlib se
    # giau mat cac duong nam tren mat cong.
    ax0 = fig.add_subplot(111, projection="3d", computed_zorder=False)

    # ============================================== (a) mat mat mat 3D + hai duong
    # Luoi day + KHONG cat cung: cat cung (np.minimum) tao ra cao nguyen phang
    # voi canh sac -- chinh la cho "gap khuc" nhin thay tren mat cong.
    gx = np.linspace(*XLIM, 340)
    gy = np.linspace(*YLIM, 260)
    X, Y = np.meshgrid(gx, gy)
    Z = loss(np.stack([X.ravel(), Y.ravel()], 1)).reshape(X.shape)
    zlo, zhi = float(Z.min()), float(Z.max())
    zfloor = zlo - 0.72 * (zhi - zlo)          # san chieu, dat duoi day mat cong
    lift = 0.012 * (zhi - zlo)                 # nhac duong len chut de khong bi mat cat

    # --- qua cau don vi cua metric, DAP LEN mat cong tai 3 diem tham do ---
    # (khong ve o mat san nua: mat san chiem nua panel va canh tranh voi mat cong)
    th = np.linspace(0, 2 * np.pi, 200)
    rings = []
    for tt in (0.16, 0.50, 0.84):
        p = A + tt * (B - A)
        w, V = np.linalg.eigh(metric(p)[0])
        semi = 0.42 / np.sqrt(w)
        e = p[:, None] + V @ np.stack([semi[0] * np.cos(th), semi[1] * np.sin(th)])
        rings.append((e, loss(e.T)))

    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=315, altdeg=45)

    surf = ax0.plot_surface(X, Y, Z, cmap=cm.coolwarm, rstride=1, cstride=1,
                            linewidth=0, edgecolor="none", antialiased=True,
                            shade=True, lightsource=ls, alpha=0.95, zorder=4)
    surf.set_rasterized(True)      # giu PDF nhe; chu va duong van la vector
    
    # Them luoi (wireframe) tren be mat de nhan manh tinh toan hoc (giam do "AI")
    wire = ax0.plot_wireframe(X, Y, Z, rstride=15, cstride=15, color="white", 
                              linewidth=0.25, alpha=0.4, zorder=4.1)
    wire.set_rasterized(True)
    # duong binh do tren mat cong: khong co trong loss-landscape, nhung o kho nho
    # cua ICLR thi mat coolwarm tron mat het cam giac dia hinh neu thieu chung.
    ax0.contour(X, Y, Z, levels=np.linspace(zlo + 0.04, zhi - 0.06, 10),
                colors="white", linewidths=0.4, alpha=0.62, zorder=5)

    for e, ze in rings:                      # ring bam theo dia hinh
        ax0.plot(e[0], e[1], ze + 2 * lift, color="white", lw=1.5,
                 alpha=0.85, zorder=5.5)
        ax0.plot(e[0], e[1], ze + 2.2 * lift, color=C_MET, lw=1.0, zorder=5.6,
                 path_effects=[pe.Stroke(linewidth=2.6, foreground="white"),
                               pe.Normal()])

    # coolwarm co ca vung do lan vung xanh, nen mot mau duong bat ky cung se chim
    # o dau do. Vien trang quanh net (path_effects) giai quyet dut diem: duong noi
    # tren MOI nen ma khong phai doi he mau da thong nhat voi fig7.
    HALO = [pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()]

    # --- hai duong, ve TREN mat cong ---
    ax0.plot(lin[:, 0], lin[:, 1], L_lin + lift, color=C_LIN, lw=1.6,
             ls=(0, (4.0, 1.8)), zorder=6, path_effects=HALO)
    ax0.plot(geo_t[:, 0], geo_t[:, 1], L_geo + lift, color=C_GEO, lw=1.8,
             zorder=7, path_effects=HALO)
    k = np.linspace(0, len(ts) - 1, 15).astype(int)          # 15 "pivot"
    ax0.plot(lin[k, 0], lin[k, 1], L_lin[k] + 1.6 * lift, "o", ms=2.9,
             mfc=C_LIN, mec="white", mew=0.55, linestyle="none", zorder=7.5)
    ax0.plot(geo_t[k, 0], geo_t[k, 1], L_geo[k] + 1.6 * lift, "o", ms=3.1,
             mfc=C_GEO, mec="white", mew=0.55, linestyle="none", zorder=7.6)

    # --- doan thang dung do rao can tai t* ---
    ax0.plot([lin[i_star, 0]] * 2, [lin[i_star, 1]] * 2,
             [L_geo[i_star] + lift, L_lin[i_star] + lift],
             color=style.INK, lw=0.8, zorder=8)
    ax0.text(lin[i_star, 0] + 0.06, lin[i_star, 1],
             0.5 * (L_geo[i_star] + L_lin[i_star]), "$B$",
             fontsize=8.5, color=style.INK, zorder=9)

    for p, Lp, name in ((A, L_lin[0], r"$\theta_A$"), (B, L_lin[-1], r"$\theta_B$")):
        ax0.plot([p[0]], [p[1]], [Lp + lift], "o", ms=4.2, mfc="white",
                 mec=style.INK, mew=1.0, zorder=9)
        ax0.text(p[0], p[1] - 0.52, Lp, name, fontsize=8.5,
                 color=style.INK, ha="center", zorder=9)

    ax0.text(-0.95, -0.12, L_lin[80] + 4 * lift, r"$\gamma_{\mathrm{lin}}$",
             fontsize=8, color=C_LIN, ha="center", zorder=9,
             path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    ax0.text(0.72, 0.86, L_geo[210] - 3 * lift, r"$\gamma_{g}$",
             fontsize=8, color=C_GEO, ha="center", zorder=9,
             path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    ax0.annotate(r"unit balls of $G_F=F+\lambda I$", (0.50, 0.075),
                 xycoords="axes fraction", fontsize=6.8, color=C_MET,
                 ha="center", va="bottom")


    # Manual colorbar placement to close the empty gap on the right
    cax = fig.add_axes([0.85, 0.35, 0.025, 0.32])
    cb = fig.colorbar(surf, cax=cax)
    cb.outline.set_visible(False); cb.set_ticks([])
    cb.ax.set_title("loss", fontsize=6.2, color=style.MUTED, pad=3)

    ax0.set_xlim(*XLIM); ax0.set_ylim(*YLIM); ax0.set_zlim(zlo - 0.05 * (zhi - zlo), zhi)
    ax0.set_box_aspect((4.1, 2.7, 2.6), zoom=1.38)   # phong truc z de doc duoc do cao
    ax0.view_init(elev=27, azim=-70)
    ax0.set_axis_off()
    ax0.annotate("parameter space (2-D slice)", (0.50, -0.045),
                 xycoords="axes fraction", fontsize=7.5, color=style.INK,
                 ha="center")
    # Canh bao nam TRONG hinh: reviewer luot hinh truoc khi doc caption.
    ax0.set_title("SCHEMATIC \u00b7 analytic two-well model, not measured data",
                  loc="center", fontsize=6.5, color=style.MUTED, pad=2)
    # C.panel(ax0, "a", dx=0.038, dy=1.005) # Khong can label (a) neu chi co 1 hinh



    print(r"  \caption{\textbf{Schematic; no experimental data appears in this "
          r"figure.} The loss field is the analytic two-well model "
          r"$\mathcal{L}=1-e^{-d^{2}/2\sigma^{2}}+\text{bowl}-\text{wells}$ on "
          r"$\mathbb{R}^{2}$. Within that model everything is computed, not drawn: "
          r"the metric is $G=(1+\beta\mathcal{L})(I+\kappa\nabla\mathcal{L}"
          r"\nabla\mathcal{L}^{\top})$, the geodesic $\gamma_g$ minimises the "
          r"discrete path energy, and the rings are its true unit balls. "
          r"The chord $\gamma_{\mathrm{lin}}$ crosses the ridge while the "
          r"geodesic follows the valley; markers are path pivots.}")
    style.save(fig, "fig0_geometry_schematic")
    


if __name__ == "__main__":
    main()
