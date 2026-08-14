# common.py -- lop TRUY CAP DU LIEU + cac manh ve dung lai.
#
# LUAT: file nay CHI DOC CSV. Khong train, khong do lai, khong ghi de bat cu CSV nao.
# Moi script figure chi goi ham o day roi ve.
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- CSV map
# Every path the figure layer will ever read is declared HERE and nowhere else.
# No fig*.py may hardcode a path. Move the data, edit this block, done.
PATHS = {
    # per pair: barrier; per net: dF            (column `kind` = "pair" / "net")
    "combined":   {m: os.path.join(ROOT, "data", "train", f"{m}_combined.csv")
                   for m in ("mlp", "cnn", "ts")},
    # the CNN combined file above is already the dF-corrected one; this alias is
    # kept so load_pairs()/load_nets() need no change.
    "combined_cnn_fixed": os.path.join(ROOT, "data", "train", "cnn_combined.csv"),

    # per cell: dF, dev_rel, flen + the alpha_* exponents
    "cell_geo":   {m: os.path.join(ROOT, "data", "geodesic", f"{m}_cells.csv")
                   for m in ("mlp", "cnn", "ts")},
    # per pair, geodesic round: flen/rq at t = 0, 1/2, 1 and dev_rel
    "geo_pair":   {m: os.path.join(ROOT, "data", "geodesic", f"{m}_pairs.csv")
                   for m in ("mlp", "cnn", "ts")},

    # per cell: B, t_star, rho_*, R_end / R_mid / R_tstar
    "cell_final": {m: os.path.join(ROOT, "data", "final", f"{m}_cells.csv")
                   for m in ("mlp", "cnn", "ts")},
    # per pair, final round (full detail)
    "pair_final": {m: os.path.join(ROOT, "data", "final", f"{m}_pairs.csv")
                   for m in ("mlp", "cnn", "ts")},

    # damping sweep (MLP only), one file per regime
    "lam_sweep":  {r: os.path.join(ROOT, "data", "final", f"lambda_sweep_{r}_cells.csv")
                   for r in ("ntk", "sp", "mup")},
    "decompose":  os.path.join(ROOT, "data", "final", "decompose_mlp.csv"),
}

MODES = ("mlp", "cnn", "ts")


def _num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load(path, numeric=()):
    if not os.path.exists(path):
        raise FileNotFoundError(f"khong thay CSV: {path}  (kiem tra PATHS trong common.py)")
    return _num(pd.read_csv(path), numeric)


# ---------------------------------------------------------------- cac loader
def load_pairs(mode):
    """Barrier theo tung cap (kind == 'pair')."""
    p = PATHS["combined_cnn_fixed"] if mode == "cnn" else PATHS["combined"][mode]
    df = load(p, ("width", "barrier", "acc", "dF_op", "wmove"))
    return df[df.kind.astype(str) == "pair"].copy()


def load_nets(mode):
    """dF_op / acc theo tung mang da train (kind == 'net')."""
    p = PATHS["combined_cnn_fixed"] if mode == "cnn" else PATHS["combined"][mode]
    df = load(p, ("width", "acc", "dF_op", "wmove"))
    return df[df.kind.astype(str) == "net"].copy()


def load_cell_geo(mode):
    """Mot dong = mot o (regime x act x width): dF_med, dev_rel_med, flen_mid_med, alpha_*."""
    return load(PATHS["cell_geo"][mode],
                ("width", "acc_med", "dF_med", "dF_q1", "dF_q3",
                 "dev_rel_med", "dev_rel_q1", "dev_rel_q3", "dev_rel2_med",
                 "gamma_mid_med", "flen_mid_med", "flen_mid_q1", "flen_mid_q3",
                 "rq_mid_med", "fd_instab_med",
                 "alpha_dF", "alpha_dF_r2", "alpha_devrel2", "alpha_devrel2_r2",
                 "alpha_flen", "alpha_flen_r2"))


def load_cell_final(mode):
    """Mot dong = mot o: B, t_star, rho_*, flen_*, R_end/R_mid/R_tstar."""
    return load(PATHS["cell_final"][mode],
                ("width", "n_pairs", "B", "t_star", "rho_A", "rho_mid", "rho_max",
                 "rho_at_tstar", "acc_mid", "flen_A", "flen_mid", "flen_B",
                 "flen_at_tstar", "rq_mid", "R_end", "R_mid", "R_tstar", "fd_instab"))


def load_pair_final(mode):
    df = load(PATHS["pair_final"][mode],
              ("width", "B", "t_star", "rho_mid", "flen_mid", "R_end", "R_mid",
               "R_tstar", "devrel_lam1e-1", "devrel_lam1e-2", "devrel_lam1e-3",
               "cg_resid", "fd_instab"))
    return df[df.get("status", "ok").astype(str) == "ok"].copy() if "status" in df else df


def load_lam_sweep(regime):
    return load(PATHS["lam_sweep"][regime],
                ("width", "devrel_1e-1", "devrel_1e-2", "devrel_1e-3"))


def load_decompose():
    """Phan ra dF = Gauss-Newton + van chuyen softmax (Menh de 4.4 / Phu luc A).

      gn_op        = ||d_z Ftilde||_op        (S dong bang -- phan DINH LY phu)
      transport_op = ||E[J^T (d_z S) J]||_op  (van chuyen softmax -- phan KHONG phu)
      rho_S        = E_x ||S(p_w(x))||_op  in [0, 1/2]
      tr_frac      = transport_op / dF_op     (da kiem: khop toi 5e-5)
    """
    return load(PATHS["decompose"],
                ("width", "dF_op", "gn_op", "transport_op", "rho_S", "tr_frac"))


def load_geo_pairs(mode):
    """param_geo_*.csv theo tung cap: flen/rq tai t=0, 0.5, 1 + dev_rel + dnorm."""
    df = load(PATHS["geo_pair"][mode],
              ("width", "dnorm", "dev_geo", "dev_rel", "gamma_mid",
               "flen_A", "flen_mid", "flen_B", "rq_A", "rq_mid", "rq_B",
               "lam_rel", "cg_resid", "fd_instab"))
    df = df[df.status.astype(str) == "ok"]
    if "lam_rel" in df and df.lam_rel.notna().any():   # chi giu damping chinh
        df = df[np.isclose(df.lam_rel, 1e-2)]
    return df.copy()


# ================================================================ BANG DAN XUAT
# Hai bang duoi day tai lap dung so lieu cua paper; moi figure deu doc tu day
# de khong bao gio co hai hinh dung hai cach gop khac nhau.

#: he so cua du bao bac hai tai TRUNG DIEM theo Singh et al.:
#: B ~ [alpha(1-alpha)/2] D^T Hess D, tai alpha=1/2 la (1/8) D^T F D.
#: flen = (1/2) D^T F D  =>  (1/8) D^T F D = flen/4  =>  R = 4B/flen.
R_FROM_FLEN = 4.0


def barrier_cells(mode):
    """Trung vi barrier theo o (regime, act, width) -- nguon cua moi so mu alpha_B.

    Dung combined_p*.csv: DA KIEM la file tai lap dung Figure 1 cua paper
    (R^2 = 0.20 / 0.11 / 0.02 / 0.91, he so goc 1.12).
    """
    p = load_pairs(mode)
    return (p.groupby(["regime", "act", "width"]).barrier
             .median().rename("B").reset_index())


def cell_table():
    """216 o (regime x act x width x 3 kien truc): B, flen, rho*, R_mid, R_end."""
    out = []
    for mode in MODES:
        B = barrier_cells(mode).set_index(["regime", "act", "width"]).B
        g = load_geo_pairs(mode).groupby(["regime", "act", "width"])[
            ["flen_A", "flen_mid", "flen_B", "rq_mid", "dev_rel"]].median()
        f = load_cell_final(mode).set_index(["regime", "act", "width"])[
            ["rho_A", "rho_mid", "acc_mid", "t_star"]]
        t = pd.concat([B, g, f], axis=1).dropna(subset=["flen_mid"]).reset_index()
        t["mode"] = mode
        t["flen_end"] = 0.5 * (t.flen_A + t.flen_B)
        t["R_mid"] = R_FROM_FLEN * t.B / t.flen_mid
        t["R_end"] = R_FROM_FLEN * t.B / t.flen_end
        out.append(t)
    return pd.concat(out, ignore_index=True)


def exponent_table():
    """36 o (kien truc x che do x activation): so mu theo width cua moi dai luong.

    Quy uoc dau: alpha > 0 = GIAM theo width (giong Figure 1 cua paper).
    """
    rows = []
    for mode in MODES:
        B = barrier_cells(mode)
        g = load_cell_geo(mode)
        for (reg, act), s in g.groupby(["regime", "act"]):
            s = s.sort_values("width")
            sb = B[(B.regime == reg) & (B.act == act)].sort_values("width")
            rows.append(dict(
                mode=mode, regime=reg, act=act,
                alpha_B=fit_alpha(sb.width.values, sb.B.values)[0],
                alpha_dF=fit_alpha(s.width.values, s.dF_med.values)[0],
                alpha_dev=fit_alpha(s.width.values, s.dev_rel_med.values)[0],
                alpha_ray=fit_alpha(s.width.values, s.rq_mid_med.values)[0],
                alpha_len=fit_alpha(s.width.values, s.flen_mid_med.values)[0]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- thong ke
def med_iqr(df, group, value):
    """Gop thanh median + Q1 + Q3. Dung o MOI hinh cho nhat quan."""
    g = df.groupby(group)[value]
    out = g.agg(med="median",
                lo=lambda s: s.quantile(.25),
                hi=lambda s: s.quantile(.75),
                n="count").reset_index()
    return out


def fit_alpha(width, value):
    """Fit y ~ w^(-alpha) tren log-log. Tra ve (alpha, R^2, n_diem).

    Quy uoc DAU giong het code do cua ban (measure_final / run_geo):
    alpha DUONG = dai luong GIAM khi width tang.
    """
    w = np.asarray(width, float)
    y = np.asarray(value, float)
    ok = (w > 0) & (y > 0) & np.isfinite(w) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan"), int(ok.sum())
    lw, ly = np.log(w[ok]), np.log(y[ok])
    b, a = np.polyfit(lw, ly, 1)
    resid = ly - (a + b * lw)
    ss = ((ly - ly.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / max(ss, 1e-30)
    return -b, float(r2), int(ok.sum())


# ================================================================ NGU PHAP VE
# Cac ham duoi day la "tu vung tao hinh" cua paper. Moi figure chi duoc dung
# nhung tu nay, nho vay 8 hinh nhin nhu do MOT nguoi ve trong MOT buoi.

def grid_with_label_gutter(fig, nrow, ncol, *, gutter=0.30, sharex=True,
                           sharey="row", **gs_kw):
    """Luoi truc + MOT COT DEM ben phai danh cho nhan cuoi duong.

    Vi nhan dat truc tiep nam NGOAI truc, phai chua san cho cho no; neu khong
    constrained_layout se bop truc lai hoac nhan bi cat mat khi xuat PDF.
    """
    gs = fig.add_gridspec(nrow, ncol + 1,
                          width_ratios=[1.0] * ncol + [gutter], **gs_kw)
    axes = np.empty((nrow, ncol), dtype=object)
    for i in range(nrow):
        for j in range(ncol):
            kw = {}
            if sharex and i > 0:
                kw["sharex"] = axes[0, j]
            if sharey == "row" and j > 0:
                kw["sharey"] = axes[i, 0]
            elif sharey is True and (i or j):
                kw["sharey"] = axes[0, 0]
            axes[i, j] = fig.add_subplot(gs[i, j], **kw)
            if sharex and i < nrow - 1:
                axes[i, j].tick_params(labelbottom=False)
            if sharey and j > 0:
                axes[i, j].tick_params(labelleft=False)
    gut = fig.add_subplot(gs[:, ncol])
    gut.set_axis_off()
    return axes, gut


def despine(ax, offset=4, which=("left", "bottom")):
    """Tach truc ra khoi vung du lieu -- thu tao ra cam giac 'thoang' cua hinh dep."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in which:
        ax.spines[side].set_position(("outward", offset))


def boxed(ax, minor=True):
    """Khung dong kin 4 canh, tick huong VAO TRONG o ca 4 canh.

    Day la lối tap chi vat ly (PRE/PRL) ma ca hai bai tham chieu deu dung.
    Nguoc voi despine(): dung cai nay khi muon hinh "nang" va day dan hon.
    """
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_position(("outward", 0))
        sp.set_linewidth(0.6)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if minor:
        ax.minorticks_on()
        ax.tick_params(which="minor", direction="in", top=True, right=True)


def hgrid(ax, alpha=0.55):
    """Chi luoi NGANG, that mo. Luoi doc gan nhu luon la nhieu thi giac."""
    import style
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=style.HAIR, lw=0.4, alpha=alpha)
    ax.xaxis.grid(False)


def panel(ax, letter, dx=-0.02, dy=1.0):
    """Chu (a) (b) (c) dat ngoai truc, kieu tap chi. Chay duoc ca voi truc 3D."""
    import style
    # Axes3D.text() doi (x, y, z); ban 2D cua no la text2D.
    put = getattr(ax, "text2D", ax.text)
    put(dx, dy, f"({letter})", transform=ax.transAxes,
        fontsize=8.5, fontweight="bold", color=style.INK,
        ha="right", va="bottom")


def curve(ax, x, med, lo=None, hi=None, *, act=None, regime=None,
          label=None, color=None, marker=None, **kw):
    """Duong median + dai IQR, mau/marker theo dung quy uoc trong style.py.

    Marker to vien trang -> cac diem chong nhau van tach bach.
    """
    import style
    color = color if color is not None else style.COLOR_ACT.get(act, style.MUTED)
    if marker is None:      # marker -> regime; neu hinh khong noi regime thi theo act
        marker = (style.MARKER_REGIME.get(regime) if regime
                  else style.MARKER_ACT.get(act)) or "o"
    ls = kw.pop("linestyle", style.LINESTYLE_REGIME.get(regime, "-"))
    x = np.asarray(x, float); med = np.asarray(med, float)
    keep = np.isfinite(med)
    if not keep.any():
        # o khong co phep do nao (vd relu bi loai khoi dF vi ham co kink):
        # KHONG ve artist rong -- bbox NaN cua no lam hong bo cuc ca hinh.
        return color
    ax.plot(x[keep], med[keep], marker=marker, color=color, linestyle=ls,
            markerfacecolor=color, markeredgecolor="white",
            label=label, zorder=3, clip_on=False, **kw)
    if lo is not None and hi is not None:
        lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        ax.fill_between(x[keep], lo[keep], hi[keep], color=color,
                        alpha=.13, lw=0, zorder=1)
    return color


def end_labels(ax, items, *, pad=0.018, fontsize=7.0, min_gap=0.052):
    """NHAN TRUC TIEP o cuoi duong, thay cho legend hop.

    items: [(y_cuoi_theo_don_vi_du_lieu, text, color), ...]
    Tu day cac nhan ra xa nhau theo truc doc de khong de len nhau.
    """
    if not items:
        return
    inv = ax.transAxes.inverted()
    tr = ax.transData
    rows = []
    for y, text, color in items:
        if y is None or not np.isfinite(y):
            continue
        yax = inv.transform(tr.transform((ax.get_xlim()[1], y)))[1]
        rows.append([yax, text, color])
    rows.sort(key=lambda r: r[0])
    for i in range(1, len(rows)):                      # day len
        if rows[i][0] - rows[i - 1][0] < min_gap:
            rows[i][0] = rows[i - 1][0] + min_gap
    over = rows[-1][0] - 1.0 if rows and rows[-1][0] > 1.0 else 0.0
    for r in rows:                                      # keo ca cum ve trong khung
        r[0] -= over
    for yax, text, color in rows:
        ax.text(1.0 + pad, yax, text, transform=ax.transAxes, color=color,
                fontsize=fontsize, va="center", ha="left", clip_on=False)


def powerlaw_line(ax, x, y, *, color=None, lo=None, hi=None, extend=1.12):
    """Ve duong khop luy thua y ~ x^(-alpha) tren truc log-log. Tra ve (alpha, r2)."""
    import style
    alpha, r2, n = fit_alpha(x, y)
    if not np.isfinite(alpha):
        return alpha, r2
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    lw_, ly_ = np.log(x[ok]), np.log(y[ok])
    b, a = np.polyfit(lw_, ly_, 1)
    xs = np.array([x[ok].min() / extend, x[ok].max() * extend])
    ax.plot(xs, np.exp(a) * xs ** b, color=color or style.MUTED,
            lw=0.7, ls=(0, (3.5, 2)), zorder=2, alpha=0.85)
    return alpha, r2


def slope_tag(ax, text, xy, *, color=None, fontsize=6.8, **kw):
    """Nhan do doc dat sat duong, khong khung, khong mui ten."""
    import style
    ax.annotate(text, xy=xy, xycoords="axes fraction", fontsize=fontsize,
                color=color or style.MUTED, **kw)


def log_decades(ax, axis="y"):
    """Tick log chi o cac bac 10, tick phu khong nhan -> truc log sach."""
    from matplotlib.ticker import LogLocator, NullFormatter
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10))
    a.set_minor_locator(LogLocator(base=10, subs=tuple(np.arange(2, 10) * 0.1)))
    a.set_minor_formatter(NullFormatter())


def wlabel(w):
    """64 -> '64', 1024 -> '1k', 4096 -> '4k'. Tranh nhan truc chong nhau."""
    w = int(w)
    return f"{w // 1024}k" if w >= 1024 and w % 1024 == 0 else str(w)


def zero_line(ax):
    """Vach 0 mo cho cac dai luong co the sat 0 (barrier)."""
    import style
    ax.axhline(0, color=style.HAIR, lw=0.6, zorder=0)


def act_handles(acts=None):
    """Chu thich MAU = activation. Khong deo marker: marker la kenh cua regime,
    de lan vao day se mau thuan voi cai dang ve tren truc."""
    import style
    from matplotlib.lines import Line2D
    acts = acts or style.ORDER_ACT
    return [Line2D([], [], color=style.COLOR_ACT[a], lw=1.8, linestyle="-",
                   label=a) for a in acts]


def regime_handles(regimes=None, dashed=False):
    """Chu thich MAU = che do (quy uoc toan cuc).

    dashed=True: hien luon KIEU NET, de chu thich khop voi duong tren truc.
    ICML yeu cau "figures should not only use colors to distinguish curves",
    nen o dau che do dung chung mot truc thi kieu net la kenh thu hai bat buoc.
    """
    import style
    from matplotlib.lines import Line2D
    regimes = regimes or style.ORDER_REGIME
    return [Line2D([], [], color=style.COLOR_REGIME[r], lw=2.0,
                   linestyle=style.LINESTYLE_REGIME[r] if dashed else "-",
                   label=style.LABEL_REGIME[r]) for r in regimes]


def mode_handles(modes=None):
    """Chu thich MARKER = kien truc. Mau xam de khong tranh voi kenh mau."""
    import style
    from matplotlib.lines import Line2D
    modes = modes or list(MODES)
    return [Line2D([], [], color=style.MUTED, marker=style.MARKER_MODE[m],
                   markeredgecolor="white", mew=0.55, linestyle="none",
                   label=style.LABEL_MODE[m].split(" /")[0]) for m in modes]
