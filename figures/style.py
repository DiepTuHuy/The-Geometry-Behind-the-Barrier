# style.py -- ngon ngu thi giac dung chung cho MOI figure cua paper.
#
# Muc tieu: hinh nhin phai giong nhu no MOC RA TU bai bao, khong phai dan vao.
# Ba thu quyet dinh dieu do:
#   1. Chu trong hinh cung ho voi chu trong bai  -> serif kieu Times (STIX Two Text),
#      cong thuc toan dung STIX -> khop voi \usepackage{times} cua template ICLR.
#   2. Truc TACH ROI (spine offset), bo top/right, tick huong ra -> khoang tho.
#   3. Khong legend hop; dat NHAN TRUC TIEP o cuoi duong.
#
# Import mot lan o dau moi script ve:  import style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont, FontProperties

# ---------------------------------------------------------------- kho giay ICLR (inch)
W_FULL = 5.50    # het be ngang trang
W_2_3  = 3.60
W_HALF = 2.70

# ---------------------------------------------------------------- chon font serif
def _pick_serif():
    """Uu tien font giong LaTeX/Times nhat co tren may."""
    for cand in ("STIX Two Text", "STIXGeneral", "Times New Roman", "Charter", "DejaVu Serif"):
        try:
            if findfont(FontProperties(family=cand), fallback_to_default=False):
                return cand
        except Exception:
            continue
    return "DejaVu Serif"

SERIF = _pick_serif()

# ---------------------------------------------------------------- mau
# Okabe-Ito: an toan voi nguoi mu mau, in den trang van phan biet duoc.
OKABE_ITO = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
    "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000",
}

# MAU = activation. Co dinh o MOI hinh trong paper.
COLOR_ACT = {
    "relu":     "#D55E00",
    "gelu":     "#0072B2",
    "tanh":     "#009E73",
    "swish":    "#CC79A7",
    "softplus": "#E69F00",
}
MARKER_ACT = {"relu": "s", "gelu": "o", "tanh": "^", "swish": "D", "softplus": "v"}

# MARKER = regime. Quy uoc TOAN CUC:  mau -> activation,  marker -> regime.
# Nho tach hai kenh nhu vay, hinh nao gop ca ba regime vao mot truc van doc duoc,
# va ban in den trang cung khong mat thong tin.
MARKER_REGIME = {"ntk": "o", "sp": "s", "mup": "^"}

# =========================== QUY UOC MA HOA TOAN CUC ===========================
#   MAU    = che do tham so hoa   (bien mang KET QUA: giai thich 75.8% phuong sai
#            cua alpha_B; activation chi 1.4%, kien truc 6.4%)
#   MARKER = kien truc            (MLP o / CNN square / TS triangle)
#   DUONG MANH cung mau = activation  (chieu KIEM CHUNG DO BEN, khong phai tuong phan)
#   NGOAI LE: hinh nao chi ve MOT che do thi mau roi ve activation (vd fig6).
#
# Ba mau che do duoc chon theo BA rang buoc:
#   (i) che do la bien CO THU TU (do dich trong so: NTK 0.011-0.072 < standard
#       0.15-1.11 < muP 1.48-2.96, Muc 7) -> mau di tu lanh sang am;
#   (ii) an toan mu mau: cap xanh-cam la cap ben nhat voi moi dang CVD;
#   (iii) do sang khac nhau -> ban in den trang van tach duoc NTK khoi hai cai kia.
COLOR_REGIME     = {"ntk": "#0B3C5D", "sp": "#3E8FC1", "mup": "#D8741A"}
# Du phong cho ban den trang khi hai duong co the cat nhau (fig1 thi khong can:
# ba che do cach nhau nhieu bac tren truc log).
LINESTYLE_REGIME = {"ntk": "-", "sp": (0, (5, 1.6)), "mup": (0, (1.4, 1.4))}
MARKER_MODE      = {"mlp": "o", "cnn": "s", "ts": "^"}

LABEL_REGIME = {"ntk": "NTK", "sp": "Standard", "mup": r"$\mu$P"}
LABEL_MODE   = {"mlp": "MLP / MNIST", "cnn": "CNN / Fashion-MNIST", "ts": "Teacher\u2013student"}
# ban dung trong \caption{}: LaTeX go "--" thanh en-dash, khong nhan ky tu Unicode.
LABEL_MODE_TEX = {k: v.replace("\u2013", "--") for k, v in LABEL_MODE.items()}
ORDER_REGIME = ["ntk", "sp", "mup"]
ORDER_ACT    = ["relu", "gelu", "tanh", "swish", "softplus"]

# muc xam dung nhat quan: INK cho chu chinh, MUTED cho phu, HAIR cho luoi
INK   = "#1a1a1a"
MUTED = "#6b6b6b"
HAIR  = "#c9ccd1"
WASH  = "#eef0f3"

RC = {
    # --- chu ---
    "font.family": "serif",
    "font.serif": [SERIF, "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "axes.labelcolor": INK,
    "text.color": INK,
    "axes.titlepad": 5.0,
    "axes.labelpad": 3.0,

    # --- truc tach roi, khong khung ---
    "axes.edgecolor": INK,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.size": 2.8, "ytick.major.size": 2.8,
    "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,

    # --- luoi: mac dinh TAT. Bat rieng, that mo, chi khi that su can. ---
    "axes.grid": False,
    "grid.color": HAIR,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.7,

    # --- duong & diem ---
    "lines.linewidth": 1.15,
    "lines.markersize": 3.0,
    "lines.markeredgewidth": 0.6,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",
    "patch.linewidth": 0.6,

    # --- legend khi buoc phai dung ---
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.3,
    "legend.columnspacing": 1.1,
    "legend.borderpad": 0.0,

    # --- bo cuc & xuat file ---
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.03,
    "figure.constrained_layout.w_pad": 0.03,
    "figure.constrained_layout.hspace": 0.03,
    "figure.constrained_layout.wspace": 0.03,
    "savefig.dpi": 400,
    "savefig.facecolor": "white",
    # KHONG dung bbox="tight": moi hinh se bi cat mot luong khac nhau, dan vao
    # LaTeX voi width=\linewidth thi co chu moi hinh mot khac. Giu dung kho da
    # khai bao va de constrained_layout lo phan le.
    "savefig.bbox": None,
    "pdf.fonttype": 42,     # font nhung duoc vao PDF -- ICLR/NeurIPS yeu cau
    "ps.fonttype": 42,
}
plt.rcParams.update(RC)


def save(fig, name, outdir="out"):
    """Luu ca PDF (dua vao LaTeX) va PNG (slide, README)."""
    import os
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p)
        print("  ->", p)
