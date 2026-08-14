#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 run_lambda_sweep_mup.py  --  NHAN CHAY LA XONG, KHONG SUA GI
============================================================================
     python run_lambda_sweep_mup.py
============================================================================

KHAC BAN CU O DIEM NAO (chi 3 diem, phan tinh toan GIU NGUYEN 100%):
  1. TU KHOI PHUC CSV: dau moi lan chay, script tu copy
     /kaggle/input/datasets/ANONYMIZED/DATASET/lam_sweep_mlp_mup.csv
     ve /kaggle/working roi moi bat dau -> nhan chay la resume ngay,
     khong can go lenh cp. Ban cu KHONG lam viec nay nen no chay lai tu dau.
     Kem theo: tu cat bo dong cuoi bi ghi do dang (neu session truoc bi giet
     dung luc dang ghi) de pandas khong hong khi doc.
  2. BO QUA O DA XONG TRUOC KHI LOAD .pt: ban cu load ca 5 checkpoint roi
     moi phat hien khong con gi de lam -- o w4096 mat vai chuc giay vo ich
     moi o. Ban nay kiem tra truoc, o nao xong thi bo qua trong tich tac.
  3. DUNG CHU DONG khi gan het gio session (BUDGET_H) de CSV kip flush,
     thay vi bi giet ngang.
  Cong thuc, tham so, thu tu vong lap, khoa resume: KHONG DOI. Ket qua
  chay tiep ghep lien mach voi so lieu cu.

MUC DICH KHOA HOC (khong doi):
  Quet damping cho dev_rel trong che do MUP (MLP/MNIST).
  Ban NTK cho: so mu dev_rel KHONG bat bien theo damping -- tut deu tu
  lam=1e-1 xuong lam=1e-3, trung binh 0.236 +- 0.032 tren 12 cap, 12/12
  cung dau. Cau hoi cua file nay: do tut do co GIONG NHAU giua cac che do
  khong? Neu offset chung cho moi o -> R^2 cua hoi quy
  alpha_B ~ alpha_devrel bat bien duoi phep tinh tien chung -> ket qua phu
  dinh R^2=0.11 (36 o) khong bi damping dung toi. Neu khac nhau -> phai
  phat bieu lai ket qua devrel trong Figure 1.
  Cau hoi phu: DAU so mu co on dinh qua 3 bac damping khong.

DAU RA:
  lam_sweep_mlp_mup.csv        theo cap (resumable)
  lam_sweep_mlp_mup_cell.csv   median theo o
  + BAO CAO cuoi log: bang so mu 3 lambda, do tut theo cap, mu_eff/lambda,
    va VERDICT.

CAN CO:
  ckpt_pmlp_v2/mup_{act}_w{w}_s{s}.pt   va  MNIST tai ./data
  (khong can nhan y: Fisher khong dung nhan)

SAU KHI CHAY XONG (hoac het gio): Save Version de CSV thanh output, roi
them output do vao dataset -- lan sau nhan chay la tiep tuc dung cho.
============================================================================
"""
import os, sys, time, math, glob, shutil, itertools, traceback
try:
    sys.stdout.reconfigure(line_buffering=True); sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.func import functional_call, jvp as _fjvp, vjp as _fvjp, jacrev as _jacrev, grad as _grad

# ======================= TAT CA THAM SO DA CHOT SAN =========================
MODE      = "mlp"
RUN_TAG   = "pmlp_v2"
REGIMES   = ["mup"]                # <-- CHO DUY NHAT khac 2 file kia
ACTS      = ["gelu","tanh","swish","softplus"]
WIDTHS    = [64,128,256,512,1024,2048,4096]
NSEEDS    = 5
PAIRS     = 3                            # 3 cap/o -- GIU NGUYEN de so voi NTK
LAM_RELS  = [1e-1, 1e-2, 1e-3]           # lambda = LAM_REL * ||F||_op

# --- so lieu tham chieu tu ban NTK da chay (de so sanh offset) ---
NTK_DROP_MEAN = 0.236                    # trung binh (so mu@1e-1 - so mu@1e-3)
NTK_DROP_SD   = 0.032                    # sd tren 12 cap
NTK_REF_NAMES = ["lam_sweep_mlp_ntk.csv", "lam_sweep_mlp.csv"]              # ten file NTK de doc truc tiep, neu co

TGRID     = 9                            # trung luoi Green cua measure_geo
FISHER_N  = 2048
MICRO     = 64
FD_EPS    = 3e-3                         # GIU NGUYEN tuyet doi = ban NTK
CG_ITERS  = 300
CG_TOL    = 1e-6
POWER_ITERS = 20
RESUME    = True
BUDGET_H  = 11.0                         # dung sach truoc khi session bi giet

DATA_ROOT  = "/kaggle/input/datasets/ANONYMIZED/DATASET"   # noi chua CSV cu + ckpt
CKPT_ROOTS = [DATA_ROOT, ".", "/kaggle/input", "/content", "/content/drive/MyDrive"]
OUT_DIR   = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUT_NAME  = "lam_sweep_mlp_mup.csv"
ALT_NAMES = []                # ten cu khac cua CHINH file nay, neu co
OUT_CSV   = os.path.join(OUT_DIR, OUT_NAME)
CELL_CSV  = os.path.join(OUT_DIR, "lam_sweep_mlp_mup_cell.csv")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DIN, K, BASE = 784, 10, 64
T_START   = time.time()
# ============================================================================

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def set_seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ------------------------------------------------------------------ MODEL
def make_act(n):
    return {"relu":nn.ReLU,"gelu":nn.GELU,"tanh":nn.Tanh,"swish":nn.SiLU,"softplus":nn.Softplus}[n]()

def param_cfg(regime, fin, fout, kind):
    ss = math.sqrt(fin)
    if regime == "sp":  return (1.0/ss, 1.0, 1.0)
    if regime == "ntk": return (1.0, 1.0/ss, 1.0)
    if regime == "mup":
        if kind == "input":  return (1.0/ss, 1.0, (fout/BASE)**1.0)
        if kind == "hidden": return (1.0/ss, 1.0, (fin/BASE)**0.7)
        return (1.0/ss, (BASE/fin)**0.5, (fin/BASE)**(-0.5))
    raise ValueError(regime)

class ScaledLinear(nn.Module):
    def __init__(self, fin, fout, regime, kind):
        super().__init__()
        istd, self.fmul, self.lr_scale = param_cfg(regime, fin, fout, kind)
        self.weight = nn.Parameter(torch.randn(fout, fin)*istd)
        self.bias   = nn.Parameter(torch.zeros(fout))
    def forward(self, x): return self.fmul*F.linear(x, self.weight) + self.bias

class NetMLP(nn.Module):
    def __init__(self, width, act, regime="ntk", din=DIN, k=K):
        super().__init__()
        self.fc1 = ScaledLinear(din, width, regime, "input")
        self.fc2 = ScaledLinear(width, width, regime, "hidden")
        self.fc3 = ScaledLinear(width, k, regime, "output")
        self.a1 = make_act(act); self.a2 = make_act(act)
    def forward(self, x): return self.fc3(self.a2(self.fc2(self.a1(self.fc1(x)))))

def build_net(width, act, regime): return NetMLP(width, act, regime)

# ------------------------------------------------------------------ PERM
def perm_spec(model):
    ag = {"fc1.weight":["h1",None],"fc1.bias":["h1"],
          "fc2.weight":["h2","h1"],"fc2.bias":["h2"],
          "fc3.weight":[None,"h2"],"fc3.bias":[None]}
    sd = model.state_dict(); gs = {}
    for n, axes in ag.items():
        for a, g in enumerate(axes):
            if g is not None: gs[g] = sd[n].shape[a]
    return ag, gs

def apply_perm(sd, ag, perms):
    out = {}
    for n, t in sd.items():
        if n in ag:
            tt = t
            for a, g in enumerate(ag[n]):
                if g is not None: tt = tt.index_select(a, perms[g])
            out[n] = tt.clone()
        else:
            out[n] = t.clone()
    return out

def _perm_except(t, axes, perms, exc):
    tt = t
    for a, g in enumerate(axes):
        if g is not None and a != exc: tt = tt.index_select(a, perms[g])
    return tt

def weight_matching(ag, gs, sdA, sdB, iters=8, seed=0):
    rng = np.random.RandomState(seed); perms = {g: torch.arange(n) for g, n in gs.items()}
    g2pa = {g: [] for g in gs}
    for n, axes in ag.items():
        for a, g in enumerate(axes):
            if g is not None: g2pa[g].append((n, a))
    groups = list(gs)
    for _ in range(iters):
        moved = 0
        for g in [groups[i] for i in rng.permutation(len(groups))]:
            n = gs[g]; S = torch.zeros(n, n, dtype=torch.float64)
            for (name, axis) in g2pa[g]:
                A = sdA[name].double()
                B = _perm_except(sdB[name].double(), ag[name], perms, axis)
                S += torch.movedim(A, axis, 0).reshape(n, -1) @ torch.movedim(B, axis, 0).reshape(n, -1).T
            new = torch.as_tensor(linear_sum_assignment(-S.numpy())[1], dtype=torch.long)
            if not torch.equal(new, perms[g]): moved += 1
            perms[g] = new
        if moved == 0: break
    return perms

# ------------------------------------------------------------------ DATA (khong can nhan)
_C = {}
def load_X():
    if "X" in _C: return _C["X"]
    import torchvision
    ds = torchvision.datasets.MNIST("./data", train=True, download=True)
    X = ((ds.data.float()/255.0) - 0.1307)/0.3081
    _C["X"] = X.reshape(-1, 784); return _C["X"]

# ------------------------------------------------------------------ PRIMITIVES
def _pb(m):
    return ({k: v.detach() for k, v in m.named_parameters()},
            {k: v.detach() for k, v in m.named_buffers()})
def _call(m, p, b, x): return functional_call(m, {**p, **b}, (x,))
def _vdot(a, b): return float(sum((a[k]*b[k]).sum() for k in a))
def _vnorm(a): return float(torch.sqrt(torch.clamp(sum((a[k]*a[k]).sum() for k in a), min=0)))
def _vscale(a, c): return {k: a[k]*c for k in a}
def _vaxpy(a, c, b): return {k: a[k] + c*b[k] for k in a}

def fisher_vp(m, p, b, x, v, micro):
    B = x.shape[0]; acc = None
    for i in range(0, B, micro):
        xb = x[i:i+micro]
        def f(pp): return _call(m, pp, b, xb)
        logits, Jv = _fjvp(f, (p,), (v,))
        pr = torch.softmax(logits, 1)
        s  = pr*Jv - pr*(pr*Jv).sum(1, keepdim=True)
        JTs = _fvjp(f, p)[1](s)[0]
        acc = {k: JTs[k].detach() for k in JTs} if acc is None else {k: acc[k] + JTs[k].detach() for k in acc}
    return {k: acc[k]/B for k in acc}

def dFz(m, p, b, x, z, v, eps, micro):
    Fp = fisher_vp(m, _vaxpy(p,  eps, z), b, x, v, micro)
    Fm = fisher_vp(m, _vaxpy(p, -eps, z), b, x, v, micro)
    return {k: (Fp[k]-Fm[k])/(2*eps) for k in p}

def quad_scalar(m, p, b, x, delta, micro):
    B = x.shape[0]; total = None
    for i in range(0, B, micro):
        xb = x[i:i+micro]
        def f(pp): return _call(m, pp, b, xb)
        logits, u = _fjvp(f, (p,), (delta,))
        pr = torch.softmax(logits, 1)
        Su = pr*u - pr*(pr*u).sum(1, keepdim=True)
        t = (u*Su).sum(); total = t if total is None else total + t
    return total/B

def grad_quad(m, p, b, x, delta, micro):
    return _grad(lambda pp: quad_scalar(m, pp, b, x, delta, micro))(p)

def gf_vp(m, p, b, x, v, lam, micro):
    Fv = fisher_vp(m, p, b, x, v, micro)
    return {k: Fv[k] + lam*v[k] for k in v}

def lam_max(m, p, b, x, micro, iters, seed=17):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    u = {k: torch.randn(v.shape, generator=gen, device=v.device, dtype=v.dtype) for k, v in p.items()}
    u = _vscale(u, 1.0/max(_vnorm(u), 1e-30)); lam = 0.0
    for _ in range(iters):
        Au = fisher_vp(m, p, b, x, u, micro); lam = _vnorm(Au)
        if lam < 1e-30: break
        u = _vscale(Au, 1.0/lam)
    return lam

def cg_solve(m, p, b, x, rhs, lam, micro, x0=None, iters=CG_ITERS, tol=CG_TOL):
    xk = {k: (torch.zeros_like(v) if x0 is None else x0[k].clone()) for k, v in rhs.items()}
    Ax = gf_vp(m, p, b, x, xk, lam, micro) if x0 is not None else {k: torch.zeros_like(v) for k, v in rhs.items()}
    r = {k: rhs[k]-Ax[k] for k in rhs}; pdir = {k: r[k].clone() for k in r}
    rs = _vdot(r, r); r0 = max(rs, 1e-300)
    for _ in range(iters):
        Ap = gf_vp(m, p, b, x, pdir, lam, micro)
        a = rs/max(_vdot(pdir, Ap), 1e-300)
        xk = {k: xk[k] + a*pdir[k] for k in xk}
        r  = {k: r[k] - a*Ap[k] for k in r}
        rs2 = _vdot(r, r)
        if rs2 <= tol*tol*r0: break
        beta = rs2/max(rs, 1e-300); pdir = {k: r[k] + beta*pdir[k] for k in pdir}; rs = rs2
    return xk, math.sqrt(_vdot(r, r)/r0)

def christoffel_dd(m, p, b, x, delta, lam, micro, x0=None):
    dn = _vnorm(delta); dhat = _vscale(delta, 1.0/dn)
    d1 = dFz(m, p, b, x, dhat, dhat, FD_EPS,   micro)
    d2 = dFz(m, p, b, x, dhat, dhat, FD_EPS/2, micro)
    t1r = {k: (4*d2[k]-d1[k])/3 for k in d1}                    # Richardson O(eps^4)
    fd  = _vnorm({k: d1[k]-d2[k] for k in d1})/max(_vnorm(t1r), 1e-30)
    t1  = {k: t1r[k]*dn*dn for k in t1r}
    mv  = grad_quad(m, p, b, x, delta, micro)
    rhs = {k: 2*t1[k] - mv[k] for k in t1}
    rhs_n = _vnorm(rhs)
    sol, resid = cg_solve(m, p, b, x, rhs, lam, micro, x0=x0)
    return _vscale(sol, 0.5), resid, fd, rhs_n

def green_matrix(ts):
    n = len(ts); G = torch.zeros(n, n, dtype=torch.float64)
    for i, t in enumerate(ts):
        for j, s in enumerate(ts):
            G[i, j] = s*(1-t) if s <= t else t*(1-s)
    return G

def dev_rel_at_lambda(ref, pA, pB, b_ref, Xf, delta, lam, ts):
    gammas = []; resids = []; fds = []; rhsn = []; x0 = None
    for tt in ts:
        pt = {k: (1-tt)*pA[k] + tt*pB[k] for k in pA}
        g, res, fd, rn = christoffel_dd(ref, pt, b_ref, Xf, delta, lam, MICRO, x0=x0)
        x0 = g
        gammas.append({k: v.detach().cpu() for k, v in g.items()}); resids.append(res); fds.append(fd); rhsn.append(rn)
    G = green_matrix(list(ts)); dt = ts[1]-ts[0]; keys = list(delta.keys()); sup = 0.0
    for ti in range(len(ts)):
        xi = None
        for si in range(len(ts)):
            w_ = float(G[ti, si])*dt
            if w_ == 0.0: continue
            xi = {k: (gammas[si][k]*w_ if xi is None else xi[k] + gammas[si][k]*w_) for k in keys}
        sup = max(sup, _vnorm(xi) if xi is not None else 0.0)
    return sup/max(_vnorm(delta), 1e-30), max(resids), max(fds), max(rhsn)

# ------------------------------------------------------------------ SELF-TEST
def self_test():
    log("  [self-test] Christoffel vs dense, hang so Green = 1/8 ...")
    old = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    torch.manual_seed(0); m = NetMLP(6, "tanh", REGIMES[0], din=4, k=3).eval(); x = torch.randn(8, 4)
    p, b = _pb(m); keys = list(p.keys())
    flat = lambda d: torch.cat([d[k].reshape(-1) for k in keys])
    def unflat(v):
        o = {}; i = 0
        for k in keys:
            n = p[k].numel(); o[k] = v[i:i+n].reshape(p[k].shape); i += n
        return o
    torch.manual_seed(3); df = torch.randn(flat(p).numel()); df = df/df.norm()*3.0
    delta = unflat(df); lam = 1e-2
    def Fmat(wv):
        pp = unflat(wv); J = _jacrev(lambda q: _call(m, q, b, x))(pp); B = x.shape[0]
        Jf = torch.cat([J[k].reshape(B, 3, -1) for k in keys], 2)
        pr = torch.softmax(_call(m, pp, b, x), 1)
        S = torch.diag_embed(pr) - pr.unsqueeze(2)*pr.unsqueeze(1)
        return torch.einsum('bki,bkl,blj->ij', Jf, S, Jf)/B
    w0 = flat(p); dF = _jacrev(Fmat)(w0); Fd = Fmat(w0); P = w0.numel()
    t1 = torch.einsum('ijl,l,j->i', dF, df, df); mv = torch.einsum('i,ijl,j->l', df, dF, df)
    g_dense = 0.5*torch.linalg.solve(Fd + lam*torch.eye(P), 2*t1 - mv)
    g_vp, _, _, _ = christoffel_dd(m, p, b, x, delta, lam, micro=8)
    rel = float((flat(g_vp) - g_dense).norm()/max(g_dense.norm(), 1e-12))
    ts = np.linspace(0, 1, 21); G = green_matrix(list(ts)); dt = ts[1]-ts[0]
    xi = (G @ torch.full((len(ts),), 0.7, dtype=torch.float64))*dt
    ge = float((xi - torch.tensor([0.7*t*(1-t)/2 for t in ts])).abs().max())
    torch.set_default_dtype(old)
    log(f"  [self-test] regime={REGIMES[0]}  Gamma rel={rel:.2e}   Green err={ge:.2e}")
    assert rel < 1e-5 and ge < 1e-10, "SELF-TEST HONG -- dung lai"

# ------------------------------------------------------------------ CKPT
_IDX = None
def _build_index():
    global _IDX
    if _IDX is not None: return _IDX
    idx = {}
    for root in CKPT_ROOTS:
        if not root or not os.path.isdir(root): continue
        for dp, _, fns in os.walk(root):
            if os.path.basename(dp) != f"ckpt_{RUN_TAG}": continue
            for fn in fns:
                if fn.endswith(".pt"): idx.setdefault(fn, os.path.join(dp, fn))
    _IDX = idx
    log(f"[ckpt] tim thay {len(idx)} file .pt")
    if not idx:
        log(f"[ckpt] !! KHONG THAY GI. Sua CKPT_ROOTS o dau file. Thu: ls /kaggle/input/*/")
    else:
        pref = sorted({fn.split("_")[0] for fn in idx})
        log(f"[ckpt] tien to (che do) co mat: {pref}")
        for rg in REGIMES:
            n = sum(1 for fn in idx if fn.startswith(rg + "_"))
            if n == 0:
                log(f"[ckpt] !! KHONG CO FILE NAO cho REGIMES='{rg}'.")
                log(f"[ckpt] !! Doi REGIMES o dau file thanh mot trong {pref} roi chay lai.")
            else:
                log(f"[ckpt] '{rg}': {n} file -- OK")
    return idx

def find_ckpt(regime, act, w, s):
    return _build_index().get(f"{regime}_{act}_w{w}_s{s}.pt")

def load_sd(path):
    try: d = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: d = torch.load(path, map_location="cpu")
    return d["sd"]

# ------------------------------------------------------------------ CSV
COLS = ["regime","act","width","seedA","seedB","dnorm","lam_max",
        "devrel_1e-1","devrel_1e-2","devrel_1e-3","rhs_1e-2","cg_resid","fd_instab","status"]

_DONE = None

def _seek_csv(name):
    """tim file CSV theo ten: uu tien dataset da biet, roi quet /kaggle/input, roi cwd."""
    p = os.path.join(DATA_ROOT, name)
    if os.path.exists(p): return p
    for c in sorted(glob.glob("/kaggle/input/**/" + name, recursive=True)):
        return c
    if os.path.exists(name): return os.path.abspath(name)
    return None

def _sanitize_csv():
    """cat bo dong bi ghi do dang (session truoc bi giet giua chung) + dong loi."""
    if not os.path.exists(OUT_CSV): return
    lines = open(OUT_CSV).read().splitlines()
    if not lines: return
    nf = len(lines[0].split(","))
    keep = [l for l in lines if l.strip() and len(l.split(",")) == nf]
    if len(keep) != len(lines):
        open(OUT_CSV, "w").write("\n".join(keep) + "\n")
        log(f"[resume] bo {len(lines)-len(keep)} dong hong/loi -> se chay lai cac cap do")

def restore_csv():
    """RIENG CUA BAN NAY: keo CSV cu tu dataset (read-only) ve OUT_DIR de done() thay."""
    if not os.path.exists(OUT_CSV):
        for name in [OUT_NAME] + ALT_NAMES:
            src = _seek_csv(name)
            if src and os.path.abspath(src) != os.path.abspath(OUT_CSV):
                try:
                    shutil.copy(src, OUT_CSV); log(f"[resume] khoi phuc CSV tu {src}")
                    break
                except Exception as e:
                    log(f"[resume] copy that bai ({e!r}) -> chay tu dau")
    _sanitize_csv()
    if os.path.exists(OUT_CSV):
        n = max(sum(1 for _ in open(OUT_CSV)) - 1, 0)
        log(f"[resume] {OUT_CSV}: {n} dong du lieu san co")
    else:
        log(f"[resume] khong tim thay CSV cu -> chay tu dau (binh thuong neu lan dau)")

def _load_done():
    global _DONE
    _DONE = set()
    if not os.path.exists(OUT_CSV): return
    for ln in open(OUT_CSV):
        f = ln.strip().split(",")
        if len(f) >= 5 and f[-1] == "ok":
            _DONE.add((f[0], f[1], f[2], f[3], f[4]))

def done(regime, act, w, i, j):
    if _DONE is None: _load_done()
    return (str(regime), str(act), str(w), str(i), str(j)) in _DONE

def write_row(row):
    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a") as f:
        if new: f.write(",".join(COLS) + "\n")
        f.write(",".join(str(row.get(c, "")) for c in COLS) + "\n"); f.flush()
        os.fsync(f.fileno())
    if _DONE is not None and row.get("status") == "ok":
        _DONE.add(tuple(str(row[c]) for c in COLS[:5]))

def out_of_time():
    return (time.time() - T_START) > BUDGET_H*3600

# ------------------------------------------------------------------ MAIN
def run():
    X = load_X(); Xf = X[:FISHER_N].to(DEVICE)
    ts = np.linspace(0, 1, TGRID)
    log(f"DEVICE={DEVICE} | {REGIMES[0].upper()} / MLP | {len(ACTS)} act x {len(WIDTHS)} width x {PAIRS} cap x {len(LAM_RELS)} lambda")
    log(f"-> {OUT_CSV}")
    _load_done()
    tot = len(ACTS)*len(WIDTHS)*PAIRS
    left = sum(1 for a in ACTS for w in WIDTHS
               for (i, j) in list(itertools.combinations(range(NSEEDS), 2))[:PAIRS]
               if not done(REGIMES[0], a, w, i, j))
    log(f"[resume] uoc tinh con {left}/{tot} cap chua chay")
    for regime in REGIMES:
        for act in ACTS:
            for w in WIDTHS:
                # --- RIENG CUA BAN NAY: bo qua o da xong TRUOC khi load .pt ---
                avail = [s for s in range(NSEEDS) if find_ckpt(regime, act, w, s) is not None]
                if len(avail) < 2:
                    log(f"  [{regime}/{act}/w{w}] <2 ckpt -> bo"); continue
                pairs = list(itertools.combinations(range(len(avail)), 2))[:PAIRS]
                if RESUME and all(done(regime, act, w, i, j) for (i, j) in pairs):
                    log(f"  [{regime}/{act}/w{w}] da xong ca {len(pairs)} cap -> bo qua"); continue
                if out_of_time():
                    log(f"[budget] het {BUDGET_H}h -> dung sach. Save Version roi chay lai de tiep."); return
                log(f"=== {regime}/{act}/w{w} ===")
                sds = [load_sd(find_ckpt(regime, act, w, s)) for s in avail]
                ref = build_net(w, act, regime).to(DEVICE).eval()
                p_ref, b_ref = _pb(ref)
                ag, gs = perm_spec(build_net(w, act, regime))
                topar = lambda sd: {k: sd[k].to(DEVICE) for k in p_ref.keys()}
                for (i, j) in pairs:
                    if RESUME and done(regime, act, w, i, j): continue
                    if out_of_time():
                        log(f"[budget] het {BUDGET_H}h -> dung sach. Save Version roi chay lai de tiep.")
                        del sds; return
                    t0 = time.time()
                    try:
                        perms = weight_matching(ag, gs, sds[i], sds[j], iters=8, seed=i*13 + j)
                        pA = topar(sds[i]); pB = topar(apply_perm(sds[j], ag, perms))
                        delta = {k: pB[k] - pA[k] for k in pA}
                        lmax = lam_max(ref, pA, b_ref, Xf, MICRO, POWER_ITERS)
                        row = dict(regime=regime, act=act, width=w, seedA=i, seedB=j,
                                   dnorm=f"{_vnorm(delta):.6e}", lam_max=f"{lmax:.6e}", status="ok")
                        msg = []
                        for lr in LAM_RELS:
                            dv, cg, fd, rn = dev_rel_at_lambda(ref, pA, pB, b_ref, Xf, delta,
                                                               max(lr*lmax, 1e-12), ts)
                            tag = {1e-1:"1e-1", 1e-2:"1e-2", 1e-3:"1e-3"}[lr]
                            row[f"devrel_{tag}"] = f"{dv:.6e}"
                            if lr == 1e-2: row["rhs_1e-2"] = f"{rn:.6e}"
                            row["cg_resid"] = f"{cg:.2e}"; row["fd_instab"] = f"{fd:.3e}"
                            msg.append(f"{tag}:{dv:.3e}")
                        write_row(row)
                        log(f"  [{i}-{j}] " + "  ".join(msg) + f"   fd={row['fd_instab']}  ({time.time()-t0:.0f}s)")
                    except Exception as e:
                        traceback.print_exc()
                        write_row(dict(regime=regime, act=act, width=w, seedA=i, seedB=j,
                                       status="error:" + repr(e)[:40].replace(",", ";")))
                del sds
                if DEVICE == "cuda": torch.cuda.empty_cache()

# ------------------------------------------------------------------ REPORT
COLS3 = ["devrel_1e-1","devrel_1e-2","devrel_1e-3"]

def _slope(s, c):
    """tra ve (so mu duong = co theo width, R^2). NaN neu <3 diem."""
    s = s.dropna(subset=[c])
    if len(s) < 3: return float("nan"), float("nan")
    b, a = np.polyfit(np.log(s.width), np.log(s[c]), 1)
    yh = np.polyval([b, a], np.log(s.width))
    ss = ((np.log(s[c]) - np.log(s[c]).mean())**2).sum()
    r2 = 1 - ((np.log(s[c]) - yh)**2).sum()/max(ss, 1e-30)
    return -b, r2

def _pair_drops(d):
    """so mu tinh RIENG cho tung cap -> do tut la hieu ung TRONG-cap, it nhieu."""
    out = []
    d = d.copy(); d["pair"] = d.seedA.astype(str) + "-" + d.seedB.astype(str)
    for (act, pr), s in d.groupby(["act", "pair"]):
        s = s.sort_values("width")
        v = [_slope(s, c)[0] for c in COLS3]
        if any(np.isnan(v)): continue
        out.append(dict(act=act, pair=pr, e1=v[0], e2=v[1], e3=v[2], drop=v[0]-v[2]))
    return out

def report():
    import pandas as pd
    RG = REGIMES[0]
    if not os.path.exists(OUT_CSV): log("khong co du lieu"); return
    d = pd.read_csv(OUT_CSV); d = d[d.status.astype(str) == "ok"].copy()
    for c in COLS3 + ["width", "lam_max"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    if len(d) == 0: log("khong co dong 'ok'"); return
    g = d.groupby(["act", "width"])[COLS3].median().reset_index()
    g = g.merge(d.groupby(["act", "width"]).lam_max.median().rename("Fop").reset_index(),
                on=["act", "width"])
    g.to_csv(CELL_CSV, index=False)

    W = 74
    print("\n" + "=" * W)
    print(f" QUET DAMPING 3 BAC -- CHE DO {RG.upper()} / MLP     ({len(d)} cap, {g.shape[0]} o)")
    ncell_full = len(ACTS)*len(WIDTHS)
    if g.shape[0] < ncell_full:
        print(f" !! MOI CO {g.shape[0]}/{ncell_full} o -- bao cao nay la SO BO, chua ket luan duoc")
    print("=" * W)

    # ---------- (1) bang so mu ----------
    print("\n(1) SO MU dev_rel THEO WIDTH  (duong = co theo width)")
    print(f"{'act':<10}{'lam=1e-1':>16}{'lam=1e-2':>16}{'lam=1e-3':>16}{'spread':>10}")
    print("-" * W)
    spreads = []; signflip = []
    for act, s in g.groupby("act"):
        v = []; r = []
        for c in COLS3:
            b, r2 = _slope(s, c); v.append(b); r.append(r2)
        sp = np.nanmax(v) - np.nanmin(v); spreads.append(sp)
        if not (all(x > 0 for x in v) or all(x < 0 for x in v)): signflip.append(act)
        print(f"{act:<10}" + "".join(f"{v[i]:>9.3f}(R2{r[i]:.2f})" for i in range(3)) + f"{sp:>10.3f}")
    print("-" * W)
    mx = np.nanmax(spreads) if spreads else float("nan")
    print(f"spread LON NHAT trong mot hang: {mx:.3f}")

    # ---------- (2) do tut theo tung cap ----------
    pd_rows = _pair_drops(d)
    print(f"\n(2) DO TUT so mu (lam 1e-1 -> 1e-3), tinh RIENG tung cap")
    if len(pd_rows) < 2:
        print("    khong du cap de thong ke"); dm = ds = float("nan")
    else:
        dr = np.array([r["drop"] for r in pd_rows])
        e1 = np.array([r["e1"] for r in pd_rows]); e3 = np.array([r["e3"] for r in pd_rows])
        dm, ds = float(dr.mean()), float(dr.std(ddof=1))
        print(f"    {RG.upper():<9} drop = {dm:+.3f} +- {ds:.3f}   (n={len(dr)} cap, cung dau: {int((dr>0).sum())}/{len(dr)})")
        print(f"    NTK       drop = {NTK_DROP_MEAN:+.3f} +- {NTK_DROP_SD:.3f}   (n=12 cap, cung dau: 12/12)  <- tham chieu")
        import collections
        sds_in = []
        by = collections.defaultdict(list)
        for r in pd_rows: by[r["act"]].append(r["e2"])
        for a, vv in by.items():
            if len(vv) > 1: sds_in.append(np.std(vv, ddof=1))
        if sds_in:
            print(f"    nhieu giua cac cap trong cung lambda: sd ~ {np.mean(sds_in):.3f}"
                  f"   -> do tut lon gap ~{abs(dm)/max(np.mean(sds_in),1e-9):.0f}x nhieu")

    # ---------- (3) co che ----------
    print("\n(3) CO CHE: gia tri rieng hieu dung mu_eff ma CG 'nhin thay' doc ve phai")
    print("    (giai tu devrel = A/(lam+mu);  in ra mu_eff / lam(1e-2))")
    l1 = g.Fop*LAM_RELS[0]; l3 = g.Fop*LAM_RELS[2]
    rr = g["devrel_1e-3"]/g["devrel_1e-1"]
    g["mu"] = (l1 - rr*l3)/(rr - 1)
    g["mu_rel"] = g["mu"]/(g.Fop*LAM_RELS[1])
    piv = g.pivot_table(index="width", columns="act", values="mu_rel")
    print(piv.round(2).to_string())
    print("    >1 = F chi phoi phep giai   |   <1 = damping chi phoi")
    for act, s in g.groupby("act"):
        s = s.dropna(subset=["mu_rel"]); s = s[s.mu_rel > 0]
        if len(s) >= 3:
            sl = np.polyfit(np.log(s.width), np.log(s.mu_rel), 1)[0]
            print(f"      {act:<10} mu_eff/lam  ~ n^{sl:+.2f}")

    # ---------- (4) so voi NTK truc tiep neu tim thay file ----------
    for nm in NTK_REF_NAMES:
        src = _seek_csv(nm)
        if not src or os.path.abspath(src) == os.path.abspath(OUT_CSV): continue
        try:
            dn = pd.read_csv(src); dn = dn[dn.status.astype(str) == "ok"].copy()
            for c in COLS3 + ["width"]: dn[c] = pd.to_numeric(dn[c], errors="coerce")
            dnr = _pair_drops(dn)
            if dnr:
                a = np.array([r["drop"] for r in dnr])
                print(f"\n    [doc truc tiep {src}] NTK drop = {a.mean():+.3f} +- {a.std(ddof=1):.3f} (n={len(a)})")
            break
        except Exception as e:
            print(f"    (khong doc duoc {src}: {e!r})")

    # ---------- (5) verdict ----------
    print("\n" + "=" * W)
    print(" KET LUAN")
    print("=" * W)
    if signflip:
        print(f" !! DOI DAU SO MU theo damping o: {signflip}")
        print("    Voi cac o nay, dau cua alpha_devrel KHONG xac dinh -> vi tri cua chung")
        print("    tren truc hoanh Figure 1 phu thuoc lambda. PHAI xu ly rieng: hoac bo")
        print("    khoi hoi quy, hoac bao cao ca dai. Day la loi nang nhat co the gap.")
    else:
        print(" [OK] DAU so mu on dinh qua ca 3 bac damping cho moi hang.")
        print("      Claim dinh tinh (devrel co / tang theo width) khong phu thuoc damping.")

    if not np.isnan(dm):
        diff = abs(dm - NTK_DROP_MEAN)
        if RG == "ntk":
            print(f"\n Tai lap ban NTK cu: |{dm:+.3f} - {NTK_DROP_MEAN:+.3f}| = {diff:.3f}")
            print(" (day la che do tham chieu, so sanh nay chi de kiem tra tai lap)")
        else:
            print(f"\n Lech offset so voi NTK: |{dm:+.3f} - {NTK_DROP_MEAN:+.3f}| = {diff:.3f}")
        if diff < 0.10:
            print(" => OFFSET CHUNG. Damping dich MOI o gan nhu cung mot luong.")
            print("    R^2 cua hoi quy alpha_B ~ alpha_devrel BAT BIEN CHINH XAC duoi phep")
            print("    tinh tien chung cua bien hoi quy -> ket qua phu dinh R^2=0.11 an toan.")
            print("    Viet vao phu luc: 'do tut do damping la offset chung giua cac che do")
            print("    (NTK %.3f, %s %.3f), nen khong tao/pha tuong quan giua cac o'."
                  % (NTK_DROP_MEAN, RG.upper(), dm))
        elif diff < 0.25:
            print(" => OFFSET LECH VUA. Phai tinh lai R^2 voi alpha_devrel do o CUNG mot")
            print("    lambda cho moi o (dieu nay von da dung neu moi o deu dung lam=1e-2),")
            print("    va bao cao do nhay nay trong phu luc. Chua sup nhung khong con sach.")
        else:
            print(" => OFFSET KHONG CHUNG. Damping dich cac che do KHAC nhau dang ke.")
            print("    Lap luan 'tinh tien chung -> R^2 bat bien' KHONG dung nua.")
            print("    Phai phat bieu lai ket qua devrel trong Figure 1: bao cao no nhu mot")
            print("    dai phu thuoc lambda, hoac bo devrel khoi hoi quy va giu no thuan tuy")
            print("    la phep do dinh tinh cua Muc 5.1 (dau + don dieu).")
    print("\n Ba ket qua chinh (rho*, R = B/[1/4 Delta^T F Delta], do dai Fisher) KHONG dung")
    print(" G_F^{-1} nen khong bi bat ky ket luan nao o tren dung toi.")
    print("=" * W)

if __name__ == "__main__":
    log(f"=== lambda sweep: {REGIMES[0].upper()} / MLP / {PAIRS} cap / lambda in {LAM_RELS} ===")
    restore_csv()
    self_test()
    run()
    report()
    log(f"XONG (tong {(time.time()-T_START)/60:.1f} phut). Nho Save Version de giu {os.path.basename(OUT_CSV)}.")
