#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 measure_final_{MODE}.py -- LAN DO CUOI tu checkpoint da co (khong train lai)
============================================================================
Gom TAT CA phep do con thieu vao mot lan chay. Tai su dung nguyen primitives
cua measure_geo.py (build_net / perm_spec / weight_matching / apply_perm /
fisher_vp / cg_solve / christoffel_dd / green) nen so lieu so sanh truc tiep
duoc voi param_geo_{MODE}.csv da co.

DO 4 NHOM:

 (1) L(t) tren luoi min      -> t* = argmax,  B = max_t L - 1/2(L_A+L_B)
 (2) rho*(t) = E||p_w(x)-e_y||_2   (Dinh nghia 2.3, thang [0, sqrt(2)])
     -> CAU HOI GATING: rho*(mid) nho => F ~= Hess o trung diem;
                        rho*(mid) lon => F != Hess ma du bao van chay.
 (3) flen(t) = 1/2 Delta^T F(gamma(t)) Delta  tren luoi tho + tai t*
     -> R(t) := B / (flen(t)/4)  = ti so voi du bao bac hai
        (da biet: R_end ~ 0.81 o NTK, R_mid ~ 1.0 o feature-learning)
 (4) SWEEP lambda cho do lech trac dia: dev_rel o lam_rel in {1e-1,1e-2,1e-3}
     -> kiem so mu bat bien theo damping (Nhan xet 5.1).
        lam_rel=1e-2 trung voi lan do cu => dong thoi la cross-check pipeline.

PHASE (bien moi truong):
    PHASE=rho    chi (1)(2)          -- chi forward pass, RE, chay truoc
    PHASE=full   (1)(2)(3)           -- them fisher_vp
    PHASE=all    (1)(2)(3)(4)        -- them CG + Green cho sweep lambda   [mac dinh]

CAN CO SAN:
    ckpt_{RUN_TAG}/{regime}_{act}_w{w}_s{s}.pt      (bat buoc)
    combined_p{MODE}*.csv                            (tuy chon, de selfcheck B)
    du lieu MNIST/FashionMNIST tai ./data            (mlp/cnn; ts la synthetic)

OUTPUT:
    param_final_{MODE}.csv        theo cap, resumable
    final_{MODE}_cell.csv         gop theo o
============================================================================
"""
import os, sys, time, math, glob, itertools, traceback
try:
    sys.stdout.reconfigure(line_buffering=True); sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.func import functional_call, jvp as _fjvp, vjp as _fvjp, jacrev as _jacrev, grad as _grad

def _tv():
    import torchvision; return torchvision

# ==================================================================== CONFIG
MODE       = "cnn"                       # <<< dat boi ban phat hanh
RESUME     = True
SELFCHECK  = True
RUN_TAG    = {"mlp":"pmlp_v2","cnn":"pcnn_v2","ts":"pts_v2"}[MODE]
PHASE      = os.environ.get("PHASE", "all")   # rho | full | all

WIDTHS   = {"mlp":[64,128,256,512,1024,2048,4096],
            "cnn":[1,2,4,8],
            "ts" :[64,128,256,512,1024,2048,4096]}[MODE]
ACTS     = ["gelu","tanh","swish","softplus"]      # C^3, dong nhat geo script
REGIMES  = ["ntk","sp","mup"]
NSEEDS   = 5
PAIRS    = int(os.environ.get("PAIRS", "10"))     # PAIRS=3 de quet nhanh truoc

TGRID_FINE   = 41       # L(t), rho*(t)  -- forward pass
TGRID_COARSE = 9        # flen(t), Gamma(t) -- trung luoi Green cua geo script
EVAL_N       = 10000    # so mau cho L va rho*
FISHER_N     = 2048     # = GEO_BATCH cu, giu nguyen de so sanh duoc
MICRO        = 64

# --- sweep lambda (chi chay tren LAM_PAIRS cap dau moi o: du de kiem so mu) ---
LAM_RELS   = [1e-1, 1e-2, 1e-3]
LAM_PAIRS  = int(os.environ.get("LAM_PAIRS", "3"))
FD_EPS     = 3e-3
CG_ITERS   = 300
CG_TOL     = 1e-6
POWER_ITERS= 20

CKPT_ROOTS = [".","/kaggle/input","/content","/content/drive/MyDrive"]
OUT_DIR    = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DIN,K      = {"mlp":(784,10),"cnn":(None,10),"ts":(64,10)}[MODE]
BASE       = 64

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def set_seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def _first(pats):
    for p in pats:
        if not p: continue
        h = sorted(glob.glob(p, recursive=True))
        if h: return h[0]
    return None

# ==================================================================== MODEL (sao y geo script)
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

class ScaledConv(nn.Module):
    def __init__(self, cin, cout, k, st, pad, regime, kind):
        super().__init__()
        fin = cin*k*k
        istd, self.fmul, self.lr_scale = param_cfg(regime, fin, cout, kind)
        self.weight = nn.Parameter(torch.randn(cout, cin, k, k)*istd); self.st = st; self.pad = pad
    def forward(self, x): return self.fmul*F.conv2d(x, self.weight, None, self.st, self.pad)

def _gn(c): return nn.GroupNorm(1, c)

class NetMLP(nn.Module):
    def __init__(self, width, act, regime="ntk", din=DIN, k=K):
        super().__init__()
        self.fc1 = ScaledLinear(din, width, regime, "input")
        self.fc2 = ScaledLinear(width, width, regime, "hidden")
        self.fc3 = ScaledLinear(width, k, regime, "output")
        self.a1 = make_act(act); self.a2 = make_act(act); self.width = width; self.regime = regime
    def forward(self, x): return self.fc3(self.a2(self.fc2(self.a1(self.fc1(x)))))

class NetCNN(nn.Module):
    def __init__(self, wm, act, regime="ntk", in_ch=1, k=K):
        super().__init__(); c = [16*wm, 32*wm, 64*wm]
        self.c1 = ScaledConv(in_ch, c[0], 3,1,1, regime, "input");  self.n1 = _gn(c[0]); self.a1 = make_act(act)
        self.c2 = ScaledConv(c[0], c[1], 3,1,1, regime, "hidden");  self.n2 = _gn(c[1]); self.a2 = make_act(act)
        self.c3 = ScaledConv(c[1], c[2], 3,1,1, regime, "hidden");  self.n3 = _gn(c[2]); self.a3 = make_act(act)
        self.pool = nn.MaxPool2d(2); self.fc = ScaledLinear(c[2], k, regime, "output")
        self.width = wm; self.regime = regime
    def forward(self, x):
        h1 = self.pool(self.a1(self.n1(self.c1(x))))
        h2 = self.pool(self.a2(self.n2(self.c2(h1))))
        h3 = self.a3(self.n3(self.c3(h2)))
        return self.fc(F.adaptive_avg_pool2d(h3,1).flatten(1))

def build_net(width, act, regime):
    return NetCNN(width, act, regime) if MODE == "cnn" else NetMLP(width, act, regime)

# ==================================================================== PERM (sao y)
def perm_spec(model):
    if MODE == "cnn":
        ag = {"c1.weight":["g1",None,None,None],"n1.weight":["g1"],"n1.bias":["g1"],
              "c2.weight":["g2","g1",None,None],"n2.weight":["g2"],"n2.bias":["g2"],
              "c3.weight":["g3","g2",None,None],"n3.weight":["g3"],"n3.bias":["g3"],
              "fc.weight":[None,"g3"],"fc.bias":[None]}
    else:
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
            ci = linear_sum_assignment(-S.numpy())[1]
            new = torch.as_tensor(ci, dtype=torch.long)
            if not torch.equal(new, perms[g]): moved += 1
            perms[g] = new
        if moved == 0: break
    return perms

# ==================================================================== DATA (CO NHAN)
_CACHE = {}
def load_data_xy():
    """Khac geo script: geo chi can X (Fisher khong can y); o day can y cho L va rho*."""
    if "xy" in _CACHE: return _CACHE["xy"]
    if MODE == "mlp":
        ds = _tv().datasets.MNIST("./data", train=True, download=True)
        X = ((ds.data.float()/255.0) - 0.1307)/0.3081; X = X.reshape(-1, 784); Y = ds.targets.clone()
    elif MODE == "cnn":
        ds = _tv().datasets.FashionMNIST("./data", train=True, download=True)
        X = ((ds.data.float()/255.0) - 0.2860)/0.3530; X = X.unsqueeze(1); Y = ds.targets.clone()
    else:
        g = torch.Generator().manual_seed(1); X = torch.randn(20000, DIN, generator=g)
        set_seed(1234); teach = NetMLP(32, "relu", "sp").eval()     # teacher co dinh, sao y geo script
        with torch.no_grad(): Y = teach(X).argmax(1)
    _CACHE["xy"] = (X, Y); return X, Y

# ==================================================================== PRIMITIVES
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

@torch.no_grad()
def loss_acc_rho(m, p, b, x, y, micro=512):
    """L = cross-entropy;  rho* = E||p_w(x)-e_y||_2 (Dinh nghia 2.3);  acc."""
    n = x.shape[0]; sL = 0.0; sR = 0.0; sC = 0
    for i in range(0, n, micro):
        xb = x[i:i+micro]; yb = y[i:i+micro]
        lg = _call(m, p, b, xb)
        sL += float(F.cross_entropy(lg, yb, reduction="sum"))
        pr = torch.softmax(lg, 1)
        e  = F.one_hot(yb, num_classes=pr.shape[1]).to(pr.dtype)
        sR += float((pr - e).norm(dim=1).sum())
        sC += int((lg.argmax(1) == yb).sum())
    return sL/n, sR/n, sC/n

def flen_at(m, pt, b, xf, delta, micro):
    q  = _vdot(delta, fisher_vp(m, pt, b, xf, delta, micro))
    d2 = max(_vdot(delta, delta), 1e-30)
    return 0.5*q, q/d2

# ---------- may do do lech trac dia (sao y geo script) ----------
def dFz(m, p, b, x, z, v, eps, micro, rich):
    def cd(e):
        Fp = fisher_vp(m, _vaxpy(p,  e, z), b, x, v, micro)
        Fm = fisher_vp(m, _vaxpy(p, -e, z), b, x, v, micro)
        return {k: (Fp[k]-Fm[k])/(2*e) for k in p}
    if rich:
        d1, d2 = cd(eps), cd(eps/2)
        return {k: (4*d2[k]-d1[k])/3 for k in p}
    return cd(eps)

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

def lam_max(m, p, b, x, micro, iters, seed=0):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    u = {k: torch.randn(v.shape, generator=gen, device=v.device, dtype=v.dtype) for k, v in p.items()}
    u = _vscale(u, 1.0/max(_vnorm(u), 1e-30)); lam = 0.0
    for _ in range(iters):
        Au = fisher_vp(m, p, b, x, u, micro); lam = _vnorm(Au)
        if lam < 1e-30: break
        u = _vscale(Au, 1.0/lam)
    return lam

def cg_solve(m, p, b, x, rhs, lam, micro, x0=None, iters=80, tol=1e-6):
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
    d1 = dFz(m, p, b, x, dhat, dhat, FD_EPS,   micro, False)
    d2 = dFz(m, p, b, x, dhat, dhat, FD_EPS/2, micro, False)
    t1r = {k: (4*d2[k]-d1[k])/3 for k in d1}
    fd_instab = _vnorm({k: d1[k]-d2[k] for k in d1})/max(_vnorm(t1r), 1e-30)
    t1 = {k: t1r[k]*dn*dn for k in t1r}
    mvec = grad_quad(m, p, b, x, delta, micro)
    rhs = {k: 2*t1[k] - mvec[k] for k in t1}
    sol, resid = cg_solve(m, p, b, x, rhs, lam, micro, x0=x0, iters=CG_ITERS, tol=CG_TOL)
    return _vscale(sol, 0.5), resid, fd_instab

def green_matrix(ts):
    n = len(ts); G = torch.zeros(n, n, dtype=torch.float64)
    for i, t in enumerate(ts):
        for j, s in enumerate(ts):
            G[i, j] = s*(1-t) if s <= t else t*(1-s)
    return G

# ==================================================================== SELF-TEST
def self_test():
    log("  [self-test] Gamma(dd) vp == dense, va hang so Green = 1/8 ...")
    old = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    torch.manual_seed(0); m = NetMLP(6, "tanh", "ntk", din=4, k=3).eval(); x = torch.randn(8, 4)
    p, b = _pb(m); keys = list(p.keys())
    flat = lambda d: torch.cat([d[k].reshape(-1) for k in keys])
    def unflat(v):
        o = {}; i = 0
        for k in keys:
            n = p[k].numel(); o[k] = v[i:i+n].reshape(p[k].shape); i += n
        return o
    torch.manual_seed(3); dflat = torch.randn(flat(p).numel()); dflat = dflat/dflat.norm()*3.0
    delta = unflat(dflat); lam = 1e-2
    def Fmat(wv):
        pp = unflat(wv); J = _jacrev(lambda q: _call(m, q, b, x))(pp); B = x.shape[0]
        Jf = torch.cat([J[k].reshape(B, 3, -1) for k in keys], 2)
        pr = torch.softmax(_call(m, pp, b, x), 1)
        S = torch.diag_embed(pr) - pr.unsqueeze(2)*pr.unsqueeze(1)
        return torch.einsum('bki,bkl,blj->ij', Jf, S, Jf)/B
    w0 = flat(p); dF = _jacrev(Fmat)(w0); Fd = Fmat(w0); P = w0.numel()
    t1 = torch.einsum('ijl,l,j->i', dF, dflat, dflat)
    mv = torch.einsum('i,ijl,j->l', dflat, dF, dflat)
    g_dense = 0.5*torch.linalg.solve(Fd + lam*torch.eye(P), 2*t1 - mv)
    global CG_ITERS, CG_TOL
    oi, ot = CG_ITERS, CG_TOL; CG_ITERS, CG_TOL = 300, 1e-12
    g_vp, _, _ = christoffel_dd(m, p, b, x, delta, lam, micro=8)
    CG_ITERS, CG_TOL = oi, ot
    rel = float((flat(g_vp) - g_dense).norm()/max(g_dense.norm(), 1e-12))
    assert rel < 1e-6, f"Gamma sai {rel:.2e}"
    ts = np.linspace(0, 1, 21); G = green_matrix(list(ts)); dt = ts[1]-ts[0]
    xi = (G @ torch.full((len(ts),), 0.7, dtype=torch.float64))*dt
    ge = float((xi - torch.tensor([0.7*t*(1-t)/2 for t in ts])).abs().max())
    assert ge < 1e-10, f"Green sai {ge:.2e}"
    log(f"  [self-test] OK  Gamma rel={rel:.2e}  Green err={ge:.2e}")
    torch.set_default_dtype(old)

# ==================================================================== CKPT IO
def ckpt_dir():
    d = os.path.join(OUT_DIR, f"ckpt_{RUN_TAG}"); os.makedirs(d, exist_ok=True); return d

def find_ckpt(regime, act, w, s):
    name = f"{regime}_{act}_w{w}_s{s}.pt"
    local = os.path.join(ckpt_dir(), name)
    if os.path.exists(local): return local
    for root in CKPT_ROOTS:
        hits = glob.glob(os.path.join(root, "**", f"ckpt_{RUN_TAG}", name), recursive=True)
        if hits: return hits[0]
    return None

def load_sd(path):
    try: d = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: d = torch.load(path, map_location="cpu")
    return d["sd"], d.get("acc"), d.get("wmove")

# ==================================================================== CSV
CSV = ["mode","regime","act","width","seedA","seedB","dnorm",
       "L_A","L_B","B","t_star","L_max",
       "rho_A","rho_mid","rho_max","rho_at_tstar","acc_mid",
       "flen_A","flen_mid","flen_B","flen_at_tstar","rq_mid",
       "R_end","R_mid","R_tstar",
       "devrel_lam1e-1","devrel_lam1e-2","devrel_lam1e-3",
       "cg_resid","fd_instab","wmoveA","status"]

def write_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new: f.write(",".join(CSV) + "\n")
        f.write(",".join(str(row.get(c, "")) for c in CSV) + "\n"); f.flush()

def already_done(path, regime, act, w, i, j):
    if not os.path.exists(path): return False
    key = f",{regime},{act},{w},{i},{j},"
    with open(path) as f:
        for ln in f:
            if key in ln and ln.strip().endswith("ok"): return True
    return False

# ==================================================================== DEV_REL
def dev_rel_at_lambda(ref, pA, pB, b_ref, Xf, delta, lam_rel, lmax, ts_c):
    """Dung Gamma(D,D) doc luoi t, tich phan Green -> dev_rel = sup_t||xi||/||D||."""
    lam = max(lam_rel*lmax, 1e-12); dn = _vnorm(delta)
    gammas = []; resids = []; fdis = []; x0 = None
    for tt in ts_c:
        pt = {k: (1-tt)*pA[k] + tt*pB[k] for k in pA}
        g, res, fdi = christoffel_dd(ref, pt, b_ref, Xf, delta, lam, MICRO, x0=x0)
        x0 = g
        gammas.append({k: v.detach().cpu() for k, v in g.items()})
        resids.append(res); fdis.append(fdi)
    G = green_matrix(list(ts_c)); dt = ts_c[1]-ts_c[0]; keys = list(delta.keys())
    sup = 0.0
    for ti in range(len(ts_c)):
        xi = None
        for si in range(len(ts_c)):
            w_ = float(G[ti, si])*dt
            if w_ == 0.0: continue
            xi = {k: (gammas[si][k]*w_ if xi is None else xi[k] + gammas[si][k]*w_) for k in keys}
        sup = max(sup, _vnorm(xi) if xi is not None else 0.0)
    return sup/max(dn, 1e-30), max(resids), max(fdis)

# ==================================================================== MAIN
def run():
    out = os.path.join(OUT_DIR, f"param_final_{MODE}.csv")
    X, Y = load_data_xy()
    Xe = X[:EVAL_N].to(DEVICE); Ye = Y[:EVAL_N].to(DEVICE)
    Xf = X[:FISHER_N].to(DEVICE)
    ts_f = np.linspace(0, 1, TGRID_FINE)
    ts_c = np.linspace(0, 1, TGRID_COARSE)
    i_mid_c = int(np.argmin(np.abs(ts_c - 0.5)))
    k_mid_f = int(np.argmin(np.abs(ts_f - 0.5)))
    log(f"MODE={MODE} PHASE={PHASE} DEVICE={DEVICE} pairs={PAIRS} lam_pairs={LAM_PAIRS}")
    log(f"widths={WIDTHS} eval_n={EVAL_N} fisher_n={FISHER_N} -> {out}")

    for regime in REGIMES:
        for act in ACTS:
            for w in WIDTHS:
                try:
                    sds = []; wmv = []
                    for s in range(NSEEDS):
                        cp = find_ckpt(regime, act, w, s)
                        if cp is None: continue
                        sd, _, wm = load_sd(cp); sds.append(sd); wmv.append(wm)
                    if len(sds) < 2:
                        log(f"  [{regime}/{act}/w{w}] <2 ckpt -> bo"); continue
                    log(f"=== {regime}/{act}/w{w}  ({len(sds)} nets) ===")
                    ref = build_net(w, act, regime).to(DEVICE).eval()
                    p_ref, b_ref = _pb(ref)
                    ag, gs = perm_spec(build_net(w, act, regime))
                    to_params = lambda sd: {k: sd[k].to(DEVICE) for k in p_ref.keys()}

                    for np_, (i, j) in enumerate(list(itertools.combinations(range(len(sds)), 2))[:PAIRS]):
                        if RESUME and already_done(out, regime, act, w, i, j):
                            continue
                        try:
                            perms = weight_matching(ag, gs, sds[i], sds[j], iters=8, seed=i*13 + j)
                            pA = to_params(sds[i]); pB = to_params(apply_perm(sds[j], ag, perms))
                            delta = {k: pB[k] - pA[k] for k in pA}; dn = _vnorm(delta)

                            # ---- (1)(2) L(t), rho*(t) ----
                            Ls, Rs = [], []
                            for tt in ts_f:
                                pt = {k: (1-tt)*pA[k] + tt*pB[k] for k in pA}
                                L_, r_, _ = loss_acc_rho(ref, pt, b_ref, Xe, Ye)
                                Ls.append(L_); Rs.append(r_)
                            Ls = np.array(Ls); Rs = np.array(Rs)
                            k_star = int(np.argmax(Ls)); t_star = float(ts_f[k_star])
                            B = float(Ls[k_star] - 0.5*(Ls[0] + Ls[-1]))
                            p_mid = {k: 0.5*(pA[k] + pB[k]) for k in pA}
                            _, _, acc_mid = loss_acc_rho(ref, p_mid, b_ref, Xe, Ye)

                            row = dict(mode=MODE, regime=regime, act=act, width=w, seedA=i, seedB=j,
                                       dnorm=f"{dn:.6e}", L_A=f"{Ls[0]:.6e}", L_B=f"{Ls[-1]:.6e}",
                                       B=f"{B:.6e}", t_star=f"{t_star:.4f}", L_max=f"{Ls[k_star]:.6e}",
                                       rho_A=f"{Rs[0]:.6e}", rho_mid=f"{Rs[k_mid_f]:.6e}",
                                       rho_max=f"{Rs.max():.6e}", rho_at_tstar=f"{Rs[k_star]:.6e}",
                                       acc_mid=f"{acc_mid:.4f}", wmoveA=wmv[i], status="ok")

                            # ---- (3) flen(t) ----
                            if PHASE in ("full", "all"):
                                fl = []
                                for tt in ts_c:
                                    pt = {k: (1-tt)*pA[k] + tt*pB[k] for k in pA}
                                    f_, _ = flen_at(ref, pt, b_ref, Xf, delta, MICRO); fl.append(f_)
                                fl_star, rq_m = flen_at(ref,
                                    {k: (1-t_star)*pA[k] + t_star*pB[k] for k in pA}, b_ref, Xf, delta, MICRO)
                                _, rq_mid = flen_at(ref, p_mid, b_ref, Xf, delta, MICRO)
                                fl_A, fl_mid, fl_B = fl[0], fl[i_mid_c], fl[-1]
                                fl_end = 0.5*(fl_A + fl_B)
                                Rf = lambda f_: (B/(f_/4) if f_ > 0 else float("nan"))
                                row.update(flen_A=f"{fl_A:.6e}", flen_mid=f"{fl_mid:.6e}",
                                           flen_B=f"{fl_B:.6e}", flen_at_tstar=f"{fl_star:.6e}",
                                           rq_mid=f"{rq_mid:.6e}",
                                           R_end=f"{Rf(fl_end):.4f}", R_mid=f"{Rf(fl_mid):.4f}",
                                           R_tstar=f"{Rf(fl_star):.4f}")

                            # ---- (4) sweep lambda cho dev_rel (chi LAM_PAIRS cap dau) ----
                            if PHASE == "all" and np_ < LAM_PAIRS:
                                lmax = lam_max(ref, pA, b_ref, Xf, MICRO, POWER_ITERS, seed=17)
                                for lr_ in LAM_RELS:
                                    dv, cgr, fdi = dev_rel_at_lambda(ref, pA, pB, b_ref, Xf, delta,
                                                                     lr_, lmax, ts_c)
                                    row[f"devrel_lam{lr_:g}"] = f"{dv:.6e}"
                                    row["cg_resid"] = f"{cgr:.2e}"; row["fd_instab"] = f"{fdi:.3e}"

                            write_row(out, row)
                            log(f"  [{i}-{j}] B={B:.4f} t*={t_star:.3f} rho*(mid)={Rs[k_mid_f]:.4f} "
                                + (f"R_end={row.get('R_end','-')} R_mid={row.get('R_mid','-')} "
                                   f"R_t*={row.get('R_tstar','-')}" if PHASE != "rho" else ""))
                        except Exception as e:
                            traceback.print_exc()
                            write_row(out, dict(mode=MODE, regime=regime, act=act, width=w,
                                                seedA=i, seedB=j, status="error:" + repr(e)[:40]))
                    del sds
                    if DEVICE == "cuda": torch.cuda.empty_cache()
                except Exception as e:
                    traceback.print_exc()
                    write_row(out, dict(mode=MODE, regime=regime, act=act, width=w,
                                        status="cell-error:" + repr(e)[:40]))
    log("DONE", MODE)
    return out

def aggregate(path):
    import pandas as pd
    d = pd.read_csv(path); d = d[d.status.astype(str) == "ok"].copy()
    num = ["B","t_star","rho_A","rho_mid","rho_max","rho_at_tstar","acc_mid","width",
           "flen_A","flen_mid","flen_B","flen_at_tstar","rq_mid","R_end","R_mid","R_tstar",
           "devrel_lam0.1","devrel_lam0.01","devrel_lam0.001","fd_instab"]
    for c in num:
        if c in d: d[c] = pd.to_numeric(d[c], errors="coerce")
    agg = {c: (c, "median") for c in num if c in d and c != "width"}
    g = d.groupby(["mode","regime","act","width"]).agg(n_pairs=("B","size"), **agg).reset_index()
    fin = os.path.join(OUT_DIR, f"final_{MODE}_cell.csv"); g.to_csv(fin, index=False)
    log(f"-> {fin}  ({len(g)} o)")
    cols = [c for c in ["t_star","rho_mid","R_end","R_mid","R_tstar"] if c in g]
    log("\n  tom tat theo che do (trung vi):")
    log(g.groupby("regime")[cols].median().round(3).to_string())
    if "devrel_lam0.1" in g and g["devrel_lam0.1"].notna().any():
        def _slope(gr, c):
            s = gr.dropna(subset=[c])
            if len(s) < 3: return np.nan
            b, _ = np.polyfit(np.log(s.width), np.log(s[c]), 1); return -b
        log("\n  so mu dev_rel theo lambda (phai gan nhau => ket luan bat bien damping):")
        for (r, a), gr in g.groupby(["regime","act"]):
            v = [f"{_slope(gr, f'devrel_lam{l:g}'):.2f}" for l in [0.1, 0.01, 0.001]]
            log(f"    {r:4s}/{a:9s}  1e-1={v[0]}  1e-2={v[1]}  1e-3={v[2]}")
    return fin

def selfcheck(path):
    import pandas as pd
    cb = _first([f"combined_p{MODE}*dFfixed*.csv", f"combined_p{MODE}*.csv",
                 f"/kaggle/input/**/combined_p{MODE}*.csv"])
    if not cb:
        log("[selfcheck] khong tim thay combined -> bo qua"); return
    a = pd.read_csv(path); a = a[a.status.astype(str) == "ok"]
    a["B"] = pd.to_numeric(a["B"], errors="coerce")
    x = a.groupby(["regime","act","width"])["B"].median().rename("B_new").reset_index()
    c = pd.read_csv(cb); c = c[c["kind"] == "pair"]
    y = c.groupby(["regime","act","width"])["barrier"].median().rename("B_old").reset_index()
    m = x.merge(y, on=["regime","act","width"], how="inner")
    if len(m) == 0:
        log("[selfcheck] khong khop o nao"); return
    rel = ((m.B_new - m.B_old).abs()/m.B_old.abs().clip(lower=1e-12))
    log(f"[selfcheck] {len(m)} o | sai so tuong doi trung vi={rel.median():.3%} max={rel.max():.3%}")
    if rel.median() > 0.05:
        log("[selfcheck] !! lech >5%: EVAL_N khac tap danh gia lan do barrier goc -> chinh lai truoc khi tin so")

if __name__ == "__main__":
    log(f"=== measure_final[{MODE}] PHASE={PHASE} ===")
    if PHASE == "all": self_test()
    p = run()
    aggregate(p)
    if SELFCHECK: selfcheck(p)
