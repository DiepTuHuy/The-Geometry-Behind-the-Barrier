#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 run_lambda_sweep.py  --  CHAY THANG, KHONG CAN CHINH GI
============================================================================
     python run_lambda_sweep.py
============================================================================

MUC DICH (mot cau):
  Chung minh so mu cua do lech trac dia dev_rel theo width KHONG phu thuoc
  hang so damping lambda trong G_F = F + lambda*I.

VI SAO CAN:
  Rayleigh doc Delta co xuong ~1e-8 o width lon, trong khi lambda = 1e-2*||F||.
  Tuc doc huong Delta thi G_F ~= lambda*I. Reviewer se noi: "dev_rel co la vi
  anh do chinh hang so damping cua anh". Quet lambda qua 3 bac; neu SO MU
  khong doi (du magnitude doi) thi ket luan theo width khong phu thuoc damping.
  Day dung la dieu Nhan xet 5.1 dang khang dinh ma chua chung minh.

PHAM VI (da chot, khong can mo rong):
  - Chi che do NTK: claim hinh dang chi duoc phat bieu trong NTK (Muc 5.1).
  - Chi MLP, 2 activation (gelu/tanh = 2 dau mut cua dai alpha_op), 2 cap/o.
    2 bo so mu doc lap la du cho mot ablation phu luc.

DAU RA:
  lam_sweep_mlp.csv        theo cap (resumable, ngat giua chung chay lai duoc)
  + BANG SO MU in ra cuoi log:  3 cot gan nhau  =>  ket luan bat bien damping.

CAN CO:
  ckpt_pmlp_v2/{ntk}_{act}_w{w}_s{s}.pt   va  MNIST tai ./data
  (khong can nhan y: Fisher khong dung nhan)
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

# ======================= TAT CA THAM SO DA CHOT SAN =========================
MODE      = "mlp"
RUN_TAG   = "pmlp_v2"
REGIMES   = ["ntk"]                      # chi NTK
ACTS      = ["gelu","tanh"]              # 2 dau mut cua dai alpha_op (1.03 / 0.86)
WIDTHS    = [64,128,256,512,1024,2048,4096]
NSEEDS    = 5
PAIRS     = 2                            # ablation can do BEN, khong can thong ke
LAM_RELS  = [1e-1, 1e-2, 1e-3]           # lambda = LAM_REL * ||F||_op

TGRID     = 9                            # trung luoi Green cua measure_geo
FISHER_N  = 2048
MICRO     = 64
FD_EPS    = 3e-3
CG_ITERS  = 300
CG_TOL    = 1e-6
POWER_ITERS = 20
RESUME    = True

CKPT_ROOTS = ["/kaggle/input/datasets/ANONYMIZED/DATASET",
              ".", "/kaggle/input", "/content", "/content/drive/MyDrive"]
OUT_DIR   = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUT_CSV   = os.path.join(OUT_DIR, "lam_sweep_mlp.csv")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DIN, K, BASE = 784, 10, 64
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
    sol, resid = cg_solve(m, p, b, x, rhs, lam, micro, x0=x0)
    return _vscale(sol, 0.5), resid, fd

def green_matrix(ts):
    n = len(ts); G = torch.zeros(n, n, dtype=torch.float64)
    for i, t in enumerate(ts):
        for j, s in enumerate(ts):
            G[i, j] = s*(1-t) if s <= t else t*(1-s)
    return G

def dev_rel_at_lambda(ref, pA, pB, b_ref, Xf, delta, lam, ts):
    gammas = []; resids = []; fds = []; x0 = None
    for tt in ts:
        pt = {k: (1-tt)*pA[k] + tt*pB[k] for k in pA}
        g, res, fd = christoffel_dd(ref, pt, b_ref, Xf, delta, lam, MICRO, x0=x0)
        x0 = g
        gammas.append({k: v.detach().cpu() for k, v in g.items()}); resids.append(res); fds.append(fd)
    G = green_matrix(list(ts)); dt = ts[1]-ts[0]; keys = list(delta.keys()); sup = 0.0
    for ti in range(len(ts)):
        xi = None
        for si in range(len(ts)):
            w_ = float(G[ti, si])*dt
            if w_ == 0.0: continue
            xi = {k: (gammas[si][k]*w_ if xi is None else xi[k] + gammas[si][k]*w_) for k in keys}
        sup = max(sup, _vnorm(xi) if xi is not None else 0.0)
    return sup/max(_vnorm(delta), 1e-30), max(resids), max(fds)

# ------------------------------------------------------------------ SELF-TEST
def self_test():
    log("  [self-test] Christoffel vs dense, hang so Green = 1/8 ...")
    old = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    torch.manual_seed(0); m = NetMLP(6, "tanh", "ntk", din=4, k=3).eval(); x = torch.randn(8, 4)
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
    g_vp, _, _ = christoffel_dd(m, p, b, x, delta, lam, micro=8)
    rel = float((flat(g_vp) - g_dense).norm()/max(g_dense.norm(), 1e-12))
    ts = np.linspace(0, 1, 21); G = green_matrix(list(ts)); dt = ts[1]-ts[0]
    xi = (G @ torch.full((len(ts),), 0.7, dtype=torch.float64))*dt
    ge = float((xi - torch.tensor([0.7*t*(1-t)/2 for t in ts])).abs().max())
    torch.set_default_dtype(old)
    log(f"  [self-test] Gamma rel={rel:.2e}   Green err={ge:.2e}")
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
    return idx

def find_ckpt(regime, act, w, s):
    return _build_index().get(f"{regime}_{act}_w{w}_s{s}.pt")

def load_sd(path):
    try: d = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: d = torch.load(path, map_location="cpu")
    return d["sd"]

# ------------------------------------------------------------------ CSV
COLS = ["regime","act","width","seedA","seedB","dnorm","lam_max",
        "devrel_1e-1","devrel_1e-2","devrel_1e-3","cg_resid","fd_instab","status"]

def write_row(row):
    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a") as f:
        if new: f.write(",".join(COLS) + "\n")
        f.write(",".join(str(row.get(c, "")) for c in COLS) + "\n"); f.flush()

def done(regime, act, w, i, j):
    if not os.path.exists(OUT_CSV): return False
    key = f"{regime},{act},{w},{i},{j},"
    with open(OUT_CSV) as f:
        for ln in f:
            if ln.startswith(key) and ln.strip().endswith("ok"): return True
    return False

# ------------------------------------------------------------------ MAIN
def run():
    X = load_X(); Xf = X[:FISHER_N].to(DEVICE)
    ts = np.linspace(0, 1, TGRID)
    log(f"DEVICE={DEVICE} | NTK / MLP | {len(ACTS)} act x {len(WIDTHS)} width x {PAIRS} cap x {len(LAM_RELS)} lambda")
    log(f"-> {OUT_CSV}")
    for regime in REGIMES:
        for act in ACTS:
            for w in WIDTHS:
                sds = []
                for s in range(NSEEDS):
                    cp = find_ckpt(regime, act, w, s)
                    if cp is not None: sds.append(load_sd(cp))
                if len(sds) < 2:
                    log(f"  [{regime}/{act}/w{w}] <2 ckpt -> bo"); continue
                log(f"=== {regime}/{act}/w{w} ===")
                ref = build_net(w, act, regime).to(DEVICE).eval()
                p_ref, b_ref = _pb(ref)
                ag, gs = perm_spec(build_net(w, act, regime))
                topar = lambda sd: {k: sd[k].to(DEVICE) for k in p_ref.keys()}
                for (i, j) in list(itertools.combinations(range(len(sds)), 2))[:PAIRS]:
                    if RESUME and done(regime, act, w, i, j): continue
                    try:
                        perms = weight_matching(ag, gs, sds[i], sds[j], iters=8, seed=i*13 + j)
                        pA = topar(sds[i]); pB = topar(apply_perm(sds[j], ag, perms))
                        delta = {k: pB[k] - pA[k] for k in pA}
                        lmax = lam_max(ref, pA, b_ref, Xf, MICRO, POWER_ITERS)
                        row = dict(regime=regime, act=act, width=w, seedA=i, seedB=j,
                                   dnorm=f"{_vnorm(delta):.6e}", lam_max=f"{lmax:.6e}", status="ok")
                        msg = []
                        for lr in LAM_RELS:
                            dv, cg, fd = dev_rel_at_lambda(ref, pA, pB, b_ref, Xf, delta,
                                                           max(lr*lmax, 1e-12), ts)
                            tag = {1e-1:"1e-1", 1e-2:"1e-2", 1e-3:"1e-3"}[lr]
                            row[f"devrel_{tag}"] = f"{dv:.6e}"
                            row["cg_resid"] = f"{cg:.2e}"; row["fd_instab"] = f"{fd:.3e}"
                            msg.append(f"{tag}:{dv:.3e}")
                        write_row(row)
                        log(f"  [{i}-{j}] " + "  ".join(msg))
                    except Exception as e:
                        traceback.print_exc()
                        write_row(dict(regime=regime, act=act, width=w, seedA=i, seedB=j,
                                       status="error:" + repr(e)[:40]))
                del sds
                if DEVICE == "cuda": torch.cuda.empty_cache()

def report():
    import pandas as pd
    if not os.path.exists(OUT_CSV): log("khong co du lieu"); return
    d = pd.read_csv(OUT_CSV); d = d[d.status.astype(str) == "ok"].copy()
    cols = ["devrel_1e-1","devrel_1e-2","devrel_1e-3"]
    for c in cols + ["width"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    g = d.groupby(["act","width"])[cols].median().reset_index()
    g.to_csv(os.path.join(OUT_DIR, "lam_sweep_mlp_cell.csv"), index=False)
    def slope(s, c):
        s = s.dropna(subset=[c])
        if len(s) < 3: return float("nan")
        b, _ = np.polyfit(np.log(s.width), np.log(s[c]), 1)
        yh = np.polyval([b, _], np.log(s.width))
        r2 = 1 - ((np.log(s[c]) - yh)**2).sum()/max(((np.log(s[c]) - np.log(s[c]).mean())**2).sum(), 1e-30)
        return -b, r2
    print("\n" + "="*66)
    print(" SO MU dev_rel THEO WIDTH, QUET 3 BAC DAMPING  (NTK / MLP)")
    print("="*66)
    print(f"{'act':<10}{'lam=1e-1':>13}{'lam=1e-2':>13}{'lam=1e-3':>13}{'do lech':>12}")
    print("-"*66)
    spread = []
    for act, s in g.groupby("act"):
        v = []
        for c in cols:
            r = slope(s, c); v.append(r[0] if isinstance(r, tuple) else float("nan"))
        sp = np.nanmax(v) - np.nanmin(v); spread.append(sp)
        print(f"{act:<10}{v[0]:>13.3f}{v[1]:>13.3f}{v[2]:>13.3f}{sp:>12.3f}")
    print("-"*66)
    mx = np.nanmax(spread) if spread else float("nan")
    print(f"do lech LON NHAT giua cac lambda: {mx:.3f}")
    if mx < 0.10:
        print("=> SO MU BAT BIEN theo damping. Ket luan hinh dang (Muc 5.1) vung.")
        print("   Viet vao phu luc: 'so mu dev_rel bat bien qua 3 bac damping'.")
    elif mx < 0.25:
        print("=> Gan bat bien. Bao cao ca 3 cot, neu ro bien do trong phu luc.")
    else:
        print("=> !! SO MU PHU THUOC DAMPING. Phai noi yeu claim hinh dang lai;")
        print("   ba ket qua chinh (rho*, R, flen) KHONG bi anh huong vi khong dung G_F^-1.")
    print("="*66)

if __name__ == "__main__":
    log("=== lambda sweep: NTK / MLP / 3 cap / lambda in {1e-1, 1e-2, 1e-3} ===")
    self_test()
    run()
    report()
