#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shape: the geodesic deviation xi(t) along the whole path, all regimes.

     python measure_profile_shape.py               # MODE=mlp (default)
     MODE=cnn python measure_profile_shape.py
     MODE=all python measure_profile_shape.py

What was missing
----------------
param_geo_{mode}.csv already carries dev_rel in all three regimes, but dev_rel
is a single number: sup_t ||xi(t)|| / ||Delta||. It says how high the bump is,
not where it sits or whether it is symmetric. This file records the whole curve
xi(t) and extracts three shape quantities:

    xirel(t) = ||xi(t)|| / ||Delta||        amplitude, comparable across cells
    shape(t) = ||xi(t)|| / sup_s ||xi(s)||  pure shape, peak = 1
    t_peak   = argmax_t ||xi(t)||           which way the bump leans
    skew     = (t_peak - 1/2) * 2           in [-1,1], 0 is symmetric

The toy model behind the Poincare-Sobolev lemma gives a symmetric bump peaking
exactly at t=1/2 when Gamma is constant along the path. A measured peak away
from 1/2 therefore means the Christoffel field is not uniform -- which is new
information, not readable from dev_rel.

Formula (identical to the geodesic script)
------------------------------------------
    xi(t) = int_0^1 G(t,s) Gamma(D,D)(gamma_lin(s)) ds
    G(t,s) = s(1-t) for s <= t,  t(1-s) for s >= t
    dev_rel = sup_t ||xi(t)|| / ||D||

Reusing the Christoffel file saves the GPU work
-----------------------------------------------
xi(t) is built from Gamma(t), and measure_profile_christoffel.py already writes
xinorm_t* into profile_christoffel_{mode}.csv. When that file is present this
script reads it directly and uses no GPU; only without it does it compute Gamma
itself.

  => cheapest order to run:
       1) measure_profile_christoffel.py   (heavy: computes Gamma, SHARD=0..5)
       2) measure_profile_shape.py         (near-instant: reads it back)
       3) measure_profile_length.py        (light, independent)
     The reverse order is still correct, it just pays for Gamma once more.

Without the Christoffel file this script still runs and computes Gamma itself,
but is then as heavy as that file, so use the same SHARD=0..5 split. Run this
way both files compute Gamma, which is why running Christoffel first is cheapest.

PAIRS defaults to 3 to match the Christoffel file. Setting PAIRS=10 here while
that file ran 3 pairs means the remaining 7 must compute Gamma themselves, which
is expensive.

REUSE=0 forces recomputation, for an independent check.

Cross-check: the dev_rel computed here is compared against the dev_rel column of
param_geo_{mode}.csv at lam_rel=1e-2. They must agree.

Output
------
  profile_shape_{mode}.csv        one row per pair, resumable
  profile_shape_{mode}_cell.csv   per-cell medians
"""
import os, sys, time, math, glob, shutil, itertools, traceback, zipfile, io
try:
    sys.stdout.reconfigure(line_buffering=True); sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.func import functional_call, jvp as _fjvp, vjp as _fvjp, jacrev as _jacrev, grad as _grad

# ======================= PARAMETERS (as in the geodesic script) =============
MODE      = os.environ.get("MODE", "mlp")
REGIMES   = ["ntk", "sp", "mup"]
ACTS      = ["gelu", "tanh", "swish", "softplus"]
NSEEDS    = 5

# PAIRS=3 matches measure_profile_christoffel.py. Setting 10 here while that
# file ran 3 means the other 7 pairs must compute Gamma themselves, which is
# expensive.
PAIRS     = int(os.environ.get("PAIRS", "3"))

# Sharded like the Christoffel file. Only needed with REUSE=0, where this file
# computes Gamma itself and is just as heavy.
SHARD      = os.environ.get("SHARD")
SHARD_PLAN = {0: ("ntk", ["gelu","tanh"]), 1: ("ntk", ["swish","softplus"]),
              2: ("sp",  ["gelu","tanh"]), 3: ("sp",  ["swish","softplus"]),
              4: ("mup", ["gelu","tanh"]), 5: ("mup", ["swish","softplus"])}

def plan_cells():
    if SHARD not in (None, ""):
        rg, acts = SHARD_PLAN[int(SHARD)]
        return [(rg, a) for a in acts if a in ACTS]
    return [(rg, a) for rg in REGIMES for a in ACTS]

TGRID     = int(os.environ.get("TGRID", "9"))      # = GEO_TGRID cu (Green quadrature)
FISHER_N  = 2048
MICRO     = 64
FD_EPS    = 3e-3
FD_RICH   = True
LAM_REL   = float(os.environ.get("LAM_REL", "1e-2"))
CG_ITERS  = 300
CG_TOL    = 1e-6
POWER_ITERS = 20
REUSE     = os.environ.get("REUSE", "1") == "1"    # doc lai profile_christoffel_*.csv
SELFTEST  = os.environ.get("SELFTEST", "1") == "1"
ANCHOR    = os.environ.get("ANCHOR", "1") == "1"
BUDGET_H  = float(os.environ.get("BUDGET_H", "11.0"))

DATA_ROOT  = "/kaggle/input/datasets/ANONYMIZED/DATASET"
CKPT_ROOTS = [DATA_ROOT, ".", "/kaggle/input", "/content", "/content/drive/MyDrive"]
OUT_DIR    = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BASE       = 64
T_START    = time.time()

_MODE_CFG = {
    "mlp": dict(tag="pmlp_v2", din=784, k=10, widths=[64,128,256,512,1024,2048,4096]),
    "cnn": dict(tag="pcnn_v2", din=None, k=10, widths=[1,2,4,8]),
    "ts":  dict(tag="pts_v2",  din=64,  k=10, widths=[64,128,256,512,1024,2048,4096]),
}
# ============================================================================

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def out_of_time(): return (time.time() - T_START)/3600.0 > BUDGET_H

# ------------------------------------------------------------------ MODEL
def make_act(n):
    return {"relu":nn.ReLU,"gelu":nn.GELU,"tanh":nn.Tanh,
            "swish":nn.SiLU,"softplus":nn.Softplus}[n]()

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
    def __init__(self, width, act, regime="ntk", din=784, k=10):
        super().__init__()
        self.fc1 = ScaledLinear(din, width, regime, "input")
        self.fc2 = ScaledLinear(width, width, regime, "hidden")
        self.fc3 = ScaledLinear(width, k, regime, "output")
        self.a1 = make_act(act); self.a2 = make_act(act)
        self.width = width; self.regime = regime
    def forward(self, x): return self.fc3(self.a2(self.fc2(self.a1(self.fc1(x)))))
    def opt_groups(self, base_lr):        # = param_{mlp,ts}_v2_shard*.py
        return [{"params":[m.weight, m.bias], "lr": base_lr*m.lr_scale}
                for m in [self.fc1, self.fc2, self.fc3]]

class NetCNN(nn.Module):
    def __init__(self, wm, act, regime="ntk", in_ch=1, k=10):
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
    def opt_groups(self, base_lr):        # = param_cnn_v2_shard*.py
        g = [{"params":[m.weight] + ([m.bias] if hasattr(m, "bias") else []),
              "lr": base_lr*m.lr_scale} for m in [self.c1, self.c2, self.c3, self.fc]]
        g.append({"params":[p for n in [self.n1, self.n2, self.n3] for p in n.parameters()],
                  "lr": base_lr}); return g

def build_net(mode, width, act, regime):
    cfg = _MODE_CFG[mode]
    if mode == "cnn": return NetCNN(width, act, regime, in_ch=1, k=cfg["k"])
    return NetMLP(width, act, regime, din=cfg["din"], k=cfg["k"])

# ------------------------------------------------------------------ PERM
def perm_spec(mode, model):
    if mode == "cnn":
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
            new = torch.as_tensor(linear_sum_assignment(-S.numpy())[1], dtype=torch.long)
            if not torch.equal(new, perms[g]): moved += 1
            perms[g] = new
        if moved == 0: break
    return perms

# ------------------------------------------------------------------ DATA
_C = {}
def load_X(mode):
    if mode in _C: return _C[mode]
    if mode == "mlp":
        import torchvision
        ds = torchvision.datasets.MNIST("./data", train=True, download=True)
        X = (((ds.data.float()/255.0) - 0.1307)/0.3081).reshape(-1, 784)
    elif mode == "cnn":
        import torchvision
        ds = torchvision.datasets.FashionMNIST("./data", train=True, download=True)
        X = (((ds.data.float()/255.0) - 0.2860)/0.3530).unsqueeze(1)
    else:
        g = torch.Generator().manual_seed(1)
        X = torch.randn(20000, _MODE_CFG["ts"]["din"], generator=g)
    _C[mode] = X; return X

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
    d1 = dFz(m, p, b, x, dhat, dhat, FD_EPS,   micro)
    d2 = dFz(m, p, b, x, dhat, dhat, FD_EPS/2, micro)
    t1r = {k: (4*d2[k]-d1[k])/3 for k in d1} if FD_RICH else d1
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

# ------------------------------------------------------------------ SELF-TEST
def self_test():
    log("  [self-test] Gamma(dd) vp == dense, va hang so Green = 1/8 ...")
    old = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    m = NetMLP(6, "tanh", "ntk", din=4, k=3).eval(); x = torch.randn(8, 4)
    p, b = _pb(m); keys = list(p.keys())
    flat = lambda d: torch.cat([d[k].reshape(-1) for k in keys])
    def unflat(v):
        o = {}; i = 0
        for k in keys:
            n = p[k].numel(); o[k] = v[i:i+n].reshape(p[k].shape); i += n
        return o
    torch.manual_seed(3)
    dflat = torch.randn(flat(p).numel()); dflat = dflat/dflat.norm()*3.0
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
    assert rel < 1e-6, f"Gamma mismatch: {rel:.2e}"
    ts = np.linspace(0, 1, 21); G = green_matrix(list(ts)); dt = ts[1]-ts[0]
    xi = (G @ torch.full((len(ts),), 0.7, dtype=torch.float64))*dt
    ge = float((xi - torch.tensor([0.7*t*(1-t)/2 for t in ts])).abs().max())
    assert ge < 1e-10, f"Green quadrature mismatch: {ge:.2e}"
    # the normalised shape of a uniform bump must be 4t(1-t), peaking at t=1/2
    sh = np.array([float(v) for v in xi]); sh = sh/sh.max()
    ref = np.array([4*t*(1-t) for t in ts])
    assert np.abs(sh - ref).max() < 1e-10, "normalised shape(t) is wrong"
    log(f"  [self-test] OK  Gamma rel={rel:.2e}  Green err={ge:.2e}  shape=4t(1-t) OK")
    torch.set_default_dtype(old)

# ------------------------------------------------------------------ CKPT
def _diagnose(tag):
    """Say plainly why no checkpoint was found, instead of repeating '<2 ckpt' dozens of times."""
    log(f"[ckpt] !! no 'ckpt_{tag}' directory containing .pt files was found.")
    seen, n_pt, zips = set(), 0, []
    for root in CKPT_ROOTS:
        if not root or not os.path.isdir(root): continue
        for dp, dns, fns in os.walk(root):
            if dp[len(root):].count(os.sep) <= 2: seen.update(dns)
            for fn in fns:
                if   fn.endswith(".pt"):  n_pt += 1
                elif fn.endswith(".zip"): zips.append(os.path.join(dp, fn))
    cand = sorted(d for d in seen if "ckpt" in d.lower())
    log(f"[ckpt]    scanning: {[r for r in CKPT_ROOTS if r and os.path.isdir(r)]}")
    for root in CKPT_ROOTS:                      # show what is actually there
        if not root or not os.path.isdir(root): continue
        try: top = sorted(os.listdir(root))[:12]
        except OSError: top = []
        log(f"[ckpt]      {root}  ->  {top or 'empty'}")
    log(f"[ckpt]    directories containing 'ckpt': {cand or 'none'}")
    log(f"[ckpt]    total .pt files found anywhere: {n_pt}")
    if zips:
        log(f"[ckpt]    .zip archives were scanned too: {[os.path.basename(z) for z in zips[:3]]} "
            f"-- but none holds a .pt matching ckpt_{tag}")
    if n_pt and not cand:
        log(f"[ckpt]    -> .pt files exist but none sits in a directory named 'ckpt_{tag}'.")
    if cand and f"ckpt_{tag}" not in cand:
        log(f"[ckpt]    -> wrong directory name; it must be 'ckpt_{tag}'")
    if os.path.isdir("/kaggle/input") and not os.listdir("/kaggle/input"):
        log(f"[ckpt]    -> /kaggle/input is empty: no dataset is attached to this notebook.")
        log(f"[ckpt]    -> attach the dataset holding ckpt_{tag}/ as an input.")
    log(f"[ckpt]    expected layout: <anything>/ckpt_{tag}/{{regime}}_{{act}}_w{{w}}_s{{s}}.pt")

_IDX = {}
_ZCACHE = {}

def _zopen(p):
    """Keep .zip handles open: reopening re-reads the central directory each time."""
    if p not in _ZCACHE: _ZCACHE[p] = zipfile.ZipFile(p)
    return _ZCACHE[p]

def _index_dir(root, tag, idx):
    for dp, _, fns in os.walk(root):
        if os.path.basename(dp) != f"ckpt_{tag}": continue
        for fn in fns:
            if fn.endswith(".pt"): idx.setdefault(fn, os.path.join(dp, fn))

def _index_zip(zpath, tag, idx):
    """Read .pt directly out of a .zip, without extracting to disk.

    Two archive layouts are accepted:
      (a) paths of the form .../ckpt_{tag}/xxx.pt  -> only this tag is taken
      (b) a flat archive of .pt files             -> all of them (e.g. ntk.zip)
    An archive holding only ckpt_ of another architecture is skipped, so nothing
    is picked up by mistake.
    """
    try:
        z = _zopen(zpath)
        names = [n for n in z.namelist() if n.endswith(".pt")]
        nested = [n for n in z.namelist() if n.endswith(".zip")]
        if not names: return 0, nested
        norm   = lambda n: "/" + n.replace("\\", "/")
        scoped = [n for n in names if f"/ckpt_{tag}/" in norm(n)]
        other  = any("/ckpt_" in norm(n) for n in names)
        use    = scoped if scoped else ([] if other else names)
        if not use and other:
            log(f"[ckpt]    (skipping {os.path.basename(zpath)}: it only holds ckpt_ of another architecture)")
        for n in use: idx.setdefault(os.path.basename(n), ("zip", zpath, n))
        return len(use), nested
    except Exception as e:
        log(f"[ckpt] !! could not read {zpath}: {e!r}")
        return 0, []

def _build_index(tag):
    if tag in _IDX: return _IDX[tag]
    idx = {}; zips = []
    for root in CKPT_ROOTS:
        if not root or not os.path.isdir(root): continue
        _index_dir(root, tag, idx)                       # already-extracted directories win
        for dp, _, fns in os.walk(root):
            zips += [os.path.join(dp, f) for f in fns if f.endswith(".zip")]
    n_zip = 0; nested = []
    # Deduplicate by real path: nested CKPT_ROOTS see the same archive under two
    # names ("kinput/a.zip" vs "./kinput/a.zip").
    for zp in sorted({os.path.realpath(z) for z in zips}):
        k, nz = _index_zip(zp, tag, idx); n_zip += k; nested += nz
    _IDX[tag] = idx
    log(f"[ckpt] ckpt_{tag}: found {len(idx)} .pt files"
        + (f"  ({n_zip} read straight from .zip)" if n_zip else ""))
    if nested:
        log(f"[ckpt] !! nested .zip inside .zip ({nested[:2]}) -- unpack one layer first:")
        log(f"[ckpt]    !unzip -q '/kaggle/input/<ten>/*.zip' -d /kaggle/working/ckpt")
    if not idx:
        _diagnose(tag)
    else:
        pass
    return idx

def find_ckpt(tag, regime, act, w, s):
    return _build_index(tag).get(f"{regime}_{act}_w{w}_s{s}.pt")

def load_sd(src):
    """src la duong dan .pt, HOAC ("zip", duong_dan_zip, ten_entry)."""
    if isinstance(src, tuple):
        buf = io.BytesIO(_zopen(src[1]).read(src[2]))
        try: d = torch.load(buf, map_location="cpu", weights_only=False)
        except TypeError: buf.seek(0); d = torch.load(buf, map_location="cpu")
    else:
        try: d = torch.load(src, map_location="cpu", weights_only=False)
        except TypeError: d = torch.load(src, map_location="cpu")
    return d["sd"], d.get("acc")

# ------------------------------------------------------------------ TRAIN (chi khi THIEU ckpt)
# Copied verbatim from the training scripts, including the seed formula. One
# detail wrong and the network trained here differs from the original, and the
# measurement no longer matches param_geo_*.csv (anchor_check would report it).
TRAIN        = os.environ.get("TRAIN", "1") == "1"      # TRAIN=0 de tat han
TRAIN_SEED_BASE = 4321
TRAIN_ACTS   = ["relu","gelu","tanh","swish","softplus"]  # relu must stay: the seed
                                                          # depends on the index in this list
TRAIN_LR, TRAIN_BATCH, TRAIN_WARMUP, TRAIN_CLIP = 0.1, 256, 8, 1.0
TRAIN_EPOCHS = {"mlp": 30, "cnn": 100, "ts": 100}         # per architecture

def set_seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def ckpt_dir(tag):
    d = os.path.join(OUT_DIR, f"ckpt_{tag}"); os.makedirs(d, exist_ok=True); return d

def _train_xy(mode, train=True):
    """Du lieu CO NHAN de train (khac load_X: Fisher khong can nhan)."""
    key = ("xy", mode, train)
    if key in _C: return _C[key]
    if mode == "mlp":
        import torchvision
        ds = torchvision.datasets.MNIST("./data", train=train, download=True)
        X = (((ds.data.float()/255.0) - 0.1307)/0.3081).reshape(-1, 784); Y = ds.targets.clone()
    elif mode == "cnn":
        import torchvision
        ds = torchvision.datasets.FashionMNIST("./data", train=train, download=True)
        X = (((ds.data.float()/255.0) - 0.2860)/0.3530).unsqueeze(1); Y = ds.targets.clone()
    else:                                   # teacher-student: teacher ReLU width 32, SP, seed 1234
        n = 20000 if train else 10000
        set_seed(1234); t = NetMLP(32, "relu", "sp", din=64, k=10).to(DEVICE).eval()
        for p in t.parameters(): p.requires_grad_(False)
        g = torch.Generator().manual_seed(1 if train else 2)
        X = torch.randn(n, 64, generator=g); ys = []
        with torch.no_grad():
            for i in range(0, n, 4096): ys.append(t(X[i:i+4096].to(DEVICE)).argmax(1).cpu())
        Y = torch.cat(ys)
    _C[key] = (X, Y); return X, Y

def train_one(mode, tag, regime, act, w, s):
    """Train exactly the original network, save the .pt, and register it in the index."""
    cfg = _MODE_CFG[mode]; ep = TRAIN_EPOCHS[mode]
    set_seed(TRAIN_SEED_BASE + REGIMES.index(regime)*100000
             + cfg["widths"].index(w)*100 + TRAIN_ACTS.index(act)*7 + s)
    m = build_net(mode, w, act, regime).to(DEVICE).train()
    X, Y = _train_xy(mode, True)
    w0 = torch.cat([p.detach().reshape(-1) for p in m.parameters()]).clone()
    opt = torch.optim.SGD(m.opt_groups(TRAIN_LR), momentum=0.9)
    wu = min(TRAIN_WARMUP, max(1, ep//5))          # linear warmup, then cosine decay
    def _ll(e):
        if e < wu: return (e+1)/wu
        pr = (e-wu)/max(ep-wu, 1); return 0.5*(1.0 + math.cos(math.pi*pr))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, _ll); n = X.shape[0]
    for e in range(ep):
        perm = torch.randperm(n)
        for i in range(0, n, TRAIN_BATCH):
            idx = perm[i:i+TRAIN_BATCH]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(X[idx].to(DEVICE)), Y[idx].to(DEVICE)).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), TRAIN_CLIP)
            opt.step()
        sch.step()
    wT = torch.cat([p.detach().reshape(-1) for p in m.parameters()])
    wmove = float((wT - w0).norm()/max(float(w0.norm()), 1e-9))
    m.eval()
    Xe, Ye = _train_xy(mode, False); c = tot = 0
    with torch.no_grad():
        for i in range(0, Xe.shape[0], 1024):
            xb = Xe[i:i+1024].to(DEVICE); yb = Ye[i:i+1024].to(DEVICE)
            c += int((m(xb).argmax(1) == yb).sum()); tot += yb.numel()
    acc = c/max(tot, 1)
    sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    fn = f"{regime}_{act}_w{w}_s{s}.pt"
    path = os.path.join(ckpt_dir(tag), fn)
    torch.save({"sd": sd, "acc": acc, "dF": None, "wmove": wmove}, path)
    _IDX.setdefault(tag, {})[fn] = path        # register at once, so no rescan is needed
    del m
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return sd, acc

def load_or_train(mode, tag, regime, act, w, want):
    """Load checkpoints, training any that are missing. Returns (sds, accs, got) in seed order."""
    def _scan():
        sds = []; accs = []; got = []
        for s in range(NSEEDS):
            cp = find_ckpt(tag, regime, act, w, s)
            if cp is None: continue
            sd, acc = load_sd(cp); sds.append(sd); accs.append(acc); got.append(s)
        return sds, accs, got
    sds, accs, got = _scan()
    need = 1 + max(max(i, j) for (i, j) in want)      # chi can toi seed nay
    miss = [s for s in range(need) if s not in got]
    if miss and TRAIN:
        log(f"  [{regime}/{act}/w{w}] THIEU ckpt seed {miss} -> TRAIN "
            f"({TRAIN_EPOCHS[mode]} epoch/mang, config = param_{mode}_v2_shard*.py)")
        for s in miss:
            t0 = time.time()
            _, acc = train_one(mode, tag, regime, act, w, s)
            log(f"      seed {s}: acc={acc:.4f}  ({time.time()-t0:.0f}s)")
        sds, accs, got = _scan()
    return sds, accs, got

# ------------------------------------------------------------------ CSV
def t_grid(): return list(np.linspace(0, 1, TGRID))
def r_cols(): return [f"xirel_t{t:.3f}" for t in t_grid()]    # ||xi(t)||/||D||
def s_cols(): return [f"shape_t{t:.3f}" for t in t_grid()]    # ||xi(t)||/sup||xi||

def COLS():
    return (["mode","regime","act","width","seedA","seedB","dnorm","lam_rel"]
            + r_cols() + s_cols()
            + ["dev_geo","dev_rel","t_peak","skew","src","cg_resid","fd_instab",
               "accA","accB","status"])

def _seek_csv(name):
    p = os.path.join(DATA_ROOT, name)
    if os.path.exists(p): return p
    for c in sorted(glob.glob("/kaggle/input/**/" + name, recursive=True)): return c
    for c in sorted(glob.glob("**/" + name, recursive=True)): return os.path.abspath(c)
    return None

def _sanitize(path):
    if not os.path.exists(path): return
    lines = open(path).read().splitlines()
    if not lines: return
    nf = len(lines[0].split(","))
    keep = [l for l in lines if l.strip() and len(l.split(",")) == nf]
    if len(keep) != len(lines):
        open(path, "w").write("\n".join(keep) + "\n")
        log(f"[resume] bo {len(lines)-len(keep)} dong hong -> se chay lai cac cap do")

def restore_csv(path, name):
    if not os.path.exists(path):
        src = _seek_csv(name)
        if src and os.path.abspath(src) != os.path.abspath(path):
            try: shutil.copy(src, path); log(f"[resume] khoi phuc CSV tu {src}")
            except Exception as e: log(f"[resume] copy failed ({e!r}) -> starting from scratch")
    # An existing CSV must carry exactly this run's columns. Changing TGRID (or
    # LAM_REL) changes their number, so appending would produce a ragged file
    # that pandas cannot read. In that case the old file is renamed and the run
    # starts over, rather than corrupting data already collected.
    if os.path.exists(path):
        head = open(path).readline().strip()
        if head and head != ",".join(COLS()):
            bak = path + ".oldcols.bak"
            os.replace(path, bak)
            log(f"[resume] !! the existing CSV has a different column set "
                f"({len(head.split(','))} vs {len(COLS())}) -- did TGRID/LAM_REL change?")
            log(f"[resume] !! renamed it to {bak} -> starting from scratch")
    _sanitize(path)
    if os.path.exists(path):
        n = max(sum(1 for _ in open(path)) - 1, 0)
        log(f"[resume] {path}: {n} existing data rows")
    else:
        log(f"[resume] no existing CSV -> starting from scratch (normal on a first run)")

def load_done(path):
    done = set()
    if not os.path.exists(path): return done
    for ln in open(path):
        f = ln.strip().split(",")
        if len(f) >= 6 and f[-1] == "ok":
            done.add((f[1], f[2], f[3], f[4], f[5]))
    return done

def write_row(path, row):
    cols = COLS(); new = not os.path.exists(path)
    with open(path, "a") as f:
        if new: f.write(",".join(cols) + "\n")
        f.write(",".join(str(row.get(c, "")) for c in cols) + "\n"); f.flush()

# ------------------------------------------------------------------ DOC LAI FILE CHRISTOFFEL
def load_reuse(mode):
    """profile_christoffel_{mode}.csv already holds xinorm_t*: read it, no GPU needed."""
    if not REUSE: return {}
    src = _seek_csv(f"profile_christoffel_{mode}.csv")
    if not src:
        log(f"[reuse] profile_christoffel_{mode}.csv not found -> computing Gamma here"); return {}
    import pandas as pd
    d = pd.read_csv(src); d = d[d.status.astype(str) == "ok"]
    xc = [f"xinorm_t{t:.3f}" for t in t_grid()]
    if not set(xc).issubset(d.columns):
        log(f"[reuse] {src} khong khop TGRID={TGRID} -> se tu tinh Gamma"); return {}
    if "lam_rel" in d:
        d = d[pd.to_numeric(d.lam_rel, errors="coerce").round(6) == round(LAM_REL, 6)]
    out = {}
    for _, r in d.iterrows():
        key = (str(r.regime), str(r.act), str(int(r.width)), str(int(r.seedA)), str(int(r.seedB)))
        out[key] = dict(xin=np.array([float(r[c]) for c in xc]), dnorm=float(r.dnorm),
                        cg_resid=r.get("cg_resid", ""), fd_instab=r.get("fd_instab", ""),
                        accA=r.get("accA", ""), accB=r.get("accB", ""))
    log(f"[reuse] read {len(out)} pairs from {src} -> no GPU needed for them")
    return out

def _shape_row(mode, regime, act, w, i, j, xin, dn, src, cg="", fd="", aA="", aB=""):
    ts = np.array(t_grid())
    sup = float(xin.max()); dev_rel = sup/max(dn, 1e-30)
    xirel = xin/max(dn, 1e-30)
    shape = xin/max(sup, 1e-30)
    kpk = int(np.argmax(xin)); t_peak = float(ts[kpk])
    row = dict(mode=mode, regime=regime, act=act, width=w, seedA=i, seedB=j,
               dnorm=f"{dn:.6e}", lam_rel=LAM_REL,
               dev_geo=f"{sup:.6e}", dev_rel=f"{dev_rel:.6e}",
               t_peak=f"{t_peak:.4f}", skew=f"{(t_peak-0.5)*2:.4f}",
               src=src, cg_resid=cg, fd_instab=fd, accA=aA, accB=aB, status="ok")
    for c, v in zip(r_cols(), xirel): row[c] = f"{v:.6e}"
    for c, v in zip(s_cols(), shape): row[c] = f"{v:.4f}"
    return row, dev_rel, t_peak

# ------------------------------------------------------------------ NEO VAO SO CU
def anchor_check(mode, path):
    import pandas as pd
    src = _seek_csv(f"param_geo_{mode}.csv") or os.path.join("result-1", f"param_geo_{mode}.csv")
    if not src or not os.path.exists(src):
        log(f"[anchor] param_geo_{mode}.csv not found -> skipping"); return
    old = pd.read_csv(src); old = old[old.status.astype(str) == "ok"]
    if "lam_rel" in old:
        old = old[pd.to_numeric(old.lam_rel, errors="coerce").round(6) == round(LAM_REL, 6)]
    new = pd.read_csv(path); new = new[new.status.astype(str) == "ok"]
    if not len(new) or not len(old):
        log("[anchor] chua co du lieu de doi chieu"); return
    key = ["regime","act","width","seedA","seedB"]
    for c in key[2:]:
        old[c] = pd.to_numeric(old[c], errors="coerce"); new[c] = pd.to_numeric(new[c], errors="coerce")
    m = new.merge(old[key + ["dev_rel"]].rename(columns={"dev_rel":"dev_rel_old"}), on=key, how="inner")
    if not len(m):
        log("[anchor] no pair matched -> skipping"); return
    a = pd.to_numeric(m["dev_rel"], errors="coerce"); b = pd.to_numeric(m["dev_rel_old"], errors="coerce")
    rel = ((a - b).abs()/b.abs().clip(lower=1e-30)).dropna()
    log(f"[anchor] doi chieu {len(m)} cap: dev_rel lech trung vi={rel.median():.3%} max={rel.max():.3%}")
    log("[anchor] " + ("OK -- khop lan do cu" if rel.max() <= 0.02 else
                       "!! lech > 2%: kiem tra FD_EPS / CG_TOL / LAM_REL"))

# ------------------------------------------------------------------ MAIN
def run_mode(mode):
    cfg = _MODE_CFG[mode]; tag = cfg["tag"]
    out = os.path.join(OUT_DIR, f"profile_shape_{mode}.csv")
    restore_csv(out, f"profile_shape_{mode}.csv")
    done = load_done(out)
    reuse = load_reuse(mode)

    ts = t_grid(); G = green_matrix(ts); dt = ts[1]-ts[0]
    n_skip = 0
    Xf = None                                    # load data only if something must be computed
    log(f"=== MODE={mode} SHARD={SHARD if SHARD not in (None,'') else 'het'} cells={plan_cells()} widths={cfg['widths']} TGRID={TGRID} "
        f"lam_rel={LAM_REL:g} pairs={PAIRS} reuse={len(reuse)} DEVICE={DEVICE} -> {out}")

    for (regime, act) in plan_cells():
        for w in cfg["widths"]:
            if out_of_time():
                log(f"[budget] het {BUDGET_H}h -> dung sach"); return out
            want = [(i, j) for (i, j) in
                    list(itertools.combinations(range(NSEEDS), 2))[:PAIRS]
                    if (regime, act, str(w), str(i), str(j)) not in done]
            if not want:
                log(f"  [{regime}/{act}/w{w}] already done -> skipping"); continue

            # ---- (A) pairs available from the Christoffel file need no GPU ----
            left = []
            for (i, j) in want:
                r = reuse.get((regime, act, str(w), str(i), str(j)))
                if r is None: left.append((i, j)); continue
                row, dv, tp = _shape_row(mode, regime, act, w, i, j, r["xin"], r["dnorm"],
                                         "reuse", r["cg_resid"], r["fd_instab"],
                                         r["accA"], r["accB"])
                write_row(out, row)
                log(f"  [{i}-{j}] (reuse) dev_rel={dv:.3e} t_peak={tp:.3f}")
            if not left: continue

            # ---- (B) con lai: tu tinh Gamma ----
            try:
                if Xf is None:
                    Xf = load_X(mode)[:FISHER_N].to(DEVICE)
                sds, accs, got = load_or_train(mode, tag, regime, act, w, want)
                if len(sds) < 2:
                    miss = [x for x in range(NSEEDS) if x not in got]
                    n_skip += 1
                    log(f"  [{regime}/{act}/w{w}] THIEU DU LIEU: chi co {len(sds)}/{NSEEDS} ckpt "
                        f"(thay seed {got}, thieu {miss}) -- can >=2 de tao 1 cap -> bo o nay")
                    continue
                log(f"=== {regime}/{act}/w{w}  ({len(sds)} nets, {len(left)} cap phai tinh) ===")
                ref = build_net(mode, w, act, regime).to(DEVICE).eval()
                p_ref, b_ref = _pb(ref)
                ag, gs = perm_spec(mode, build_net(mode, w, act, regime))
                to_params = lambda sd: {k: sd[k].to(DEVICE) for k in p_ref.keys()}

                for (i, j) in left:
                    if i >= len(sds) or j >= len(sds): continue
                    if out_of_time():
                        log(f"[budget] het {BUDGET_H}h -> dung sach"); return out
                    try:
                        perms = weight_matching(ag, gs, sds[i], sds[j], iters=8, seed=i*13 + j)
                        pA = to_params(sds[i]); pB = to_params(apply_perm(sds[j], ag, perms))
                        delta = {k: pB[k] - pA[k] for k in pA}; dn = _vnorm(delta)
                        lmax = lam_max(ref, pA, b_ref, Xf, MICRO, POWER_ITERS, seed=17)
                        lam = max(LAM_REL*lmax, 1e-12)

                        gammas = []; resids = []; fdis = []; x0 = None
                        for tt in ts:
                            pt = {k: (1-tt)*pA[k] + tt*pB[k] for k in pA}
                            g, res, fdi = christoffel_dd(ref, pt, b_ref, Xf, delta, lam, MICRO, x0=x0)
                            x0 = g
                            gammas.append({k: v.detach().cpu() for k, v in g.items()})
                            resids.append(res); fdis.append(fdi)

                        keys = list(delta.keys()); xin = []
                        for ti in range(len(ts)):
                            xi = None
                            for si in range(len(ts)):
                                w_ = float(G[ti, si])*dt
                                if w_ == 0.0: continue
                                xi = {k: (gammas[si][k]*w_ if xi is None else xi[k] + gammas[si][k]*w_)
                                      for k in keys}
                            xin.append(_vnorm(xi) if xi is not None else 0.0)
                        xin = np.array(xin)

                        row, dv, tp = _shape_row(mode, regime, act, w, i, j, xin, dn, "compute",
                                                 f"{max(resids):.2e}", f"{max(fdis):.3e}",
                                                 accs[i], accs[j])
                        write_row(out, row)
                        log(f"  [{i}-{j}] ||d||={dn:.3g} dev_rel={dv:.3e} t_peak={tp:.3f} "
                            f"cgres<={max(resids):.1e}")
                    except Exception as e:
                        traceback.print_exc()
                        write_row(out, dict(mode=mode, regime=regime, act=act, width=w,
                                            seedA=i, seedB=j, status="error:"+repr(e)[:40]))
                del sds
                if DEVICE == "cuda": torch.cuda.empty_cache()
            except Exception as e:
                traceback.print_exc()
                write_row(out, dict(mode=mode, regime=regime, act=act, width=w,
                                    status="cell-error:"+repr(e)[:40]))
    if n_skip:
        log(f"!! {n_skip} o bi BO vi thieu ckpt -- xem cac dong 'THIEU DU LIEU' o tren.")
    log(f"DONE shape profile [{mode}]")
    return out

def aggregate(mode, path):
    import pandas as pd
    if not os.path.exists(path): return
    d = pd.read_csv(path); d = d[d.status.astype(str) == "ok"].copy()
    if not len(d): return
    cols = [c for c in (r_cols()+s_cols()+["dev_rel","t_peak","skew","dnorm"]) if c in d]
    for c in cols + ["width"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    g = d.groupby(["mode","regime","act","width"])[cols].median().reset_index()
    g.insert(4, "n_pairs", d.groupby(["mode","regime","act","width"]).size().values)
    fin = os.path.join(OUT_DIR, f"profile_shape_{mode}_cell.csv")
    g.to_csv(fin, index=False); log(f"-> {fin}  ({len(g)} o)")

    log("\n  shape of the deviation bump (median per regime):")
    log(f"    {'regime':6s} {'dev_rel':>11s} {'t_peak':>8s} {'skew':>7s}   "
        f"(t_peak=0.5 & skew=0 => doi xung nhu mo hinh do choi)")
    for reg, s in g.groupby("regime"):
        # index with [], not attribute access: 'skew' collides with a DataFrame method
        log(f"    {reg:6s} {s['dev_rel'].median():11.4e} "
            f"{s['t_peak'].median():8.3f} {s['skew'].median():+7.3f}")

def main():
    modes = list(_MODE_CFG) if MODE == "all" else [MODE]
    log(f"=== measure_profile_shape  modes={modes}  TGRID={TGRID} lam_rel={LAM_REL:g} "
        f"reuse={'on' if REUSE else 'off'} ===")
    if SELFTEST: self_test()
    for m in modes:
        p = run_mode(m)
        aggregate(m, p)
        if ANCHOR:
            try: anchor_check(m, p)
            except Exception as e: log(f"[anchor] bo qua ({e!r})")

if __name__ == "__main__":
    main()
