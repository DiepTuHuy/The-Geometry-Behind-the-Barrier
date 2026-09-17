#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
measure_tightness.py -- is the Fisher-barrier inequality (Theorem 4.1) TIGHT,
or is it only a bound?   MEASURE_TIGHTNESS_SOURCE_V1

THE QUESTION.  Theorem 4.1 states

    B(gamma_lin)  <=  min{R(w_A), R(w_B)}  +  (1/2)|L(w_A) - L(w_B)| ,

    R(w_0) = ||D|| ||grad L(w_0)||            (T1, endpoint gradient)
           + (1/2) D' F(w_0) D                (T2, Fisher energy)
           + (1/2) rho*(w_0) M_2 ||D||^2      (T3, predictive uncertainty)
           + (M_3/6) ||D||^3                  (T4, third-order smoothness)

with D = w_B - w_A.  Asking whether it is tight means asking for the size of
the ratio B / RHS.  The ratio is not computable from the released measurement
round: of the four terms of R, only T2 is in the CSVs (`flen_A`, `flen_B` are
already (1/2) D'F D, not D'F D), and rho* is there for the A anchor only.  The
endpoint gradient norm was never written out, and M_2, M_3 were never measured
at all -- `src/09_scale/measure_scale.py` says so in its own header.  This file
supplies exactly the missing terms and evaluates the whole inequality.

THE ONE-SIDED TRICK, WHICH IS THE POINT OF THE FILE.  M_2 and M_3 are suprema
over an open convex neighbourhood U of the segment,

    M_2 = sqrt(K) sup_{w in U, x, k} ||grad^2_w f_k(x;w)||_op ,
    M_3 = sup_{w in U} sup_{||u||=||v||=||z||=1} |D^3 L(w)[u,v,z]| ,

and no finite computation evaluates a supremum over a neighbourhood.  What IS
computable is the same quantity restricted to the segment and to the single
direction D-hat = D/||D||:

    M2_dir = sqrt(K) max_{t, x, k} |D^2_w f_k(x; gamma(t))[D-hat, D-hat]| ,
    M3_dir =         max_t         |D^3   L(gamma(t))[D-hat, D-hat, D-hat]| .

Because the segment lies in U, and because |u' H u| <= ||H||_op for unit u,

    M2_dir <= M_2  and  M3_dir <= M_3   ==>   R_hat <= R ,

where R_hat is R with the two suprema replaced by these.  Hence

    B / R_hat  >=  B / R .

So B / R_hat is a CERTIFIED UPPER BOUND on the tightness ratio.  If the
measured B / R_hat comes out at 1e-3, the true inequality is loose by at least
three orders of magnitude and the answer is settled in the loose direction --
without ever having evaluated the supremum the theorem actually names.  This is
the only direction in which a finite measurement can settle the question, and
it happens to be the direction the paper needs, since the main text already
declines to claim sharpness.  A ratio near 1 would NOT prove tightness; it
would only mean the surrogate is too small to decide, and the run says so.

WHY THE SCALE FACTORS NEVER APPEAR.  ||D|| = Theta(n) here, so ||D||^3 at
n = 4096 is around 1e11 and M3_dir is around 1e-11: forming either alone in
float32 is pointless.  Parametrise by the segment instead.  With
phi(t) = L(w_A + tD) and psi_k(t) = f_k(x; w_A + tD),

    phi'''(t)   = D^3 L[D,D,D]     = ||D||^3 D^3 L[D-hat,D-hat,D-hat] ,
    psi_k''(t)  = D^2 f_k[D,D]     = ||D||^2 D^2 f_k[D-hat,D-hat] ,

so the two terms that carry the scale factors collapse to

    T3 = (1/2) rho*(w_0) sqrt(K) max_{t,x,k} |psi_k''(t)| ,
    T4 = (1/6)           max_t   |phi'''(t)| ,

and the powers of ||D|| cancel exactly.  Nothing large is ever formed.  M2_dir
and M3_dir are still written to the CSV, divided back out, because the prose
wants the constants themselves.

HOW EACH DERIVATIVE IS TAKEN.  phi''(t) and psi_k''(t) are EXACT: forward-over-
forward JVP with tangent D, no finite difference anywhere.  Only phi'''(t) is a
difference, and it is a central difference OF THE EXACT phi'' -- the same
"difference an exact derivative" pattern
`04_profile/measure_profile_christoffel.py` uses for dF, chosen for the same
reason: differencing an exact second derivative amplifies float error by 1/h,
differencing the loss itself three times amplifies it by 1/h^3.  Steps are
taken in t, not in parameter norm: the segment is the scale on which the loss
actually varies.

phi'' is sampled on ONE uniform grid of NGRID_PHI points and the stencil is
then taken between neighbours, rather than planting an independent Richardson
stencil at each point where the supremum is wanted.  That is not a detail: the
per-point form costs four phi'' evaluations per sup point, so 9 sup points cost
36 sweeps, while 33 grid points resolve the sup at 29 places for 33 sweeps --
cheaper AND three times the coverage of the supremum, which is the whole object
M_3 names.  It also makes the error estimate free: the O(h^4) five-point
stencil is the value taken and the O(h^2) three-point stencil on the same
samples is what it is compared against, so `fd_instab` costs no forward pass at
all, where a second step size would have cost another entire grid.  A row whose
M_3 is numerical noise is identifiable rather than merely believed.

WHAT ELSE IT RECOMPUTES, AND WHY.  B, t*, L_A, L_B, ||D||, flen_A, flen_B and
rho_A are all already released.  This file measures them again anyway, from the
same checkpoints through the same primitives, for two reasons: the row is then
self-contained, so the tightness ratio needs no join against another CSV and no
convention has to be re-derived downstream; and the recomputed columns are a
free cross-check of the whole pipeline against `param_final_{MODE}.csv`, which
`selfcheck()` performs and reports.  rho_B is new -- the released round wrote
rho at A, mid, max and t*, never at B, and min{R(w_A), R(w_B)} needs both.

COST.  Everything is forward or forward-mode AD except one backward per anchor
for ||grad L||.  There is no conjugate gradient, no Christoffel symbol, no
geodesic and no inverse anywhere, so none of the caveats of Appendix D.5
applies to these numbers.

Counting in sweeps of the EVAL_N set, one sweep = one forward pass over all
EVAL_N samples, a pair costs about

    41    barrier grid          L(t) at TGRID_FINE points
     2    the two anchors       L, rho* at w_A and w_B
     6    ||grad L||            2 anchors x ~3 sweeps for forward+backward
     2    Fisher energy         2 anchors, on the smaller FISHER_N set
    14    psi_k'' sup           NGRID_M2 points, FISHER_N set, ~4x per point
   132    phi'' grid            NGRID_PHI points x ~4x per nested JVP
   ----
   ~197 sweeps

so phi'' is two thirds of it and the barrier grid a fifth.  That is roughly 4x
PHASE=full of `03_final/*_measure_final.py` on the same pair, and still well
under PHASE=all, which adds 300-iteration CG solves this file does not do.
Cost per sweep grows with the parameter count, so the widest one or two grid
points dominate the total: for the MLP, n=4096 alone is about half the run.
Set NGRID_PHI=17 to halve the dominant term at the price of coarser sup
coverage, or PAIRS=3 to cut every cell.

NEEDS:
    ckpt_{RUN_TAG}/{regime}_{act}_w{w}_s{s}.pt      (required)
    param_final_{MODE}.csv                          (optional, for selfcheck)
    MNIST / FashionMNIST under ./data               (mlp / cnn; ts is synthetic)

OUTPUT:
    tightness_{MODE}.csv        per seed pair, resumable
    tightness_{MODE}_cell.csv   aggregated per cell

ENV:
    MODE=mlp|cnn|ts   PAIRS=10   NGRID_PHI=33   NGRID_M2=17   SELFCHECK=1
    BENCH=1           time one pair per width and project the full grid, then stop
    MICRO_L=128       loss-sweep micro-batch (default 512, or 128 for the CNN)
"""
import os, sys, time, math, glob, itertools, traceback
try:
    sys.stdout.reconfigure(line_buffering=True); sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.func import functional_call, jvp as _fjvp, vjp as _fvjp, grad as _grad

def _tv():
    import torchvision; return torchvision

# ==================================================================== CONFIG
MODE       = os.environ.get("MODE", "mlp")
RESUME     = True
SELFCHECK  = os.environ.get("SELFCHECK", "1") == "1"
RUN_TAG    = {"mlp":"pmlp_v2","cnn":"pcnn_v2","ts":"pts_v2"}[MODE]

WIDTHS   = {"mlp":[64,128,256,512,1024,2048,4096],
            "cnn":[1,2,4,8],
            "ts" :[64,128,256,512,1024,2048,4096]}[MODE]
ACTS     = ["gelu","tanh","swish","softplus"]      # C^3, same four as every other script
REGIMES  = ["ntk","sp","mup"]
NSEEDS   = 5
PAIRS    = int(os.environ.get("PAIRS", "10"))
# BENCH=1 measures ONE pair per width and prints the projected cost of the full
# grid, then stops.  Two or three minutes of the real machine beats any
# arithmetic done off it: the projection uses this GPU, this torch build and
# these checkpoints, so it prices the run that is actually about to happen.
N_REGIMES, N_ACTS = len(REGIMES), len(ACTS)   # the FULL grid, before BENCH
N_CELLS  = N_REGIMES*N_ACTS                   # narrows REGIMES/ACTS below
BENCH    = os.environ.get("BENCH", "0") == "1"
if BENCH:
    PAIRS = 1
    RESUME = False          # a benchmark must not skip the pair it is timing
    # One cell is enough to price all of them: regime changes no tensor shape
    # and the activation changes only an elementwise op, so cost per pair is
    # set by the width.  Time one pair at each width, project the rest.
    REGIMES, ACTS = REGIMES[:1], ACTS[:1]

TGRID_FINE = 41     # L(t) -> B and t*, identical to the released round
NGRID_PHI  = int(os.environ.get("NGRID_PHI", "33"))  # uniform grid for phi''
NGRID_M2   = int(os.environ.get("NGRID_M2", "17"))   # grid for the psi_k'' sup
EVAL_N     = 10000  # samples for L, rho*, grad L        (= released round)
FISHER_N   = 2048   # samples for D'FD and for psi_k''   (= released round)
MICRO      = 64
# Loss sweeps take a wider micro-batch than the Fisher path, but `phi2_grid`
# runs a NESTED JVP through the same batch and so holds two tangent copies of
# every activation.  For the MLP and teacher-student the activations are
# (batch, n) and 512 is comfortable; the CNN's are (batch, 128, 28, 28), which
# at 512 is ~200 MB per tensor before tangents and will OOM a 16 GB card at the
# widest multiplier.  Hence the smaller CNN default; raise it if memory allows.
MICRO_L    = int(os.environ.get("MICRO_L", "128" if MODE == "cnn" else "512"))

CKPT_ROOTS = [".","/kaggle/input","/content","/content/drive/MyDrive"]
OUT_DIR    = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DIN,K      = {"mlp":(784,10),"cnn":(None,10),"ts":(64,10)}[MODE]
BASE       = 64
SQRT_K     = math.sqrt(K)

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def set_seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def _first(pats):
    for p in pats:
        if not p: continue
        h = sorted(glob.glob(p, recursive=True))
        if h: return h[0]
    return None

# ==================================================================== MODEL
# Copied verbatim from 03_final/*_measure_final.py.  These scripts duplicate
# their primitives rather than share a module, so that any one of them can be
# dropped into a fresh Kaggle session alone; the copy is deliberate.
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

# ==================================================================== PERM
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

# ==================================================================== DATA
_CACHE = {}
def load_data_xy():
    if "xy" in _CACHE: return _CACHE["xy"]
    if MODE == "mlp":
        ds = _tv().datasets.MNIST("./data", train=True, download=True)
        X = ((ds.data.float()/255.0) - 0.1307)/0.3081; X = X.reshape(-1, 784); Y = ds.targets.clone()
    elif MODE == "cnn":
        ds = _tv().datasets.FashionMNIST("./data", train=True, download=True)
        X = ((ds.data.float()/255.0) - 0.2860)/0.3530; X = X.unsqueeze(1); Y = ds.targets.clone()
    else:
        g = torch.Generator().manual_seed(1); X = torch.randn(20000, DIN, generator=g)
        set_seed(1234); teach = NetMLP(32, "relu", "sp").eval()   # same fixed teacher as the geo script
        with torch.no_grad(): Y = teach(X).argmax(1)
    _CACHE["xy"] = (X, Y); return X, Y

# ==================================================================== PRIMITIVES
def _pb(m):
    return ({k: v.detach() for k, v in m.named_parameters()},
            {k: v.detach() for k, v in m.named_buffers()})
def _call(m, p, b, x): return functional_call(m, {**p, **b}, (x,))
def _vdot(a, b): return float(sum((a[k]*b[k]).sum() for k in a))
def _vnorm(a): return float(torch.sqrt(torch.clamp(sum((a[k]*a[k]).sum() for k in a), min=0)))
def _vaxpy(a, c, b): return {k: a[k] + c*b[k] for k in a}
def _lerp(pA, pB, t): return {k: (1-t)*pA[k] + t*pB[k] for k in pA}

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
def loss_acc_rho(m, p, b, x, y, micro=MICRO_L):
    """L = cross-entropy;  rho* = E||p_w(x)-e_y||_2 (Definition 2.3);  acc."""
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

def flen_at(m, pt, b, xf, delta, micro=MICRO):
    """Returns ((1/2) D'FD, D'FD/||D||^2) -- the T2 term and the Rayleigh quotient.
    Same convention as 03_final: `flen_*` in the released CSVs is ALREADY halved."""
    q  = _vdot(delta, fisher_vp(m, pt, b, xf, delta, micro))
    d2 = max(_vdot(delta, delta), 1e-30)
    return 0.5*q, q/d2

# ------------------------------------------------------- THE MISSING TERMS
def grad_norm(m, p, b, x, y, micro=MICRO_L):
    """||grad L(w)||_2 over the same EVAL_N samples that define L.  T1.

    The only reverse-mode call in the file.  L is the MEAN cross-entropy, so the
    per-micro-batch sums are accumulated and divided once at the end."""
    n = x.shape[0]; acc = None
    for i in range(0, n, micro):
        xb = x[i:i+micro]; yb = y[i:i+micro]
        def Lsum(pp): return F.cross_entropy(_call(m, pp, b, xb), yb, reduction="sum")
        g = _grad(Lsum)(p)
        acc = {k: g[k].detach() for k in g} if acc is None else {k: acc[k] + g[k].detach() for k in acc}
    return _vnorm({k: acc[k]/n for k in acc})


def d2f_seg(m, pt, b, x, delta, micro=MICRO):
    """max over the batch and over k of |psi_k''| at one t, where
    psi_k(s) = f_k(x; gamma(t) + s D).  EXACT: forward-over-forward JVP.

    Returns ||D||^2 max_{x,k} |D^2 f_k[D-hat, D-hat]|, i.e. already carrying the
    ||D||^2 that T3 needs, so the caller never forms ||D||^2 itself."""
    B = x.shape[0]; mx = 0.0
    for i in range(0, B, micro):
        xb = x[i:i+micro]
        def f(pp):  return _call(m, pp, b, xb)                    # [b, K]
        def df(pp): return _fjvp(f, (pp,), (delta,))[1]           # [b, K]
        d2 = _fjvp(df, (pt,), (delta,))[1]                        # [b, K]
        mx = max(mx, float(d2.abs().max()))
    return mx


def d2L_seg(m, pt, b, x, y, delta, micro=MICRO):
    """phi''(t) for phi(s) = L(gamma(t) + s D).  EXACT, forward-over-forward."""
    n = x.shape[0]; tot = 0.0
    for i in range(0, n, micro):
        xb = x[i:i+micro]; yb = y[i:i+micro]
        def Lsum(pp): return F.cross_entropy(_call(m, pp, b, xb), yb, reduction="sum")
        def dL(pp):   return _fjvp(Lsum, (pp,), (delta,))[1]
        tot += float(_fjvp(dL, (pt,), (delta,))[1])
    return tot/n


def phi2_grid(m, pA, pB, b, x, y, delta, ts, micro=MICRO):
    """phi''(t) at every t of a UNIFORM grid, exact at each point.

    This is the expensive object in the file -- roughly two thirds of the work
    of a pair -- so it is evaluated once per grid point and nothing else.  An
    earlier version placed an independent 4-point Richardson stencil at each of
    9 sup points, which is 36 evaluations of this same function to resolve the
    segment at 9 places.  Sampling phi'' on one uniform grid instead lets the
    stencil be taken BETWEEN neighbouring grid points, so 33 evaluations
    resolve it at 29 places: fewer forward sweeps and three times the coverage
    of the supremum, which is what M_3 is."""
    return np.array([d2L_seg(m, _lerp(pA, pB, float(t)), b, x, y, delta, micro)
                     for t in ts])


def d3_from_grid(g2, h):
    """phi'''(t) at the interior points of a uniform grid of phi'' values.

    Two stencils on the same samples, both standard central differences of the
    first derivative of g2:  the 5-point one is O(h^4) and is the value taken;
    the 3-point one is O(h^2) and exists only to be disagreed with.  Their
    relative gap is the `fd_instab` diagnostic, so it costs no extra forward
    pass at all -- unlike a second step size, which would cost another full
    grid.  Returns (t index array, phi''' values, instability)."""
    i = np.arange(2, len(g2) - 2)
    d4 = (-g2[i+2] + 8*g2[i+1] - 8*g2[i-1] + g2[i-2])/(12*h)   # O(h^4)
    d2 = (g2[i+1] - g2[i-1])/(2*h)                             # O(h^2)
    instab = np.abs(d4 - d2)/np.maximum(np.abs(d4), 1e-30)
    return i, d4, instab

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
       # recomputed, and cross-checked against param_final_{MODE}.csv
       "L_A","L_B","B","t_star","rho_A","rho_B",
       # the four terms of R at each anchor
       "T1_A","T1_B","T2_A","T2_B","T3_A","T3_B","T4",
       # the constants themselves, scale factors divided back out
       "gnorm_A","gnorm_B","m2_dir","m3_dir","t_arg_m2","t_arg_m3",
       # the verdict
       "R_hat_A","R_hat_B","R_hat","floor","tight_ratio",
       # which term carries R_hat, and how far the numerics can be trusted
       "dom_term","share_T1","share_T2","share_T3","share_T4",
       "fd_instab","status"]

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

# ==================================================================== BENCH
def project(timings, n_cells):
    """Print what the full grid will cost, from one timed pair per width.

    `timings` maps width -> seconds for one pair.  Cost is dominated by the
    widest one or two grid points, so a projection that has actually timed
    those is worth far more than a FLOP count: it carries this GPU's real
    throughput, the micro-batch sizes, and the checkpoint-loading overhead."""
    if not timings:
        log("[bench] nothing timed"); return
    log("")
    log(f"[bench] projected cost of the full grid  "
        f"({n_cells} cells = {N_REGIMES} regimes x {N_ACTS} activations, "
        f"{PAIRS_FULL} pairs per cell per width)")
    log(f"  {'width':>7}  {'s / pair':>9}  {'cell-widths':>11}  {'hours':>7}")
    total = 0.0
    for w in sorted(timings):
        per_cell = timings[w]*PAIRS_FULL
        hrs = per_cell*n_cells/3600.0
        total += hrs
        log(f"  {w:>7}  {timings[w]:9.1f}  {n_cells:>11}  {hrs:7.2f}")
    log(f"  {'TOTAL':>7}  {'':>9}  {'':>11}  {total:7.2f}  hours for MODE={MODE}")
    log("  knobs: NGRID_PHI=17 roughly halves the dominant term; PAIRS=3 cuts")
    log("         every cell to 3 of 10 seed pairs; both are ~linear in cost.")


# ==================================================================== MAIN
PAIRS_FULL = int(os.environ.get("PAIRS_FULL", "10"))   # what a real run uses


def run():
    out = os.path.join(OUT_DIR,
                       f"tightness_{MODE}{'_bench' if BENCH else ''}.csv")
    X, Y = load_data_xy()
    Xe = X[:EVAL_N].to(DEVICE); Ye = Y[:EVAL_N].to(DEVICE)
    Xf = X[:FISHER_N].to(DEVICE); Yf = Y[:FISHER_N].to(DEVICE)
    ts_f = np.linspace(0, 1, TGRID_FINE)

    # phi'' on a uniform grid over the whole segment; phi''' then lives on the
    # interior points the 5-point stencil can reach, t in [2h, 1-2h].  U is an
    # open neighbourhood of the CLOSED segment in the theorem, so dropping the
    # four outermost points only makes the surrogate smaller, which is the safe
    # direction: R_hat stays a lower bound on R.
    ts_phi = np.linspace(0.0, 1.0, NGRID_PHI)
    h_phi = float(ts_phi[1] - ts_phi[0])
    ts_m2 = np.linspace(0.0, 1.0, NGRID_M2)

    log(f"MODE={MODE} DEVICE={DEVICE} pairs={PAIRS} "
        f"ngrid_phi={NGRID_PHI} (h={h_phi:.4g}) ngrid_m2={NGRID_M2}"
        + ("  [BENCH: 1 pair per cell, then project]" if BENCH else ""))
    timings = {}
    log(f"widths={WIDTHS} eval_n={EVAL_N} fisher_n={FISHER_N} -> {out}")

    for regime in REGIMES:
        for act in ACTS:
            for w in WIDTHS:
                try:
                    sds = []
                    for s in range(NSEEDS):
                        cp = find_ckpt(regime, act, w, s)
                        if cp is None: continue
                        sd, _, _ = load_sd(cp); sds.append(sd)
                    if len(sds) < 2:
                        log(f"  [{regime}/{act}/w{w}] <2 ckpt -> skip"); continue
                    log(f"=== {regime}/{act}/w{w}  ({len(sds)} nets) ===")
                    ref = build_net(w, act, regime).to(DEVICE).eval()
                    p_ref, b_ref = _pb(ref)
                    ag, gs = perm_spec(build_net(w, act, regime))
                    to_params = lambda sd: {k: sd[k].to(DEVICE) for k in p_ref.keys()}

                    for (i, j) in list(itertools.combinations(range(len(sds)), 2))[:PAIRS]:
                        if RESUME and already_done(out, regime, act, w, i, j):
                            continue
                        t_pair = time.time()
                        try:
                            perms = weight_matching(ag, gs, sds[i], sds[j], iters=8, seed=i*13 + j)
                            pA = to_params(sds[i]); pB = to_params(apply_perm(sds[j], ag, perms))
                            delta = {k: pB[k] - pA[k] for k in pA}; dn = _vnorm(delta)

                            # ---- barrier, endpoint losses, rho* at BOTH anchors ----
                            Ls = []
                            for tt in ts_f:
                                L_, _, _ = loss_acc_rho(ref, _lerp(pA, pB, tt), b_ref, Xe, Ye)
                                Ls.append(L_)
                            Ls = np.array(Ls)
                            k_star = int(np.argmax(Ls)); t_star = float(ts_f[k_star])
                            B = float(Ls[k_star] - 0.5*(Ls[0] + Ls[-1]))
                            L_A, rho_A, _ = loss_acc_rho(ref, pA, b_ref, Xe, Ye)
                            L_B, rho_B, _ = loss_acc_rho(ref, pB, b_ref, Xe, Ye)

                            # ---- T1: endpoint gradient (the reverse-mode part) ----
                            gA = grad_norm(ref, pA, b_ref, Xe, Ye)
                            gB = grad_norm(ref, pB, b_ref, Xe, Ye)
                            T1_A, T1_B = dn*gA, dn*gB

                            # ---- T2: Fisher energy, already halved by flen_at ----
                            T2_A, _ = flen_at(ref, pA, b_ref, Xf, delta)
                            T2_B, _ = flen_at(ref, pB, b_ref, Xf, delta)

                            # ---- the two suprema over the segment ----
                            # d2f_seg already carries ||D||^2 and the phi'' grid
                            # ||D||^3, so T3 and T4 form without touching either.
                            sup2 = 0.0; t2 = float("nan")
                            for tt in ts_m2:
                                v2 = d2f_seg(ref, _lerp(pA, pB, tt), b_ref, Xf, delta)
                                if v2 > sup2: sup2, t2 = v2, float(tt)

                            g2 = phi2_grid(ref, pA, pB, b_ref, Xe, Ye, delta, ts_phi,
                                           micro=MICRO_L)
                            idx, d3, inst = d3_from_grid(g2, h_phi)
                            k3 = int(np.argmax(np.abs(d3)))
                            sup3 = float(abs(d3[k3])); t3 = float(ts_phi[idx[k3]])
                            fdi = float(inst[k3])

                            T3_A = 0.5*rho_A*SQRT_K*sup2
                            T3_B = 0.5*rho_B*SQRT_K*sup2
                            T4   = sup3/6.0
                            # the constants themselves, for the prose
                            m2_dir = SQRT_K*sup2/max(dn**2, 1e-30)
                            m3_dir = sup3/max(dn**3, 1e-30)

                            # ---- the verdict ----
                            R_hat_A = T1_A + T2_A + T3_A + T4
                            R_hat_B = T1_B + T2_B + T3_B + T4
                            floor   = 0.5*abs(L_A - L_B)
                            R_hat   = min(R_hat_A, R_hat_B) + floor
                            ratio   = B/max(R_hat, 1e-30)

                            # which term carries the anchor that min{} selected
                            terms = ((T1_A, T2_A, T3_A, T4) if R_hat_A <= R_hat_B
                                     else (T1_B, T2_B, T3_B, T4))
                            tot = max(sum(terms), 1e-30)
                            sh = [t_/tot for t_ in terms]
                            dom = ["T1","T2","T3","T4"][int(np.argmax(sh))]

                            row = dict(
                                mode=MODE, regime=regime, act=act, width=w, seedA=i, seedB=j,
                                dnorm=f"{dn:.6e}",
                                L_A=f"{L_A:.6e}", L_B=f"{L_B:.6e}", B=f"{B:.6e}",
                                t_star=f"{t_star:.4f}",
                                rho_A=f"{rho_A:.6e}", rho_B=f"{rho_B:.6e}",
                                T1_A=f"{T1_A:.6e}", T1_B=f"{T1_B:.6e}",
                                T2_A=f"{T2_A:.6e}", T2_B=f"{T2_B:.6e}",
                                T3_A=f"{T3_A:.6e}", T3_B=f"{T3_B:.6e}", T4=f"{T4:.6e}",
                                gnorm_A=f"{gA:.6e}", gnorm_B=f"{gB:.6e}",
                                m2_dir=f"{m2_dir:.6e}", m3_dir=f"{m3_dir:.6e}",
                                t_arg_m2=f"{t2:.4f}", t_arg_m3=f"{t3:.4f}",
                                R_hat_A=f"{R_hat_A:.6e}", R_hat_B=f"{R_hat_B:.6e}",
                                R_hat=f"{R_hat:.6e}", floor=f"{floor:.6e}",
                                tight_ratio=f"{ratio:.6e}",
                                dom_term=dom,
                                share_T1=f"{sh[0]:.4f}", share_T2=f"{sh[1]:.4f}",
                                share_T3=f"{sh[2]:.4f}", share_T4=f"{sh[3]:.4f}",
                                fd_instab=f"{fdi:.3e}", status="ok")
                            write_row(out, row)
                            dt = time.time() - t_pair
                            timings.setdefault(w, []).append(dt)
                            log(f"  [{i}-{j}] B={B:.4f} R_hat={R_hat:.4g} "
                                f"B/R_hat={ratio:.3e} dom={dom} fd={fdi:.1e} "
                                f"({dt:.1f}s)")
                        except Exception as e:
                            traceback.print_exc()
                            write_row(out, dict(mode=MODE, regime=regime, act=act, width=w,
                                                seedA=i, seedB=j,
                                                status="pair-error:" + repr(e)[:40]))
                    del sds
                    if DEVICE == "cuda": torch.cuda.empty_cache()
                except Exception as e:
                    traceback.print_exc()
                    write_row(out, dict(mode=MODE, regime=regime, act=act, width=w,
                                        status="cell-error:" + repr(e)[:40]))
    if BENCH:
        project({w: float(np.median(v)) for w, v in timings.items()}, N_CELLS)
    elif timings:
        done = sum(len(v) for v in timings.values())
        log(f"  timed {done} pairs, median "
            f"{np.median([t for v in timings.values() for t in v]):.1f}s each")
    log("DONE", MODE)
    return out


def aggregate(path):
    import pandas as pd
    d = pd.read_csv(path); d = d[d.status.astype(str) == "ok"].copy()
    num = ["dnorm","L_A","L_B","B","t_star","rho_A","rho_B",
           "T1_A","T1_B","T2_A","T2_B","T3_A","T3_B","T4",
           "gnorm_A","gnorm_B","m2_dir","m3_dir",
           "R_hat_A","R_hat_B","R_hat","floor","tight_ratio",
           "share_T1","share_T2","share_T3","share_T4","fd_instab","width"]
    for c in num:
        if c in d: d[c] = pd.to_numeric(d[c], errors="coerce")
    agg = {c: (c, "median") for c in num if c in d and c != "width"}
    g = (d.groupby(["mode","regime","act","width"])
           .agg(n_pairs=("B","size"), **agg).reset_index())
    fin = os.path.join(OUT_DIR, f"tightness_{MODE}_cell.csv"); g.to_csv(fin, index=False)
    log(f"-> {fin}  ({len(g)} cells)")

    log("\n  tightness ratio B / R_hat  (median; R_hat <= R, so this bounds B/R from ABOVE):")
    log(d.groupby("regime")["tight_ratio"].median().to_string())
    log("\n  which term carries R_hat:")
    log(d.groupby(["regime","dom_term"]).size().to_string())
    log("\n  term shares by regime (median):")
    log(d.groupby("regime")[["share_T1","share_T2","share_T3","share_T4"]]
          .median().round(3).to_string())
    bad = float((d["fd_instab"] > 0.5).mean())
    log(f"\n  rows whose phi''' disagrees by >50% between the two steps: {100*bad:.1f}%")
    over = float((d["tight_ratio"] > 1).mean())
    log(f"  rows with B > R_hat (surrogate too small to decide, NOT a violation): "
        f"{100*over:.1f}%")
    return fin


def selfcheck(path):
    """The recomputed columns against the released round.  B, t*, L_A, L_B,
    ||D|| and rho_A should reproduce; a drift means the checkpoints or the
    alignment differ from the ones the paper reports, not that R_hat is wrong."""
    import pandas as pd
    ref = _first([f"param_final_{MODE}.csv", f"**/param_final_{MODE}.csv",
                  f"/kaggle/input/**/param_final_{MODE}.csv"])
    if not ref:
        log("[selfcheck] param_final not found -> skipped"); return
    a = pd.read_csv(path); a = a[a.status.astype(str) == "ok"]
    b = pd.read_csv(ref);  b = b[b.status.astype(str).str.startswith("ok")]
    key = ["regime","act","width","seedA","seedB"]
    m = a.merge(b, on=key, suffixes=("", "_ref"))
    if not len(m):
        log("[selfcheck] no overlapping pairs -> skipped"); return
    log(f"[selfcheck] {len(m)} pairs against {ref}")
    for c in ["B","L_A","L_B","dnorm","rho_A","t_star"]:
        if f"{c}_ref" not in m: continue
        x = pd.to_numeric(m[c], errors="coerce"); y = pd.to_numeric(m[f"{c}_ref"], errors="coerce")
        rel = ((x - y).abs()/y.abs().clip(lower=1e-12)).median()
        log(f"    {c:7s} median relative drift {rel:.2e}")


if __name__ == "__main__":
    p = run()
    if BENCH:
        log("[bench] timing only -- no aggregate, no selfcheck.  "
            f"Rows went to {p}, which a real run ignores.")
    else:
        aggregate(p)
        if SELFCHECK: selfcheck(p)
