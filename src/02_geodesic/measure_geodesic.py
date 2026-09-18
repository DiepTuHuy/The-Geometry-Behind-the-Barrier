#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure the geodesic-linear deviation for all three architectures.

Merges mlp_/cnn_/ts_measure_geodesic.py into one file. The three were nearly
line-for-line identical (mlp and ts differed only in MODE), so this keeps the
common body, applies the CNN memory fix to all three modes, and moves what used
to be hardcoded into environment variables.

The question: does the Fisher geodesic stay close to the linear interpolation?

        D_rel = sup_t || gamma_g(t) - gamma_lin(t) ||  /  ||Delta||

to first order, through the Green representation of the Poincare-Sobolev lemma
and the Christoffel symbols of G_F = F + lam I.

Since gamma_lin'' = 0, the geodesic residual of the straight line is
Gamma(Delta, Delta), and with xi = gamma_g - gamma_lin,

  xi''    ~= -Gamma_{gamma_lin}(Delta, Delta)           (first order)
  xi(t)    = int_0^1 G(t,s) Gamma(Delta,Delta)(gamma_lin(s)) ds
  G(t,s)   = s(1-t) for s <= t,  t(1-s) for s >= t

The Levi-Civita connection of G_F at u = v = Delta is

  Gamma(Delta,Delta) = 1/2 G_F^{-1} [ 2 (d_Delta F) Delta - m ],
     m_l = Delta^T (d_l F) Delta = grad_w [ Delta^T F(w) Delta ]_l

Grid: 3 parameterisations x 3 architectures x 5 activations.
  regime : ntk / sp / mup
  arch   : mlp (MNIST) / cnn (FashionMNIST) / ts (teacher-student, synthetic)
  act    : relu, gelu, tanh, swish, softplus

  relu is not C^3, so a finite-difference Christoffel symbol is not valid for it
  (F and the Hessian themselves remain well defined; see the numerical
  methodology in the appendix). It is measured when explicitly asked for, but
  marked smooth=0 and status="ok:nonsmooth" so downstream analysis can exclude
  it. GEO_ACTS defaults to the four smooth activations.

Reading the output:

  * Absolute magnitudes carry the CG damping lambda, so only width exponents and
    scale-invariant ratios are meaningful. The script reports
    dev_rel = sup_t ||xi|| / ||Delta||.
  * D_rel falls only if the flattening of the metric outpaces the growth of
    ||Delta||. That is a race between two quantities, not an automatic
    consequence of metric flattening.
  * cg_resid > CG_TOL means CG hit its iteration ceiling instead of converging,
    so the number at that cell is not usable. Column `cg_iters` records how many
    iterations were spent.

Usage
-----
  python measure_geodesic.py                      # mlp, full grid, all GPUs
  GEO_MODE=cnn python measure_geodesic.py
  GEO_MODE=all python measure_geodesic.py         # mlp, then cnn, then ts
  GEO_GPUS=0 python measure_geodesic.py           # force a single GPU

  # one shard, to run several processes in parallel
  GEO_MODE=mlp GEO_SHARD=2 python measure_geodesic.py

  # targeted re-measurement: sp only, last two widths, deeper CG
  GEO_MODE=mlp GEO_REGIMES=sp GEO_WIDTHS=2048,4096 GEO_CG_ITERS=3000 \
      python measure_geodesic.py

  # end-to-end smoke test (minutes; trains small nets itself)
  GEO_SMOKE=1 python measure_geodesic.py

Environment variables
---------------------
  GEO_MODE       mlp | cnn | ts | all               (default mlp)
  GEO_SHARD      0..5, one (regime, act group)      (default: all)
  GEO_REGIMES    e.g. "sp" or "ntk,mup"             (default: all)
  GEO_ACTS       e.g. "gelu,tanh"                   (default: 4 smooth)
  GEO_WIDTHS     e.g. "2048,4096"                   (default: the mode grid)
  GEO_PAIRS      seed pairs per cell                (default 10)
  GEO_CG_ITERS   CG iteration ceiling               (default 300)
  GEO_CG_TOL     CG convergence tolerance           (default 1e-6)
  GEO_MICRO      fisher_vp micro-batch size         (default: chosen from P)
  GEO_RESUME     progress file, or "none"           (default: best available)
  GEO_GPUS       "0,1" to pick, "0" for one GPU     (default: all GPUs)
  GEO_LAMSWEEP   =1 sweeps lam_rel over [1e-1,1e-2,1e-3]
  GEO_SMOKE      =1 trains small checkpoints first
  GEO_ALLOW_RELU =1 adds relu to the default GEO_ACTS (see the caveat above)
  COMBINED       path to combined_p{mode}*.csv, to merge in ||dF||_op

Performance
-----------
* GEO_MICRO: fisher_vp accumulates over micro-batches to bound memory. The
  earlier version hardcoded 64 at every width, which is pure waste at small
  width: each slice is ~0.02 GFLOP but still pays a full torch.func trace and
  kernel launch, which is then ~99.9% of the time. micro is now chosen so that
  micro*P is roughly constant, i.e. few large slices at small width and many
  small ones at large width. The result is unchanged up to floating-point
  summation order, since fisher_vp sums everything and divides by B.
* Multi-GPU is on by default. The script counts the GPUs and, with more than
  one, splits cells (regime, act, width) across them, one process each, writing
  separate CSVs that are merged at the end. A two-GPU machine is used fully
  without extra configuration; GEO_GPUS=0 forces one.
  DataParallel is deliberately not used: fisher_vp is a torch.func loop, and
  splitting the batch across GPUs would ship a P-dimensional vector on every
  call (80MB at width 4096), costing more than the arithmetic saved. Splitting
  by cell scales nearly linearly and touches no arithmetic, so each cell matches
  a single-GPU run exactly.

Output: param_geo_{mode}.csv (one row per seed pair per lam_rel), resumable,
        and {mode}_final.csv aggregated per cell.
"""
import os, sys, time, math, glob, itertools, traceback
try: sys.stdout.reconfigure(line_buffering=True); sys.stderr.reconfigure(line_buffering=True)
except Exception: pass
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.func import functional_call, jvp as _fjvp, vjp as _fvjp, jacrev as _jacrev, grad as _grad
def _tv():   # torchvision is needed for mlp/cnn only; ts is synthetic
    import torchvision; return torchvision

# ==================================================================== CONFIG
def _env(name, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v
def _env_list(name, default=None, cast=str):
    v = _env(name)
    return default if v is None else [cast(s.strip()) for s in v.split(",") if s.strip()]

ALL_MODES  = ["mlp", "cnn", "ts"]
MODES      = ALL_MODES if _env("GEO_MODE", "mlp").lower() == "all" else [_env("GEO_MODE", "mlp").lower()]
for _m in MODES:
    if _m not in ALL_MODES: raise SystemExit(f"invalid GEO_MODE: {_m} (choose {ALL_MODES} or 'all')")

RESUME     = True
SMOKE      = _env("GEO_SMOKE", "0") == "1"
ONLY_SHARD = _env("GEO_SHARD")
ALLOW_RELU = _env("GEO_ALLOW_RELU", "0") == "1"

SMOOTH_ACTS = ["gelu", "tanh", "swish", "softplus"]     # C^3, so a finite-difference Christoffel symbol is valid
ALL_ACTS    = ["relu"] + SMOOTH_ACTS
DEFAULT_ACTS = ALL_ACTS if ALLOW_RELU else SMOOTH_ACTS

# Width grid per mode. For cnn the stored value is the channel multiplier wm;
# channels are c=[16wm,32wm,64wm], so the width the paper reports (the widest
# layer) is 64*wm.
MODE_WIDTHS = {"mlp": [64, 128, 256, 512, 1024, 2048, 4096],
               "cnn": [1, 2, 4, 8],
               "ts":  [64, 128, 256, 512, 1024, 2048, 4096]}
MODE_TAG    = {"mlp": "pmlp_v2", "cnn": "pcnn_v2", "ts": "pts_v2"}
MODE_DIN_K  = {"mlp": (784, 10), "cnn": (None, 10), "ts": (64, 10)}

if SMOKE:
    GEO_ACTS  = _env_list("GEO_ACTS", ["gelu", "tanh"])
    GEO_PAIRS = int(_env("GEO_PAIRS", 1)); GEO_TGRID = 5
    GEO_BATCH = 128; NSEEDS = 2
    CG_ITERS  = int(_env("GEO_CG_ITERS", 60)); POWER_ITERS = 12
    SMOKE_WIDTHS = {"mlp": [16, 32], "cnn": [1, 2], "ts": [16, 32]}
else:
    GEO_ACTS  = _env_list("GEO_ACTS", DEFAULT_ACTS)
    GEO_PAIRS = int(_env("GEO_PAIRS", 10))   # combinations(5,2), the same pairs as the barrier
    GEO_TGRID = 9                            # Green quadrature nodes; the integrand is smooth
    GEO_BATCH = 2048                         # same batch dF was measured on
    NSEEDS = 5
    CG_ITERS  = int(_env("GEO_CG_ITERS", 300)); POWER_ITERS = 20

# Micro-batch size: fixed if the user sets it, otherwise chosen from the
# activation memory budget.
#
# The total FLOP count of one fisher_vp does not depend on micro -- it always
# covers the whole GEO_BATCH. Splitting finer only adds overhead passes
# (torch.func trace + kernel launch), measured at ~4.3ms each. So larger micro
# is faster, and the only constraint is the activation memory of jvp+vjp. Hence
# estimate bytes per sample and divide the budget, rather than guessing from the
# parameter count -- which is badly wrong for the CNN: few parameters, very
# large activations.
GEO_MICRO_ENV = _env("GEO_MICRO")
ACT_BUDGET    = float(_env("GEO_ACT_BUDGET", 2.0e9))   # ~2GB for activations
ACT_COPIES    = 6      # primal + tangent (jvp) + the vjp tape; a rough estimate

def act_units_per_sample(width):
    """Activation elements per sample; a rough estimate, used to split the budget."""
    if MODE == "cnn":   # FashionMNIST 28x28, c=[16wm,32wm,64wm], pooled after c1 and c2
        wm = width
        return 28 * 28 * 16 * wm + 14 * 14 * 32 * wm + 7 * 7 * 64 * wm
    return (DIN or 0) + 2 * width + (K or 0)

def pick_micro(width, nparams=None):
    """Largest micro the activation budget allows, rounded down to a power of two."""
    if GEO_MICRO_ENV: return max(1, min(int(GEO_MICRO_ENV), GEO_BATCH))
    per_sample = max(act_units_per_sample(width) * ACT_COPIES * 4, 1)   # 4 bytes per float
    m = int(ACT_BUDGET // per_sample)
    m = min(max(m, 8), GEO_BATCH)
    return 1 << int(math.floor(math.log2(m)))

FD_EPS  = 3e-3; FD_RICH = True; LAM_REL = 1e-2
LAM_SWEEP = [1e-1, 1e-2, 1e-3] if _env("GEO_LAMSWEEP", "0") == "1" else None
CG_TOL  = float(_env("GEO_CG_TOL", 1e-6))
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_ROOTS = [".", "/kaggle/input", "/content", "/content/drive/MyDrive"]
OUT_DIR    = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

SHARD_PLAN = {0: ("ntk", ["relu", "gelu", "tanh"]), 1: ("ntk", ["swish", "softplus"]),
              2: ("sp",  ["relu", "gelu", "tanh"]), 3: ("sp",  ["swish", "softplus"]),
              4: ("mup", ["relu", "gelu", "tanh"]), 5: ("mup", ["swish", "softplus"])}
ONLY_REGIMES = _env_list("GEO_REGIMES")
SHARD_ID = ""   # unused here; kept so the CSV columns match the training runs

# --- mode-dependent state, set by set_mode() ---
MODE = None; RUN_TAG = None; DIN = None; K = None; GEO_WIDTHS = None; COMBINED = None
_CACHE = {}

def _first(pats):
    for p in pats:
        if not p: continue
        h = sorted(glob.glob(p, recursive=True))
        if h: return h[0]
    return None

def set_mode(mode):
    """Set the current mode and every constant that depends on it. Call before running."""
    global MODE, RUN_TAG, DIN, K, GEO_WIDTHS, COMBINED, _CACHE, _CKPT_INDEX
    MODE = mode; RUN_TAG = MODE_TAG[mode]; DIN, K = MODE_DIN_K[mode]
    _CKPT_INDEX = None      # GEO_MODE=all: each mode has its own ckpt_dir(), so rescan
    GEO_WIDTHS = _env_list("GEO_WIDTHS", (SMOKE_WIDTHS if SMOKE else MODE_WIDTHS)[mode], cast=int)
    COMBINED = _first([_env("COMBINED"),
        f"combined_p{mode}*dFfixed*.csv", f"combined_p{mode}*.csv",
        f"/kaggle/input/**/combined_p{mode}*dFfixed*.csv", f"/kaggle/input/**/combined_p{mode}*.csv",
        f"/content/drive/MyDrive/**/combined_p{mode}*.csv"])
    _CACHE = {}

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def set_seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ==================================================================== MODEL
BASE = 64
def make_act(n): return {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh,
                         "swish": nn.SiLU, "softplus": nn.Softplus}[n]()
def param_cfg(regime, fin, fout, kind):   # (init_std, fwd_mult, lr_scale) -- coord-check v6
    ss = math.sqrt(fin)
    if regime == "sp":  return (1.0 / ss, 1.0, 1.0)
    if regime == "ntk": return (1.0, 1.0 / ss, 1.0)
    if regime == "mup":
        if kind == "input":  return (1.0 / ss, 1.0, (fout / BASE) ** 1.0)
        if kind == "hidden": return (1.0 / ss, 1.0, (fin / BASE) ** 0.7)
        return (1.0 / ss, (BASE / fin) ** 0.5, (fin / BASE) ** (-0.5))
    raise ValueError(regime)

class ScaledLinear(nn.Module):
    def __init__(self, fin, fout, regime, kind):
        super().__init__(); istd, self.fmul, self.lr_scale = param_cfg(regime, fin, fout, kind)
        self.weight = nn.Parameter(torch.randn(fout, fin) * istd); self.bias = nn.Parameter(torch.zeros(fout))
    def forward(self, x): return self.fmul * F.linear(x, self.weight) + self.bias

class ScaledConv(nn.Module):
    def __init__(self, cin, cout, k, st, pad, regime, kind):
        super().__init__(); fin = cin * k * k
        istd, self.fmul, self.lr_scale = param_cfg(regime, fin, cout, kind)
        self.weight = nn.Parameter(torch.randn(cout, cin, k, k) * istd); self.st = st; self.pad = pad
    def forward(self, x): return self.fmul * F.conv2d(x, self.weight, None, self.st, self.pad)

def _gn(c): return nn.GroupNorm(1, c)

class NetMLP(nn.Module):     # mlp & ts
    def __init__(self, width, act, regime="ntk", din=None, k=None):
        super().__init__(); din = DIN if din is None else din; k = K if k is None else k
        self.fc1 = ScaledLinear(din, width, regime, "input")
        self.fc2 = ScaledLinear(width, width, regime, "hidden")
        self.fc3 = ScaledLinear(width, k, regime, "output")
        self.a1 = make_act(act); self.a2 = make_act(act); self.width = width; self.regime = regime
    def forward(self, x): return self.fc3(self.a2(self.fc2(self.a1(self.fc1(x)))))

class NetCNN(nn.Module):     # cnn (ScaledConv + GroupNorm)
    def __init__(self, wm, act, regime="ntk", in_ch=1, k=None):
        super().__init__(); k = K if k is None else k; c = [16 * wm, 32 * wm, 64 * wm]
        self.c1 = ScaledConv(in_ch, c[0], 3, 1, 1, regime, "input"); self.n1 = _gn(c[0]); self.a1 = make_act(act)
        self.c2 = ScaledConv(c[0], c[1], 3, 1, 1, regime, "hidden"); self.n2 = _gn(c[1]); self.a2 = make_act(act)
        self.c3 = ScaledConv(c[1], c[2], 3, 1, 1, regime, "hidden"); self.n3 = _gn(c[2]); self.a3 = make_act(act)
        self.pool = nn.MaxPool2d(2); self.fc = ScaledLinear(c[2], k, regime, "output")
        self.width = wm; self.regime = regime
    def forward(self, x):
        h1 = self.pool(self.a1(self.n1(self.c1(x))))
        h2 = self.pool(self.a2(self.n2(self.c2(h1))))
        h3 = self.a3(self.n3(self.c3(h2)))
        return self.fc(F.adaptive_avg_pool2d(h3, 1).flatten(1))
    def opt_groups(self, base_lr):
        g = [{"params": [m.weight] + ([m.bias] if hasattr(m, "bias") else []), "lr": base_lr * m.lr_scale}
             for m in [self.c1, self.c2, self.c3, self.fc]]
        g.append({"params": [p for n in [self.n1, self.n2, self.n3] for p in n.parameters()], "lr": base_lr})
        return g

def build_net(width, act, regime):
    return NetCNN(width, act, regime) if MODE == "cnn" else NetMLP(width, act, regime)

# ---- perm spec ----
def perm_spec(model):
    if MODE == "cnn":
        ag = {"c1.weight": ["g1", None, None, None], "n1.weight": ["g1"], "n1.bias": ["g1"],
              "c2.weight": ["g2", "g1", None, None], "n2.weight": ["g2"], "n2.bias": ["g2"],
              "c3.weight": ["g3", "g2", None, None], "n3.weight": ["g3"], "n3.bias": ["g3"],
              "fc.weight": [None, "g3"], "fc.bias": [None]}
    else:
        ag = {"fc1.weight": ["h1", None], "fc1.bias": ["h1"],
              "fc2.weight": ["h2", "h1"], "fc2.bias": ["h2"],
              "fc3.weight": [None, "h2"], "fc3.bias": [None]}
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
        else: out[n] = t.clone()
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
    for it in range(iters):
        moved = 0
        for g in [groups[i] for i in rng.permutation(len(groups))]:
            n = gs[g]; S = torch.zeros(n, n, dtype=torch.float64)
            for (name, axis) in g2pa[g]:
                A = sdA[name].double(); B = _perm_except(sdB[name].double(), ag[name], perms, axis)
                S += torch.movedim(A, axis, 0).reshape(n, -1) @ torch.movedim(B, axis, 0).reshape(n, -1).T
            ci = linear_sum_assignment(-S.numpy())[1]; new = torch.as_tensor(ci, dtype=torch.long)
            if not torch.equal(new, perms[g]): moved += 1
            perms[g] = new
        if moved == 0: break
    return perms

# ==================================================================== PRIMITIVES
def _pb(m): return ({k: v.detach() for k, v in m.named_parameters()},
                    {k: v.detach() for k, v in m.named_buffers()})
def _call(m, p, b, x): return functional_call(m, {**p, **b}, (x,))

def fisher_vp(m, p, b, x, v, micro):
    B = x.shape[0]; acc = None
    for i in range(0, B, micro):
        xb = x[i:i + micro]
        def f(pp): return _call(m, pp, b, xb)
        logits, Jv = _fjvp(f, (p,), (v,)); pr = torch.softmax(logits, 1)
        s = pr * Jv - pr * (pr * Jv).sum(1, keepdim=True); JTs = _fvjp(f, p)[1](s)[0]
        acc = {k: JTs[k].detach() for k in JTs} if acc is None else {k: acc[k] + JTs[k].detach() for k in acc}
    return {k: acc[k] / B for k in acc}

def _vnorm(a): return float(torch.sqrt(torch.clamp(sum((a[k] * a[k]).sum() for k in a), min=0)))
def _vscale(a, c): return {k: a[k] * c for k in a}
def _vaxpy(a, c, b): return {k: a[k] + c * b[k] for k in a}
def _vdot(a, b): return float(sum((a[k] * b[k]).sum() for k in a))

def dFz(m, p, b, x, z, v, eps, micro, rich):
    def cd(e):
        Fp = fisher_vp(m, _vaxpy(p, e, z), b, x, v, micro)
        Fm = fisher_vp(m, _vaxpy(p, -e, z), b, x, v, micro)
        return {k: (Fp[k] - Fm[k]) / (2 * e) for k in p}
    if rich:
        d1, d2 = cd(eps), cd(eps / 2); return {k: (4 * d2[k] - d1[k]) / 3 for k in d1}
    return cd(eps)

# ==================================================================== GEOMETRY
def _quad_sum(m, p, b, xb, delta):     # sum of u^T S u over a chunk, not divided by B
    def f(pp): return _call(m, pp, b, xb)
    logits, u = _fjvp(f, (p,), (delta,)); pr = torch.softmax(logits, 1)
    Su = pr * u - pr * (pr * u).sum(1, keepdim=True); return (u * Su).sum()

def grad_quad(m, p, b, x, delta, micro):
    """m_l = Delta^T (d_l F) Delta = grad_w <Delta, F Delta>.

    Accumulating the gradient per micro-batch bounds memory by `micro` instead
    of holding a graph over the whole batch. This fix was CNN-only; it is applied
    to all three modes here."""
    B = x.shape[0]; acc = None
    for i in range(0, B, micro):
        gi = _grad(lambda pp: _quad_sum(m, pp, b, x[i:i + micro], delta))(p)
        acc = {k: gi[k].detach() for k in gi} if acc is None else {k: acc[k] + gi[k].detach() for k in acc}
    return {k: acc[k] / B for k in acc}

def gf_vp(m, p, b, x, v, lam, micro):
    Fv = fisher_vp(m, p, b, x, v, micro); return {k: Fv[k] + lam * v[k] for k in v}

def lam_max(m, p, b, x, micro, iters, seed=0):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    u = {k: torch.randn(v.shape, generator=gen, device=v.device, dtype=v.dtype) for k, v in p.items()}
    u = _vscale(u, 1.0 / max(_vnorm(u), 1e-30)); lam = 0.0
    for _ in range(iters):
        Au = fisher_vp(m, p, b, x, u, micro); lam = _vnorm(Au)
        if lam < 1e-30: break
        u = _vscale(Au, 1.0 / lam)
    return lam

def cg_solve(m, p, b, x, rhs, lam, micro, x0=None, iters=80, tol=1e-6):
    """CG for G_F x = rhs. Returns (solution, relative residual, iterations used).

    `used == iters` means the ceiling was hit rather than convergence, so the
    number measured there is not usable."""
    xk = {k: (torch.zeros_like(v) if x0 is None else x0[k].clone()) for k, v in rhs.items()}
    Ax = gf_vp(m, p, b, x, xk, lam, micro) if x0 is not None else {k: torch.zeros_like(v) for k, v in rhs.items()}
    r = {k: rhs[k] - Ax[k] for k in rhs}; pdir = {k: r[k].clone() for k in r}
    rs = _vdot(r, r); r0 = max(rs, 1e-300); used = 0
    for _ in range(iters):
        used += 1
        Ap = gf_vp(m, p, b, x, pdir, lam, micro); a = rs / max(_vdot(pdir, Ap), 1e-300)
        xk = {k: xk[k] + a * pdir[k] for k in xk}; r = {k: r[k] - a * Ap[k] for k in r}
        rs2 = _vdot(r, r)
        if rs2 <= tol * tol * r0: break
        beta = rs2 / max(rs, 1e-300); pdir = {k: r[k] + beta * pdir[k] for k in pdir}; rs = rs2
    return xk, math.sqrt(_vdot(r, r) / r0), used

def christoffel_dd(m, p, b, x, delta, lam, micro, x0=None):
    """Gamma(Delta,Delta). Tra ve (Gamma, cg_resid, fd_instab, cg_iters, ||rhs||)."""
    dn = _vnorm(delta); dhat = _vscale(delta, 1.0 / dn)
    # (d_Delta F) Delta at two eps levels, to report finite-difference stability.
    d1 = dFz(m, p, b, x, dhat, dhat, FD_EPS, micro, False)          # central diff @ eps
    d2 = dFz(m, p, b, x, dhat, dhat, FD_EPS / 2, micro, False)      # @ eps/2
    t1r = {k: (4 * d2[k] - d1[k]) / 3 for k in d1} if FD_RICH else d1
    fd_instab = _vnorm({k: d1[k] - d2[k] for k in d1}) / max(_vnorm(t1r), 1e-30)
    t1 = {k: t1r[k] * dn * dn for k in t1r}
    mvec = grad_quad(m, p, b, x, delta, micro)
    rhs = {k: 2 * t1[k] - mvec[k] for k in t1}
    rhs_norm = _vnorm(rhs)
    sol, resid, used = cg_solve(m, p, b, x, rhs, lam, micro, x0=x0, iters=CG_ITERS, tol=CG_TOL)
    return _vscale(sol, 0.5), resid, fd_instab, used, rhs_norm

def _is_oom(e):
    """OOM must reach the per-cell retry loop, not be swallowed by except Exception."""
    if isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())): return True
    return "out of memory" in str(e).lower() or "CUDA_ERROR_OUT_OF_MEMORY" in str(e)

def green_matrix(ts):
    n = len(ts); G = torch.zeros(n, n, dtype=torch.float64)
    for i, t in enumerate(ts):
        for j, s in enumerate(ts):
            G[i, j] = s * (1 - t) if s <= t else t * (1 - s)
    return G

# ==================================================================== DATA
def load_data():
    if "d" in _CACHE: return _CACHE["d"]
    if MODE == "mlp":
        ds = _tv().datasets.MNIST("./data", train=True, download=True)
        X = ((ds.data.float() / 255.0) - 0.1307) / 0.3081; X = X.reshape(-1, 784)
    elif MODE == "cnn":
        ds = _tv().datasets.FashionMNIST("./data", train=True, download=True)
        X = ((ds.data.float() / 255.0) - 0.2860) / 0.3530; X = X.unsqueeze(1)
    else:   # ts needs inputs only: F is an expectation over x
        g = torch.Generator().manual_seed(1); X = torch.randn(20000, DIN, generator=g)
    _CACHE["d"] = X; return X

# ==================================================================== CKPT IO
def ckpt_dir():
    d = os.path.join(OUT_DIR, f"ckpt_{RUN_TAG}"); os.makedirs(d, exist_ok=True); return d

_CKPT_INDEX = None
def _ckpt_index(force=False):
    """{basename: [paths]} for every *.pt under CKPT_ROOTS, scanned once.

    The earlier version re-globbed recursively for every seed of every cell,
    which on an input mount holding thousands of files dominated startup. One
    scan, then O(1) lookups."""
    global _CKPT_INDEX
    if _CKPT_INDEX is not None and not force: return _CKPT_INDEX
    idx = {}
    for root in [ckpt_dir()] + CKPT_ROOTS:
        if not os.path.isdir(root): continue
        for p in glob.glob(os.path.join(root, "**", "*.pt"), recursive=True):
            idx.setdefault(os.path.basename(p), []).append(p)
    _CKPT_INDEX = idx
    return idx

def _sd_matches(sd, w):
    """Architecture gate: is this state dict really MODE at width w?

    Needed because mlp and ts share checkpoint filenames -- both sweep widths
    64..4096 -- so a file found by name may belong to the other architecture.
    Verified by shape, not by directory name."""
    try:
        if MODE == "cnn":
            return "c1.weight" in sd and int(sd["c1.weight"].shape[0]) == 16 * int(w)
        return ("fc1.weight" in sd
                and int(sd["fc1.weight"].shape[1]) == int(DIN)
                and int(sd["fc1.weight"].shape[0]) == int(w))
    except Exception:
        return False

_LOOSE_WARNED = set()
def find_ckpt(regime, act, w, s):
    """Path to a checkpoint, or None.

    Prefers a file inside a directory named `ckpt_{RUN_TAG}`, which is what the
    training scripts produce. A dataset uploaded as loose .pt files lands them
    directly under the mount with no such directory, and the earlier version
    then reported "no checkpoint" while the file sat right there. Hence the
    fallback: look up by filename, then verify by shape before accepting."""
    name = f"{regime}_{act}_w{w}_s{s}.pt"
    cands = _ckpt_index().get(name, [])
    if not cands: return None
    tagged = [p for p in cands
              if os.path.basename(os.path.dirname(p)) == f"ckpt_{RUN_TAG}"]
    if tagged: return tagged[0]
    for p in cands:                       # fallback: accept only on a shape match
        try: sd, _, _ = load_sd(p)
        except Exception: continue
        if _sd_matches(sd, w):
            if MODE not in _LOOSE_WARNED:
                _LOOSE_WARNED.add(MODE)
                log(f"  [ckpt] no 'ckpt_{RUN_TAG}/' directory; accepting files by"
                    f" name + shape. Example: {p}")
            return p
    return None
def load_sd(path):
    try: d = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: d = torch.load(path, map_location="cpu")
    return d["sd"], d.get("acc"), d.get("dF")

# ==================================================================== CSV
# `rhs_norm`, `cg_iters` and `smooth` were added immediately before `status`.
# fig_data.py selects by act and status, so it reads both old and new files.
CSV = ["kind", "mode", "shard", "regime", "act", "width", "seedA", "seedB",
       "dnorm", "dev_geo", "dev_rel", "gamma_mid",
       "flen_A", "flen_mid", "flen_B", "rq_A", "rq_mid", "rq_B",
       "lam", "lam_rel", "cg_resid", "fd_instab", "rhs_norm", "cg_iters",
       "accA", "accB", "smooth", "status"]

def write_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new: f.write(",".join(CSV) + "\n")
        f.write(",".join(str(row.get(c, "")) for c in CSV) + "\n"); f.flush()

def already_done(paths, regime, act, w, i, j, lam_rel):
    """Is there already an 'ok' row for (cell, pair, lam_rel) in any file?

    Takes a list of paths: under multi-GPU each worker writes its own file, but
    resume must also see the merged progress of earlier runs. Matching is by
    column index rather than endswith('ok'), so 'ok:nonsmooth' counts as done."""
    if isinstance(paths, str): paths = [paths]
    return any(_done_in(p, regime, act, w, i, j, lam_rel) for p in paths)

def _done_in(path, regime, act, w, i, j, lam_rel):
    if not os.path.exists(path): return False
    try:
        with open(path) as f:
            head = f.readline().rstrip("\n").split(",")
            idx = {c: k for k, c in enumerate(head)}
            need = ["regime", "act", "width", "seedA", "seedB", "lam_rel", "status"]
            if any(c not in idx for c in need): return False
            for ln in f:
                p = ln.rstrip("\n").split(",")
                if len(p) <= idx["status"]: continue
                if (p[idx["regime"]] == str(regime) and p[idx["act"]] == str(act)
                        and p[idx["width"]] == str(w) and p[idx["seedA"]] == str(i)
                        and p[idx["seedB"]] == str(j) and p[idx["lam_rel"]] == str(lam_rel)
                        and p[idx["status"]].startswith("ok")):
                    return True
    except Exception:
        return False
    return False

# ==================================================================== SELF-TEST
def self_test():
    """Gamma by vector products equals dense Gamma, and the Green quadrature equals the analytic solution."""
    log("  [self-test] Gamma(dd) vp==dense + Green ...")
    old = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    torch.manual_seed(0); m = NetMLP(6, "tanh", "ntk", din=4, k=3).eval(); x = torch.randn(8, 4)
    p, b = _pb(m); keys = list(p.keys())
    flat = lambda d: torch.cat([d[k].reshape(-1) for k in keys])
    def unflat(v):
        o = {}; i = 0
        for k in keys: n = p[k].numel(); o[k] = v[i:i + n].reshape(p[k].shape); i += n
        return o
    torch.manual_seed(3); dflat = torch.randn(flat(p).numel()); dflat = dflat / dflat.norm() * 3.0
    delta = unflat(dflat); lam = 1e-2
    def Fmat(wv):
        pp = unflat(wv); J = _jacrev(lambda q: _call(m, q, b, x))(pp); B = x.shape[0]
        Jf = torch.cat([J[k].reshape(B, 3, -1) for k in keys], 2)
        pr = torch.softmax(_call(m, pp, b, x), 1); S = torch.diag_embed(pr) - pr.unsqueeze(2) * pr.unsqueeze(1)
        return torch.einsum('bki,bkl,blj->ij', Jf, S, Jf) / B
    w0 = flat(p); dF = _jacrev(Fmat)(w0); Fd = Fmat(w0); P = w0.numel()
    t1 = torch.einsum('ijl,l,j->i', dF, dflat, dflat); mv = torch.einsum('i,ijl,j->l', dflat, dF, dflat)
    g_dense = 0.5 * torch.linalg.solve(Fd + lam * torch.eye(P), 2 * t1 - mv)
    global CG_ITERS, CG_TOL; oi, ot = CG_ITERS, CG_TOL; CG_ITERS, CG_TOL = 300, 1e-12
    g_vp, _, _, _, _ = christoffel_dd(m, p, b, x, delta, lam, micro=8); CG_ITERS, CG_TOL = oi, ot
    rel = float((flat(g_vp) - g_dense).norm() / max(g_dense.norm(), 1e-12))
    assert rel < 1e-6, f"Gamma sai {rel:.2e}"
    ts = np.linspace(0, 1, 21); G = green_matrix(list(ts)); dt = ts[1] - ts[0]
    xi = (G @ torch.full((len(ts),), 0.7, dtype=torch.float64)) * dt
    ge = float((xi - torch.tensor([0.7 * t * (1 - t) / 2 for t in ts])).abs().max())
    assert ge < 1e-10, f"Green sai {ge:.2e}"
    log(f"  [self-test] OK  Gamma rel={rel:.2e}  Green err={ge:.2e}")
    torch.set_default_dtype(old)

def _smoke_train(regime, act, w, seed):
    """Train one small net quickly, to smoke-test the path when no checkpoint exists."""
    set_seed(seed); m = build_net(w, act, regime).to(DEVICE).train()
    if MODE == "cnn":
        ds = _tv().datasets.FashionMNIST("./data", train=True, download=True)
        X = (((ds.data.float() / 255.0) - 0.2860) / 0.3530).unsqueeze(1)[:2000]; Y = ds.targets[:2000]
    elif MODE == "mlp":
        ds = _tv().datasets.MNIST("./data", train=True, download=True)
        X = (((ds.data.float() / 255.0) - 0.1307) / 0.3081).reshape(-1, 784)[:2000]; Y = ds.targets[:2000]
    else:
        g = torch.Generator().manual_seed(1); X = torch.randn(2000, DIN, generator=g)
        set_seed(1234); teach = NetMLP(32, "relu", "sp").eval()
        with torch.no_grad(): Y = teach(X).argmax(1)
    groups = m.opt_groups(0.1) if hasattr(m, "opt_groups") else [{"params": list(m.parameters()), "lr": 0.1}]
    opt = torch.optim.SGD(groups, momentum=0.9)
    for ep in range(3):
        perm = torch.randperm(X.shape[0])
        for i in range(0, X.shape[0], 256):
            idx = perm[i:i + 256]; opt.zero_grad()
            F.cross_entropy(m(X[idx].to(DEVICE)), Y[idx].to(DEVICE)).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval(); sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    torch.save({"sd": sd, "acc": 0.0, "dF": None, "wmove": 0.0},
               os.path.join(ckpt_dir(), f"{regime}_{act}_w{w}_s{seed}.pt"))
    _ckpt_index(force=True)          # the file just written must enter the index

# ==================================================================== MAIN
def _fit_alpha(width, val):
    w = np.asarray(width, float); y = np.asarray(val, float)
    ok = (w > 0) & (y > 0) & np.isfinite(w) & np.isfinite(y)
    if ok.sum() < 2: return (float("nan"), float("nan"))
    lw, ly = np.log(w[ok]), np.log(y[ok]); b, a = np.polyfit(lw, ly, 1); yh = a + b * lw
    r2 = 1 - ((ly - yh) ** 2).sum() / max(((ly - ly.mean()) ** 2).sum(), 1e-30)
    return (round(-b, 4), round(float(r2), 4))

def _cells():
    """The (regime, act) pairs left after filtering by GEO_SHARD / GEO_REGIMES / GEO_ACTS."""
    plan = [SHARD_PLAN[int(ONLY_SHARD)]] if ONLY_SHARD is not None else list(SHARD_PLAN.values())
    out = []
    for regime, acts_all in plan:
        if ONLY_REGIMES and regime not in ONLY_REGIMES: continue
        for act in acts_all:
            if act in GEO_ACTS and (regime, act) not in out: out.append((regime, act))
    return out

def base_csv():
    return os.path.join(OUT_DIR, f"param_geo_{MODE}_SMOKE.csv" if SMOKE else f"param_geo_{MODE}.csv")

def all_tasks():
    """Every cell to measure, as (regime, act, width)."""
    return [(r, a, w) for (r, a) in _cells() for w in GEO_WIDTHS]

def run_geo(tasks=None, out=None):
    base = base_csv()
    out = out or base
    read_paths = [out] if out == base else [out, base]   # a worker also reads the merged file
    lam_rels = LAM_SWEEP if LAM_SWEEP else [LAM_REL]
    X = load_data(); Xgeo = X[:GEO_BATCH].to(DEVICE)
    tasks = all_tasks() if tasks is None else tasks
    ns = [a for (_, a, _) in tasks if a not in SMOOTH_ACTS]
    if ns: log(f"!! WARNING: {sorted(set(ns))} not C^3 -> finite-difference Christoffel "
               f"is not valid; rows are marked smooth=0 / status=ok:nonsmooth")
    log(f"GEO[{MODE}] {len(tasks)} cells  pairs={GEO_PAIRS} batch={GEO_BATCH} "
        f"cg_iters={CG_ITERS} cg_tol={CG_TOL:.1e} lam_rels={lam_rels} -> {out}")
    for (regime, act, w) in tasks:
            smooth = 1 if act in SMOOTH_ACTS else 0
            st_ok = "ok" if smooth else "ok:nonsmooth"
            try:
                # --- load the NSEEDS checkpoints ---
                sds = []; accs = []
                for s in range(NSEEDS):
                    cp = find_ckpt(regime, act, w, s)
                    if cp is None and SMOKE: _smoke_train(regime, act, w, s); cp = find_ckpt(regime, act, w, s)
                    if cp is None: continue
                    sd, acc, _ = load_sd(cp); sds.append(sd); accs.append(acc)
                if len(sds) < 2:
                    log(f"  [{regime}/{act}/w{w}] fewer than 2 checkpoints ({len(sds)}) -> skip")
                    write_row(out, dict(kind="geo", mode=MODE, shard="", regime=regime, act=act,
                                        width=w, smooth=smooth, status="skip:nockpt")); continue
                ag, gs = perm_spec(build_net(w, act, regime))
                ref = build_net(w, act, regime).to(DEVICE).eval(); p_ref, b_ref = _pb(ref)
                nP = sum(v.numel() for v in p_ref.values()); micro = pick_micro(w, nP)
                log(f"=== {regime}/{act}/w{w}  ({len(sds)} nets)  P={nP/1e6:.3f}M  "
                    f"micro={micro} ({GEO_BATCH // micro} slices per fisher_vp) ===")
                def to_params(sd):   # parameter keys only, in named_parameters order
                    return {k: sd[k].to(DEVICE) for k in p_ref.keys()}

                pairs = list(itertools.combinations(range(len(sds)), 2))[:GEO_PAIRS]
                while True:      # on OOM: lower micro and redo this cell; finished pairs are skipped
                    try:
                        for (i, j) in pairs:
                            if RESUME and all(already_done(read_paths, regime, act, w, i, j, lr_) for lr_ in lam_rels):
                                log(f"  [pair {i}-{j}] every lam_rel already measured -> skip"); continue
                            # ---- lambda-independent part ----
                            try:
                                perms = weight_matching(ag, gs, sds[i], sds[j], iters=8, seed=i * 13 + j)
                                sdB = apply_perm(sds[j], ag, perms)
                                pA = to_params(sds[i]); pB = to_params(sdB)
                                delta = {k: (pB[k] - pA[k]) for k in pA}; dn = _vnorm(delta); d2 = max(dn * dn, 1e-30)
                                lmax = lam_max(ref, pA, b_ref, Xgeo, micro, POWER_ITERS, seed=17)
                                # Fisher length 1/2 Delta^T F Delta and Rayleigh quotient, at A / midpoint / B
                                def _flen(pt):
                                    q = _vdot(delta, fisher_vp(ref, pt, b_ref, Xgeo, delta, micro))
                                    return 0.5 * q, q / d2
                                pmid = {k: 0.5 * (pA[k] + pB[k]) for k in pA}
                                flA, rqA = _flen(pA); flM, rqM = _flen(pmid); flB, rqB = _flen(pB)
                            except Exception as e:
                                if _is_oom(e): raise
                                traceback.print_exc()
                                write_row(out, dict(kind="geo", mode=MODE, shard="", regime=regime, act=act, width=w,
                                                    seedA=i, seedB=j, smooth=smooth,
                                                    status="error:" + repr(e)[:40])); continue
                            for lr_ in lam_rels:
                                if RESUME and already_done(read_paths, regime, act, w, i, j, lr_):
                                    log(f"  [pair {i}-{j} lam_rel={lr_}] already measured -> skip"); continue
                                try:
                                    lam = max(lr_ * lmax, 1e-12)
                                    ts = list(np.linspace(0, 1, GEO_TGRID))
                                    gammas = []; resids = []; fdis = []; iters_used = []; rhs_ns = []; x0 = None
                                    try:
                                        for tt in ts:
                                            pt = {k: (1 - tt) * pA[k] + tt * pB[k] for k in pA}
                                            g, res, fdi, used, rn = christoffel_dd(ref, pt, b_ref, Xgeo, delta,
                                                                                   lam, micro, x0=x0)
                                            x0 = g                       # warm-start CG (GPU)
                                            gammas.append({k: v.detach().cpu() for k, v in g.items()})  # CPU: tranh OOM
                                            resids.append(res); fdis.append(fdi); iters_used.append(used); rhs_ns.append(rn)
                                        mid = int(np.argmin([abs(t - 0.5) for t in ts])); gamma_mid = _vnorm(gammas[mid])
                                        # Green integral on CPU: xi(t) = int G(t,s) Gamma(s) ds -> sup_t ||xi||
                                        G = green_matrix(ts); dt = ts[1] - ts[0]
                                        keys = list(delta.keys()); n = len(ts); sup = 0.0
                                        for ti in range(n):
                                            xi = None
                                            for si in range(n):
                                                w_ = float(G[ti, si]) * dt
                                                if w_ == 0.0: continue
                                                xi = {k: (gammas[si][k] * w_ if xi is None else xi[k] + gammas[si][k] * w_)
                                                      for k in keys}
                                            sup = max(sup, _vnorm(xi) if xi is not None else 0.0)
                                    finally:
                                        try: del gammas, x0
                                        except Exception: pass
                                        if DEVICE == "cuda": torch.cuda.empty_cache()
                                    dev_rel = sup / max(dn, 1e-30)
                                    cg_max = max(resids); hit_cap = max(iters_used) >= CG_ITERS
                                    write_row(out, dict(kind="geo", mode=MODE, shard="", regime=regime, act=act, width=w,
                                        seedA=i, seedB=j, dnorm=round(dn, 4), dev_geo=f"{sup:.6e}", dev_rel=f"{dev_rel:.6e}",
                                        gamma_mid=f"{gamma_mid:.6e}",
                                        flen_A=f"{flA:.6e}", flen_mid=f"{flM:.6e}", flen_B=f"{flB:.6e}",
                                        rq_A=f"{rqA:.6e}", rq_mid=f"{rqM:.6e}", rq_B=f"{rqB:.6e}",
                                        lam=f"{lam:.4e}", lam_rel=lr_, cg_resid=f"{cg_max:.2e}",
                                        fd_instab=f"{max(fdis):.3e}", rhs_norm=f"{max(rhs_ns):.6e}",
                                        cg_iters=max(iters_used), accA=accs[i], accB=accs[j],
                                        smooth=smooth, status=st_ok))
                                    warn = "  !!CG hit its iteration cap (not converged)" if hit_cap else ""
                                    log(f"  [pair {i}-{j} lam_rel={lr_}] ||d||={dn:.3g} dev_rel={dev_rel:.3e} "
                                        f"flen_mid={flM:.3e} rq_mid={rqM:.3e} cgres<={cg_max:.1e} "
                                        f"cg_it={max(iters_used)}{warn}")
                                except Exception as e:
                                    if _is_oom(e): raise
                                    traceback.print_exc()
                                    write_row(out, dict(kind="geo", mode=MODE, shard="", regime=regime, act=act, width=w,
                                                        seedA=i, seedB=j, lam_rel=lr_, smooth=smooth,
                                                        status="error:" + repr(e)[:40]))
                        break
                    except Exception as e:
                        if not _is_oom(e) or micro <= 8: raise
                        micro = max(8, micro // 2)
                        log(f"  !! OOM -> lowering micro to {micro} ({GEO_BATCH // micro} slices) and redoing this cell")
                        if DEVICE == "cuda": torch.cuda.empty_cache()
                try: del sds, ref, p_ref, b_ref
                except Exception: pass
                import gc; gc.collect()
                if DEVICE == "cuda": torch.cuda.empty_cache(); torch.cuda.synchronize()
            except Exception as e:
                traceback.print_exc()
                write_row(out, dict(kind="geo", mode=MODE, shard="", regime=regime, act=act, width=w,
                                    smooth=smooth, status="cell-error:" + repr(e)[:40]))
    log(f"DONE geo {MODE}")
    return out

def build_final(geo_csv):
    import pandas as pd
    def _col(df, cands):
        low = {c.lower(): c for c in df.columns}
        for k in cands:
            if k in low: return low[k]
        return None
    comb = None
    if COMBINED and os.path.exists(COMBINED):
        c = pd.read_csv(COMBINED)
        if _col(c, ["kind"]): c = c[c[_col(c, ["kind"])].astype(str) == "net"].copy()
        ren = {_col(c, ["regime"]): "regime", _col(c, ["act", "activation"]): "act",
               _col(c, ["width", "n"]): "width", _col(c, ["df_op", "df"]): "dF", _col(c, ["acc"]): "acc"}
        ren = {k: v for k, v in ren.items() if k}
        comb = c.rename(columns=ren)
        for cc in ["width", "dF", "acc"]:
            if cc in comb: comb[cc] = pd.to_numeric(comb[cc], errors="coerce")
    g = pd.read_csv(geo_csv)
    if "status" in g: g = g[g["status"].astype(str).str.startswith("ok")].copy()
    for cc in ["width", "dev_rel", "dev_geo", "dnorm", "gamma_mid", "flen_mid", "rq_mid",
               "fd_instab", "lam_rel", "rhs_norm", "cg_iters", "cg_resid"]:
        if cc in g: g[cc] = pd.to_numeric(g[cc], errors="coerce")
    if set(["dev_geo", "dnorm"]).issubset(g.columns): g["dev_rel2"] = g["dev_geo"] / g["dnorm"] ** 2
    # with a lambda sweep, report lam_rel=1e-2; the sweep itself stays in param_geo_*.csv
    if "lam_rel" in g and g["lam_rel"].notna().any():
        gg2 = g[np.isclose(g["lam_rel"], 1e-2)]
        if len(gg2) > 0: g = gg2
    def mi(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return (float(s.median()), float(s.quantile(.25)), float(s.quantile(.75)), int(len(s))) \
            if len(s) else (np.nan, np.nan, np.nan, 0)
    cells = set()
    for src in [comb, g]:
        if src is not None and set(["regime", "act", "width"]).issubset(src.columns):
            cells |= set(map(tuple, src[["regime", "act", "width"]].dropna().values))
    rows = []
    for (r, a, w) in sorted(cells, key=lambda x: (str(x[0]), str(x[1]), float(x[2]))):
        rec = dict(mode=MODE, regime=r, act=a, width=int(float(w)), smooth=int(a in SMOOTH_ACTS))
        if comb is not None:
            cc = comb[(comb.regime == r) & (comb.act == a) & (comb.width == float(w))]
            m, q1, q3, n = mi(cc["dF"]) if "dF" in cc else (np.nan, np.nan, np.nan, 0)
            rec.update(dF_med=m, dF_q1=q1, dF_q3=q3, n_seeds=n,
                       acc_med=mi(cc["acc"])[0] if "acc" in cc else np.nan)
        gc_ = g[(g.regime == r) & (g.act == a) & (g.width == float(w))]
        dm, dq1, dq3, ng = mi(gc_["dev_rel"]) if "dev_rel" in gc_ else (np.nan, np.nan, np.nan, 0)
        fm, fq1, fq3, _ = mi(gc_["flen_mid"]) if "flen_mid" in gc_ else (np.nan, np.nan, np.nan, 0)
        rec.update(n_pairs=ng, dev_rel_med=dm, dev_rel_q1=dq1, dev_rel_q3=dq3,
            dev_rel2_med=mi(gc_["dev_rel2"])[0] if "dev_rel2" in gc_ else np.nan,
            gamma_mid_med=mi(gc_["gamma_mid"])[0] if "gamma_mid" in gc_ else np.nan,
            flen_mid_med=fm, flen_mid_q1=fq1, flen_mid_q3=fq3,
            rq_mid_med=mi(gc_["rq_mid"])[0] if "rq_mid" in gc_ else np.nan,
            fd_instab_med=mi(gc_["fd_instab"])[0] if "fd_instab" in gc_ else np.nan,
            rhs_norm_med=mi(gc_["rhs_norm"])[0] if "rhs_norm" in gc_ else np.nan,
            cg_resid_max=float(gc_["cg_resid"].max()) if "cg_resid" in gc_ and len(gc_) else np.nan,
            cg_iters_max=float(gc_["cg_iters"].max()) if "cg_iters" in gc_ and len(gc_) else np.nan)
        rows.append(rec)
    T = pd.DataFrame(rows)
    def add_alpha(colmed, name):
        if colmed not in T or T[colmed].notna().sum() == 0:
            T[name] = np.nan; T[name + "_r2"] = np.nan; return
        for (r, a), gr in T.groupby(["regime", "act"]):
            al, r2 = _fit_alpha(gr["width"].values, gr[colmed].values)
            T.loc[(T.regime == r) & (T.act == a), name] = al
            T.loc[(T.regime == r) & (T.act == a), name + "_r2"] = r2
    add_alpha("dF_med", "alpha_dF"); add_alpha("dev_rel2_med", "alpha_devrel2")
    add_alpha("flen_mid_med", "alpha_flen"); add_alpha("dev_rel_med", "alpha_devrel")
    order = ["mode", "regime", "act", "width", "smooth", "n_seeds", "n_pairs", "acc_med",
             "dF_med", "dF_q1", "dF_q3", "dev_rel_med", "dev_rel_q1", "dev_rel_q3", "dev_rel2_med",
             "gamma_mid_med", "flen_mid_med", "flen_mid_q1", "flen_mid_q3", "rq_mid_med",
             "fd_instab_med", "rhs_norm_med", "cg_resid_max", "cg_iters_max",
             "alpha_dF", "alpha_dF_r2", "alpha_devrel", "alpha_devrel_r2",
             "alpha_devrel2", "alpha_devrel2_r2", "alpha_flen", "alpha_flen_r2"]
    for c in order:
        if c not in T: T[c] = np.nan
    T = T[order].sort_values(["regime", "act", "width"]).reset_index(drop=True)
    fin = os.path.join(OUT_DIR, f"{MODE}_final.csv"); T.to_csv(fin, index=False)
    hv = "dev_rel_med" in T and T["dev_rel_med"].notna().any()
    log(f"-> {fin}  ({len(T)} cells)  shape/length={'yes' if hv else 'NaN (no geo yet)'}  "
        f"dF={'yes' if comb is not None else 'missing'}")

def _count_ok(path):
    """Number of status=ok rows in a param_geo_*.csv (0 if it cannot be read)."""
    try:
        with open(path) as f:
            head = f.readline().rstrip("\n").split(",")
            if "status" not in head: return 0
            i = head.index("status")
            return sum(1 for ln in f
                       if len(ln.rstrip("\n").split(",")) > i
                       and ln.rstrip("\n").split(",")[i].startswith("ok"))
    except Exception:
        return 0

_KEY = ("regime", "act", "width", "seedA", "seedB", "lam_rel")

def _rows_of(path):
    """[(key, row_text)] for every status=ok row in a param_geo_*.csv."""
    out = []
    try:
        with open(path) as f:
            head = f.readline().rstrip("\n").split(",")
            if "status" not in head or any(k not in head for k in _KEY): return []
            si = head.index("status"); ki = [head.index(k) for k in _KEY]
            for ln in f:
                q = ln.rstrip("\n").split(",")
                if len(q) > si and q[si].startswith("ok"):
                    out.append((tuple(q[i] for i in ki), ln))
    except Exception:
        pass
    return out

def _harvest_worker_csvs(dst):
    """Collect leftover worker files into `dst`, dropping duplicate rows.

    Under multi-GPU, progress lives in param_geo_<mode>.gpu<k>.<pid>.csv and is
    folded into the main file only after every worker finishes
    (`_merge_worker_csvs`). If the session times out partway -- which is what
    happened on the CNN run -- that merge never runs, so the next `_seed_resume`
    finds no `param_geo_<mode>.csv` and starts over. Hence worker files are
    collected too, including ones mounted from a previous run."""
    parts = []
    for pat in (os.path.join(OUT_DIR, f"param_geo_{MODE}.gpu*.csv"),
                f"/kaggle/input/**/param_geo_{MODE}.gpu*.csv",
                f"/content/**/param_geo_{MODE}.gpu*.csv"):
        parts += glob.glob(pat, recursive=True)
    parts = sorted({os.path.abspath(x) for x in parts})
    if not parts: return 0
    seen = {k for k, _ in _rows_of(dst)} if os.path.exists(dst) else set()
    add = []
    for q in parts:
        for k, ln in _rows_of(q):
            if k in seen: continue
            seen.add(k); add.append(ln)
    if not add:
        log(f"[resume] found {len(parts)} worker files, no new rows"); return 0
    new_file = not os.path.exists(dst)
    with open(dst, "a") as f:
        if new_file: f.write(",".join(CSV) + "\n")
        f.writelines(add)
    log(f"[resume] collected {len(add)} rows from {len(parts)} leftover worker files -> {dst}")
    return len(add)

def _seed_resume():
    """Inherit progress from an earlier run so finished cells are skipped.

    Candidates are ranked by the number of `ok` rows, not alphabetically. The
    earlier version took `sorted(...)[0]`, so with several param_geo_<mode>.csv
    copies attached (say one full 840-row file and two half-grid 420-row ones)
    it could pick a half file -- and once did. Every candidate found is printed
    so the choice can be checked.

    GEO_RESUME=<path> forces that exact file.
    GEO_RESUME=none         disable resume and measure everything again."""
    import shutil
    dst = os.path.join(OUT_DIR, f"param_geo_{MODE}.csv")
    forced = _env("GEO_RESUME")
    if forced and forced.lower() == "none":
        log("[resume] GEO_RESUME=none -> ignoring earlier progress, starting from scratch"); return
    if os.path.exists(dst):
        _harvest_worker_csvs(dst)
        log(f"[resume] found {dst} ({_count_ok(dst)} ok rows) -> continuing"); return
    if forced:
        if not os.path.exists(forced):
            raise SystemExit(f"GEO_RESUME={forced} does not exist")
        cands = [forced]
    else:
        cands = []
        for pat in (f"param_geo_{MODE}.csv",
                    f"/kaggle/input/**/param_geo_{MODE}.csv",
                    f"/kaggle/working/**/param_geo_{MODE}.csv",
                    f"/content/**/param_geo_{MODE}.csv",
                    f"/content/drive/MyDrive/**/param_geo_{MODE}.csv"):
            cands += glob.glob(pat, recursive=True)
        cands = sorted({os.path.abspath(c) for c in cands}
                       - {os.path.abspath(dst)})
    if not cands:
        if _harvest_worker_csvs(dst):
            log(f"[resume] recovered from worker files: {_count_ok(dst)} ok rows")
        else:
            log("[resume] no earlier progress -> starting from scratch")
        return
    scored = sorted(((_count_ok(c), c) for c in cands), reverse=True)
    if len(scored) > 1:
        log(f"[resume] found {len(scored)} copies of param_geo_{MODE}.csv:")
        for n, c in scored: log(f"           {n:5d} ok rows  {c}")
    n, best = scored[0]
    if n == 0:
        if _harvest_worker_csvs(dst):
            log(f"[resume] recovered from worker files: {_count_ok(dst)} ok rows")
        else:
            log("[resume] no copy has an ok row -> starting from scratch")
        return
    shutil.copy(best, dst)
    log(f"[resume] chose the copy with the most ok rows: {best} -> {dst} ({n} rows done)")
    _harvest_worker_csvs(dst)

# ==================================================================== MULTI-GPU
# Cells are split across GPUs, one process each. Every worker writes its own
# CSV -- two processes never append to the same file, so rows cannot interleave
# -- and the parent merges them at the end. No arithmetic changes: this only
# partitions the work list.
WORKER_ID    = _env("GEO_WORKER")               # set by the parent; None means this is the parent
WORKER_TASKS = _env("GEO_TASKS")                # "regime:act:width,..."

def _resolve_gpus():
    """Which GPUs to use. By default every visible GPU.

    - worker (GEO_WORKER set): returns [] -- a child sees one GPU through
      CUDA_VISIBLE_DEVICES and must not spawn further children.
    - GEO_GPUS="0"   -> one GPU, run inline in this process (the off switch).
    - GEO_GPUS="0,1" -> pick explicitly.
    - unset          -> torch.cuda.device_count().
    SMOKE does not enable multi-GPU: the work is too small for spawning to pay."""
    if WORKER_ID is not None: return []
    env = _env("GEO_GPUS")
    nv = torch.cuda.device_count()
    if env is not None:
        want = [s.strip() for s in env.split(",") if s.strip()]
        if len(want) > nv: raise SystemExit(f"GEO_GPUS={want} but only {nv} GPU(s) visible")
        return want
    if SMOKE: return []
    return [str(i) for i in range(nv)] if nv > 1 else []

GPUS = _resolve_gpus()

def _self_path():
    """Source file to spawn workers from, or None if it cannot be determined.

    In a notebook there is no `__file__` -- the script lives in a cell. In that
    case the cell's own source is written to disk and workers are spawned from
    that temporary file. If even that fails, return None and let the caller fall
    back to one GPU: a speed optimisation must not kill the job."""
    try:
        p = os.path.abspath(__file__)
        if os.path.exists(p): return p
    except NameError:
        pass
    try:                                   # running inside a notebook
        from IPython import get_ipython
        cells = get_ipython().user_ns.get("In") or []
        src = next(c for c in reversed(cells) if "def run_geo" in c and "def _spawn_workers" in c)
        p = os.path.join(OUT_DIR, "_measure_geodesic_self.py")
        with open(p, "w") as f: f.write(src)
        log(f"[multi-gpu] no __file__ (running in a notebook) -> wrote the source to {p}")
        return p
    except Exception as e:
        log(f"[multi-gpu] could not recover the source to spawn workers: {e!r}")
        return None

def _enc_tasks(ts): return ",".join(f"{r}:{a}:{w}" for (r, a, w) in ts)
def _dec_tasks(s):
    out = []
    for it in s.split(","):
        if not it.strip(): continue
        r, a, w = it.split(":"); out.append((r, a, int(w)))
    return out

def worker_csv(k, pid=None):
    """One worker's own CSV. The PID is included so that even if two processes
    somehow take the same WORKER_ID they cannot overwrite each other: parallel
    appends to one file interleave rows and corrupt the CSV. The merge globs
    every gpu*.csv."""
    return os.path.join(OUT_DIR, f"param_geo_{MODE}.gpu{k}.{pid or os.getpid()}.csv")

def _merge_worker_csvs(base):
    """Merge param_geo_{mode}.gpu*.csv into the main file, dropping repeated headers."""
    parts = sorted(glob.glob(os.path.join(OUT_DIR, f"param_geo_{MODE}.gpu*.csv")))
    if not parts: return
    have_header = os.path.exists(base)
    n = 0
    with open(base, "a") as dst:
        if not have_header: dst.write(",".join(CSV) + "\n")
        for p in parts:
            with open(p) as f:
                f.readline()                     # drop the worker header
                for ln in f:
                    if ln.strip(): dst.write(ln); n += 1
            os.remove(p)
    log(f"[merge] gop {len(parts)} file worker -> {base} (+{n} dong)")

def _spawn_workers(tasks, self_path):
    """Run one child process per GPU; True if all of them exited cleanly."""
    import subprocess
    groups = [tasks[k::len(GPUS)] for k in range(len(GPUS))]   # interleave, so the load is balanced
    procs = []
    for k, (gpu, grp) in enumerate(zip(GPUS, groups)):
        if not grp: continue
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)   # the child sees one GPU, so DEVICE=cuda:0
        env["GEO_WORKER"] = str(k); env["GEO_TASKS"] = _enc_tasks(grp)
        env["GEO_MODE"] = MODE                   # the child runs this mode only, never 'all'
        env.pop("GEO_GPUS", None)                # prevent recursive spawning
        log(f"[gpu {gpu}] worker {k}: {len(grp)} o, o dau = {grp[0]}")
        procs.append(subprocess.Popen([sys.executable, self_path], env=env))
    rc = [p.wait() for p in procs]
    for k, r in enumerate(rc):
        if r != 0: log(f"!! worker {k} exited with code {r}")
    return all(r == 0 for r in rc)

def run_mode(mode):
    set_mode(mode)
    log(f"=== run_geo[{MODE}] DEVICE={DEVICE} SMOKE={SMOKE} SHARD={ONLY_SHARD} "
        f"REGIMES={ONLY_REGIMES or 'het'} ACTS={GEO_ACTS} WIDTHS={GEO_WIDTHS} "
        f"GPUS={GPUS or '1'} WORKER={WORKER_ID or '-'} ===")
    log(f"  ckpt_roots={CKPT_ROOTS}")
    log(f"  combined  ={COMBINED}")

    if WORKER_ID is not None:                    # --- child process ---
        _t = _dec_tasks(WORKER_TASKS)
        log(f"[worker {WORKER_ID}] pid={os.getpid()} {len(_t)} o, o dau = {_t[0] if _t else '-'}")
        run_geo(tasks=_t, out=worker_csv(WORKER_ID))
        return                                   # the parent handles merging and build_final

    _seed_resume()
    base = base_csv()
    self_path = _self_path() if (GPUS and len(GPUS) > 1) else None
    if GPUS and len(GPUS) > 1 and self_path:     # --- parent, multiple GPUs ---
        # Download the dataset before spawning. Left to themselves the workers
        # write into ./data at the same time: one is mid-write while the other
        # checks the md5, reports "File not found or corrupted" and dies. With
        # the parent fetching it once, the workers only read, and download=True
        # becomes harmless.
        log("[multi-gpu] loading the dataset in the parent before spawning ...")
        load_data(); _CACHE.clear()
        ok = _spawn_workers(all_tasks(), self_path)
        _merge_worker_csvs(base)
        if not ok: log("!! a worker failed -> rerun to resume what is missing")
        geo = base
    else:                                        # --- single GPU ---
        if GPUS and len(GPUS) > 1:
            log(f"!! could not spawn workers -> falling back to 1 GPU (still correct, about {len(GPUS)}x slower).")
            log("   For multi-GPU, save this as a .py file and run: python measure_geodesic.py")
        geo = run_geo()
    try: build_final(geo)
    except Exception as e:
        traceback.print_exc(); log("!! merge failed (param_geo_*.csv is still on disk):", repr(e)[:100])

def main():
    if WORKER_ID is None: self_test()            # architecture-independent: run once
    for mode in MODES: run_mode(mode)
    log("DONE all modes:", MODES)

if __name__ == "__main__": main()
