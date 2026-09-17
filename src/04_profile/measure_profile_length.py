#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fisher length: the Rayleigh quotient rq(t) along the whole path, all regimes.

     python measure_profile_length.py              # MODE=mlp (default)
     MODE=cnn python measure_profile_length.py
     MODE=all python measure_profile_length.py     # mlp, then cnn, then ts

What was missing
----------------
param_geo_{mode}.csv already carries rq in all three regimes, but only at three
nodes, t in {0, 1/2, 1} (columns rq_A / rq_mid / rq_B). That does not show the
shape of the curve: whether it bulges or dips in the middle, and where the peak
sits. This file measures rq on a 21-point grid, in all three regimes and all
three architectures.

Definitions (identical to _flen in the geodesic script)
-------------------------------------------------------
    q(t)    = Delta^T F(gamma_lin(t)) Delta          (>= 0)
    rq(t)   = q(t) / ||Delta||^2     the Rayleigh quotient, invariant to the
                                     scale of Delta
    flen(t) = q(t) / 2               the Fisher length

The CSV stores rq_t* and dnorm only, since flen follows from them:
        flen(t) = 0.5 * rq(t) * dnorm^2
No redundant column is written, so the file stays small.

Cheaper than the other two profile files
----------------------------------------
Each t node costs exactly one fisher_vp -- no CG, no finite differences -- so a
21-point grid costs far less than the Christoffel file.

Cross-check against the earlier run
-----------------------------------
Afterwards the script compares the rq(0), rq(1/2), rq(1) it just measured with
rq_A, rq_mid, rq_B in param_geo_{mode}.csv. Those must agree: it is the same
checkpoints, the same Fisher batch and the same permutation alignment, so this
is a free pipeline check. A gap above 1% means some configuration differs, and
the script says so.

Resume
------
  - Copies an existing CSV from the input mount into the working directory first.
  - Truncates a half-written row if the previous session was killed partway.
  - Skips finished cells before loading any .pt.
  - Stops on its own near the time limit (BUDGET_H) so the CSV can flush.

Required on disk
----------------
  ckpt_{RUN_TAG}/{regime}_{act}_w{w}_s{s}.pt, and MNIST / FashionMNIST
  (ts is synthetic and needs no download; F does not use the labels)

Output
------
  profile_length_{mode}.csv        one row per pair, resumable
  profile_length_{mode}_cell.csv   per-cell medians (regime x act x width)
"""
import os, sys, time, math, glob, shutil, itertools, traceback, zipfile, io
try:
    sys.stdout.reconfigure(line_buffering=True); sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.func import functional_call, jvp as _fjvp, vjp as _fvjp, jacrev as _jacrev

# ======================= PARAMETERS (as in the geodesic script) =============
MODE      = os.environ.get("MODE", "mlp")          # mlp | cnn | ts | all
REGIMES   = ["ntk", "sp", "mup"]
ACTS      = ["gelu", "tanh", "swish", "softplus"]  # C^3 only; relu is excluded
NSEEDS    = 5
# This file is cheap (one fisher_vp per t node, no CG), so it keeps all 10 pairs.
PAIRS     = int(os.environ.get("PAIRS", "10"))     # combinations(5,2), the same pairs as the barrier

# Sharded like the other two files, so all three split the work the same way.
SHARD      = os.environ.get("SHARD")
SHARD_PLAN = {0: ("ntk", ["gelu","tanh"]), 1: ("ntk", ["swish","softplus"]),
              2: ("sp",  ["gelu","tanh"]), 3: ("sp",  ["swish","softplus"]),
              4: ("mup", ["gelu","tanh"]), 5: ("mup", ["swish","softplus"])}

def plan_cells():
    if SHARD not in (None, ""):
        rg, acts = SHARD_PLAN[int(SHARD)]
        return [(rg, a) for a in acts if a in ACTS]
    return [(rg, a) for rg in REGIMES for a in ACTS]

TGRID     = int(os.environ.get("TGRID", "21"))     # t grid, fine enough to show the shape
FISHER_N  = 2048                                   # same as GEO_BATCH, so results compare directly
MICRO     = 64
RESUME    = True
SELFTEST  = os.environ.get("SELFTEST", "1") == "1"
ANCHOR    = os.environ.get("ANCHOR", "1") == "1"   # cross-check against param_geo_*.csv
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

# ------------------------------------------------------------------ DATA (labels are not needed)
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

def fisher_vp(m, p, b, x, v, micro):
    """F v, with F = E_x[J^T S J] and S = diag(p) - p p^T, the softmax Fisher."""
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

def rayleigh_at(m, pt, b, xf, delta, d2, micro):
    """tra ve (q, rq) voi q = Delta^T F(pt) Delta, rq = q/||Delta||^2."""
    q = _vdot(delta, fisher_vp(m, pt, b, xf, delta, micro))
    return q, q/d2

# ------------------------------------------------------------------ SELF-TEST
def self_test():
    """rq via fisher_vp must match the dense Fisher to machine precision."""
    log("  [self-test] rq via fisher_vp == dense Fisher ...")
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
    delta = unflat(dflat)
    J = _jacrev(lambda q: _call(m, q, b, x))(p); B = x.shape[0]
    Jf = torch.cat([J[k].reshape(B, 3, -1) for k in keys], 2)
    pr = torch.softmax(_call(m, p, b, x), 1)
    S = torch.diag_embed(pr) - pr.unsqueeze(2)*pr.unsqueeze(1)
    Fd = torch.einsum('bki,bkl,blj->ij', Jf, S, Jf)/B
    q_dense = float(dflat @ Fd @ dflat)
    q_vp, rq_vp = rayleigh_at(m, p, b, x, delta, float(dflat.norm())**2, micro=8)
    rel = abs(q_vp - q_dense)/max(abs(q_dense), 1e-30)
    assert rel < 1e-9, f"rq sai {rel:.2e}"
    assert q_dense > 0, "F must be positive semi-definite"
    log(f"  [self-test] OK  rel={rel:.2e}  rq={rq_vp:.6e}")
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
        for rg in REGIMES:
            n = sum(1 for fn in idx if fn.startswith(rg + "_"))
            log(f"[ckpt]   '{rg}': {n} file" + ("  -- OK" if n else "  !! THIEU"))
    return idx

def find_ckpt(tag, regime, act, w, s):
    return _build_index(tag).get(f"{regime}_{act}_w{w}_s{s}.pt")

def load_sd(src):
    """src is a .pt path, or the tuple ("zip", zip_path, entry_name)."""
    if isinstance(src, tuple):
        buf = io.BytesIO(_zopen(src[1]).read(src[2]))
        try: d = torch.load(buf, map_location="cpu", weights_only=False)
        except TypeError: buf.seek(0); d = torch.load(buf, map_location="cpu")
    else:
        try: d = torch.load(src, map_location="cpu", weights_only=False)
        except TypeError: d = torch.load(src, map_location="cpu")
    return d["sd"], d.get("acc")

# --------------------------------------------------------- TRAIN (only when checkpoints are missing)
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
    """Labelled data for training, unlike load_X: F does not need the labels."""
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
    need = 1 + max(max(i, j) for (i, j) in want)      # only seeds up to this index are needed
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
def t_cols():  return [f"rq_t{t:.3f}" for t in t_grid()]
def COLS():    return (["mode","regime","act","width","seedA","seedB","dnorm"]
                       + t_cols() + ["rq_min","rq_max","t_argmin","accA","accB","status"])

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
        log(f"[resume] dropped {len(lines)-len(keep)} corrupt rows -> those pairs will be redone")

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
            done.add((f[1], f[2], f[3], f[4], f[5]))     # regime, act, width, seedA, seedB
    return done

def write_row(path, row):
    cols = COLS(); new = not os.path.exists(path)
    with open(path, "a") as f:
        if new: f.write(",".join(cols) + "\n")
        f.write(",".join(str(row.get(c, "")) for c in cols) + "\n"); f.flush()

# ------------------------------------------------------------- CROSS-CHECK vs EARLIER RUN
def anchor_check(mode, path):
    """Compare the rq(0), rq(1/2), rq(1) just measured with rq_A/rq_mid/rq_B in param_geo."""
    import pandas as pd
    src = _seek_csv(f"param_geo_{mode}.csv") or os.path.join("result-1", f"param_geo_{mode}.csv")
    if not src or not os.path.exists(src):
        log(f"[anchor] param_geo_{mode}.csv not found -> skipping the cross-check"); return
    old = pd.read_csv(src)
    old = old[old.status.astype(str) == "ok"]
    if "lam_rel" in old:
        old = old[pd.to_numeric(old.lam_rel, errors="coerce").round(6) == 0.01]
    new = pd.read_csv(path); new = new[new.status.astype(str) == "ok"]
    if not len(new) or not len(old):
        log("[anchor] no data to cross-check against"); return
    key = ["regime","act","width","seedA","seedB"]
    for c in key[2:]: old[c] = pd.to_numeric(old[c], errors="coerce"); new[c] = pd.to_numeric(new[c], errors="coerce")
    m = new.merge(old[key + ["rq_A","rq_mid","rq_B"]], on=key, how="inner")
    if not len(m):
        log("[anchor] no pair matched (different checkpoints or seeds?) -> skipping"); return
    ts = t_grid()
    c0, cm, c1 = f"rq_t{ts[0]:.3f}", f"rq_t{ts[int(np.argmin(np.abs(np.array(ts)-0.5)))]:.3f}", f"rq_t{ts[-1]:.3f}"
    log(f"[anchor] cross-checking {len(m)} pairs against param_geo_{mode}.csv (lam_rel=1e-2):")
    worst = 0.0
    for new_c, old_c, nm in [(c0,"rq_A","t=0"), (cm,"rq_mid","t=1/2"), (c1,"rq_B","t=1")]:
        a = pd.to_numeric(m[new_c], errors="coerce"); b = pd.to_numeric(m[old_c], errors="coerce")
        rel = ((a - b).abs()/b.abs().clip(lower=1e-30)).dropna()
        if not len(rel): continue
        worst = max(worst, float(rel.max()))
        log(f"[anchor]   {nm:6s}: lech trung vi={rel.median():.3%}  max={rel.max():.3%}")
    if worst > 0.01:
        log("[anchor] !! off by more than 1%: check FISHER_N, the pair order, and the permutation alignment")
    else:
        log("[anchor] OK -- the pipeline matches the earlier run")

# ------------------------------------------------------------------ MAIN
def run_mode(mode):
    cfg = _MODE_CFG[mode]; tag = cfg["tag"]
    out  = os.path.join(OUT_DIR, f"profile_length_{mode}.csv")
    restore_csv(out, f"profile_length_{mode}.csv")
    done = load_done(out)

    if not _build_index(tag):      # check for checkpoints before loading any data
        if not TRAIN:
            log("[ckpt] -> stopping: checkpoints (.pt) are missing and TRAIN=0.")
            log("[ckpt]    (.pt files are the input. Set TRAIN=1 to train them here, or attach a checkpoint dataset.)")
            return out
        log(f"[ckpt] -> no checkpoints found: training them here ({TRAIN_EPOCHS[mode]} epochs per net).")
        log(f"[ckpt]    saved to {ckpt_dir(tag)} -- keep them so the next run need not retrain.")
    X = load_X(mode); Xf = X[:FISHER_N].to(DEVICE)
    n_skip = 0
    ts = t_grid(); cols = t_cols()
    log(f"=== MODE={mode} SHARD={SHARD if SHARD not in (None,'') else 'het'} cells={plan_cells()} widths={cfg['widths']} "
        f"TGRID={TGRID} pairs={PAIRS} fisher_n={FISHER_N} DEVICE={DEVICE} -> {out}")

    for (regime, act) in plan_cells():
        for w in cfg["widths"]:
            if out_of_time():
                log(f"[budget] het {BUDGET_H}h -> dung sach"); return out
            # --- skip finished cells before loading any .pt ---
            want = [(i, j) for (i, j) in
                    list(itertools.combinations(range(NSEEDS), 2))[:PAIRS]
                    if (regime, act, str(w), str(i), str(j)) not in done]
            if not want:
                log(f"  [{regime}/{act}/w{w}] already done -> skipping"); continue
            try:
                sds, accs, got = load_or_train(mode, tag, regime, act, w, want)
                if len(sds) < 2:
                    miss = [x for x in range(NSEEDS) if x not in got]
                    n_skip += 1
                    log(f"  [{regime}/{act}/w{w}] MISSING DATA: only {len(sds)}/{NSEEDS} checkpoints "
                        f"(found seeds {got}, missing {miss}) -- 2 are needed for a pair, skipping this cell")
                    continue
                log(f"=== {regime}/{act}/w{w}  ({len(sds)} nets, {len(want)} pairs left) ===")
                ref = build_net(mode, w, act, regime).to(DEVICE).eval()
                p_ref, b_ref = _pb(ref)
                ag, gs = perm_spec(mode, build_net(mode, w, act, regime))
                to_params = lambda sd: {k: sd[k].to(DEVICE) for k in p_ref.keys()}

                for (i, j) in want:
                    if i >= len(sds) or j >= len(sds): continue
                    if out_of_time():
                        log(f"[budget] het {BUDGET_H}h -> dung sach"); return out
                    try:
                        perms = weight_matching(ag, gs, sds[i], sds[j], iters=8, seed=i*13 + j)
                        pA = to_params(sds[i]); pB = to_params(apply_perm(sds[j], ag, perms))
                        delta = {k: pB[k] - pA[k] for k in pA}
                        dn = _vnorm(delta); d2 = max(dn*dn, 1e-30)

                        rqs = []
                        for tt in ts:
                            pt = {k: (1-tt)*pA[k] + tt*pB[k] for k in pA}
                            _, rq = rayleigh_at(ref, pt, b_ref, Xf, delta, d2, MICRO)
                            rqs.append(rq)
                        rqs = np.array(rqs)
                        kmin = int(np.argmin(rqs))

                        row = dict(mode=mode, regime=regime, act=act, width=w,
                                   seedA=i, seedB=j, dnorm=f"{dn:.6e}",
                                   rq_min=f"{rqs.min():.6e}", rq_max=f"{rqs.max():.6e}",
                                   t_argmin=f"{ts[kmin]:.4f}",
                                   accA=accs[i], accB=accs[j], status="ok")
                        for c, v in zip(cols, rqs): row[c] = f"{v:.6e}"
                        write_row(out, row)
                        mid = int(np.argmin(np.abs(np.array(ts) - 0.5)))
                        log(f"  [{i}-{j}] ||d||={dn:.3g}  rq(0)={rqs[0]:.3e} "
                            f"rq(1/2)={rqs[mid]:.3e} rq(1)={rqs[-1]:.3e}  "
                            f"mid/end={rqs[mid]/max(0.5*(rqs[0]+rqs[-1]),1e-30):.3f}")
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
        log(f"!! {n_skip} cells skipped for missing checkpoints -- see the MISSING DATA lines above.")
    log(f"DONE length profile [{mode}]")
    return out

def aggregate(mode, path):
    import pandas as pd
    if not os.path.exists(path): return
    d = pd.read_csv(path); d = d[d.status.astype(str) == "ok"].copy()
    if not len(d): return
    cols = [c for c in t_cols() if c in d]
    for c in cols + ["dnorm", "width"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    g = d.groupby(["mode","regime","act","width"])[cols].median().reset_index()
    g.insert(4, "n_pairs", d.groupby(["mode","regime","act","width"]).size().values)
    fin = os.path.join(OUT_DIR, f"profile_length_{mode}_cell.csv")
    g.to_csv(fin, index=False); log(f"-> {fin}  ({len(g)} o)")

    ts = np.array(t_grid()); mid = int(np.argmin(np.abs(ts - 0.5)))
    ratio = g[cols[mid]]/(0.5*(g[cols[0]] + g[cols[-1]]))
    log("\n  curvature ratio, midpoint over endpoints (median per regime) -- below 1 means a dip in the middle:")
    for reg, v in ratio.groupby(g.regime).median().items():
        log(f"    {reg:4s}  mid/end = {v:.3f}  " + ("(sap)" if v < 1 else "(phinh)"))

def main():
    modes = list(_MODE_CFG) if MODE == "all" else [MODE]
    log(f"=== measure_profile_length  modes={modes}  TGRID={TGRID} ===")
    if SELFTEST: self_test()
    for m in modes:
        p = run_mode(m)
        aggregate(m, p)
        if ANCHOR:
            try: anchor_check(m, p)
            except Exception as e: log(f"[anchor] bo qua ({e!r})")

if __name__ == "__main__":
    main()
