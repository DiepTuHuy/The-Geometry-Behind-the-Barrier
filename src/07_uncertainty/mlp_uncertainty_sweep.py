#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 mlp_uncertainty_sweep.py -- de-confound predictive uncertainty from regime
============================================================================
WHY THIS RUN EXISTS

The paper claims: "barrier collapse requires Fisher length reduction in the
presence of controlled predictive uncertainty."  The released grid cannot test
the second half of that sentence, because rho* and the parameterisation are
perfectly collinear in it:

    rho*(endpoint, largest width)   NTK-lazy   [0.203, 0.615]
                                    Standard   [0.000, 0.068]
                                    muP        [0.000, 0.104]

The two ranges do not overlap in a single cell, and inside the 24
feature-learning cells the residual rho* explains NONE of the barrier exponent
(R^2 = 0.000, p = 0.99; adding it on top of the Fisher-length exponent moves
R^2 by +0.0000).  So "controlled uncertainty" and "not NTK-lazy" are the same
variable in that data, and no amount of re-analysis separates them.

This run separates them.  It holds the parameterisation, architecture, width
grid and optimiser fixed, and moves rho* with a training-time knob --
LABEL SMOOTHING -- which raises the uncertainty attainable at the minimum
without touching anything else.  In the feature-learning regimes rho* is
~0 at eps=0, so the sweep walks it up through the NTK-lazy range while the
regime stays feature-learning.  That is the decisive direction: if collapse
survives a large rho* at fixed regime, the "controlled uncertainty" clause is
not doing work and the claim should be reworded; if collapse weakens as rho*
rises, the clause is earned.

WHAT IS MEASURED (per aligned seed pair, one round, no cross-round join)

    dnorm, L_A, L_B, B, t_star            barrier on the T=41 grid
    rho_A, rho_mid, rho_max, acc_mid      Definition 2.3, on TRUE labels
    flen_A, flen_mid, flen_B, rq_mid      1/2 Delta^T F Delta, Rayleigh
    R_end, R_mid                          B / (flen/4), as in 03_final

WHAT THIS KNOB DOES NOT DO.  Label smoothing is not a surgical rho*-only
control: raising eps also moves the minimum, hence Delta, hence the Fisher
length.  A smoke run at w=64/muP already shows both moving together
(eps 0 -> 0.2: rho_A 0.024 -> 0.235, flen_A 0.34 -> 2.68).  So the analysis
cannot be "vary rho*, hold everything else"; it has to regress the barrier
exponent on BOTH alpha_flen and rho* and ask whether rho* carries an
independent coefficient.  That is why the grid is 4 widths x 5 eps rather than
one width x many eps: the two variables need to vary along different
directions before a regression can tell them apart.  What the sweep does buy,
and what no re-analysis of the released data can, is rho* varying WITHIN a
fixed parameterisation -- which is the collinearity that currently makes the
claim untestable.

IMPORTANT -- the smoothing enters TRAINING ONLY.  L(t), B and rho* are all
evaluated with plain cross-entropy against the true one-hot labels, exactly as
in src/03_final, so every number here is directly comparable with
param_final_mlp.csv and param_geo_mlp.csv.

PRIMITIVES are copied verbatim from src/01_train/mlp_train_shard0.py and
src/03_final/mlp_measure_final.py (param_cfg, ScaledLinear, MLP, perm_spec,
apply_perm, weight_matching, fisher_vp, loss_acc_rho, flen_at) so this run is
not a second pipeline.  EPS_LIST = [0.0, ...] means the eps=0 column
reproduces the released cells and is the built-in cross-check: its B and rho_A
should match param_final_mlp.csv for the same (regime, act, width).

COST.  Trainings = REGIMES x ACTS x WIDTHS x EPS x NSEEDS; measurement =
cells x PAIRS.  With the defaults below (2 x 1 x 4 x 5 x 5 = 200 trainings,
40 cells x 10 pairs) this is roughly 2 h on a Kaggle T4.  Trim WIDTHS or
EPS_LIST first if the session is tight; the CSV is resumable, so a run that is
cut off continues where it stopped.

OUTPUT
    param_uncert_mlp.csv     one row per seed pair, resumable
    uncert_mlp_cell.csv      per-cell medians
============================================================================
"""
import os, sys, time, math, itertools, traceback
try:
    sys.stdout.reconfigure(line_buffering=True); sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.func import functional_call, jvp as _fjvp, vjp as _fvjp

def _tv():
    import torchvision; return torchvision

# ==================================================================== CONFIG
# This block is the source of truth for the run (the repo convention: the
# docstrings drift, the CONFIG does not).
SMOKE      = False
RESUME     = True

REGIMES    = ["mup", "sp"]                      # the two regimes whose rho* is ~0
ACTS       = ["gelu"]                           # one smooth activation is enough here
WIDTHS     = [64, 256, 1024, 4096]              # >=3 widths, else no exponent
EPS_LIST   = [0.0, 0.05, 0.10, 0.20, 0.40]      # label smoothing -> rho* knob
NSEEDS     = 5
PAIRS      = int(os.environ.get("PAIRS", "10")) # C(5,2) = 10

# training -- identical to src/01_train/mlp_train_shard0.py
EPOCHS     = 30
LR         = 0.1
BATCH      = 256
WARMUP_EPOCHS = 8
CLIP_NORM  = 1.0
SEED_BASE  = 4321

# measurement -- identical to src/03_final/mlp_measure_final.py
TGRID_FINE   = 41
TGRID_COARSE = 9
EVAL_N       = 10000
FISHER_N     = 2048
MICRO        = 64
MATCH_ITERS  = 8

if SMOKE:
    REGIMES = ["mup"]; WIDTHS = [64, 128]; EPS_LIST = [0.0, 0.2]
    NSEEDS = 2; PAIRS = 1; EPOCHS = 4; TGRID_FINE = 9; TGRID_COARSE = 5
    EVAL_N = 2000; FISHER_N = 512

DIN, K   = 784, 10
BASE     = 64
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR  = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUT_CSV  = os.path.join(OUT_DIR, "param_uncert_mlp.csv")
CELL_CSV = os.path.join(OUT_DIR, "uncert_mlp_cell.csv")

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def set_seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ===================================================================== DATA
_CACHE = {}
def load_data_xy():
    """MNIST, normalised and flattened exactly as in src/03_final."""
    if "xy" in _CACHE: return _CACHE["xy"]
    root = "./data"
    for cand in ("/kaggle/input/mnist-dataset", "/kaggle/input/mnist", "./data"):
        if os.path.isdir(cand): root = cand; break
    ds = _tv().datasets.MNIST(root, train=True, download=True)
    X = ((ds.data.float()/255.0) - 0.1307)/0.3081
    X = X.reshape(-1, 784); Y = ds.targets.clone()
    _CACHE["xy"] = (X, Y); return X, Y

# ==================================================================== MODEL
# Verbatim from src/01_train/mlp_train_shard0.py -- do not "improve" these.
def make_act(n):
    return {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh,
            "swish": nn.SiLU, "softplus": nn.Softplus}[n]()

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

class MLP(nn.Module):
    def __init__(self, width, act, regime="mup", din=DIN, k=K):
        super().__init__()
        self.fc1 = ScaledLinear(din, width, regime, "input")
        self.fc2 = ScaledLinear(width, width, regime, "hidden")
        self.fc3 = ScaledLinear(width, k, regime, "output")
        self.a1 = make_act(act); self.a2 = make_act(act)
        self.width = width; self.regime = regime
    def forward(self, x):
        return self.fc3(self.a2(self.fc2(self.a1(self.fc1(x)))))
    def opt_groups(self, base_lr):
        return [{"params": [m.weight, m.bias], "lr": base_lr*m.lr_scale}
                for m in (self.fc1, self.fc2, self.fc3)]

def build_net(w, act, regime):
    return MLP(w, act, regime).to(DEVICE)

# ============================================================== PERMUTATIONS
def perm_spec(model):
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
        else:
            out[n] = t.clone()
    return out

def _perm_except(t, axes, perms, exc):
    tt = t
    for a, g in enumerate(axes):
        if g is not None and a != exc: tt = tt.index_select(a, perms[g])
    return tt

def weight_matching(ag, gs, sdA, sdB, iters, seed=0):
    rng = np.random.RandomState(seed)
    perms = {g: torch.arange(n) for g, n in gs.items()}
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
                S += torch.movedim(A, axis, 0).reshape(n, -1) @ \
                     torch.movedim(B, axis, 0).reshape(n, -1).T
            ci = linear_sum_assignment(-S.numpy())[1]
            new = torch.as_tensor(ci, dtype=torch.long)
            if not torch.equal(new, perms[g]): moved += 1
            perms[g] = new
        if moved == 0: break
    return perms

# ================================================================ PRIMITIVES
def _pb(m):
    return ({k: v.detach() for k, v in m.named_parameters()},
            {k: v.detach() for k, v in m.named_buffers()})
def _call(m, p, b, x): return functional_call(m, {**p, **b}, (x,))
def _vdot(a, b): return float(sum((a[k]*b[k]).sum() for k in a))
def _vnorm(a):
    return float(torch.sqrt(torch.clamp(sum((a[k]*a[k]).sum() for k in a), min=0)))

def fisher_vp(m, p, b, x, v, micro):
    B = x.shape[0]; acc = None
    for i in range(0, B, micro):
        xb = x[i:i+micro]
        def f(pp): return _call(m, pp, b, xb)
        logits, Jv = _fjvp(f, (p,), (v,))
        pr = torch.softmax(logits, 1)
        s  = pr*Jv - pr*(pr*Jv).sum(1, keepdim=True)
        JTs = _fvjp(f, p)[1](s)[0]
        acc = {k: JTs[k].detach() for k in JTs} if acc is None else \
              {k: acc[k] + JTs[k].detach() for k in acc}
    return {k: acc[k]/B for k in acc}

@torch.no_grad()
def loss_acc_rho(m, p, b, x, y, micro=512):
    """L = plain cross-entropy; rho* = E||p_w(x) - e_y||_2 (Definition 2.3).

    Deliberately UNSMOOTHED, whatever the training used: B and rho* have to
    stay on the same scale as param_final_mlp.csv or the sweep measures
    nothing comparable."""
    n = x.shape[0]; sL = sR = 0.0; sC = 0
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

# ================================================================== TRAINING
def train(model, X, Y, epochs, eps):
    """Identical to src/01_train except for `label_smoothing=eps`.

    That one argument is the whole experiment: it moves the uncertainty of the
    minimum without touching width, parameterisation, optimiser or schedule."""
    model.train()
    opt = torch.optim.SGD(model.opt_groups(LR), momentum=0.9)
    wu  = min(WARMUP_EPOCHS, max(1, epochs//5))
    def _lr_lambda(e):
        if e < wu: return (e + 1)/wu
        return 0.5*(1 + math.cos(math.pi*(e - wu)/max(1, epochs - wu)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            x = X[idx].to(DEVICE); y = Y[idx].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y, label_smoothing=eps).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()
        sched.step()
    model.eval()
    return model

# ====================================================================== I/O
FIELDS = ["mode", "regime", "act", "width", "eps", "seedA", "seedB",
          "dnorm", "L_A", "L_B", "B", "t_star", "L_max",
          "rho_A", "rho_mid", "rho_max", "rho_at_tstar", "acc_mid",
          "flen_A", "flen_mid", "flen_B", "rq_mid", "R_end", "R_mid", "status"]

def _read_done():
    done = set()
    if RESUME and os.path.exists(OUT_CSV):
        import csv
        with open(OUT_CSV) as fh:
            for r in csv.DictReader(fh):
                done.add((r["regime"], r["act"], int(r["width"]),
                          float(r["eps"]), int(r["seedA"]), int(r["seedB"])))
    return done

def _append(row):
    import csv
    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new: w.writeheader()
        w.writerow(row)

def summarise():
    import csv, collections
    if not os.path.exists(OUT_CSV): return
    rows = list(csv.DictReader(open(OUT_CSV)))
    num = [f for f in FIELDS if f not in ("mode","regime","act","seedA","seedB","status")]
    g = collections.defaultdict(list)
    for r in rows:
        if r["status"] != "ok": continue
        g[(r["regime"], r["act"], r["width"], r["eps"])].append(r)
    with open(CELL_CSV, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["regime","act","width","eps","n_pairs"] + num)
        for k, v in sorted(g.items()):
            med = []
            for f in num:
                xs = [float(x[f]) for x in v if x.get(f) not in (None, "", "nan")]
                med.append(f"{np.median(xs):.6e}" if xs else "")
            w.writerow(list(k) + [len(v)] + med)
    log(f"wrote {CELL_CSV}  ({len(g)} cells)")

# ====================================================================== MAIN
def main():
    X, Y = load_data_xy()
    Xe, Ye = X[:EVAL_N].to(DEVICE), Y[:EVAL_N].to(DEVICE)   # same fixed eval subset
    Xf     = X[:FISHER_N].to(DEVICE)                        # same fixed Fisher batch
    ts_f = np.linspace(0.0, 1.0, TGRID_FINE)
    ts_c = np.linspace(0.0, 1.0, TGRID_COARSE)
    k_mid_f = TGRID_FINE//2; i_mid_c = TGRID_COARSE//2
    done = _read_done()
    log(f"device={DEVICE}  cells={len(REGIMES)*len(ACTS)*len(WIDTHS)*len(EPS_LIST)}  "
        f"already done pairs={len(done)}")

    for regime in REGIMES:
        for act in ACTS:
            for w in WIDTHS:
                for eps in EPS_LIST:
                    todo = [(i, j) for (i, j) in
                            list(itertools.combinations(range(NSEEDS), 2))[:PAIRS]
                            if (regime, act, w, eps, i, j) not in done]
                    if not todo:
                        log(f"skip  {regime}/{act}/w{w}/eps{eps}  (done)"); continue
                    t0 = time.time()
                    sds = []
                    for s in range(NSEEDS):
                        set_seed(SEED_BASE + 1000*s + w + int(1000*eps))
                        net = build_net(w, act, regime)
                        net = train(net, X, Y, EPOCHS, eps)
                        sds.append({k: v.detach().cpu().clone()
                                    for k, v in net.state_dict().items()})
                        del net
                        if DEVICE == "cuda": torch.cuda.empty_cache()
                    log(f"trained {regime}/{act}/w{w}/eps{eps}  {NSEEDS} seeds "
                        f"in {time.time()-t0:.0f}s")

                    ref = build_net(w, act, regime); ref.eval()
                    p_ref, b_ref = _pb(ref)
                    ag, gs = perm_spec(ref)
                    to_params = lambda sd: {k: sd[k].to(DEVICE) for k in p_ref.keys()}

                    for (i, j) in todo:
                        try:
                            perms = weight_matching(ag, gs, sds[i], sds[j],
                                                    iters=MATCH_ITERS, seed=i*13 + j)
                            pA = to_params(sds[i])
                            pB = to_params(apply_perm(sds[j], ag, perms))
                            delta = {k: pB[k] - pA[k] for k in pA}
                            dn = _vnorm(delta)

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

                            fl = []
                            for tt in ts_c:
                                pt = {k: (1-tt)*pA[k] + tt*pB[k] for k in pA}
                                f_, _ = flen_at(ref, pt, b_ref, Xf, delta, MICRO)
                                fl.append(f_)
                            _, rq_mid = flen_at(ref, p_mid, b_ref, Xf, delta, MICRO)
                            fl_A, fl_mid, fl_B = fl[0], fl[i_mid_c], fl[-1]
                            fl_end = 0.5*(fl_A + fl_B)
                            Rf = lambda f_: (B/(f_/4) if f_ > 0 else float("nan"))

                            _append(dict(
                                mode="mlp", regime=regime, act=act, width=w, eps=eps,
                                seedA=i, seedB=j,
                                dnorm=f"{dn:.6e}", L_A=f"{Ls[0]:.6e}", L_B=f"{Ls[-1]:.6e}",
                                B=f"{B:.6e}", t_star=f"{t_star:.4f}",
                                L_max=f"{Ls[k_star]:.6e}",
                                rho_A=f"{Rs[0]:.6e}", rho_mid=f"{Rs[k_mid_f]:.6e}",
                                rho_max=f"{Rs.max():.6e}", rho_at_tstar=f"{Rs[k_star]:.6e}",
                                acc_mid=f"{acc_mid:.4f}",
                                flen_A=f"{fl_A:.6e}", flen_mid=f"{fl_mid:.6e}",
                                flen_B=f"{fl_B:.6e}", rq_mid=f"{rq_mid:.6e}",
                                R_end=f"{Rf(fl_end):.4f}", R_mid=f"{Rf(fl_mid):.4f}",
                                status="ok"))
                            log(f"  {regime}/{act}/w{w}/eps{eps} pair {i}-{j}: "
                                f"B={B:.4e} rho_A={Rs[0]:.4e} flen_A={fl_A:.4e} "
                                f"R_end={Rf(fl_end):.3f}")
                        except Exception as e:
                            traceback.print_exc()
                            _append(dict({f: "" for f in FIELDS},
                                         mode="mlp", regime=regime, act=act, width=w,
                                         eps=eps, seedA=i, seedB=j,
                                         status=f"fail:{type(e).__name__}"))
                    del ref, sds
                    if DEVICE == "cuda": torch.cuda.empty_cache()
                    summarise()
    summarise()
    log("done")

if __name__ == "__main__":
    main()
