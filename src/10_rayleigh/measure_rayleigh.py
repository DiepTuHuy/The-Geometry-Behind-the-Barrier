#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
measure_rayleigh.py -- the Rayleigh quotient, end to end.   RAYLEIGH_V2

    python3 measure_rayleigh.py audit      # what exists, what is missing (no GPU)
    python3 measure_rayleigh.py selftest   # numerics vs a dense Fisher   (seconds)
    python3 measure_rayleigh.py measure    # measure ONLY the gaps       (GPU)
    python3 measure_rayleigh.py merge      # old + new -> canonical      (no GPU)
    python3 measure_rayleigh.py figures    # canonical -> PNG/PDF        (no GPU)
    python3 measure_rayleigh.py all        # audit, measure, merge, figures

Only `measure` needs a GPU or torch; torch is imported lazily so the other three
run on a laptop.

ON KAGGLE, SET THE COMMAND IN THE ENVIRONMENT, NOT ON THE COMMAND LINE:

      import os
      os.environ.update(RQ_CMD="measure", RQ_ARCH="cnn,ts",
                        RQ_DATA="/kaggle/input/<your-dataset>/data")

A notebook is executed by papermill, which puts its own parameter file on argv,
so `sys.argv[1]` arrives as something like /tmp/tmpXXXX.json.  `RQ_CMD` wins over
argv, and an argv entry that is not a known command is now ignored rather than
fatal.

MULTI-GPU IS ON BY DEFAULT.  A "GPU T4 x2" session uses both without being asked:
one process per visible device, split BY CELL and balanced BY COST (see
`split_tasks`).  Nothing crosses between processes, so the numbers are
bit-for-bit what one GPU would produce.  `RQ_GPUS=0` forces a single device.

WHAT THIS FILE IS FOR
---------------------
R_F(w) = d^T F(w) d  with  d = Delta/||Delta||, the Rayleigh quotient of the
Fisher metric along the displacement between two aligned minima.  The paper uses
it as the negative control of Figure 5, and reads R_F -> 0 as the displacement
"increasingly concentrating in the low-eigenvalue subspace of F".

That reading has two possible causes and R_F alone cannot separate them:

    directional   the spectrum stays put and d rotates down it
    global        every eigenvalue shrinks under a d that has not moved

The separator is an identity.  For v uniform on the unit sphere,

        E[v^T F v]  =  tr F / P                                          (*)

exactly.  So tr F / P is the value R_F would take if Delta carried no
directional information whatsoever, and the ALIGNMENT RATIO

        A(w)  =  R_F(w) / (tr F(w) / P)

is 1 when d is spectrally indistinguishable from a random direction, below 1
when it avoids the large eigenvalues, above 1 when it seeks them.  Since
R_F = (tr F/P) * A, the width exponents are additive,

        alpha_{R_F}  =  alpha_{tr F/P}  +  alpha_A

and "directional" is alpha_A > 0 while "global" is alpha_A ~ 0.

MEASURE THE GAP, NOT THE GRID
-----------------------------
Most of what a Rayleigh study needs was measured long ago and is already in this
repository.  Re-measuring it would spend GPU hours reproducing numbers to seven
decimal places: two independent runs of R_F on the same checkpoints agree to
7e-05, which is floating-point summation order (a different micro-batch size) and
nothing else.  So this file AUDITS first and measures only what is absent.

  quantity            already at                            coverage
  ------------------  ------------------------------------  --------------------
  R_F at t=0,1/2,1    data/geodesic/{arch}_pairs.csv        216 cells, 3 arch,
                                                            4 smooth acts
  lambda_max(F)       same file, as lam / lam_rel           216 cells
                      (02_geodesic power-iterated it to set
                       the CG damping and wrote out both)
  barrier B           data/train/{arch}_combined.csv        270 cells, incl. relu
  R_F(t), 21 points   data/profile/{arch}_length.csv        84 cells, MLP ONLY

  tr F / P            NOWHERE -- no trace or Hutchinson     0 cells
                      estimator exists in src/01..08
  R_F for relu        NOWHERE -- relu rode along with the   0 cells
                      Christoffel run, which needs C^3
  R_F(t), CNN and TS  NOWHERE                               0 cells

So the measurement phase computes tr F/P everywhere, R_F for relu, and the R_F(t)
profile for CNN and TS.  Everything else is read off disk.  R_F takes no
derivative of F -- one JVP and one VJP per micro-batch, no finite difference, no
conjugate gradient, no damping -- which is exactly why relu is admissible here
although the Christoffel run had to exclude it.

Note also that data/final/*_pairs.csv HAS rq_* and flen_* columns and they are
empty in all 840 released rows, so data/geodesic is the only source of R_F.

OUTPUT -- THE CONTRACT FOR REDRAWING FIGURES
--------------------------------------------
`merge` writes the canonical artefacts.  Every figure, here and in scripts/,
reads these and nothing else.  Each row carries `src_*` columns naming where the
number came from, so provenance never requires re-running anything.

  rayleigh_pairs.csv    one row per (arch, regime, act, width, seedA, seedB)
  rayleigh_profile.csv  one row per (pair, t)            -- long format
  rayleigh_cells.csv    one row per cell, with the fitted exponents
  rayleigh_summary.json machine-readable headline numbers
  fig_rayleigh_*.png/pdf  reading figures

CONVENTIONS.  `width` is stored RAW, as the source CSVs store it -- for the CNN
that is the channel MULTIPLIER in {1,2,4,8}, and the paper's width is 64x that;
`width_paper` carries the converted value so no figure has to know the rule.
Exponents follow the paper's sign: alpha > 0 means the quantity DECREASES with
width.

A CELL AGGREGATE OF A RATIO IS THE RATIO OF THE AGGREGATES.  The median does not
commute with division, so median(R_F/trP) and median(R_F)/median(trP) differ by
a few percent and the exponent decomposition stops adding up (measured: up to
0.032 on alpha_rq).  `align_mid_med` is the median of the per-pair ratios, which
is what to quote for a typical pair; `align_mid_rat` is the ratio of the medians
and is what alpha_A is fitted on, because only that makes the identity exact.

ENVIRONMENT
  RQ_ARCH      mlp | cnn | ts | all          (default all)
  RQ_REGIMES   e.g. "sp" or "ntk,mup"        (default all three)
  RQ_ACTS      e.g. "relu,gelu"              (default all five)
  RQ_WIDTHS    e.g. "2048,4096"              (default: the arch's grid)
  RQ_PAIRS     seed pairs per cell           (default 10 = C(5,2))
  RQ_TGRID     points in t for new profiles  (default 21, matching 04_profile)
  RQ_PROBES    unit-sphere probes for trP    (default 24)
  RQ_POWER     power iterations for lmax     (default 24)
  RQ_BATCH     Fisher batch                  (default 2048 -- do NOT change, the
                                              cross-check against the old runs
                                              needs the same batch)
  RQ_MICRO     fisher_vp micro-batch         (default: automatic)
  RQ_DATA      path to the repo's data/      (default: found from this file)
  RQ_OUT       output directory              (default /kaggle/working or .)
  RQ_CMD       audit | selftest | measure | merge | figures | all
  RQ_GPUS      "0,1" manual | "0" force one   (default: every visible GPU)
  RQ_SMOKE=1   tiny end-to-end rehearsal
  RQ_FORCE=1   re-measure even what the audit says exists
  RQ_NOTEST=1  skip the self-test
"""
import os
import sys
import csv
import json
import math
import glob
import time
import itertools
import traceback

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np

# torch is imported LAZILY, inside the measurement phase only, so audit, merge
# and figures run with no torch installed and no GPU present.
torch = None
nn = None
Fnn = None


# ==================================================================== CONFIG
def _env(name, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _env_list(name, default=None, cast=str):
    v = _env(name)
    return default if v is None else [cast(s.strip()) for s in v.split(",") if s.strip()]


def _flag(name, default="0"):
    return _env(name, default) == "1"


ALL_ARCHS = ["mlp", "cnn", "ts"]
_a = _env("RQ_ARCH", "all").lower()
ARCHS = ALL_ARCHS if _a == "all" else [s.strip() for s in _a.split(",")]
for _x in ARCHS:
    if _x not in ALL_ARCHS:
        raise SystemExit(f"RQ_ARCH invalid: {_x} (choose {ALL_ARCHS} or 'all')")

SMOOTH_ACTS = ["gelu", "tanh", "swish", "softplus"]
ALL_ACTS = ["relu"] + SMOOTH_ACTS
ACTS = _env_list("RQ_ACTS", ALL_ACTS)
REGIMES = _env_list("RQ_REGIMES", ["ntk", "sp", "mup"])

SMOKE = _flag("RQ_SMOKE")
FORCE = _flag("RQ_FORCE")

ARCH_WIDTHS = {"mlp": [64, 128, 256, 512, 1024, 2048, 4096],
               "cnn": [1, 2, 4, 8],
               "ts":  [64, 128, 256, 512, 1024, 2048, 4096]}
SMOKE_WIDTHS = {"mlp": [16, 32], "cnn": [1, 2], "ts": [16, 32]}
ARCH_TAG = {"mlp": "pmlp_v2", "cnn": "pcnn_v2", "ts": "pts_v2"}
ARCH_DIN_K = {"mlp": (784, 10), "cnn": (None, 10), "ts": (64, 10)}
CNN_CHANNELS_PER_WM = 64          # paper width = 64 * channel multiplier

N_PAIRS = int(_env("RQ_PAIRS", 1 if SMOKE else 10))
TGRID = int(_env("RQ_TGRID", 5 if SMOKE else 21))     # 21 = the 04_profile grid
NPROBE = int(_env("RQ_PROBES", 6 if SMOKE else 24))
POWER_ITERS = int(_env("RQ_POWER", 8 if SMOKE else 24))
RQ_BATCH = int(_env("RQ_BATCH", 128 if SMOKE else 2048))
NSEEDS = 2 if SMOKE else 5
LAM_REL_MAIN = 1e-2               # the damping level every released figure uses

OUT_DIR = _env("RQ_OUT") or ("/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
CKPT_ROOTS = [".", "/kaggle/input", "/content", "/content/drive/MyDrive"]


def _find_data_dir():
    """The repo's data/ directory, however this file was invoked."""
    if _env("RQ_DATA"):
        return _env("RQ_DATA")
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = "."
    for c in (os.path.join(here, "..", "..", "data"), "./data",
              os.path.join(OUT_DIR, "data")):
        if os.path.isdir(os.path.normpath(c)):
            return os.path.normpath(c)
    for c in sorted(glob.glob("/kaggle/input/**/geodesic", recursive=True)):
        return os.path.dirname(c)
    return None


DATA_DIR = _find_data_dir()


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def smooth_of(act):
    return 1 if act in SMOOTH_ACTS else 0


def width_paper(arch, w):
    return int(w) * CNN_CHANNELS_PER_WM if arch == "cnn" else int(w)


# ======================================================================= CSV
# No pandas anywhere: the measurement phase runs on Kaggle images where a pandas
# import is pure overhead, and merge has to run wherever the user happens to be.
def read_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, cols, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def append_csv(path, cols, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()


def num(x, default=float("nan")):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def is_ok(row):
    return str(row.get("status", "ok")).startswith("ok")


def median(xs):
    xs = sorted(x for x in xs if x is not None and math.isfinite(x))
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def quantile(xs, q):
    xs = sorted(x for x in xs if x is not None and math.isfinite(x))
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


# ================================================================ OLD SOURCES
def old_geodesic(arch):
    """R_F at t = 0, 1/2, 1 and lambda_max, per aligned seed pair.

    lambda_max is not stored directly.  02_geodesic power-iterated it only to set
    the CG damping lam = lam_rel * lambda_max, and wrote out both the product and
    the factor -- so the eigenvalue divides straight back out.  Only lam_rel =
    1e-2 rows are taken: that is the damping level every released figure uses."""
    out = []
    for r in read_csv(os.path.join(DATA_DIR or "", "geodesic", f"{arch}_pairs.csv")):
        if not is_ok(r):
            continue
        lr = num(r.get("lam_rel"))
        if math.isfinite(lr) and abs(lr - LAM_REL_MAIN) > 1e-12:
            continue
        lam = num(r.get("lam"))
        lmax = lam / lr if (math.isfinite(lam) and math.isfinite(lr) and lr > 0) else float("nan")
        out.append(dict(arch=arch, regime=r.get("regime"), act=r.get("act"),
                        width=int(num(r.get("width"), 0)),
                        seedA=int(num(r.get("seedA"), -1)),
                        seedB=int(num(r.get("seedB"), -1)),
                        dnorm=num(r.get("dnorm")), rq_A=num(r.get("rq_A")),
                        rq_mid=num(r.get("rq_mid")), rq_B=num(r.get("rq_B")),
                        lmax_A=lmax))
    return out


def old_profile(arch):
    """R_F(t) on the 21-point grid, reshaped long.  MLP only in this repo.

    The file is stored WIDE -- one rq_t0.000 ... rq_t1.000 column per grid point
    -- because that is how 04_profile wrote it."""
    out = []
    for r in read_csv(os.path.join(DATA_DIR or "", "profile", f"{arch}_length.csv")):
        if not is_ok(r):
            continue
        for k, v in r.items():
            if not k.startswith("rq_t"):
                continue
            t, y = num(k[len("rq_t"):]), num(v)
            if math.isfinite(t) and math.isfinite(y):
                out.append(dict(arch=arch, regime=r.get("regime"), act=r.get("act"),
                                width=int(num(r.get("width"), 0)),
                                seedA=int(num(r.get("seedA"), -1)),
                                seedB=int(num(r.get("seedB"), -1)),
                                t=round(t, 4), rq_t=y, src="04_profile"))
    return out


def old_barrier(arch):
    """Median barrier per cell, from data/train.

    data/train covers all five activations where data/final covers only the four
    smooth ones -- and data/final's rq_*/flen_* columns are empty in every
    released row, so it carries no Rayleigh information at all."""
    acc = {}
    for r in read_csv(os.path.join(DATA_DIR or "", "train", f"{arch}_combined.csv")):
        if r.get("kind") != "pair" or not is_ok(r):
            continue
        b = num(r.get("barrier"))
        if math.isfinite(b):
            acc.setdefault((r.get("regime"), r.get("act"),
                            int(num(r.get("width"), 0))), []).append(b)
    return {k: median(v) for k, v in acc.items()}


# ====================================================================== AUDIT
def cell_grid(arch):
    widths = _env_list("RQ_WIDTHS", (SMOKE_WIDTHS if SMOKE else ARCH_WIDTHS)[arch], cast=int)
    return [(r, a, w) for r in REGIMES for a in ACTS for w in widths]


def audit(verbose=True):
    """What exists per cell, and therefore what has to be measured.

    Returns {(arch, regime, act, width): {"rq","lmax","prof","trP" -> bool}}."""
    prev = {(r["arch"], r["regime"], r["act"], int(num(r["width"], 0)))
            for r, _src in trP_sources()
            if math.isfinite(num(r.get("trP_mid")))}
    have = {}
    for arch in ARCHS:
        geo = old_geodesic(arch)
        rq_cells = {(r["regime"], r["act"], r["width"]) for r in geo
                    if math.isfinite(r["rq_mid"])}
        lm_cells = {(r["regime"], r["act"], r["width"]) for r in geo
                    if math.isfinite(r["lmax_A"])}
        pf_cells = {(r["regime"], r["act"], r["width"]) for r in old_profile(arch)}
        for (r, a, w) in cell_grid(arch):
            have[(arch, r, a, w)] = dict(rq=(r, a, w) in rq_cells,
                                         lmax=(r, a, w) in lm_cells,
                                         prof=(r, a, w) in pf_cells,
                                         trP=(arch, r, a, w) in prev)
    if verbose:
        _audit_report(have)
    return have


def gap_tasks(have):
    """{cell: set of quantities still to measure}.  RQ_FORCE=1 marks all."""
    out = {}
    for key, h in have.items():
        want = set()
        if FORCE or not h["trP"]:
            want.add("trP")
        if FORCE or not h["rq"]:
            want.add("rq")
        if FORCE or not h["prof"]:
            want.add("profile")
        if want:
            out[key] = want
    return out


def _audit_report(have):
    from collections import Counter
    print("=" * 78)
    print("AUDIT -- what already exists, and what this run has to measure")
    print("=" * 78)
    print(f"  data/ resolved to: {DATA_DIR}")
    if DATA_DIR is None:
        print("  !! data/ NOT FOUND -- nothing can be reused, every cell is a gap.")
        print("     Set RQ_DATA=/path/to/data or stage it under /kaggle/input.")
    hdr = f"  {'arch':5s} {'cells':>6s} {'R_F':>11s} {'lmax':>11s} {'R_F(t)':>11s} {'trF/P':>11s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for arch in ARCHS:
        ks = [k for k in have if k[0] == arch]
        n = len(ks) or 1

        def pct(f, ks=ks, n=n):
            return f"{sum(1 for k in ks if have[k][f]):4d}/{n:<5d}"
        print(f"  {arch:5s} {len(ks):6d} {pct('rq'):>11s} {pct('lmax'):>11s} "
              f"{pct('prof'):>11s} {pct('trP'):>11s}")
    gaps = gap_tasks(have)
    print()
    if not gaps:
        print("  -> nothing to measure; every quantity is on disk.  Run `merge`.")
        return gaps
    print(f"  -> {len(gaps)} cells need GPU work:")
    need = {}
    for (arch, _r, _a, _w), what in gaps.items():
        need.setdefault(", ".join(sorted(what)), []).append(arch)
    for what, archs in sorted(need.items()):
        c = Counter(archs)
        print(f"       {what:26s}  " + "  ".join(f"{k}:{v}" for k, v in sorted(c.items())))
    # Cost, in the only unit this file has: one fisher_vp is one pass of
    # RQ_BATCH samples through a jvp and a vjp.  Everything else is vector
    # arithmetic.  Endpoint anchors are per SEED, the rest per PAIR.
    per_seed = POWER_ITERS + NPROBE
    total = 0
    for _key, want in gaps.items():
        c = NSEEDS * per_seed if "trP" in want else 0        # endpoint anchors
        c += N_PAIRS * (POWER_ITERS + NPROBE) if "trP" in want else 0   # midpoints
        if "profile" in want:
            c += N_PAIRS * TGRID
        elif "rq" in want:
            c += N_PAIRS * 3
        total += c
    print()
    print(f"  cost ~{total:,} fisher_vp at batch {RQ_BATCH}.")
    print("  For scale: measuring the full grid from scratch, ignoring what is")
    print("  already on disk, would be roughly 4x that -- the audit is the")
    print("  cheapest optimisation in this file.")
    print()
    print("  Everything else is reused, not re-measured: two independent runs of")
    print("  R_F on these checkpoints agree to 7e-05 (float summation order), so")
    print("  re-measuring buys precision that does not exist.")
    return gaps


# ==================================================================== MEASURE
def _import_torch():
    global torch, nn, Fnn
    if torch is not None:
        return
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch as _t
    import torch.nn as _nn
    import torch.nn.functional as _F
    torch, nn, Fnn = _t, _nn, _F


BASE = 64


def build_kit():
    """Model, permutation and Fisher primitives.

    Built inside a function so importing this file costs no torch.  The
    architecture is byte-for-byte src/01_train's -- this file reads those
    checkpoints, and one changed scale factor would make every number here
    incomparable with the released ones."""
    _import_torch()
    from torch.func import functional_call, jvp as _fjvp, vjp as _fvjp, jacrev as _jacrev
    from scipy.optimize import linear_sum_assignment
    F = Fnn

    def make_act(n):
        return {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh,
                "swish": nn.SiLU, "softplus": nn.Softplus}[n]()

    def param_cfg(regime, fin, fout, kind):
        ss = math.sqrt(fin)
        if regime == "sp":
            return (1.0 / ss, 1.0, 1.0)
        if regime == "ntk":
            return (1.0, 1.0 / ss, 1.0)
        if regime == "mup":
            if kind == "input":
                return (1.0 / ss, 1.0, (fout / BASE) ** 1.0)
            if kind == "hidden":
                return (1.0 / ss, 1.0, (fin / BASE) ** 0.7)
            return (1.0 / ss, (BASE / fin) ** 0.5, (fin / BASE) ** (-0.5))
        raise ValueError(regime)

    class ScaledLinear(nn.Module):
        def __init__(self, fin, fout, regime, kind):
            super().__init__()
            istd, self.fmul, self.lr_scale = param_cfg(regime, fin, fout, kind)
            self.weight = nn.Parameter(torch.randn(fout, fin) * istd)
            self.bias = nn.Parameter(torch.zeros(fout))

        def forward(self, x):
            return self.fmul * F.linear(x, self.weight) + self.bias

    class ScaledConv(nn.Module):
        def __init__(self, cin, cout, k, st, pad, regime, kind):
            super().__init__()
            fin = cin * k * k
            istd, self.fmul, self.lr_scale = param_cfg(regime, fin, cout, kind)
            self.weight = nn.Parameter(torch.randn(cout, cin, k, k) * istd)
            self.st, self.pad = st, pad

        def forward(self, x):
            return self.fmul * F.conv2d(x, self.weight, None, self.st, self.pad)

    class NetMLP(nn.Module):
        def __init__(self, width, act, regime, din, k):
            super().__init__()
            self.fc1 = ScaledLinear(din, width, regime, "input")
            self.fc2 = ScaledLinear(width, width, regime, "hidden")
            self.fc3 = ScaledLinear(width, k, regime, "output")
            self.a1, self.a2 = make_act(act), make_act(act)

        def forward(self, x):
            return self.fc3(self.a2(self.fc2(self.a1(self.fc1(x)))))

    class NetCNN(nn.Module):
        def __init__(self, wm, act, regime, in_ch, k):
            super().__init__()
            c = [16 * wm, 32 * wm, 64 * wm]
            self.c1 = ScaledConv(in_ch, c[0], 3, 1, 1, regime, "input")
            self.n1, self.a1 = nn.GroupNorm(1, c[0]), make_act(act)
            self.c2 = ScaledConv(c[0], c[1], 3, 1, 1, regime, "hidden")
            self.n2, self.a2 = nn.GroupNorm(1, c[1]), make_act(act)
            self.c3 = ScaledConv(c[1], c[2], 3, 1, 1, regime, "hidden")
            self.n3, self.a3 = nn.GroupNorm(1, c[2]), make_act(act)
            self.pool = nn.MaxPool2d(2)
            self.fc = ScaledLinear(c[2], k, regime, "output")

        def forward(self, x):
            h1 = self.pool(self.a1(self.n1(self.c1(x))))
            h2 = self.pool(self.a2(self.n2(self.c2(h1))))
            h3 = self.a3(self.n3(self.c3(h2)))
            return self.fc(F.adaptive_avg_pool2d(h3, 1).flatten(1))

    def build_net(arch, width, act, regime):
        din, k = ARCH_DIN_K[arch]
        return (NetCNN(width, act, regime, 1, k) if arch == "cnn"
                else NetMLP(width, act, regime, din, k))

    def perm_spec(arch, model):
        if arch == "cnn":
            ag = {"c1.weight": ["g1", None, None, None], "n1.weight": ["g1"], "n1.bias": ["g1"],
                  "c2.weight": ["g2", "g1", None, None], "n2.weight": ["g2"], "n2.bias": ["g2"],
                  "c3.weight": ["g3", "g2", None, None], "n3.weight": ["g3"], "n3.bias": ["g3"],
                  "fc.weight": [None, "g3"], "fc.bias": [None]}
        else:
            ag = {"fc1.weight": ["h1", None], "fc1.bias": ["h1"],
                  "fc2.weight": ["h2", "h1"], "fc2.bias": ["h2"],
                  "fc3.weight": [None, "h2"], "fc3.bias": [None]}
        sd = model.state_dict()
        gs = {}
        for n, axes in ag.items():
            for a, g in enumerate(axes):
                if g is not None:
                    gs[g] = sd[n].shape[a]
        return ag, gs

    def _perm_except(t, axes, perms, exc):
        tt = t
        for a, g in enumerate(axes):
            if g is not None and a != exc:
                tt = tt.index_select(a, perms[g])
        return tt

    def apply_perm(sd, ag, perms):
        out = {}
        for n, t in sd.items():
            if n in ag:
                tt = t
                for a, g in enumerate(ag[n]):
                    if g is not None:
                        tt = tt.index_select(a, perms[g])
                out[n] = tt.clone()
            else:
                out[n] = t.clone()
        return out

    def weight_matching(ag, gs, sdA, sdB, iters=8, seed=0):
        """Identical to 02_geodesic, RNG stream included: the cross-check
        compares R_F pair by pair, so Delta must be the SAME Delta."""
        rng = np.random.RandomState(seed)
        perms = {g: torch.arange(n) for g, n in gs.items()}
        g2pa = {g: [] for g in gs}
        for n, axes in ag.items():
            for a, g in enumerate(axes):
                if g is not None:
                    g2pa[g].append((n, a))
        groups = list(gs)
        for _ in range(iters):
            moved = 0
            for g in [groups[i] for i in rng.permutation(len(groups))]:
                n = gs[g]
                S = torch.zeros(n, n, dtype=torch.float64)
                for (name, axis) in g2pa[g]:
                    A = sdA[name].double()
                    Bm = _perm_except(sdB[name].double(), ag[name], perms, axis)
                    S += (torch.movedim(A, axis, 0).reshape(n, -1)
                          @ torch.movedim(Bm, axis, 0).reshape(n, -1).T)
                new = torch.as_tensor(linear_sum_assignment(-S.numpy())[1], dtype=torch.long)
                if not torch.equal(new, perms[g]):
                    moved += 1
                perms[g] = new
            if moved == 0:
                break
        return perms

    def pb(m):
        return ({k: v.detach() for k, v in m.named_parameters()},
                {k: v.detach() for k, v in m.named_buffers()})

    def call(m, p, b, x):
        return functional_call(m, {**p, **b}, (x,))

    def vdot(a, b):
        return float(sum((a[k] * b[k]).sum() for k in a))

    def vnorm(a):
        return math.sqrt(max(vdot(a, a), 0.0))

    def vscale(a, c):
        return {k: a[k] * c for k in a}

    def fisher_vp(m, p, b, x, v, micro):
        """F v, with F = E_x J^T S J and S = diag(pr) - pr pr^T (MODEL Fisher).

        Accumulated over micro-batches and divided by the batch at the end, so
        the value does not depend on `micro` beyond summation order."""
        B = x.shape[0]
        acc = None
        for i in range(0, B, micro):
            xb = x[i:i + micro]

            def f(pp):
                return call(m, pp, b, xb)

            logits, Jv = _fjvp(f, (p,), (v,))
            pr = torch.softmax(logits, 1)
            s = pr * Jv - pr * (pr * Jv).sum(1, keepdim=True)
            JTs = _fvjp(f, p)[1](s)[0]
            acc = ({k: JTs[k].detach() for k in JTs} if acc is None
                   else {k: acc[k] + JTs[k].detach() for k in acc})
        return {k: acc[k] / B for k in acc}

    def fisher_quad(m, p, b, x, v, micro):
        return vdot(v, fisher_vp(m, p, b, x, v, micro))

    def power_top(m, p, b, x, micro, iters, seed=17):
        """(lmax, u_1, gap).

        lmax is the Rayleigh quotient of the final iterate rather than ||F u||:
        for a symmetric PSD operator it converges at twice the rate and is a
        lower bound, so an lmax that falls short is honest rather than
        optimistic.  `gap` is the last relative change -- a cell where it is not
        small has not resolved its top eigenvalue."""
        gen = torch.Generator(device=x.device).manual_seed(seed)
        u = {k: torch.randn(v.shape, generator=gen, device=v.device, dtype=v.dtype)
             for k, v in p.items()}
        u = vscale(u, 1.0 / max(vnorm(u), 1e-30))
        lam = prev = 0.0
        for _ in range(iters):
            Au = fisher_vp(m, p, b, x, u, micro)
            prev, lam = lam, vdot(u, Au)
            nrm = vnorm(Au)
            if nrm < 1e-30:
                break
            u = vscale(Au, 1.0 / nrm)
        return lam, u, abs(lam - prev) / max(abs(lam), 1e-30)

    def sphere_probes(m, p, b, x, micro, nprobe, seed=23):
        """(mean, sem, n) of v^T F v over v uniform on the unit sphere.

        E[v^T F v] = tr F / P exactly, so the mean is an unbiased estimate of the
        MEAN EIGENVALUE of F -- and is exactly what R_F would be if Delta carried
        no directional information.  The s.e.m. is written out: the alignment
        ratio is a ratio of two measured numbers and its error bar has to come
        from somewhere."""
        gen = torch.Generator(device=x.device).manual_seed(seed)
        vals = []
        for _ in range(nprobe):
            v = {k: torch.randn(t.shape, generator=gen, device=t.device, dtype=t.dtype)
                 for k, t in p.items()}
            v = vscale(v, 1.0 / max(vnorm(v), 1e-30))
            vals.append(fisher_quad(m, p, b, x, v, micro))
        a = np.asarray(vals, float)
        sem = float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else float("nan")
        return float(a.mean()), sem, len(a)

    return dict(build_net=build_net, perm_spec=perm_spec, apply_perm=apply_perm,
                weight_matching=weight_matching, pb=pb, call=call, vdot=vdot,
                vnorm=vnorm, vscale=vscale, fisher_vp=fisher_vp,
                fisher_quad=fisher_quad, power_top=power_top,
                sphere_probes=sphere_probes, jacrev=_jacrev, NetMLP=NetMLP)


def act_units_per_sample(arch, width, din, k):
    if arch == "cnn":
        return 28 * 28 * 16 * width + 14 * 14 * 32 * width + 7 * 7 * 64 * width
    return (din or 0) + 2 * width + (k or 0)


def pick_micro(arch, width):
    """The largest micro-batch activation memory allows.

    One fisher_vp's total FLOP does not depend on micro -- the whole batch is
    always consumed -- so splitting only adds per-chunk overhead.  Budget by
    activation bytes per sample, not parameter count: for the CNN those two
    disagree badly (few parameters, very large activations)."""
    if _env("RQ_MICRO"):
        return max(1, min(int(_env("RQ_MICRO")), RQ_BATCH))
    din, k = ARCH_DIN_K[arch]
    per = max(act_units_per_sample(arch, width, din, k) * 6 * 4, 1)
    m = min(max(int(float(_env("RQ_ACT_BUDGET", 2.0e9)) // per), 8), RQ_BATCH)
    return 1 << int(math.floor(math.log2(m)))


_DATA_CACHE = {}


def load_fisher_batch(arch):
    """The Fisher batch: same source, normalisation and size as 02_geodesic and
    04_profile.  F is an expectation over inputs, so no labels are needed."""
    if arch in _DATA_CACHE:
        return _DATA_CACHE[arch]
    if arch == "mlp":
        import torchvision
        ds = torchvision.datasets.MNIST("./data", train=True, download=True)
        X = (((ds.data.float() / 255.0) - 0.1307) / 0.3081).reshape(-1, 784)
    elif arch == "cnn":
        import torchvision
        ds = torchvision.datasets.FashionMNIST("./data", train=True, download=True)
        X = (((ds.data.float() / 255.0) - 0.2860) / 0.3530).unsqueeze(1)
    else:
        g = torch.Generator().manual_seed(1)
        X = torch.randn(20000, ARCH_DIN_K["ts"][0], generator=g)
    _DATA_CACHE[arch] = X
    return X


def prefetch_data(archs):
    """Download any dataset the run needs, ONCE, in the parent.

    Workers are split by cost, not by architecture, so two of them can both be
    handed CNN cells -- and `load_fisher_batch` would then have both call
    torchvision with download=True on the same `./data` at the same moment.
    Two processes writing one archive is a corrupted archive.  So the parent
    touches every dataset first and the workers all find it on disk.

    TS is synthetic and needs nothing; MNIST and FashionMNIST are ~10 s each
    when already cached, so this is cheap even when it is redundant."""
    for arch in sorted(set(archs)):
        if arch == "ts":
            continue
        try:
            import torchvision
            ds = (torchvision.datasets.MNIST if arch == "mlp"
                  else torchvision.datasets.FashionMNIST)
            ds("./data", train=True, download=True)
            log(f"[prefetch] {arch} dataset ready")
        except Exception as e:
            log(f"[prefetch] {arch} failed ({e!r}) -- workers will each try")


def find_ckpt(arch, regime, act, w, s):
    name = f"{regime}_{act}_w{w}_s{s}.pt"
    tag = ARCH_TAG[arch]
    local = os.path.join(OUT_DIR, f"ckpt_{tag}", name)
    if os.path.exists(local):
        return local
    for root in CKPT_ROOTS:
        hits = glob.glob(os.path.join(root, "**", f"ckpt_{tag}", name), recursive=True)
        if hits:
            return hits[0]
    return None


def load_sd(path):
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        d = torch.load(path, map_location="cpu")
    return d["sd"], d.get("acc")


NEW_PAIR_COLS = ["arch", "regime", "act", "width", "smooth", "seedA", "seedB",
                 "nparams", "dnorm", "rq_A", "rq_mid", "rq_B",
                 "lmax_A", "lmax_mid", "lmax_B",
                 "trP_A", "trP_mid", "trP_B",
                 "trP_sem_A", "trP_sem_mid", "trP_sem_B",
                 "ov1_A", "ov1_mid", "ov1_B", "pow_gap_mid",
                 "batch", "micro", "probes", "power_iters", "status"]
NEW_PROF_COLS = ["arch", "regime", "act", "width", "seedA", "seedB", "t", "rq_t",
                 "dnorm", "status"]


def new_pairs_path():
    return os.path.join(OUT_DIR, "rayleigh_new_pairs.csv")


def new_prof_path():
    return os.path.join(OUT_DIR, "rayleigh_new_profile.csv")


def trP_sources():
    """Every table that carries tr F/P, this session's and any earlier one.

    An earlier run of this file may already be committed under data/rayleigh/,
    and its probes are as good as today's -- tr F/P is the expensive quantity
    here, so orphaning it would mean paying for it twice.  Rows are keyed by
    (arch, regime, act, width, seedA, seedB), so a cell measured twice simply
    resolves to one row and the later file wins."""
    out = []
    for arch in ARCHS:
        for r in read_csv(os.path.join(DATA_DIR or "", "rayleigh", f"{arch}_pairs.csv")):
            # v1 of this file named the architecture column `mode`.
            r.setdefault("arch", r.get("mode", arch))
            if is_ok(r) and math.isfinite(num(r.get("trP_mid"))):
                out.append((r, "data/rayleigh"))
    out += [(r, "this run") for r in read_csv(new_pairs_path()) if is_ok(r)]
    return out


def profile_sources():
    """Every long-format R_F(t) table other than 04_profile's."""
    out = []
    for arch in ARCHS:
        for r in read_csv(os.path.join(DATA_DIR or "", "rayleigh", f"{arch}_profile.csv")):
            r.setdefault("arch", r.get("mode", arch))
            if is_ok(r):
                out.append(r)
    out += [r for r in read_csv(new_prof_path()) if is_ok(r)]
    return out


def selftest():
    """Validate the numerics against a DENSE Fisher built by explicit Jacobian,
    in float64 on a net small enough to materialise:

      1. fisher_vp v  ==  F v                  the operator is the right one
      2. power_top    ==  eigvalsh(F).max()    the top of the spectrum
      3. mean of sphere probes -> tr F / P     identity (*), which the whole
                                               alignment ratio rests on
      4. R_F          ==  d^T F d              the read-out itself
    """
    K = build_kit()
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        m = K["NetMLP"](6, "tanh", "ntk", 4, 3).eval()
        x = torch.randn(8, 4)
        p, b = K["pb"](m)
        keys = list(p)

        def flat(d):
            return torch.cat([d[k].reshape(-1) for k in keys])

        def unflat(v):
            o, i = {}, 0
            for k in keys:
                n = p[k].numel()
                o[k] = v[i:i + n].reshape(p[k].shape)
                i += n
            return o

        def Fmat():
            J = K["jacrev"](lambda q: K["call"](m, q, b, x))(p)
            B = x.shape[0]
            Jf = torch.cat([J[k].reshape(B, 3, -1) for k in keys], 2)
            pr = torch.softmax(K["call"](m, p, b, x), 1)
            S = torch.diag_embed(pr) - pr.unsqueeze(2) * pr.unsqueeze(1)
            return torch.einsum('bki,bkl,blj->ij', Jf, S, Jf) / B

        Fd = Fmat()
        P = flat(p).numel()
        torch.manual_seed(3)
        df = torch.randn(P)
        df = df / df.norm() * 3.0
        delta = unflat(df)

        got = flat(K["fisher_vp"](m, p, b, x, delta, 8))
        want = Fd @ df
        e1 = float((got - want).norm() / max(float(want.norm()), 1e-300))
        assert e1 < 1e-10, f"fisher_vp wrong by {e1:.2e}"

        ev = torch.linalg.eigvalsh(Fd)
        lmax_p, _u, _g = K["power_top"](m, p, b, x, 8, 400, 5)
        e2 = abs(lmax_p - float(ev.max())) / max(abs(float(ev.max())), 1e-300)
        assert e2 < 1e-6, f"power_top wrong by {e2:.2e}"

        trP_d = float(ev.sum()) / P
        mean, sem, n = K["sphere_probes"](m, p, b, x, 8, 4000, 11)
        z = abs(mean - trP_d) / max(sem, 1e-300)
        assert z < 5.0, f"probe mean {mean:.6e} vs tr F/P {trP_d:.6e} ({z:.1f} sigma)"

        d2 = float(df.dot(df))
        rq = K["fisher_quad"](m, p, b, x, delta, 8) / d2
        rq_d = float(df @ Fd @ df) / d2
        e4 = abs(rq - rq_d) / max(abs(rq_d), 1e-300)
        assert e4 < 1e-10, f"R_F wrong by {e4:.2e}"

        log(f"[selftest] OK   Fv {e1:.1e}   lmax {e2:.1e}   "
            f"trF/P {z:.2f} sigma (n={n})   R_F {e4:.1e}")
    finally:
        torch.set_default_dtype(old)


def smoke_train(K, arch, regime, act, w, seed):
    """Train a tiny net so RQ_SMOKE=1 exercises the NUMERICS, not just the
    plumbing.  Without this a box with no checkpoints skips every cell and the
    rehearsal proves nothing -- which is the exact failure mode src/09_scale's
    docstring warns about.  Never used in a real run."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    m = K["build_net"](arch, w, act, regime).train()
    if arch == "ts":
        g = torch.Generator().manual_seed(1)
        X = torch.randn(1500, ARCH_DIN_K["ts"][0], generator=g)
        torch.manual_seed(1234)
        teach = K["build_net"]("ts", 32, "relu", "sp").eval()
        with torch.no_grad():
            Y = teach(X).argmax(1)
    else:
        import torchvision
        ds = (torchvision.datasets.MNIST if arch == "mlp"
              else torchvision.datasets.FashionMNIST)("./data", train=True, download=True)
        if arch == "mlp":
            X = (((ds.data.float() / 255.0) - 0.1307) / 0.3081).reshape(-1, 784)[:1500]
        else:
            X = (((ds.data.float() / 255.0) - 0.2860) / 0.3530).unsqueeze(1)[:1500]
        Y = ds.targets[:1500]
    opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
    for _ in range(2):
        perm = torch.randperm(X.shape[0])
        for i in range(0, X.shape[0], 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            Fnn.cross_entropy(m(X[idx]), Y[idx]).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    m.eval()
    d = os.path.join(OUT_DIR, f"ckpt_{ARCH_TAG[arch]}")
    os.makedirs(d, exist_ok=True)
    torch.save({"sd": {k: v.detach().cpu().clone() for k, v in m.state_dict().items()},
                "acc": 0.0}, os.path.join(d, f"{regime}_{act}_w{w}_s{seed}.pt"))


def measure(have=None, tasks=None, out_pair=None, out_prof=None):
    """Measure the gaps the audit found.  Resumable at pair granularity.

    The parent splits the work across every visible GPU and spawns one process
    per device; a worker (RQ_WORKER set) runs only the cells it was handed."""
    if WORKER_ID is None:
        have = audit(verbose=False) if have is None else have
        tasks = gap_tasks(have) if tasks is None else tasks
        if not tasks:
            log("nothing to measure -- the audit found every quantity on disk")
            return
        gpus = resolve_gpus()
        path = self_path() if len(gpus) > 1 else None
        if len(gpus) > 1 and path:
            log(f"[multi-gpu] {len(gpus)} GPUs, {len(tasks)} cells, split by cost")
            prefetch_data({k[0] for k in tasks})
            ok = spawn_workers(gpus, tasks, path)
            merge_worker_files()
            if not ok:
                log("!! a worker failed -> rerun to RESUME what is missing")
            return
        if len(gpus) > 1:
            log(f"!! could not spawn workers -> falling back to one GPU "
                f"(still correct, about {len(gpus)}x slower)")
    else:
        tasks = dec_tasks(WORKER_TASKS or "")
        log(f"[worker {WORKER_ID}] pid={os.getpid()} {len(tasks)} cells")

    K = build_kit()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    out_pair = out_pair or new_pairs_path()
    out_prof = out_prof or new_prof_path()
    if WORKER_ID is not None:
        out_pair, out_prof = worker_paths(WORKER_ID)
    # A worker must also see progress already merged into the main file, or a
    # rerun would redo everything the previous session finished.
    read_pair = [out_pair] if out_pair == new_pairs_path() else [out_pair,
                                                                new_pairs_path()]
    done = {(r["arch"], r["regime"], r["act"], int(num(r["width"], 0)),
             int(num(r["seedA"], -1)), int(num(r["seedB"], -1)))
            for q in read_pair for r in read_csv(q) if is_ok(r)}
    log(f"MEASURE {len(tasks)} cells on {DEVICE}  batch={RQ_BATCH} probes={NPROBE} "
        f"power={POWER_ITERS} tgrid={TGRID}  (already done: {len(done)} pairs)")

    by_arch = {}
    for key, want in sorted(tasks.items()):
        by_arch.setdefault(key[0], []).append((key, want))

    for arch, items in by_arch.items():
        Xf = load_fisher_batch(arch)[:RQ_BATCH].to(DEVICE)
        for (key, want) in items:
            _, regime, act, w = key
            try:
                sds, accs = [], []
                for s in range(NSEEDS):
                    cp = find_ckpt(arch, regime, act, w, s)
                    if cp is None and SMOKE:
                        smoke_train(K, arch, regime, act, w, s)
                        cp = find_ckpt(arch, regime, act, w, s)
                    if cp is None:
                        sds.append(None)
                        accs.append(None)
                        continue
                    sd, acc = load_sd(cp)
                    sds.append(sd)
                    accs.append(acc)
                avail = [s for s in range(NSEEDS) if sds[s] is not None]
                if len(avail) < 2:
                    append_csv(out_pair, NEW_PAIR_COLS,
                               dict(arch=arch, regime=regime, act=act, width=w,
                                    smooth=smooth_of(act), status="skip:nockpt"))
                    log(f"  [{arch}/{regime}/{act}/w{w}] <2 checkpoints -> skip")
                    continue

                ref = K["build_net"](arch, w, act, regime).to(DEVICE).eval()
                p_ref, b_ref = K["pb"](ref)
                nP = sum(v.numel() for v in p_ref.values())
                micro = pick_micro(arch, w)
                ag, gs = K["perm_spec"](arch, K["build_net"](arch, w, act, regime))
                log(f"=== {arch}/{regime}/{act}/w{w}  P={nP/1e6:.3f}M  micro={micro}  "
                    f"need={sorted(want)} ===")

                def to_params(sd):
                    return {kk: sd[kk].to(DEVICE) for kk in p_ref.keys()}

                # Endpoint anchors are per SEED, not per pair -- cache them.
                anchors = {}
                for s in avail:
                    ps = to_params(sds[s])
                    lmax, u1, gap = K["power_top"](ref, ps, b_ref, Xf, micro, POWER_ITERS)
                    trP, sem, _ = K["sphere_probes"](ref, ps, b_ref, Xf, micro, NPROBE)
                    anchors[s] = (lmax, trP, sem, u1)
                    log(f"  [seed {s}] lmax={lmax:.3e} (gap {gap:.1e})  "
                        f"trF/P={trP:.3e} +- {sem:.1e}")

                for (i, j) in list(itertools.combinations(avail, 2))[:N_PAIRS]:
                    if (arch, regime, act, w, i, j) in done and not FORCE:
                        continue
                    perms = K["weight_matching"](ag, gs, sds[i], sds[j], 8, i * 13 + j)
                    sdB = K["apply_perm"](sds[j], ag, perms)
                    pA, pB = to_params(sds[i]), to_params(sdB)
                    delta = {kk: pB[kk] - pA[kk] for kk in pA}
                    dn = K["vnorm"](delta)
                    d2 = max(dn * dn, 1e-300)
                    dhat = K["vscale"](delta, 1.0 / max(dn, 1e-300))
                    pmid = {kk: 0.5 * (pA[kk] + pB[kk]) for kk in pA}

                    ts = [round(float(x), 4) for x in np.linspace(0.0, 1.0, TGRID)]
                    rq_t = []
                    for tt in ts:
                        pt = {kk: (1 - tt) * pA[kk] + tt * pB[kk] for kk in pA}
                        q = K["fisher_quad"](ref, pt, b_ref, Xf, delta, micro)
                        rq_t.append(q / d2)
                        if "profile" in want:
                            append_csv(out_prof, NEW_PROF_COLS,
                                       dict(arch=arch, regime=regime, act=act, width=w,
                                            seedA=i, seedB=j, t=f"{tt:.4f}",
                                            rq_t=f"{q/d2:.6e}", dnorm=round(dn, 6),
                                            status="ok"))
                        del pt
                    i_mid = int(np.argmin([abs(t - 0.5) for t in ts]))
                    rqA, rqM, rqB = rq_t[0], rq_t[i_mid], rq_t[-1]

                    lmaxM, u1M, gapM = K["power_top"](ref, pmid, b_ref, Xf, micro,
                                                      POWER_ITERS, 17 + 101 * (i * 13 + j))
                    trPM, semM, _ = K["sphere_probes"](ref, pmid, b_ref, Xf, micro,
                                                       NPROBE, 23 + 101 * (i * 13 + j))
                    lmaxA, trPA, semA, u1A = anchors[i]
                    lmaxB, trPB, semB, u1B = anchors[j]

                    def ov(u):
                        return K["vdot"](dhat, u) ** 2

                    append_csv(out_pair, NEW_PAIR_COLS, dict(
                        arch=arch, regime=regime, act=act, width=w,
                        smooth=smooth_of(act), seedA=i, seedB=j, nparams=nP,
                        dnorm=round(dn, 6),
                        rq_A=f"{rqA:.6e}", rq_mid=f"{rqM:.6e}", rq_B=f"{rqB:.6e}",
                        lmax_A=f"{lmaxA:.6e}", lmax_mid=f"{lmaxM:.6e}",
                        lmax_B=f"{lmaxB:.6e}",
                        trP_A=f"{trPA:.6e}", trP_mid=f"{trPM:.6e}", trP_B=f"{trPB:.6e}",
                        trP_sem_A=f"{semA:.3e}", trP_sem_mid=f"{semM:.3e}",
                        trP_sem_B=f"{semB:.3e}",
                        ov1_A=f"{ov(u1A):.6e}", ov1_mid=f"{ov(u1M):.6e}",
                        ov1_B=f"{ov(u1B):.6e}", pow_gap_mid=f"{gapM:.2e}",
                        batch=RQ_BATCH, micro=micro, probes=NPROBE,
                        power_iters=POWER_ITERS,
                        status="ok" if smooth_of(act) else "ok:nonsmooth"))
                    A_ = rqM / trPM if trPM > 0 else float("nan")
                    log(f"  [pair {i}-{j}] rq_mid={rqM:.3e}  trF/P={trPM:.3e}  A={A_:.3f}")
                    del pA, pB, pmid, delta, dhat, sdB, u1M

                del sds, ref, p_ref, b_ref, anchors
                import gc
                gc.collect()
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                traceback.print_exc()
                append_csv(out_pair, NEW_PAIR_COLS,
                           dict(arch=arch, regime=regime, act=act, width=w,
                                smooth=smooth_of(act),
                                status="error:" + repr(e)[:40].replace(",", ";")))
    log(f"measurement done (worker {WORKER_ID})" if WORKER_ID is not None
        else "measurement done")


# ====================================================================== MERGE
PAIR_COLS = ["arch", "regime", "act", "width", "width_paper", "smooth",
             "seedA", "seedB", "dnorm",
             "rq_A", "rq_mid", "rq_B", "flen_A", "flen_mid", "flen_B",
             "lmax_A", "lmax_mid", "lmax_B", "trP_A", "trP_mid", "trP_B",
             "trP_sem_A", "trP_sem_mid", "trP_sem_B",
             "align_A", "align_mid", "align_B", "spec_A", "spec_mid", "spec_B",
             "ov1_mid", "top1_mid", "src_rq", "src_lmax", "src_trP", "status"]
PROF_COLS = ["arch", "regime", "act", "width", "width_paper", "seedA", "seedB",
             "t", "rq_t", "src", "status"]


def merge():
    """Old sources + this run -> the canonical tables.

    PRECEDENCE.  Where a quantity exists in both, the OLDER measurement wins.
    Not because it is more accurate -- the two agree to 7e-05, which is float
    summation order -- but because it covers far more: 216 cells against 44 for
    R_F, and an MLP profile on 21 t-points against 9.  The union is then the
    widest grid available, and the src_* columns record which row came from
    where."""
    log("merging old sources with this run ...")
    merged = {}
    for arch in ARCHS:
        for r in old_geodesic(arch):
            merged[(arch, r["regime"], r["act"], r["width"], r["seedA"], r["seedB"])] = dict(
                arch=arch, regime=r["regime"], act=r["act"], width=r["width"],
                width_paper=width_paper(arch, r["width"]), smooth=smooth_of(r["act"]),
                seedA=r["seedA"], seedB=r["seedB"], dnorm=r["dnorm"],
                rq_A=r["rq_A"], rq_mid=r["rq_mid"], rq_B=r["rq_B"],
                lmax_A=r["lmax_A"], src_rq="02_geodesic", src_lmax="02_geodesic",
                status="ok")

    n_rq = n_trP = 0
    for r, _src in trP_sources():
        if not is_ok(r):
            continue
        arch = r["arch"]
        w = int(num(r["width"], 0))
        k = (arch, r["regime"], r["act"], w,
             int(num(r["seedA"], -1)), int(num(r["seedB"], -1)))
        row = merged.get(k)
        if row is None:
            row = dict(arch=arch, regime=r["regime"], act=r["act"], width=w,
                       width_paper=width_paper(arch, w), smooth=smooth_of(r["act"]),
                       seedA=k[4], seedB=k[5], dnorm=num(r.get("dnorm")), status="ok")
            merged[k] = row
        # R_F and lmax only fill a hole; they never overwrite the older run.
        if not math.isfinite(num(row.get("rq_mid"))):
            row.update(rq_A=num(r.get("rq_A")), rq_mid=num(r.get("rq_mid")),
                       rq_B=num(r.get("rq_B")), src_rq="10_rayleigh")
            n_rq += 1
        if not math.isfinite(num(row.get("lmax_A"))):
            row.update(lmax_A=num(r.get("lmax_A")), src_lmax="10_rayleigh")
        # tr F/P exists nowhere else, so it is always taken.
        row.update(lmax_mid=num(r.get("lmax_mid")), lmax_B=num(r.get("lmax_B")),
                   trP_A=num(r.get("trP_A")), trP_mid=num(r.get("trP_mid")),
                   trP_B=num(r.get("trP_B")), trP_sem_A=num(r.get("trP_sem_A")),
                   trP_sem_mid=num(r.get("trP_sem_mid")),
                   trP_sem_B=num(r.get("trP_sem_B")),
                   ov1_mid=num(r.get("ov1_mid")), src_trP="10_rayleigh")
        n_trP += 1

    for row in merged.values():
        d2 = num(row.get("dnorm")) ** 2
        for t in ("A", "mid", "B"):
            rq, trP, lm = (num(row.get(f"rq_{t}")), num(row.get(f"trP_{t}")),
                           num(row.get(f"lmax_{t}")))
            row[f"flen_{t}"] = (0.5 * d2 * rq if math.isfinite(rq) and math.isfinite(d2)
                                else "")
            row[f"align_{t}"] = rq / trP if (math.isfinite(rq) and trP > 0) else ""
            row[f"spec_{t}"] = rq / lm if (math.isfinite(rq) and lm > 0) else ""
        ov1, lm, rq = (num(row.get("ov1_mid")), num(row.get("lmax_mid")),
                       num(row.get("rq_mid")))
        row["top1_mid"] = (ov1 * lm / rq if (math.isfinite(ov1) and math.isfinite(lm)
                                             and rq > 0) else "")

    rows = sorted(merged.values(),
                  key=lambda r: (r["arch"], r["regime"], r["act"], r["width"],
                                 r["seedA"], r["seedB"]))
    p1 = write_csv(os.path.join(OUT_DIR, "rayleigh_pairs.csv"), PAIR_COLS, rows)
    log(f"-> {p1}  ({len(rows)} pair rows; {n_rq} R_F filled from this run, "
        f"{n_trP} carry tr F/P)")

    prof, seen = [], set()
    for arch in ARCHS:
        for r in old_profile(arch):
            r["width_paper"] = width_paper(arch, r["width"])
            r["status"] = "ok"
            prof.append(r)
            seen.add((r["arch"], r["regime"], r["act"], r["width"],
                      r["seedA"], r["seedB"], r["t"]))
    n_new = 0
    for r in profile_sources():
        if not is_ok(r):
            continue
        w = int(num(r["width"], 0))
        key = (r["arch"], r["regime"], r["act"], w, int(num(r["seedA"], -1)),
               int(num(r["seedB"], -1)), round(num(r["t"]), 4))
        if key in seen:
            continue                      # the 21-point grid wins where it exists
        prof.append(dict(arch=r["arch"], regime=r["regime"], act=r["act"], width=w,
                         width_paper=width_paper(r["arch"], w),
                         seedA=key[4], seedB=key[5], t=key[6],
                         rq_t=num(r["rq_t"]), src="10_rayleigh", status="ok"))
        n_new += 1
    p2 = write_csv(os.path.join(OUT_DIR, "rayleigh_profile.csv"), PROF_COLS, prof)
    log(f"-> {p2}  ({len(prof)} profile rows, {n_new} from this run)")
    return rows, prof, build_cells(rows, prof)


# ============================================================= CELLS + ALPHAS
CELL_COLS = ["arch", "regime", "act", "width", "width_paper", "smooth", "n_pairs",
             "rq_mid_med", "rq_mid_q1", "rq_mid_q3", "trP_mid_med", "trP_sem_mid_med",
             "lmax_mid_med", "lmax_A_med", "dnorm_med", "flen_mid_med",
             "align_mid_med", "align_mid_rat", "spec_mid_rat", "top1_mid_med",
             "dip_depth", "t_at_peak", "n_profile_t", "barrier",
             "alpha_rq", "alpha_rq_r2", "alpha_trP", "alpha_trP_r2",
             "alpha_align", "alpha_align_r2", "alpha_dnorm", "alpha_dnorm_r2",
             "alpha_flen", "alpha_flen_r2", "alpha_lmax", "alpha_lmax_r2",
             "alpha_B", "alpha_B_r2", "alpha_dip", "alpha_dip_r2",
             "n_widths", "resid_rq"]


def fit_alpha(widths, vals):
    """OLS of log(val) on log(width).  alpha > 0 means DECREASING with width."""
    w = np.asarray(widths, float)
    y = np.asarray(vals, float)
    ok = (w > 0) & (y > 0) & np.isfinite(w) & np.isfinite(y)
    if ok.sum() < 3:
        return (float("nan"), float("nan"))
    lw, ly = np.log(w[ok]), np.log(y[ok])
    b, a = np.polyfit(lw, ly, 1)
    yh = a + b * lw
    r2 = 1 - ((ly - yh) ** 2).sum() / max(((ly - ly.mean()) ** 2).sum(), 1e-300)
    return (round(-float(b), 4), round(float(r2), 4))


def build_cells(pair_rows, prof_rows):
    bar = {}
    for arch in ARCHS:
        for k, v in old_barrier(arch).items():
            bar[(arch,) + k] = v

    prof_by = {}
    for r in prof_rows:
        prof_by.setdefault((r["arch"], r["regime"], r["act"], r["width"]), []).append(r)
    pair_by = {}
    for r in pair_rows:
        pair_by.setdefault((r["arch"], r["regime"], r["act"], r["width"]), []).append(r)

    cells = {}
    for key, rs in pair_by.items():
        arch, regime, act, w = key

        def col(c):
            return [num(r.get(c)) for r in rs]

        rq = col("rq_mid")
        rec = dict(arch=arch, regime=regime, act=act, width=w,
                   width_paper=width_paper(arch, w), smooth=smooth_of(act),
                   n_pairs=len(rs), rq_mid_med=median(rq),
                   rq_mid_q1=quantile(rq, .25), rq_mid_q3=quantile(rq, .75),
                   trP_mid_med=median(col("trP_mid")),
                   trP_sem_mid_med=median(col("trP_sem_mid")),
                   lmax_mid_med=median(col("lmax_mid")),
                   lmax_A_med=median(col("lmax_A")), dnorm_med=median(col("dnorm")),
                   flen_mid_med=median(col("flen_mid")),
                   align_mid_med=median(col("align_mid")),
                   top1_mid_med=median(col("top1_mid")),
                   barrier=bar.get(key, float("nan")))
        # Ratios are formed FROM the medians, never medianed as ratios: the
        # median does not commute with division, and only this makes
        # alpha_rq = alpha_trP + alpha_align exact.
        rec["align_mid_rat"] = (rec["rq_mid_med"] / rec["trP_mid_med"]
                                if rec["trP_mid_med"] > 0 else float("nan"))
        rec["spec_mid_rat"] = (rec["rq_mid_med"] / rec["lmax_mid_med"]
                               if rec["lmax_mid_med"] > 0 else float("nan"))
        pr = prof_by.get(key, [])
        rec["n_profile_t"] = len({round(num(r["t"]), 4) for r in pr})
        if pr:
            by_t = {}
            for r in pr:
                by_t.setdefault(round(num(r["t"]), 4), []).append(num(r["rq_t"]))
            med_t = {t: median(v) for t, v in by_t.items()}
            if 0.5 in med_t and med_t[0.5] > 0:
                peak = max(med_t, key=lambda t: med_t[t])
                rec["dip_depth"] = med_t[peak] / med_t[0.5]
                rec["t_at_peak"] = peak
        cells[key] = rec

    groups = {}
    for key, rec in cells.items():
        groups.setdefault(key[:3], []).append(rec)
    for recs in groups.values():
        recs.sort(key=lambda r: r["width"])
        ws = [r["width"] for r in recs]
        for src, name in (("rq_mid_med", "alpha_rq"), ("trP_mid_med", "alpha_trP"),
                          ("align_mid_rat", "alpha_align"), ("dnorm_med", "alpha_dnorm"),
                          ("flen_mid_med", "alpha_flen"), ("lmax_mid_med", "alpha_lmax"),
                          ("barrier", "alpha_B"), ("dip_depth", "alpha_dip")):
            a, r2 = fit_alpha(ws, [r.get(src, float("nan")) for r in recs])
            for r in recs:
                r[name], r[name + "_r2"] = a, r2
        for r in recs:
            r["n_widths"] = len(recs)
            ar, at, aa = r["alpha_rq"], r["alpha_trP"], r["alpha_align"]
            r["resid_rq"] = (round(abs(ar - at - aa), 4)
                             if all(math.isfinite(x) for x in (ar, at, aa)) else "")

    rows = sorted(cells.values(),
                  key=lambda r: (r["arch"], r["regime"], r["act"], r["width"]))
    p = write_csv(os.path.join(OUT_DIR, "rayleigh_cells.csv"), CELL_COLS, rows)
    log(f"-> {p}  ({len(rows)} cell rows)")
    return rows


# ==================================================================== SUMMARY
def summarise(cells):
    """Headline numbers, printed and written as JSON so a later session can read
    them without re-deriving anything."""
    uniq = {}
    for r in cells:
        uniq.setdefault((r["arch"], r["regime"], r["act"]), r)
    U = list(uniq.values())
    al = [r for r in U if math.isfinite(num(r.get("alpha_align")))]

    def col(c, rows=None):
        return [num(r.get(c)) for r in (rows or U) if math.isfinite(num(r.get(c)))]

    def reg(xcol, rows):
        x = np.array([num(r.get(xcol)) for r in rows], float)
        y = np.array([num(r.get("alpha_B")) for r in rows], float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return dict(n=int(m.sum()), r2=None, slope=None)
        b, a = np.polyfit(x[m], y[m], 1)
        yh = a + b * x[m]
        r2 = 1 - ((y[m] - yh) ** 2).sum() / max(((y[m] - y[m].mean()) ** 2).sum(), 1e-300)
        return dict(n=int(m.sum()), r2=round(float(r2), 4), slope=round(float(b), 4))

    smooth = [r for r in U if int(num(r.get("smooth"), 0)) == 1]
    out = {
        "n_cell_groups": len(U),
        "with_trP": len(al),
        "alpha_align": {
            "median": median([num(r["alpha_align"]) for r in al]),
            "min": min([num(r["alpha_align"]) for r in al], default=float("nan")),
            "max": max([num(r["alpha_align"]) for r in al], default=float("nan")),
            "n_abs_le_0.1": sum(1 for r in al if abs(num(r["alpha_align"])) <= 0.1)},
        "alpha_trP_median": median([num(r["alpha_trP"]) for r in al]),
        "alpha_rq_median": median(col("alpha_rq")),
        "dip": {"n_cells": len(col("alpha_dip")),
                "alpha_dip_median": median(col("alpha_dip")),
                "n_deepening": sum(1 for v in col("alpha_dip") if v < 0)},
        "barrier_regression": {
            "all": {c: reg(c, U) for c in ("alpha_flen", "alpha_rq", "alpha_trP",
                                           "alpha_align", "alpha_dnorm")},
            "smooth_only": {c: reg(c, smooth) for c in ("alpha_flen", "alpha_rq")}},
        "published_reference": {"fisher_length_r2": 0.904, "fisher_length_slope": 1.13,
                                "rayleigh_r2": 0.026},
    }
    p = os.path.join(OUT_DIR, "rayleigh_summary.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  cell groups (arch x regime x act): {out['n_cell_groups']}, "
          f"{out['with_trP']} carry tr F/P")
    a = out["alpha_align"]
    if math.isfinite(a["median"]):
        print(f"  alpha_A      median {a['median']:+.3f}   range "
              f"[{a['min']:+.3f}, {a['max']:+.3f}]   "
              f"|alpha_A| <= 0.1 in {a['n_abs_le_0.1']}/{out['with_trP']}")
        print(f"  alpha_trF/P  median {out['alpha_trP_median']:+.3f}")
        print("  -> alpha_A ~ 0 beside a large alpha_trF/P means R_F is tracking")
        print("     the spectrum, not the direction of Delta.")
    else:
        print("  alpha_A: unavailable -- no cell carries tr F/P yet (run `measure`).")
    d = out["dip"]
    if d["n_cells"]:
        print(f"  midpoint dip: median alpha_dip {d['alpha_dip_median']:+.3f} over "
              f"{d['n_cells']} cells, deepening with width in {d['n_deepening']}")
    print("  barrier regression   alpha_B ~ predictor:")
    for c, v in out["barrier_regression"]["all"].items():
        print(f"    {c:14s} n={v['n']:3d}  R2={v['r2']}  slope={v['slope']}")
    print("  published MNIST grid: Fisher length R2 0.904 slope 1.13, Rayleigh R2 0.026")
    log(f"-> {p}")
    return out


# ==================================================================== FIGURES
def figures(cells=None, prof=None):
    """Reading figures, written next to the tables.

    Self-contained on purpose: these run on Kaggle, where scripts/fig_style.py
    does not exist.  The paper figures live in scripts/ and read the same
    canonical CSVs; these are for reading a run, not for the paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7, "figure.dpi": 130,
                         "savefig.dpi": 200, "axes.facecolor": "#F2F2F2",
                         "grid.color": "#ffffff", "axes.grid": True,
                         "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "legend.frameon": False})
    CR = {"ntk": "#0072B2", "sp": "#56B4E9", "mup": "#D55E00"}
    LAB = {"ntk": "NTK", "sp": "SP", "mup": "muP"}
    LS = {"relu": (0, (1, 1.2)), "gelu": "-", "tanh": (0, (5, 1.6)),
          "swish": (0, (4, 1.3, 1, 1.3)), "softplus": (0, (2.4, 1.2))}

    cells = cells if cells is not None else read_csv(
        os.path.join(OUT_DIR, "rayleigh_cells.csv"))
    prof = prof if prof is not None else read_csv(
        os.path.join(OUT_DIR, "rayleigh_profile.csv"))
    if not cells:
        log("no cell table -- run `merge` first")
        return []

    groups = {}
    for r in cells:
        groups.setdefault((r["arch"], r["regime"], r["act"]), []).append(r)
    for v in groups.values():
        v.sort(key=lambda r: int(num(r["width"], 0)))
    archs = sorted({r["arch"] for r in cells})
    written = []

    def wticks(ax, xs):
        """Real widths on the x axis, not powers of two."""
        xs = sorted({int(x) for x in xs})
        if not xs:
            return
        step = 1 if len(xs) <= 4 else 2      # 7 widths do not fit a 2.6 in panel
        keep = xs[::step]
        if xs[-1] not in keep:
            keep.append(xs[-1])
        ax.set_xticks(keep)
        ax.set_xticklabels([f"{x//1024}k" if x >= 1024 else str(x) for x in keep],
                           fontsize=6.2)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = os.path.join(OUT_DIR, f"{name}.{ext}")
            fig.savefig(p, bbox_inches="tight")
            if ext == "png":
                written.append(p)
        plt.close(fig)
        log(f"-> {os.path.join(OUT_DIR, name)}.png / .pdf")

    # --- 1. alignment ratio vs width, one panel per architecture -----------
    if any(math.isfinite(num(r.get("align_mid_rat"))) for r in cells):
        fig, axes = plt.subplots(1, len(archs), figsize=(2.6 * len(archs), 2.7),
                                 squeeze=False)
        for ax, arch in zip(axes[0], archs):
            ax.axhline(1.0, color="#8a8a8a", lw=0.9, ls=(0, (4, 2)))
            for (a_, rg, act), g in sorted(groups.items()):
                if a_ != arch:
                    continue
                xy = [(num(r["width_paper"]), num(r.get("align_mid_rat"))) for r in g]
                xy = [(x, y) for x, y in xy if math.isfinite(y)]
                if len(xy) < 2:
                    continue
                ax.plot([p[0] for p in xy], [p[1] for p in xy],
                        color=CR.get(rg, "#444"), ls=LS.get(act, "-"), lw=1.2,
                        marker="o", ms=2.6, label=f"{LAB.get(rg, rg)} {act}")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(arch, fontsize=8)
            ax.set_xlabel("width $n$")
            wticks(ax, [num(r["width_paper"]) for (a_, _r, _a), g in groups.items()
                        if a_ == arch for r in g])
            # An empty panel means tr F/P has not been measured for that
            # architecture yet -- say so, rather than leaving a blank box that
            # reads as "measured, and zero".
            if not ax.lines[1:]:
                ax.text(0.5, 0.5, "tr $F/P$ not measured yet\n(run `measure`)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=6.6, color="#8a8a8a")
        axes[0][0].set_ylabel(r"$A=\mathcal{R}_F/(\mathrm{tr}\,F/P)$")
        h, l = [], []
        for ax in axes[0]:            # the leftmost panel may be the empty one
            hh, ll = ax.get_legend_handles_labels()
            for a_, b_ in zip(hh, ll):
                if b_ not in l:
                    h.append(a_)
                    l.append(b_)
        if h:
            fig.legend(h, l, loc="lower center", ncol=min(5, len(l)), fontsize=6,
                       bbox_to_anchor=(0.5, -0.14))
        fig.suptitle("A = 1 means the displacement is spectrally "
                     "indistinguishable from a random direction", fontsize=7, y=1.02)
        save(fig, "fig_rayleigh_alignment")

    # --- 2. the exponent split ---------------------------------------------
    rows = [(k, v[0]) for k, v in sorted(groups.items())
            if math.isfinite(num(v[0].get("alpha_align")))]
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, max(2.2, 0.26 * len(rows) + 1.1)))
        y = np.arange(len(rows))[::-1]
        for yi, ((arch, rg, act), r) in zip(y, rows):
            ax.barh(yi, num(r["alpha_trP"]), height=0.55, color=CR.get(rg, "#444"),
                    alpha=0.8, edgecolor="white", lw=0.5)
            ax.plot([num(r["alpha_rq"])], [yi], marker="D", ms=4.0, color="#262626",
                    markerfacecolor="white", mew=1.0, ls="none")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{a} {LAB.get(rg, rg)} {act}" for (a, rg, act), _ in rows],
                           fontsize=6.0)
        ax.axvline(0, color="#4d4d4d", lw=0.7)
        ax.set_xlabel(r"width exponent:  bar $=\alpha_{\mathrm{tr}F/P}$,  "
                      r"diamond $=\alpha_{\mathcal{R}_F}$,  gap $=\alpha_A$")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        save(fig, "fig_rayleigh_split")

    # --- 3. R_F(t) profile and the dip -------------------------------------
    if prof:
        pb = {}
        for r in prof:
            k = (r["arch"], r["regime"], r["act"], int(num(r["width"], 0)))
            pb.setdefault(k, {}).setdefault(round(num(r["t"]), 4), []).append(num(r["rq_t"]))
        med = {k: {t: median(v) for t, v in d.items()} for k, d in pb.items()}
        widest = {}
        for (arch, rg, act, w) in med:
            widest[(arch, rg, act)] = max(widest.get((arch, rg, act), 0), w)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 2.7))
        for (arch, rg, act, w), m in sorted(med.items()):
            if w != widest[(arch, rg, act)] or 0.0 not in m or m[0.0] <= 0:
                continue
            ts = sorted(m)
            ax1.plot(ts, [m[t] / m[0.0] for t in ts], color=CR.get(rg, "#444"),
                     ls=LS.get(act, "-"), lw=1.0, alpha=0.85)
        ax1.axvline(0.5, color="#8a8a8a", lw=0.9, ls=(0, (4, 2)))
        ax1.set_yscale("log")
        ax1.set_xlabel("$t$")
        ax1.set_ylabel(r"$\mathcal{R}_F(t)/\mathcal{R}_F(0)$")
        ax1.set_title("widest $n$ of each cell; dashed = the anchor", fontsize=7)
        for (arch, rg, act), g in sorted(groups.items()):
            xy = [(num(r["width_paper"]), num(r.get("dip_depth"))) for r in g]
            xy = [(x, y) for x, y in xy if math.isfinite(y)]
            if len(xy) >= 2:
                ax2.plot([p[0] for p in xy], [p[1] for p in xy],
                         color=CR.get(rg, "#444"), ls=LS.get(act, "-"), lw=1.0,
                         marker="o", ms=2.2)
        ax2.axhline(1.0, color="#8a8a8a", lw=0.9, ls=(0, (4, 2)))
        ax2.set_xscale("log", base=2)
        ax2.set_yscale("log")
        ax2.set_xlabel("width $n$")
        wticks(ax2, [num(r["width_paper"]) for g in groups.values() for r in g
                     if math.isfinite(num(r.get("dip_depth")))])
        ax2.set_ylabel(r"$\max_t\mathcal{R}_F(t)\,/\,\mathcal{R}_F(\frac{1}{2})$")
        ax2.set_title("how unrepresentative the anchor is", fontsize=7)
        save(fig, "fig_rayleigh_profile")

    # --- 4. barrier regression ---------------------------------------------
    U = [v[0] for v in groups.values()]
    pts = [(num(r.get("alpha_flen")), num(r.get("alpha_rq")), num(r.get("alpha_B")),
            r["regime"]) for r in U]
    pts = [p for p in pts if all(math.isfinite(v) for v in p[:3])]
    if len(pts) >= 3:
        fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.5, 2.6), sharey=True)
        for ax, idx, name in ((axl, 0, r"$\alpha_{\mathcal{L}_F}$  (Fisher length)"),
                              (axr, 1, r"$\alpha_{\mathcal{R}_F}$  (Rayleigh)")):
            x = np.array([p[idx] for p in pts])
            yv = np.array([p[2] for p in pts])
            for p in pts:
                ax.scatter(p[idx], p[2], s=15, color=CR.get(p[3], "#444"),
                           edgecolor="white", lw=0.4, zorder=5)
            b, a = np.polyfit(x, yv, 1)
            xs = np.linspace(x.min(), x.max(), 20)
            ax.plot(xs, a + b * xs, color="#777", ls="--", lw=0.9)
            r2 = 1 - ((yv - (a + b * x)) ** 2).sum() / max(((yv - yv.mean()) ** 2).sum(),
                                                           1e-300)
            ax.set_title(f"$R^2$={r2:.3f}, slope={b:+.2f}  (n={len(pts)})", fontsize=7.5)
            ax.set_xlabel(name)
        axl.set_ylabel(r"$\alpha_B$")
        save(fig, "fig_rayleigh_predicts")

    return written


# ======================================================================= MAIN
# =================================================================== WORKERS
# Every visible GPU gets a process.  The split is BY CELL, not by batch:
# fisher_vp is a torch.func loop over a P-dimensional vector, so splitting a
# batch across devices would ship 80 MB of parameters per call at width 4096 --
# more than the arithmetic it saves.  By cell, nothing crosses between
# processes, so the numbers are bit-for-bit what one GPU would produce.
WORKER_ID = _env("RQ_WORKER")
WORKER_TASKS = _env("RQ_TASKS")


def resolve_gpus():
    """GPUs to use.  A worker returns [] -- it sees one GPU through
    CUDA_VISIBLE_DEVICES and must not spawn further workers."""
    if WORKER_ID is not None:
        return []
    env = _env("RQ_GPUS")
    try:
        _import_torch()
        nv = torch.cuda.device_count()
    except Exception:
        nv = 0
    if env is not None:
        want = [x.strip() for x in env.split(",") if x.strip()]
        if len(want) > max(nv, 1):
            raise SystemExit(f"RQ_GPUS={want} but only {nv} GPU(s) visible")
        return want
    if SMOKE:
        return []
    return [str(i) for i in range(nv)] if nv > 1 else []


def split_tasks(tasks, n):
    """Balance cells across n workers by COST, not by position.

    The obvious `tasks[k::n]` is what 02_geodesic does, and on an ODD width grid
    it self-corrects: the phase flips every cell so the widths even out (MLP,
    7 widths: 1.08x).  On an EVEN grid it never flips and every heavy width
    lands on the same worker -- the CNN's 4 widths give a 4.0x split, i.e. one
    T4 idle for most of the run.  So sort by cost and give each cell to the
    currently-least-loaded worker (longest-processing-time greedy, within 4/3 of
    optimal).  Cost proxy is width^2: parameters dominate fisher_vp for the MLP
    and TS, and the CNN's conv weights are quadratic in the channel multiplier
    too."""
    items = sorted(tasks.items(), key=lambda kv: -(float(kv[0][3]) ** 2))
    groups = [{} for _ in range(n)]
    load = [0.0] * n
    for key, want in items:
        k = min(range(n), key=lambda i: load[i])
        groups[k][key] = want
        load[k] += float(key[3]) ** 2
    return groups


def enc_tasks(tasks):
    return ";".join(f"{a}:{r}:{ac}:{w}:{'+'.join(sorted(v))}"
                    for (a, r, ac, w), v in tasks.items())


def dec_tasks(spec):
    out = {}
    for it in spec.split(";"):
        if not it.strip():
            continue
        a, r, ac, w, want = it.split(":")
        out[(a, r, ac, int(w))] = set(want.split("+"))
    return out


def worker_paths(k):
    pid = os.getpid()
    return (os.path.join(OUT_DIR, f"rayleigh_new_pairs.gpu{k}.{pid}.csv"),
            os.path.join(OUT_DIR, f"rayleigh_new_profile.gpu{k}.{pid}.csv"))


def merge_worker_files():
    """Fold worker CSVs into the main ones, dropping repeated headers.

    Workers never append to a shared file: two processes appending to one file
    interleave lines and corrupt it.  The PID in the name means even two workers
    accidentally given the same id cannot collide."""
    for base, cols, pat in ((new_pairs_path(), NEW_PAIR_COLS,
                             "rayleigh_new_pairs.gpu*.csv"),
                            (new_prof_path(), NEW_PROF_COLS,
                             "rayleigh_new_profile.gpu*.csv")):
        parts = sorted(glob.glob(os.path.join(OUT_DIR, pat)))
        if not parts:
            continue
        have = os.path.exists(base)
        n = 0
        with open(base, "a", newline="") as dst:
            if not have:
                dst.write(",".join(cols) + "\n")
            for q in parts:
                with open(q) as f:
                    f.readline()
                    for ln in f:
                        if ln.strip():
                            dst.write(ln)
                            n += 1
                os.remove(q)
        log(f"[merge] {len(parts)} worker files -> {os.path.basename(base)} (+{n} rows)")


def self_path():
    """The source file to spawn workers from, or None.

    In a Kaggle notebook there is no __file__ -- the script lives in a cell --
    so the cell source is written to disk and that temp file is spawned.  If
    even that fails we return None and the caller falls back to one GPU: a
    speed optimisation must never be able to kill the job."""
    try:
        q = os.path.abspath(__file__)
        if os.path.exists(q):
            return q
    except NameError:
        pass
    try:
        from IPython import get_ipython
        cells = get_ipython().user_ns.get("In") or []
        src = next(c for c in reversed(cells)
                   if "def sphere_probes" in c and "def merge()" in c)
        q = os.path.join(OUT_DIR, "_measure_rayleigh_self.py")
        with open(q, "w") as f:
            f.write(src)
        log(f"[multi-gpu] no __file__ (notebook) -> wrote source to {q}")
        return q
    except Exception as e:
        log(f"[multi-gpu] cannot recover source to spawn workers: {e!r}")
        return None


def spawn_workers(gpus, tasks, path):
    import subprocess
    groups = split_tasks(tasks, len(gpus))
    procs = []
    for k, (gpu, grp) in enumerate(zip(gpus, groups)):
        if not grp:
            continue
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["RQ_WORKER"] = str(k)
        env["RQ_TASKS"] = enc_tasks(grp)
        env["RQ_CMD"] = "measure"
        env.pop("RQ_GPUS", None)
        cost = sum(float(key[3]) ** 2 for key in grp)
        log(f"[gpu {gpu}] worker {k}: {len(grp)} cells, cost share {cost:.3g}")
        procs.append(subprocess.Popen([sys.executable, path], env=env))
    rc = [pr.wait() for pr in procs]
    for k, r in enumerate(rc):
        if r != 0:
            log(f"!! worker {k} exited with {r}")
    return all(r == 0 for r in rc)


VALID_CMDS = ("audit", "selftest", "measure", "merge", "figures", "all")


def resolve_cmd():
    """The command to run, from RQ_CMD or argv.

    argv is NOT reliable here.  A Kaggle notebook is executed by papermill,
    which puts its own parameter file on the command line, so sys.argv[1] comes
    through as something like /tmp/tmpXXXX.json and the old code exited with
    "unknown command".  So: RQ_CMD wins, then the first argv entry that is
    actually one of the known commands, then "all".  Anything unrecognised is
    ignored rather than fatal -- an argument this file did not put there is not
    this file's to object to."""
    env = (_env("RQ_CMD") or "").strip().lower()
    if env:
        if env not in VALID_CMDS:
            raise SystemExit(f"RQ_CMD={env!r} invalid; use {' | '.join(VALID_CMDS)}")
        return env
    for a in sys.argv[1:]:
        if a.strip().lower() in VALID_CMDS:
            return a.strip().lower()
    return "all"


def main():
    cmd = resolve_cmd()
    os.makedirs(OUT_DIR, exist_ok=True)
    log(f"measure_rayleigh v2   cmd={cmd}   archs={ARCHS}   out={OUT_DIR}"
        + (f"   worker={WORKER_ID}" if WORKER_ID is not None else ""))
    log(f"  data={DATA_DIR}")

    if cmd == "audit":
        audit()
    elif cmd == "selftest":
        selftest()
    elif cmd == "measure":
        if not _flag("RQ_NOTEST") and WORKER_ID is None:
            selftest()          # once, in the parent -- not per worker
        measure()
    elif cmd == "merge":
        _, _, cells = merge()
        summarise(cells)
    elif cmd == "figures":
        figures()
    elif cmd == "all":
        if WORKER_ID is not None:      # a worker measures its slice and stops
            measure()
            return
        have = audit()
        try:
            if not _flag("RQ_NOTEST"):
                selftest()
            measure(have=have)
        except Exception as e:
            traceback.print_exc()
            log(f"!! measurement failed ({e!r}) -- merging what exists anyway")
        _, prof, cells = merge()
        summarise(cells)
        figures(cells=cells, prof=prof)
    else:                       # resolve_cmd only returns known commands
        raise SystemExit(f"unknown command {cmd!r}; use {' | '.join(VALID_CMDS)}")


if __name__ == "__main__":
    main()
