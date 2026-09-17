#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
measure_scale.py -- does the barrier analysis still work on a larger task?
MEASURE_SCALE_SOURCE_V2

TWO PHASES, ONE FILE, selected with PHASE=:

  barrier   (default)  trains the grid, then measures B, L_F, R_F, rho* and
                       ||grad L||.  Answers: does the Fisher-length PREDICTION
                       still hold?
  geodesic             reads those checkpoints and measures D_rel, the
                       deviation of the Fisher geodesic from the straight line.
                       Answers: is the straight line still a legitimate PROXY
                       for the geodesic, which is what the whole barrier
                       analysis is read off?
  all                  barrier, then geodesic.

They live in one file on purpose.  The geodesic phase loads the checkpoints the
barrier phase wrote, and a state_dict only loads back into the network that
produced it -- as two files that meant keeping two copies of the model, the
parameterisations, the permutation spec and the weight matching in step, and a
silent drift there would have measured the wrong pair of minima with nothing in
the output to show it.  One definition cannot drift from itself.

THE QUESTION.  The paper carries two separate claims and they need different
evidence.  Theorem 4.1 is an INEQUALITY, and testing whether it is tight needs
M_2 and M_3.  Figure 5 is a PREDICTION METHOD: the width exponent of the
Fisher length predicts the width exponent of the barrier, across cells, at
R^2 = 0.90 and slope 1.13, while the normalised Rayleigh quotient does not
(R^2 = 0.03).  This file tests the second, on a task bigger than MNIST, plus
(PHASE=geodesic) the Section 5.1 justification that lets it be read off a
straight line.  Nothing here computes M_2 or M_3.

WHY THE CHECKPOINTS CANNOT BE REUSED.  A trained weight vector is a minimum of
the task it was trained on and of nothing else.  Feeding CIFAR to the released
MNIST weights fails twice: the first layer is (n, 784) and the head is (10, n),
so the shapes do not match; and on a task with matching shapes but different
data the endpoint is no longer a minimum and B is no longer a barrier between
minima.  This script therefore TRAINS as well as measures, in one pass, and
both phases resume.

WHAT IT MEASURES.  Per aligned seed pair, all of it forward or forward-mode AD:

    B          max_t L(gamma_lin(t)) - (1/2)(L(w_A)+L(w_B)), TGRID points in t
    L_F(w_0)   (1/2) Delta' F(w_0) Delta          one JVP per micro-batch
    R_F(w_0)   Delta' F(w_0) Delta / ||Delta||^2  the normalised control
    rho*(w_0)  E||p_w(x) - e_y||_2                Definition 2.3
    ||Delta||  after weight-matching alignment

both at the MIDPOINT w_0 = (w_A+w_B)/2, which is what the published Figure 5
regresses on, and at the two endpoints, which is what the main text's prose
describes.  Then per cell (regime x activation) it fits alpha_B, alpha_LF and
alpha_RF over the width grid and regresses alpha_B on each of the two
predictors.

WHAT COUNTS AS AN ANSWER.  Not "R^2 is high".  The method survives only if the
Fisher length predicts AND the Rayleigh quotient still fails.  If both predict,
the test has lost its power at this scale rather than confirmed anything, and
the figure says so.

CHOOSE THE ARCHITECTURE WITH CARE.  The paper's headline "0.90 against 0.03"
pools three architectures, and the control's failure is largely a
CROSS-architecture effect.  Refitting the released cells one architecture at a
time:

    all three   36 cells   Fisher 0.904   Rayleigh 0.026   <- the headline
    MLP only    12 cells   Fisher 0.973   Rayleigh 0.630   <- control does NOT fail
    TS only     12 cells   Fisher 0.947   Rayleigh 0.001
    CNN only    12 cells   Fisher 0.903   Rayleigh 0.111

So an MLP-only run is a weak test: on the MLP the Rayleigh quotient tracks the
barrier too, even on MNIST where the answer is known.  MODE=cnn is both the
right architecture for image data at scale and the one whose control still
discriminates.

EVAL SET.  EVAL_N = 10000, taken from the head of the TRAIN split, exactly as
src/03_final does.  That is deliberate: B here has to be the same functional
as the published B, or the two runs are not comparable.

--------------------------------------------------------------------------
JUST RUN IT.  Paste the whole file into one Kaggle cell and Save Version.  The
defaults already are the experiment: CIFAR-10, CNN, widths 1/2/4/8, 3 regimes
x 4 smooth activations, 4 seeds, 100 epochs, and PHASE=all -- the barrier phase
followed by the geodesic phase.  Nothing needs to be set for the intended run.

Resuming is automatic.  The barrier phase skips every (cell, width) whose pairs
are already measured and whose checkpoints are already present, so a rerun
passes through the finished work in seconds, trains only what is missing, and
goes on to the geodesics.  Stage each session's output as an input dataset and
the next session picks up where it stopped.

Everything below is a knob, not a step.  PHASE=barrier or PHASE=geodesic runs
one phase alone; SMOKE=1 does a tiny end-to-end pass in a couple of minutes;
PLAN=1 prints the grid and times a real fisher_vp to project the cost without
doing any work; --selfcheck runs ten checks in seconds; --merge rebuilds the
CSVs and figures from what is on disk.

WHY CIFAR-10 AND NOT CIFAR-100.  The first attempt at this ran CIFAR-100 and
produced numbers that looked fine and meant nothing: endpoint loss 3.97 against
a chance loss of ln(100) = 4.61, so 14% below chance, and rho* = 0.99, which is
a softmax that is still essentially uniform.  The released cells sit at 87%
below chance (NTK) to 99.7% (SP/muP).  Two causes, and only one of them is
fixable by waiting longer:

  * 30 epochs is 5,880 gradient steps, where the released CNN run took 100
    epochs = ~23,400.  Hence EPOCHS defaults to 100 for MODE=cnn.
  * at width 1 this CNN holds 30,196 parameters.  That cannot reach a minimum
    on 100-way CIFAR-100 at any epoch count, so the small widths would be
    gated out and the power law would be fitted on two points.  CIFAR-10 they
    can fit.

CIFAR-10 also keeps K = 10, so B is the same functional as the published B and
the two clouds are directly comparable -- while still being a real step up:
the first natural-colour task in the paper, 3072-dimensional input against
784, and one that needs learned convolutional features rather than pixel
templates.  B is meaningless unless the endpoints are minima, so every row now
carries L_A, ln(K) and ||grad L|| at both endpoints, and a cell whose training
stopped short is skipped and says so.

On Kaggle paste the file into one cell and set the env at the top of the cell:

      import os
      os.environ.update(DATASET="cifar100", MODE="cnn", SMOKE="1")

Both phases resume from staged output, so a session that hits the 12 h wall is
continued by adding the previous version's output as an input dataset and
re-running.

DATA, AND WHY IT NEED NOT BE SLOW.  torchvision fetches CIFAR-100 from
cs.toronto.edu, which from a Kaggle session runs at about 80 kB/s -- 169 MB is
half an hour, and every worker used to pay it separately.  So the data path
looks, in order, at: the decoded uint8 cache `_raw_{DATASET}_{HW}.pt`; any copy
already on the box, in ANY layout (the cifar-*-python pickles and the MNIST
idx-ubyte files are read directly, so an attached Kaggle dataset works whatever
its folders are called); and only then the network, from the launcher alone,
never from a worker.  The decoded cache is always written.  In practice:

  * attach any CIFAR-100 dataset as a notebook input -> no download, ever;
  * or pay for it once, then stage that session's `_raw_cifar100_32.pt` as an
    input and every later session loads it in about a second.

The direct readers were checked against torchvision byte for byte.

READ THE VALIDITY COLUMNS BEFORE THE GEODESIC NUMBERS.  cg_iters == CG_ITERS
means CG hit its cap instead of converging; fd_instab is the disagreement
between the eps and eps/2 finite differences.  And cg_resid is NOT what it
looks like: CG never forms G_F x - rhs, it carries the residual forward by
r <- r - a A p, which loses the very cancellation it is measuring.  On a test
problem here, float32 CG reports 5.9e-13 where the truth is 1.2e-06 -- six
orders of magnitude optimistic, and that is the definition the released
param_geo_*.csv uses.  One extra matrix-vector product per solve buys the
honest number, so both are recorded: cg_resid keeps the released definition and
cg_resid_true is the one to trust.  Absolute magnitudes also depend on the CG
damping lam (Remark 5.1), so compare exponents in width, never a bare D_rel
against a different lam.  relu is excluded by default: Christoffel needs C^3
and relu has a kink; such rows are written but flagged smooth=0.

OUTPUT (all under OUT_DIR)
    ckpt_scale_{TAG}/                     trained checkpoints, resumable
    param_scale_{TAG}_shard*.csv          barrier: one row per seed pair
    param_scale_{TAG}.csv                 merged
    cell_scale_{TAG}.csv                  barrier: one row per cell
    figscale_{TAG}.pdf/.png               the two regressions
    param_geo_scale_{TAG}_shard*.csv      geodesic: one row per pair per lam
    param_geo_scale_{TAG}.csv             merged
    cell_geo_scale_{TAG}.csv              geodesic: one row per cell
    figgeo_scale_{TAG}.pdf/.png           D_rel vs width, plus the validity plot
    log_scale_{TAG}_shard*.txt            every worker's full log
                                          (TAG = {DATASET}_{MODE})
"""
import os, sys, time, math, glob, csv, itertools, traceback
import subprocess, threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.func import (functional_call, jvp as _fjvp,
                        vjp as _fvjp, grad as _grad)

# The string that lets this file recognise its own source on disk.  Everything
# about spawning workers hangs off it -- see _script_path().
SENTINEL = "MEASURE_SCALE_SOURCE_V2"

# ==================================================================== CONFIG
def _tv():
    import torchvision
    return torchvision


def _env(name, default):
    return os.environ.get(name, default)


def _flag(name, default="0"):
    return _env(name, default) not in ("0", "", "false", "False")


SMOKE = _flag("SMOKE")
PLAN  = _flag("PLAN")

# Two measurements, one file, because a Kaggle cell is one paste and because
# the geodesic phase reads the checkpoints the barrier phase trains -- keeping
# them in separate files meant keeping two copies of the network in step, and a
# silent drift there would have measured the wrong minima with no way to tell.
#
#   barrier   B, L_F, R_F, rho*, ||grad L||     trains, then measures
#   geodesic  D_rel via the Christoffel symbol  measures only, needs the ckpts
#   all       barrier, then geodesic  <- THE DEFAULT
#
# The default is "all", and it is deliberate.  The geodesic phase cannot train,
# so it needs the barrier phase's checkpoints; defaulting to "barrier" meant
# that running this file after the geodesic phase was added did the barrier
# work over again and measured no geodesics at all -- a whole session spent
# retraining checkpoints for pairs that were already measured.  With "all" the
# barrier phase skips every (cell, width) whose pairs are measured and whose
# checkpoints are present, trains only what is genuinely missing, and then the
# geodesic phase runs.  Ask for one phase by name when you want only that one.
PHASE = _env("PHASE", "all").lower()
if PHASE not in ("barrier", "geodesic", "all"):
    raise SystemExit(f"PHASE={PHASE!r}: pick barrier | geodesic | all")
PHASES = ["barrier", "geodesic"] if PHASE == "all" else [PHASE]

MODE = _env("MODE", "cnn")
if MODE not in ("mlp", "cnn"):
    raise SystemExit(f"MODE={MODE!r}: this file covers mlp and cnn")

# Input/output geometry of the task.  Changing DATASET changes the network's
# first and last layer, which is exactly why old checkpoints do not transfer.
# mean/std are per channel; a single scalar would leave the blue channel of a
# colour dataset ~18% off centre, and NTK-lazy is sensitive to input scale.
DATASETS = {
    "mnist":     dict(k=10,   ch=1, hw=28, tv="MNIST",
                      mean=(0.1307,), std=(0.3081,)),
    "fashion":   dict(k=10,   ch=1, hw=28, tv="FashionMNIST",
                      mean=(0.2860,), std=(0.3530,)),
    "cifar10":   dict(k=10,   ch=3, hw=32, tv="CIFAR10",
                      mean=(0.4914, 0.4822, 0.4465),
                      std=(0.2470, 0.2435, 0.2616)),
    "cifar100":  dict(k=100,  ch=3, hw=32, tv="CIFAR100",
                      mean=(0.5071, 0.4865, 0.4409),
                      std=(0.2673, 0.2564, 0.2762)),
    "imagenet32":   dict(k=1000, ch=3, hw=32, tv=None,
                         mean=(0.4810, 0.4574, 0.4078),
                         std=(0.2604, 0.2532, 0.2682)),
    "tinyimagenet": dict(k=200,  ch=3, hw=64, tv=None,
                         mean=(0.4802, 0.4481, 0.3975),
                         std=(0.2770, 0.2691, 0.2821)),
}
DATASET = _env("DATASET", "cifar10")
if DATASET not in DATASETS:
    raise SystemExit(f"DATASET={DATASET!r} unknown; pick {sorted(DATASETS)}")
_DS   = DATASETS[DATASET]
K     = _DS["k"]
IN_CH = _DS["ch"]
HW    = _DS["hw"]
DIN   = IN_CH*HW*HW                      # the MLP sees the flattened image

REGIMES = [r for r in _env("REGIMES", "ntk,sp,mup").split(",") if r]
ACTS    = [a for a in _env("ACTS", "gelu,tanh,swish,softplus").split(",") if a]
WIDTHS  = [int(v) for v in _env(
    "WIDTHS", "64,128,256,512,1024" if MODE == "mlp" else "1,2,4,8").split(",")]
# The published run used 5 seeds.  Four gives C(4,2) = 6 pairs per cell, so
# each point of the power-law fit is a median over 6 measurements rather than
# 3 -- the cheapest available improvement in the quality of the fit, since the
# exponent is fitted on those medians.
NSEEDS  = int(_env("NSEEDS", "4"))
PAIRS   = int(_env("PAIRS", "6"))        # C(4,2) = 6

# Training recipe, identical to src/01_train: SGD+momentum, per-layer lr_scale
# from the parameterisation, linear warmup into cosine, gradient clipping.
# docs/RUNBOOK.md: the released MLP/MNIST run used 30 epochs, the released
# CNN/FashionMNIST run used 100.  CIFAR is harder than either, so 30 epochs of
# CNN stops the network halfway down the loss curve -- above chance, nowhere
# near a minimum, and B measured there is not a barrier between minima.
EPOCHS        = int(_env("EPOCHS", "30" if MODE == "mlp" else "100"))
BATCH         = int(_env("BATCH", "256"))
LR            = float(_env("LR", "0.1"))
WARMUP_EPOCHS = int(_env("WARMUP_EPOCHS", "8"))
CLIP_NORM     = float(_env("CLIP_NORM", "1.0"))
TRAIN_N       = int(_env("TRAIN_N", "0"))   # 0 = the whole training set

# Measurement.  EVAL_N = 10000 from the head of the TRAIN split is what
# src/03_final used; keeping it identical is what makes B comparable to the
# published B.  Raising it makes the barrier 5x dearer and comparable to
# nothing.
TGRID       = int(_env("TGRID", "41"))
EVAL_N      = int(_env("EVAL_N", "10000"))
FISHER_N    = int(_env("FISHER_N", "2048"))
MICRO       = int(_env("MICRO", "512"))
MICRO_JVP   = int(_env("MICRO_JVP", "128"))
MATCH_ITERS = int(_env("MATCH_ITERS", "8"))

# --- PHASE=geodesic ------------------------------------------------------
# GEO_BATCH matches the released run's 2048, so the Fisher in the geodesic step
# is evaluated on the same batch the released dF was.  TGRID=9 is the Green
# quadrature grid: the integrand is smooth and every extra point costs a whole
# Christoffel solve.
GEO_BATCH   = int(_env("GEO_BATCH", "2048"))
GEO_TGRID   = int(_env("GEO_TGRID", "9"))
CG_ITERS    = int(_env("GEO_CG_ITERS", "300"))
CG_TOL      = float(_env("GEO_CG_TOL", "1e-6"))
POWER_ITERS = int(_env("GEO_POWER_ITERS", "20"))
FD_EPS      = float(_env("GEO_FD_EPS", "3e-3"))
FD_RICH     = _flag("GEO_FD_RICH", "1")
LAM_REL     = float(_env("GEO_LAM_REL", "1e-2"))
LAM_SWEEP   = ([1e-1, 1e-2, 1e-3] if _flag("GEO_LAMSWEEP") else None)
GEO_PAIRS   = int(_env("GEO_PAIRS", "3"))   # Christoffel is far dearer than B

# Christoffel is computed with a finite difference of F, so it needs C^3.
# relu has a kink: rows for it are still written but flagged, never pooled.
SMOOTH_ACTS = ["gelu", "tanh", "swish", "softplus"]

# micro-batch for fisher_vp.  The TOTAL flops do not depend on micro -- the
# whole batch is summed either way -- so a smaller micro only adds tracing and
# launch overhead.  Take the largest micro whose activation memory fits, which
# for a CNN is set by feature-map size and not at all by the parameter count.
GEO_MICRO_ENV = _env("GEO_MICRO", None)
ACT_BUDGET    = float(_env("GEO_ACT_BUDGET", "2.0e9"))
ACT_COPIES    = 6            # primal + jvp tangent + vjp tape, rough

# A cell whose networks did not learn is not a cell: the barrier between two
# points that are not minima is not the object being predicted.
#
# "3x chance accuracy" is a much weaker demand at K=100 than at K=10 -- it lets
# a 3%-accurate network through -- so the binding test is on the LOSS: how far
# below ln(K) the endpoint actually got.  For reference, the released cells sat
# at 87% below chance (NTK) to 99.7% (SP/muP).  A run at 14% below chance is
# not measuring the same object, and should say so rather than quietly produce
# a figure.
MIN_ACC_OVER_CHANCE = float(_env("MIN_ACC_OVER_CHANCE", "3.0"))
# Set to catch the catastrophe (the CIFAR-100 run that reached 14% below
# chance) without deleting NTK, which is lazy by construction and sat at 87%
# in the released run while SP/muP sat at 99.7%.  Every cell's endpoint loss
# and ||grad L|| are recorded either way, so convergence can be judged per
# cell at analysis time instead of only by this one threshold.
MIN_LOSS_DROP = float(_env("MIN_LOSS_DROP", "0.3"))   # fraction below ln(K)

BASE   = 64
SEED   = 20260909
RESUME = _flag("RESUME", "1")

# Kaggle commits /kaggle/working only when the version FINISHES.  A run killed
# at the 12 h wall loses everything it computed, so stop taking new work in
# time to exit cleanly and let the notebook commit; the next session resumes
# from the staged output.  Hours, measured from process start.
TIME_BUDGET_H = float(_env("TIME_BUDGET_H", "10.5"))
_T_START      = time.time()

CKPT_ROOTS = [".", "/kaggle/input", "/kaggle/working", "/content",
              "/content/drive/MyDrive"]
OUT_DIR    = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
DATA_DIR   = _env("DATA_DIR", None)
REF_DIR    = _env("REF_DIR", None)          # the released repo's data/ dir
_SHARD_ENV = os.environ.get("SHARD_ID")
SHARD_ID   = int(_SHARD_ENV or "0")
NUM_SHARDS = int(_env("NUM_SHARDS", "1"))
DEVICE     = _env("DEVICE", None) or ("cuda" if torch.cuda.is_available() else "cpu")
WORKERS    = _env("WORKERS", "auto")        # auto | 1 | 2 | ...
NO_LAUNCH  = _flag("NO_LAUNCH")             # set in every spawned child
LAUNCH_TIMEOUT = float(_env("LAUNCH_TIMEOUT", "180"))
SCAN_SECONDS   = float(_env("SCAN_SECONDS", "90"))
LOG_EVERY_S    = float(_env("LOG_EVERY_S", "60"))

# SMOKE runs the entire pipeline -- data, training, alignment, barrier, Fisher,
# exponent fit, figure -- on a grid small enough to finish on a CPU in about a
# minute.  It is not a test of the science; it is a test that the plumbing is
# connected, which is the failure this file exists to make impossible.
if SMOKE:
    DATASET = _env("SMOKE_DATASET", "fashion")
    _DS = DATASETS[DATASET]
    K, IN_CH, HW = _DS["k"], _DS["ch"], _DS["hw"]
    DIN = IN_CH*HW*HW
    # A smoke default must never silently overrule something the caller asked
    # for: SMOKE=1 EPOCHS=25 has to mean 25 epochs, or the smoke run cannot be
    # used to check that a longer recipe actually converges.
    def _sm(name, val):
        return type(val)(os.environ[name]) if name in os.environ else val
    REGIMES = ["ntk", "sp", "mup"]
    ACTS    = [a for a in _env("ACTS", "gelu,tanh").split(",") if a]
    WIDTHS  = ([int(v) for v in os.environ["WIDTHS"].split(",")]
               if "WIDTHS" in os.environ
               else ([8, 12, 16] if MODE == "mlp" else [1, 2, 3]))
    NSEEDS  = _sm("NSEEDS", 2)
    PAIRS   = _sm("PAIRS", 1)
    EPOCHS  = _sm("EPOCHS", 1)
    TRAIN_N = _sm("TRAIN_N", 2048)
    EVAL_N  = _sm("EVAL_N", 512)
    FISHER_N = _sm("FISHER_N", 256)
    TGRID   = _sm("TGRID", 9)
    MATCH_ITERS = _sm("MATCH_ITERS", 3)
    GEO_BATCH = _sm("GEO_BATCH", 128)
    GEO_TGRID = _sm("GEO_TGRID", 5)
    GEO_PAIRS = _sm("GEO_PAIRS", 1)
    CG_ITERS  = _sm("GEO_CG_ITERS", 60)
    POWER_ITERS = _sm("GEO_POWER_ITERS", 12)
    MIN_ACC_OVER_CHANCE = _sm("MIN_ACC_OVER_CHANCE", 1.2)
    MIN_LOSS_DROP = _sm("MIN_LOSS_DROP", 0.0)   # a 1-epoch smoke cannot converge
    TIME_BUDGET_H = _sm("TIME_BUDGET_H", 0.5)

TAG    = f"{DATASET}_{MODE}" + ("_smoke" if SMOKE else "")
_CACHE = {}
_LOGFH = None


def out_of_time(margin_h=0.0):
    return (time.time() - _T_START)/3600.0 > (TIME_BUDGET_H - margin_h)


_TRAIN_TIMES = []


def _note_train_time(sec):
    _TRAIN_TIMES.append(sec)


def _worst_train_h():
    """Leave room for one more training of the size this run actually does.

    Measured, not guessed: widths differ by more than a factor of ten in cost,
    so a fixed margin is either wasteful at width 1 or too small at width 8."""
    if not _TRAIN_TIMES:
        return 0.25
    return min(2.0, 1.3*max(_TRAIN_TIMES)/3600.0 + 0.05)


def log(*a):
    msg = f"[{time.strftime('%H:%M:%S')}][s{SHARD_ID}] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    if _LOGFH is not None:
        try:
            _LOGFH.write(msg + "\n"); _LOGFH.flush()
        except (OSError, ValueError):
            pass


def open_log():
    """Every worker keeps its own log file.  A notebook that loses stdout, or a
    worker whose pipe is not being relayed, still leaves a record on disk that
    the next session can read."""
    global _LOGFH
    try:
        _LOGFH = open(os.path.join(OUT_DIR, f"log_scale_{TAG}_shard{SHARD_ID}.txt"),
                      "a", buffering=1)
    except OSError:
        _LOGFH = None


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ===================================================================== MODEL
def make_act(n):
    return {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh,
            "swish": nn.SiLU, "softplus": nn.Softplus}[n]()


def param_cfg(regime, fin, fout, kind):
    """Init std, forward multiplier, per-layer lr scale.  Verbatim from the
    released scripts -- this is what makes a regime a regime."""
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
        self.bias = nn.Parameter(torch.zeros(fout))

    def forward(self, x):
        return self.fmul*F.linear(x, self.weight) + self.bias


class ScaledConv(nn.Module):
    def __init__(self, cin, cout, k, st, pad, regime, kind):
        super().__init__()
        fin = cin*k*k
        istd, self.fmul, self.lr_scale = param_cfg(regime, fin, cout, kind)
        self.weight = nn.Parameter(torch.randn(cout, cin, k, k)*istd)
        self.st = st; self.pad = pad

    def forward(self, x):
        return self.fmul*F.conv2d(x, self.weight, None, self.st, self.pad)


def _gn(c): return nn.GroupNorm(1, c)


class NetMLP(nn.Module):
    def __init__(self, width, act, regime="ntk", din=None, k=None):
        super().__init__()
        din = DIN if din is None else din
        k = K if k is None else k
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


class NetCNN(nn.Module):
    """GroupNorm(1, c) is a whole-tensor normaliser, so permuting channels is
    an exact symmetry of the network -- which is what makes weight matching
    valid here.  BatchNorm's running statistics would not survive it."""

    def __init__(self, wm, act, regime="ntk", in_ch=None, k=None):
        super().__init__()
        in_ch = IN_CH if in_ch is None else in_ch
        k = K if k is None else k
        c = [16*wm, 32*wm, 64*wm]
        self.c1 = ScaledConv(in_ch, c[0], 3, 1, 1, regime, "input"); self.n1 = _gn(c[0]); self.a1 = make_act(act)
        self.c2 = ScaledConv(c[0], c[1], 3, 1, 1, regime, "hidden"); self.n2 = _gn(c[1]); self.a2 = make_act(act)
        self.c3 = ScaledConv(c[1], c[2], 3, 1, 1, regime, "hidden"); self.n3 = _gn(c[2]); self.a3 = make_act(act)
        self.pool = nn.MaxPool2d(2)
        self.fc = ScaledLinear(c[2], k, regime, "output")
        self.width = wm; self.regime = regime

    def forward(self, x):
        h1 = self.pool(self.a1(self.n1(self.c1(x))))
        h2 = self.pool(self.a2(self.n2(self.c2(h1))))
        h3 = self.a3(self.n3(self.c3(h2)))
        return self.fc(F.adaptive_avg_pool2d(h3, 1).flatten(1))

    def opt_groups(self, base_lr):
        gs = [{"params": [m.weight], "lr": base_lr*m.lr_scale}
              for m in (self.c1, self.c2, self.c3)]
        gs.append({"params": [self.fc.weight, self.fc.bias],
                   "lr": base_lr*self.fc.lr_scale})
        gs.append({"params": [p for n in (self.n1, self.n2, self.n3)
                              for p in n.parameters()], "lr": base_lr})
        return gs


def build_net(width, act, regime):
    return (NetCNN(width, act, regime) if MODE == "cnn"
            else NetMLP(width, act, regime)).to(DEVICE)


def n_params(width, act, regime):
    return sum(p.numel() for p in build_net(width, act, regime).parameters())


# ============================================================== PERMUTATIONS
def perm_spec(model):
    if MODE == "cnn":
        ag = {"c1.weight": ["h1", None, None, None],
              "n1.weight": ["h1"], "n1.bias": ["h1"],
              "c2.weight": ["h2", "h1", None, None],
              "n2.weight": ["h2"], "n2.bias": ["h2"],
              "c3.weight": ["h3", "h2", None, None],
              "n3.weight": ["h3"], "n3.bias": ["h3"],
              "fc.weight": [None, "h3"], "fc.bias": [None]}
    else:
        ag = {"fc1.weight": ["h1", None], "fc1.bias": ["h1"],
              "fc2.weight": ["h2", "h1"], "fc2.bias": ["h2"],
              "fc3.weight": [None, "h2"], "fc3.bias": [None]}
    sd = model.state_dict(); gs = {}
    for n, axes in ag.items():
        for a, g in enumerate(axes):
            if g: gs[g] = sd[n].shape[a]
    return ag, gs


def _idx(perms, g, t):
    """The permutations come out of scipy's linear_sum_assignment, so they are
    always CPU LongTensors, while the state_dict they index lives wherever the
    model was trained.  index_select demands both on one device, and on a CPU
    box the mismatch cannot occur -- which is exactly why it survived CPU
    testing and only broke on Kaggle's GPUs."""
    i = perms[g]
    return i if i.device == t.device else i.to(t.device)


def _perm_except(t, axes, perms, exc):
    for a, g in enumerate(axes):
        if g and g != exc:
            t = t.index_select(a, _idx(perms, g, t))
    return t


def apply_perm(sd, ag, perms):
    out = {}
    for n, t in sd.items():
        axes = ag.get(n)
        if axes is None:
            out[n] = t.clone(); continue
        for a, g in enumerate(axes):
            if g: t = t.index_select(a, _idx(perms, g, t))
        out[n] = t
    return out


def weight_matching(ag, gs, sdA, sdB, iters=8, seed=0):
    """Align B onto A by permuting hidden units, the released procedure."""
    rng = np.random.RandomState(seed)
    perms = {g: torch.arange(n) for g, n in gs.items()}
    for _ in range(iters):
        for g in rng.permutation(list(gs.keys())):
            n = gs[g]; M = torch.zeros(n, n, dtype=torch.float64)
            for name, axes in ag.items():
                if g not in axes: continue
                a = axes.index(g)
                A = sdA[name].detach().cpu().to(torch.float64)
                Bt = _perm_except(sdB[name].detach().cpu().to(torch.float64),
                                  axes, perms, g)
                A = A.movedim(a, 0).reshape(n, -1)
                Bt = Bt.movedim(a, 0).reshape(n, -1)
                M += A @ Bt.T
            r, c = linear_sum_assignment(-M.numpy())
            perms[g] = torch.as_tensor(c[np.argsort(r)], dtype=torch.long)
    return perms


# ======================================================== STAGED-INPUT INDEX
# Every lookup of a checkpoint or a released CSV used to be its own
# glob("/kaggle/input/**", recursive=True).  With 180 checkpoints in the grid
# that is 180 full walks of an input mount that may hold a million image files,
# and it is indistinguishable from a hang.  Walk once, with a wall-clock cap
# and directory pruning, and answer every later lookup from the index.
_INDEX = None


def input_index():
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    hits = {"ckpt": {}, "param": [], "geoparam": [], "final": {}, "geo": {},
            "data": [], "dirs": {}, "raw": [], "cifar": [], "idx": [],
            "archives": []}
    t0, nfile, pruned = time.time(), 0, 0
    seen_roots = []
    roots = list(CKPT_ROOTS)
    if REF_DIR:
        roots.insert(0, REF_DIR)
    # ../../data relative to this file is the released repo's own CSVs, so a
    # run from a checkout gets the published-MNIST overlay with no staging.
    here = globals().get("__file__")
    if here:
        roots.insert(0, os.path.join(os.path.dirname(os.path.abspath(here)),
                                     "..", "..", "data"))
    for root in roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root) or root in seen_roots:
            continue
        seen_roots.append(root)
        depth0 = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root):
            # An ImageFolder tree is thousands of sibling class directories.
            # Nothing we index lives inside one, so do not descend into it.
            if len(dirnames) > 200 or dirpath.count(os.sep) - depth0 >= 6:
                pruned += len(dirnames); dirnames[:] = []
            dirnames[:] = [d for d in dirnames
                           if not d.startswith(".") and d != "__pycache__"]
            nfile += len(filenames)
            if time.time() - t0 > SCAN_SECONDS:
                log(f"input scan: capped at {SCAN_SECONDS:.0f}s "
                    f"({nfile} files seen). Set SCAN_SECONDS higher, or "
                    f"DATA_DIR/CKPT_DIR, if something staged was missed.")
                _INDEX = hits
                return hits
            base = os.path.basename(dirpath)
            # Directories that mean "this dataset is already on disk here".
            # Finding one turns a 35-minute download into a file read.
            if base in _DATA_MARKERS:
                hits["dirs"].setdefault(base, []).append(dirpath)
            # Kaggle slugs a dataset's folder however it likes, and uploaders
            # routinely strip the archive's own parent directory, so the name
            # 'cifar-100-python' may appear nowhere at all.  Recognise the
            # payload instead: the cifar pickles, or the idx-ubyte files.
            fset = set(filenames)
            if {"train", "meta"} <= fset or {"train", "test"} <= fset:
                hits["cifar"].append(dirpath)
            if any(f.startswith("data_batch_") for f in fset):
                hits["cifar"].append(dirpath)
            if any(f.startswith("train-images-idx3-ubyte") for f in fset):
                hits["idx"].append(dirpath)
            for f in filenames:
                if ("cifar" in f.lower() or "mnist" in f.lower()) and (
                        f.endswith(".tar.gz") or f.endswith(".tar")):
                    hits["archives"].append(os.path.join(dirpath, f))
            for f in filenames:
                p = os.path.join(dirpath, f)
                if f.startswith(f"_raw_{DATASET}_") and f.endswith(".pt"):
                    hits["raw"].append(p)
                elif base.startswith("ckpt_scale_") and f.endswith(".pt"):
                    if base == f"ckpt_scale_{TAG}":
                        hits["ckpt"].setdefault(f, p)
                elif (f.startswith(f"param_geo_scale_{TAG}")
                      and f.endswith(".csv")):
                    hits["geoparam"].append(p)
                elif f.startswith(f"param_scale_{TAG}") and f.endswith(".csv"):
                    hits["param"].append(p)
                elif f.endswith("_pairs.csv") and base in ("final", "geodesic"):
                    # the RELEASED repo layout: data/final/{arch}_pairs.csv
                    # and data/geodesic/{arch}_pairs.csv
                    k = "final" if base == "final" else "geo"
                    hits[k].setdefault(f.split("_")[0], p)
                elif f.startswith("param_final_") and f.endswith(".csv"):
                    # the raw Kaggle output layout
                    hits["final"].setdefault(f[len("param_final_"):-4], p)
                elif f.startswith("param_geo_") and f.endswith(".csv"):
                    hits["geo"].setdefault(f[len("param_geo_"):-4], p)
                elif f.startswith(f"_data_{DATASET}_") and f.endswith(".pt"):
                    hits["data"].append(p)
    log(f"input scan: {nfile} files in {time.time()-t0:.0f}s "
        f"({pruned} dirs pruned) -> {len(hits['ckpt'])} staged checkpoints, "
        f"{len(hits['param'])} barrier CSVs, "
        f"{len(hits['geoparam'])} geodesic CSVs, "
        f"{len(hits['raw'])+len(hits['data'])} data caches, "
        f"{len(hits['cifar'])} cifar dirs, {len(hits['idx'])} idx dirs, "
        f"{len(hits['archives'])} archives")
    _INDEX = hits
    return hits


# ====================================================================== DATA
# Downloading CIFAR-100 from cs.toronto.edu inside a Kaggle session runs at
# about 80 kB/s: 169 MB, so ~35 minutes -- and once per worker, because each
# one calls torchvision independently.  None of that is necessary.  The archive
# is almost always already on the box (attached as an input dataset, or left in
# ./data by an earlier version), and after the first read the decoded tensor is
# cached, so every later session starts in seconds.
_DATA_MARKERS = {
    "cifar-100-python": "CIFAR100",
    "cifar-10-batches-py": "CIFAR10",
    "MNIST": "MNIST",
    "FashionMNIST": "FashionMNIST",
}


def raw_cache_path():
    return os.path.join(OUT_DIR, f"_raw_{DATASET}_{HW}.pt")


def _read_cifar_pickle(d):
    """The cifar-*-python release, read straight from its pickles.

    torchvision can only load these through a root laid out exactly the way it
    expects, and it verifies an MD5 before it will touch them.  Reading the
    pickles ourselves means ANY attached copy of CIFAR works, whatever the
    dataset that carries it chose to call its folders."""
    import pickle
    lab = "fine_labels" if DATASET == "cifar100" else "labels"
    names = ["train"] if DATASET == "cifar100" else [f"data_batch_{i}" for i in range(1, 6)]
    xs, ys = [], []
    for n in names:
        f = os.path.join(d, n)
        if not os.path.exists(f):
            return None
        with open(f, "rb") as fh:
            e = pickle.load(fh, encoding="bytes")
        e = {(k.decode() if isinstance(k, bytes) else k): v for k, v in e.items()}
        xs.append(np.asarray(e["data"], dtype=np.uint8).reshape(-1, 3, 32, 32))
        ys.append(np.asarray(e[lab], dtype=np.int64))
    X = torch.from_numpy(np.concatenate(xs))
    Y = torch.from_numpy(np.concatenate(ys))
    log(f"  read {X.shape[0]} images from the cifar pickles at {d}")
    return X, Y


def _read_idx_ubyte(d):
    """MNIST/FashionMNIST idx-ubyte, read directly (gzipped or not)."""
    import gzip
    def _one(stem):
        for cand in (os.path.join(d, stem), os.path.join(d, stem + ".gz")):
            if os.path.exists(cand):
                op = gzip.open if cand.endswith(".gz") else open
                with op(cand, "rb") as fh:
                    b = fh.read()
                magic = int.from_bytes(b[:4], "big")
                ndim = magic & 0xFF
                shape = [int.from_bytes(b[4+4*i:8+4*i], "big") for i in range(ndim)]
                return np.frombuffer(b[4+4*ndim:], dtype=np.uint8).reshape(shape)
        return None
    xi = _one("train-images-idx3-ubyte")
    yi = _one("train-labels-idx1-ubyte")
    if xi is None or yi is None:
        return None
    X = torch.from_numpy(xi.copy()).unsqueeze(1)          # (N,1,28,28)
    Y = torch.from_numpy(yi.astype(np.int64))
    log(f"  read {X.shape[0]} images from the idx-ubyte files at {d}")
    return X, Y


def _extract_archive(path):
    """Unpack a staged cifar/mnist tarball into OUT_DIR and return where."""
    import tarfile
    dst = os.path.join(OUT_DIR, "_extracted")
    os.makedirs(dst, exist_ok=True)
    log(f"  extracting {path} -> {dst}")
    with tarfile.open(path) as tf:
        tf.extractall(dst)
    return dst


def _describe_inputs():
    """When discovery fails, say what IS mounted.  A run that reports 'not
    found' without showing what it looked at costs another whole session to
    diagnose."""
    for root in ("/kaggle/input", "./data"):
        if not os.path.isdir(root):
            continue
        log(f"  contents of {root}:")
        try:
            for d in sorted(os.listdir(root))[:20]:
                sub = os.path.join(root, d)
                inner = (sorted(os.listdir(sub))[:12]
                         if os.path.isdir(sub) else [])
                log(f"    {d}/  ->  {inner}")
        except OSError as e:
            log(f"    unreadable ({e})")


def _find_local_dataset():
    """Look for this dataset already on disk, in ANY layout, before the network
    is even considered.  Returns (X uint8 [N,C,H,W], Y int64) or None.

    Layouts seen in the wild, all of which must work:
      <x>/cifar-100-python/{train,test,meta}   the archive unpacked as shipped
      <x>/{train,test,meta}                    uploader stripped the parent
      <x>/cifar-100-python.tar.gz              the archive itself
      <x>/FashionMNIST/raw/*-idx3-ubyte[.gz]   torchvision's own layout
      <x>/*-idx3-ubyte[.gz]                    the idx files, loose
    """
    want = _DS["tv"]
    idx = input_index()
    cifar = want in ("CIFAR100", "CIFAR10")
    marks = [m for m, t in _DATA_MARKERS.items() if t == want]

    cands = []
    if DATA_DIR:
        cands += [DATA_DIR, os.path.join(DATA_DIR, marks[0] if marks else "")]
    for m in marks:
        cands += idx["dirs"].get(m, [])
        for r in ("./data", "/kaggle/input", "/kaggle/working"):
            cands.append(os.path.join(r, m))
    cands += idx["cifar"] if cifar else idx["idx"]

    def _try(d):
        if not d or not os.path.isdir(d):
            return None
        try:
            if cifar:
                return _read_cifar_pickle(d)
            return (_read_idx_ubyte(os.path.join(d, "raw"))
                    or _read_idx_ubyte(d))
        except Exception as e:
            log(f"  {d}: unreadable ({type(e).__name__}: {e})")
            return None

    for d in dict.fromkeys(cands):
        got = _try(d)
        if got is not None:
            return got

    # Nothing unpacked; is the archive itself staged?
    for a in dict.fromkeys(idx["archives"]):
        try:
            root = _extract_archive(a)
        except Exception as e:
            log(f"  {a}: could not extract ({type(e).__name__}: {e})")
            continue
        for dp, _dn, fn in os.walk(root):
            got = _try(dp)
            if got is not None:
                return got

    log(f"  {DATASET} not found in any staged input.")
    _describe_inputs()
    return None


def ensure_raw():
    """(X uint8, Y) for a torchvision dataset, from -- in order -- the decoded
    cache, a copy already on disk, and only then the network.  Writes the cache
    so no later session pays for this again.

    Called by the LAUNCHER before it forks, so the download can never happen
    twice at once, and by the inline path for the single-GPU case."""
    if not _DS["tv"]:
        return None
    cache = raw_cache_path()
    for c in [cache] + input_index()["raw"]:
        if os.path.exists(c):
            d = torch.load(c, map_location="cpu")
            log(f"data: decoded cache hit {c} -> X{tuple(d['X'].shape)}")
            return d["X"], d["Y"]

    got = _find_local_dataset()
    if got is None:
        if _flag("NO_DOWNLOAD"):
            raise SystemExit(
                f"{DATASET} is not on disk and this process is not allowed to "
                f"download it. The launcher was supposed to have staged "
                f"{os.path.basename(cache)} before forking -- rerun, or attach "
                f"a {DATASET} dataset as a notebook input.")
        log(f"!! {DATASET} is not on this machine, so it has to be downloaded. "
            f"On Kaggle that runs at roughly 80 kB/s "
            f"(~{169 if DATASET=='cifar100' else 170} MB, so half an hour). "
            f"To skip it next time: attach any {DATASET} dataset as an input, "
            f"or stage this session's _raw_{DATASET}_{HW}.pt.")
        t0 = time.time()
        ds = getattr(_tv().datasets, _DS["tv"])("./data", train=True,
                                                download=True)
        log(f"data: downloaded in {time.time()-t0:.0f}s")
        X = torch.as_tensor(np.array(ds.data))
        Y = torch.as_tensor(np.array(ds.targets)).long()
        if X.dim() == 3: X = X.unsqueeze(1)
        elif X.shape[-1] in (1, 3): X = X.permute(0, 3, 1, 2).contiguous()
        got = (X.to(torch.uint8), Y)

    X, Y = got
    try:
        torch.save({"X": X, "Y": Y}, cache)
        log(f"data: cached -> {os.path.basename(cache)} "
            f"({X.numel()/2**20:.0f} MiB, uint8). Stage it as an input and "
            f"the next session loads it in seconds.")
    except OSError as e:
        log(f"data: could not write the cache ({e})")
    return X, Y


def _find_imagefolder(name):
    if DATA_DIR: return DATA_DIR
    pats = []
    for root in CKPT_ROOTS:
        pats += [os.path.join(root, f"*{name}*"),
                 os.path.join(root, "*", f"*{name}*")]
    for pat in pats:
        for cand in sorted(glob.glob(pat)):
            if not os.path.isdir(cand): continue
            for sub in ("train", "Train", "images", ""):
                d = os.path.join(cand, sub) if sub else cand
                if os.path.isdir(d) and any(
                        os.path.isdir(os.path.join(d, e))
                        for e in os.listdir(d)[:50]):
                    return d
    raise FileNotFoundError(
        f"no ImageFolder tree for {name!r} under {CKPT_ROOTS}; set "
        f"DATA_DIR=/kaggle/input/<dataset>/train")


def _imagenet32_pickle(root, cap):
    """The official downsampled-ImageNet release is ten pickled batches, not a
    tree of 1.28M files.  Reading those takes seconds; walking the tree takes
    hours.  Try the pickles first."""
    import pickle
    hits = sorted(glob.glob(os.path.join(root, "**", "train_data_batch_*"),
                            recursive=True))
    if not hits:
        return None
    xs, ys = [], []
    for i, f in enumerate(hits):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        x = torch.as_tensor(d["data"]).float()/255.0
        x = x.reshape(-1, 3, 32, 32)
        y = torch.as_tensor(d["labels"]).long() - 1        # 1-indexed on disk
        xs.append(x); ys.append(y)
        log(f"  pickle {i+1}/{len(hits)}: {x.shape[0]} images")
        if sum(t.shape[0] for t in xs) >= cap:
            break
    X, Y = torch.cat(xs), torch.cat(ys)
    g = torch.Generator().manual_seed(SEED)
    sel = torch.randperm(X.shape[0], generator=g)[:cap]
    return X[sel], Y[sel]


def _load_imagefolder(name, hw, cap):
    """Decode `cap` images, then CACHE the tensor.  num_workers=0 on purpose --
    this process may itself have been spawned by the launcher and already holds
    a CUDA context, so forking DataLoader children under it risks a deadlock,
    and the work is I/O bound anyway."""
    cache = os.path.join(OUT_DIR, f"_data_{name}_{hw}_{cap}.pt")
    for c in [cache] + input_index()["data"]:
        if os.path.exists(c) and os.path.basename(c) == os.path.basename(cache):
            log(f"data cache hit: {c}")
            d = torch.load(c, map_location="cpu")
            return d["X"], d["Y"]

    tv = _tv()
    log(f"locating {name} ...")
    root = _find_imagefolder(name)
    log(f"root = {root}")

    fast = _imagenet32_pickle(root, cap) if name.startswith("imagenet") else None
    if fast is not None:
        X, Y = fast
    else:
        log("scanning the tree (this is the slow path; stage the pickled "
            "release instead and it loads in seconds) ...")
        t0 = time.time()
        ds = tv.datasets.ImageFolder(root)
        log(f"  {len(ds)} images, {len(ds.classes)} classes "
            f"({time.time()-t0:.0f}s to scan)")
        if len(ds.classes) != K:
            raise RuntimeError(f"{name} has {len(ds.classes)} classes on disk, "
                               f"DATASETS says k={K}")
        idx = list(range(0, len(ds), max(1, len(ds)//cap)))[:cap]
        g = torch.Generator().manual_seed(SEED)
        idx = [idx[i] for i in torch.randperm(len(idx), generator=g).tolist()]
        tf = tv.transforms.Compose([tv.transforms.Resize((hw, hw)),
                                    tv.transforms.ToTensor()])
        xs, ys, t0 = [], [], time.time()
        for n, i in enumerate(idx):
            img, lab = ds[i]
            xs.append(tf(img)); ys.append(lab)
            if (n + 1) % 2000 == 0:
                rate = (n + 1)/(time.time() - t0)
                log(f"  decoded {n+1}/{len(idx)}  ({rate:.0f} img/s, "
                    f"~{(len(idx)-n-1)/max(rate,1e-9)/60:.0f} min left)")
        X, Y = torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    try:
        torch.save({"X": X, "Y": Y}, cache)
        log(f"data cached -> {os.path.basename(cache)}")
    except OSError as e:
        log(f"could not cache the tensor ({e}); the other shard will redo this")
    return X, Y


def load_data_xy():
    """(X, Y) on CPU, normalised per channel; MLP sees it flattened."""
    if "xy" in _CACHE: return _CACHE["xy"]
    t0 = time.time()
    if _DS["tv"]:
        X, Y = ensure_raw()
        X = X.float()/255.0
    else:
        cap = max(TRAIN_N or 200000, EVAL_N + 4096)
        X, Y = _load_imagefolder(DATASET, HW, cap)
    mean = torch.tensor(_DS["mean"]).view(1, -1, 1, 1)
    std = torch.tensor(_DS["std"]).view(1, -1, 1, 1)
    X = (X - mean)/std
    if X.shape[1] != IN_CH or X.shape[2] != HW:
        raise RuntimeError(f"{DATASET} gives {tuple(X.shape[1:])}, "
                           f"DATASETS says ({IN_CH},{HW},{HW})")
    if MODE != "cnn":
        X = X.reshape(X.shape[0], -1)
    if TRAIN_N:
        X, Y = X[:TRAIN_N], Y[:TRAIN_N]
    X = X.contiguous(); Y = Y.contiguous()
    _CACHE["xy"] = (X, Y)
    log(f"data {DATASET}: X{tuple(X.shape)} K={K} "
        f"({X.numel()*4/2**30:.2f} GiB, {time.time()-t0:.0f}s)")
    return X, Y


# ===================================================================== TRAIN
def train(model, X, Y, epochs, tag=""):
    """Progress is logged from the first step.  A training loop that says
    nothing for an hour is the same observation as a hung one, and the whole
    point of this rewrite is that the two never look alike again."""
    model.to(DEVICE).train()
    opt = torch.optim.SGD(model.opt_groups(LR), momentum=0.9)
    wu = min(WARMUP_EPOCHS, max(1, epochs//5))

    def _lr(ep):
        if ep < wu: return (ep + 1)/wu
        return 0.5*(1.0 + math.cos(math.pi*(ep - wu)/max(epochs - wu, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
    n = X.shape[0]
    total = epochs*math.ceil(n/BATCH)
    step, t0, t_last, last = 0, time.time(), time.time(), float("nan")
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            x = X[idx].to(DEVICE, non_blocking=True)
            y = Y[idx].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()
            step += 1
            if step == 1 or time.time() - t_last > LOG_EVERY_S:
                last = float(loss.detach())
                rate = step/max(time.time() - t0, 1e-9)
                log(f"    {tag} step {step}/{total} loss={last:.4f} "
                    f"{rate:.1f} it/s eta {(total-step)/max(rate,1e-9)/60:.1f} min")
                t_last = time.time()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def acc_of(model, X, Y):
    c = t = 0
    for i in range(0, X.shape[0], 1024):
        x = X[i:i+1024].to(DEVICE); y = Y[i:i+1024].to(DEVICE)
        c += int((model(x).argmax(1) == y).sum()); t += y.numel()
    return c/max(t, 1)


# ================================================================ PRIMITIVES
def _pb(m):
    return ({k: v.detach() for k, v in m.named_parameters()},
            {k: v.detach() for k, v in m.named_buffers()})


def _call(m, p, b, x): return functional_call(m, {**p, **b}, (x,))
def _vdot(a, b): return float(sum((a[k]*b[k]).sum() for k in a))
def _vnorm(a): return math.sqrt(max(_vdot(a, a), 0.0))
def _vscale(a, c): return {k: a[k]*c for k in a}
def _vaxpy(a, c, b): return {k: a[k] + c*b[k] for k in a}


@torch.no_grad()
def loss_acc_rho(m, p, b, x, y, micro=None):
    """L, rho* = E||p_w(x)-e_y||_2 (Definition 2.3), and accuracy."""
    micro = MICRO if micro is None else micro
    n = x.shape[0]; sL = sR = 0.0; sC = 0
    for i in range(0, n, micro):
        xb = x[i:i+micro].to(DEVICE); yb = y[i:i+micro].to(DEVICE)
        lg = _call(m, p, b, xb)
        sL += float(F.cross_entropy(lg, yb, reduction="sum"))
        pr = torch.softmax(lg, 1)
        e = F.one_hot(yb, num_classes=pr.shape[1]).to(pr.dtype)
        sR += float((pr - e).norm(dim=1).sum())
        sC += int((lg.argmax(1) == yb).sum())
    return sL/n, sR/n, sC/n


def fisher_quad(m, p, b, x, delta, micro=None):
    """Delta' F(w) Delta by forward-mode AD: one JVP per micro-batch, then
    u' S(p) u with S(p) = diag(p) - p p'.  No solve, no finite difference, no
    sampling over output coordinates -- this is the whole reason the prediction
    method survives at large K where M_2 would not."""
    micro = MICRO_JVP if micro is None else micro
    n = x.shape[0]; tot = 0.0
    for i in range(0, n, micro):
        xb = x[i:i+micro].to(DEVICE)
        logits, u = _fjvp(lambda pp: _call(m, pp, b, xb), (p,), (delta,))
        pr = torch.softmax(logits, 1)
        Su = pr*u - pr*(pr*u).sum(1, keepdim=True)
        tot += float((u*Su).sum())
    return tot/n


def grad_norm(m, p, b, x, y, micro=None):
    """||grad L(w)||_2 at w.

    The whole construction assumes the endpoints are minima -- that is what
    makes B a barrier BETWEEN minima rather than a bump on a slope.  Accuracy
    above chance does not establish it: a network stopped halfway down is well
    above chance and nowhere near a minimum.  The gradient norm says so
    directly, and it costs one backward pass."""
    micro = (MICRO if micro is None else micro)
    from torch.func import grad
    n = x.shape[0]; acc = None; seen = 0
    for i in range(0, n, micro):
        xb = x[i:i+micro].to(DEVICE); yb = y[i:i+micro].to(DEVICE)
        g = grad(lambda pp: F.cross_entropy(_call(m, pp, b, xb), yb))(p)
        w = xb.shape[0]
        acc = ({k: v.detach()*w for k, v in g.items()} if acc is None
               else {k: acc[k] + g[k].detach()*w for k in acc})
        seen += w
    return math.sqrt(max(sum(float((v/seen).pow(2).sum()) for v in acc.values()), 0.0))


def flen_rq(m, p, b, xf, delta):
    """(Fisher length (1/2) D'FD, Rayleigh quotient D'FD/||D||^2)."""
    q = fisher_quad(m, p, b, xf, delta)
    return 0.5*q, q/max(_vdot(delta, delta), 1e-30)


def barrier(m, p_ref, b, pA, pB, X, Y):
    """max_t L(t) - (1/2)(L(0)+L(1)) over a TGRID-point grid, plus t*."""
    ts = np.linspace(0.0, 1.0, TGRID)
    Ls = []
    for t in ts:
        pt = {k: (1 - t)*pA[k] + t*pB[k] for k in pA}
        Ls.append(loss_acc_rho(m, pt, b, X, Y)[0])
    Ls = np.array(Ls)
    i = int(Ls.argmax())
    return float(Ls.max() - 0.5*(Ls[0] + Ls[-1])), float(ts[i]), Ls[0], Ls[-1]



# ============================================== GEOMETRY (PHASE=geodesic)
# Ported unchanged from src/02_geodesic/measure_geodesic.py.  The only
# edits are that micro-batching is sized from this task's feature maps
# rather than FashionMNIST's, and that cg_solve also returns the TRUE
# residual (see its docstring).
def act_units_per_sample(width):
    """Activation elements per sample, roughly -- only used to split a budget.

    For the CNN this is governed by feature-map area, not by the parameter
    count: c = [16w, 32w, 64w] with a pool after c1 and after c2, on HW x HW
    inputs.  Getting this from the parameter count instead (which the original
    did for the MLP) is wrong for a CNN by more than an order of magnitude."""
    if MODE == "cnn":
        return (HW*HW*16*width + (HW//2)*(HW//2)*32*width
                + (HW//4)*(HW//4)*64*width)
    return DIN + 2*width + K


def pick_micro(width):
    if GEO_MICRO_ENV:
        return max(1, min(int(GEO_MICRO_ENV), GEO_BATCH))
    per_sample = max(act_units_per_sample(width)*ACT_COPIES*4, 1)
    m = int(ACT_BUDGET // per_sample)
    m = min(max(m, 8), GEO_BATCH)
    return 1 << int(math.floor(math.log2(m)))



def fisher_vp(m, p, b, x, v, micro):
    """F v, accumulated over micro-batches.  One jvp and one vjp per chunk."""
    B = x.shape[0]; acc = None
    for i in range(0, B, micro):
        xb = x[i:i+micro]

        def f(pp): return _call(m, pp, b, xb)
        logits, Jv = _fjvp(f, (p,), (v,))
        pr = torch.softmax(logits, 1)
        s = pr*Jv - pr*(pr*Jv).sum(1, keepdim=True)
        JTs = _fvjp(f, p)[1](s)[0]
        acc = ({k: JTs[k].detach() for k in JTs} if acc is None
               else {k: acc[k] + JTs[k].detach() for k in acc})
    return {k: acc[k]/B for k in acc}


def dFz(m, p, b, x, z, v, eps, micro, rich):
    def cd(e):
        Fp = fisher_vp(m, _vaxpy(p, e, z), b, x, v, micro)
        Fm = fisher_vp(m, _vaxpy(p, -e, z), b, x, v, micro)
        return {k: (Fp[k] - Fm[k])/(2*e) for k in p}
    if rich:
        d1, d2 = cd(eps), cd(eps/2)
        return {k: (4*d2[k] - d1[k])/3 for k in d1}
    return cd(eps)


def _quad_sum(m, p, b, xb, delta):
    def f(pp): return _call(m, pp, b, xb)
    logits, u = _fjvp(f, (p,), (delta,))
    pr = torch.softmax(logits, 1)
    Su = pr*u - pr*(pr*u).sum(1, keepdim=True)
    return (u*Su).sum()


def grad_quad(m, p, b, x, delta, micro):
    """m_l = Delta^T (d_l F) Delta = grad_w <Delta, F Delta>, micro-batched so
    no graph is held over the whole batch."""
    B = x.shape[0]; acc = None
    for i in range(0, B, micro):
        gi = _grad(lambda pp: _quad_sum(m, pp, b, x[i:i+micro], delta))(p)
        acc = ({k: gi[k].detach() for k in gi} if acc is None
               else {k: acc[k] + gi[k].detach() for k in acc})
    return {k: acc[k]/B for k in acc}


def gf_vp(m, p, b, x, v, lam, micro):
    Fv = fisher_vp(m, p, b, x, v, micro)
    return {k: Fv[k] + lam*v[k] for k in v}


def lam_max(m, p, b, x, micro, iters, seed=0):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    u = {k: torch.randn(v.shape, generator=gen, device=v.device, dtype=v.dtype)
         for k, v in p.items()}
    u = _vscale(u, 1.0/max(_vnorm(u), 1e-30)); lam = 0.0
    for _ in range(iters):
        Au = fisher_vp(m, p, b, x, u, micro); lam = _vnorm(Au)
        if lam < 1e-30: break
        u = _vscale(Au, 1.0/lam)
    return lam


def cg_solve(m, p, b, x, rhs, lam, micro, x0=None, iters=300, tol=1e-6):
    """CG for G_F x = rhs.
    Returns (solution, recursive residual, TRUE residual, iterations).

    `used == iters` means CG ran out of iterations rather than converging.

    THE TWO RESIDUALS ARE NOT THE SAME NUMBER, and in float32 they are not even
    close.  CG never forms G_F x - rhs; it carries r forward by the recursion
    r <- r - a A p, which loses the very cancellation it is measuring.  On a
    tiny test problem here:

        float32   35 iters   recursion says 5.9e-13   truth is 1.2e-06
        float64   23 iters   recursion says 6.3e-13   truth is 6.3e-13

    -- the recursion understates the error by six orders of magnitude at single
    precision.  A `cg_resid` of 1e-12 therefore says nothing about how well the
    Christoffel symbol was actually solved for.  One extra matrix-vector
    product at the end, about 1% of the cost of the solve, buys a residual that
    means what a reader assumes it means, so both are recorded: cg_resid keeps
    the released run's definition, cg_resid_true is the one to trust."""
    xk = {k: (torch.zeros_like(v) if x0 is None else x0[k].clone())
          for k, v in rhs.items()}
    Ax = (gf_vp(m, p, b, x, xk, lam, micro) if x0 is not None
          else {k: torch.zeros_like(v) for k, v in rhs.items()})
    r = {k: rhs[k] - Ax[k] for k in rhs}
    pdir = {k: r[k].clone() for k in r}
    rs = _vdot(r, r); r0 = max(rs, 1e-300); used = 0
    for _ in range(iters):
        used += 1
        Ap = gf_vp(m, p, b, x, pdir, lam, micro)
        a = rs/max(_vdot(pdir, Ap), 1e-300)
        xk = {k: xk[k] + a*pdir[k] for k in xk}
        r = {k: r[k] - a*Ap[k] for k in r}
        rs2 = _vdot(r, r)
        if rs2 <= tol*tol*r0: break
        beta = rs2/max(rs, 1e-300)
        pdir = {k: r[k] + beta*pdir[k] for k in pdir}
        rs = rs2
    Ax = gf_vp(m, p, b, x, xk, lam, micro)
    true = math.sqrt(_vdot({k: Ax[k] - rhs[k] for k in rhs},
                           {k: Ax[k] - rhs[k] for k in rhs})/r0)
    return xk, math.sqrt(_vdot(r, r)/r0), true, used


def christoffel_dd(m, p, b, x, delta, lam, micro, x0=None):
    """Gamma(Delta,Delta)
    -> (Gamma, cg_resid, cg_resid_true, fd_instab, cg_iters, ||rhs||)."""
    dn = _vnorm(delta); dhat = _vscale(delta, 1.0/dn)
    d1 = dFz(m, p, b, x, dhat, dhat, FD_EPS, micro, False)
    d2 = dFz(m, p, b, x, dhat, dhat, FD_EPS/2, micro, False)
    t1r = ({k: (4*d2[k] - d1[k])/3 for k in d1} if FD_RICH else d1)
    fd_instab = _vnorm({k: d1[k] - d2[k] for k in d1})/max(_vnorm(t1r), 1e-30)
    t1 = {k: t1r[k]*dn*dn for k in t1r}
    mvec = grad_quad(m, p, b, x, delta, micro)
    rhs = {k: 2*t1[k] - mvec[k] for k in t1}
    rhs_norm = _vnorm(rhs)
    sol, resid, resid_true, used = cg_solve(m, p, b, x, rhs, lam, micro,
                                            x0=x0, iters=CG_ITERS, tol=CG_TOL)
    return _vscale(sol, 0.5), resid, resid_true, fd_instab, used, rhs_norm


def green_matrix(ts):
    n = len(ts); G = torch.zeros(n, n, dtype=torch.float64)
    for i, t in enumerate(ts):
        for j, s in enumerate(ts):
            G[i, j] = s*(1 - t) if s <= t else t*(1 - s)
    return G


def _is_oom(e):
    """An OOM has to reach the retry loop, not be swallowed by a broad except."""
    if isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    return ("out of memory" in str(e).lower()
            or "CUDA_ERROR_OUT_OF_MEMORY" in str(e))



@torch.no_grad()
def acc_of_params(m, p, b, X, Y, micro=512):
    c = t = 0
    for i in range(0, X.shape[0], micro):
        xb = X[i:i+micro].to(DEVICE); yb = Y[i:i+micro].to(DEVICE)
        c += int((_call(m, p, b, xb).argmax(1) == yb).sum()); t += yb.numel()
    return c/max(t, 1)



# =============================================================== CHECKPOINTS
def ckpt_dir():
    d = os.path.join(OUT_DIR, f"ckpt_scale_{TAG}")
    os.makedirs(d, exist_ok=True)
    return d


def ckpt_name(regime, act, w, s): return f"{regime}_{act}_w{w}_s{s}.pt"


def find_ckpt(regime, act, w, s):
    name = ckpt_name(regime, act, w, s)
    local = os.path.join(ckpt_dir(), name)
    if os.path.exists(local): return local
    return input_index()["ckpt"].get(name)


def get_or_train(regime, act, w, s, X, Y, Xe, Ye):
    """Train once, then reuse.  This is the resume seam that lets a Kaggle
    session that runs out of time be continued by staging its output."""
    path = find_ckpt(regime, act, w, s) if RESUME else None
    if path:
        try:
            d = torch.load(path, map_location=DEVICE, weights_only=False)
            net = build_net(w, act, regime)
            net.load_state_dict(d["sd"]); net.eval()
            return net, d.get("acc", float("nan")), True
        except Exception as e:
            log(f"  checkpoint {os.path.basename(path)} will not load "
                f"({type(e).__name__}: {e}) -- retraining it")
            if os.path.dirname(os.path.abspath(path)) == os.path.abspath(ckpt_dir()):
                try: os.remove(path)
                except OSError: pass
    set_seed(1000*s + 7)
    net = build_net(w, act, regime)
    t0 = time.time()
    net = train(net, X, Y, EPOCHS, tag=f"{regime}/{act}/w{w}/s{s}")
    a = acc_of(net, Xe, Ye)
    # Write through a temporary file and rename.  The heartbeat SIGTERMs the
    # workers when the session budget runs out, and a checkpoint caught
    # half-written would be loaded by the NEXT session, raise, and silently
    # cost that cell -- the failure would show up as a missing cell hours
    # later, with nothing in the log pointing at it.  os.replace is atomic, so
    # a file either exists complete or does not exist.
    dst = os.path.join(ckpt_dir(), ckpt_name(regime, act, w, s))
    tmp = dst + f".tmp{os.getpid()}"
    torch.save({"sd": net.state_dict(), "acc": a, "dataset": DATASET,
                "mode": MODE, "epochs": EPOCHS, "train_n": X.shape[0]}, tmp)
    os.replace(tmp, dst)
    log(f"  trained {regime}/{act}/w{w}/s{s}: acc={a:.4f} ({time.time()-t0:.0f}s)")
    return net, a, False



def load_ckpt(regime, act, w, s):
    path = find_ckpt(regime, act, w, s)
    if path is None:
        return None, None
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        log(f"  {os.path.basename(path)} will not load "
            f"({type(e).__name__}: {e})")
        return None, None
    return d["sd"], d.get("acc", float("nan"))



# ======================================================================= CSV
# flen_mid / rq_mid are the quantities the published Figure 5 regresses on.
# The endpoint columns are kept because the main text motivates the bound at
# the endpoint anchors, so both are recorded and the figure says which it used.
FIELDS = ["dataset", "mode", "regime", "act", "width", "nparam", "seedA", "seedB",
          "accA", "accB", "dnorm", "L_A", "L_B", "B", "t_star",
          "L_chance", "gnorm_A", "gnorm_B",
          "rho_A", "rho_B", "rho_mid",
          "flen_mid", "rq_mid", "flen_A", "flen_B", "rq_A", "rq_B", "status"]


def _warn_if_shrunk(sources, n_ok, what):
    """A merge that ends up smaller than one of the files it read has lost
    something.  Say so: the output still looks like a complete result, and the
    figure drawn from it still looks like a figure."""
    worst = 0
    worst_path = None
    for p in sources:
        try:
            with open(p) as fh:
                n = sum(1 for r in csv.DictReader(fh)
                        if str(r.get("status", "")).startswith("ok"))
        except OSError:
            continue
        if n > worst:
            worst, worst_path = n, p
    if worst > n_ok:
        log(f"!! MERGE SHRANK: {n_ok} {what} out, but "
            f"{os.path.basename(worst_path)} alone already held {worst}. "
            f"Rows from earlier sessions are not staged. Attach every "
            f"session's output as an input dataset and rerun --merge before "
            f"reading the figure; nothing has been deleted, but this merged "
            f"file is not the full result.")


def _uniq_paths(paths):
    """Same file, two spellings: the direct glob returns './x.csv' and the
    input walk returns '/abs/x.csv'.  Counting both makes a two-shard run
    report four shards."""
    seen, out = set(), []
    for p in sorted(paths):
        rp = os.path.realpath(p)
        if rp in seen: continue
        seen.add(rp); out.append(p)
    return out


def out_csv(shard=None):
    sh = SHARD_ID if shard is None else shard
    return os.path.join(OUT_DIR, f"param_scale_{TAG}_shard{sh}.csv")


def append(row):
    new = not os.path.exists(out_csv())
    with open(out_csv(), "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new: w.writeheader()
        w.writerow(row)


def read_done():
    done = set()
    if not RESUME: return done
    paths = _uniq_paths(
        glob.glob(os.path.join(OUT_DIR, f"param_scale_{TAG}*.csv"))
        + input_index()["param"])
    for path in paths:
        try:
            with open(path) as fh:
                for r in csv.DictReader(fh):
                    if not str(r.get("status", "")).startswith("ok"):
                        continue
                    done.add((r["regime"], r["act"], int(r["width"]),
                              int(r["seedA"]), int(r["seedB"])))
        except (OSError, KeyError, ValueError):
            continue
    if paths:
        log(f"resume: {len(done)} pairs already measured, from {len(paths)} CSVs")
    return done


def merge():
    # Read BOTH the per-shard CSVs and any already-merged param_scale_{TAG}.csv
    # that an earlier session left behind.  Taking shards only looks right and
    # is not: a Kaggle version's output holds what THAT version wrote, so a
    # session that measured nothing new writes no shard file, and rebuilding
    # from whatever shards happen to be staged SHRINKS the merged result.  That
    # is not hypothetical -- it turned a complete 288-row, 12-cell run into 48
    # rows and 3 cells, and the figure was redrawn from the 48 without a word.
    # The merged file is a first-class source of history, not a derived
    # artefact that can always be rebuilt.
    parts = _uniq_paths(
        glob.glob(os.path.join(OUT_DIR, f"param_scale_{TAG}*.csv"))
        + input_index()["param"])
    if not parts:
        log("nothing to merge"); return None
    # A pair that failed in one session and was re-measured in the next has
    # TWO rows: an old 'fail:' and a newer 'ok'.  Keeping whichever came first
    # silently throws the successful re-measurement away and leaves the run
    # looking permanently broken -- resume would never be able to repair
    # anything.  An 'ok' always wins; between two of equal standing the later
    # row wins, because it was measured later.
    best = {}
    order = []
    for p in parts:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                key = (r.get("regime"), r.get("act"), r.get("width"),
                       r.get("seedA"), r.get("seedB"))
                if key not in best:
                    order.append(key)
                prev = best.get(key)
                if prev is None or (
                        not str(prev.get("status", "")).startswith("ok")):
                    best[key] = r
    rows = [best[k] for k in order]
    n_ok = sum(1 for r in rows if str(r.get("status", "")).startswith("ok"))
    if len(rows) != n_ok:
        log(f"merge: {len(rows)-n_ok} of {len(rows)} pairs are still 'fail:' "
            f"rows -- rerun to retry them, they are not cached as done")
    _warn_if_shrunk(parts, n_ok, "ok pairs")
    dst = os.path.join(OUT_DIR, f"param_scale_{TAG}.csv")
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)
    log(f"merged {len(rows)} rows from {len(parts)} shards -> {os.path.basename(dst)}")
    return dst



# ------------------------------------------- PHASE=geodesic rows
FIELDS_GEO = ["dataset", "mode", "regime", "act", "width", "nparam",
          "seedA", "seedB", "accA", "accB", "smooth",
          "dnorm", "dev_geo", "dev_rel", "gamma_mid",
          "flen_A", "flen_mid", "flen_B", "rq_A", "rq_mid", "rq_B",
          "lam", "lam_rel", "lam_max", "cg_resid", "cg_resid_true",
          "cg_iters", "fd_instab",
          "rhs_norm", "micro", "secs", "status"]


def out_csv_geo(shard=None):
    sh = SHARD_ID if shard is None else shard
    return os.path.join(OUT_DIR, f"param_geo_scale_{TAG}_shard{sh}.csv")


def append_geo(row):
    new = not os.path.exists(out_csv_geo())
    with open(out_csv_geo(), "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS_GEO, extrasaction="ignore")
        if new: w.writeheader()
        w.writerow(row)


def geo_csv_paths():
    return _uniq_paths(
        glob.glob(os.path.join(OUT_DIR, f"param_geo_scale_{TAG}*.csv"))
        + input_index()["geoparam"])


def read_done_geo():
    """Resume at the granularity of one (cell, width, pair, lam_rel).

    A Christoffel solve is the most expensive thing in the paper, so the resume
    key has to include lam_rel: a lambda sweep that got through 1e-1 and died
    in 1e-2 must not redo 1e-1.  Only 'ok' rows count as done -- a failed row
    is a thing to retry, not a result."""
    done = set()
    if not RESUME: return done
    paths = [p for p in geo_csv_paths() if "_shard" in os.path.basename(p)
             or os.path.basename(p) == f"param_geo_scale_{TAG}.csv"]
    for path in paths:
        try:
            with open(path) as fh:
                for r in csv.DictReader(fh):
                    if not str(r.get("status", "")).startswith("ok"):
                        continue
                    done.add((r["regime"], r["act"], int(r["width"]),
                              int(r["seedA"]), int(r["seedB"]),
                              f'{float(r["lam_rel"]):.6g}'))
        except (OSError, KeyError, ValueError, TypeError):
            continue
    if paths:
        log(f"resume: {len(done)} (pair, lam) rows already measured, "
            f"from {len(paths)} CSVs")
    return done


def merge_geo():
    # Both the shards and any already-merged CSV -- see merge() for why taking
    # shards alone silently shrinks the result.
    parts = geo_csv_paths()
    if not parts:
        log("nothing to merge"); return None
    best, order = {}, []
    for p in parts:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                key = (r.get("regime"), r.get("act"), r.get("width"),
                       r.get("seedA"), r.get("seedB"), r.get("lam_rel"))
                if key not in best: order.append(key)
                prev = best.get(key)
                if prev is None or not str(prev.get("status", "")).startswith("ok"):
                    best[key] = r
    rows = [best[k] for k in order]
    n_ok = sum(1 for r in rows if str(r.get("status", "")).startswith("ok"))
    _warn_if_shrunk(parts, n_ok, "ok rows")
    dst = os.path.join(OUT_DIR, f"param_geo_scale_{TAG}.csv")
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS_GEO, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)
    log(f"merged {len(rows)} rows ({n_ok} ok) from {len(parts)} shards "
        f"-> {os.path.basename(dst)}")
    return dst



# ================================================= MEASURE (PHASE=geodesic)
def measure_pair(ref, p_ref, b_ref, Xgeo, pA, pB, lam_rel, lmax, micro):
    """One (pair, lam_rel): Christoffel along the straight line, then the Green
    integral, then sup_t ||xi||.  Returns a dict of measurements."""
    delta = {k: pB[k] - pA[k] for k in pA}
    dn = _vnorm(delta); d2 = max(dn*dn, 1e-30)
    lam = max(lam_rel*lmax, 1e-12)

    def _flen(pt):
        q = _vdot(delta, fisher_vp(ref, pt, b_ref, Xgeo, delta, micro))
        return 0.5*q, q/d2

    pmid = {k: 0.5*(pA[k] + pB[k]) for k in pA}
    flA, rqA = _flen(pA); flM, rqM = _flen(pmid); flB, rqB = _flen(pB)

    ts = list(np.linspace(0.0, 1.0, GEO_TGRID))
    gammas, resids, trues, fdis, iters_used, rhs_ns = [], [], [], [], [], []
    x0 = None
    try:
        for tt in ts:
            pt = {k: (1 - tt)*pA[k] + tt*pB[k] for k in pA}
            g, res, res_t, fdi, used, rn = christoffel_dd(
                ref, pt, b_ref, Xgeo, delta, lam, micro, x0=x0)
            x0 = g                                  # warm-start the next CG
            # Park each Gamma on the CPU: GEO_TGRID of them at width 8 is
            # several times the model itself, and the GPU still has to hold the
            # CG workspace.
            gammas.append({k: v.detach().cpu() for k, v in g.items()})
            resids.append(res); trues.append(res_t); fdis.append(fdi)
            iters_used.append(used); rhs_ns.append(rn)
        mid = int(np.argmin([abs(t - 0.5) for t in ts]))
        gamma_mid = _vnorm(gammas[mid])
        G = green_matrix(ts); dt = ts[1] - ts[0]
        keys = list(delta.keys()); n = len(ts); sup = 0.0
        for ti in range(n):
            xi = None
            for si in range(n):
                w_ = float(G[ti, si])*dt
                if w_ == 0.0: continue
                xi = {k: (gammas[si][k]*w_ if xi is None
                          else xi[k] + gammas[si][k]*w_) for k in keys}
            sup = max(sup, _vnorm(xi) if xi is not None else 0.0)
    finally:
        try: del gammas, x0
        except Exception: pass
        if DEVICE == "cuda": torch.cuda.empty_cache()

    return dict(dnorm=dn, dev_geo=sup, dev_rel=sup/max(dn, 1e-30),
                gamma_mid=gamma_mid, flen_A=flA, flen_mid=flM, flen_B=flB,
                rq_A=rqA, rq_mid=rqM, rq_B=rqB, lam=lam, lam_max=lmax,
                cg_resid=max(resids), cg_resid_true=max(trues),
                cg_iters=max(iters_used),
                fd_instab=max(fdis), rhs_norm=max(rhs_ns))


_XCHECK = {"n": 0, "worst": 0.0, "warned": False}


def run_worker_geo():
    open_log()
    print(f"[s{SHARD_ID}] WORKER-READY pid={os.getpid()} device={DEVICE} "
          f"cuda={os.environ.get('CUDA_VISIBLE_DEVICES','-')}", flush=True)
    log(f"shard {SHARD_ID}/{NUM_SHARDS} dataset={DATASET} K={K} mode={MODE} "
        f"out={OUT_DIR}")
    input_index()
    cells = [(r, a) for r in REGIMES for a in ACTS]
    mine = [c for i, c in enumerate(cells) if i % NUM_SHARDS == SHARD_ID]
    pair_list = list(itertools.combinations(range(NSEEDS), 2))[:GEO_PAIRS]
    lam_rels = LAM_SWEEP if LAM_SWEEP else [LAM_REL]
    ns = sorted({a for (_, a) in mine if a not in SMOOTH_ACTS})
    if ns:
        log(f"!! {ns} are not C^3 -- the finite-difference Christoffel is not "
            f"valid there; those rows carry smooth=0 / status=ok:nonsmooth")
    log(f"cells={len(mine)}/{len(cells)} widths={WIDTHS} pairs={pair_list} "
        f"lam_rels={lam_rels} geo_batch={GEO_BATCH} tgrid={GEO_TGRID} "
        f"cg_iters={CG_ITERS} cg_tol={CG_TOL:.1e}")

    # COVERAGE PREFLIGHT.  A Kaggle version's output holds only what THAT
    # version wrote; the checkpoints an earlier session trained live in the
    # earlier version's output.  Attach one of them and the grid looks fine
    # until, hours in, nine cells turn out to have nothing to measure.  Count
    # first, and say exactly which cells cannot be done, in the first seconds.
    seeds_needed = sorted({s for pr in pair_list for s in pr})
    have = miss = 0
    dead, partial = [], []
    for (regime, act) in mine:
        for w in WIDTHS:
            n = sum(1 for s in seeds_needed if find_ckpt(regime, act, w, s))
            have += n; miss += len(seeds_needed) - n
            if n == 0: dead.append(f"{regime}/{act}/w{w}")
            elif n < len(seeds_needed): partial.append(f"{regime}/{act}/w{w}({n})")
    fittable = sum(1 for (regime, act) in mine
                   if sum(1 for w in WIDTHS
                          if sum(1 for s in seeds_needed
                                 if find_ckpt(regime, act, w, s)) >= 2) >= 3)
    log(f"checkpoints: {have} found, {miss} missing (of {have+miss})")
    if miss:
        log(f"!! {len(dead)} of {len(mine)*len(WIDTHS)} (cell,width) slots have "
            f"NO checkpoint: {dead[:8]}{' ...' if len(dead) > 8 else ''}")
        if partial:
            log(f"!! partial: {partial[:8]}{' ...' if len(partial) > 8 else ''}")
        log(f"!! only {fittable} of {len(mine)} cells have >=3 usable widths, "
            f"so only {fittable} can yield an exponent. This file does not "
            f"train. If that is not what you expected, attach the output of "
            f"EVERY measure_scale.py session as an input dataset -- one "
            f"session's output contains only the checkpoints that session "
            f"wrote.")
    else:
        log("checkpoints: complete for this shard")


    # Only now is it worth paying for the data: on a box with nothing
    # staged that download is half an hour, and finding out afterwards
    # that there was nothing to measure wastes all of it.
    X, Y = load_data_xy()
    Xgeo = X[:GEO_BATCH].to(DEVICE)
    done = read_done_geo()
    n_ok = n_bad = n_skip = 0
    stopped_early = False
    for (regime, act) in mine:
        if stopped_early: break
        smooth = 1 if act in SMOOTH_ACTS else 0
        st_ok = "ok" if smooth else "ok:nonsmooth"
        for w in WIDTHS:
            if stopped_early: break
            todo = [(i, j, lr) for (i, j) in pair_list for lr in lam_rels
                    if (regime, act, w, i, j, f"{lr:.6g}") not in done]
            if not todo:
                log(f"  {regime}/{act}/w{w}: all pairs already measured")
                continue
            sds, accs = {}, {}
            for s in sorted({s for pr in pair_list for s in pr}):
                sd, acc = load_ckpt(regime, act, w, s)
                if sd is not None:
                    sds[s], accs[s] = sd, acc
            if len(sds) < 2:
                log(f"  skip {regime}/{act}/w{w}: found {len(sds)} checkpoints "
                    f"-- stage the measure_scale.py output as an input dataset")
                append_geo(dict(dataset=DATASET, mode=MODE, regime=regime, act=act,
                            width=w, smooth=smooth, status="skip:nockpt"))
                n_skip += 1
                continue

            ref = build_net(w, act, regime).eval()
            p_ref, b_ref = _pb(ref)
            npar = sum(v.numel() for v in p_ref.values())
            ag, gs = perm_spec(ref)
            micro = pick_micro(w)
            log(f"=== {regime}/{act}/w{w}  P={npar/1e6:.3f}M  micro={micro} "
                f"({math.ceil(GEO_BATCH/micro)} chunks/fisher_vp)  "
                f"{len(todo)} rows to do ===")

            while True:                     # on OOM: halve micro, redo the cell
                try:
                    for (i, j) in pair_list:
                        if all((regime, act, w, i, j, f"{lr:.6g}") in done
                               for lr in lam_rels):
                            continue
                        if i not in sds or j not in sds:
                            continue
                        if out_of_time(margin_h=_worst_row_h()):
                            log(f"TIME BUDGET: stopping before "
                                f"{regime}/{act}/w{w} {i}-{j} -- everything "
                                f"measured so far is on disk; the next session "
                                f"resumes from it.")
                            stopped_early = True
                            break
                        perms = weight_matching(ag, gs, sds[i], sds[j],
                                                MATCH_ITERS, seed=i*13 + j)
                        sdBp = apply_perm(sds[j], ag, perms)
                        pA = {k: sds[i][k].to(DEVICE) for k in p_ref}
                        pB = {k: sdBp[k].to(DEVICE) for k in p_ref}
                        lmax = lam_max(ref, pA, b_ref, Xgeo, micro,
                                       POWER_ITERS, seed=17)
                        for lr in lam_rels:
                            if (regime, act, w, i, j, f"{lr:.6g}") in done:
                                continue
                            t0 = time.time()
                            try:
                                m = measure_pair(ref, p_ref, b_ref, Xgeo,
                                                 pA, pB, lr, lmax, micro)
                            except Exception as e:
                                if _is_oom(e): raise
                                n_bad += 1
                                traceback.print_exc()
                                append_geo(dict(dataset=DATASET, mode=MODE,
                                            regime=regime, act=act, width=w,
                                            seedA=i, seedB=j, lam_rel=lr,
                                            smooth=smooth,
                                            status=f"fail:{type(e).__name__}"))
                                continue
                            secs = time.time() - t0
                            _note_row_time(secs)
                            n_ok += 1
                            append_geo(dict(
                                dataset=DATASET, mode=MODE, regime=regime,
                                act=act, width=w, nparam=npar, seedA=i,
                                seedB=j, accA=f"{accs.get(i, float('nan')):.6f}",
                                accB=f"{accs.get(j, float('nan')):.6f}",
                                smooth=smooth, lam_rel=lr, micro=micro,
                                secs=f"{secs:.1f}", status=st_ok,
                                **{k: (f"{v:.6e}" if isinstance(v, float) else v)
                                   for k, v in m.items()}))
                            cap = m["cg_iters"] >= CG_ITERS
                            log(f"  {regime}/{act}/w{w} {i}-{j} lam_rel={lr}: "
                                f"||d||={m['dnorm']:.3g} "
                                f"dev_rel={m['dev_rel']:.3e} "
                                f"gamma_mid={m['gamma_mid']:.3e} "
                                f"flen_mid={m['flen_mid']:.3e} "
                                f"cgres<={m['cg_resid_true']:.1e} "
                                f"cg_it={m['cg_iters']}"
                                f"{'  !!CG HIT THE CAP' if cap else ''} "
                                f"fd={m['fd_instab']:.2e} ({secs:.0f}s)")
                    break
                except Exception as e:
                    if not _is_oom(e) or micro <= 8: raise
                    micro = max(8, micro//2)
                    log(f"  !! OOM -> micro down to {micro} "
                        f"({math.ceil(GEO_BATCH/micro)} chunks), redoing "
                        f"this cell; finished pairs are skipped on the retry")
                    if DEVICE == "cuda": torch.cuda.empty_cache()
            del sds, ref
            import gc; gc.collect()
            if DEVICE == "cuda": torch.cuda.empty_cache()

    n_row = sum(max(0, sum(1 for _ in open(f)) - 1)
                for f in glob.glob(os.path.join(
                    OUT_DIR, f"param_geo_scale_{TAG}_shard*.csv")))
    log(f"{'STOPPED EARLY' if stopped_early else 'done'}: {n_row} rows on "
        f"disk, {n_ok} measured / {n_bad} failed / {n_skip} cells without "
        f"checkpoints this session ({(time.time()-_T_START)/3600:.1f} h)")
    if n_bad and not n_ok:
        log(f"!! every one of {n_bad} attempted rows failed -- see the "
            f"traceback above; exiting non-zero so the launcher says so")
        raise SystemExit(2)


# ============================================================ EXPONENTS + FIG
def _fit(widths, vals):
    """OLS of log Q on log n over per-width medians, App. C's procedure.
    Returns the DECAY exponent alpha (Q ~ n^-alpha) and R^2."""
    w = np.asarray(widths, float); v = np.asarray(vals, float)
    m = np.isfinite(w) & np.isfinite(v) & (v > 0)
    if m.sum() < 3: return np.nan, np.nan
    x, y = np.log(w[m]), np.log(v[m])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope*x + intercept
    ss_res = float(((y - pred)**2).sum())
    ss_tot = float(((y - y.mean())**2).sum())
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return -slope, r2


def cell_table(src):
    import pandas as pd
    d = pd.read_csv(src)
    n_all = len(d)
    d = d[d["status"].astype(str).str.startswith("ok")]
    if len(d) < n_all:
        log(f"cells: dropped {n_all-len(d)} failed rows")
    for c in ("B", "flen_mid", "rq_mid", "flen_A", "flen_B", "rq_A", "rq_B",
              "rho_A", "rho_B", "rho_mid", "accA", "accB", "width",
              "L_A", "gnorm_A"):
        d[c] = d[c].astype(float) if c in d else np.nan
    d["flen_end"] = 0.5*(d["flen_A"] + d["flen_B"])
    d["rq_end"] = 0.5*(d["rq_A"] + d["rq_B"])
    rows = []
    for (r, a), g in d.groupby(["regime", "act"]):
        med = g.groupby("width")[["B", "flen_mid", "rq_mid", "flen_end",
                                  "rq_end", "rho_A", "rho_mid",
                                  "accA", "L_A", "gnorm_A"]].median()
        ws = med.index.values
        aB, rB = _fit(ws, med["B"].values)
        aL, rL = _fit(ws, med["flen_mid"].values)
        aR, rR = _fit(ws, med["rq_mid"].values)
        aLe, _ = _fit(ws, med["flen_end"].values)
        aRe, _ = _fit(ws, med["rq_end"].values)
        rows.append(dict(dataset=DATASET, mode=MODE, regime=r, act=a,
                         n_widths=len(ws), n_pairs=len(g),
                         acc_min=float(med["accA"].min()),
                         L_end_max=float(med["L_A"].max()),
                         loss_drop_min=float(1.0 - med["L_A"].max()/math.log(K)),
                         gnorm_max=float(med["gnorm_A"].max()),
                         rho_end=float(med["rho_A"].iloc[-1]),
                         rho_mid=float(med["rho_mid"].iloc[-1]),
                         alpha_B=aB, r2_B=rB,
                         alpha_LF=aL, r2_LF=rL, alpha_RF=aR, r2_RF=rR,
                         alpha_LF_end=aLe, alpha_RF_end=aRe))
    out = pd.DataFrame(rows)
    dst = os.path.join(OUT_DIR, f"cell_scale_{TAG}.csv")
    out.to_csv(dst, index=False)
    log(f"cells -> {os.path.basename(dst)}  ({len(out)} cells)")
    if len(out):
        log("  " + out[["regime", "act", "n_widths", "loss_drop_min",
                        "gnorm_max", "alpha_B", "r2_B", "alpha_LF",
                        "alpha_RF"]]
            .to_string(index=False).replace("\n", "\n  "))
    return out


def _released_reference():
    """The published MNIST/FashionMNIST/Teacher-Student result, as per-cell
    (alpha_LF, alpha_RF, alpha_B) triples -- the 36 cells behind the paper's
    "0.90 against 0.03".

    This follows scripts/Figure5.py exactly: the barrier comes
    from data/final/{arch}_pairs.csv, the Fisher length and Rayleigh quotient
    from data/geodesic/{arch}_pairs.csv, each exponent an OLS fit on the seed
    medians of its own cell, the three quantities then merged on
    (arch, regime, act).  Note it fits each quantity independently rather than
    joining per seed pair -- that is the released procedure, and matching it is
    what makes the two clouds comparable.

    It is a REFERENCE, not a controlled baseline: the published run used a
    scalar input normalisation, so read it as "where MNIST sits", not as an arm
    of a matched experiment."""
    import pandas as pd
    idx = input_index()
    if not (idx["final"] and idx["geo"]):
        log("reference: no released *_pairs.csv found -- drawing the new task "
            "only. Point REF_DIR at the repo's data/ directory (or stage "
            "data/final and data/geodesic on Kaggle) to overlay the "
            "published cells.")
        return None

    def _cells(path, col):
        d = pd.read_csv(path)
        if "status" in d:
            d = d[d["status"].astype(str).str.startswith("ok")]
        if col not in d:
            return None
        d = d.dropna(subset=[col])
        d = d[d["act"].isin(ACTS)]
        out = {}
        for (r, a), g in d.groupby(["regime", "act"]):
            med = g.groupby("width")[col].median()
            out[(r, a)] = _fit(med.index.values.astype(float), med.values)[0]
        return out

    rows, archs = [], []
    for arch in sorted(set(idx["final"]) & set(idx["geo"])):
        try:
            cB = _cells(idx["final"][arch], "B")
            cL = _cells(idx["geo"][arch], "flen_mid")
            cR = _cells(idx["geo"][arch], "rq_mid")
        except Exception as e:
            log(f"reference: {arch} unreadable ({type(e).__name__}: {e})")
            continue
        if not (cB and cL and cR):
            continue
        for k in sorted(set(cB) & set(cL) & set(cR)):
            t = (cL[k], cR[k], cB[k])
            if np.isfinite(t).all():
                rows.append(t); archs.append(arch)
    if not rows:
        log("reference: released CSVs found but no cell survived the fit")
        return None
    out = np.array(rows)
    r2L, slL = _r2_slope(out[:, 0], out[:, 2])
    r2R, _ = _r2_slope(out[:, 1], out[:, 2])
    log(f"reference: {len(out)} published cells from "
        f"{sorted(set(archs))} -> Fisher R2={r2L:.3f} slope={slL:.2f}, "
        f"Rayleigh R2={r2R:.3f}   (the paper reports 0.90 / 1.13 / 0.03)")
    # The control's failure is largely a CROSS-architecture effect, so say
    # what it looks like one architecture at a time -- an MLP-only comparison
    # is a weaker test and the reader should be able to see that here.
    for a in sorted(set(archs)):
        m = np.array([x == a for x in archs])
        if m.sum() >= 3:
            rl, _ = _r2_slope(out[m, 0], out[m, 2])
            rr, _ = _r2_slope(out[m, 1], out[m, 2])
            log(f"  {a:>4}-only ({int(m.sum())} cells): "
                f"Fisher R2={rl:.3f}, Rayleigh R2={rr:.3f}")
    return out


def _r2_slope(x, y):
    if len(x) < 3: return np.nan, np.nan
    sl = np.polyfit(x, y, 1)[0]
    return np.corrcoef(x, y)[0, 1]**2, sl


def make_figure(cells):
    """One figure that answers the question: does the relation that predicts
    the barrier on MNIST still predict it on a larger task?

    (a) the predictor, (b) the control.  The published MNIST cells sit behind
    the new ones in both panels, so "still works" is visible as the same
    relationship on a bigger problem rather than asserted in a caption."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {"ntk": "#0072B2", "sp": "#56B4E9", "mup": "#D55E00", "ink": "#262626"}
    LAB = {"ntk": "NTK-lazy", "sp": "Standard", "mup": r"$\mu$P"}
    MK = {"gelu": "o", "tanh": "^", "swish": "s", "softplus": "D", "relu": "v"}
    REF = "#b9b9b9"

    plt.rcParams.update({"font.size": 8, "axes.facecolor": "#F2F2F2",
                         "axes.edgecolor": "#262626", "axes.linewidth": 0.8,
                         "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.85),
                             gridspec_kw={"wspace": 0.28})
    ref = _released_reference()
    verdict = {}

    for ax, col, ridx, name, title in (
            (axes[0], "alpha_LF", 0, r"Fisher length $\alpha_{\mathcal{L}_F}$",
             "the predictor"),
            (axes[1], "alpha_RF", 1, r"Rayleigh $\alpha_{\mathcal{R}_F}$",
             "the control: must still fail")):
        d = cells.dropna(subset=[col, "alpha_B"])
        if ref is not None:
            ax.plot(ref[:, ridx], ref[:, 2], ls="none", marker="o", ms=3.2,
                    markerfacecolor="none", markeredgecolor=REF,
                    markeredgewidth=0.9, zorder=2)
            r2r, slr = _r2_slope(ref[:, ridx], ref[:, 2])
            xs = np.linspace(ref[:, ridx].min(), ref[:, ridx].max(), 10)
            ic = ref[:, 2].mean() - slr*ref[:, ridx].mean()
            ax.plot(xs, slr*xs + ic, color=REF, lw=1.2, ls=(0, (4, 1.6)),
                    zorder=3)
            ax.annotate(f"published MNIST  $R^2$={r2r:.2f}",
                        xy=(0.035, 0.035), xycoords="axes fraction",
                        fontsize=6.1, color="#8f8f8f", style="italic")
        for _, r in d.iterrows():
            ax.plot([r[col]], [r["alpha_B"]], ls="none",
                    marker=MK.get(r["act"], "o"), ms=5.0,
                    markerfacecolor=C.get(r["regime"], "#666"),
                    markeredgecolor="white", markeredgewidth=0.6, zorder=7)
        r2, sl = _r2_slope(d[col].values, d["alpha_B"].values)
        verdict[col] = r2
        if np.isfinite(r2):
            x = d[col].values
            ic = d["alpha_B"].values.mean() - sl*x.mean()
            xs = np.linspace(x.min(), x.max(), 10)
            ax.plot(xs, sl*xs + ic, color=C["ink"], lw=1.5, zorder=6)
            ax.text(0.035, 0.965, f"{DATASET}\n$R^2$ = {r2:.2f}"
                                  f"\nslope = {sl:.2f}",
                    transform=ax.transAxes, fontsize=7.2, va="top",
                    linespacing=1.35, weight="bold")
        ax.set_xlabel(name)
        ax.set_title(title, fontsize=8)
        ax.grid(True, color="white", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(r"barrier exponent $\alpha_B$")

    rL, rR = verdict.get("alpha_LF", np.nan), verdict.get("alpha_RF", np.nan)
    if np.isfinite(rL) and np.isfinite(rR):
        if rL >= 0.6 and rR <= 0.35:
            msg, col = ("the relation transfers: the predictor holds and the "
                        "control still fails"), "#1a7f37"
        elif rL >= 0.6:
            msg, col = ("inconclusive: both predict, so the test has lost its "
                        "power at this scale"), "#9a6700"
        else:
            msg, col = "the relation does not transfer at this scale", "#b42318"
        fig.text(0.5, -0.055, msg, ha="center", fontsize=7.2, color=col,
                 style="italic")

    h = [plt.Line2D([], [], ls="none", marker="o", ms=5.0, color=C[r],
                    markerfacecolor=C[r], markeredgecolor="white",
                    label=LAB[r]) for r in REGIMES if r in C]
    h += [plt.Line2D([], [], ls="none", marker=MK[a], ms=4.4, color="#6f6f6f",
                     markerfacecolor="#6f6f6f", markeredgecolor="white",
                     label=a) for a in ACTS if a in MK]
    if ref is not None:
        h += [plt.Line2D([], [], ls="none", marker="o", ms=3.4, color=REF,
                         markerfacecolor="none", markeredgecolor=REF,
                         label="MNIST (published)")]
    fig.legend(handles=h, ncol=min(len(h), 5), loc="lower center",
               bbox_to_anchor=(0.5, 0.005), fontsize=6.4, columnspacing=1.0,
               handletextpad=0.4)
    fig.suptitle(f"The barrier prediction on {DATASET} "
                 f"({MODE.upper()}, K={K}, {len(cells)} cells)",
                 fontsize=8.5, y=1.01)
    fig.subplots_adjust(bottom=0.33, top=0.86)
    stem = os.path.join(OUT_DIR, f"figscale_{TAG}")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"figure -> {os.path.basename(stem)}.pdf/.png")
    log(f"VERDICT on {DATASET}: Fisher length R2={rL:.2f}, "
        f"Rayleigh R2={rR:.2f}   (published MNIST: 0.90 and 0.03)")
    return rL, rR


def finalize():
    """Never let the last step of a ten-hour run be the thing that throws.
    A run with too few cells has a diagnosis to report, not a traceback."""
    src = merge()
    if not src:
        return None
    cells = cell_table(src)
    if len(cells) < 3:
        log(f"!! only {len(cells)} usable cells -- the regression needs at "
            f"least 3 points, so no figure was drawn. Check the 'fail:' rows "
            f"in {os.path.basename(src)} and the shard logs, then resume.")
        return None
    try:
        return make_figure(cells)
    except Exception as e:
        traceback.print_exc()
        log(f"!! the figure failed to draw ({type(e).__name__}: {e}); the CSVs "
            f"are on disk and intact, so rerun with --merge after fixing it")
        return None



# ------------------------------------------ PHASE=geodesic exponents
def cell_table_geo(src):
    import pandas as pd
    d = pd.read_csv(src)
    n_all = len(d)
    d = d[d["status"].astype(str).str.startswith("ok")]
    # A row whose CG ran out of iterations is not a measurement.  Drop it here
    # rather than letting it set an exponent, and say how many went.
    d["cg_iters"] = pd.to_numeric(d["cg_iters"], errors="coerce")
    capped = int((d["cg_iters"] >= CG_ITERS).sum())
    if capped:
        log(f"cells: {capped} of {len(d)} ok rows had CG hit the "
            f"{CG_ITERS}-iteration cap -- excluded from the fits")
        d = d[d["cg_iters"] < CG_ITERS]
    if len(d) < n_all:
        log(f"cells: using {len(d)} of {n_all} rows")
    if LAM_SWEEP:
        d = d[np.isclose(d["lam_rel"].astype(float), LAM_REL)]
    for c in ("dev_rel", "dev_geo", "gamma_mid", "flen_mid", "rq_mid",
              "dnorm", "width", "cg_resid", "cg_resid_true", "fd_instab"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    rows = []
    for (r, a), g in d.groupby(["regime", "act"]):
        med = g.groupby("width")[["dev_rel", "dev_geo", "gamma_mid",
                                  "flen_mid", "rq_mid", "dnorm",
                                  "cg_resid_true", "fd_instab"]].median()
        ws = med.index.values
        aD, rD = _fit(ws, med["dev_rel"].values)
        aG, rG = _fit(ws, med["gamma_mid"].values)
        aL, _ = _fit(ws, med["flen_mid"].values)
        aR, _ = _fit(ws, med["rq_mid"].values)
        rows.append(dict(dataset=DATASET, mode=MODE, regime=r, act=a,
                         n_widths=len(ws), n_pairs=len(g),
                         dev_rel_min=float(med["dev_rel"].min()),
                         dev_rel_max=float(med["dev_rel"].max()),
                         cg_resid_true_max=float(med["cg_resid_true"].max()),
                         fd_instab_max=float(med["fd_instab"].max()),
                         alpha_dev=aD, r2_dev=rD,
                         alpha_gamma=aG, r2_gamma=rG,
                         alpha_LF=aL, alpha_RF=aR))
    out = pd.DataFrame(rows)
    dst = os.path.join(OUT_DIR, f"cell_geo_scale_{TAG}.csv")
    out.to_csv(dst, index=False)
    log(f"cells -> {os.path.basename(dst)}  ({len(out)} cells)")
    if len(out):
        log("  " + out[["regime", "act", "n_widths", "dev_rel_min",
                        "dev_rel_max", "alpha_dev", "r2_dev",
                        "fd_instab_max"]]
            .to_string(index=False).replace("\n", "\n  "))
    return out


def _published_dev():
    """Median D_rel per (arch, regime, width) from data/geodesic/*_pairs.csv."""
    import pandas as pd
    idx = input_index()
    out = {}
    for arch, path in idx["geo"].items():
        try:
            d = pd.read_csv(path)
            d = d[d["status"].astype(str).str.startswith("ok")]
            if "dev_rel" not in d: continue
            d["dev_rel"] = pd.to_numeric(d["dev_rel"], errors="coerce")
            d["width"] = pd.to_numeric(d["width"], errors="coerce")
            out[arch] = d.groupby(["regime", "width"])["dev_rel"].median()
        except Exception as e:
            log(f"published dev_rel for {arch} unreadable "
                f"({type(e).__name__}: {e})")
    return out


def make_figure_geo(cells, src):
    """(a) D_rel against width on the new task, with the published CNN curve
    behind it; (b) the validity columns, because a D_rel from a CG that never
    converged is not a measurement and the reader has to be able to see that."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    C = {"ntk": "#0072B2", "sp": "#56B4E9", "mup": "#D55E00", "ink": "#262626"}
    LAB = {"ntk": "NTK-lazy", "sp": "Standard", "mup": r"$\mu$P"}
    REF = "#b9b9b9"

    d = pd.read_csv(src)
    d = d[d["status"].astype(str).str.startswith("ok")]
    for c in ("dev_rel", "width", "cg_iters", "fd_instab", "cg_resid"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if LAM_SWEEP:
        d = d[np.isclose(pd.to_numeric(d["lam_rel"], errors="coerce"), LAM_REL)]
    good = d[d["cg_iters"] < CG_ITERS]

    plt.rcParams.update({"font.size": 8, "axes.facecolor": "#F2F2F2",
                         "axes.edgecolor": "#262626", "axes.linewidth": 0.8,
                         "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.85),
                             gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    pub = _published_dev()
    if "cnn" in pub:
        for reg, sub in pub["cnn"].groupby(level=0):
            ax.plot(sub.index.get_level_values(1), sub.values, ls=(0, (4, 1.6)),
                    lw=1.1, color=REF, marker="o", ms=2.8,
                    markerfacecolor="none", zorder=2)
        ax.annotate("published CNN / FashionMNIST", xy=(0.03, 0.955),
                    xycoords="axes fraction", fontsize=6.1, color="#8f8f8f",
                    style="italic", va="top")
    for reg, sub in good.groupby("regime"):
        med = sub.groupby("width")["dev_rel"].median()
        ax.plot(med.index.values, med.values, marker="o", ms=4.5, lw=1.4,
                color=C.get(reg, "#666"), markeredgecolor="white",
                markeredgewidth=0.6, label=LAB.get(reg, reg), zorder=6)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("width multiplier $w$")
    ax.set_ylabel(r"relative deviation $D_{\mathrm{rel}}$")
    ax.set_title(f"the linear path as a proxy, {DATASET}", fontsize=8)
    ax.grid(True, color="white", lw=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=6.2, frameon=False, loc="lower left",
              bbox_to_anchor=(0.0, 0.0), borderaxespad=0.3)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    ax = axes[1]
    n_cap = int((d["cg_iters"] >= CG_ITERS).sum())
    for reg, sub in good.groupby("regime"):
        ax.scatter(sub["fd_instab"], sub["dev_rel"], s=11,
                   color=C.get(reg, "#666"), edgecolor="white", linewidth=0.4,
                   zorder=6, label=LAB.get(reg, reg))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("finite-difference instability")
    ax.set_title("is the number trustworthy?", fontsize=8)
    ax.grid(True, color="white", lw=0.6); ax.set_axisbelow(True)
    ax.text(0.03, 0.96, f"{len(good)} usable rows\n{n_cap} dropped (CG hit "
                        f"the cap)", transform=ax.transAxes, fontsize=6.4,
            va="top", linespacing=1.35)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    fig.suptitle(f"Geodesic deviation on {DATASET} "
                 f"({MODE.upper()}, K={K}, {len(cells)} cells)",
                 fontsize=8.5, y=1.01)
    stem = os.path.join(OUT_DIR, f"figgeo_scale_{TAG}")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"figure -> {os.path.basename(stem)}.pdf/.png")
    if len(good):
        log(f"D_rel on {DATASET}: median {good['dev_rel'].median():.3e}, "
            f"range {good['dev_rel'].min():.3e}-{good['dev_rel'].max():.3e}")


def finalize_geo():
    src = merge_geo()
    if not src:
        return None
    cells = cell_table_geo(src)
    if len(cells) < 1:
        log("!! no usable geodesic cell -- check the 'fail:'/'skip:' rows and "
            "the shard logs, then resume")
        return None
    try:
        return make_figure_geo(cells, src)
    except Exception as e:
        traceback.print_exc()
        log(f"!! the geodesic figure failed to draw ({type(e).__name__}: {e}); "
            f"the CSVs are intact, so rerun with --merge after fixing it")
        return None



# ================================================================== LAUNCHER
def _looks_like_me(path):
    """The one check that makes worker spawning safe.  sys.argv[0] inside a
    notebook is .../ipykernel_launcher.py -- a real, existing .py file.  The
    previous version accepted it, spawned two bare Jupyter kernels that sat
    waiting for a client that never came, and reported '0 checkpoints' every
    five minutes for as long as the session lasted.  A candidate is this
    script only if it contains this script's sentinel."""
    try:
        if not (path and os.path.isfile(path)):
            return False
        with open(path, "r", errors="ignore") as fh:
            return SENTINEL in fh.read(400000)
    except OSError:
        return False


def _ipython_cell_sources():
    try:
        ip = get_ipython()                                   # noqa: F821
    except Exception:
        return []
    hist = ip.user_ns.get("In") or []
    return [s for s in reversed(hist) if isinstance(s, str)]


def _materialise_cell_source():
    """Recover the pasted cell's own source so workers can be spawned from it."""
    for src in _ipython_cell_sources():
        if SENTINEL not in src:
            continue
        lines = [l for l in src.splitlines()
                 if not l.lstrip().startswith(("%", "!"))]
        dst = os.path.join(OUT_DIR, "_measure_scale_worker.py")
        try:
            with open(dst, "w") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError:
            return None
        return dst if _looks_like_me(dst) else None
    return None


def _script_path():
    for cand in (globals().get("__file__"),
                 sys.argv[0] if sys.argv else None,
                 os.path.join(os.getcwd(), "measure_scale.py"),
                 os.path.join(OUT_DIR, "measure_scale.py")):
        if _looks_like_me(cand):
            return os.path.abspath(cand)
    return _materialise_cell_source()


def n_workers():
    if NO_LAUNCH or _SHARD_ENV is not None or WORKERS == "1":
        return 1
    ngpu = torch.cuda.device_count()
    if WORKERS != "auto":
        # An explicit WORKERS is honoured as given.  On a GPU box it should not
        # exceed the GPU count; on a CPU box it is how the launcher itself gets
        # exercised without a GPU, which is the only way to test it before
        # committing a session to it.
        return max(1, int(WORKERS))
    return ngpu if ngpu >= 2 else 1


def launch_workers(ngpu):
    """One process per GPU, with a handshake.  If the children do not announce
    themselves, they are not workers, and the run falls back to doing the job
    itself rather than watching them."""
    path = _script_path()
    if not path:
        log("launcher: cannot recover this script's source (not a file, and "
            "no notebook cell carries the sentinel) -- running inline on one "
            "GPU. Set WORKERS=1 to silence this.")
        return False
    log(f"launcher: {ngpu} workers, {len(REGIMES)*len(ACTS)} cells, source={path}")
    # Every worker would otherwise fetch and decode the same archive at the
    # same moment -- two 35-minute downloads racing into one ./data.  Do it
    # once here, write the decoded cache, and forbid the children from
    # downloading at all, so a miss is a loud error rather than a silent
    # half-hour each.
    staged = False
    if _DS["tv"]:
        try:
            t0 = time.time()
            X, _ = ensure_raw()
            log(f"launcher: data ready in {time.time()-t0:.0f}s, "
                f"X{tuple(X.shape)} -- workers will read the cache")
            del X
            staged = True
        except Exception as e:
            log(f"launcher: could not stage the data ({type(e).__name__}: {e})")

    # A notebook captures sys.stdout at the PYTHON level, not at file
    # descriptor 1.  A subprocess that merely inherits the parent's fd writes
    # somewhere the notebook log never shows.  Pipe each worker and relay its
    # lines through the parent's own print(), which IS captured.
    ready = [threading.Event() for _ in range(ngpu)]

    def _relay(proc, idx):
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if "WORKER-READY" in line:
                    ready[idx].set()
                print(f"[w{idx}] {line}", flush=True)
        except Exception:
            pass
        finally:
            ready[idx].set()          # an exited worker unblocks the handshake

    if PHASES == ["geodesic"]:
        n_ck = len(input_index()["ckpt"]) + len(
            glob.glob(os.path.join(ckpt_dir(), "*.pt")))
        if n_ck == 0:
            log("launcher: !! no ckpt_scale_* checkpoints anywhere. "
                "PHASE=geodesic does not train -- stage a PHASE=barrier "
                "output as an input dataset first.")

    procs, relays = [], []
    for i in range(ngpu):
        env = dict(os.environ, SHARD_ID=str(i), NUM_SHARDS=str(ngpu),
                   NO_LAUNCH="1", PYTHONUNBUFFERED="1")
        if staged:
            env["NO_DOWNLOAD"] = "1"
        if torch.cuda.device_count() > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(i)
        pr = subprocess.Popen([sys.executable, path], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1)
        th = threading.Thread(target=_relay, args=(pr, i), daemon=True)
        th.start()
        procs.append(pr); relays.append(th)

    t0 = time.time()
    for i, ev in enumerate(ready):
        ev.wait(max(1.0, LAUNCH_TIMEOUT - (time.time() - t0)))
    alive = [i for i, pr in enumerate(procs) if pr.poll() is None]
    spoke = [i for i, ev in enumerate(ready) if ev.is_set()]
    if not alive or len(spoke) < ngpu:
        log(f"launcher: handshake failed after {time.time()-t0:.0f}s "
            f"(alive={alive}); killing the children and running inline.")
        for pr in procs:
            try: pr.kill()
            except Exception: pass
        return False
    log(f"launcher: {ngpu} workers up in {time.time()-t0:.0f}s")

    stop = threading.Event()

    def _beat():
        while not stop.wait(300):
            n_ck = len(glob.glob(os.path.join(ckpt_dir(), "*.pt")))
            stems = (["param_scale"] if PHASES == ["barrier"] else
                     ["param_geo_scale"] if PHASES == ["geodesic"] else
                     ["param_scale", "param_geo_scale"])
            n_row = sum(
                max(0, sum(1 for _ in open(f)) - 1)
                for st in stems
                for f in glob.glob(os.path.join(
                    OUT_DIR, f"{st}_{TAG}_shard*.csv")))
            live = sum(1 for pr in procs if pr.poll() is None)
            mins = (time.time() - _T_START)/60
            print(f"[beat] {mins:.0f} min | {live}/{ngpu} workers alive | "
                  f"{n_ck} checkpoints, {n_row} pairs", flush=True)
            if out_of_time(margin_h=0.25):
                print(f"[beat] TIME BUDGET {TIME_BUDGET_H} h reached -- "
                      f"stopping the workers so the notebook can commit. "
                      f"Stage this version's output as an input dataset and "
                      f"rerun to resume.", flush=True)
                for pr in procs:
                    try: pr.terminate()
                    except Exception: pass
                return

    hb = threading.Thread(target=_beat, daemon=True); hb.start()
    rc = [p.wait() for p in procs]
    stop.set()
    for th in relays:
        th.join(timeout=10)
    log(f"launcher: workers finished rc={rc}")
    if any(c not in (0, -15, 143) for c in rc):
        log(f"launcher: !! at least one worker exited non-zero -- read "
            f"log_scale_{TAG}_shard*.txt before trusting the figure")
    finalize()
    return True



_ROW_TIMES = []


def _note_row_time(sec): _ROW_TIMES.append(sec)


def _worst_row_h():
    """Leave room for one more row of the size this run actually produces.

    Measured, not guessed: a Christoffel solve at width 8 costs more than ten
    times one at width 1, so a fixed margin is either wasteful at the bottom of
    the grid or useless at the top."""
    # The margin must always leave room to do SOME work: a fixed 0.5 h reserve
    # against a 0.5 h budget means the very first check fires before anything
    # has been measured, and the run stops having done nothing at all.  Cap it
    # at a quarter of the budget, always.
    cap = max(0.02, 0.25*TIME_BUDGET_H)
    if not _ROW_TIMES:
        return min(0.5, cap)
    return min(3.0, cap, 1.4*max(_ROW_TIMES)/3600.0 + 0.05)



# ====================================================================== MAIN
def plan():
    cells = [(r, a) for r in REGIMES for a in ACTS]
    pair_list = list(itertools.combinations(range(NSEEDS), 2))[:PAIRS]
    seeds = sorted({s for pr in pair_list for s in pr})
    n_train = len(cells)*len(WIDTHS)*len(seeds)
    n_pair = len(cells)*len(WIDTHS)*len(pair_list)
    log(f"PLAN  dataset={DATASET} K={K} mode={MODE} device={DEVICE} "
        f"gpus={torch.cuda.device_count()}")
    log(f"  regimes={REGIMES} acts={ACTS} widths={WIDTHS}")
    log(f"  seeds={seeds} pairs={pair_list}")
    log(f"  {len(cells)} cells -> {len(cells)} regression points "
        f"(the published run had 36 across three architectures)")
    log(f"  {n_train} trainings of {EPOCHS} epochs, {n_pair} measured pairs")
    for w in WIDTHS:
        log(f"    width {w}: {n_params(w, ACTS[0], REGIMES[0]):,} parameters")
    ev = TGRID*EVAL_N
    log(f"  per pair: {ev:,} forward images for B (TGRID={TGRID} x "
        f"EVAL_N={EVAL_N}) + 3 x {FISHER_N} JVP images")
    log(f"  total measurement: {n_pair*ev/1e6:.0f}M forward images")
    log(f"  budget {TIME_BUDGET_H} h/session; the run resumes, so more than "
        f"one session is fine")
    if "geodesic" in PHASES:
        plan_geo(cells, pair_list)


def plan_geo(cells, pair_list):
    """Time a REAL fisher_vp at every width, then project the geodesic phase.

    A Christoffel solve is a CG whose every iteration is one fisher_vp, so the
    cost is (number of fisher_vp) x (cost of one), and only the second factor
    needs a GPU to know.  Guessing it has already cost this project two wasted
    sessions; measuring it takes about a minute."""
    geo_pairs = list(itertools.combinations(range(NSEEDS), 2))[:GEO_PAIRS]
    lam_rels = LAM_SWEEP if LAM_SWEEP else [LAM_REL]
    rows = len(cells)*len(WIDTHS)*len(geo_pairs)*len(lam_rels)
    log(f"  PHASE=geodesic: {rows} rows "
        f"({len(geo_pairs)} pairs x {len(lam_rels)} lam per cell-width)")
    log(f"    per row: {GEO_TGRID} Christoffel solves (each 4 fisher_vp for "
        f"dF, 1 grad_quad, up to {CG_ITERS} CG iterations) "
        f"+ {POWER_ITERS} for lam_max + 3 for the Fisher lengths")
    have = miss = 0
    for w in WIDTHS:
        for (r, a) in cells:
            for sd in range(NSEEDS):
                if find_ckpt(r, a, w, sd): have += 1
                else: miss += 1
    log(f"    checkpoints: {have} found, {miss} missing (of {have+miss})")
    if miss:
        log("    !! PHASE=geodesic does not train. Attach the output of EVERY "
            "PHASE=barrier session as an input dataset -- one session's "
            "output holds only the checkpoints that session wrote.")
    try:
        X, _Y = load_data_xy()
    except SystemExit:
        log("    (no data staged, so no timing)"); return
    Xgeo = X[:GEO_BATCH].to(DEVICE)
    total_h = 0.0
    for w in WIDTHS:
        net = build_net(w, ACTS[0], REGIMES[0]).eval()
        p, b = _pb(net)
        micro = pick_micro(w)
        v = {k: torch.randn_like(t) for k, t in p.items()}
        fisher_vp(net, p, b, Xgeo[:micro], v, micro)          # warm up
        if DEVICE == "cuda": torch.cuda.synchronize()
        t0 = time.time(); n_rep = 3
        for _ in range(n_rep):
            fisher_vp(net, p, b, Xgeo, v, micro)
        if DEVICE == "cuda": torch.cuda.synchronize()
        t_fvp = (time.time() - t0)/n_rep
        for label, cg in (("CG uses 1/3 the cap", CG_ITERS//3),
                          ("CG hits the cap", CG_ITERS)):
            per_row = (GEO_TGRID*(4 + 1 + cg) + POWER_ITERS + 3)*t_fvp
            n_w = len(cells)*len(geo_pairs)*len(lam_rels)
            if cg != CG_ITERS:
                total_h += n_w*per_row/3600.0
            log(f"    width {w:>2}: fisher_vp {t_fvp*1000:7.1f} ms "
                f"(micro={micro})  row {per_row/60:6.1f} min  if {label}")
        del net, p, b, v
        if DEVICE == "cuda": torch.cuda.empty_cache()
    ngpu = max(torch.cuda.device_count(), 1)
    log(f"    PROJECTED ~{total_h:.1f} GPU-hours (~{total_h/ngpu:.1f} h on "
        f"{ngpu} GPU) at the optimistic CG count")


def run_worker():
    open_log()
    print(f"[s{SHARD_ID}] WORKER-READY pid={os.getpid()} device={DEVICE} "
          f"cuda={os.environ.get('CUDA_VISIBLE_DEVICES','-')}", flush=True)
    log(f"shard {SHARD_ID}/{NUM_SHARDS} dataset={DATASET} K={K} mode={MODE} "
        f"out={OUT_DIR}")
    input_index()
    X, Y = load_data_xy()
    Xe, Ye = X[:EVAL_N], Y[:EVAL_N]
    Xf = X[:FISHER_N].to(DEVICE)
    done = read_done()
    cells = [(r, a) for r in REGIMES for a in ACTS]
    mine = [c for i, c in enumerate(cells) if i % NUM_SHARDS == SHARD_ID]
    pair_list = list(itertools.combinations(range(NSEEDS), 2))[:PAIRS]
    need = sorted({s for pr in pair_list for s in pr})
    log(f"cells={len(mine)}/{len(cells)} widths={WIDTHS} pairs={pair_list} "
        f"eval_n={EVAL_N} fisher_n={FISHER_N} tgrid={TGRID}")
    chance = 1.0/K

    n_ok = n_bad = 0
    stopped_early = False
    for (regime, act) in mine:
        if stopped_early: break
        for w in WIDTHS:
            if out_of_time():
                log(f"TIME BUDGET {TIME_BUDGET_H} h reached -- stopping before "
                    f"{regime}/{act}/w{w} so the notebook can commit its "
                    f"output. Stage it as an input dataset and rerun.")
                stopped_early = True
                break
            all_done = all((regime, act, w, i, j) in done
                           for (i, j) in pair_list)
            have_ck = all(find_ckpt(regime, act, w, s) for s in need)
            if all_done and have_ck:
                log(f"  {regime}/{act}/w{w}: every pair measured and every "
                    f"checkpoint present -- nothing to do")
                continue
            if all_done and not have_ck:
                log(f"  {regime}/{act}/w{w}: every pair is already measured, "
                    f"but checkpoints are missing. Training them anyway -- "
                    f"PHASE=geodesic needs them and cannot train. Set "
                    f"PHASE=geodesic if the barrier numbers are what you "
                    f"already have.")
            nets, accs = {}, {}
            for s in need:
                # One budget check per WIDTH used to cover all four seeds.  At
                # width 8 that is four 20-minute trainings behind a single
                # check, so the run could pass the check with minutes to spare
                # and then work for over an hour past it.  Check per training,
                # and leave the finished checkpoints for the next session.
                if out_of_time(margin_h=_worst_train_h()):
                    log(f"TIME BUDGET: stopping before {regime}/{act}/w{w}/s{s}"
                        f" -- {len(nets)}/{len(need)} seeds of this cell are "
                        f"trained and saved; the next session resumes them.")
                    stopped_early = True
                    nets = {}; break
                try:
                    t_one = time.time()
                    nets[s], accs[s], reused = get_or_train(regime, act, w, s,
                                                            X, Y, Xe, Ye)
                    if not reused:
                        _note_train_time(time.time() - t_one)
                except Exception:
                    traceback.print_exc()
                    log(f"!! training failed at {regime}/{act}/w{w}/s{s}")
                    nets = {}; break
            if stopped_early:
                break
            if len(nets) < len(need):
                log(f"skip {regime}/{act}/w{w}: training failed"); continue
            if min(accs.values()) < MIN_ACC_OVER_CHANCE*chance:
                log(f"skip {regime}/{act}/w{w}: acc {min(accs.values()):.4f} "
                    f"below {MIN_ACC_OVER_CHANCE}x chance ({chance:.4f}) -- "
                    f"these are not minima, so B is not a barrier")
                continue
            # How far down the loss curve did training actually get?
            ref0 = build_net(w, act, regime).eval()
            pr0, br0 = _pb(ref0)
            L_end = min(loss_acc_rho(ref0, {k: nets[s].state_dict()[k]
                                            for k in pr0}, br0, Xe, Ye)[0]
                        for s in need)
            drop = 1.0 - L_end/math.log(K)
            del ref0
            if drop < MIN_LOSS_DROP:
                log(f"skip {regime}/{act}/w{w}: endpoint loss {L_end:.3f} is "
                    f"only {100*drop:.0f}% below chance ({math.log(K):.3f}); "
                    f"the published cells were 87-99% below. Training stopped "
                    f"short of a minimum -- raise EPOCHS (currently {EPOCHS}) "
                    f"or lower MIN_LOSS_DROP={MIN_LOSS_DROP} to measure it "
                    f"anyway.")
                continue
            log(f"  {regime}/{act}/w{w}: endpoint loss {L_end:.3f} "
                f"({100*drop:.0f}% below chance ln({K})={math.log(K):.2f})")
            ref = build_net(w, act, regime).eval()
            p_ref, b_ref = _pb(ref)
            ag, gs = perm_spec(ref)
            npar = sum(v.numel() for v in p_ref.values())
            sds = {s: nets[s].state_dict() for s in need}
            for (i, j) in pair_list:
                if (regime, act, w, i, j) in done: continue
                t0 = time.time()
                try:
                    perms = weight_matching(ag, gs, sds[i], sds[j],
                                            MATCH_ITERS, seed=i*13 + j)
                    pA = {k: sds[i][k].to(DEVICE) for k in p_ref}
                    sdBp = apply_perm(sds[j], ag, perms)
                    pB = {k: sdBp[k].to(DEVICE) for k in p_ref}
                    delta = {k: pB[k] - pA[k] for k in pA}
                    dn = _vnorm(delta)
                    Bv, tstar, L0, L1 = barrier(ref, p_ref, b_ref, pA, pB, Xe, Ye)
                    LA, rA, _ = loss_acc_rho(ref, pA, b_ref, Xe, Ye)
                    LB, rB, _ = loss_acc_rho(ref, pB, b_ref, Xe, Ye)
                    gA = grad_norm(ref, pA, b_ref, Xf, Y[:FISHER_N])
                    gB = grad_norm(ref, pB, b_ref, Xf, Y[:FISHER_N])
                    flA, rqA = flen_rq(ref, pA, b_ref, Xf, delta)
                    flB, rqB = flen_rq(ref, pB, b_ref, Xf, delta)
                    pM = {k: 0.5*(pA[k] + pB[k]) for k in pA}
                    flM, rqM = flen_rq(ref, pM, b_ref, Xf, delta)
                    _, rM, _ = loss_acc_rho(ref, pM, b_ref, Xe, Ye)
                    append(dict(dataset=DATASET, mode=MODE, regime=regime,
                                act=act, width=w, nparam=npar, seedA=i, seedB=j,
                                accA=f"{accs[i]:.6f}", accB=f"{accs[j]:.6f}",
                                dnorm=f"{dn:.6e}", L_A=f"{LA:.6e}",
                                L_B=f"{LB:.6e}", B=f"{Bv:.6e}",
                                t_star=f"{tstar:.4f}",
                                L_chance=f"{math.log(K):.6e}",
                                gnorm_A=f"{gA:.6e}", gnorm_B=f"{gB:.6e}",
                                rho_A=f"{rA:.6e}",
                                rho_B=f"{rB:.6e}", rho_mid=f"{rM:.6e}",
                                flen_mid=f"{flM:.6e}", rq_mid=f"{rqM:.6e}",
                                flen_A=f"{flA:.6e}", flen_B=f"{flB:.6e}",
                                rq_A=f"{rqA:.6e}", rq_B=f"{rqB:.6e}",
                                status="ok"))
                    n_ok += 1
                    log(f"{regime}/{act}/w{w} {i}-{j}: B={Bv:.3e} "
                        f"L_A={LA:.3e} ({100*(1-LA/math.log(K)):.0f}% below "
                        f"chance) |gradL|={gA:.2e} rho_mid={rM:.3e} "
                        f"LF_mid={flM:.3e} RF_mid={rqM:.3e} "
                        f"({time.time()-t0:.0f}s)")
                except Exception as e:
                    n_bad += 1
                    traceback.print_exc()
                    append(dict({f: "" for f in FIELDS}, dataset=DATASET,
                                mode=MODE, regime=regime, act=act, width=w,
                                seedA=i, seedB=j,
                                status=f"fail:{type(e).__name__}"))
            del nets, sds, ref
            if DEVICE == "cuda": torch.cuda.empty_cache()

    n_ck = len(glob.glob(os.path.join(ckpt_dir(), "*.pt")))
    n_row = sum(max(0, sum(1 for _ in open(f)) - 1)
                for f in glob.glob(os.path.join(OUT_DIR,
                                                f"param_scale_{TAG}_shard*.csv")))
    log(f"{'STOPPED EARLY' if stopped_early else 'done'}: "
        f"{n_ck} checkpoints, {n_row} pairs on disk, "
        f"{n_ok} measured / {n_bad} failed this session "
        f"({(time.time()-_T_START)/3600:.1f} h)")
    # A shard whose every attempt raised is a broken shard, not a finished one.
    # Reporting rc=0 for that is what let a run of pure tracebacks look like a
    # success in the launcher's summary line.
    if n_bad and not n_ok:
        log(f"!! every one of {n_bad} attempted pairs failed -- see the "
            f"traceback above; exiting non-zero so the launcher says so")
        raise SystemExit(2)
    if n_bad:
        log(f"!! {n_bad} pairs failed and were written as 'fail:' rows")
    return n_ok, n_bad


def selfcheck():
    """Three properties the whole file rests on, on tiny nets."""
    global MODE
    ok = True
    set_seed(0)
    m = NetMLP(6, "tanh", "ntk", din=5, k=3).to(DEVICE).eval()
    p, b = _pb(m)
    x = torch.randn(9, 5, device=DEVICE)
    d = {k: torch.randn_like(v) for k, v in p.items()}
    # (1) Delta'F Delta by JVP == the dense Fisher quadratic form
    from torch.func import jacrev
    keys = list(p.keys())
    J = jacrev(lambda q: _call(m, q, b, x))(p)
    Jf = torch.cat([J[k].reshape(9, 3, -1) for k in keys], 2)
    pr = torch.softmax(_call(m, p, b, x), 1)
    S = torch.diag_embed(pr) - pr.unsqueeze(2)*pr.unsqueeze(1)
    Fm = torch.einsum('bki,bkl,blj->ij', Jf, S, Jf)/9
    df = torch.cat([d[k].reshape(-1) for k in keys])
    dense = float(df @ Fm @ df)
    jvp_v = fisher_quad(m, p, b, x, d, micro=4)
    rel = abs(jvp_v - dense)/max(abs(dense), 1e-30)
    log(f"selfcheck Fisher quad: jvp={jvp_v:.8e} dense={dense:.8e} rel={rel:.1e}"
        f"  {'PASS' if rel < 1e-5 else 'FAIL'}")
    ok &= rel < 1e-5
    # (2) weight matching is exact on a known permutation, for BOTH nets
    for name, net in (("mlp", NetMLP(8, "tanh", "sp", din=5, k=3)),
                      ("cnn", NetCNN(2, "tanh", "sp", in_ch=3, k=4))):
        keep, MODE = MODE, name
        try:
            ag, gs = perm_spec(net)
            rng = np.random.RandomState(3)
            perms = {g: torch.as_tensor(rng.permutation(n), dtype=torch.long)
                     for g, n in gs.items()}
            sdA = net.state_dict()
            sdB = apply_perm(sdA, ag, perms)
            back = weight_matching(ag, gs, sdA, sdB, iters=8, seed=1)
            rec = apply_perm(sdB, ag, back)
            err = max(float((rec[k] - sdA[k]).abs().max()) for k in sdA)
        finally:
            MODE = keep
        log(f"selfcheck weight matching ({name}): "
            f"max|recovered - original| = {err:.2e} "
            f"{'PASS' if err < 1e-6 else 'FAIL'}")
        ok &= err < 1e-6
    # (3) channel permutation is an exact symmetry of the CNN's function
    set_seed(1)
    net = NetCNN(2, "gelu", "sp", in_ch=3, k=4).eval()
    keep, MODE = MODE, "cnn"
    try:
        ag, gs = perm_spec(net)
        rng = np.random.RandomState(5)
        perms = {g: torch.as_tensor(rng.permutation(n), dtype=torch.long)
                 for g, n in gs.items()}
        xb = torch.randn(4, 3, 16, 16)
        net2 = NetCNN(2, "gelu", "sp", in_ch=3, k=4).eval()
        net2.load_state_dict(apply_perm(net.state_dict(), ag, perms))
        with torch.no_grad():
            fd = float((net(xb) - net2(xb)).abs().max())
    finally:
        MODE = keep
    log(f"selfcheck cnn permutation symmetry: max|f(x)-f'(x)| = {fd:.2e} "
        f"{'PASS' if fd < 1e-4 else 'FAIL'}")
    ok &= fd < 1e-4
    # (4) the whole per-pair path, on the REAL device.  Checks (1)-(3) run on
    # tensors this function made itself, and every one of them passes on a CPU
    # box even when the pipeline cannot survive a GPU: the permutations come
    # from scipy on the CPU while the weights sit on the accelerator, and
    # index_select refuses to mix them.  That mismatch is invisible unless the
    # trained state_dicts really are on DEVICE, so put them there and walk the
    # exact sequence run_worker() walks.
    set_seed(2)
    dev = pdev = {"?"}
    Bv = tstar = fl = rq = float("nan")
    try:
        w0 = 2 if MODE == "cnn" else 16
        nets = [build_net(w0, "tanh", "sp") for _ in range(2)]
        for n in nets: n.eval()
        xb = (torch.randn(24, IN_CH, HW, HW) if MODE == "cnn"
              else torch.randn(24, DIN))
        yb = torch.randint(0, K, (24,))
        ref = build_net(w0, "tanh", "sp").eval()
        p_ref, b_ref = _pb(ref)
        ag, gs = perm_spec(ref)
        sds = [n.state_dict() for n in nets]      # these live on DEVICE
        dev = {str(v.device) for v in sds[0].values()}
        perms = weight_matching(ag, gs, sds[0], sds[1], iters=2, seed=0)
        pdev = {str(v.device) for v in perms.values()}
        pA = {k: sds[0][k].to(DEVICE) for k in p_ref}
        pB = {k: v.to(DEVICE) for k, v in apply_perm(sds[1], ag, perms).items()
              if k in p_ref}
        delta = {k: pB[k] - pA[k] for k in pA}
        Bv, tstar, _, _ = barrier(ref, p_ref, b_ref, pA, pB, xb, yb)
        fl, rq = flen_rq(ref, pA, b_ref, xb[:8].to(DEVICE), delta)
        good = all(np.isfinite([Bv, tstar, fl, rq]))
    except Exception as e:
        traceback.print_exc()
        log(f"selfcheck end-to-end pair RAISED: {type(e).__name__}: {e}")
        good = False
    log(f"selfcheck end-to-end pair on {DEVICE}: weights on {dev}, "
        f"perms on {pdev}, B={Bv:.3e} flen={fl:.3e} rq={rq:.3e} "
        f"{'PASS' if good else 'FAIL'}")
    ok &= bool(good)
    # (5) PHASE=geodesic: fisher_vp agrees with the dense Fisher, CG really
    # solves, and the Green quadrature reproduces the analytic solution.  Also
    # the check that matters most in one file: fisher_quad (the barrier path,
    # a JVP alone) and fisher_vp (the geodesic path, jvp+vjp contracted back)
    # must agree, because the two phases are supposed to be measuring the same
    # Fisher on the same network.
    set_seed(0)
    m = NetMLP(6, "tanh", "ntk", din=5, k=3).to(DEVICE).eval()
    p, b = _pb(m)
    x = torch.randn(9, 5, device=DEVICE)
    v = {k: torch.randn_like(t) for k, t in p.items()}
    keys = list(p.keys())
    J = jacrev(lambda q: _call(m, q, b, x))(p)
    Jf = torch.cat([J[k].reshape(9, 3, -1) for k in keys], 2)
    pr = torch.softmax(_call(m, p, b, x), 1)
    S = torch.diag_embed(pr) - pr.unsqueeze(2)*pr.unsqueeze(1)
    Fm = torch.einsum('bki,bkl,blj->ij', Jf, S, Jf)/9
    vf = torch.cat([v[k].reshape(-1) for k in keys])
    got = fisher_vp(m, p, b, x, v, micro=4)
    gotf = torch.cat([got[k].reshape(-1) for k in keys])
    rel = float((gotf - Fm @ vf).norm()/max(float((Fm @ vf).norm()), 1e-30))
    log(f"selfcheck fisher_vp vs dense F: rel={rel:.1e} "
        f"{'PASS' if rel < 1e-5 else 'FAIL'}")
    ok &= rel < 1e-5
    q_vp = _vdot(v, got)
    q_jvp = fisher_quad(m, p, b, x, v, micro=4)
    rel = abs(q_vp - q_jvp)/max(abs(q_vp), 1e-30)
    log(f"selfcheck fisher_quad (barrier) vs fisher_vp (geodesic): "
        f"{q_jvp:.8e} vs {q_vp:.8e} rel={rel:.1e} "
        f"{'PASS' if rel < 1e-5 else 'FAIL'}")
    ok &= rel < 1e-5
    lam = 1e-2
    m64 = NetMLP(6, "tanh", "ntk", din=5, k=3).to(DEVICE).double().eval()
    p64, b64 = _pb(m64)
    rhs64 = {k: torch.randn_like(t) for k, t in p64.items()}
    _, _, true64, used64 = cg_solve(m64, p64, b64, x.double(), rhs64, lam, 4,
                                    iters=400, tol=1e-12)
    log(f"selfcheck CG (float64): true residual={true64:.1e} in {used64} iters "
        f"{'PASS' if true64 < 1e-10 else 'FAIL'}")
    ok &= true64 < 1e-10
    rhs = {k: torch.randn_like(t) for k, t in p.items()}
    sol, rec32, true32, used = cg_solve(m, p, b, x, rhs, lam, 4,
                                        iters=400, tol=1e-12)
    chk = gf_vp(m, p, b, x, sol, lam, 4)
    indep = _vnorm({k: chk[k] - rhs[k] for k in rhs})/max(_vnorm(rhs), 1e-30)
    honest = abs(true32 - indep) <= 1e-3*max(indep, 1e-30)
    log(f"selfcheck CG (float32): true residual={true32:.1e} in {used} iters, "
        f"CG's own recursion claims {rec32:.1e} "
        f"({true32/max(rec32,1e-30):.0e}x optimistic) "
        f"{'PASS' if (true32 < 1e-3 and honest) else 'FAIL'}")
    ok &= (true32 < 1e-3 and honest)
    ts = np.linspace(0, 1, 201)
    xi = green_matrix(list(ts)).numpy().sum(1)*(ts[1] - ts[0])
    exact = ts*(1 - ts)/2
    err = float(np.abs(xi - exact).max()/max(np.abs(exact).max(), 1e-30))
    log(f"selfcheck Green quadrature: max rel err={err:.1e} "
        f"{'PASS' if err < 2e-2 else 'FAIL'}")
    ok &= err < 2e-2
    log("selfcheck: " + ("ALL PASS" if ok else "FAILED"))
    return ok


def finalize_all():
    for ph in PHASES:
        (finalize if ph == "barrier" else finalize_geo)()


def smoke_assert():
    """A SMOKE run that prints happily but leaves no CSV and no figure has told
    you nothing.  Check the artefacts of every phase that ran -- and that the
    rows actually say 'ok', because a run where every pair raised still leaves
    a complete set of files behind, which passed this check once already."""
    if not SMOKE:
        return
    for ph in PHASES:
        stem = "param_scale" if ph == "barrier" else "param_geo_scale"
        cstem = "cell_scale" if ph == "barrier" else "cell_geo_scale"
        fstem = "figscale" if ph == "barrier" else "figgeo_scale"
        want = [f"{stem}_{TAG}.csv", f"{cstem}_{TAG}.csv",
                f"{fstem}_{TAG}.png", f"{fstem}_{TAG}.pdf"]
        miss = [f for f in want if not os.path.exists(os.path.join(OUT_DIR, f))]
        if miss:
            log(f"SMOKE[{ph}]: FAILED -- missing {miss}")
            raise SystemExit(1)
        ok = bad = 0
        with open(os.path.join(OUT_DIR, f"{stem}_{TAG}.csv")) as fh:
            for r in csv.DictReader(fh):
                if str(r.get("status", "")).startswith("ok"): ok += 1
                else: bad += 1
        ncell = max(0, sum(1 for _ in open(
            os.path.join(OUT_DIR, f"{cstem}_{TAG}.csv"))) - 1)
        log(f"SMOKE[{ph}]: {ok} measured rows, {bad} failed, {ncell} cells")
        need = 3 if ph == "barrier" else 2
        if bad or ok < need:
            log(f"SMOKE[{ph}]: FAILED -- do not start the real run")
            raise SystemExit(1)
    log("SMOKE: PASS -- pipeline connected, every row measured")


def main():
    if "--selfcheck" in sys.argv:
        sys.exit(0 if selfcheck() else 1)
    if "--merge" in sys.argv:
        finalize_all(); return
    if PLAN or "--plan" in sys.argv:
        plan(); return
    if SMOKE:
        log("SMOKE: tiny end-to-end run. It proves the plumbing, not the "
            "science. Clear SMOKE before the real run.")
        if not selfcheck():
            raise SystemExit("selfcheck failed -- do not launch the real run")
    ngpu = n_workers()
    if ngpu >= 2 and launch_workers(ngpu):
        smoke_assert(); return
    if ngpu >= 2:
        log("running inline on one GPU after the launcher declined")
    for ph in PHASES:
        if len(PHASES) > 1:
            log(f"===== PHASE {ph} =====")
        (run_worker if ph == "barrier" else run_worker_geo)()
    if _SHARD_ENV is not None:
        return                      # a worker merges nothing and draws nothing
    finalize_all()
    smoke_assert()


if __name__ == "__main__":
    main()
