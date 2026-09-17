# Runbook — what to run, in what order

Three layers, strictly separated. Each layer writes files; the layer above only reads
them and never calls back down.

```
  Layer 1   train        GPU, days     ->  ckpt_*/  (*.pt, not in this repo)
  Layer 2   measure      GPU, hours    ->  data/**/*.csv
  Layer 3   plot         CPU, seconds  ->  figures/{main,appendix}/*.pdf
```

**To reproduce the figures, skip layers 1 and 2.** Every CSV they produce is already
committed.

```bash
pip install -r requirements-figures.txt
bash scripts/make_all.sh
```

---

## Layer 1 — Training

Six shards per architecture; each shard was run in its own session (one Kaggle account
per shard). Set `SHARD_ID` at the top of the file, then:

```bash
python3 src/01_train/mlp_train_shard0.py     # -> mlp_shard0.csv  + ckpt_pmlp_v2/
python3 src/01_train/cnn_train_shard0.py     # -> cnn_shard0.csv  + ckpt_pcnn_v2/
python3 src/01_train/ts_train_shard0.py      # -> ts_shard0.csv   + ckpt_pts_v2/
```

| Architecture | Data | Epochs | Cells | Status |
|---|---|---:|---:|---|
| mlp | MNIST | 30 | 105 | complete |
| cnn | FashionMNIST | 100 | 60 | complete |
| ts | Teacher–Student | 100 | 105 | complete |

The shard plan is identical in all three files:

| `SHARD_ID` | regime | activations |
|---|---|---|
| 0 / 1 | ntk | `relu, gelu, tanh` / `swish, softplus` |
| 2 / 3 | sp | `relu, gelu, tanh` / `swish, softplus` |
| 4 / 5 | mup | `relu, gelu, tanh` / `swish, softplus` |

Merging the six shard CSVs gives `data/train/{arch}_combined.csv`.

---

## Layer 2 — Measurement

Every script below reads the `.pt` checkpoints produced by layer 1.

### 2.1 Geodesic deviation (Sec. 5.1) → `data/geodesic/{mode}_pairs.csv`

```bash
python3 src/02_geodesic/mlp_measure_geodesic.py
python3 src/02_geodesic/cnn_measure_geodesic.py
python3 src/02_geodesic/ts_measure_geodesic.py
```

Produces `dev_rel`, `gamma_mid`, and `flen`/`rq` at t = 0, ½, 1.

### 2.2 Final measurement → `data/final/{mode}_pairs.csv`, `{mode}_cells.csv`

```bash
PHASE=all python3 src/03_final/mlp_measure_final.py
PHASE=all python3 src/03_final/cnn_measure_final.py
PHASE=all python3 src/03_final/ts_measure_final.py
```

> **`PHASE=all` matters.** The recorded run stopped at `PHASE=rho`, so the `flen_*`,
> `R_*` and `devrel_lam*` columns in `data/final/` are currently empty for some rows.
> Re-run with `PHASE=all` before building the remaining appendix figures.

### 2.3 Damping sweep → `data/final/lambda_sweep_*`

```bash
python3 src/05_lambda_sweep/lambda_sweep_ntk.py
python3 src/05_lambda_sweep/lambda_sweep_sp.py
python3 src/05_lambda_sweep/lambda_sweep_mup.py
```

> Use the three per-regime scripts. `lambda_sweep_v1_legacy.py` is an earlier, smaller
> run (2 activations, 2 pairs per cell, no wall-clock guard). It is kept only because
> some early CSVs trace back to it — do not pool its numbers with the others.

### 2.4 ∂F audit (CNN) → `data/train/cnn_remeasure_dF_audit.csv`

```bash
python3 src/06_analysis/remeasure_dF_cnn.py
```

Re-measures suspect ∂F values under five configurations — original, two alternative
probe seeds, `eps` halved and doubled, and Richardson extrapolation disabled — then
labels each as `ARTIFACT` (not reproducible outside the original configuration, replaced
by the median of the alternatives) or `REPRODUCED` (genuine). This produced the
canonical `data/train/cnn_combined.csv`.

---

## Layer 2b — Along-path profiles (not yet run)

`data/geodesic/` records `dev_rel` and `rq` at only **three** points, t ∈ {0, ½, 1}, so
the *shape* of the curve is invisible. These three scripts recover it.

### Where the output lands

`OUT_DIR` is `/kaggle/working` on Kaggle and the **current directory** elsewhere, so run
them from the directory you want the CSVs in:

```bash
mkdir -p data/profile && cd data/profile
python3 ../../src/04_profile/measure_profile_christoffel.py
```

### Cheapest order

```
1. measure_profile_christoffel.py    EXPENSIVE  — computes Gamma; shard with SHARD=0..5
2. measure_profile_shape.py          near-instant — reuses step 1's output
3. measure_profile_length.py         cheap, independent
```

ξ(t) is derived from Γ(t), which step 1 already wrote as `xinorm_t*`. If step 2 finds
that file it reads it directly and **never touches the GPU**. Running them in the other
order still gives correct results, it just recomputes Γ once.

### Sharding the expensive one

A full MLP pass is 3 regimes × 4 activations × 7 widths × 3 pairs = 252 pairs ≈ 2 268 CG
solves — too much for one session. Spread it over six:

| `SHARD` | regime | activations |
|---|---|---|
| 0 / 1 | ntk | `gelu, tanh` / `swish, softplus` |
| 2 / 3 | sp | `gelu, tanh` / `swish, softplus` |
| 4 / 5 | mup | `gelu, tanh` / `swish, softplus` |

All shards append to the same CSV; the resume key is
`(regime, act, width, seedA, seedB)`, so they never collide.

> Note this shard table differs from the training one: the profile scripts exclude
> `relu`, so even shards cover two activations rather than three.

### Set environment variables per command, never export them

```bash
PAIRS=3 python3 ../../src/04_profile/measure_profile_christoffel.py   # correct
export PAIRS=3                                                        # WRONG
```

`PAIRS` and `TGRID` deliberately differ between these three scripts (see
[CONFIGURATION.md](CONFIGURATION.md)). Exporting one silently drops
`measure_profile_length.py` from 10 pairs to 3, with no warning — the CSV still appears
and the numbers still look plausible.

`PAIRS=0` is worse: the resume logic sees `0 >= 0`, marks every cell finished, prints
`DONE` and exits 0 **having written no CSV at all**.

### Free cross-validation

On completion each script compares its own `rq(0)`, `rq(½)`, `rq(1)` against
`rq_A`, `rq_mid`, `rq_B` in `data/geodesic/{mode}_pairs.csv`. Same checkpoints, same
Fisher batch, same permutation alignment, so the three numbers must agree; a deviation
above 1% means the configuration differs and is reported. Disable with `ANCHOR=0`.

---

## Layer 3 — Figures

```bash
bash scripts/make_all.sh
```

Seconds, no GPU, no torch, no dataset. All fifteen figure scripts run from the
committed CSVs alone, and re-running them reproduces every PDF and PNG committed
under `figures/` byte for byte.

---

## Health check

```bash
python3 tools/check_config.py --strict
```

Parses all 30+ experiment scripts and exits non-zero if a hyperparameter drifts between
files that are supposed to match. Read-only.

---

## 07 — Uncertainty sweep (de-confounding rho* from the regime)

`src/07_uncertainty/mlp_uncertainty_sweep.py`

**Why.** The paper's summary claim has two clauses; the released grid tests only
one of them. `rho*` at the endpoint anchor is `[0.203, 0.615]` in every NTK-lazy
cell and `[0.000, 0.104]` in every feature-learning cell — the ranges are
disjoint — so "controlled predictive uncertainty" and "not NTK-lazy" are the
same variable, and inside the 24 feature-learning cells the residual `rho*`
explains none of the barrier exponent (`R^2 = 0.000`, `p = 0.99`). This run
moves `rho*` **within** a fixed parameterisation, using label smoothing, so the
two can be told apart.

**Kaggle.** Single file, no repo imports; writes to `/kaggle/working`.

```
!python mlp_uncertainty_sweep.py            # ~2 h on a T4 with the defaults
PAIRS=3 python mlp_uncertainty_sweep.py     # quick first pass
```

Set `SMOKE = True` at the top for a 30-second end-to-end check (validated on
CPU). The CSV is resumable: a session that is cut off continues where it
stopped, so the run can be split across sessions.

**Grid (defaults).** `regime ∈ {mup, sp}` × `act = gelu` ×
`width ∈ {64, 256, 1024, 4096}` × `eps ∈ {0, 0.05, 0.10, 0.20, 0.40}` × 5 seeds
= 200 trainings, 40 cells, 10 pairs each. Trim `WIDTHS` or `EPS_LIST` first if
the session is tight.

**Built-in cross-check.** `eps = 0` reproduces released cells: its `B` and
`rho_A` should match `param_final_mlp.csv` at the same `(regime, act, width)`.
Verify that column before trusting the rest.

**Known limitation, stated in the file.** Label smoothing is not a
`rho*`-only knob — it moves the minimum, so the Fisher length moves too (smoke
run, muP/w64: `eps` 0→0.2 sends `rho_A` 0.024→0.235 *and* `flen_A` 0.34→2.68).
The analysis must therefore regress the barrier exponent on **both**
`alpha_flen` and `rho*`; the 4×5 width×eps grid exists to give the two enough
independent variation for that regression to separate them.

**Output.** `param_uncert_mlp.csv` (per pair), `uncert_mlp_cell.csv` (per cell).

---

## 08 — The three unmeasured terms of Theorem 4.1

`src/08_remainder/measure_remainder.py`

**Why.** The bound is `B <= min{R(w_A),R(w_B)} + floor` with four terms in `R`,
and the released round evaluates only the Fisher term `(1/2) D^T F D`. The
appendix can therefore report what share of the barrier that term carries, but
not whether the bound is tight, because `||grad L(w_0)||`, `M_2` and `M_3` were
never computed. This run computes them and assembles the whole right-hand side.

**What it loads.** Nothing is retrained. Two kinds of file:

| what | path | used for |
|---|---|---|
| trained parameters | `ckpt_{RUN_TAG}/{regime}_{act}_w{w}_s{s}.pt` — `pmlp_v2` / `pcnn_v2` / `pts_v2` by `MODE` | the endpoints `w_A`, `w_B` themselves |
| released per-pair measurements | `data/final/{MODE}_pairs.csv`, `data/geodesic/{MODE}_pairs.csv` | `B`, `L_A`, `L_B`, `rho_A`, `dnorm`, `flen_A`, `flen_B` |

Checkpoints are found by recursive glob under `.`, `/kaggle/input`, `/content`,
`/content/drive/MyDrive`, so a Kaggle dataset containing `ckpt_pmlp_v2/` is
picked up wherever it is mounted. With the default `PAIRS=3` only seeds 0–3 are
touched, so `MODE=mlp` needs 4 seeds × 3 regimes × 4 activations × 7 widths =
**336 checkpoints** (192 for `cnn`, 336 for `ts`) — not the full 420.

Reusing the CSVs (`REUSE=1`, the default; set `REUSE=0` to recompute) is not
only faster. The bound has to be compared against the **same** `B` the paper
reports, or the comparison means nothing. Only `rho*(w_B)` is recomputed, since
the released round never stored it.

As an integrity check, `‖Δ‖` is recomputed from the checkpoints and compared
with the CSV value; a relative disagreement above `1e-3` aborts that pair with
`fail:RuntimeError` rather than producing a number from the wrong weights.

**What is exact and what is a lower bound.**

| term | how | status |
|---|---|---|
| `T1 = ‖D‖·‖grad L‖` | one backward pass | exact |
| `T2 = (1/2) D'FD` | matrix-free Fisher–vector product | exact |
| `M_2` | power iteration on `grad^2_w f_k(x;w)`, exact HVPs | exact per `(x,k)`; **max over a sample** of `x`, `k` and of `w` on the segment |
| `M_3` | `D^3 L[u,u,u]` by exact nested AD, direction optimised by the symmetric-3-tensor power iteration `u <- T[.,u,u]/‖·‖` | exact per `u`; **max over sampled `u`** |

`M_2` and `M_3` are therefore **lower** bounds on the suprema, so the assembled
`R` **understates** the true `R`. Read `bound/B >= 1` as evidence the inequality
holds with room; `bound/B < 1` is *not* evidence it fails, only that the sampled
directions do not reach far enough.

`M_3` is deliberately not a finite difference. Differencing `L` three times
divides by `h^3`, and since `u` is a unit vector while `‖D‖` is of order `10^2`,
the loss differences fall under float32 resolution well before the truncation
error is small — the estimate would be noise.

**Check it first (seconds, no checkpoints, no dataset):**

```bash
python3 src/08_remainder/measure_remainder.py --selfcheck
```

It rebuilds the Hessian of `f_k` densely on a tiny net and compares the power
iteration against `‖·‖_2`, and compares the nested-AD third derivative against a
finite difference of the *second* derivative (which is well conditioned). Both
must print `PASS`.

**Kaggle, two T4s in one session.** Paste the whole file into a cell and run
it. Nothing to upload, no `%%writefile`. The launcher notices two GPUs, writes
the running cell's own source to `_measure_remainder_worker.py` (IPython keeps
it in `In[-1]`), starts one worker per card, waits, merges and draws the figure.

```python
# ... paste the whole file into one cell and run it ...
```

`MODE` need not be set. With no `MODE` in the environment the file counts the
`ckpt_p{mlp,cnn,ts}_v2` directories it can reach and covers **every**
architecture whose weights are staged, one full pass each, merging and drawing
that architecture's figure before starting the next. It prints what it found
first. `MODE=cnn` restricts it to one; `MODE=all` is the explicit form of the
default.

Or, if the file is already on disk, the ordinary way still works:

```python
!MODE=cnn PAIRS=3 python measure_remainder.py
```

Children inherit stdout, so progress is live and every line carries its shard
tag (`[s0]`, `[s1]`). Lines beginning with `%` or `!` are dropped when the cell
is written out, so a `!pip install` at the top of the cell is harmless. Outside
IPython, or if the source cannot be recovered, the run says so and does the
whole grid single-process on one GPU — complete, just slower. Setting `SHARD_ID`
in the environment bypasses the launcher entirely; `--merge` and `--figure`
re-run those stages alone.

`MODE` is `mlp`, `ts` or `cnn`; run one mode at a time. Knobs, all environment
variables: `PAIRS` (default 3), `M2_BATCH`/`M2_CLASSES`/`M2_ITERS`,
`M3_BATCH`/`M3_RESTART`/`M3_ITERS`, `SEG_TS`, `EVAL_N`, `REUSE`, `DEVICE`.

**The figure.** `figD13_remainder_{MODE}.pdf` (and `.png`) is drawn at the end,
in the appendix's style — Okabe–Ito regime colours, STIX serif at true physical
size, despined axes. Panel (a) is `bound/B` against width per parameterisation,
log scale, with a rule at 1 where the bound would be exactly tight; panel (b)
is the share each of the four terms of `R` takes at the anchor that minimises
it, in the four-term palette `fig_style.py` reserves for this decomposition and
keeps disjoint from the regime triple. The style is inlined rather than
imported so the file still runs standalone on Kaggle.

**Resuming.** The CSV is appended one row per pair, and a rerun skips every
pair already in it, so an interrupted run continues where it stopped. What it
reads is deliberately wide: every `param_remainder_{MODE}*.csv` it can find in
`OUT_DIR` *and* recursively under `/kaggle/input` and the other checkpoint
roots, shard files and merged files alike.

- **Same session** — just run the cell again. Nothing to do.
- **New session** — `/kaggle/working` starts empty, so bring the CSVs back:
  download them at the end of a session, upload them as a small Kaggle dataset,
  attach it, and the next run finds them under `/kaggle/input` and picks up.
  They are a few hundred KB.
- **Failed pairs** count as done, so a deterministic failure is not retried
  forever. `RETRY_FAILED=1` reattempts them — what you want after changing a
  setting that caused an out-of-memory.
- `--merge` and `--figure` rebuild the merged CSV and the plot from whatever
  rows exist, without measuring anything.

Because a full pass over all three architectures is likely to outlast one
session, the cheap ordering is `MODE=cnn` first (48 cells, the smallest
networks), then `mlp`, then `ts` — each a session of its own, each resumable.

**Output.** Per architecture covered: `param_remainder_{MODE}_shard{i}.csv` per
worker, then `param_remainder_{MODE}.csv`, with `T1..T4` at both anchors, `R_A`,
`R_B`, `bound`, the released `B`, and `bound_over_B` — the number the question
is about — plus `figD13_remainder_{MODE}.pdf`/`.png`.

## 08b. Remainder on a larger task (`DATASET=`)

`src/08_remainder/measure_remainder.py` now takes a `DATASET` knob so the same
measurement can be pointed at a bigger problem. Unset, it is exactly the
released run (`mlp`→mnist, `cnn`→fashion, `ts`→synth) and nothing changes.

| DATASET | din | K | ch × hw | source |
|---|---|---|---|---|
| `mnist` / `fashion` | 784 | 10 | 1×28 | torchvision |
| `cifar10` / `cifar100` | 3072 | 10 / 100 | 3×32 | torchvision |
| `imagenet32` | 3072 | 1000 | 3×32 | ImageFolder under `/kaggle/input` |
| `imagenet64` | 12288 | 1000 | 3×64 | ImageFolder |
| `tinyimagenet` | 12288 | 200 | 3×64 | ImageFolder |

**The checkpoints must match the task.** `DATASET` changes the input and output
dimensions of the network, so the released weights (784-in, 10-out, trained on
MNIST) cannot be measured on any other task. Two guards enforce this rather
than letting it fail somewhere inside a JVP:

* `check_ckpt_task` compares the checkpoint's first-layer input dimension and
  output width against the task and raises naming both sides.
* `CHECK_MINIMA` (default on) refuses when the endpoint loss is at chance for
  `K`, because a barrier between two points that are not minima of the task
  being measured is not the object Theorem 4.1 bounds. `CHECK_MINIMA=0`
  overrides, for a deliberate control.

Reuse of the released per-pair CSVs is switched off automatically whenever
`DATASET` is not the default one — those files hold `B`, `L_A`, `‖Δ‖` and the
Fisher lengths for the *original* task. Output files are tagged
(`param_remainder_mlp_imagenet32_shard0.csv`) so a larger-problem run can never
overwrite or resume from the released one.

### Order of operations

1. Train on the new task with `src/01_train` (this is the unavoidable cost —
   there is no way to reuse MNIST minima on another dataset).
2. Stage those checkpoints as `ckpt_<tag>/` under `/kaggle/input`.
3. Stage the data as an ImageFolder tree, or set `DATA_DIR=/kaggle/input/<ds>/train`.
4. `DATASET=imagenet32 MODE=mlp TERMS=grad PAIRS=3` for a first cheap pass;
   drop `TERMS` for the full M₂/M₃ run.

### Cost notes

`M_2 = √K · sup‖∇²f_k‖` samples `M2_CLASSES` of `K` output coordinates. At
K=1000 the default of 3 samples is a far weaker lower bound than at K=10, and
the `√K` prefactor is 31.6 rather than 3.16 — raise `M2_CLASSES` or read `M_2`
as a much looser bound. The MLP's first layer is `din × n`, so full-resolution
ImageNet (150528-in) is not a width sweep this codebase can run: at n=4096 the
first layer alone is 616M parameters. `imagenet32` keeps ImageNet's 1000-way
task while leaving the width grid intact.

## 09. Does the prediction method survive a larger task?

`src/09_scale/measure_scale.py`. Self-contained: it **trains and measures** in
one pass, because the released checkpoints cannot be reused (see below). Paste
the whole file into one Kaggle cell and Save Version.

### What it tests

The paper carries two separable claims:

| claim | needs | file |
|---|---|---|
| Theorem 4.1 is *tight* | `M₂`, `M₃`, `‖∇L‖` | `08_remainder` — **parked** |
| Figure 5 *predicts* the barrier | `B`, `L_F`, `R_F`, `ρ*` | **this one** |

Only the second is a method someone could apply. It needs no `M₂`, no `M₃`, no
Christoffel symbol, no geodesic, no conjugate gradient and no damping `λ`:
`L_F = ½Δ'F(w₀)Δ` is one JVP per micro-batch. That is why it survives at large
`K` where the `M₂` sampling (3 of 1000 output coordinates) would not.

### Why it has to train

A trained weight vector is a minimum of its own task and nothing else. The
released MLP is `(n,784)` in and `(10,n)` out, so CIFAR/ImageNet do not even fit
the shapes; and on a *shape-compatible* different task — pixel-permuted MNIST —
the loss goes 0.26 → 3.65 (chance `ln 10 = 2.30`) and `‖∇L‖` grows 67×. The
endpoints stop being minima, so `B` stops being a barrier between minima.

### The verdict rule

Not "R² is high". The method transfers only if **the predictor works and the
control still fails**, as on MNIST (`R²` 0.90 vs 0.03). The figure states one of
three verdicts under the panels, from the two R² it just plotted:

| condition | verdict printed |
|---|---|
| Fisher ≥ 0.6 and Rayleigh ≤ 0.35 | *the relation transfers* |
| Fisher ≥ 0.6, Rayleigh > 0.35 | *inconclusive: both predict, so the test has lost its power at this scale* |
| Fisher < 0.6 | *the relation does not transfer at this scale* |

The middle row is the one to watch: if the control also predicts, the
experiment has lost its discriminating power rather than confirmed anything.

### Grid

Cells, not seeds, are the unit: the regression has one point per
(regime × activation) cell, so **activations generate the points**. Four
activations give 12 cells; cutting to one would leave three points and no
regression.

The run that was actually done, and the one the defaults reproduce:

| | |
|---|---|
| dataset | CIFAR-10 (`K = 10`, 3×32×32) |
| architecture | CNN only — **one**, not the released run's three |
| grid | 3 regimes × 4 smooth activations = **12 cells** |
| widths | `1, 2, 4, 8` (channel multiplier; 24 346 → 1 484 938 parameters) |
| seeds | 4, so C(4,2) = **6 pairs** per (cell, width) |
| epochs | **100** — the released CNN recipe, ≈19 600 steps |
| rows | 12 × 4 × 6 = **288**, all `ok` |

**CNN only, on purpose.** Refitting the released cells one architecture at a
time shows the control's failure is largely a *cross*-architecture effect:

| | Fisher R² | Rayleigh R² |
|---|---:|---:|
| released, 3 architectures pooled | 0.904 | 0.026 |
| released, MLP only | 0.973 | **0.630** — control does *not* fail |
| released, CNN only | 0.903 | 0.111 |

An MLP-only run is therefore a weak test: on the MLP the Rayleigh quotient
tracks the barrier too, *even on MNIST where the answer is known*. CNN is both
the right architecture for image data and the one whose control still
discriminates. There is no teacher–student arm and there cannot be: `ts` inputs
are `X ~ N(0, I)` with labels from a random teacher, so it has no dataset to
scale up.

**CIFAR-10 and not CIFAR-100.** The first attempt ran CIFAR-100 and produced
numbers that looked fine and meant nothing: endpoint loss 3.97 against a chance
loss of `ln 100 = 4.61`, so 14% below chance, with `ρ* = 0.99` — a softmax still
essentially uniform. The released cells sit at 87% below chance (NTK) to 99.7%
(SP/µP). Two causes, only one fixable by waiting:

* 30 epochs is 5 880 steps where the released CNN run took ≈23 400;
* at width 1 the CNN holds 24 346 parameters, which cannot reach a minimum on
  100-way CIFAR-100 at any epoch count — the small widths gate out and the
  power law is left with two points.

CIFAR-10 keeps `K = 10`, so `B` is the same functional as the published `B` and
the two clouds are directly comparable, while still being a real step up: the
first natural-colour task in the paper, 3072-dimensional input against 784, and
one that needs learned convolutional features rather than pixel templates.

### Running

    SMOKE=1 PHASE=barrier python measure_scale.py  # ~2 min, whole pipeline
    PLAN=1  PHASE=barrier python measure_scale.py  # grid + cost, no work
    PHASE=barrier python measure_scale.py          # this phase only
    python measure_scale.py --selfcheck            # 10 checks, seconds
    python measure_scale.py --merge                # rebuild CSVs + figures

(`PHASE` defaults to `all`; name `barrier` when you want only this phase.)

Do not skip `SMOKE=1`. It is the only thing standing between a typo and another
ten-hour empty log, and it fails loudly: a run where every pair raised still
leaves a complete set of output files behind, so the check also requires that
the rows actually say `ok`.

Both training and measurement resume from `/kaggle/input`, so a session that
runs out of time is continued by staging its own output as a dataset. Two GPUs
are sharded by cell automatically.

`EVAL_N` is 10000 from the head of the **train** split, exactly as
`src/03_final` used it — keeping it identical is what makes `B` comparable to
the published `B`. Normalisation is per channel, unlike the released scalar
constants, because NTK-lazy is sensitive to input scale.

Data is looked for locally before the network is considered: the decoded uint8
cache `_raw_{DATASET}_{HW}.pt`, then any copy already on the box in **any**
layout (the `cifar-*-python` pickles and the MNIST idx-ubyte files are read
directly, so an attached Kaggle dataset works whatever its folders are called),
and only then a download — from the launcher alone, never from a worker.
torchvision fetches CIFAR from `cs.toronto.edu` at about 80 kB/s inside a Kaggle
session, which is half an hour per worker otherwise.

### Is the endpoint a minimum?

`B` is meaningless unless it separates two minima, and accuracy above chance
does not establish that: a network stopped halfway down the loss curve is well
above chance and nowhere near a minimum. Every row therefore carries `L_A`,
`ln K` and `‖∇L‖` at both endpoints, and a cell whose training stopped short is
skipped and says so (`MIN_LOSS_DROP`, default 0.3 of the way below `ln K`). The
threshold is 0.3 and not 0.5 because NTK is lazy *by construction* — it sat at
87% below chance in the released run against SP/µP's 99.7% — and a stricter gate
would delete the whole regime, which is the high-`α_B` end of the range the
regression is fitted over.

### Result

    Fisher length   R² = 0.887   slope 0.74     (published: 0.904, slope 1.13)
    Rayleigh        R² = 0.281                  (published: 0.026)
    loss below chance  0.48 – 0.88     ‖∇L‖  0.12 – 0.47

The predictor holds and the control still fails, so the relation transfers.
Three cells have weak power-law fits (`r2_B` 0.21 ntk/swish, 0.31 ntk/gelu,
0.36 mup/tanh), so their `α_B` is poorly determined; the median `r2_B` is 0.88.
The Rayleigh R² is higher than the released CNN-only 0.111, as expected from a
single-architecture cloud.

## 09b. Does the linear-path proxy survive the same task?

`src/09_scale/measure_scale.py` with `PHASE=geodesic` — the same file, second
phase. Everything the barrier analysis
measures is read off the straight line `γ_lin(t) = (1-t)w_A + t w_B`;
Section 5.1 earns that by showing the Fisher geodesic stays close to it. This
file carries that justification to CIFAR-10, on the same checkpoints, so both
halves of the argument are tested at the same scale rather than one being
assumed.

    D_rel = sup_t ‖γ_g(t) − γ_lin(t)‖ / ‖Δ‖

to first order via the Green representation of Lemma 4.10 and the Christoffel
symbol of `G_F = F + λI`. The mathematics is `src/02_geodesic/measure_geodesic.py`
unchanged. The model, parameterisations, permutation spec and weight matching
are shared with the barrier phase — and that is why the two are one file rather
than two. A `state_dict` only loads back into the network that produced it; as
separate files, keeping two copies of the model in step was a standing
liability, and a silent drift would have measured the wrong pair of minima with
nothing in the output to show it. One definition cannot drift from itself.

**It does not train.** If the checkpoints are not staged it says so and stops.

### Running

    SMOKE=1 PHASE=geodesic python measure_scale.py   # ~4 min, whole pipeline
    PLAN=1  PHASE=geodesic python measure_scale.py   # TIMES a real fisher_vp
    PHASE=geodesic python measure_scale.py           # just this phase

`PHASE` defaults to **`all`**: the barrier phase first (which skips every
(cell, width) whose pairs are measured and whose checkpoints are present, so a
resumed run passes through it in seconds and trains only what is missing), then
the geodesic phase. Name a single phase only when you want just that one.

`PLAN=1` does not guess: it times an actual `fisher_vp` at each width on the
GPU it is given and projects from that. This is by far the dearest measurement
in the paper — one Christoffel symbol is a CG solve whose every iteration is a
full Fisher-vector product — so the budget is worth knowing before a session is
committed. For scale, on this laptop's CPU one row at width 8 takes 209 minutes
and the whole run is about nine days.

### Stage EVERY session's output

A Kaggle version's output holds only what *that* version wrote. The checkpoints
an earlier session trained live in the earlier version's output, so attaching
only the last one leaves most of the grid missing — the merged CSV survives,
because `merge()` reads staged inputs, but the checkpoints do not follow it.
A coverage preflight runs in the first seconds, before the data is even loaded,
and prints exactly which cells cannot be done:

    checkpoints: 29 found, 163 missing (of 192)
    !! only 2 of 12 cells have >=3 usable widths

`PLAN=1` must report `192 found, 0 missing` before the real run is worth
starting.

### Read the validity columns before the numbers

| column | meaning |
|---|---|
| `cg_iters == GEO_CG_ITERS` | CG hit its cap instead of converging |
| `cg_resid_true` | the residual of `G_F x − rhs`, computed |
| `cg_resid` | CG's own recursive residual — see below |
| `fd_instab` | disagreement between the `ε` and `ε/2` finite differences |

**`cg_resid` is not what it looks like.** CG never forms `G_F x − rhs`; it
carries the residual forward by `r ← r − a A p`, which loses the very
cancellation it is measuring. On a test problem here:

    float32   35 iters   recursion says 5.9e-13   truth is 1.2e-06
    float64   23 iters   recursion says 6.3e-13   truth is 6.3e-13

— six orders of magnitude optimistic at single precision, which is what the
released `param_geo_*.csv` reports. One extra matrix-vector product at the end
of each solve, about 1% of its cost, buys a residual that means what a reader
assumes, so both are recorded: `cg_resid` keeps the released definition and
`cg_resid_true` is the one to trust. It does not invalidate the released
numbers — 1e-6 is ample for a quantity compared on a log scale across widths —
but the column should say what it means.

Absolute magnitudes also depend on the damping `λ` (Remark 5.1), so compare
exponents in width and relative sizes, never a bare `D_rel` against a different
`λ`. `relu` is excluded by default: Γ is computed with a finite difference of
`F`, which needs `C³`, and relu has a kink. Rows for a non-smooth activation are
still written, flagged `smooth=0` / `status="ok:nonsmooth"`.

### Cross-check

`--selfcheck` contracts `fisher_vp` (the geodesic path: jvp+vjp) back against Δ
and compares it with `fisher_quad` (the barrier path: a JVP alone) on the same
network. They are the same quantity by two independent routes, so a
disagreement means the two phases are not measuring the same Fisher. It runs in
seconds and is part of the ten checks `--selfcheck` performs.

## 10. What does the Rayleigh quotient actually measure?

`src/10_rayleigh/measure_rayleigh.py`. Four phases; only `measure` needs a GPU,
and `torch` is imported lazily so the rest run on a laptop.

    python3 measure_rayleigh.py audit      # what exists, what is missing
    python3 measure_rayleigh.py selftest   # numerics vs a dense Fisher, seconds
    python3 measure_rayleigh.py measure    # ONLY the gaps                 (GPU)
    python3 measure_rayleigh.py merge      # old + new -> canonical tables
    python3 measure_rayleigh.py figures    # canonical -> PNG/PDF
    python3 measure_rayleigh.py all        # all of the above in order

**On Kaggle, pass the command in the environment, not on argv:**

```python
import os
os.environ.update(RQ_CMD="measure", RQ_ARCH="cnn,ts",
                  RQ_DATA="/kaggle/input/<dataset>/data")
```

A notebook is run by papermill, which puts its own parameter file on the command
line, so `sys.argv[1]` arrives as `/tmp/tmpXXXX.json`. `RQ_CMD` takes precedence,
and an unrecognised argv entry is now ignored instead of fatal.

One run covers every architecture named: `RQ_ARCH="cnn,ts"` does both in a
single pass — the task list is grouped by architecture inside `measure`, and a
worker handles whatever mix it was dealt. The parent pre-downloads each dataset
before spawning (`prefetch_data`), because workers are split by cost rather than
by architecture, so two of them can both be handed CNN cells and would otherwise
race each other writing the same `./data` archive.

**Both T4s are used by default** — one process per visible GPU, split *by cell*
and balanced *by cost* (`split_tasks`, proxy `width²`, least-loaded first).
Round-robin — what `02_geodesic` does — is 1.08× imbalanced on the MLP's 7-width
grid but **4.0×** on the CNN's 4-width grid, because an even grid never flips the
phase and every heavy width lands on one worker. Cost-based assignment measures
1.000× on the CNN+TS gap. Nothing crosses between processes, so the output is
bit-for-bit what one GPU gives. `RQ_GPUS=0` forces a single device.

### What it tests

`R_F = Δ̂'FΔ̂` is Figure 5's negative control, and §5 reads `R_F → 0` as the
displacement concentrating in F's **low-eigenvalue subspace**. A falling `R_F`
has two causes it cannot separate: Δ̂ rotating down a fixed spectrum
(*directional*), or the whole spectrum shrinking under a Δ̂ that has not moved
(*global*).

The separator is an identity: for `v` uniform on the unit sphere,
`E[v'Fv] = tr F/P` **exactly**. So the alignment ratio `A = R_F/(tr F/P)` is 1
when Δ̂ is spectrally indistinguishable from a random direction, and since
`R_F = (tr F/P)·A` the exponents are additive:

    α_{R_F} = α_{tr F/P} + α_A

`α_A > 0` is directional; `α_A ≈ 0` is global. The script prints the split and
**does not decide it** — if `α_A ≈ 0`, the honest report is that `R_F` tracks
`tr F` and §5's sentence has to be weakened.

### Audit first — it is the cheapest optimisation in the file

Most of what a Rayleigh study needs was already measured and is in `data/`.
Re-measuring it reproduces numbers to seven decimals: two independent runs of
`R_F` on the same checkpoints agree to **7e-05**, which is floating-point
summation order (a different micro-batch) and nothing else.

| quantity | already at | coverage |
|---|---|---|
| `R_F` at t=0,½,1 | `data/geodesic/{arch}_pairs.csv` | 216 cells, 3 arch, 4 smooth acts |
| `λ_max(F)` | same file, as `lam / lam_rel` | 216 cells |
| barrier `B` | `data/train/{arch}_combined.csv` | 270 cells, **incl. ReLU** |
| `R_F(t)`, 21 points | `data/profile/{arch}_length.csv` | 84 cells, **MLP only** |
| **`tr F/P`** | **nowhere** — no trace estimator exists in `src/01..08` | **0 cells** |
| `R_F` for ReLU | nowhere (ReLU rode along with the Christoffel run) | 0 cells |
| `R_F(t)` for CNN/TS | nowhere | 0 cells |

`data/final/*_pairs.csv` *has* `rq_*` and `flen_*` columns and they are empty in
all 840 released rows — `data/geodesic` is the only source of `R_F`.

So `measure` computes `tr F/P` everywhere, `R_F` for ReLU, and the `R_F(t)`
profile for CNN and TS. `audit` prints the remaining cost: **~202k `fisher_vp`**
at the current state, against roughly 4× that for a from-scratch run.

ReLU is admissible here although the Christoffel run had to exclude it: `R_F`
takes no derivative of F — one JVP and one VJP per micro-batch, no finite
difference, no CG, no damping.

### Output — the contract for redrawing figures

`merge` writes the canonical tables into `RQ_OUT`; the committed copies live in
`data/rayleigh/`. Every figure reads these and nothing else, and each row carries
`src_*` columns naming where the number came from.

| file | one row per |
|---|---|
| `rayleigh_pairs.csv` | (arch, regime, act, width, seedA, seedB) — `rq`, `flen`, `lmax`, `trP`+sem, `align`, `spec`, `ov1`, `top1` at A/mid/B |
| `rayleigh_profile.csv` | (pair, `t`) — long format, `src` says which run |
| `rayleigh_cells.csv` | cell — medians, `dip_depth`, `barrier`, and every fitted `alpha_*` |
| `rayleigh_summary.json` | machine-readable headline numbers |
| `fig_rayleigh_*.png/pdf` | reading figures, self-contained (no `fig_style`) |

**Precedence on merge:** where a quantity exists in both, the **older** run wins
— not because it is more accurate (they agree to 7e-05) but because it covers
far more: 216 cells against 44 for `R_F`, and 21 `t`-points against 9. A 9-point
grid understates the dip depth by up to 13%.

**`width` is raw**, as the source CSVs store it — for the CNN that is the channel
multiplier in {1,2,4,8}; `width_paper` is 64× that. Exponent sign: `α > 0` means
the quantity decreases with width.

**`*_med` vs `*_rat`.** A cell aggregate of a ratio is the ratio of the
aggregates: the median does not commute with division, and using
`median(R_F/trP)` leaves a residual up to 0.032 on `α_{R_F}`. `align_mid_med` is
the median of per-pair ratios (quote it for a typical pair); `align_mid_rat` is
what `α_A` is fitted on, because only that makes the identity exact.

### Validation

`selftest` checks `fisher_vp` against a dense Jacobian-built F, `power_top`
against `eigvalsh`, the sphere-probe mean against `tr F/P`, and `R_F` against
`d'Fd` — float64, seconds. Measured: `4.6e-16`, `1.0e-15`, `0.51σ`, `1.7e-16`.

The merged table reproduces the paper's headline regression from a different
barrier source: on the 36 smooth cells, **Fisher length R² 0.906 slope 1.120,
Rayleigh R² 0.023**, against the published 0.904 / 1.13 / 0.026.

