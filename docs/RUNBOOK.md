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
