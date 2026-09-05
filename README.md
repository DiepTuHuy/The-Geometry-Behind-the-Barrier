# The Geometry Behind the Barrier

Code and data for the ICLR submission on the **Fisher geometry of linear mode
connectivity** across three parameterizations — **NTK**, **Standard (SP)** and **μP**.

We measure the loss **barrier**, the metric-flattening rate **‖∂F‖**, the **geodesic
deviation** ξ(t), the **Fisher length** and the quadratic-approximation ratio **ρ\***
over a full grid:

> 3 architectures (MLP/MNIST · CNN/FashionMNIST · Teacher–Student)
> × 3 regimes × 5 activations × 4–7 widths × 5 seeds
> = **270 cells, 3 150 measured rows, all complete.**

---

## Reproduce every figure — two commands, no GPU

Every CSV the figures need is committed here (~1 MB total). No training, no dataset
download, no `torch`.

```bash
pip install -r requirements-figures.txt
bash scripts/make_all.sh
```

Runs in seconds. Output appears in `figures/main/` (the six main-text figures) and
`figures/appendix/` (the nine appendix figures) — `.pdf` for the paper, `.png` where a
raster is useful. Each script also prints the measured numbers it plotted, so a number
quoted in the text can be checked against the figure that carries it.

*Verified: all fifteen figure scripts run from the committed CSVs alone, with no GPU,
no `torch` and no dataset download, and reproduce every PDF and PNG committed under
`figures/` byte for byte.*

Re-running the **measurements** needs a GPU and days of compute — see
[docs/RUNBOOK.md](docs/RUNBOOK.md).

---

## Main result

`scripts/figp5_length_predicts.py` regresses the barrier exponent `α_B` on two
candidate predictors across **n = 36 cells** (3 architectures × 3 regimes × 4 smooth
activations):

| Predictor | Quantity | R² |
|---|---|---:|
| `α_Δ̂ᵀFΔ̂` | Rayleigh quotient (direction only) | 0.03 |
| **`α_ΔᵀFΔ`** | **Fisher length** | **0.90** |

Only **Fisher length** predicts how the barrier scales with width — direction alone
does not. Reproduce with `python3 scripts/figp5_length_predicts.py`; it prints
`n cells = 36 | Fisher length R2 = 0.904, slope = 1.13 | Rayleigh R2 = 0.026`.

The companion claim — the barrier grows while the metric flattens — is
`scripts/figp1_paradox.py`, which prints the measured `α_B` and `α_∂F` per regime.

---

## Repository layout

```
src/                       every experiment, numbered in execution order
  01_train/                18 files = 3 architectures × 6 shards
  02_geodesic/             geodesic deviation, Sec. 5.1
  03_final/                rho*, R, Fisher length
  04_profile/              along-path profiles (Christoffel / length / shape)
  05_lambda_sweep/         damping-robustness sweep
  06_analysis/             dF decomposition, dF re-measurement audit

data/                      all measured results, committed
  train/                   per-shard and combined training CSVs
  geodesic/                measurement round 1
  final/                   measurement round 2 + damping sweep
  regrid/                  barrier re-measured on a 401-point grid (App. E.1 check)

scripts/                   one script per figure; READS CSV ONLY, never measures
  fig_style.py             shared rcParams, palette, markers, physical figure size
  fig_data.py              the single CSV map (PATHS) + the conventions applied once
  fig_schematic.py         drawing primitives shared by the schematic figures
  figp<N>_*.py             main text   -> figures/main/figp<N>_*.pdf
  fig{B,C,D,E,F,R}_*.py    appendix    -> figures/appendix/*.pdf
  make_all.sh              one command, every figure

figures/                   generated .pdf and .png -- exactly what the paper includes
  main/                    the six main-text figures
  appendix/                the nine appendix figures

docs/                      runbook, configuration snapshot, data dictionary
tools/                     read-only helpers (no GPU, no torch)
```

**The layering is strict.** Training writes checkpoints; measurement reads checkpoints
and writes CSV; plotting reads CSV and writes PDF. Nothing in `scripts/` ever trains or
re-measures, which is why the figures reproduce in seconds on any machine.

---

## Figure → script → data

Main text (`figures/main/`):

| Figure | Script | Reads |
|---|---|---|
| 1 | `scripts/figp1_paradox.py` | `data/final/*_pairs.csv` + `data/train/*_combined.csv` |
| 2 | `scripts/figp2_deviation.py` | `data/geodesic/*_pairs.csv` |
| 3 | `scripts/figp3_rayleigh.py` | `data/geodesic/*_pairs.csv` |
| 4 | `scripts/figp4_regime_barrier.py` | `data/final/*_pairs.csv` |
| 5 | `scripts/figp5_length_predicts.py` | `data/final/*_pairs.csv` + `data/geodesic/*_pairs.csv` |
| 6 | `scripts/figp6_regime_uncertainty.py` | `data/final/*_pairs.csv` |

Appendix (`figures/appendix/`):

| Figure | Script | Reads |
|---|---|---|
| roadmap | `scripts/figR_roadmap.py` | *(none — schematic)* |
| B.2 | `scripts/figB_scaling.py` | `data/final/*_pairs.csv` |
| C.4–C.5 | `scripts/figC_geodesic.py` | `data/final/*_pairs.csv` + `data/train/*_combined.csv` |
| D.1 | `scripts/fig_counterexample_D.py` | *(none — closed-form counterexample)* |
| D.2 | `scripts/figD_vacuous.py` | `data/final/*_pairs.csv` + `data/geodesic/*_pairs.csv` |
| E.1 | `scripts/figE_pipeline.py` | *(none — schematic)* |
| E | `scripts/figE_gridcheck.py` | `data/regrid/summary.csv` |
| F.1 | `scripts/figF_exponents.py` | `data/final/*` + `data/geodesic/*` + `data/train/*` |
| F.7 | `scripts/figF_within.py` | via `figp5_length_predicts.measured_cells()` |

No plotting script hardcodes a path. The complete CSV map is the `PATHS` dictionary at
the top of [`scripts/fig_data.py`](scripts/fig_data.py) — move the data, edit one block.
`data/regrid/` is the only exception: it is read directly by `figE_gridcheck.py`, which
is the one figure that is a convergence check on the quadrature grid rather than a plot
of the main measurement.

---

## Data completeness

| File | Cells | Rows |
|---|---|---|
| `data/train/mlp_combined.csv` | **105 / 105** | 525 `net` + 1050 `pair` |
| `data/train/cnn_combined.csv` | **60 / 60** | 300 `net` + 600 `pair` |
| `data/train/ts_combined.csv` | **105 / 105** | 525 `net` + 1050 `pair` |

Every row has `status = ok`; no `barrier` value is missing.

`∂F` is empty for `relu` **by design** — ∂F requires a smooth activation and ReLU has a
kink, so ReLU is excluded from `DF_ACTS`. This is not missing data.

The CNN run has **12 additional ∂F measurements withdrawn** after an audit
(`src/06_analysis/remeasure_dF_cnn.py`); 10 of the 12 fall in the NTK regime and 6 in
`softplus`. `data/train/cnn_remeasure_dF_audit.csv` is the full audit log. The corrected
file is the canonical `cnn_combined.csv`; the pre-audit version is kept as
`cnn_combined_v1_superseded.csv` for comparison.

**Checkpoints are not included** (~12 GB) and are not needed for any figure.

---

## Documentation

| Document | Answers |
|---|---|
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | What to run, in what order, and what it costs |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Every hyperparameter of every script, in one table |
| [docs/DATA.md](docs/DATA.md) | What each CSV column means |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | What the experiment files are and how they differ |

## Tools

Read-only. No GPU, no torch, no data required.

```bash
python3 tools/check_config.py --strict   # fails if hyperparameters drift between scripts
```

---

## Reproducibility notes

Things we did that reviewers may want to check:

* Every measurement script ships a `self_test()` that compares against a dense
  reference implementation, and an `ANCHOR` check that cross-validates a new
  measurement round against the previous one at shared grid points (it reports any
  deviation above 1%).
* Failed and suspect measurements are **kept and labelled**, not dropped: see the
  `status`, `fd_instab`, `cg_resid` and `dF_note` columns documented in
  [docs/DATA.md](docs/DATA.md).
* Conclusions are checked against the damping parameter λ over three decades
  (`data/final/lambda_sweep_*`), so they do not depend on one choice of λ.
* Figures are generated by one command, never edited by hand, and are deterministic.

Two caveats we state rather than hide:

1. Warmup is `min(8, epochs // 5)`, which is **6 epochs for MLP** but **8 for CNN and
   Teacher–Student**, because the epoch budgets differ (30 vs 100). It is not identical
   across architectures.
2. `data/geodesic/*_cells.csv` also carries an `alpha_devrel2` column (normalised by
   ‖Δ‖² rather than ‖Δ‖). **No figure uses it**; Figure 3 uses `dev_rel`. It is retained
   only for comparison.

## License

MIT — see [LICENSE](LICENSE).
