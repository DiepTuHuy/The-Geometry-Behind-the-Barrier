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
bash figures/make_all.sh
```

Runs in seconds. Output appears in `figures/out/` — `.pdf` for the paper, `.png` for
slides. Each script also prints a ready-to-paste LaTeX `\caption{...}`, so the numbers
in a caption can never drift from the numbers in its figure.

*Verified: all ten plotting scripts complete with `import torch` blocked (10/10), and
two consecutive runs produce byte-identical output.*

Re-running the **measurements** needs a GPU and days of compute — see
[docs/RUNBOOK.md](docs/RUNBOOK.md).

---

## Main result

`figures/fig2_exponent_scatter.py` regresses the barrier exponent `α_B` on four
candidate predictors across **n = 36 cells** (3 architectures × 3 regimes × 4 smooth
activations):

| Predictor | Quantity | R² |
|---|---|---:|
| `α_∂F` | metric flattening | 0.20 |
| `α_dev_rel` | geodesic deviation | 0.11 |
| `α_Δ̂ᵀFΔ̂` | Rayleigh quotient (direction only) | 0.02 |
| **`α_ΔᵀFΔ`** | **Fisher length** | **0.91** |

Only **Fisher length** predicts how the barrier scales with width. Reproduce with
`python3 figures/fig2_exponent_scatter.py` — it prints exactly these numbers.

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

figures/                   one script per figure; READS CSV ONLY, never measures
  style.py                 shared rcParams, palette, markers
  common.py                the single CSV map (PATHS) + shared plotting helpers
  fig*.py                  fig<N> ↔ out/fig<N>*.pdf ↔ \includegraphics{fig<N>*}
  make_all.sh              one command, every figure
  out/                     generated .pdf and .png

docs/                      runbook, configuration snapshot, data dictionary
tools/                     read-only helpers (no GPU, no torch)
```

**The layering is strict.** Training writes checkpoints; measurement reads checkpoints
and writes CSV; plotting reads CSV and writes PDF. Nothing in `figures/` ever trains or
re-measures, which is why the figures reproduce in seconds on any machine.

---

## Figure → script → data

| Figure | Script | Reads |
|---|---|---|
| 1 | `figures/fig0_geometry_schematic.py` | *(none — loss field, metric and geodesic are computed)* |
| 2 | `figures/fig1_barrier_dF.py` | `data/train/{mlp,cnn,ts}_combined.csv` |
| 3 | `figures/fig2_exponent_scatter.py` | `data/train/*_combined.csv` + `data/geodesic/*_cells.csv` |
| 4 | `figures/fig3_rho_star.py` | `data/final/{mode}_pairs.csv` |
| 5 | `figures/fig4_rho_gate.py` | `data/train/*` + `data/geodesic/*_pairs.csv` + `data/final/*_cells.csv` |
| 6 | `figures/fig5_path_profile.py` | `data/geodesic/{mode}_pairs.csv` |
| 7 | `figures/fig6_deviation.py` | `data/geodesic/{mode}_cells.csv` |
| — | `figures/fig7_three_spaces.py` | *(none — schematic for Eq. (4))* |
| A1 | `figures/figA1_decompose.py` | `data/final/decompose_mlp.csv` |
| A2 | `figures/figA2_by_activation.py` | `data/train/*_combined.csv` |

No plotting script hardcodes a path. The complete CSV map is the `PATHS` dictionary at
the top of [`figures/common.py`](figures/common.py) — move the data, edit one block.

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
