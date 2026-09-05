# Data dictionary

Every column name below is taken from the actual CSV headers.

## Conventions

| Term | Meaning |
|---|---|
| **cell** | a triple `(regime, act, width)`. Per-cell statistics are **medians** over the pairs or seeds in that cell. |
| **pair** | two networks `seedA`, `seedB` **after permutation alignment** (weight matching), before interpolation. |
| `regime` | `ntk` · `sp` (standard) · `mup` (μP) |
| `act` | `relu` · `gelu` · `tanh` · `swish` · `softplus` |
| `width` | MLP/TS: hidden width (64…4096). **CNN: channel multiplier** (1, 2, 4, 8) → real channels `[16w, 32w, 64w]` |
| `t` | position along the interpolation path; `t=0` at net A, `t=1` at net B |
| `Δ` (`dnorm`) | `w_B − w_A` after alignment |
| `status` | `ok` if the row is valid. Anything else means the measurement failed — **kept deliberately**, never deleted |

Sign convention for exponents: **`α > 0` means the quantity decreases with width.**

---

## `data/train/{mlp,cnn,ts}_combined.csv`

Training, barrier and ∂F. Source for Figures 2, 3, 5 and A2. 17 columns (CNN has 18).

| Column | Present for | Meaning |
|---|---|---|
| `kind` | — | `net` = one trained network · `pair` = one aligned pair · `cell` = failed cell |
| `shard` | both | which shard (0–5) produced the row |
| `regime`, `act`, `width` | both | cell coordinates |
| `seed` | `net` | seed of this network |
| `seedA`, `seedB` | `pair` | the two seeds forming the pair |
| `acc` | `net` | test accuracy |
| `accA`, `accB` | `pair` | endpoint accuracies |
| `barrier` | `pair` | **loss barrier** on the linear path after permutation alignment |
| `dF_op` | `net` | ‖∂F‖ operator norm. **Empty for `relu` by design** |
| `wmove` | `net` | distance travelled in weight space during training |
| `epochs` | both | epochs actually run (30 for MLP, 100 for CNN/TS) |
| `status` | both | `ok` on every row of all three files |
| `_src` | both | originating shard file — provenance |
| `dF_note` | `net`, CNN only | annotation from the ∂F audit |

`cnn_combined.csv` is the **dF-corrected** file and the one every figure uses.
`cnn_combined_v1_superseded.csv` is the pre-audit version, kept only for comparison.

---

## `data/geodesic/{mode}_pairs.csv`

Geodesic deviation, per pair. 25 columns. Source for Figure 6.

| Column | Meaning |
|---|---|
| `dnorm` | ‖Δ‖ — Euclidean chord length between the aligned networks |
| `dev_geo` | `sup_t ‖ξ(t)‖` — absolute geodesic deviation |
| `dev_rel` | `sup_t ‖ξ(t)‖ / ‖Δ‖` — **normalised**, comparable across cells |
| `gamma_mid` | ‖Γ(Δ,Δ)‖ at `t = ½` — Christoffel magnitude at the midpoint |
| `flen_A`, `flen_mid`, `flen_B` | **Fisher length** `q(t)/2` at `t = 0, ½, 1` |
| `rq_A`, `rq_mid`, `rq_B` | **Rayleigh quotient** `q(t)/‖Δ‖²` at the same points — invariant to the scale of Δ |
| `lam`, `lam_rel` | damping: `λ = lam_rel · ‖F‖_op`, default `lam_rel = 1e-2` |
| `cg_resid` | conjugate-gradient residual — **numerical diagnostic**; large means suspect |
| `fd_instab` | finite-difference instability — numerical diagnostic |
| `accA`, `accB` | endpoint accuracies |

Relation between the two: `flen(t) = ½ · rq(t) · dnorm²`.

---

## `data/geodesic/{mode}_cells.csv`

Per-cell aggregate plus fitted exponents. 26 columns, 105 rows. Source for Figures 3 and 7.

| Group | Meaning |
|---|---|
| `n_seeds`, `n_pairs` | sample size for the cell — **always report this in the caption** |
| `acc_med` | median accuracy |
| `dF_med`, `dF_q1`, `dF_q3` | median and quartiles of ‖∂F‖ → the IQR band in the figures |
| `dev_rel_med/q1/q3` | same, for relative geodesic deviation |
| `dev_rel2_med` | variant normalised by ‖Δ‖² |
| `gamma_mid_med`, `flen_mid_med/q1/q3`, `rq_mid_med` | median geometric quantities at the midpoint |
| `fd_instab_med` | median numerical diagnostic |
| `alpha_dF` + `alpha_dF_r2` | width exponent of ‖∂F‖ with the **R²** of the fit |
| `alpha_devrel2` + `_r2` | width exponent of the ‖Δ‖²-normalised deviation |
| `alpha_flen` + `_r2` | width exponent of Fisher length |

> **Two deviation columns exist; only one is used.** The figures use `dev_rel`
> (normalised by ‖Δ‖). That choice is applied in
> `scripts/figp2_deviation.py` / `scripts/figF_exponents.py`, and matches the
> comment in the λ-sweep scripts.
> `dev_rel2_med` / `alpha_devrel2` (normalised by ‖Δ‖²) is used by **no figure** —
> retained for comparison only. Do not cite it by accident.

---

## `data/final/{mode}_pairs.csv`

Final measurement round, per pair. 32 columns. Source for Figure 4.

| Column | Meaning |
|---|---|
| `L_A`, `L_B`, `L_max` | endpoint losses and the maximum along the path |
| `B` | barrier |
| `t_star` | position of the loss maximum — **where the barrier peaks** |
| `rho_A`, `rho_mid`, `rho_max`, `rho_at_tstar` | **ρ\*** — the Fisher ≈ Hessian agreement ratio, at t = 0, ½, its maximum, and at `t_star` |
| `acc_mid` | midpoint accuracy — used by the collapse gate |
| `flen_A/mid/B/at_tstar` | Fisher length at those points |
| `rq_mid` | Rayleigh quotient at the midpoint |
| `R_end`, `R_mid`, `R_tstar` | **ratio R** = constant × `B / flen`, at three reference points |
| `devrel_lam1e-1`, `devrel_lam1e-2`, `devrel_lam1e-3` | relative deviation at **three damping levels** — evidence that conclusions do not depend on λ |
| `cg_resid`, `fd_instab` | numerical diagnostics |
| `wmoveA` | weight-space distance travelled by net A |

`data/final/{mode}_cells.csv` (21 columns) is the per-cell median of the above, plus `n_pairs`.

---

## `data/final/lambda_sweep_{ntk,sp,mup}_cells.csv`

5 columns: `act`, `width`, `devrel_1e-1`, `devrel_1e-2`, `devrel_1e-3`.
Answers: *would the conclusion change under a different λ?*

The matching `*_pairs.csv` files hold the same sweep before aggregation.

---

## `data/final/decompose_mlp.csv`

10 columns, 126 rows. Source for Figure A1; verifies `∂F = gauge + transport`.

| Column | Meaning |
|---|---|
| `dF_op` | total ‖∂F‖ |
| `gn_op` | **Gauss–Newton** component |
| `transport_op` | **transport** component |
| `rho_S` | residual-related ratio |
| `tr_frac` | transport share of the total — *where ∂F mostly comes from* |

---

## `data/train/cnn_remeasure_dF_audit.csv`

12 columns, 83 rows. **Not experimental data — an audit log.**

| Column | Meaning |
|---|---|
| `role` | suspect network, or healthy control |
| `dF_goc` | the original ∂F value under suspicion |
| `cfg` | which re-measurement configuration: original / alternate probe seed / halved or doubled `eps` / Richardson disabled |
| `probe_seed`, `eps`, `rich` | the parameters of that configuration |
| `dF` | the value obtained |
| `sec` | wall-clock time |

Verdicts are `ARTIFACT` (original not reproducible outside its own configuration →
replaced by the median of the alternatives) or `REPRODUCED` (genuine effect).

---

## Not yet produced — along-path profiles

`src/04_profile/` will write, once run (see [RUNBOOK.md](RUNBOOK.md)):

| File | Key columns |
|---|---|
| `profile_christoffel_{mode}.csv` + `_cell` | `xinorm_t*` — ‖Γ‖ along the `t` grid |
| `profile_length_{mode}.csv` + `_cell` | `rq_t*` on a 21-point grid, `dnorm` |
| `profile_shape_{mode}.csv` + `_cell` | `xirel(t)`, `shape(t)`, `t_peak`, `skew` |

Because they have not been run, some `flen_*`, `R_*` and `devrel_lam*` values in
`data/final/` are still empty, and the remaining appendix figures cannot yet be built.
