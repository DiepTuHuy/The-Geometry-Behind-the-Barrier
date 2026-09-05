# Barrier grid-density check

Answers one question: does the `TGRID_FINE = 41` grid used by `measure_final_*.py`
miss the maximum of `L(t)`?

**Method.** `regrid.py` rebuilds the exact pipeline of `src/03_final/mlp_measure_final.py`
(same `param_cfg`, same `NetMLP`, same `weight_matching` with `seed = i*13 + j`, same
`EVAL_N = 10000` first MNIST samples) and then evaluates `L(t)` on a 401-point grid that
**contains the original 41-point grid as a subset**, so the two are directly comparable.

**Pipeline validation.** On `ntk_gelu_w1024` the 41-point values reproduce
`data/final/mlp_pairs.csv` to six digits: 0.806255 / 0.858114 / 0.824332 / 0.759831.

**Result** (128 seed pairs, 32 cells, all three parameterisations; MLP checkpoints only):

| | |
|---|---|
| median gap | +0.047% |
| maximum gap | +0.383% |
| pairs with a gap > 0.5% | 0 / 128 |
| largest absolute gap | 9.4e-4 (loss units) |
| `t*` moves | 103 / 128 pairs |

`t*` moves but `B` barely changes, so `L(t)` is flat around its maximum. The 41-point
grid is sufficient; these numbers are the ones reported in Appendix E.1.

`summary.csv` — one row per seed pair. `regrid.py` — the script that reproduces it.
