# Experiment inventory

What each experiment file is, and how the files differ from one another. Every number
below was measured from the source or the CSVs, not recalled.

---

## The 18 training files are three programs

`src/01_train/` holds 18 files, but they are **three programs × six copies**. Within an
architecture the six shards differ in **exactly one line**:

```python
SHARD_ID = 0        # 0..5, one per session
```

Evidence: within an architecture every shard is byte-identical in size (MLP: 19 501 B
for all six), because `SHARD_ID = 0` and `SHARD_ID = 1` are the same length.

Shard 0/2/4 cover three activations, shard 1/3/5 cover two, which is why their CSVs have
315 and 210 rows respectively for the MLP.

---

## How the three programs differ

`mlp_train_shard0.py` vs `cnn_train_shard0.py` differ in 80 lines, nearly all of it the
model definition.

| | `mlp` | `cnn` | `ts` |
|---|---|---|---|
| `RUN_TAG` | `pmlp_v2` | `pcnn_v2` | `pts_v2` |
| data | MNIST | FashionMNIST | synthetic teacher–student |
| `DIN, K` | `784, 10` | `784, 10` *(vestigial — model is convolutional)* | `64, 10` |
| model | 2-hidden-layer MLP, `ScaledLinear` | 3 × `ScaledConv` + `GroupNorm(1, c)` + `MaxPool` | reuses the MLP |
| `WIDTHS` | `[64…4096]` | `[1, 2, 4, 8]` channel multiplier | `[64…4096]` |
| `EPOCHS` | **30** | **100** | **100** |
| cells | 105 | 60 | 105 |

`GroupNorm(1, c)` is chosen deliberately: with one group it is invariant to channel
permutation, so normalisation does not interfere with weight matching. The CNN class is
still named `MLP` in the source, with a comment acknowledging it.

Everything else — regimes, activations, seeds, learning rate, batch size, alignment
iterations, and all eight `DF_*` parameters — is identical, which is what makes the
three architectures comparable. See [CONFIGURATION.md](CONFIGURATION.md).

---

## Data completeness

| File | `net` | `pair` | Cells | `status` |
|---|---:|---:|---|---|
| `data/train/mlp_combined.csv` | 525 | 1050 | **105 / 105** | `ok` ×1575 |
| `data/train/cnn_combined.csv` | 300 | 600 | **60 / 60** | `ok` ×900 |
| `data/train/ts_combined.csv` | 525 | 1050 | **105 / 105** | `ok` ×1575 |

Arithmetic check: MLP `net` = 3 regimes × 5 activations × 7 widths × 5 seeds = 525;
`pair` = 3 × 5 × 7 × C(5,2) = 1050. No `barrier` value is missing anywhere.

Per-shard CSVs are present for MLP and CNN. For TS only the combined file survives.

---

## Where `∂F` is empty, and why

| Architecture | `net` rows without `dF_op` | Explanation |
|---|---:|---|
| mlp | 105 / 525 | **100% `relu`** — by design |
| ts | 105 / 525 | **100% `relu`** — by design |
| cnn | 72 / 300 | 60 `relu` (by design) **+ 12 genuinely withdrawn** |

`relu` is excluded from `DF_ACTS` because ∂F requires a smooth activation and ReLU has a
kink. This is not missing data.

The **12 withdrawn CNN measurements** are not random:

| regime | act | width | seed |
|---|---|---|---|
| ntk | gelu | 8 | 1 |
| ntk | tanh | 1 | 4 |
| ntk | tanh | 8 | 4 |
| ntk | swish | 4 | 1 |
| ntk | swish | 8 | 1 |
| ntk | softplus | 1 | 1 |
| ntk | softplus | 1 | 2 |
| ntk | softplus | 2 | 3 |
| ntk | softplus | 4 | 4 |
| ntk | softplus | 8 | 1 |
| sp | softplus | 2 | 3 |
| mup | gelu | 8 | 1 |

**10 of 12 fall in `ntk`, 6 of 12 in `softplus`.** That is a pattern, not noise, and is
worth stating in the appendix rather than reporting only the count. The audit that
produced this list is `src/06_analysis/remeasure_dF_cnn.py`; its log is
`data/train/cnn_remeasure_dF_audit.csv`.

---

## Provenance of `src/02_geodesic/`

These three files were originally saved as `run-{mlp,cnn,ts}.py` but their **contents
were Jupyter notebook JSON** — `nbformat 4.4`, a single code cell on a single line.
Consequences in the original form: `wc -l` reported 0 lines for 574 lines of code,
`git diff` was useless, and Jupyter could not open them because of the extension.

They are published here as flat Python, extracted verbatim from the single code cell
(byte-for-byte, verified by round-trip and `py_compile`). The extraction is lossless:
the cell contained no IPython magics and no stored output.

Comparing the flattened files shows `mlp` and `ts` differ in **exactly one line**
(`MODE = "mlp"` vs `"ts"`), while `cnn` differs in three places, all GPU-memory
handling.

---

## Docstring drift

All three training programs share one original docstring block with only the first line
edited. The remaining lines have not kept up with the code:

| Docstring claims | Reality |
|---|---|
| `python ts_mlp_shard.py` — in **both** mlp and cnn | no such file exists |
| `Output ts_mlp_shard{K}.csv` — in **both** | actually `param_{arch}_v2_shard{K}.csv` |
| "two MLPs trained on real MNIST (no teacher, no over-realization)" — in **both cnn and ts** | cnn uses FashionMNIST; ts *is* teacher–student with over-realization, contradicting its own first line |
| "no BN, no REPAIR" — in **cnn** | cnn does normalise, with `GroupNorm(1, c)` |
| "sweep width up to 4096" — in **cnn** | cnn sweeps `[1, 2, 4, 8]` channel multipliers |
| ts: "merged version: runs all 6 shards sequentially" | `main()` calls `run_one_shard(SHARD_ID)` **once**; the inline comment just above it is correct |

None of this affects the measurements — the `CONFIG` block is authoritative and is what
`tools/check_config.py` reads. **When writing the paper, take facts from `CONFIG`, not
from the docstrings.**

For the record: `run_one_shard(sid, …)` does declare `global SHARD_ID`, so the function
*is* capable of running all six shards in a loop. The docstring describes an earlier
version; the code is not wrong, it simply does less than the docstring claims.

---

## Legacy files kept on purpose

| File | Why it is still here |
|---|---|
| `src/05_lambda_sweep/lambda_sweep_v1_legacy.py` | earlier, smaller sweep (2 activations, 2 pairs/cell, no wall-clock guard). Some early CSVs trace to it. **Do not pool its numbers** with the three per-regime scripts. |
| `data/train/cnn_combined_v1_superseded.csv` | pre-audit CNN data, kept so the effect of the ∂F audit can be inspected. Every figure uses `cnn_combined.csv`. |
| `data/geodesic/{mode}_cells_v1_superseded.csv` | earlier aggregation of the geodesic round. |
