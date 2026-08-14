# Configuration snapshot

Hyperparameters are hardcoded in each script, one copy per file. This document is the
single place to look them up.

**Every value below is extracted from the source with `ast`, not typed by hand.**
Regenerate at any time:

```bash
python3 tools/check_config.py            # full tables
python3 tools/check_config.py --md       # markdown
python3 tools/check_config.py --strict   # exits 1 on unexplained drift
```

> **Read hyperparameters from the `CONFIG` block, not from the docstrings.** The
> docstrings of the training scripts have drifted from the code — see
> [EXPERIMENTS.md](EXPERIMENTS.md#docstring-drift).

---

## 1. Training — `src/01_train/`

### Identical across all three architectures

This is what makes the three architectures comparable:

```python
REGIMES  = ["ntk", "sp", "mup"]
ACTS     = ["relu", "gelu", "tanh", "swish", "softplus"]
NSEEDS   = 5          LR          = 0.1      BATCH   = 256
WARMUP_EPOCHS = 8     CLIP_NORM   = 1.0      N_EVAL  = 5000
T_GRID   = 21         MATCH_ITERS = 8        SEED_BASE = 4321
NUM_SHARDS = 6        RESUME      = True     SMOKE   = False
BASE     = 64                                          # muP anchor width

DF_ENABLE = True   DF_ACTS = ["gelu","tanh","swish","softplus"]   # no relu
DF_BATCH  = 2048   DF_MICRO = 64   DF_ITERS = 20   DF_NZ = 5
DF_EPS    = 3e-3   DF_RICHARDSON = True
```

### The four lines that differ

| | `mlp` | `cnn` | `ts` |
|---|---|---|---|
| `RUN_TAG` | `pmlp_v2` | `pcnn_v2` | `pts_v2` |
| dataset | MNIST | FashionMNIST | synthetic teacher–student |
| `DIN, K` | `784, 10` | `784, 10` *(vestigial — the model is convolutional)* | `64, 10` |
| `WIDTHS` | `[64…4096]` (7) | `[1, 2, 4, 8]` — **channel multiplier**, real channels `[16w, 32w, 64w]` | `[64…4096]` (7) |
| `EPOCHS` | **30** | **100** | **100** |

> Effective warmup is `min(8, epochs // 5)`: **6 epochs for MLP**, **8 for CNN and TS**.
> State this in the paper — warmup is not identical across architectures.

The CNN uses `GroupNorm(1, c)` (LayerNorm-style, one group) precisely because it is
**invariant to channel permutation**, so it does not interfere with weight matching.

### Shard plan (identical in all three)

```python
SHARD_PLAN = {
    0: ("ntk", ["relu","gelu","tanh"]),  1: ("ntk", ["swish","softplus"]),
    2: ("sp",  ["relu","gelu","tanh"]),  3: ("sp",  ["swish","softplus"]),
    4: ("mup", ["relu","gelu","tanh"]),  5: ("mup", ["swish","softplus"]),
}
```

The 18 training files are **three programs × six copies**, differing in exactly one
line: `SHARD_ID`.

---

## 2. Geodesic measurement — `src/02_geodesic/`

```python
MODE   = "mlp" | "cnn" | "ts"        # the only difference between mlp and ts
NSEEDS = 5      FD_EPS   = 3e-3      FD_RICH = True
LAM_REL = 0.01  CG_ITERS = 300       CG_TOL  = 1e-6
POWER_ITERS = 20    RESUME = True    BASE = 64
```

`cnn_measure_geodesic.py` differs in three further places, all GPU-memory handling and
no science: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; `grad_quad()` accumulates
gradients per micro-batch so memory is bounded by `micro` rather than the whole batch;
and an explicit cleanup block.

These three files were originally single-cell Jupyter notebooks saved with a `.py`
extension; they are published here as flat Python.

---

## 3. Final measurement — `src/03_final/`

```python
MODE    = "mlp" | "cnn" | "ts"
ACTS    = ["gelu","tanh","swish","softplus"]     # relu excluded
PAIRS   = env:PAIRS=10     NSEEDS = 5     FISHER_N = 2048   MICRO = 64
FD_EPS  = 3e-3   CG_ITERS = 300   CG_TOL = 1e-6   POWER_ITERS = 20
RESUME  = True   BASE = 64
WIDTHS  = {"mlp": [64…4096], "cnn": [1,2,4,8], "ts": [64…4096]}
```

All three files are identical apart from `MODE`.

---

## 4. Damping sweep — `src/05_lambda_sweep/`

| | `lambda_sweep_ntk` | `_sp` | `_mup` | `_v1_legacy` |
|---|---|---|---|---|
| `REGIMES` | `["ntk"]` | `["sp"]` | `["mup"]` | `["ntk"]` |
| `ACTS` | 4 | 4 | 4 | **2** |
| `PAIRS` | 3 | 3 | 3 | **2** |
| `BUDGET_H` | 11.0 | 11.0 | 11.0 | *(absent)* |

Shared: `MODE="mlp"`, `RUN_TAG="pmlp_v2"`, `TGRID=9`, `NSEEDS=5`, `FISHER_N=2048`,
`CG_ITERS=300`, `WIDTHS=[64…4096]`.

> `lambda_sweep_v1_legacy.py` is an earlier, smaller run. Do not pool its numbers with
> the three per-regime scripts.

---

## 5. Along-path profiles — `src/04_profile/`

```python
REGIMES = ["ntk","sp","mup"]
ACTS    = ["gelu","tanh","swish","softplus"]     # relu excluded
NSEEDS  = 5   FISHER_N = 2048   MICRO = 64   BASE = 64
SELFTEST = env:SELFTEST=1   ANCHOR = env:ANCHOR=1   BUDGET_H = env:BUDGET_H=11.0
```

Deliberate differences, with the reason from each file's own docstring:

| | `christoffel` | `length` | `shape` |
|---|---|---|---|
| `TGRID` | **9** | **21** | **9** |
| `PAIRS` | **3** | **10** | **3** |
| `LAM_REL`, `CG_ITERS`, `FD_EPS` | used | not used | used |

`christoffel` and `shape` are expensive — each pair costs `TGRID` CG solves — so they
sample 3 pairs on the 9-point grid inherited from the geodesic round (Green's-function
quadrature). `length` is cheap — one Fisher-vector product per grid point, no CG — so it
keeps all 10 pairs and uses 21 points to resolve the *shape* of the curve.

> **Set these per command; never `export` them.** They are environment variables, so an
> exported `PAIRS=3` silently drops `measure_profile_length.py` from 10 pairs to 3 with
> no warning. `PAIRS=0` is worse: the script prints `DONE`, exits 0 and writes nothing.
> Values that are merely malformed fail fast and loudly (`TGRID=abc`, `MODE=cnnn`,
> `SHARD=9`).

---

## 6. Two different tables named `SHARD_PLAN`

| | `src/01_train/` | `src/04_profile/` |
|---|---|---|
| `ACTS` | 5 (**includes** `relu`) | 4 (**excludes** `relu`) |
| even shards | 3 activations | 2 activations |
| odd shards | `swish, softplus` | `swish, softplus` |

Same name, same keys `0..5`, different contents. Do not confuse them when comparing
results.

---

## 7. Paths and environment

```python
DATA_ROOT  = "/kaggle/input/datasets/ANONYMIZED/DATASET"
CKPT_ROOTS = [DATA_ROOT, ".", "/kaggle/input", "/content", "/content/drive/MyDrive"]
OUT_DIR    = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
```

`DATA_ROOT` is anonymized for double-blind review. It is only the first of five
checkpoint roots and the others include a recursive `/kaggle/input` glob, so checkpoint
discovery is unaffected.

`OUT_DIR = "."` off Kaggle means CSVs land in the **current directory** — `cd` into the
target directory first (see [RUNBOOK.md](RUNBOOK.md)).

---

## 8. Automated check

```
$ python3 tools/check_config.py --strict
TOTAL: no unexplained differences within any group.
```

Every difference found has been traced and is declared, with its reason, in
`KNOWN_INTENTIONAL` inside [`tools/check_config.py`](../tools/check_config.py). Any
*new* drift makes `--strict` exit non-zero.
