# Figures

Generated output only. Every file here is produced by a script in [`../scripts/`](../scripts/)
from the CSVs in [`../data/`](../data/), by:

```bash
bash ../scripts/make_all.sh
```

Nothing in this directory is hand-drawn or hand-edited, and nothing is reconstructed:
each array plotted is read from a released CSV through `scripts/fig_data.py`. Re-running
the scripts reproduces every file below byte for byte.

## `main/` — Figures 1–5 of the paper

Each file is named for the figure number it carries in the paper.

| file | paper | what it shows |
|------|-------|---------------|
| `Figure1` | Figure 1 | the paradox: the barrier grows with width while the metric flattens |
| `Figure2` | Figure 2 | relative geodesic–linear deviation $D_{\mathrm{rel}}$ against width |
| `Figure3` | Figure 3 | the Rayleigh quotient $\mathcal{R}_F$ against width, by parameterisation |
| `Figure4` | Figure 4 | the dual controls on the Length axis: midpoint Fisher length and endpoint $\rho^*$ |
| `Figure5` | Figure 5 | the main result: Fisher length predicts α_B (R² = 0.90, n = 36 cells) |

## `appendix/` — Figures 6–12 of the paper

| file | paper | appendix | what it carries |
|------|-------|----------|-----------------|
| `Figure6` | Figure 6 | D.1 | Figure 1 repeated on the teacher–student family and the CNN |
| `Figure7` | Figure 7 | D.2 | the Rayleigh quotient on the other two architectures |
| `Figure8` | Figure 8 | D.3 | endpoint $\rho^*$ at the widest width, every cell of the grid |
| `Figure9` | Figure 9 | D.3 | endpoint $\rho^*$ against width, one panel per cell |
| `Figure10` | Figure 10 | D.3 | the two conditions of the summary claim in one plane |
| `Figure11` | Figure 11 | D.4 | the width measurements repeated on CIFAR-10 |
| `Figure12` | Figure 12 | D.4 | barrier against Fisher length and Rayleigh quotient on CIFAR-10 |

### Not in the paper

Kept under their working names, and deliberately not renumbered:
`figp2_deviation`, `figp6_regime_uncertainty`, `figD9_quadrants`,
`figD16_rayleigh_split`, `figD17_rayleigh_profile`, `figR_roadmap`.

`.pdf` is what LaTeX includes; `.png` is committed where a raster preview is useful.

## Rayleigh appendix (`src/10_rayleigh`)

| Script | Says |
|---|---|
| `figD16_rayleigh_split` | $\mathcal{R}_F$ decays because the whole Fisher spectrum shrinks, not because $\hat\Delta$ rotates down it |
| `figD17_rayleigh_profile` | the $t=\frac12$ anchor sits in a dip of $\mathcal{R}_F(t)$, and the dip deepens with width |

**They read different files, on purpose.**

`figD11` reads `data/rayleigh/mlp_*.csv` (`fig_data.load_rayleigh` /
`rayleigh_cells`), because the alignment ratio needs $\mathrm{tr}\,F/P$ and
**no earlier run measures a trace at all** — `src/10_rayleigh` is the only
source of it. That loader bypasses `_canon`, since `_canon` drops non-smooth
activations and **ReLU is legitimate here**: the $C^3$ requirement comes from
differentiating $F$, and the Rayleigh quotient differentiates nothing. Coverage
is **six cells**, MLP only — the cells whose checkpoints spanned at least three
widths, which is the threshold `_fit_alpha` needs. Read it as a complete result
on six cells, not a partial one on the full grid.

`figD12` reads `data/profile/mlp_length.csv` (`fig_data.load_profile_length`),
from `src/04_profile` — **84 cells on a 21-point $t$ grid**, against the newer
run's 42 cells on 9 points. The two are independent and agree to a maximum
relative deviation of `7e-05` over the 1800 rows they share, so the choice is
coverage, not trust. Do not "update" this figure to the newer file: it would
halve the cells and coarsen the grid.

## How the figures are kept honest

**Physical size.** `fig_style.save()` pads each figure out to exactly the width declared
in `figsize`, which must equal the width it is `\includegraphics`'d at, so the scale
factor in LaTeX is always 1.000 and a font declared at 8 pt prints at 8 pt. It also
widens the axes to fill that canvas, and warns (`[warn]` / `[thin]`) if a figure
overflows or under-fills. Do not reintroduce `bbox_inches="tight"`: it crops the
declared margins away, and each figure then gets rescaled by a different factor
(measured spread before this was fixed: 7.8 pt to 9.6 pt for a declared 8 pt).

**Colour discipline.** The three regime colours are reserved for parameterisations
(NTK-lazy / Standard / $\mu$P) and are never reused for anything else. Schematics use
`C["accent"]` (Okabe–Ito bluish green); the four terms of $R(w_0)$ in `figD_vacuous` use
`C["term_*"]`, chosen disjoint from the regime palette.

**Numbers printed, not just drawn.** Each script prints the quantities it plotted — for
example `Figure5.py` prints `n cells = 36 | Fisher length R2 = 0.904,
slope = 1.13 | Rayleigh R2 = 0.026` — so a number quoted in the text can always be
checked against the figure that carries it.

Modules: `fig_style.py` is the shared style (Okabe–Ito semantic colours, STIX serif at
true physical size, despined axes, log–log helpers); `fig_data.py` is the single place
that reads CSV; `fig_schematic.py` holds the box/arrow primitives used by the diagram
figures (`figE_pipeline`, `figR_roadmap`).
