# Figures

Generated output only. Every file here is produced by a script in [`../scripts/`](../scripts/)
from the CSVs in [`../data/`](../data/), by:

```bash
bash ../scripts/make_all.sh
```

Nothing in this directory is hand-drawn or hand-edited, and nothing is reconstructed:
each array plotted is read from a released CSV through `scripts/fig_data.py`. Re-running
the scripts reproduces every file below byte for byte.

## `main/` — the six main-text figures

| file | what it shows |
|------|---------------|
| `figp1_paradox` | the paradox: the barrier grows with width while the metric flattens |
| `figp2_deviation` | geodesic deviation ξ(t), by architecture and by parameterisation |
| `figp3_rayleigh` | the Rayleigh quotient — direction alone does not carry the effect |
| `figp4_regime_barrier` | barrier against width, split by parameterisation |
| `figp5_length_predicts` | the main result: Fisher length predicts α_B (R² = 0.90, n = 36 cells) |
| `figp6_regime_uncertainty` | spread within a parameterisation |

## `appendix/` — the nine appendix figures

| file | appendix location | what it carries |
|------|-------------------|-----------------|
| `figR_roadmap` | roadmap | dependency graph of every appendix result; solid edge = "used in the proof of", dashed = "bounds the regime in which it applies" |
| `figB_scaling` | B.2 | $\|\Delta\|_2=\Theta(\sqrt P)$: growth (a) and the $\sqrt P$-compensated two-sided corridor (b) |
| `figC_geodesic` | C.4–C.5 | deviation profile inside the tube, envelope = Green kernel (a); the regime the theorem does not reach (b) |
| `fig_counterexample_D` | D.1 | the constant-Fisher counterexample |
| `figD_vacuous` | D.2 | the four terms of $R(w_0)$ and the diverging admissible interval |
| `figE_pipeline` | E.1 | the matrix-free measurement pipeline; the barrier branch bypasses $F$ |
| `figE_gridcheck` | E | $B(41)$ against $B(401)$: the quadrature grid is converged, no pair reaches the 0.5% line |
| `figF_exponents` | F.1 | the fitted width exponents as a dot-and-whisker plot, all four families on one axis |
| `figF_within` | F.7 | does the Fisher-length relation survive splitting the 36 cells — and does the Rayleigh quotient |

`.pdf` is what LaTeX includes; `.png` is committed where a raster preview is useful.

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
example `figp5_length_predicts.py` prints `n cells = 36 | Fisher length R2 = 0.904,
slope = 1.13 | Rayleigh R2 = 0.026` — so a number quoted in the text can always be
checked against the figure that carries it.

Modules: `fig_style.py` is the shared style (Okabe–Ito semantic colours, STIX serif at
true physical size, despined axes, log–log helpers); `fig_data.py` is the single place
that reads CSV; `fig_schematic.py` holds the box/arrow primitives used by the diagram
figures (`figE_pipeline`, `figR_roadmap`).
