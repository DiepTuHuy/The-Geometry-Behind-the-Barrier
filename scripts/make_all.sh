#!/usr/bin/env bash
# make_all.sh -- ONE command that redraws EVERY figure in the paper.
#
#   bash scripts/make_all.sh
#
# Every script below READS CSV ONLY. None of them trains or re-measures
# anything, so running this any number of times is safe and takes seconds.
# Output goes to figures/main/ and figures/appendix/ as .pdf (for LaTeX) and,
# where a raster is useful, .png.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Main text (figures/main) ==="
python3 Figure1.py    # Figure 1: the paradox -- barrier grows, curvature flattens
python3 Figure2.py    # Figure 2: geodesic deviation xi(t) across width and regime
python3 Figure3.py    # Figure 3: Rayleigh quotient -- direction alone is not enough
python3 Figure4.py    # Figure 4: barrier by parameterisation
python3 Figure5.py    # Figure 5: Fisher length is the predictor that works

echo
echo "=== Appendix D (figures/appendix) ==="
python3 Figure6.py    # Figure 6: Figure 1 on the other two architectures
python3 Figure7.py    # Figure 7: Rayleigh quotient on the other two architectures
python3 Figure8.py    # Figure 8: endpoint rho* across all cells
python3 Figure9.py    # Figure 9: endpoint rho* against width, every cell
python3 Figure10.py   # Figure 10: the two conditions in one plane
python3 Figure11.py   # Figure 11: the width measurements repeated on CIFAR-10
python3 Figure12.py   # Figure 12: barrier vs Fisher length / Rayleigh on CIFAR-10

echo
echo "=== Not in the paper (kept for reference) ==="
python3 figp2_deviation.py            # superseded by Figure2.py
python3 figp6_regime_uncertainty.py   # spread within a regime
python3 figD9_quadrants.py            # the two conditions as two axes
python3 figD16_rayleigh_split.py      # R_F decays with the spectrum, not by rotating
python3 figD17_rayleigh_profile.py    # the midpoint anchor sits in a dip
python3 figR_roadmap.py               # roadmap: dependency graph

echo
echo "Done. Figures are in figures/main/ and figures/appendix/"
