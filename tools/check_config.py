#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scan every experiment script and report configuration drift.

    python3 tools/check_config.py            # human-readable tables
    python3 tools/check_config.py --md       # markdown (for docs/)
    python3 tools/check_config.py --only train,profile
    python3 tools/check_config.py --strict   # exit 1 on unexplained drift

Read-only and safe: it parses the source with `ast` and never imports, runs
or edits your code. No GPU, no torch, no data required.

WHY IT EXISTS
    Hyperparameters are hardcoded in more than thirty files, one copy each.
    There is no single place that answers "which widths did this experiment
    use, and for how many epochs?". This tool reconstructs that table from
    the source, so the answer can never drift from the code.

WHAT IT UNDERSTANDS
    module-level assignment      NSEEDS = 5
    the `else` branch of         if SMOKE: ...  else: <-- this branch wins,
      an `if SMOKE:` block                          because SMOKE is False
                                                    for real runs
    environment overrides        int(os.environ.get("PAIRS", "3")) -> env:PAIRS=3

HOW TO READ THE OUTPUT
    Files are compared within GROUPS.
      * Drift inside a group is almost certainly a bug: those files are meant
        to be copies of each other. Flagged "<<< DRIFT".
      * Some differences are deliberate (a heavy script samples fewer pairs
        than a cheap one). Those are declared in KNOWN_INTENTIONAL below,
        together with the reason, and printed as "(intended)".
      * The final table compares groups against each other, where differences
        are expected.
"""
import argparse
import ast
import os
import re
import sys
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KEYS = [
    "SMOKE", "RUN_TAG", "SHARD_ID", "NUM_SHARDS", "SEED_BASE", "MODE", "SHARD",
    "REGIMES", "ACTS", "WIDTHS", "NSEEDS", "EPOCHS", "LR",
    "WARMUP_EPOCHS", "CLIP_NORM", "BATCH", "N_EVAL", "T_GRID", "TGRID",
    "MATCH_ITERS", "PAIRS", "DIN", "K",
    "DF_ENABLE", "DF_ACTS", "DF_BATCH", "DF_MICRO", "DF_ITERS", "DF_NZ",
    "DF_EPS", "DF_RICHARDSON", "FD_EPS", "FD_RICH",
    "FISHER_N", "MICRO", "LAM_REL", "CG_ITERS", "CG_TOL", "POWER_ITERS",
    "BUDGET_H", "RESUME", "SELFTEST", "ANCHOR", "BASE",
]

GROUPS = OrderedDict([
    ("train-mlp  (MLP / MNIST)",
     [f"src/01_train/mlp_train_shard{i}.py" for i in range(6)]),
    ("train-cnn  (CNN / FashionMNIST)",
     [f"src/01_train/cnn_train_shard{i}.py" for i in range(6)]),
    ("train-ts   (Teacher-Student)",
     [f"src/01_train/ts_train_shard{i}.py" for i in range(6)]),
    ("geodesic   (Sec. 5.1)",
     [f"src/02_geodesic/{m}_measure_geodesic.py" for m in ("mlp", "cnn", "ts")]),
    ("final      (rho*, R, Fisher length)",
     [f"src/03_final/{m}_measure_final.py" for m in ("mlp", "cnn", "ts")]),
    ("profile    (along-path)",
     [f"src/04_profile/measure_profile_{q}.py"
      for q in ("christoffel", "length", "shape")]),
    ("sweep      (damping lambda)",
     ["src/05_lambda_sweep/lambda_sweep_ntk.py",
      "src/05_lambda_sweep/lambda_sweep_sp.py",
      "src/05_lambda_sweep/lambda_sweep_mup.py",
      "src/05_lambda_sweep/lambda_sweep_v1_legacy.py"]),
    ("analysis",
     ["src/06_analysis/decompose_dF.py", "src/06_analysis/remeasure_dF_cnn.py"]),
])

# These MUST differ -- that is their whole purpose.
EXPECTED_TO_DIFFER = {"SHARD_ID", "MODE"}

# Deliberate differences, with the reason taken from the file's own docstring.
KNOWN_INTENTIONAL = {
    ("sweep", "REGIMES"): "each file sweeps exactly one regime (ntk / sp / mup)",
    ("sweep", "ACTS"): "the v1 legacy script used 2 activations; the three "
                       "per-regime scripts widened this to 4",
    ("sweep", "PAIRS"): "v1 legacy used 2 pairs per cell; the three per-regime "
                        "scripts use 3",
    ("sweep", "BUDGET_H"): "v1 legacy predates the wall-clock budget guard",
    ("profile", "PAIRS"): "christoffel/shape are EXPENSIVE (each pair costs TGRID "
                          "CG solves) -> 3 pairs; length is cheap (one Fisher-vector "
                          "product per grid point, no CG) -> keeps all 10 pairs",
    ("profile", "TGRID"): "christoffel/shape use the 9-point grid inherited from the "
                          "geodesic round (Green's-function quadrature); length uses "
                          "21 points to resolve the SHAPE of the curve",
}


def _val(node):
    """Render an AST node as a short readable string."""
    try:
        s = ast.unparse(node)
    except Exception:
        return "?"
    m = re.search(r"os\.environ\.get\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]?([^'\")]*)['\"]?\s*\)", s)
    if m:
        return f"env:{m.group(1)}={m.group(2)}"
    return re.sub(r"\s+", " ", s).strip()


def scan(path):
    """Return {NAME: value} parsed from source. Never executes the file."""
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None
    try:
        tree = ast.parse(open(full, encoding="utf-8", errors="replace").read())
    except SyntaxError as e:
        return {"__error__": f"parse failed: {e}"}

    found = {}

    def take(stmts, prefer=False):
        for n in stmts:
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id in KEYS:
                        if prefer or t.id not in found:
                            found[t.id] = _val(n.value)
            elif isinstance(n, ast.If):
                if _val(n.test) == "SMOKE":
                    take(n.orelse, prefer=True)   # real-run branch wins
                    take(n.body, prefer=False)
                else:
                    take(n.body)
                    take(n.orelse)

    take(tree.body)
    return found


def render(groups, md=False):
    drift_total, lines = 0, []
    for gname, files in groups.items():
        data = OrderedDict((f, d) for f in files for d in [scan(f)] if d is not None)
        if not data:
            continue
        keys = [k for k in KEYS if any(k in d for d in data.values())]
        if not keys:
            continue

        short = [os.path.basename(f) for f in data]
        w = max([len(k) for k in keys] + [8])
        cw = max(max((len(s) for s in short), default=8), 14)

        if md:
            lines.append(f"\n### {gname}  ({len(data)} files)\n")
            lines.append("| variable |" + "|".join(f" `{s}` " for s in short) + "| |")
            lines.append("|---|" + "|".join("---" for _ in short) + "|---|")
        else:
            lines.append(f"\n{'='*78}\n {gname}  ({len(data)} files)\n{'='*78}")
            lines.append(f"{'variable':<{w}} " + " ".join(f"{s[:cw]:<{cw}}" for s in short))
            lines.append("-" * (w + 1 + (cw + 1) * len(short)))

        gkey = gname.split()[0]
        notes = []
        for k in keys:
            vals = [data[f].get(k, "-") for f in data]
            differs = len({v for v in vals if v != "-"}) > 1 and k not in EXPECTED_TO_DIFFER
            reason = KNOWN_INTENTIONAL.get((gkey, k))
            drift = differs and reason is None
            drift_total += bool(drift)
            if differs and reason:
                flag, mdflag = "  (intended)", " *(intended)*"
                notes.append(f"`{k}`: {reason}" if md else f"  (intended) {k}: {reason}")
            elif drift:
                flag, mdflag = " <<< DRIFT", " **DRIFT**"
            else:
                flag, mdflag = "", ""
            if md:
                lines.append(f"| `{k}` |" + "|".join(f" `{v}` " for v in vals) + f"|{mdflag} |")
            else:
                lines.append(f"{k:<{w}} " + " ".join(f"{v[:cw]:<{cw}}" for v in vals) + flag)

        if notes:
            lines.append("\n**Intended differences in this group:**\n" if md
                         else "\nIntended differences in this group:")
            lines.extend((f"- {n}" if md else n) for n in notes)
    return "\n".join(lines), drift_total


def cross_group(md=False):
    rep = OrderedDict()
    for gname, files in GROUPS.items():
        for f in files:
            d = scan(f)
            if d:
                rep[gname] = d
                break
    names = list(rep)
    keys = [k for k in KEYS if sum(k in d for d in rep.values()) > 1]
    out = ["\n### Across groups (differences here are expected)\n" if md else
           f"\n{'='*78}\n Across groups -- differences here are expected\n{'='*78}"]
    if md:
        out.append("| variable |" + "|".join(f" {n.split()[0]} " for n in names) + "|")
        out.append("|---|" + "|".join("---" for _ in names) + "|")
    for k in keys:
        vals = [rep[n].get(k, "-") for n in names]
        if len({v for v in vals if v != "-"}) <= 1:
            continue
        if md:
            out.append(f"| `{k}` |" + "|".join(f" `{v}` " for v in vals) + "|")
        else:
            out.append(f"{k:<16} " + " ".join(f"{v[:18]:<18}" for v in vals))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Report configuration drift across experiment scripts.")
    ap.add_argument("--md", action="store_true", help="emit markdown")
    ap.add_argument("--only", help="comma-separated group prefixes, e.g. train,profile")
    ap.add_argument("--strict", action="store_true", help="exit 1 on unexplained drift")
    a = ap.parse_args()

    groups = GROUPS
    if a.only:
        want = [x.strip() for x in a.only.split(",")]
        groups = OrderedDict((g, f) for g, f in GROUPS.items()
                             if any(g.startswith(w) for w in want))

    body, drift = render(groups, md=a.md)
    print(body)
    print(cross_group(md=a.md))
    print(f"\nTOTAL: {drift} unexplained difference(s) within a group."
          if drift else "\nTOTAL: no unexplained differences within any group.")
    return 1 if (a.strict and drift) else 0


if __name__ == "__main__":
    sys.exit(main())
