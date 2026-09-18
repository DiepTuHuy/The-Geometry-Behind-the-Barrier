# remeasure_dF_cnn.py -- re-measure dF for CNN checkpoints with suspect outliers.
# =================================================================================
# Run: place this file alongside a dataset holding ckpt_pcnn_v2 and one
# param_cnn*shard*.py, then `python remeasure_dF_cnn.py`. Paths under
# /kaggle/input/** and /content/drive/MyDrive/** are found automatically;
# override with MODULE_PATH=... CKPT_DIR=... Internet must be on to fetch
# FashionMNIST into ./data.
#
# Idea: measure_dF takes a maximum over 5 probe directions z, drawn from a
# generator seeded at 999+seed, so it is deterministic. Each flagged net is
# measured again (a) under the original configuration, to test reproduction,
# (b) with two other probe seeds, (c) at eps halved and doubled, and (d) with
# Richardson extrapolation off. A healthy seed from the same cell is measured
# once as a control.
#
# Verdict per net:
#   ARTIFACT   the original value does not reproduce outside the original
#              configuration -> replace it with med(alt) and note this in the
#              appendix. The row is never deleted and the mean is never used.
#   REPRODUCED every configuration gives a large value -> the effect is real
#              and needs further investigation.
#
# Output: remeasure_df_cnn.csv plus a verdict table.
# Note: the v1 or v2 CNN training file works equally well as the module source;
# the measurement machinery (network, measure_dF, DF config) is identical and
# this script never trains.
import os, glob, csv, sys, time
import importlib.util
import numpy as np
import torch

def _first(patterns):
    for p in patterns:
        if not p: continue
        hits = sorted(glob.glob(p, recursive=True))
        if hits: return hits[0]
    return None

MODULE_PATH = _first([
    os.environ.get("MODULE_PATH"),
    "param_cnn_v2_shard*.py", "param_cnn_shard*.py",                     # same directory
    "/kaggle/input/**/param_cnn_v2_shard*.py",
    "/kaggle/input/**/param_cnn_shard*.py",                              # either training version works
    "/content/param_cnn*shard*.py", "/content/drive/MyDrive/**/param_cnn*shard*.py",  # Colab
])
if MODULE_PATH is None:
    sys.exit("!! no param_cnn*shard*.py found anywhere -- place one training file (any shard) beside this script.")

CKPT_DIR_FOUND = _first([
    os.environ.get("CKPT_DIR"),
    "ckpt_pcnn_v2",
    "/kaggle/input/**/ckpt_pcnn_v2",
    "/content/drive/MyDrive/ckpt_pcnn_v2", "/content/drive/MyDrive/**/ckpt_pcnn_v2",
])
if CKPT_DIR_FOUND is None:
    sys.exit("!! ckpt_pcnn_v2 not found -- attach the dataset holding the checkpoints.")

npt = len(glob.glob(os.path.join(CKPT_DIR_FOUND, "*.pt")))
print(f"[preflight] module = {MODULE_PATH}")
print(f"[preflight] ckpt   = {CKPT_DIR_FOUND}  ({npt} .pt files; a complete CNN set is 300)")
if npt < 300:
    print("[preflight] !! WARNING: fewer than 300 checkpoints; some cells will report a missing checkpoint and be skipped.")

QUICK = os.environ.get("QUICK", "0") == "1"   # plumbing check only: small iters/batch, never for a verdict

spec = importlib.util.spec_from_file_location("cnnmod", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules["cnnmod"] = mod
spec.loader.exec_module(mod)
DEV = mod.DEVICE

# The 12 points whose dF exceeds 5x the cell median: (regime, act, width_mult, seed, original dF)
FLAGGED = [
    ("ntk", "softplus", 8, 1, 40.955), ("ntk", "softplus", 4, 4, 52.148),
    ("ntk", "tanh",     1, 4, 298.259), ("ntk", "tanh",    8, 4, 4.088),
    ("ntk", "softplus", 2, 3, 13.387), ("ntk", "gelu",     8, 1, 4.547),
    ("ntk", "softplus", 1, 2, 8.496),  ("ntk", "swish",    4, 1, 3.324),
    ("ntk", "softplus", 1, 1, 4.421),  ("sp",  "softplus", 2, 3, 6.270),
    ("ntk", "swish",    8, 1, 0.998),  ("mup", "gelu",     8, 1, 0.897),
]
# Controls: a healthy seed from the same cell (lowest dF in that cell)
CONTROLS = [
    ("ntk", "softplus", 8, 2), ("ntk", "softplus", 4, 3), ("ntk", "tanh", 1, 3),
    ("ntk", "tanh", 8, 0), ("ntk", "softplus", 2, 0), ("ntk", "gelu", 8, 4),
    ("ntk", "softplus", 1, 0), ("ntk", "swish", 4, 2), ("sp", "softplus", 2, 0),
    ("ntk", "swish", 8, 3), ("mup", "gelu", 8, 4),
]

def find_ckpt(regime, act, w, s):
    name = f"{regime}_{act}_w{w}_s{s}.pt"
    cand = os.path.join(CKPT_DIR_FOUND, name)
    return cand if os.path.exists(cand) else None

def load_net(regime, act, w, s):
    cp = find_ckpt(regime, act, w, s)
    if cp is None:
        return None, None
    try:
        d = torch.load(cp, map_location="cpu", weights_only=False)
    except TypeError:
        d = torch.load(cp, map_location="cpu")
    m = mod.MLP(w, act, regime)
    m.load_state_dict(d["sd"]); m.to(DEV).eval()
    return m, d

def main():
    iters = 3 if QUICK else mod.DF_ITERS
    nz = 2 if QUICK else mod.DF_NZ
    nb = 128 if QUICK else mod.DF_BATCH
    Xtr, _ = mod.load_mnist(True)
    Xdf = Xtr[:nb].to(DEV)

    rows = []
    def meas(m, seed, eps, rich, tag, meta):
        t0 = time.time()
        v = mod.measure_dF(m, Xdf, eps, iters, nz, mod.DF_MICRO, rich, seed=seed)
        rows.append(dict(**meta, cfg=tag, probe_seed=seed, eps=eps, rich=rich,
                         dF=v, sec=round(time.time() - t0, 1)))
        print(f"  {tag:14s} seed={seed:<6d} eps={eps:g} rich={int(rich)} -> dF={v:.4g} ({rows[-1]['sec']}s)", flush=True)
        return v

    print(f"DEVICE={DEV} iters={iters} nz={nz} batch={nb} (QUICK={QUICK})")
    verdicts = []
    for (r, a, w, s, orig) in FLAGGED:
        meta = dict(regime=r, act=a, width=w, seed=s, role="flagged", dF_goc=orig)
        m, d = load_net(r, a, w, s)
        print(f"\n### FLAGGED {r}/{a}/w{w}/s{s}  dF_orig={orig:g}  " + ("" if m else "!! checkpoint missing, skipping"))
        if m is None:
            continue
        v0 = meas(m, 999 + s, 3e-3, True,  "goc(taidien)", meta)   # production config; this string is
                                                                  # a value in the released audit CSV
        v1 = meas(m, 20001,   3e-3, True,  "probe#2",      meta)
        v2 = meas(m, 20002,   3e-3, True,  "probe#3",      meta)
        v3 = meas(m, 999 + s, 1.5e-3, True, "eps/2",       meta)
        v4 = meas(m, 999 + s, 6e-3, True,  "eps*2",        meta)
        v5 = meas(m, 999 + s, 3e-3, False, "no-richardson", meta)
        alts = [v1, v2, v3, v4, v5]
        big = orig / 5.0  # "still large" means above a fifth of the original value
        n_big = sum(x > big for x in alts)
        if n_big == 0:
            verdict = "ARTIFACT (does not reproduce outside the original config)"
        elif n_big == len(alts):
            verdict = "REPRODUCED (real effect -- investigate further)"
        else:
            verdict = f"UNCLEAR ({n_big}/{len(alts)} configs still large -- see the rows)"
        verdicts.append((r, a, w, s, orig, v0, float(np.median(alts)), verdict))
        del m; torch.cuda.empty_cache() if DEV == "cuda" else None

    for (r, a, w, s) in CONTROLS:
        meta = dict(regime=r, act=a, width=w, seed=s, role="control", dF_goc=None)
        m, d = load_net(r, a, w, s)
        print(f"\n--- control {r}/{a}/w{w}/s{s}  " + ("" if m else "!! checkpoint missing"))
        if m is None:
            continue
        meas(m, 999 + s, 3e-3, True, "goc(taidien)", meta)
        del m; torch.cuda.empty_cache() if DEV == "cuda" else None

    with open("remeasure_df_cnn.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader(); [wcsv.writerow(x) for x in rows]
    print("\n================ VERDICT ================")
    print(f"{'cell':28s} {'dF_orig':>9s} {'repeat':>9s} {'med(alt)':>9s}  verdict")
    for (r, a, w, s, orig, v0, medalt, verdict) in verdicts:
        print(f"{r}/{a}/w{w}/s{s:<10} {orig:9.3g} {v0:9.3g} {medalt:9.3g}  {verdict}")
    print("\n-> remeasure_df_cnn.csv holds every measurement. On ARTIFACT, replace dF_op for")
    print("   that row in the combined CSV with med(alt) and note it in the appendix.")
    print("   Never delete the row, and never use the mean.")

if __name__ == "__main__":
    main()