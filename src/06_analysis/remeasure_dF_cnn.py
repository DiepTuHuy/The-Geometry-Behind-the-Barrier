# remeasure_df_cnn.py — do lai dF cho cac net bi nghi DO NO (artifact) trong CNN v2
# =================================================================================
# CHAY: up file nay + (dataset chua ckpt_pcnn_v2 va param_cnn*shard*.py) -> !python remeasure_df_cnn.py
# Tu do duong dan tren KAGGLE (/kaggle/input/**) va COLAB (/content/drive/MyDrive/**),
# khong can set gi. Muon chi tay: MODULE_PATH=... CKPT_DIR=... python remeasure_df_cnn.py
# Kaggle: bat Internet ON (tai FashionMNIST ve ./data).
#
# Y TUONG: measure_dF lay max tren 5 probe z voi generator seed=999+sd (deterministic).
#   -> Do lai voi (a) DUNG config goc (tai lap?), (b) 2 probe-seed khac, (c) eps x1/2 va x2,
#      (d) tat Richardson. Net khoe cung cell do 1 lan lam control.
# VERDICT:
#   - ARTIFACT: gia tri goc KHONG tai lap ngoai config goc -> thay bang med(alt), ghi chu appendix
#   - TAI_LAP : moi config deu lon -> hien tuong that, dieu tra tiep
# Ket qua: remeasure_df_cnn.csv + bang verdict.
# LUU Y: file code v1 (param_cnn_shard0.py) hay v2 deu dung duoc — bo may do (MLP,
# measure_dF, DF config) identical; hai ban chi khac train()/tag, ma script khong train.
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
    "param_cnn_v2_shard*.py", "param_cnn_shard*.py",                     # cung thu muc
    "/kaggle/input/**/param_cnn_v2_shard*.py",
    "/kaggle/input/**/param_cnn_shard*.py",                              # dataset Kaggle (v1 cung ok)
    "/content/param_cnn*shard*.py", "/content/drive/MyDrive/**/param_cnn*shard*.py",  # Colab
])
if MODULE_PATH is None:
    sys.exit("!! Khong tim thay file param_cnn*shard*.py o dau ca — up 1 file code (bat ky shard nao) len canh script/dataset.")

CKPT_DIR_FOUND = _first([
    os.environ.get("CKPT_DIR"),
    "ckpt_pcnn_v2",
    "/kaggle/input/**/ckpt_pcnn_v2",
    "/content/drive/MyDrive/ckpt_pcnn_v2", "/content/drive/MyDrive/**/ckpt_pcnn_v2",
])
if CKPT_DIR_FOUND is None:
    sys.exit("!! Khong tim thay thu muc ckpt_pcnn_v2 — add dataset chua ckpt (Kaggle) hoac mount Drive (Colab).")

npt = len(glob.glob(os.path.join(CKPT_DIR_FOUND, "*.pt")))
print(f"[preflight] module = {MODULE_PATH}")
print(f"[preflight] ckpt   = {CKPT_DIR_FOUND}  ({npt} file .pt; du bo CNN v2 = 300)")
if npt < 300:
    print("[preflight] !! CANH BAO: it hon 300 ckpt — mot so cell se bi bao THIEU CKPT, van chay tiep.")

QUICK = os.environ.get("QUICK", "0") == "1"   # test nhanh plumbing: iters/batch nho (KHONG dung de phan xu)

spec = importlib.util.spec_from_file_location("cnnmod", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules["cnnmod"] = mod
spec.loader.exec_module(mod)
DEV = mod.DEVICE

# 12 diem ratio>5x median trong combined_pcnn.csv (v2) — (regime, act, width_mult, seed, dF_goc)
FLAGGED = [
    ("ntk", "softplus", 8, 1, 40.955), ("ntk", "softplus", 4, 4, 52.148),
    ("ntk", "tanh",     1, 4, 298.259), ("ntk", "tanh",    8, 4, 4.088),
    ("ntk", "softplus", 2, 3, 13.387), ("ntk", "gelu",     8, 1, 4.547),
    ("ntk", "softplus", 1, 2, 8.496),  ("ntk", "swish",    4, 1, 3.324),
    ("ntk", "softplus", 1, 1, 4.421),  ("sp",  "softplus", 2, 3, 6.270),
    ("ntk", "swish",    8, 1, 0.998),  ("mup", "gelu",     8, 1, 0.897),
]
# control: seed khoe cung cell (dF thap nhat trong cell theo CSV)
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
        print(f"\n### FLAGGED {r}/{a}/w{w}/s{s}  dF_goc={orig:g}  " + ("" if m else "!! THIEU CKPT, BO QUA"))
        if m is None:
            continue
        v0 = meas(m, 999 + s, 3e-3, True,  "goc(taidien)", meta)   # dung config production
        v1 = meas(m, 20001,   3e-3, True,  "probe#2",      meta)
        v2 = meas(m, 20002,   3e-3, True,  "probe#3",      meta)
        v3 = meas(m, 999 + s, 1.5e-3, True, "eps/2",       meta)
        v4 = meas(m, 999 + s, 6e-3, True,  "eps*2",        meta)
        v5 = meas(m, 999 + s, 3e-3, False, "no-richardson", meta)
        alts = [v1, v2, v3, v4, v5]
        big = orig / 5.0  # nguong "van lon": con >1/5 gia tri goc
        n_big = sum(x > big for x in alts)
        if n_big == 0:
            verdict = "ARTIFACT (khong tai lap ngoai config goc)"
        elif n_big == len(alts):
            verdict = "TAI_LAP (hien tuong that — dieu tra tiep)"
        else:
            verdict = f"MO ({n_big}/{len(alts)} config van lon — xem chi tiet)"
        verdicts.append((r, a, w, s, orig, v0, float(np.median(alts)), verdict))
        del m; torch.cuda.empty_cache() if DEV == "cuda" else None

    for (r, a, w, s) in CONTROLS:
        meta = dict(regime=r, act=a, width=w, seed=s, role="control", dF_goc=None)
        m, d = load_net(r, a, w, s)
        print(f"\n--- control {r}/{a}/w{w}/s{s}  " + ("" if m else "!! THIEU CKPT"))
        if m is None:
            continue
        meas(m, 999 + s, 3e-3, True, "goc(taidien)", meta)
        del m; torch.cuda.empty_cache() if DEV == "cuda" else None

    with open("remeasure_df_cnn.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader(); [wcsv.writerow(x) for x in rows]
    print("\n================ VERDICT ================")
    print(f"{'cell':28s} {'dF_goc':>9s} {'taidien':>9s} {'med(alt)':>9s}  verdict")
    for (r, a, w, s, orig, v0, medalt, verdict) in verdicts:
        print(f"{r}/{a}/w{w}/s{s:<10} {orig:9.3g} {v0:9.3g} {medalt:9.3g}  {verdict}")
    print("\n-> remeasure_df_cnn.csv (day du moi phep do). Neu ARTIFACT: thay dF_op cua dong do")
    print("   trong combined CSV bang med(alt) + ghi chu appendix; KHONG xoa dong, KHONG dung mean.")

if __name__ == "__main__":
    main()