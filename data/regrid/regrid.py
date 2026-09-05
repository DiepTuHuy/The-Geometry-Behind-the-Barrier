#!/usr/bin/env python3
"""Re-measure the barrier on a finer grid.  The model, param_cfg, weight-matching and
apply_perm are copied verbatim from src/03_final/mlp_measure_final.py so the numbers are
directly comparable with data/final/mlp_pairs.csv."""
import math, sys, os, itertools, time, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DIN, K, BASE = 784, 10, 64
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
EVAL_N = 10000
TG_PAPER = 41                      # TGRID_FINE trong measure_final_mlp.py

def make_act(n): return {"relu":nn.ReLU,"gelu":nn.GELU,"tanh":nn.Tanh,
                         "swish":nn.SiLU,"softplus":nn.Softplus}[n]()
def param_cfg(regime, fin, fout, kind):
    ss = math.sqrt(fin)
    if regime=="sp":  return (1.0/ss, 1.0, 1.0)
    if regime=="ntk": return (1.0, 1.0/ss, 1.0)
    if regime=="mup":
        if kind=="input":  return (1.0/ss, 1.0, (fout/BASE)**1.0)
        if kind=="hidden": return (1.0/ss, 1.0, (fin/BASE)**0.7)
        return (1.0/ss, (BASE/fin)**0.5, (fin/BASE)**(-0.5))
    raise ValueError(regime)

class ScaledLinear(nn.Module):
    def __init__(self, fin, fout, regime, kind):
        super().__init__()
        istd, self.fmul, self.lr_scale = param_cfg(regime, fin, fout, kind)
        self.weight = nn.Parameter(torch.randn(fout, fin)*istd)
        self.bias   = nn.Parameter(torch.zeros(fout))
    def forward(self, x): return self.fmul*F.linear(x, self.weight) + self.bias

class NetMLP(nn.Module):
    def __init__(self, width, act, regime="ntk"):
        super().__init__()
        self.fc1 = ScaledLinear(DIN, width, regime, "input")
        self.fc2 = ScaledLinear(width, width, regime, "hidden")
        self.fc3 = ScaledLinear(width, K, regime, "output")
        self.a1, self.a2 = make_act(act), make_act(act)
    def forward(self, x): return self.fc3(self.a2(self.fc2(self.a1(self.fc1(x)))))

AG = {"fc1.weight":["h1",None],"fc1.bias":["h1"],
      "fc2.weight":["h2","h1"],"fc2.bias":["h2"],
      "fc3.weight":[None,"h2"],"fc3.bias":[None]}

def apply_perm(sd, perms):
    out={}
    for n,t in sd.items():
        if n in AG:
            for ax,g in enumerate(AG[n]):
                if g is not None: t = torch.index_select(t, ax, perms[g])
        out[n]=t
    return out

def _perm_except(t, axes, perms, exc):
    tt = t
    for a, g in enumerate(axes):
        if g is not None and a != exc: tt = tt.index_select(a, perms[g])
    return tt

def weight_matching(gs, sdA, sdB, iters=8, seed=0):
    """Sao y NGUYEN VAN measure_final_mlp.py."""
    from scipy.optimize import linear_sum_assignment
    rng = np.random.RandomState(seed); perms = {g: torch.arange(n) for g, n in gs.items()}
    g2pa = {g: [] for g in gs}
    for n_, axes in AG.items():
        for a, g in enumerate(axes):
            if g is not None: g2pa[g].append((n_, a))
    groups = list(gs)
    for _ in range(iters):
        moved = 0
        for g in [groups[i] for i in rng.permutation(len(groups))]:
            n = gs[g]; S = torch.zeros(n, n, dtype=torch.float64)
            for (name, axis) in g2pa[g]:
                A = sdA[name].double()
                B = _perm_except(sdB[name].double(), AG[name], perms, axis)
                S += torch.movedim(A, axis, 0).reshape(n, -1) @ torch.movedim(B, axis, 0).reshape(n, -1).T
            ci = linear_sum_assignment(-S.numpy())[1]
            new_ = torch.as_tensor(ci, dtype=torch.long)
            if not torch.equal(new_, perms[g]): moved += 1
            perms[g] = new_
        if moved == 0: break
    return perms

@torch.no_grad()
def loss_at(model, sdA, sdB, t, X, Y, bs=2000):
    sd = {k: (1-t)*sdA[k] + t*sdB[k] for k in sdA}
    model.load_state_dict(sd); model.eval()
    s, n = 0.0, X.shape[0]
    for i in range(0, n, bs):
        s += float(F.cross_entropy(model(X[i:i+bs]), Y[i:i+bs], reduction="sum"))
    return s/n

def main():
    cell = sys.argv[1]                       # vd ntk_gelu_w1024
    tg_fine = int(sys.argv[2]) if len(sys.argv)>2 else 401
    npairs  = int(sys.argv[3]) if len(sys.argv)>3 else 10
    regime, act, wtag = cell.split("_"); width = int(wtag[1:])
    ck = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt", "ckpt_pmlp_v2")

    import torchvision
    ds = torchvision.datasets.MNIST(os.path.expanduser("~/Downloads/Exp_hung/data"),
                                    train=True, download=False)
    X = (((ds.data.float()/255.0)-0.1307)/0.3081).reshape(-1,784)[:EVAL_N].to(DEV)
    Y = ds.targets.clone()[:EVAL_N].to(DEV)

    sds = []
    for s in range(5):
        p = os.path.join(ck, f"{cell}_s{s}.pt")
        o = torch.load(p, map_location="cpu", weights_only=False)
        sds.append(o["sd"])          # cau truc ckpt: {"sd":..., "acc":..., "wmove":...}
    model = NetMLP(width, act, regime).to(DEV)
    gs = {"h1":width, "h2":width}

    # luoi min chua tron ven luoi 41 diem cua bai:  (tg_fine-1) % 40 == 0
    assert (tg_fine-1) % (TG_PAPER-1) == 0, "luoi min phai chua luoi 41 diem"
    step = (tg_fine-1)//(TG_PAPER-1)
    ts = np.linspace(0,1,tg_fine)

    rows=[]
    for i,j in list(itertools.combinations(range(5),2))[:npairs]:
        t0=time.time()
        perms = weight_matching(gs, sds[i], sds[j], iters=8, seed=i*13 + j)   # sao y: seed=i*13+j
        sdB   = apply_perm(sds[j], perms)
        sdA   = {k:v.to(DEV) for k,v in sds[i].items()}
        sdB   = {k:v.to(DEV) for k,v in sdB.items()}
        Ls = np.array([loss_at(model, sdA, sdB, float(t), X, Y) for t in ts])
        base = 0.5*(Ls[0]+Ls[-1])
        B_paper = Ls[::step].max() - base
        B_fine  = Ls.max() - base
        rows.append(dict(cell=cell, pair=f"{i}{j}", B_paper=B_paper, B_fine=B_fine,
                         rel=(B_fine-B_paper)/max(B_paper,1e-12),
                         t_paper=float(ts[::step][Ls[::step].argmax()]),
                         t_fine=float(ts[Ls.argmax()]), sec=time.time()-t0))
        print(f"  {cell} pair{i}{j}: B41={B_paper:.5f}  B{tg_fine}={B_fine:.5f}  "
              f"lech={100*rows[-1]['rel']:+.2f}%  ({rows[-1]['sec']:.0f}s)", flush=True)
    out=os.path.join(os.path.dirname(os.path.abspath(__file__)), f"regrid_{cell}.json")
    json.dump(rows, open(out,"w"), indent=1)
    r=np.array([x["rel"] for x in rows])
    print(f"  => {cell}: lech trung vi {100*np.median(r):+.2f}%, toi da {100*r.max():+.2f}%")

if __name__ == "__main__":
    main()
