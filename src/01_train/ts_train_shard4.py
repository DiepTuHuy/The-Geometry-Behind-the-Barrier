#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEACHER-STUDENT 3-REGIME x 5 act (over-realization). Param module dung chung MLP (validated).
Vá đúng ba điểm yếu bản cũ: 5 seed (thay vì 2), train tới HỘI TỤ (kiểm acc),
và quét width tới 4096 (thay vì dừng ~1024) --- để (A) ∂F và (B) barrier CHỒNG dải width.

Đo: barrier sau căn permutation (MLP -> KHÔNG BN, KHÔNG REPAIR) và ||∂F||_op (hàm smooth).
Setup: hai MLP KHOI TAO DOC LAP train tren MNIST that (khong teacher, khong over-realization).

CÁCH DÙNG: python param_ts_merged.py  (SMOKE=True để test ~phút; đặt False khi chạy thật)
BAN GOP: file nay chay CA 6 shard tuan tu trong 1 phien (setup/self-test/coord-check chi 1 lan).
Output van la param_ts_shard{0..5}.csv rieng tung shard -> gộp bằng script merge (kind=net/pair như ResNet).
Resume: tự kéo ckpt cũ từ /kaggle/input (như file ResNet).
"""
import os, sys, time, math, glob, traceback, itertools, shutil
try:
    sys.stdout.reconfigure(line_buffering=True); sys.stderr.reconfigure(line_buffering=True)
except Exception: pass
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import torchvision
try:
    from torch.func import functional_call, jvp as _fjvp, vjp as _fvjp, jacrev as _jacrev
    _HAS_FUNC = True
except Exception:
    _HAS_FUNC = False

# ============================================================ CONFIG
SMOKE      = False         # chay that
NUM_SHARDS = 6
SHARD_ID   = 4             # <<< FILE NAY = SHARD 4: mup relu,gelu,tanh (moi acc Kaggle chay 1 file)
RESUME     = True
RUN_TAG    = "pts_v2"        # <<< v2: warmup+clip; tag moi -> KHONG nap ckpt v1
OUT_DIR    = "."
SEED_BASE  = 4321
DIN, K     = 64, 10        # teacher-student synthetic

if SMOKE:
    REGIMES= ["ntk","sp","mup"]
    ACTS   = ["gelu","tanh"]; WIDTHS=[64,128]; NSEEDS=2; EPOCHS=6; LR=0.1
    WARMUP_EPOCHS = 8; CLIP_NORM = 1.0   # warmup thuc te = min(8, epochs//5) -> SMOKE tu ngan lai
    T_GRID=5; MATCH_ITERS=5; BATCH=256; N_EVAL=2000
    DF_ENABLE=True; DF_ACTS=["gelu","tanh","swish","softplus"]; DF_BATCH=128; DF_MICRO=64; DF_ITERS=6; DF_NZ=2
    DF_EPS=3e-3; DF_RICHARDSON=False
else:
    REGIMES= ["ntk","sp","mup"]
    ACTS   = ["relu","gelu","tanh","swish","softplus"]     # 5 ham
    WIDTHS = [64,128,256,512,1024,2048,4096]
    NSEEDS = 5; EPOCHS = 100; LR = 0.1              # SGD dong nhat; v1 cho thay LR 0.1 KHONG an toan o SP width lon nhat -> v2 them warmup+clip (ap DONG DEU moi cell)
    WARMUP_EPOCHS = 8; CLIP_NORM = 1.0
    T_GRID=21; MATCH_ITERS=8; BATCH=256; N_EVAL=5000
    DF_ENABLE=True; DF_ACTS=["gelu","tanh","swish","softplus"]; DF_BATCH=2048; DF_MICRO=64; DF_ITERS=20; DF_NZ=5   # dong nhat voi MLP
    DF_EPS=3e-3; DF_RICHARDSON=True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def set_seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ============================================================ MODEL (MLP 2 hidden)
def make_act(n): return {"relu":nn.ReLU,"gelu":nn.GELU,"tanh":nn.Tanh,"swish":nn.SiLU,"softplus":nn.Softplus}[n]()
SMOOTH={"gelu","tanh","swish","softplus"}   # relu kink -> chi barrier
BASE=64
def param_cfg(regime, fin, fout, kind):   # (init_std, fwd_mult, lr_scale) -- validated coord-check v6
    ss=math.sqrt(fin)
    if regime=="sp":   return (1.0/ss, 1.0, 1.0)
    if regime=="ntk":  return (1.0, 1.0/ss, 1.0)
    if regime=="mup":
        if kind=="input":   return (1.0/ss, 1.0, (fout/BASE)**1.0)
        if kind=="hidden":  return (1.0/ss, 1.0, (fin/BASE)**0.7)
        return (1.0/ss, (BASE/fin)**0.5, (fin/BASE)**(-0.5))          # readout
    raise ValueError(regime)
class ScaledLinear(nn.Module):
    def __init__(self,fin,fout,regime,kind):
        super().__init__()
        istd,self.fmul,self.lr_scale=param_cfg(regime,fin,fout,kind)
        self.weight=nn.Parameter(torch.randn(fout,fin)*istd); self.bias=nn.Parameter(torch.zeros(fout))
    def forward(self,x): return self.fmul*F.linear(x,self.weight)+self.bias
class MLP(nn.Module):
    def __init__(self, width, act, regime="ntk", din=DIN, k=K):
        super().__init__()
        self.fc1=ScaledLinear(din,width,regime,"input"); self.fc2=ScaledLinear(width,width,regime,"hidden"); self.fc3=ScaledLinear(width,k,regime,"output")
        self.a1=make_act(act); self.a2=make_act(act); self.width=width; self.regime=regime
    def forward(self,x,return_acts=False):
        h1=self.a1(self.fc1(x)); h2=self.a2(self.fc2(h1)); o=self.fc3(h2)
        return (o,[h1,h2,o]) if return_acts else o
    def opt_groups(self,base_lr):
        return [{"params":[m.weight,m.bias],"lr":base_lr*m.lr_scale} for m in [self.fc1,self.fc2,self.fc3]]

# perm groups: h1 (sau fc1), h2 (sau fc2). fc3 out = lớp (không hoán vị).
def perm_spec(model):
    ag = {"fc1.weight":["h1",None], "fc1.bias":["h1"],
          "fc2.weight":["h2","h1"], "fc2.bias":["h2"],
          "fc3.weight":[None,"h2"], "fc3.bias":[None]}
    sd = model.state_dict(); gs = {}
    for n,axes in ag.items():
        for a,g in enumerate(axes):
            if g is not None: gs[g] = sd[n].shape[a]
    return ag, gs
def apply_perm(sd, ag, perms):
    out={}
    for n,t in sd.items():
        if n in ag:
            tt=t
            for a,g in enumerate(ag[n]):
                if g is not None: tt=tt.index_select(a, perms[g])
            out[n]=tt.clone()
        else: out[n]=t.clone()
    return out
def _perm_except(t, axes, perms, exc):
    tt=t
    for a,g in enumerate(axes):
        if g is not None and a!=exc: tt=tt.index_select(a, perms[g])
    return tt
def weight_matching(ag, gs, sdA, sdB, iters, seed=0):
    rng=np.random.RandomState(seed); perms={g:torch.arange(n) for g,n in gs.items()}
    g2pa={g:[] for g in gs}
    for n,axes in ag.items():
        for a,g in enumerate(axes):
            if g is not None: g2pa[g].append((n,a))
    groups=list(gs)
    for it in range(iters):
        moved=0
        for g in [groups[i] for i in rng.permutation(len(groups))]:
            n=gs[g]; S=torch.zeros(n,n,dtype=torch.float64)
            for (name,axis) in g2pa[g]:
                A=sdA[name].double(); B=_perm_except(sdB[name].double(), ag[name], perms, axis)
                S += torch.movedim(A,axis,0).reshape(n,-1) @ torch.movedim(B,axis,0).reshape(n,-1).T
            ci=linear_sum_assignment(-S.numpy())[1]; new=torch.as_tensor(ci,dtype=torch.long)
            if not torch.equal(new,perms[g]): moved+=1
            perms[g]=new
        if moved==0: break
    return perms

# ============================================================ ∂F (như file ResNet, đã validate máy-precision)
def _pb(m): return ({k:v.detach() for k,v in m.named_parameters()},{k:v.detach() for k,v in m.named_buffers()})
def _call(m,p,b,x): return functional_call(m,{**p,**b},(x,))
def fisher_vp(m,p,b,x,v,micro):
    B=x.shape[0]; acc=None
    for i in range(0,B,micro):
        xb=x[i:i+micro]
        def f(pp): return _call(m,pp,b,xb)
        logits,Jv=_fjvp(f,(p,),(v,)); pr=torch.softmax(logits,1)
        s=pr*Jv-pr*(pr*Jv).sum(1,keepdim=True)
        JTs=_fvjp(f,p)[1](s)[0]
        acc={k:JTs[k].detach() for k in JTs} if acc is None else {k:acc[k]+JTs[k].detach() for k in acc}
    return {k:acc[k]/B for k in acc}
def _vnorm(a): return float(torch.sqrt(torch.clamp(sum((a[k]*a[k]).sum() for k in a),min=0)))
def _vscale(a,c): return {k:a[k]*c for k in a}
def _vaxpy(a,c,b): return {k:a[k]+c*b[k] for k in a}
def _vrand(ref,gen): 
    r={k:torch.randn(v.shape,generator=gen,device=v.device,dtype=v.dtype) for k,v in ref.items()}
    return _vscale(r,1.0/max(_vnorm(r),1e-30))
def dFz(m,p,b,x,z,v,eps,micro,rich):
    def cd(e):
        Fp=fisher_vp(m,_vaxpy(p,e,z),b,x,v,micro); Fm=fisher_vp(m,_vaxpy(p,-e,z),b,x,v,micro)
        return {k:(Fp[k]-Fm[k])/(2*e) for k in p}
    if rich:
        d1,d2=cd(eps),cd(eps/2); return {k:(4*d2[k]-d1[k])/3 for k in p}
    return cd(eps)
def dF_for_z(m,p,b,x,z,eps,iters,micro,rich,gen):
    u=_vrand(p,gen); lam=0.0
    for _ in range(iters):
        Au=dFz(m,p,b,x,z,u,eps,micro,rich); lam=_vnorm(Au)
        if lam<1e-30: break
        u=_vscale(Au,1.0/lam)
    return lam
def measure_dF(m,x,eps,iters,nz,micro,rich,seed):
    if not _HAS_FUNC: return None
    m.eval(); p,b=_pb(m); gen=torch.Generator(device=x.device); gen.manual_seed(seed)
    return float(np.max([dF_for_z(m,p,b,x,_vrand(p,gen),eps,iters,micro,rich,gen) for _ in range(nz)]))

# ============================================================ SELF-TESTS
def self_test_perm():
    log("  [self-test] perm invariance MLP ...")
    set_seed(0); m=MLP(48,"tanh").eval(); ag,gs=perm_spec(m)
    rng=np.random.RandomState(5); perms={g:torch.as_tensor(rng.permutation(n),dtype=torch.long) for g,n in gs.items()}
    m2=MLP(48,"tanh").eval(); m2.load_state_dict(apply_perm(m.state_dict(),ag,perms))
    x=torch.randn(8,DIN); d=(m(x)-m2(x)).abs().max().item()
    assert d<1e-4, f"PERM MLP sai: {d:.2e}"; log(f"  [self-test] OK perm  max|f(w)-f(pi.w)|={d:.2e}")
def self_test_dF():
    if not _HAS_FUNC: log("  [self-test] BỎ ∂F: thiếu torch.func"); return
    log("  [self-test] ∂F vs Fisher tường minh (float64) ...")
    torch.manual_seed(0)
    m=MLP(6,"tanh",din=4,k=3).double().eval(); x=torch.randn(6,4,dtype=torch.float64)
    p,b=_pb(m); keys=list(p.keys())
    flat=lambda d: torch.cat([d[k].reshape(-1) for k in keys])
    def unflat(v):
        o={};i=0
        for k in keys: n=p[k].numel(); o[k]=v[i:i+n].reshape(p[k].shape); i+=n
        return o
    def Fexp(pp):
        J=_jacrev(lambda q:_call(m,q,b,x))(pp); B=x.shape[0]
        Jf=torch.cat([J[k].reshape(B,3,-1) for k in keys],2); pr=torch.softmax(_call(m,pp,b,x),1)
        S=torch.diag_embed(pr)-pr.unsqueeze(2)*pr.unsqueeze(1)
        return torch.einsum('bki,bkl,blj->ij',Jf,S,Jf)/B
    vf=torch.randn(flat(p).numel(),dtype=torch.float64)
    e1=(flat(fisher_vp(m,p,b,x,unflat(vf),micro=6))-Fexp(p)@vf).abs().max().item()
    assert e1<1e-8, f"FVP sai {e1:.2e}"; log(f"  [self-test] FVP==F tường minh: {e1:.2e}")
    gen=torch.Generator().manual_seed(1); z=_vrand(p,gen); eps=1e-3
    fd=flat(dFz(m,p,b,x,z,unflat(vf),eps,micro=6,rich=False))
    ex=((Fexp(_vaxpy(p,eps,z))-Fexp(_vaxpy(p,-eps,z)))/(2*eps))@vf
    rel=(fd-ex).abs().max().item()/max(ex.abs().max().item(),1e-12)
    assert rel<1e-6, f"∂F sai {rel:.2e}"; log(f"  [self-test] ∂F==tường minh: {rel:.2e}")

# ============================================================ DATA (teacher-student)
TEACHER_W=32; _CACHE={}
def _teacher():
    if "t" in _CACHE: return _CACHE["t"]
    set_seed(1234); t=MLP(TEACHER_W,"relu","sp").to(DEVICE).eval()   # teacher: ReLU width 32, SP
    for p in t.parameters(): p.requires_grad_(False)
    _CACHE["t"]=t; return t
@torch.no_grad()
def load_mnist(train):   # ten giu de khoi sua main; that ra la teacher-student
    key=("ts",train); 
    if key in _CACHE: return _CACHE[key]
    n=20000 if train else 10000; t=_teacher(); g=torch.Generator().manual_seed(1 if train else 2)
    X=torch.randn(n,DIN,generator=g); ys=[]
    for i in range(0,n,4096): ys.append(t(X[i:i+4096].to(DEVICE)).argmax(1).cpu())
    Y=torch.cat(ys); _CACHE[key]=(X,Y); return X,Y

# ============================================================ TRAIN / EVAL / BARRIER
def train(model, X, Y, epochs):
    model.to(DEVICE).train()
    w0=torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()
    opt=torch.optim.SGD(model.opt_groups(LR),momentum=0.9)   # SGD dong nhat + per-layer lr_scale (regime)
    wu=min(WARMUP_EPOCHS, max(1, epochs//5))                 # v2: warmup tuyen tinh -> cosine; SMOKE tu co wu ngan
    def _lr_lambda(ep):
        if ep < wu: return (ep+1)/wu
        prog=(ep-wu)/max(epochs-wu,1); return 0.5*(1.0+math.cos(math.pi*prog))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,_lr_lambda); n=X.shape[0]
    for ep in range(epochs):
        perm=torch.randperm(n)
        for i in range(0,n,BATCH):
            idx=perm[i:i+BATCH]; x=X[idx].to(DEVICE); y=Y[idx].to(DEVICE)
            opt.zero_grad(set_to_none=True); F.cross_entropy(model(x),y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)   # v2: chan spike dau run
            opt.step()
        sched.step()
    wT=torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    wmove=float((wT-w0).norm()/max(float(w0.norm()),1e-9)); model.eval(); return model, wmove
@torch.no_grad()
def acc_of(model,X,Y):
    c=t=0
    for i in range(0,X.shape[0],1024):
        x=X[i:i+1024].to(DEVICE); y=Y[i:i+1024].to(DEVICE); c+=(model(x).argmax(1)==y).sum().item(); t+=y.numel()
    return c/max(t,1)
@torch.no_grad()
def loss_of(model,X,Y):
    s=0.;t=0
    for i in range(0,X.shape[0],1024):
        x=X[i:i+1024].to(DEVICE); y=Y[i:i+1024].to(DEVICE); s+=F.cross_entropy(model(x),y,reduction="sum").item(); t+=y.numel()
    return s/max(t,1)
def barrier(scratch, sdA, sdB, X, Y, tgrid):
    L=[]
    for tt in tgrid:
        scratch.load_state_dict({k:(1-tt)*sdA[k]+tt*sdB[k] for k in sdA}); scratch.to(DEVICE)
        L.append(loss_of(scratch,X,Y))
    L=np.array(L); return float(L.max()-0.5*(L[0]+L[-1]))

# ============================================================ SHARDING + resume + CSV
SHARD_PLAN = {   # 6 shard = 3 regime x 2 nhom ham; moi shard chay MOI width x 5 seed
    0:("ntk",["relu","gelu","tanh"]), 1:("ntk",["swish","softplus"]),
    2:("sp", ["relu","gelu","tanh"]), 3:("sp", ["swish","softplus"]),
    4:("mup",["relu","gelu","tanh"]), 5:("mup",["swish","softplus"]),
}
CKPT_DIR = os.path.join(OUT_DIR, f"ckpt_{RUN_TAG}")
def ckpt_path(regime,act,w,s):
    os.makedirs(CKPT_DIR,exist_ok=True); return os.path.join(CKPT_DIR,f"{regime}_{act}_w{w}_s{s}.pt")
def import_prev_ckpts():
    os.makedirs(CKPT_DIR,exist_ok=True)
    found=glob.glob(f"/kaggle/input/**/ckpt_{RUN_TAG}/*.pt",recursive=True); n=0
    for src in found:
        dst=os.path.join(CKPT_DIR,os.path.basename(src))
        if not os.path.exists(dst):
            try: shutil.copy(src,dst); n+=1
            except Exception as e: log(f"  (import ckpt lỗi: {e!r})")
    log(f"  [resume] nạp {n}/{len(found)} ckpt từ /kaggle/input" if found else "  [resume] không thấy ckpt cũ")
def build_worklist():
    if SMOKE: return [(r,a,WIDTHS) for r in REGIMES for a in ACTS]
    regime,acts=SHARD_PLAN[SHARD_ID]
    return [(regime,a,WIDTHS) for a in acts]
CSV=["kind","shard","regime","act","width","seed","seedA","seedB",
     "acc","accA","accB","barrier","dF_op","wmove","epochs","status"]
def write_row(path,row):
    new=not os.path.exists(path)
    with open(path,"a") as f:
        if new: f.write(",".join(CSV)+"\n")
        f.write(",".join(str(row.get(c,"")) for c in CSV)+"\n"); f.flush()

# ============================================================ MAIN
@torch.no_grad()
def _cv(m,x): return [float(t.abs().mean()) for t in m(x,return_acts=True)[1]]
def coordinate_check():
    log("  [coord-check] (muP nen PHANG; NTK lazy |upd|->0; SP lech theo width)")
    x=torch.randn(64,DIN,device=DEVICE); y=torch.randint(0,K,(64,),device=DEVICE)
    for r in REGIMES:
        row=[]
        for w in [64,256,1024]:
            torch.manual_seed(0); m=MLP(w,"gelu",r).to(DEVICE)
            a0=[t.detach().clone() for t in m(x,return_acts=True)[1]]
            opt=torch.optim.SGD(m.opt_groups(LR),momentum=0.0); opt.zero_grad(); F.cross_entropy(m(x),y).backward(); opt.step()
            with torch.no_grad(): a1=m(x,return_acts=True)[1]
            upd=[float((a1[i]-a0[i]).abs().mean()) for i in range(3)]
            row.append(f"w{w}:upd={upd[0]:.3f},{upd[1]:.3f},{upd[2]:.3f}")
        log(f"    {r:4s} | "+" | ".join(row))

def run_one_shard(sid, tgrid, Xtr, Ytr, Xte, Yte, Xbar, Ybar, Xdf):
    global SHARD_ID
    SHARD_ID = sid                                   # build_worklist()/write_row/out doc bien nay
    out=os.path.join(OUT_DIR, "param_ts_v2_shard%s.csv" % (SHARD_ID if not SMOKE else "SMOKE"))
    cells=build_worklist(); log(f"cells shard {SHARD_ID}: {cells} -> {out}")
    for (regime,act,widths) in cells:
        df_here=DF_ENABLE and act in DF_ACTS
        for w in widths:
            try:
                log(f"=== {act}/w{w} ===")
                sds,accs,dfs,wmvs=[],[],[],[]
                for sd_i in range(NSEEDS):
                    cp=ckpt_path(regime,act,w,sd_i)
                    if RESUME and os.path.exists(cp):
                        try: d=torch.load(cp,map_location="cpu",weights_only=False)
                        except TypeError: d=torch.load(cp,map_location="cpu")
                        sds.append(d["sd"]); accs.append(d["acc"]); dfs.append(d.get("dF")); wmvs.append(d.get("wmove")); log(f"  s{sd_i}: NAP ckpt acc={d['acc']:.3f}")
                    else:
                        set_seed(SEED_BASE+REGIMES.index(regime)*100000+WIDTHS.index(w)*100+ACTS.index(act)*7+sd_i)
                        m,wmove=train(MLP(w,act,regime),Xtr,Ytr,EPOCHS); a=acc_of(m,Xte,Yte)
                        dF=None
                        if df_here:
                            try: m.eval(); dF=measure_dF(m,Xdf,DF_EPS,DF_ITERS,DF_NZ,DF_MICRO,DF_RICHARDSON,seed=999+sd_i)
                            except Exception as e: traceback.print_exc(); log(f"  s{sd_i}: dF loi {e!r}")
                        sd={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
                        try: torch.save({"sd":sd,"acc":a,"dF":dF,"wmove":wmove},cp)
                        except Exception as e: log(f"  (ko luu ckpt {e!r})")
                        sds.append(sd); accs.append(a); dfs.append(dF); wmvs.append(wmove); log(f"  s{sd_i}: acc={a:.3f} dF={dF} wmove={wmove:.3f}")
                        del m; torch.cuda.empty_cache() if DEVICE=="cuda" else None
                    write_row(out,dict(kind="net",shard=SHARD_ID,regime=regime,act=act,width=w,seed=sd_i,
                                       acc=round(accs[-1],4),dF_op=("" if dfs[-1] is None else round(dfs[-1],6)),
                                       wmove=("" if wmvs[-1] is None else round(wmvs[-1],4)),epochs=EPOCHS,status="ok"))
                ag,gs=perm_spec(MLP(w,act,regime)); scratch=MLP(w,act,regime).to(DEVICE)
                for i,j in itertools.combinations(range(NSEEDS),2):
                    try:
                        perms=weight_matching(ag,gs,sds[i],sds[j],MATCH_ITERS,seed=i*13+j)
                        b=barrier(scratch,sds[i],apply_perm(sds[j],ag,perms),Xbar,Ybar,tgrid)
                        write_row(out,dict(kind="pair",shard=SHARD_ID,regime=regime,act=act,width=w,seedA=i,seedB=j,
                                           accA=round(accs[i],4),accB=round(accs[j],4),barrier=round(b,4),epochs=EPOCHS,status="ok"))
                        log(f"  [pair {i}-{j}] barrier={b:.3f}")
                    except Exception as e:
                        traceback.print_exc(); write_row(out,dict(kind="pair",shard=SHARD_ID,regime=regime,act=act,width=w,seedA=i,seedB=j,status="error:"+repr(e)[:50]))
                del sds,scratch; torch.cuda.empty_cache() if DEVICE=="cuda" else None
            except Exception as e:
                traceback.print_exc(); write_row(out,dict(kind="cell",shard=SHARD_ID,act=act,width=w,status="cell-error:"+repr(e)[:50]))
    log("DONE shard",SHARD_ID)

def main():
    log(f"DEVICE={DEVICE} SMOKE={SMOKE} SHARD={SHARD_ID}/{NUM_SHARDS} RECIPE=v2(warmup=min({WARMUP_EPOCHS},ep//5), clip={CLIP_NORM})")
    self_test_perm(); self_test_dF()          # setup chi chay 1 lan
    if RESUME: import_prev_ckpts()
    coordinate_check()
    tgrid=list(np.linspace(0,1,T_GRID))
    Xtr,Ytr=load_mnist(True); Xte,Yte=load_mnist(False)
    Xbar,Ybar=Xtr[:N_EVAL],Ytr[:N_EVAL]; Xdf=Xtr[:DF_BATCH].to(DEVICE)
    # 6 ACC KAGGLE SONG SONG: moi file chay dung 1 shard (SHARD_ID o CONFIG). SMOKE: full grid nho.
    run_one_shard(0 if SMOKE else SHARD_ID, tgrid, Xtr, Ytr, Xte, Yte, Xbar, Ybar, Xdf)
    log("DONE")

if __name__=="__main__": main()

# GHI CHÚ:
# - Muc dich: barrier co TRU toi 4096 khong (khong co co che sup) -> chot (A) va neg(B) khong phai artifact-width.
#   Nếu tụt dần -> tanh cũng sụp chậm (khớp ResNet) -> đổi phát biểu §5.5 sang "tốc độ sụp".
# - GATE tien-dang-ky (ap luc PLOT, khong ap luc chay): loai cell barrier neu median acc rot >5 diem
#   duoi plateau cua CHINH duong do (median acc tren width 256..2048). v1: rule nay cat dung 1 cell sp/softplus/4096.
# - Accuracy phai ~cao & ngang giua cac ham TRONG CUNG regime; NTK tran ~0.645 la tran kernel, KHONG chua.
# - Toan bo suite (MLP/CNN/ResNet-GroupNorm/TS) deu BN-free -> KHONG REPAIR o dau ca; barrier la hinh hoc tho sau weight matching.
