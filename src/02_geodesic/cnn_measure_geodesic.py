#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 measure_geo.py  --  DO DO LECH TRAC DIA (§5.1) tu checkpoint da train
============================================================================
Vá mat xich con thieu cua paper: cac experiment (param_{mlp,cnn,ts}_v2) moi do
wmove / dF / barrier / acc, NHUNG chua do "truong trac dia gan duong noi suy
tuyen tinh". §5.1 hua "xay dung day du bo may do" cho:

        sup_t || gamma_g(t) - gamma_lin(t) ||          (do lech trac dia)

o BAC NHAT, qua bieu dien Green cua Bo de 4.10 va Christoffel cua G_F=F+lambda I:

  gamma_lin'' = 0  =>  residual trac dia cua duong thang = Gamma(delta,delta)
  xi = gamma_g - gamma_lin,  xi'' ~= -Gamma_gamma_lin(delta,delta)   (bac nhat)
  xi(t) = int_0^1 G(t,s) Gamma(delta,delta)(gamma_lin(s)) ds
  G(t,s) = s(1-t) neu s<=t, t(1-s) neu s>=t

Christoffel (Levi-Civita cho G_F, delta=delta):
  Gamma(delta,delta) = 1/2 G_F^{-1} [ 2 (d_delta F) delta  -  m ],
     m_l = delta^T (d_l F) delta = grad_w [ delta^T F(w) delta ]_l

Bo may TAI SU DUNG fisher_vp/dFz da validate may-precision cua ban; THEM:
  - grad_quad : m = grad_w <delta,F delta> qua autograd (jvp long trong grad)
  - cg_solve  : G_F^{-1} qua conjugate gradient (chi can fisher_vp)
  - green     : cau phuong Green

CANH BAO DIEN GIAI (theo Remark 5.1 + 4.5 cua paper):
  * Do lon TUYET DOI phu thuoc CG damping lambda -> CHI doc so mu theo width va
    ti so tuong doi (bat bien-scale). Script bao cao dev_rel = sup||xi||/||delta||.
  * Can Thm4.7(III) chi triet tieu khi alpha>2 (ly thuyet KHONG bao dam) -> ket
    luan phai neo tren PHEP DO nay, khong tren nguong tiem can.
  * dFz lay theo huong don vi delta_hat (khop eps da hieu chuan cho dF).

CACH DUNG:
  - Chinh MODE ("mlp"/"cnn"/"ts"), SHARD_ID (0..5), tro CKPT toi noi luu .pt.
  - SMOKE=True: tu train 2 net tinh de test toan tuyen (~phut) roi do.
  - Output: param_geo_{MODE}_shard{ID}.csv  (kind="geo"), resumable.
============================================================================
"""
import os, sys, time, math, glob, itertools, traceback
try: sys.stdout.reconfigure(line_buffering=True); sys.stderr.reconfigure(line_buffering=True)
except Exception: pass
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.func import functional_call, jvp as _fjvp, vjp as _fvjp, jacrev as _jacrev, grad as _grad
def _tv():   # torchvision chi can cho mlp/cnn (ts la synthetic)
    import torchvision; return torchvision

# ==================================================================== CONFIG (STANDALONE — hardcoded MODE, chay HET, chuan ICLR)
MODE       = "cnn"                     # <<< kien truc cua file nay
RESUME     = True
SMOKE      = os.environ.get("GEO_SMOKE","0")=="1"     # tuy chon: self-train ckpt nho de test toan tuyen (~phut)
ONLY_SHARD = os.environ.get("GEO_SHARD")              # tuy chon: "0".."5" chay 1 shard (song song nhieu account); None = HET
RUN_TAG    = {"mlp":"pmlp_v2","cnn":"pcnn_v2","ts":"pts_v2"}[MODE]

def _first(pats):
    for p in pats:
        if not p: continue
        h=sorted(glob.glob(p,recursive=True))
        if h: return h[0]
    return None
CKPT_ROOTS = [".","/kaggle/input","/content","/content/drive/MyDrive"]   # find_ckpt glob de-quy trong day
COMBINED   = _first([os.environ.get("COMBINED"),
    f"combined_p{MODE}*dFfixed*.csv", f"combined_p{MODE}*.csv",
    f"/kaggle/input/**/combined_p{MODE}*dFfixed*.csv", f"/kaggle/input/**/combined_p{MODE}*.csv",
    f"/content/drive/MyDrive/**/combined_p{MODE}*.csv"])
OUT_DIR    = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

if SMOKE:
    GEO_WIDTHS={"mlp":[16,32],"cnn":[1,2],"ts":[16,32]}[MODE]
    GEO_ACTS=["gelu","tanh"]; GEO_PAIRS=1; GEO_TGRID=5; GEO_BATCH=128; GEO_MICRO=64; NSEEDS=2; CG_ITERS=60; POWER_ITERS=12
else:
    GEO_WIDTHS={"mlp":[64,128,256,512,1024,2048,4096],"cnn":[1,2,4,8],"ts":[64,128,256,512,1024,2048,4096]}[MODE]
    GEO_ACTS=["gelu","tanh","swish","softplus"]      # C^3: KHONG relu (kink) — dong nhat DF_ACTS
    GEO_PAIRS=10                # = combinations(5,2): DONG NHAT voi barrier
    GEO_TGRID=9                 # Green quadrature (integrand tron)
    GEO_BATCH=2048              # = DF_BATCH: Fisher trong geo TRUNG dung batch da do dF (Xgeo == Xdf)
    GEO_MICRO=64; NSEEDS=5
    CG_ITERS=300; POWER_ITERS=20   # CG cap cao (auto-stop o tol; du cho lam_rel nho khi sweep)

FD_EPS=3e-3; FD_RICH=True; LAM_REL=1e-2
LAM_SWEEP=[1e-1,1e-2,1e-3] if os.environ.get("GEO_LAMSWEEP","0")=="1" else None  # bat: kiem so mu bat bien theo lambda (Rmk 5.1)
CG_TOL=1e-6
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DIN,K={"mlp":(784,10),"cnn":(None,10),"ts":(64,10)}[MODE]
SHARD_PLAN={0:("ntk",["relu","gelu","tanh"]),1:("ntk",["swish","softplus"]),
            2:("sp",["relu","gelu","tanh"]),3:("sp",["swish","softplus"]),
            4:("mup",["relu","gelu","tanh"]),5:("mup",["swish","softplus"])}
SHARD_ID=""   # khong dung (giu tuong thich cot CSV)

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def set_seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ==================================================================== MODEL
BASE=64
def make_act(n): return {"relu":nn.ReLU,"gelu":nn.GELU,"tanh":nn.Tanh,"swish":nn.SiLU,"softplus":nn.Softplus}[n]()
def param_cfg(regime, fin, fout, kind):
    ss=math.sqrt(fin)
    if regime=="sp":  return (1.0/ss,1.0,1.0)
    if regime=="ntk": return (1.0,1.0/ss,1.0)
    if regime=="mup":
        if kind=="input":  return (1.0/ss,1.0,(fout/BASE)**1.0)
        if kind=="hidden": return (1.0/ss,1.0,(fin/BASE)**0.7)
        return (1.0/ss,(BASE/fin)**0.5,(fin/BASE)**(-0.5))
    raise ValueError(regime)
class ScaledLinear(nn.Module):
    def __init__(self,fin,fout,regime,kind):
        super().__init__(); istd,self.fmul,self.lr_scale=param_cfg(regime,fin,fout,kind)
        self.weight=nn.Parameter(torch.randn(fout,fin)*istd); self.bias=nn.Parameter(torch.zeros(fout))
    def forward(self,x): return self.fmul*F.linear(x,self.weight)+self.bias
class ScaledConv(nn.Module):
    def __init__(self,cin,cout,k,st,pad,regime,kind):
        super().__init__(); fin=cin*k*k; istd,self.fmul,self.lr_scale=param_cfg(regime,fin,cout,kind)
        self.weight=nn.Parameter(torch.randn(cout,cin,k,k)*istd); self.st=st; self.pad=pad
    def forward(self,x): return self.fmul*F.conv2d(x,self.weight,None,self.st,self.pad)
def _gn(c): return nn.GroupNorm(1,c)

class NetMLP(nn.Module):     # mlp & ts
    def __init__(self, width, act, regime="ntk", din=DIN, k=K):
        super().__init__()
        self.fc1=ScaledLinear(din,width,regime,"input"); self.fc2=ScaledLinear(width,width,regime,"hidden"); self.fc3=ScaledLinear(width,k,regime,"output")
        self.a1=make_act(act); self.a2=make_act(act); self.width=width; self.regime=regime
    def forward(self,x): return self.fc3(self.a2(self.fc2(self.a1(self.fc1(x)))))
class NetCNN(nn.Module):     # cnn (ScaledConv + GroupNorm)
    def __init__(self, wm, act, regime="ntk", in_ch=1, k=K):
        super().__init__(); c=[16*wm,32*wm,64*wm]
        self.c1=ScaledConv(in_ch,c[0],3,1,1,regime,"input"); self.n1=_gn(c[0]); self.a1=make_act(act)
        self.c2=ScaledConv(c[0],c[1],3,1,1,regime,"hidden"); self.n2=_gn(c[1]); self.a2=make_act(act)
        self.c3=ScaledConv(c[1],c[2],3,1,1,regime,"hidden"); self.n3=_gn(c[2]); self.a3=make_act(act)
        self.pool=nn.MaxPool2d(2); self.fc=ScaledLinear(c[2],k,regime,"output"); self.width=wm; self.regime=regime
    def forward(self,x):
        h1=self.pool(self.a1(self.n1(self.c1(x)))); h2=self.pool(self.a2(self.n2(self.c2(h1)))); h3=self.a3(self.n3(self.c3(h2)))
        return self.fc(F.adaptive_avg_pool2d(h3,1).flatten(1))
    def opt_groups(self,base_lr):
        g=[{"params":[m.weight]+([m.bias] if hasattr(m,"bias") else []),"lr":base_lr*m.lr_scale} for m in [self.c1,self.c2,self.c3,self.fc]]
        g.append({"params":[p for n in [self.n1,self.n2,self.n3] for p in n.parameters()],"lr":base_lr}); return g

def build_net(width, act, regime):
    return NetCNN(width,act,regime) if MODE=="cnn" else NetMLP(width,act,regime)

# ---- perm spec (giong script tuong ung) ----
def perm_spec(model):
    if MODE=="cnn":
        ag={"c1.weight":["g1",None,None,None],"n1.weight":["g1"],"n1.bias":["g1"],
            "c2.weight":["g2","g1",None,None],"n2.weight":["g2"],"n2.bias":["g2"],
            "c3.weight":["g3","g2",None,None],"n3.weight":["g3"],"n3.bias":["g3"],
            "fc.weight":[None,"g3"],"fc.bias":[None]}
    else:
        ag={"fc1.weight":["h1",None],"fc1.bias":["h1"],
            "fc2.weight":["h2","h1"],"fc2.bias":["h2"],
            "fc3.weight":[None,"h2"],"fc3.bias":[None]}
    sd=model.state_dict(); gs={}
    for n,axes in ag.items():
        for a,g in enumerate(axes):
            if g is not None: gs[g]=sd[n].shape[a]
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
def weight_matching(ag, gs, sdA, sdB, iters=8, seed=0):
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

# ==================================================================== PRIMITIVES (giong script)
def _pb(m): return ({k:v.detach() for k,v in m.named_parameters()},{k:v.detach() for k,v in m.named_buffers()})
def _call(m,p,b,x): return functional_call(m,{**p,**b},(x,))
def fisher_vp(m,p,b,x,v,micro):
    B=x.shape[0]; acc=None
    for i in range(0,B,micro):
        xb=x[i:i+micro]
        def f(pp): return _call(m,pp,b,xb)
        logits,Jv=_fjvp(f,(p,),(v,)); pr=torch.softmax(logits,1)
        s=pr*Jv-pr*(pr*Jv).sum(1,keepdim=True); JTs=_fvjp(f,p)[1](s)[0]
        acc={k:JTs[k].detach() for k in JTs} if acc is None else {k:acc[k]+JTs[k].detach() for k in acc}
    return {k:acc[k]/B for k in acc}
def _vnorm(a): return float(torch.sqrt(torch.clamp(sum((a[k]*a[k]).sum() for k in a),min=0)))
def _vscale(a,c): return {k:a[k]*c for k in a}
def _vaxpy(a,c,b): return {k:a[k]+c*b[k] for k in a}
def _vdot(a,b): return float(sum((a[k]*b[k]).sum() for k in a))
def dFz(m,p,b,x,z,v,eps,micro,rich):
    def cd(e):
        Fp=fisher_vp(m,_vaxpy(p,e,z),b,x,v,micro); Fm=fisher_vp(m,_vaxpy(p,-e,z),b,x,v,micro)
        return {k:(Fp[k]-Fm[k])/(2*e) for k in p}
    if rich:
        d1,d2=cd(eps),cd(eps/2); return {k:(4*d2[k]-d1[k])/3 for k in p}
    return cd(eps)

# ==================================================================== NEW GEOMETRY
def quad_scalar(m,p,b,x,delta,micro):
    B=x.shape[0]; total=None
    for i in range(0,B,micro):
        xb=x[i:i+micro]
        def f(pp): return _call(m,pp,b,xb)
        logits,u=_fjvp(f,(p,),(delta,)); pr=torch.softmax(logits,1)
        Su=pr*u-pr*(pr*u).sum(1,keepdim=True); t=(u*Su).sum()
        total=t if total is None else total+t
    return total/B
def _quad_sum(m,p,b,xb,delta):          # SUM tren chunk cua u^T S u (khong chia B)
    def f(pp): return _call(m,pp,b,xb)
    logits,u=_fjvp(f,(p,),(delta,)); pr=torch.softmax(logits,1)
    Su=pr*u-pr*(pr*u).sum(1,keepdim=True); return (u*Su).sum()
def grad_quad(m,p,b,x,delta,micro):    # m_l = delta^T (d_l F) delta = grad_w<delta,F delta>
    B=x.shape[0]; acc=None                # cong grad theo micro-batch -> bo nho chan boi micro (KHONG giu graph ca batch)
    for i in range(0,B,micro):
        gi=_grad(lambda pp: _quad_sum(m,pp,b,x[i:i+micro],delta))(p)
        acc={k:gi[k].detach() for k in gi} if acc is None else {k:acc[k]+gi[k].detach() for k in acc}
    return {k:acc[k]/B for k in acc}

def gf_vp(m,p,b,x,v,lam,micro):
    Fv=fisher_vp(m,p,b,x,v,micro); return {k:Fv[k]+lam*v[k] for k in v}
def lam_max(m,p,b,x,micro,iters,seed=0):
    gen=torch.Generator(device=x.device).manual_seed(seed)
    u={k:torch.randn(v.shape,generator=gen,device=v.device,dtype=v.dtype) for k,v in p.items()}
    u=_vscale(u,1.0/max(_vnorm(u),1e-30)); lam=0.0
    for _ in range(iters):
        Au=fisher_vp(m,p,b,x,u,micro); lam=_vnorm(Au)
        if lam<1e-30: break
        u=_vscale(Au,1.0/lam)
    return lam
def cg_solve(m,p,b,x,rhs,lam,micro,x0=None,iters=80,tol=1e-6):
    xk={k:(torch.zeros_like(v) if x0 is None else x0[k].clone()) for k,v in rhs.items()}
    Ax=gf_vp(m,p,b,x,xk,lam,micro) if x0 is not None else {k:torch.zeros_like(v) for k,v in rhs.items()}
    r={k:rhs[k]-Ax[k] for k in rhs}; pdir={k:r[k].clone() for k in r}
    rs=_vdot(r,r); r0=max(rs,1e-300)
    for _ in range(iters):
        Ap=gf_vp(m,p,b,x,pdir,lam,micro); a=rs/max(_vdot(pdir,Ap),1e-300)
        xk={k:xk[k]+a*pdir[k] for k in xk}; r={k:r[k]-a*Ap[k] for k in r}
        rs2=_vdot(r,r)
        if rs2<=tol*tol*r0: break
        beta=rs2/max(rs,1e-300); pdir={k:r[k]+beta*pdir[k] for k in pdir}; rs=rs2
    return xk, math.sqrt(_vdot(r,r)/r0)
def christoffel_dd(m,p,b,x,delta,lam,micro,x0=None):
    dn=_vnorm(delta); dhat=_vscale(delta,1.0/dn)
    # (d_delta F) delta: tach hai muc eps de do eps-stability (parity voi validation dF). Cung so fisher_vp nhu Richardson.
    d1=dFz(m,p,b,x,dhat,dhat,FD_EPS,micro,False)                       # central diff @ eps
    d2=dFz(m,p,b,x,dhat,dhat,FD_EPS/2,micro,False)                     # @ eps/2
    t1r={k:(4*d2[k]-d1[k])/3 for k in d1} if FD_RICH else d1           # Richardson
    fd_instab=_vnorm({k:d1[k]-d2[k] for k in d1})/max(_vnorm(t1r),1e-30)  # |cd(eps)-cd(eps/2)|/|Richardson|
    t1={k:t1r[k]*dn*dn for k in t1r}
    mvec=grad_quad(m,p,b,x,delta,micro)
    rhs={k:2*t1[k]-mvec[k] for k in t1}
    sol,resid=cg_solve(m,p,b,x,rhs,lam,micro,x0=x0,iters=CG_ITERS,tol=CG_TOL)
    return _vscale(sol,0.5), resid, fd_instab

def green_matrix(ts):
    n=len(ts); G=torch.zeros(n,n,dtype=torch.float64)
    for i,t in enumerate(ts):
        for j,s in enumerate(ts):
            G[i,j]= s*(1-t) if s<=t else t*(1-s)
    return G

# ==================================================================== DATA
_CACHE={}
def load_data():
    if MODE=="mlp":
        if "d" in _CACHE: return _CACHE["d"]
        ds=_tv().datasets.MNIST("./data",train=True,download=True)
        X=((ds.data.float()/255.0)-0.1307)/0.3081; X=X.reshape(-1,784); _CACHE["d"]=X; return X
    if MODE=="cnn":
        if "d" in _CACHE: return _CACHE["d"]
        ds=_tv().datasets.FashionMNIST("./data",train=True,download=True)
        X=((ds.data.float()/255.0)-0.2860)/0.3530; X=X.unsqueeze(1); _CACHE["d"]=X; return X
    # ts: chi can input X ~ N(0,I_64) (Fisher = ky vong theo x)
    if "d" in _CACHE: return _CACHE["d"]
    g=torch.Generator().manual_seed(1); X=torch.randn(20000,DIN,generator=g); _CACHE["d"]=X; return X

# ==================================================================== CKPT IO
def ckpt_dir(): 
    d=os.path.join(OUT_DIR,f"ckpt_{RUN_TAG}"); os.makedirs(d,exist_ok=True); return d
def find_ckpt(regime,act,w,s):
    name=f"{regime}_{act}_w{w}_s{s}.pt"
    local=os.path.join(ckpt_dir(),name)
    if os.path.exists(local): return local
    for root in CKPT_ROOTS:
        hits=glob.glob(os.path.join(root,"**",f"ckpt_{RUN_TAG}",name),recursive=True)
        if hits: return hits[0]
    return None
def load_sd(path):
    try: d=torch.load(path,map_location="cpu",weights_only=False)
    except TypeError: d=torch.load(path,map_location="cpu")
    return d["sd"], d.get("acc"), d.get("dF")

# ==================================================================== CSV
CSV=["kind","mode","shard","regime","act","width","seedA","seedB",
     "dnorm","dev_geo","dev_rel","gamma_mid",
     "flen_A","flen_mid","flen_B","rq_A","rq_mid","rq_B",
     "lam","lam_rel","cg_resid","fd_instab","accA","accB","status"]
def write_row(path,row):
    new=not os.path.exists(path)
    with open(path,"a") as f:
        if new: f.write(",".join(CSV)+"\n")
        f.write(",".join(str(row.get(c,"")) for c in CSV)+"\n"); f.flush()
def already_done(path,regime,act,w,i,j,lam_rel):
    if not os.path.exists(path): return False
    key=f",{regime},{act},{w},{i},{j},"
    with open(path) as f:
        for ln in f:
            if key in ln and f"{lam_rel}" in ln and ln.strip().endswith("ok"): return True
    return False

# ==================================================================== SELF-TEST (dense, tiny)
def self_test():
    log("  [self-test] Gamma(dd) vp==dense + Green ...")
    old=torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    torch.manual_seed(0); m=NetMLP(6,"tanh","ntk",din=4,k=3).eval(); x=torch.randn(8,4)
    p,b=_pb(m); keys=list(p.keys())
    flat=lambda d: torch.cat([d[k].reshape(-1) for k in keys])
    def unflat(v):
        o={};i=0
        for k in keys: n=p[k].numel(); o[k]=v[i:i+n].reshape(p[k].shape); i+=n
        return o
    torch.manual_seed(3); dflat=torch.randn(flat(p).numel()); dflat=dflat/dflat.norm()*3.0
    delta=unflat(dflat); lam=1e-2
    def Fmat(wv):
        pp=unflat(wv); J=_jacrev(lambda q:_call(m,q,b,x))(pp); B=x.shape[0]
        Jf=torch.cat([J[k].reshape(B,3,-1) for k in keys],2)
        pr=torch.softmax(_call(m,pp,b,x),1); S=torch.diag_embed(pr)-pr.unsqueeze(2)*pr.unsqueeze(1)
        return torch.einsum('bki,bkl,blj->ij',Jf,S,Jf)/B
    w0=flat(p); dF=_jacrev(Fmat)(w0); Fd=Fmat(w0); P=w0.numel()
    t1=torch.einsum('ijl,l,j->i',dF,dflat,dflat); mv=torch.einsum('i,ijl,j->l',dflat,dF,dflat)
    g_dense=0.5*torch.linalg.solve(Fd+lam*torch.eye(P),2*t1-mv)
    global CG_ITERS,CG_TOL; oi,ot=CG_ITERS,CG_TOL; CG_ITERS,CG_TOL=300,1e-12
    g_vp,_,_=christoffel_dd(m,p,b,x,delta,lam,micro=8); CG_ITERS,CG_TOL=oi,ot
    rel=float((flat(g_vp)-g_dense).norm()/max(g_dense.norm(),1e-12))
    assert rel<1e-6, f"Gamma sai {rel:.2e}"
    ts=np.linspace(0,1,21); G=green_matrix(list(ts)); dt=ts[1]-ts[0]
    xi=(G@torch.full((len(ts),),0.7,dtype=torch.float64))*dt
    ge=float((xi-torch.tensor([0.7*t*(1-t)/2 for t in ts])).abs().max())
    assert ge<1e-10, f"Green sai {ge:.2e}"
    log(f"  [self-test] OK  Gamma rel={rel:.2e}  Green err={ge:.2e}")
    torch.set_default_dtype(old)

def _smoke_train(regime,act,w,seed):
    """train NHANH 1 net tinh de smoke-test toan tuyen (chi khi khong co ckpt)."""
    set_seed(seed); m=build_net(w,act,regime).to(DEVICE).train()
    if MODE=="cnn":
        ds=_tv().datasets.FashionMNIST("./data",train=True,download=True)
        X=(((ds.data.float()/255.0)-0.2860)/0.3530).unsqueeze(1)[:2000]; Y=ds.targets[:2000]
    elif MODE=="mlp":
        ds=_tv().datasets.MNIST("./data",train=True,download=True)
        X=(((ds.data.float()/255.0)-0.1307)/0.3081).reshape(-1,784)[:2000]; Y=ds.targets[:2000]
    else:
        g=torch.Generator().manual_seed(1); X=torch.randn(2000,DIN,generator=g)
        set_seed(1234); teach=NetMLP(32,"relu","sp").eval()
        with torch.no_grad(): Y=teach(X).argmax(1)
    groups=m.opt_groups(0.1) if hasattr(m,"opt_groups") else [{"params":[pp for pp in m.parameters()],"lr":0.1}]
    opt=torch.optim.SGD(groups,momentum=0.9)
    for ep in range(3):
        perm=torch.randperm(X.shape[0])
        for i in range(0,X.shape[0],256):
            idx=perm[i:i+256]; opt.zero_grad(); F.cross_entropy(m(X[idx].to(DEVICE)),Y[idx].to(DEVICE)).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    m.eval(); sd={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    torch.save({"sd":sd,"acc":0.0,"dF":None,"wmove":0.0},os.path.join(ckpt_dir(),f"{regime}_{act}_w{w}_s{seed}.pt"))

# ==================================================================== MAIN (chay HET cell -> gop -> {mode}_final.csv)
def _fit_alpha(width,val):
    w=np.asarray(width,float); y=np.asarray(val,float); ok=(w>0)&(y>0)&np.isfinite(w)&np.isfinite(y)
    if ok.sum()<2: return (float("nan"),float("nan"))
    lw,ly=np.log(w[ok]),np.log(y[ok]); b,a=np.polyfit(lw,ly,1); yh=a+b*lw
    r2=1-((ly-yh)**2).sum()/max(((ly-ly.mean())**2).sum(),1e-30); return (round(-b,4),round(float(r2),4))

def run_geo():
    out=os.path.join(OUT_DIR, f"param_geo_{MODE}_SMOKE.csv" if SMOKE else f"param_geo_{MODE}.csv")
    lam_rels=LAM_SWEEP if LAM_SWEEP else [LAM_REL]
    X=load_data(); Xgeo=X[:GEO_BATCH].to(DEVICE)
    plan=[SHARD_PLAN[int(ONLY_SHARD)]] if ONLY_SHARD is not None else list(SHARD_PLAN.values())
    cells=[]
    for regime,acts_all in plan:
        for act in acts_all:
            if act in GEO_ACTS and (regime,act) not in cells: cells.append((regime,act))
    log(f"GEO cells={cells} widths={GEO_WIDTHS} pairs={GEO_PAIRS} batch={GEO_BATCH} lam_rels={lam_rels} -> {out}")
    for (regime,act) in cells:
        for w in GEO_WIDTHS:
            try:
                # --- nap NSEEDS ckpt ---
                sds=[]; accs=[]
                for s in range(NSEEDS):
                    cp=find_ckpt(regime,act,w,s)
                    if cp is None and SMOKE: _smoke_train(regime,act,w,s); cp=find_ckpt(regime,act,w,s)
                    if cp is None: continue
                    sd,acc,_=load_sd(cp); sds.append(sd); accs.append(acc)
                if len(sds)<2:
                    log(f"  [{act}/w{w}] < 2 ckpt (thay {len(sds)}) -> bo"); 
                    write_row(out,dict(kind="geo",mode=MODE,shard="",regime=regime,act=act,width=w,status="skip:nockpt")); continue
                log(f"=== {act}/w{w}  ({len(sds)} nets) ===")
                ag,gs=perm_spec(build_net(w,act,regime))
                ref=build_net(w,act,regime).to(DEVICE).eval(); p_ref,b_ref=_pb(ref); pkeys=set(p_ref.keys())
                def to_params(sd):   # chi lay PARAM keys (khong buffer), theo thu tu named_parameters
                    return {k:sd[k].to(DEVICE) for k in p_ref.keys()}

                pairs=list(itertools.combinations(range(len(sds)),2))[:GEO_PAIRS]
                for (i,j) in pairs:
                    if RESUME and all(already_done(out,regime,act,w,i,j,lr_) for lr_ in lam_rels):
                        log(f"  [pair {i}-{j}] tat ca lam_rel da co -> skip"); continue
                    # ---- phan dung chung (khong phu thuoc lambda): align, delta, lmax, do dai Fisher ----
                    try:
                        perms=weight_matching(ag,gs,sds[i],sds[j],iters=8,seed=i*13+j)
                        sdB=apply_perm(sds[j],ag,perms)
                        pA=to_params(sds[i]); pB=to_params(sdB)
                        delta={k:(pB[k]-pA[k]) for k in pA}; dn=_vnorm(delta); d2=max(dn*dn,1e-30)
                        lmax=lam_max(ref,pA,b_ref,Xgeo,GEO_MICRO,POWER_ITERS,seed=17)
                        # DO DAI FISHER (§5.2): 1/2 delta^T F delta va Rayleigh delta_hat^T F delta_hat, tai A/mid/B
                        def _flen(pt):
                            q=_vdot(delta, fisher_vp(ref,pt,b_ref,Xgeo,delta,GEO_MICRO))  # = delta^T F(pt) delta >=0
                            return 0.5*q, q/d2
                        pmid={k:0.5*(pA[k]+pB[k]) for k in pA}
                        flA,rqA=_flen(pA); flM,rqM=_flen(pmid); flB,rqB=_flen(pB)
                    except Exception as e:
                        traceback.print_exc()
                        write_row(out,dict(kind="geo",mode=MODE,shard="",regime=regime,act=act,width=w,seedA=i,seedB=j,status="error:"+repr(e)[:40])); continue
                    for lr_ in lam_rels:
                        if RESUME and already_done(out,regime,act,w,i,j,lr_):
                            log(f"  [pair {i}-{j} lam_rel={lr_}] da co -> skip"); continue
                        try:
                            lam=max(lr_*lmax,1e-12)
                            # Gamma(dd) doc luoi t, warm-start CG
                            ts=list(np.linspace(0,1,GEO_TGRID))
                            gammas=[]; resids=[]; fdis=[]; x0=None
                            for tt in ts:
                                pt={k:(1-tt)*pA[k]+tt*pB[k] for k in pA}
                                g,res,fdi=christoffel_dd(ref,pt,b_ref,Xgeo,delta,lam,GEO_MICRO,x0=x0)
                                x0=g                                   # warm-start CG (GPU)
                                gammas.append({k:v.detach().cpu() for k,v in g.items()})   # luu CPU: tranh OOM o width lon
                                resids.append(res); fdis.append(fdi)
                            mid=int(np.argmin([abs(t-0.5) for t in ts])); gamma_mid=_vnorm(gammas[mid])
                            # tich phan Green tren CPU: xi(t)=int G(t,s) Gamma(s) ds -> sup_t ||xi||
                            G=green_matrix(ts); dt=ts[1]-ts[0]
                            keys=list(delta.keys()); n=len(ts)
                            sup=0.0
                            for ti in range(n):
                                xi=None
                                for si in range(n):
                                    w_=float(G[ti,si])*dt
                                    if w_==0.0: continue
                                    xi={k:(gammas[si][k]*w_ if xi is None else xi[k]+gammas[si][k]*w_) for k in keys}
                                sup=max(sup,_vnorm(xi) if xi is not None else 0.0)
                            dev_rel=sup/max(dn,1e-30)
                            write_row(out,dict(kind="geo",mode=MODE,shard="",regime=regime,act=act,width=w,
                                seedA=i,seedB=j,dnorm=round(dn,4),dev_geo=f"{sup:.6e}",dev_rel=f"{dev_rel:.6e}",
                                gamma_mid=f"{gamma_mid:.6e}",
                                flen_A=f"{flA:.6e}",flen_mid=f"{flM:.6e}",flen_B=f"{flB:.6e}",
                                rq_A=f"{rqA:.6e}",rq_mid=f"{rqM:.6e}",rq_B=f"{rqB:.6e}",
                                lam=f"{lam:.4e}",lam_rel=lr_,cg_resid=f"{max(resids):.2e}",
                                fd_instab=f"{max(fdis):.3e}",
                                accA=accs[i],accB=accs[j],status="ok"))
                            log(f"  [pair {i}-{j} lam_rel={lr_}] ||d||={dn:.3g} dev_rel={dev_rel:.3e} flen_mid={flM:.3e} rq_mid={rqM:.3e} cgres<={max(resids):.1e}")
                        except Exception as e:
                            traceback.print_exc()
                            write_row(out,dict(kind="geo",mode=MODE,shard="",regime=regime,act=act,width=w,seedA=i,seedB=j,lam_rel=lr_,status="error:"+repr(e)[:40]))
                        finally:
                            try: del gammas, x0
                            except Exception: pass
                            if DEVICE=="cuda": torch.cuda.empty_cache()
                try: del sds, ref, p_ref, b_ref, pkeys
                except Exception: pass
                import gc; gc.collect()
                if DEVICE=="cuda": torch.cuda.empty_cache(); torch.cuda.synchronize()
            except Exception as e:
                traceback.print_exc(); write_row(out,dict(kind="geo",mode=MODE,shard="",act=act,width=w,status="cell-error:"+repr(e)[:40]))
    log("DONE geo", MODE)
    return out

def build_final(geo_csv):
    import pandas as pd
    def _col(df,cands):
        low={c.lower():c for c in df.columns}
        for k in cands:
            if k in low: return low[k]
        return None
    comb=None
    if COMBINED and os.path.exists(COMBINED):
        c=pd.read_csv(COMBINED)
        if _col(c,["kind"]): c=c[c[_col(c,["kind"])].astype(str)=="net"].copy()
        ren={_col(c,["regime"]):"regime",_col(c,["act","activation"]):"act",_col(c,["width","n"]):"width",
             _col(c,["df_op","df"]):"dF",_col(c,["acc"]):"acc"}
        ren={k:v for k,v in ren.items() if k}
        comb=c.rename(columns=ren)
        for cc in ["width","dF","acc"]:
            if cc in comb: comb[cc]=pd.to_numeric(comb[cc],errors="coerce")
    g=pd.read_csv(geo_csv)
    if "status" in g: g=g[g["status"].astype(str)=="ok"].copy()
    for cc in ["width","dev_rel","dev_geo","dnorm","gamma_mid","flen_mid","rq_mid","fd_instab","lam_rel"]:
        if cc in g: g[cc]=pd.to_numeric(g[cc],errors="coerce")
    if set(["dev_geo","dnorm"]).issubset(g.columns): g["dev_rel2"]=g["dev_geo"]/g["dnorm"]**2
    # neu co sweep lambda: bao cao o lam_rel=1e-2 (chinh); giu du lieu sweep o param_geo_*.csv
    if "lam_rel" in g and g["lam_rel"].notna().any():
        gg2=g[np.isclose(g["lam_rel"],1e-2)]
        if len(gg2)>0: g=gg2
    def mi(s):
        s=pd.to_numeric(s,errors="coerce").dropna()
        return (float(s.median()),float(s.quantile(.25)),float(s.quantile(.75)),int(len(s))) if len(s) else (np.nan,np.nan,np.nan,0)
    cells=set()
    for src in [comb,g]:
        if src is not None and set(["regime","act","width"]).issubset(src.columns):
            cells|=set(map(tuple, src[["regime","act","width"]].dropna().values))
    rows=[]
    for (r,a,w) in sorted(cells,key=lambda x:(str(x[0]),str(x[1]),float(x[2]))):
        rec=dict(mode=MODE,regime=r,act=a,width=int(float(w)))
        if comb is not None:
            cc=comb[(comb.regime==r)&(comb.act==a)&(comb.width==float(w))]
            m,q1,q3,n=mi(cc["dF"]) if "dF" in cc else (np.nan,np.nan,np.nan,0)
            rec.update(dF_med=m,dF_q1=q1,dF_q3=q3,n_seeds=n,acc_med=mi(cc["acc"])[0] if "acc" in cc else np.nan)
        gc=g[(g.regime==r)&(g.act==a)&(g.width==float(w))]
        dm,dq1,dq3,ng=mi(gc["dev_rel"]) if "dev_rel" in gc else (np.nan,np.nan,np.nan,0)
        fm,fq1,fq3,_=mi(gc["flen_mid"]) if "flen_mid" in gc else (np.nan,np.nan,np.nan,0)
        rec.update(n_pairs=ng,dev_rel_med=dm,dev_rel_q1=dq1,dev_rel_q3=dq3,
            dev_rel2_med=mi(gc["dev_rel2"])[0] if "dev_rel2" in gc else np.nan,
            gamma_mid_med=mi(gc["gamma_mid"])[0] if "gamma_mid" in gc else np.nan,
            flen_mid_med=fm,flen_mid_q1=fq1,flen_mid_q3=fq3,
            rq_mid_med=mi(gc["rq_mid"])[0] if "rq_mid" in gc else np.nan,
            fd_instab_med=mi(gc["fd_instab"])[0] if "fd_instab" in gc else np.nan)
        rows.append(rec)
    T=pd.DataFrame(rows)
    def add_alpha(colmed,name):
        if colmed not in T or T[colmed].notna().sum()==0: T[name]=np.nan; T[name+"_r2"]=np.nan; return
        for (r,a),gr in T.groupby(["regime","act"]):
            al,r2=_fit_alpha(gr["width"].values,gr[colmed].values)
            T.loc[(T.regime==r)&(T.act==a),name]=al; T.loc[(T.regime==r)&(T.act==a),name+"_r2"]=r2
    add_alpha("dF_med","alpha_dF"); add_alpha("dev_rel2_med","alpha_devrel2"); add_alpha("flen_mid_med","alpha_flen")
    order=["mode","regime","act","width","n_seeds","n_pairs","acc_med","dF_med","dF_q1","dF_q3",
           "dev_rel_med","dev_rel_q1","dev_rel_q3","dev_rel2_med","gamma_mid_med",
           "flen_mid_med","flen_mid_q1","flen_mid_q3","rq_mid_med","fd_instab_med",
           "alpha_dF","alpha_dF_r2","alpha_devrel2","alpha_devrel2_r2","alpha_flen","alpha_flen_r2"]
    for c in order:
        if c not in T: T[c]=np.nan
    T=T[order].sort_values(["regime","act","width"]).reset_index(drop=True)
    fin=os.path.join(OUT_DIR,f"{MODE}_final.csv"); T.to_csv(fin,index=False)
    hv = "dev_rel_med" in T and T["dev_rel_med"].notna().any()
    log(f"-> {fin}  ({len(T)} o)  shape/length={'CO' if hv else 'NaN(chua co geo)'}  dF={'CO' if comb is not None else 'thieu'}")

def _seed_resume():
    # Ke thua tien do tu lan chay truoc: neu OUT_DIR chua co param_geo_{MODE}.csv,
    # tim ban cu trong input (vd dataset da add) va copy vao de RESUME bo qua cell da xong.
    import shutil
    dst=os.path.join(OUT_DIR, f"param_geo_{MODE}.csv")
    if os.path.exists(dst): 
        n=sum(1 for _ in open(dst))-1; log(f"[resume] da co {dst} ({n} dong) -> tinh tiep"); return
    old=_first([f"param_geo_{MODE}.csv",
                f"/kaggle/input/**/param_geo_{MODE}.csv",
                f"/content/drive/MyDrive/**/param_geo_{MODE}.csv"])
    if old and os.path.abspath(old)!=os.path.abspath(dst):
        shutil.copy(old,dst); n=sum(1 for _ in open(dst))-1
        log(f"[resume] nap tien do cu: {old} -> {dst} ({n} dong da xong)")
    else:
        log("[resume] chua co tien do cu -> tinh tu dau")

def main():
    log(f"=== run_geo[{MODE}] DEVICE={DEVICE} SMOKE={SMOKE} ONLY_SHARD={ONLY_SHARD} ===")
    log(f"  ckpt_roots={CKPT_ROOTS}")
    log(f"  combined  ={COMBINED}")
    self_test()
    _seed_resume()
    geo=run_geo()
    try: build_final(geo)
    except Exception as e:
        traceback.print_exc(); log("!! gop loi (param_geo_*.csv van con):", repr(e)[:100])

if __name__=="__main__": main()
