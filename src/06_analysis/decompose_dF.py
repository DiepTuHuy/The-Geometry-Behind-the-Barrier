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
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.func import functional_call, jvp as _fjvp, vjp as _fvjp, jacrev as _jacrev, grad as _grad
def _tv():   # torchvision chi can cho mlp/cnn (ts la synthetic)
    import torchvision; return torchvision

# ==================================================================== CONFIG (DECOMPOSE dF = GN + transport)
MODE       = os.environ.get("GEO_MODE","mlp")
RUN_TAG    = {"mlp":"pmlp_v2","cnn":"pcnn_v2","ts":"pts_v2"}[MODE]
def _first(pats):
    for p in pats:
        if not p: continue
        h=sorted(glob.glob(p,recursive=True))
        if h: return h[0]
    return None
CKPT_ROOTS=[".","/kaggle/input","/content","/content/drive/MyDrive"]
OUT_DIR="."
WIDTHS={"mlp":[64,128,256,512,1024,2048,4096],"cnn":[1,2,4,8],"ts":[64,128,256,512,1024,2048,4096]}[MODE]
ACTS=["gelu","tanh"]              # hai ham trom dai dien (them swish/softplus neu muon)
REGIMES=["ntk","sp","mup"]
NSEEDS=3                          # 3 seed/o du de doc slope
BATCH=1024; MICRO=64; FD_EPS=3e-3
NZ=3; PI_ITERS=8                  # so huong z + so vong power-iteration cho op-norm
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DIN,K={"mlp":(784,10),"cnn":(None,10),"ts":(64,10)}[MODE]
# cac hyperparam geo khong dung o day nhung 1 so ham middle tham chieu:
GEO_MICRO=MICRO; FD_RICH=True; LAM_REL=1e-2; CG_ITERS=1; CG_TOL=1e-6; POWER_ITERS=1; GEO_BATCH=BATCH
GEO_WIDTHS=WIDTHS; GEO_ACTS=ACTS; SHARD_ID=""; RESUME=True; SMOKE=False; LAM_SWEEP=None
SHARD_PLAN={0:("ntk",["relu","gelu","tanh"])}
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

# ==================================================================== MAIN (decompose)
def fisher_vp_frozen(m,p,b,x,v,micro,pr0):
    """F~ = E J^T S0 J : S0 dong bang (pr0 precompute tai w0)."""
    B=x.shape[0]; acc=None
    for i in range(0,B,micro):
        xb=x[i:i+micro]; prb=pr0[i:i+micro]
        def f(pp): return _call(m,pp,b,xb)
        logits,Jv=_fjvp(f,(p,),(v,))
        s=prb*Jv-prb*(prb*Jv).sum(1,keepdim=True)
        JTs=_fvjp(f,p)[1](s)[0]
        acc={k:JTs[k].detach() for k in JTs} if acc is None else {k:acc[k]+JTs[k].detach() for k in acc}
    return {k:acc[k]/B for k in acc}

def _dFz_apply(m,p,b,x,z,v,micro,pr0):
    """(d_z F)v hoac (d_z F~)v qua sai phan trung tam (pr0=None: F that)."""
    if pr0 is None:
        Fp=fisher_vp(m,_vaxpy(p,FD_EPS,z),b,x,v,micro); Fm=fisher_vp(m,_vaxpy(p,-FD_EPS,z),b,x,v,micro)
    else:
        Fp=fisher_vp_frozen(m,_vaxpy(p,FD_EPS,z),b,x,v,micro,pr0); Fm=fisher_vp_frozen(m,_vaxpy(p,-FD_EPS,z),b,x,v,micro,pr0)
    return {k:(Fp[k]-Fm[k])/(2*FD_EPS) for k in p}

def opnorm_dz(apply_fn, p, seed):
    """||A||_op cho A doi xung (A=d_z F...) qua power-iteration."""
    g=torch.Generator(device=DEVICE).manual_seed(seed)
    v={k:torch.randn(x_.shape,generator=g,device=DEVICE,dtype=x_.dtype) for k,x_ in p.items()}
    v=_vscale(v,1.0/max(_vnorm(v),1e-30)); lam=0.0
    for _ in range(PI_ITERS):
        Av=apply_fn(v); lam=_vnorm(Av)
        if lam<1e-30: break
        v=_vscale(Av,1.0/lam)
    return lam

def rho_S(m,p,b,x,micro):
    """E_x ||S(p_w(x))||_op ; S(p)=diag(p)-pp^T (K x K)."""
    B=x.shape[0]; tot=0.0
    for i in range(0,B,micro):
        pr=torch.softmax(_call(m,p,b,x[i:i+micro]),1)
        S=torch.diag_embed(pr)-pr.unsqueeze(2)*pr.unsqueeze(1)
        tot+=torch.linalg.matrix_norm(S,ord=2).sum().item()
    return tot/B

CSVD=["mode","regime","act","width","seed","dF_op","gn_op","transport_op","rho_S","tr_frac"]
def wrow(path,r):
    new=not os.path.exists(path)
    with open(path,"a") as f:
        if new: f.write(",".join(CSVD)+"\n")
        f.write(",".join(str(r.get(c,"")) for c in CSVD)+"\n"); f.flush()

def self_test_decomp():
    log("  [self-test] tach dF=dF~+transport tren model nho...")
    old=torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    torch.manual_seed(0); mm=NetMLP(6,"tanh","ntk",din=4,k=3).eval(); xx=torch.randn(10,4)
    pp,bb=_pb(mm); pr0=torch.softmax(_call(mm,pp,bb,xx),1).detach()
    torch.manual_seed(1); z={k:torch.randn_like(pp[k]) for k in pp}; z=_vscale(z,1/_vnorm(z))
    torch.manual_seed(2); v={k:torch.randn_like(pp[k]) for k in pp}; v=_vscale(v,1/_vnorm(v))
    dF =_dFz_apply(mm,pp,bb,xx,z,v,5,None)
    dFt=_dFz_apply(mm,pp,bb,xx,z,v,5,pr0)
    tr ={k:dF[k]-dFt[k] for k in pp}
    rec={k:dFt[k]+tr[k] for k in pp}
    err=_vnorm({k:rec[k]-dF[k] for k in pp})/max(_vnorm(dF),1e-30)
    assert err<1e-9, f"tach sai {err:.2e}"
    log(f"  [self-test] OK  ||dF-(dF~+transport)||/||dF|| = {err:.1e}")
    torch.set_default_dtype(old)

def _slope(w,y):
    import numpy as _np
    w=_np.asarray(w,float); y=_np.asarray(y,float); ok=(w>0)&(y>0)&_np.isfinite(y)
    return float(_np.polyfit(_np.log(w[ok]),_np.log(y[ok]),1)[0]) if ok.sum()>=3 else float("nan")

def main():
    log(f"=== DECOMPOSE dF [{MODE}] DEVICE={DEVICE} batch={BATCH} nz={NZ} pi={PI_ITERS} ===")
    self_test_decomp()
    X=load_data(); Xb=X[:BATCH].to(DEVICE)
    out=os.path.join(OUT_DIR,f"decompose_{MODE}.csv")
    agg={}  # (regime,act) -> {width:[gn],..}
    for regime in REGIMES:
        for act in ACTS:
            for w in WIDTHS:
                for s in range(NSEEDS):
                    cp=find_ckpt(regime,act,w,s)
                    if cp is None: continue
                    sd,_,_=load_sd(cp); ref=build_net(w,act,regime).to(DEVICE).eval()
                    p_ref,b_ref=_pb(ref); p={k:sd[k].to(DEVICE) for k in p_ref.keys()}
                    # frozen softmax tai chinh checkpoint
                    pr0=[]
                    with torch.no_grad():
                        for i in range(0,Xb.shape[0],MICRO):
                            pr0.append(torch.softmax(_call(ref,p,b_ref,Xb[i:i+MICRO]),1))
                    pr0=torch.cat(pr0,0)
                    # op-norm cho tung phan, max tren NZ huong z
                    gn=dt=df=0.0
                    for zi in range(NZ):
                        g=torch.Generator(device=DEVICE).manual_seed(1000+zi)
                        z={k:torch.randn(v.shape,generator=g,device=DEVICE,dtype=v.dtype) for k,v in p.items()}
                        z=_vscale(z,1/max(_vnorm(z),1e-30))
                        df=max(df,opnorm_dz(lambda v: _dFz_apply(ref,p,b_ref,Xb,z,v,MICRO,None), p, 7+zi))
                        gn=max(gn,opnorm_dz(lambda v: _dFz_apply(ref,p,b_ref,Xb,z,v,MICRO,pr0), p, 7+zi))
                        dt=max(dt,opnorm_dz(lambda v: {k:_dFz_apply(ref,p,b_ref,Xb,z,v,MICRO,None)[k]-_dFz_apply(ref,p,b_ref,Xb,z,v,MICRO,pr0)[k] for k in p}, p, 7+zi))
                    rs=rho_S(ref,p,b_ref,Xb,MICRO)
                    wrow(out,dict(mode=MODE,regime=regime,act=act,width=w,seed=s,
                        dF_op=f"{df:.6e}",gn_op=f"{gn:.6e}",transport_op=f"{dt:.6e}",
                        rho_S=f"{rs:.6e}",tr_frac=f"{dt/max(df,1e-30):.4f}"))
                    agg.setdefault((regime,act),{}).setdefault(w,{"gn":[],"dt":[],"df":[],"rs":[]})
                    for kk,vv in [("gn",gn),("dt",dt),("df",df),("rs",rs)]: agg[(regime,act)][w][kk].append(vv)
                    log(f"  {regime}/{act}/w{w}/s{s}: dF={df:.2e} GN={gn:.2e} transport={dt:.2e} rho_S={rs:.3f} (tr={dt/max(df,1e-30):.0%})")
                    del ref
                    if DEVICE=="cuda": torch.cuda.empty_cache()
    # ===== PHAN TICH (mo ta trung thuc, khong ep gia thuyet) =====
    import numpy as np
    log("\n================ PHAN TICH (slope log-log theo width) ================")
    log(f"{'cell':16s} {'dF~n^':>7s} {'GN~n^':>7s} {'trans~n^':>8s} {'rhoS~n^':>8s} {'tro-luc':>8s}  dien giai")
    for (regime,act),d in sorted(agg.items()):
        ws=sorted(d)
        if len(ws)<3: continue
        med=lambda kk:[float(np.median(d[w][kk])) for w in ws]
        sdF,sGN,sTR,sRS=_slope(ws,med("df")),_slope(ws,med("gn")),_slope(ws,med("dt")),_slope(ws,med("rs"))
        gnL,dtL=med("gn"),med("dt")
        dom="GN" if gnL[-1]>=dtL[-1] else "transp"
        parts=["S phang" if abs(sRS)<0.15 else f"S doi(n^{sRS:.2f})"]
        parts.append("GN xuong" if sGN<-0.15 else "GN dung")
        parts.append("transp PHANG" if abs(sTR)<0.2 else ("transp xuong" if sTR<-0.15 else "transp tang"))
        log(f"{regime}/{act:9s} {sdF:7.2f} {sGN:7.2f} {sTR:8.2f} {sRS:8.2f} {dom:>8s}  dF xuong do {dom}; "+"; ".join(parts))
    log("\nDoc: rhoS phang => S mu-width (dung truc giac). GN xuong ~n^-0.5 & transport PHANG")
    log("     => Gauss-Newton (linear-hoa, Phu luc A) la thu pham. Neu transport CUNG xuong => ca hai co")
    log("     vi transport chua J (khong chi S) nen cung linear-hoa. So mu thuc quyet dinh, khong phai gia thuyet.")
    log(f"\n-> {out}")

if __name__=="__main__": main()
