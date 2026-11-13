"""Exact CTMC checks for top-level operational equilibrium (TLOE) on a two-stage tandem.
Model J : open tandem M/M/1 -> M/M/1 (Jackson, product form).
Model B : same, but stage 2 has a finite buffer K2 = 3 (blocking: 1 -> 2 moves wait when stage 2 is full).
Top-level observable Y = N = n1 + n2 ; control theta = ln(lambda).
Requires: python3, numpy, scipy.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
from scipy.sparse.linalg import expm_multiply

lam, mu1, mu2, K1 = 1.0, 2.0, 1.6, 70
T, NT, h = 8.0, 9, 1e-5

def build(l, m1, m2, K2):
    idx = lambda a, b: a * (K2 + 1) + b
    S = (K1 + 1) * (K2 + 1); r, c, v = [], [], []
    for a in range(K1 + 1):
        for b in range(K2 + 1):
            i = idx(a, b)
            if a < K1:          r.append(i); c.append(idx(a + 1, b));     v.append(l)
            if a > 0 and b < K2: r.append(i); c.append(idx(a - 1, b + 1)); v.append(m1)
            if b > 0:           r.append(i); c.append(idx(a, b - 1));     v.append(m2)
    M = sp.csr_matrix((v, (r, c)), shape=(S, S))
    Q = (M - sp.diags(np.asarray(M.sum(1)).ravel())).tocsc()
    n1 = np.repeat(np.arange(K1 + 1), K2 + 1).astype(float)
    n2 = np.tile(np.arange(K2 + 1), K1 + 1).astype(float)
    return Q, n1, n2

def stationary(Q):
    A = Q.T.tolil(); A[0, :] = 1; b = np.zeros(Q.shape[0]); b[0] = 1
    p = sla.spsolve(A.tocsc(), b); return p / p.sum()

import numpy as np
rng=np.random.default_rng(0)
def model(m1,m2,K2=K1,KK=None):
    Q,n1,n2=build(lam,m1,m2,K2); pi=stationary(Q); return Q,pi,n1,n2
# 1) three-time reversibility of N, with positive control on full-state functions
Q,pi,n1,n2=model(mu1,mu2); N=n1+n2; Ni=N.astype(int)
def three(f,g,hh,t):   # E[f(X0) g(Xt) h(X2t)]
    v=expm_multiply(Q*t, hh)          # E[h(X_{2t})|X_t]
    v=expm_multiply(Q*t, g*v)         # E[g(X_t)h(X_2t)|X_0]
    return pi@(f*v)
mx=0; mxc=0
for _ in range(8):
    a,b,c=[rng.normal(size=200) for _ in range(3)]
    f,g,hh=a[Ni],b[Ni],c[Ni]
    mx=max(mx,abs(three(f,g,hh,0.7)-three(hh,g,f,0.7)))
    F,G,H=[np.tanh(rng.normal(size=n1.size)) for _ in range(3)]
    mxc=max(mxc,abs(three(F,G,H,0.7)-three(H,G,F,0.7)))
print(f"[3-time] N: max asym={mx:.1e} | positive control (full-state fns): {mxc:.2e}")
# 2) stage-permutation invariance of the N process (two-time joint of N)
def jointN(m1,m2,t):
    Q,pi,n1,n2=model(m1,m2); N=(n1+n2).astype(int); M=30
    D=np.array([(N==k).astype(float) for k in range(M)])
    return np.array([expm_multiply(Q.T*t, pi*d) for d in D])@D.T
for t in (0.5,2.0):
    print(f"[perm] t={t}: max|J(2,1.6)-J(1.6,2)| = {np.abs(jointN(2.0,1.6,t)-jointN(1.6,2.0,t)).max():.1e}")
# 3) (R) does not imply (F): Phi=n1, control ln lambda
B=N-pi@N; n1i=n1.astype(int)
m=np.bincount(n1i,weights=pi*B)/np.bincount(n1i,weights=pi)
print(f"[R!=>F] Phi=n1: unexplained Var fraction = {(pi@(B-m[n1i])**2)/(pi@B**2):.3f}; "
      f"n1 two-time asym = {abs((expm_multiply(Q.T*1.0,pi*n1**2)@n1)-(expm_multiply(Q.T*1.0,pi*n1)@n1**2)):.1e}")
print(f"[static FDT] Var(N) = {pi@N**2-(pi@N)**2:.10f}")
# 4) small-delta KL law, blocking model, on N marginal
Qb,pib,a1,a2=model(mu1,mu2,3); Nb=(a1+a2); Nbi=Nb.astype(int)
marg=lambda p: np.bincount(Nbi,weights=p)
p0=marg(pib); k=np.arange(len(p0))
h=1e-5; score=(np.log(marg(stationary(build(lam*np.exp(h),mu1,mu2,3)[0])))-np.log(marg(stationary(build(lam*np.exp(-h),mu1,mu2,3)[0]))))/(2*h)
mis=score-k; halfvar=0.5*(p0@mis**2-(p0@mis)**2)
for d in (1e-3,1e-2,5e-2,1e-1):
    po=marg(stationary(build(lam*np.exp(d),mu1,mu2,3)[0])); pe=p0*np.exp(d*k); pe/=pe.sum()
    print(f"[KL] d={d:g}: KL/d^2={np.sum(po*np.log(po/pe))/d**2:.4f}  (1/2 Var misprediction = {halfvar:.4f})")
# 5) truncation adequacy, blocking model
print(f"[trunc] blocking: E[n1]={pib@a1:.3f}, P(n1=K1)={pib[a1==K1].sum():.1e}")
