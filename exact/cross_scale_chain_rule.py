# M/M/1 whose service rate is modulated by a fast 2-state process z (e.g. CPU throttled / not throttled).
# Fast control: theta = ln a (rate of switching into the fast state). Test cross-scale chain rule:
#   d<n>/dtheta  ~=  [slow static FDT: -Var(n)/mu_eff] x [fast static FDT: Cov_fast(mu(z), z)]   as gamma -> inf
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
lam, mu0, mu1, a, b, K = 1.0, 0.8, 2.4, 1.0, 1.0, 400
def stat(g, a_):
    S=2*(K+1); idx=lambda n,z: 2*n+z; r,c,v=[],[],[]
    for n in range(K+1):
        for z in (0,1):
            i=idx(n,z)
            if n<K: r+=[i]; c+=[idx(n+1,z)]; v+=[lam]
            if n>0: r+=[i]; c+=[idx(n-1,z)]; v+=[mu1 if z else mu0]
            r+=[i]; c+=[idx(n,1-z)]; v+=[g*(a_ if z==0 else b)]
    M=sp.csr_matrix((v,(r,c)),shape=(S,S)); Q=(M-sp.diags(np.asarray(M.sum(1)).ravel())).T.tolil()
    Q[0,:]=1; rhs=np.zeros(S); rhs[0]=1; p=sla.spsolve(Q.tocsc(),rhs); return p/p.sum()
n=np.repeat(np.arange(K+1),2).astype(float)
p1=a/(a+b); mueff=mu0*(1-p1)+mu1*p1; rho=lam/mueff; varn=rho/(1-rho)**2
pred=-varn/mueff*(mu1-mu0)*p1*(1-p1)
print(f"chain-rule prediction d<n>/dln(a) = {pred:.5f}   (mu_eff={mueff}, rho={rho:.3f})")
h=1e-5
for g in [0.3,1,10,100,1000]:
    ex=(stat(g,a*np.exp(h))@n-stat(g,a*np.exp(-h))@n)/(2*h)
    print(f"gamma={g:>6}: exact={ex:.5f}  rel.err={abs(ex-pred)/abs(pred):.2e}")
