"""Exact reference quantities for the finite-sample study (models J and B of the WP)."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
from scipy.sparse.linalg import expm_multiply
lam, mu1, mu2, K1 = 1.0, 2.0, 1.6, 70

def build(l, m1, m2, K2):
    idx = lambda a, b: a * (K2 + 1) + b
    S = (K1 + 1) * (K2 + 1); r, c, v = [], [], []
    for a in range(K1 + 1):
        for b in range(K2 + 1):
            i = idx(a, b)
            if a < K1:           r.append(i); c.append(idx(a + 1, b));     v.append(l)
            if a > 0 and b < K2: r.append(i); c.append(idx(a - 1, b + 1)); v.append(m1)
            if b > 0:            r.append(i); c.append(idx(a, b - 1));     v.append(m2)
    M = sp.csr_matrix((v, (r, c)), shape=(S, S))
    Q = (M - sp.diags(np.asarray(M.sum(1)).ravel())).tocsc()
    n1 = np.repeat(np.arange(K1 + 1), K2 + 1).astype(float); n2 = np.tile(np.arange(K2 + 1), K1 + 1).astype(float)
    return Q, n1, n2

def stationary(Q):
    A = Q.T.tolil(); A[0, :] = 1; b = np.zeros(Q.shape[0]); b[0] = 1
    p = sla.spsolve(A.tocsc(), b); return p / p.sum()

def reference(K2, tgrid, h=1e-5):
    Q, n1, n2 = build(lam, mu1, mu2, K2); pi = stationary(Q); N = n1 + n2
    Qp = build(lam*np.exp(h), mu1, mu2, K2)[0]; Qm = build(lam*np.exp(-h), mu1, mu2, K2)[0]
    traj = lambda Q_: expm_multiply(Q_.T, pi, start=tgrid[0], stop=tgrid[-1], num=len(tgrid), endpoint=True) @ N
    R_exact = (traj(Qp) - traj(Qm)) / (2*h)                      # step response of <N> to d ln(lambda)
    C = expm_multiply(Q.T, pi*N, start=tgrid[0], stop=tgrid[-1], num=len(tgrid), endpoint=True) @ N - (pi@N)**2
    R_fdt = C[0] - C                                              # FDT prediction with B = N (exact data)
    # integrated autocorrelation time of N:  tau_int = <dN, (-L)^{-1} dN>_pi / Var(N)
    dN = N - pi @ N
    A = (-Q).tolil(); A[0, :] = pi; rhs = dN.copy(); rhs[0] = 0.0     # solve -Q g = dN with pi.g = 0
    g = sla.spsolve(A.tocsc(), rhs)
    tau_int = (pi * dN) @ g / (pi @ dN**2)
    # static susceptibility d<N>/d ln lambda (exact) = R_exact(inf)
    sp_ = stationary(Qp) @ N; sm = stationary(Qm) @ N; chi = (sp_ - sm) / (2*h)
    return dict(Q=Q, pi=pi, N=N, n1=n1, n2=n2, R_exact=R_exact, R_fdt=R_fdt, C=C, tau_int=tau_int,
                chi=chi, varN=pi@dN**2, meanN=pi@N)

def cross_asym(ref, lags):
    """exact E[u(N0) v(Nt)] - E[v(N0) u(Nt)] with u = standardized N, v = standardized N^2"""
    Q, pi, N = ref['Q'], ref['pi'], ref['N']
    u = (N - pi@N) / np.sqrt(pi@(N - pi@N)**2); w = N**2; v = (w - pi@w) / np.sqrt(pi@(w - pi@w)**2)
    out = []
    for t in lags:
        out.append(expm_multiply(Q.T*t, pi*u) @ v - expm_multiply(Q.T*t, pi*v) @ u)
    return np.array(out)

if __name__ == "__main__":
    tg = np.linspace(0, 40, 161)
    for K2, lab in [(K1, "J"), (3, "B")]:
        r = reference(K2, tg)
        print(f"{lab}: E[N]={r['meanN']:.4f} Var(N)={r['varN']:.4f} chi=dE[N]/dln(lam)={r['chi']:.4f} tau_int(N)={r['tau_int']:.3f}")
        print(f"   R_exact(1,4)={r['R_exact'][4]:.3f},{r['R_exact'][16]:.3f}  R_fdt(1,4)={r['R_fdt'][4]:.3f},{r['R_fdt'][16]:.3f}"
              f"  max|R_fdt-R_exact|/chi over t<=40: {np.abs(r['R_fdt']-r['R_exact']).max()/r['chi']:.3f}")
        lags = [0.5, 1, 2, 4, 8, 16]
        print("   cross-asym (std N, std N^2) at lags", lags, np.round(cross_asym(r, lags), 4))
