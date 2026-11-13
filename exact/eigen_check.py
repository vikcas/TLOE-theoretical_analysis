"""Generalised eigenvalues of K(tau) v = kappa K(0) v (indicator dictionary on N, tau = 0.5), models J and B.
Shows that they stay real in model B although K(tau) is asymmetric (report §5.3)."""
import numpy as np, scipy.linalg as sl
from scipy.sparse.linalg import expm_multiply
from exact import build, stationary, lam, mu1, mu2
for K2, lab in [(70, 'J'), (3, 'B')]:
    Q, n1, n2 = build(lam, mu1, mu2, K2); pi = stationary(Q); Ni = (n1 + n2).astype(int)
    D = np.array([(Ni == k).astype(float) for k in range(12)] + [(Ni >= 12).astype(float)])
    K0 = (D * pi) @ D.T; Kt = np.array([expm_multiply(Q.T * 0.5, pi * d) for d in D]) @ D.T
    ev = sl.eigvals(Kt, K0)
    print(f"{lab}: ||K-K^T||/||K|| = {np.linalg.norm(Kt-Kt.T)/np.linalg.norm(Kt):.1e}, max|Im kappa| = {np.abs(ev.imag).max():.1e}, min Re kappa = {ev.real.min():.3f}")
