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

# KL(observed || TLOE-predicted) on the total-WIP marginal after d = delta ln(lambda)  (report §6.5)
def marg(pi, N): return np.bincount(N.astype(int), weights=pi)
for K2, label in [(K1, "Jackson"), (3, "blocking K2=3")]:
    Q, n1, n2 = build(lam, mu1, mu2, K2); N = n1 + n2; p0 = marg(stationary(Q), N); k = np.arange(len(p0))
    out = []
    for d in [0.05, 0.1, 0.2, -0.2]:
        pobs = marg(stationary(build(lam * np.exp(d), mu1, mu2, K2)[0]), N)
        pexp = p0 * np.exp(d * k); pexp /= pexp.sum()
        m = pobs > 1e-300
        out.append(f"d={d:+.2f}: KL={np.sum(pobs[m] * np.log(pobs[m] / pexp[m])):.1e}")
    print(label, " | ".join(out))
