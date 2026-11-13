"""K-stage ./M/1 tandem, truncated at total N <= NC (arrivals blocked at the cap; checked by truncation sweep)."""
import numpy as np, itertools, scipy.sparse as sp, scipy.sparse.linalg as sla
from scipy.sparse.linalg import expm_multiply
def build(lam, mus, NC):
    K = len(mus)
    st = [s for s in itertools.product(range(NC + 1), repeat=K) if sum(s) <= NC]
    ix = {s: i for i, s in enumerate(st)}; r, c, v = [], [], []
    for s, i in ix.items():
        if sum(s) < NC: t = list(s); t[0] += 1; r.append(i); c.append(ix[tuple(t)]); v.append(lam)
        for k in range(K):
            if s[k] > 0:
                t = list(s); t[k] -= 1
                if k < K - 1: t[k + 1] += 1
                r.append(i); c.append(ix[tuple(t)]); v.append(mus[k])
    M = sp.csr_matrix((v, (r, c)), shape=(len(st),) * 2)
    Q = (M - sp.diags(np.asarray(M.sum(1)).ravel())).tocsc()
    return Q, np.array(st)
def stationary(Q):
    A = Q.T.tolil(); A[0, :] = 1; b = np.zeros(Q.shape[0]); b[0] = 1
    p = sla.spsolve(A.tocsc(), b); return p / p.sum()
def joint(Q, pi, lab, t, pmin=1e-4):
    """two-time joint P(L0=a, Lt=b) over labels with stationary mass > pmin"""
    labs = sorted(set(map(tuple, lab))); m = {l: i for i, l in enumerate(labs)}
    li = np.array([m[tuple(l)] for l in lab]); pl = np.bincount(li, weights=pi)
    keep = np.where(pl > pmin)[0]
    D = np.array([(li == a).astype(float) for a in keep])
    return np.array([expm_multiply(Q.T * t, pi * d) for d in D]) @ D.T
