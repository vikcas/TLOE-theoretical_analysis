"""Counter-model A: a queue with a hysteretic replica autoscaler.
State (n, r): n jobs, r in {1, 2} replicas, each serving at rate mu (service rate min(n, r) mu).
Scale-up at rate alpha when r = 1 and n >= U; scale-down at rate alpha when r = 2 and n <= L.
Top-level observable Y = N = n; control ln(lambda). 1/alpha is the autoscaler's reaction time."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
from scipy.sparse.linalg import expm_multiply
lam0, mu, U, L, NMAX = 1.5, 1.0, 6, 2, 160
def build(lam, alpha):
    idx = lambda n, r: 2 * n + (r - 1); S = 2 * (NMAX + 1); rr, cc, vv = [], [], []
    for n in range(NMAX + 1):
        for r in (1, 2):
            i = idx(n, r)
            if n < NMAX: rr.append(i); cc.append(idx(n + 1, r)); vv.append(lam)
            if n > 0: rr.append(i); cc.append(idx(n - 1, r)); vv.append(min(n, r) * mu)
            if r == 1 and n >= U: rr.append(i); cc.append(idx(n, 2)); vv.append(alpha)
            if r == 2 and n <= L: rr.append(i); cc.append(idx(n, 1)); vv.append(alpha)
    M = sp.csr_matrix((vv, (rr, cc)), shape=(S, S))
    Q = (M - sp.diags(np.asarray(M.sum(1)).ravel())).tocsc()
    nv = np.repeat(np.arange(NMAX + 1), 2).astype(float); rv = np.tile([1., 2.], NMAX + 1)
    return Q, nv, rv
def stationary(Q):
    A = Q.T.tolil(); A[0, :] = 1; b = np.zeros(Q.shape[0]); b[0] = 1
    p = sla.spsolve(A.tocsc(), b); return p / p.sum()
def analyse(alpha, h=1e-5, verbose=True):
    Q, N, R = build(lam0, alpha); pi = stationary(Q); Ni = N.astype(int)
    pp, pm = stationary(build(lam0 * np.exp(h), alpha)[0]), stationary(build(lam0 * np.exp(-h), alpha)[0])
    B = (np.log(pp) - np.log(pm)) / (2 * h); B -= pi @ B
    m = np.bincount(Ni, weights=pi * B) / np.maximum(np.bincount(Ni, weights=pi), 1e-300)
    Fres = (pi @ (B - m[Ni]) ** 2) / (pi @ B ** 2)
    marg = lambda p: np.bincount(Ni, weights=p)
    BY = (np.log(marg(pp)) - np.log(marg(pm))) / (2 * h); BYx = BY[Ni] - pi @ BY[Ni]
    dN = N - pi @ N; A_ = (-Q).tolil(); A_[0, :] = pi; rhs = dN.copy(); rhs[0] = 0
    g = sla.spsolve(A_.tocsc(), rhs); tau = (pi * dN) @ g / (pi @ dN ** 2)
    chi = (pp @ N - pm @ N) / (2 * h)
    tg = np.linspace(0, 3 * tau, 121)
    traj = lambda Q_: expm_multiply(Q_.T, pi, start=0, stop=tg[-1], num=len(tg), endpoint=True) @ N
    Rex = (traj(build(lam0 * np.exp(h), alpha)[0]) - traj(build(lam0 * np.exp(-h), alpha)[0])) / (2 * h)
    C = lambda Bv: expm_multiply(Q.T, pi * Bv, start=0, stop=tg[-1], num=len(tg), endpoint=True) @ N - (pi @ N) * (pi @ Bv)
    eY = np.abs((C(BYx)[0] - C(BYx)) - Rex).max() / abs(chi)
    eN = np.abs((C(N)[0] - C(N)) - Rex).max() / abs(chi)
    # (R): indicator dictionary on N (bins 0..11, >=12), lag tau/4: relative asymmetry and largest standardized component
    D = np.array([(Ni == k).astype(float) for k in range(12)] + [(Ni >= 12).astype(float)]); p = D @ pi
    lag = max(0.5, tau / 4)
    Kt = np.array([expm_multiply(Q.T * lag, pi * d) for d in D]) @ D.T
    rel = np.linalg.norm(Kt - Kt.T) / np.linalg.norm(Kt); stdmax = np.abs((Kt - Kt.T) / np.sqrt(np.outer(p, p))).max()
    if verbose:
        print(f"alpha={alpha:>7}: tau_int={tau:7.2f} E[N]={pi@N:6.2f} | (F) unexplained Var(B) by N = {Fres:.3f} | "
              f"(R) ||K-K^T||/||K|| (lag {lag:.1f}) = {rel:.3f}, max std asym = {stdmax:.3f} | pred. error sup/chi: B^Y {eY:.3f}, B=N {eN:.3f}", flush=True)
    return dict(alpha=alpha, tau=tau, EN=float(pi @ N), Fres=float(Fres), Rrel=float(rel), Rstd=float(stdmax), eY=float(eY), eN=float(eN), chi=float(chi))
if __name__ == "__main__":
    import json
    res = [analyse(a) for a in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)]
    json.dump(res, open('autoscaler_exact.json', 'w'), indent=1)
