"""Model registry for the finite-sample study: exact references and exact simulators (uniformisation)."""
import numpy as np, scipy.sparse.linalg as sla
from scipy.sparse.linalg import expm_multiply
import exact as T, autoscaler as A
A.lam0, A.U, A.L = 1.2, 8, 1          # counter-model A parameters (fixed for the report)

def _ref_generic(Q, pi, N, Qp, Qm, chi, tg):
    traj = lambda Q_: expm_multiply(Q_.T, pi, start=0, stop=tg[-1], num=len(tg), endpoint=True) @ N
    Rex = (traj(Qp) - traj(Qm)) / (2e-5)
    C = expm_multiply(Q.T, pi * N, start=0, stop=tg[-1], num=len(tg), endpoint=True) @ N - (pi @ N) ** 2
    dN = N - pi @ N; Am = (-Q).tolil(); Am[0, :] = pi; rhs = dN.copy(); rhs[0] = 0
    g = sla.spsolve(Am.tocsc(), rhs); tau = (pi * dN) @ g / (pi @ dN ** 2)
    return dict(R_exact=Rex, R_fdt=C[0] - C, tau=tau, chi=chi)

def tandem(K2):
    h = 1e-5
    def ref(tg):
        Q, n1, n2 = T.build(T.lam, T.mu1, T.mu2, K2); pi = T.stationary(Q); N = n1 + n2
        Qp, Qm = T.build(T.lam*np.exp(h), T.mu1, T.mu2, K2)[0], T.build(T.lam*np.exp(-h), T.mu1, T.mu2, K2)[0]
        chi = (T.stationary(Qp) @ N - T.stationary(Qm) @ N) / (2*h)
        return _ref_generic(Q, pi, N, Qp, Qm, chi, tg)
    def sim(Tmax, dt, R, seed):
        from sim import simulate
        return simulate(K2, Tmax, dt, R, seed)
    return ref, sim

def autoscaler(alpha):
    h = 1e-5
    def ref(tg):
        Q, N, _ = A.build(A.lam0, alpha); pi = A.stationary(Q)
        Qp, Qm = A.build(A.lam0*np.exp(h), alpha)[0], A.build(A.lam0*np.exp(-h), alpha)[0]
        chi = (A.stationary(Qp) @ N - A.stationary(Qm) @ N) / (2*h)
        return _ref_generic(Q, pi, N, Qp, Qm, chi, tg)
    def sim(Tmax, dt, R, seed):
        rng = np.random.default_rng(seed)
        Q, N, _ = A.build(A.lam0, alpha); pi = A.stationary(Q); Qc = Q.tocsr()
        S = Q.shape[0]; out_rate = -Qc.diagonal(); Lam = out_rate.max()
        W = max(np.diff(Qc.indptr)); nbr = np.zeros((S, W), int); cum = np.ones((S, W))
        for s in range(S):
            cols, vals = Qc.indices[Qc.indptr[s]:Qc.indptr[s+1]], Qc.data[Qc.indptr[s]:Qc.indptr[s+1]]
            k = 0; acc = 0.0
            for c, v in zip(cols, vals):
                if c == s: continue
                acc += v / Lam; nbr[s, k] = c; cum[s, k] = acc; k += 1
            for kk in range(k, W): nbr[s, kk] = s; cum[s, kk] = 1.0 + kk   # self-loop for the remaining probability
            if k < W: cum[s, k:] = 2.0
        s = rng.choice(S, size=R, p=pi); Nint = N.astype(np.int16)
        nsteps = int(round(Tmax / dt)); out = np.empty((R, nsteps + 1), np.int16); out[:, 0] = Nint[s]
        for t in range(1, nsteps + 1):
            m = rng.poisson(Lam * dt, size=R)
            for j in range(m.max()):
                act = m > j; u = rng.random(R)
                k = (u[:, None] > cum[s]).sum(1)          # index of the chosen neighbour; == number of real nbrs -> self-loop
                nxt = np.where(k < W, nbr[s, np.minimum(k, W - 1)], s)
                nxt = np.where(cum[s, np.minimum(k, W - 1)] > 1.0, s, nxt)
                s = np.where(act, nxt, s)
            out[:, t] = Nint[s]
        return out
    return ref, sim

MODELS = {'J': tandem(T.K1), 'B3': tandem(3), 'B2': tandem(2), 'A': autoscaler(0.1)}
