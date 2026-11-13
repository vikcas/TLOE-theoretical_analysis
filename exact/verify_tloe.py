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

def analyse(K2, label):
    Q, n1, n2 = build(lam, mu1, mu2, K2); pi = stationary(Q); N = n1 + n2; Ni = N.astype(int)
    Qp = build(lam * np.exp(h), mu1, mu2, K2)[0]; Qm = build(lam * np.exp(-h), mu1, mu2, K2)[0]
    B = (np.log(stationary(Qp)) - np.log(stationary(Qm))) / (2 * h); B -= pi @ B   # d ln pi / d ln lambda
    traj = lambda Q_, A: expm_multiply(Q_.T, pi, start=0, stop=T, num=NT, endpoint=True) @ A
    C = lambda A, Bv: expm_multiply(Q.T, pi * Bv, start=0, stop=T, num=NT, endpoint=True) @ A - (pi @ A) * (pi @ Bv)
    step = lambda A, Bv: C(A, Bv)[0] - C(A, Bv)          # FDT-predicted step response
    print(f"\n==== {label} ====")
    L = np.log(pi).reshape(K1 + 1, K2 + 1)
    print(f"[product form] max|ln pi - additive| (n<=5): {np.abs((L - L[:, :1] - L[:1, :] + L[0, 0])[:6, :min(6, K2 + 1)]).max():.1e}")
    for nm, A in [("n2", n2), ("N", N)]:
        ex = (traj(Qp, A) - traj(Qm, A)) / (2 * h)
        print(f"[FDT] A={nm}: exact t=1,4 = {ex[1]:.3f},{ex[4]:.3f} | B=N {step(A,N)[1]:.3f},{step(A,N)[4]:.3f}"
              f" | B=n2 (per-stage) {step(A,n2)[1]:.3f},{step(A,n2)[4]:.3f}"
              f" | B=n1 {step(A,n1)[1]:.3f},{step(A,n1)[4]:.3f} | B=dln(pi): max err {np.abs(ex - step(A,B)).max():.1e}")
    m = np.bincount(Ni, weights=pi * B) / np.maximum(np.bincount(Ni, weights=pi), 1e-300)
    print(f"[F sufficiency] Var(B - E[B|N]) / Var(B) = {(pi @ (B - m[Ni])**2) / (pi @ B**2):.1e}")
    E = lambda w, g, t: expm_multiply(Q.T * t, pi * w) @ g
    print("[R two-time] N (x^2,x) asym t=1,4:", [f"{E(N**2,N,t)-E(N,N**2,t):+.1e}" for t in (1, 4)],
          "| (n1,n2) cross asym t=1:", f"{E(n1,n2,1)-E(n2,n1,1):+.2e}")
    D = np.array([(Ni == k).astype(float) for k in range(12)] + [(Ni >= 12).astype(float)])
    Kt = np.array([expm_multiply(Q.T * 0.5, pi * d) for d in D]) @ D.T
    print(f"[R lagged-cov] ||K-K^T||/||K|| (indicator dictionary on N, tau=0.5) = {np.linalg.norm(Kt-Kt.T)/np.linalg.norm(Kt):.1e}")
    if K2 == K1:  # reciprocity between controls ln(lambda) [conj N] and ln(mu1) [conj -n1]
        print("[Onsager] step resp. of N to d ln mu1  vs  of -n1 to d ln lambda (t=1,4):",
              np.round(step(N, -n1)[[1, 4]], 3), np.round(step(-n1, N)[[1, 4]], 3))
        Qa = build(lam, mu1 * np.exp(h), mu2, K2)[0]; Qb = build(lam, mu1 * np.exp(-h), mu2, K2)[0]
        for nm, A in [("n1", n1), ("n2", n2)]:
            ex = (traj(Qa, A) - traj(Qb, A)) / (2 * h)
            print(f"[FDT mu1] A={nm}: max|exact - FDT[B=-n1]| = {np.abs(ex - step(A, -n1)).max():.1e}")

analyse(K1, "Model J: open tandem (Jackson)")
analyse(3, "Model B: tandem with blocking, K2 = 3")
