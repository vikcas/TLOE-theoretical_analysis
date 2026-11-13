"""Section 10 (limitations, item 3) of the report  [numerical].
Single queue, Erlang-2 service (mean 1, each phase rate 2), Poisson arrivals rate lam.
Compare FCFS (M/E2/1) and processor sharing (M/E2/1-PS).
Question: is the score B = d ln pi / d ln lam a function of n alone?  Report Var(B - E[B|n]) / Var(B).
Then: is the passive prediction of the response of <n> to d ln lam with B = n - <n> exact? (sup error over t<=20 / static)."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
from scipy.sparse.linalg import expm_multiply
NMAX = 120
def build_fcfs(lam):
    states = [(0, 0)] + [(n, ph) for n in range(1, NMAX+1) for ph in (1, 2)]
    idx = {s: i for i, s in enumerate(states)}; r, c, v = [], [], []
    def add(s, t, rate):
        if t in idx: r.append(idx[s]); c.append(idx[t]); v.append(rate)
    for (n, ph) in states:
        s = (n, ph)
        if n == 0: add(s, (1, 1), lam)
        else:
            add(s, (n+1, ph), lam)
            if ph == 1: add(s, (n, 2), 2.0)
            else: add(s, (n-1, 1) if n > 1 else (0, 0), 2.0)
    Q = sp.csr_matrix((v, (r, c)), shape=(len(states),)*2)
    Q = Q - sp.diags(np.asarray(Q.sum(1)).ravel())
    n = np.array([s[0] for s in states], float)
    return Q, n
def build_ps(lam):
    states = [(a, b) for a in range(NMAX+1) for b in range(NMAX+1-a)]  # a jobs in phase 1, b in phase 2
    idx = {s: i for i, s in enumerate(states)}; r, c, v = [], [], []
    def add(s, t, rate):
        if t in idx and rate > 0: r.append(idx[s]); c.append(idx[t]); v.append(rate)
    for (a, b) in states:
        s = (a, b); n = a + b
        add(s, (a+1, b), lam)
        if n > 0:
            add(s, (a-1, b+1), 2.0*a/n)
            add(s, (a, b-1), 2.0*b/n)
    Q = sp.csr_matrix((v, (r, c)), shape=(len(states),)*2)
    Q = Q - sp.diags(np.asarray(Q.sum(1)).ravel())
    n = np.array([s[0]+s[1] for s in states], float)
    return Q, n
def stationary(Q):
    A = Q.T.tolil(); A[0, :] = 1; b = np.zeros(Q.shape[0]); b[0] = 1
    p = sla.spsolve(A.tocsc(), b); return np.maximum(p, 0) / np.maximum(p, 0).sum()
lam, h = 0.7, 1e-5
for name, build_ in (('FCFS', build_fcfs), ('PS', build_ps)):
    Q, n = build_(lam); pi = stationary(Q)
    pp = stationary(build_(lam*np.exp(h))[0]); pm = stationary(build_(lam*np.exp(-h))[0])
    keep = pi > 1e-13
    B = np.zeros_like(pi); B[keep] = (np.log(pp[keep]) - np.log(pm[keep])) / (2*h)
    B -= pi @ B
    # E[B | n]
    EB = np.zeros_like(B)
    for k in np.unique(n):
        m = n == k; w = pi[m].sum()
        if w > 0: EB[m] = (pi[m] @ B[m]) / w
    unexpl = (pi @ (B - EB)**2) / (pi @ B**2)
    # passive prediction of response of <n> with B = n - <n> vs exact step response
    tg = np.linspace(0, 20, 81)
    traj = lambda Q_: expm_multiply(Q_.T, pi, start=0, stop=tg[-1], num=len(tg), endpoint=True) @ n
    exact = (traj(build_(lam*np.exp(h))[0]) - traj(build_(lam*np.exp(-h))[0])) / (2*h)
    Bn = n - pi @ n
    C = expm_multiply(Q.T, pi*Bn, start=0, stop=tg[-1], num=len(tg), endpoint=True) @ n
    pred = C[0] - C
    chi = pi @ (n * B)   # exact static response
    print(f"{name}: <n>={pi@n:.4f}  Var(n)={pi@(n-pi@n)**2:.4f}  static response={chi:.4f}  "
          f"unexplained fraction of score by n = {unexpl:.3e}  "
          f"max |pred - exact| / static over t<=20 = {np.max(np.abs(pred-exact))/chi:.3e}  "
          f"(t=1: exact {exact[4]:.4f}, pred {pred[4]:.4f})  P(n=NMAX)~{pi[n==NMAX].sum():.1e}")
