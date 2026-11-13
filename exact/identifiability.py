"""Model B: FDT with the marginal (Y-only) conjugate B^Y(N) = d ln pi^N / d ln lambda.
Static response is reproduced exactly (missing-information identity); the transient is not."""
import numpy as np
from scipy.sparse.linalg import expm_multiply
from exact import build, stationary, lam, mu1, mu2
h = 1e-5
for K2, lab in [(70, 'J'), (3, 'B3')]:
    Q, n1, n2 = build(lam, mu1, mu2, K2); pi = stationary(Q); N = n1 + n2; Ni = N.astype(int)
    marg = lambda p: np.bincount(Ni, weights=p)
    sp_, sm_ = stationary(build(lam*np.exp(h), mu1, mu2, K2)[0]), stationary(build(lam*np.exp(-h), mu1, mu2, K2)[0])
    BY = (np.log(marg(sp_)) - np.log(marg(sm_))) / (2*h); BYx = BY[Ni]; BYx -= pi @ BYx
    tg = np.array([0, 1, 4, 16, 64, 200.])
    C = lambda B: np.array([expm_multiply(Q.T*t, pi*B) @ N for t in tg]) - (pi@N)*(pi@B)
    RY = C(BYx)[0] - C(BYx); RN = C(N)[0] - C(N)
    traj = lambda Q_: np.array([expm_multiply(Q_.T*t, pi) @ N for t in tg])
    Rex = (traj(build(lam*np.exp(h), mu1, mu2, K2)[0]) - traj(build(lam*np.exp(-h), mu1, mu2, K2)[0])) / (2*h)
    chi = (sp_ @ N - sm_ @ N) / (2*h)
    print(f"{lab}: t={tg.tolist()}\n   exact   {np.round(Rex,3)}  (static {chi:.3f})\n   B=B^Y(N){np.round(RY,3)}  (static Cov(N,B^Y)={pi@(N*BYx):.3f})\n   B=N     {np.round(RN,3)}  (static Var N={pi@(N*(N-pi@N)):.3f})")
