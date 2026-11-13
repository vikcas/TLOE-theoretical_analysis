import numpy as np, scipy.sparse.linalg as sla
from scipy.sparse.linalg import expm_multiply
from exact import build, stationary, lam, mu1, mu2, K1
h = 1e-5
for K2, lab in [(K1, 'J'), (3, 'B3'), (2, 'B2')]:
    Q, n1, n2 = build(lam, mu1, mu2, K2); pi = stationary(Q); N = n1 + n2; Ni = N.astype(int)
    pp, pm = stationary(build(lam*np.exp(h), mu1, mu2, K2)[0]), stationary(build(lam*np.exp(-h), mu1, mu2, K2)[0])
    B = (np.log(pp) - np.log(pm)) / (2*h); B -= pi @ B
    m = np.bincount(Ni, weights=pi*B) / np.maximum(np.bincount(Ni, weights=pi), 1e-300); Fres = (pi @ (B - m[Ni])**2) / (pi @ B**2)
    marg = lambda p: np.bincount(Ni, weights=p); BY = (np.log(marg(pp)) - np.log(marg(pm))) / (2*h); BYx = BY[Ni] - pi @ BY[Ni]
    dN = N - pi @ N; A_ = (-Q).tolil(); A_[0, :] = pi; rhs = dN.copy(); rhs[0] = 0; g = sla.spsolve(A_.tocsc(), rhs); tau = (pi*dN) @ g / (pi @ dN**2)
    chi = (pp @ N - pm @ N) / (2*h); tg = np.linspace(0, 3*tau, 121)
    traj = lambda Q_: expm_multiply(Q_.T, pi, start=0, stop=tg[-1], num=len(tg), endpoint=True) @ N
    Rex = (traj(build(lam*np.exp(h), mu1, mu2, K2)[0]) - traj(build(lam*np.exp(-h), mu1, mu2, K2)[0])) / (2*h)
    C = lambda Bv: expm_multiply(Q.T, pi*Bv, start=0, stop=tg[-1], num=len(tg), endpoint=True) @ N - (pi @ N)*(pi @ Bv)
    eY = np.abs((C(BYx)[0]-C(BYx)) - Rex).max()/abs(chi); eN = np.abs((C(N)[0]-C(N)) - Rex).max()/abs(chi)
    D = np.array([(Ni == k).astype(float) for k in range(12)] + [(Ni >= 12).astype(float)]); p = D @ pi; lag = max(0.5, tau/4)
    Kt = np.array([expm_multiply(Q.T*lag, pi*d) for d in D]) @ D.T
    print(f"| {lab} | {tau:.1f} | {pi@N:.2f} | {Fres:.3f} | {np.linalg.norm(Kt-Kt.T)/np.linalg.norm(Kt):.3f} | {np.abs((Kt-Kt.T)/np.sqrt(np.outer(p,p))).max():.3f} | {eN:.3f} | {eY:.3f} |  (lag {lag:.2f})")
