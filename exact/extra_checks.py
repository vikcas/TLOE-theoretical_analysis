import numpy as np, scipy.linalg as sl, json
from scipy.sparse.linalg import expm_multiply
from exact import build, stationary, reference, lam, mu1, mu2, K1
rng = np.random.default_rng(0)
# (1) N Markov? Chapman-Kolmogorov on the two-time joint of N (model J)
for K2, lab in [(K1, 'J'), (3, 'B3')]:
    Q, n1, n2 = build(lam, mu1, mu2, K2); pi = stationary(Q); Ni = (n1 + n2).astype(int); M = 25
    D = np.array([(Ni == k).astype(float) for k in range(M)]); p = D @ pi
    Jt = lambda t: np.array([expm_multiply(Q.T * t, pi * d) for d in D]) @ D.T
    for t in (0.5, 1.0):
        J1, J2 = Jt(t), Jt(2 * t); pred = J1 @ np.diag(1 / p) @ J1
        print(f"[CK {lab}] t={t}: max|J(2t) - J(t)P(t)| = {np.abs(J2 - pred).max():.2e}, max J(2t) = {J2.max():.2e}, rel = {np.abs(J2-pred).max()/J2.max():.3f}")
    # implied timescales of the 13-bin indicator pencil
    Db = np.array([(Ni == k).astype(float) for k in range(12)] + [(Ni >= 12).astype(float)]); K0 = (Db * pi) @ Db.T
    its = []
    for tau in (0.5, 1, 2, 4, 8, 16):
        Kt = np.array([expm_multiply(Q.T * tau, pi * d) for d in Db]) @ Db.T
        ev = np.sort(np.real(sl.eigvals(Kt, K0)))[::-1]
        its.append(-tau / np.log(ev[1]))
    print(f"[ITS {lab}] slowest implied timescale at tau=0.5,1,2,4,8,16: " + ", ".join(f"{x:.3f}" for x in its))
    # asymptotic decay rate of C_NN
    tg = np.array([100., 150., 200.]); N = n1 + n2
    C = np.array([expm_multiply(Q.T * t, pi * N) @ N for t in tg]) - (pi @ N) ** 2
    print(f"[decay {lab}] -dlnC/dt over [100,200]: {np.log(C[0]/C[2])/100:.4f} -> timescale {100/np.log(C[0]/C[2]):.2f}")
print(f"[gap J] stage gaps (sqrt mu - sqrt lam)^2: stage1 {(np.sqrt(mu1)-1)**2:.4f} ({1/(np.sqrt(mu1)-1)**2:.2f}), stage2 {(np.sqrt(mu2)-1)**2:.4f} ({1/(np.sqrt(mu2)-1)**2:.2f})")
# (2) Table 1 capacity row: response of n1, n2 to d ln mu1 at t=1,4, exact vs FDT with B=-n1
Q, n1, n2 = build(lam, mu1, mu2, K1); pi = stationary(Q); h = 1e-5
tg = np.array([0., 1., 4.])
traj = lambda Q_, A: np.array([expm_multiply(Q_.T * t, pi) @ A for t in tg])
Qa, Qb = build(lam, mu1 * np.exp(h), mu2, K1)[0], build(lam, mu1 * np.exp(-h), mu2, K1)[0]
for nm, A in [('n1', n1), ('n2', n2), ('N', n1 + n2)]:
    ex = (traj(Qa, A) - traj(Qb, A)) / (2 * h)
    C = np.array([expm_multiply(Q.T * t, pi * (-n1)) @ A for t in tg]) - (pi @ A) * (pi @ -n1); pr = C[0] - C
    print(f"[cap] d ln mu1 -> {nm}: exact t=1,4 = {ex[1]:.3f}, {ex[2]:.3f}; FDT(B=-n1) = {pr[1]:.3f}, {pr[2]:.3f}")
# (3) three-time asymmetry for model B, with positive control
def three(Q, pi, f, g, hh, t):
    v = expm_multiply(Q * t, hh); v = expm_multiply(Q * t, g * v); return pi @ (f * v)
for K2, lab in [(K1, 'J'), (3, 'B3')]:
    Q, n1, n2 = build(lam, mu1, mu2, K2); pi = stationary(Q); Ni = (n1 + n2).astype(int)
    mx = mxc = 0
    for _ in range(8):
        a, b, c = [rng.normal(size=200) for _ in range(3)]
        mx = max(mx, abs(three(Q, pi, a[Ni], b[Ni], c[Ni], 0.7) - three(Q, pi, c[Ni], b[Ni], a[Ni], 0.7)))
        F, G, H = [np.tanh(rng.normal(size=n1.size)) for _ in range(3)]
        mxc = max(mxc, abs(three(Q, pi, F, G, H, 0.7) - three(Q, pi, H, G, F, 0.7)))
    print(f"[3-time {lab}] N: max asym = {mx:.1e}; positive control = {mxc:.2e}")
# (4) KL coefficient: linear extrapolation from delta = 1e-3, 1e-2 (more digits)
Qb_, a1, a2 = build(lam, mu1, mu2, 3); pib = stationary(Qb_); Nbi = (a1 + a2).astype(int)
marg = lambda p: np.bincount(Nbi, weights=p); p0 = marg(pib); k = np.arange(len(p0))
sc = (np.log(marg(stationary(build(lam*np.exp(h), mu1, mu2, 3)[0]))) - np.log(marg(stationary(build(lam*np.exp(-h), mu1, mu2, 3)[0])))) / (2*h)
mis = sc - k; half = 0.5 * (p0 @ mis**2 - (p0 @ mis)**2); r = {}
for d in (1e-3, 1e-2):
    po = marg(stationary(build(lam*np.exp(d), mu1, mu2, 3)[0])); pe = p0*np.exp(d*k); pe /= pe.sum(); r[d] = np.sum(po*np.log(po/pe))/d**2
slope = (r[1e-2]-r[1e-3])/(9e-3); print(f"[KL] ratios {r[1e-3]:.5f}, {r[1e-2]:.5f}; slope {slope:.3f}; extrapolated {r[1e-3]-slope*1e-3:.5f}; predicted {half:.5f}")
# (5) cross-scale: relative errors w.r.t. exact
pred = -1.11111; ex = np.array([-2.26263, -1.51337, -1.15715, -1.11582, -1.11158])
print("[xscale] |pred-exact|/|exact|:", np.round(np.abs(pred-ex)/np.abs(ex)*100, 3), "  /|pred|:", np.round(np.abs(pred-ex)/abs(pred)*100, 3))
