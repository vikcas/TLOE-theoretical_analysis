"""Section 5.3 of the report  [numerical].
Model B (tandem, lambda=1, mu1=2, mu2=1.6, blocking after service, buffer K2 at stage 2).
(1) Minimal sufficient statistic for the load knob: group states whose probability ratio does not change with lambda.
(2) How the violation of (F) at N shrinks as the buffer grows."""
import numpy as np
import exact as E   # run from repo/exact (see run_all.sh)
h = 1e-5
# (1) grouping, buffer 3, states with n1 <= 12, lambda in {0.8, 0.9, 1.0, 1.1}
K2 = 3
lp = {l: np.log(E.stationary(E.build(l, E.mu1, E.mu2, K2)[0])) for l in (0.8, 0.9, 1.0, 1.1)}
_, n1, n2 = E.build(1.0, E.mu1, E.mu2, K2)
sel = np.where(n1 <= 12)[0]
V = np.stack([lp[l][sel] - lp[1.0][sel] for l in (0.8, 0.9, 1.1)], 1)   # how each state's log-prob moves with lambda
# two states are in the same group iff their rows of V coincide
D = np.abs(V[:, None, :] - V[None, :, :]).max(2); np.fill_diagonal(D, np.inf)
print(f"(1) buffer 3: {len(sel)} states with n1<=12; smallest difference in log-probability movement between two distinct states = {D.min():.3e}")
i, j = np.unravel_index(D.argmin(), D.shape)
print(f"    closest pair: {(int(n1[sel[i]]), int(n2[sel[i]]))} and {(int(n1[sel[j]]), int(n2[sel[j]]))}")
# (2) violation vs buffer size
print("(2) buffer | static bias of prediction with B=N | conjugate variance not explained by N | P(stage 2 full)")
for K2 in (2, 3, 5, 8, 12, 20):
    Q, n1, n2 = E.build(E.lam, E.mu1, E.mu2, K2); pi = E.stationary(Q); N = n1 + n2
    pp = E.stationary(E.build(E.lam*np.exp(h), E.mu1, E.mu2, K2)[0]); pm = E.stationary(E.build(E.lam*np.exp(-h), E.mu1, E.mu2, K2)[0])
    k = pi > 1e-14; B = np.zeros_like(pi); B[k] = (np.log(pp[k]) - np.log(pm[k]))/(2*h); B -= pi @ B
    EB = np.zeros_like(B)
    for v in np.unique(N):
        m = N == v; w = pi[m].sum()
        if w > 0: EB[m] = pi[m] @ B[m] / w
    chi = pi @ (N * B); varN = pi @ (N - pi@N)**2
    print(f"   {K2:5d} | {1 - varN/chi:8.4f} | {(pi @ (B-EB)**2)/(pi @ B**2):10.3e} | {pi[n2 == K2].sum():.4f}")
