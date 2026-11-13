"""Model A: the minimal sufficient statistic for ln(lambda) is the full state (n, r): every pair of states with n <= 20
is separated by a change of the load (their probability ratio changes)."""
import numpy as np, autoscaler as A
A.lam0, A.U, A.L = 1.2, 8, 1
for alpha in (0.003, 0.1, 100.0):
    Q, N, R = A.build(A.lam0, alpha); p0 = A.stationary(Q); p1 = A.stationary(A.build(A.lam0*1.05, alpha)[0])
    keep = np.where((N <= 20) & (p0 > 1e-14))[0]; lr = np.log(p1[keep]) - np.log(p0[keep])
    d = np.abs(lr[:, None] - lr[None, :]); np.fill_diagonal(d, np.inf); i, j = np.unravel_index(d.argmin(), d.shape)
    r8 = abs((np.log(p1[16]) - np.log(p0[16])) - (np.log(p1[17]) - np.log(p0[17])))
    print(f"alpha={alpha}: {len(keep)} states; closest pair ({int(N[keep[i]])},{int(R[keep[i]])})-({int(N[keep[j]])},{int(R[keep[j]])}): {d.min():.2e}; (8,1) vs (8,2): {r8:.3f}")
