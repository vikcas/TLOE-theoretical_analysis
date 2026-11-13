"""Model A (hysteretic autoscaler): no reversible sufficient aggregate for ln(lambda)  [numerical].

Argument (Section 5.4 of the report). Let B = d ln pi / d ln(lambda) be the exact score at the operating point.
Any sufficient Phi satisfies B = b(Phi), so the process B(X) is a coarsening of Phi(X). If Phi(X) were
time-reversible, so would be B(X), and a reversible coarse-grained process satisfies lumped detailed balance:
for every pair of classes a != b of the partition of S by level sets of B,
    J(a->b) = sum_{x in a, y in b} pi(x) q(x,y)  =  J(b->a).
We compute J on that partition and report the largest imbalance. To make sure that numerical rounding of B
does not split a true level set, we also report, for the two classes with the largest imbalance, the distance
of their score values from the score of every other state (rounding is at 1e-6).
Truncation: n <= NMAX, for NMAX = 160 and 240; states with pi < 1e-12 are dropped."""
import numpy as np
import autoscaler as A

A.lam0, A.U, A.L = 1.2, 8, 1
h = 1e-6
for NMAX in (160, 240):
    A.NMAX = NMAX
    for alpha in (0.003, 0.03, 0.1, 1.0, 100.0):
        Q, N, R = A.build(A.lam0, alpha)
        pi = A.stationary(Q)
        with np.errstate(all='ignore'):
            B = (np.log(A.stationary(A.build(A.lam0 * np.exp(h), alpha)[0]))
                 - np.log(A.stationary(A.build(A.lam0 * np.exp(-h), alpha)[0]))) / (2 * h)
        keep = pi > 1e-12
        lev = np.round(B, 6)
        cls = {v: i for i, v in enumerate(np.unique(lev[keep]))}
        c = np.array([cls.get(v, -1) for v in lev])
        Qc = Q.tocoo()
        m = (Qc.row != Qc.col) & keep[Qc.row] & keep[Qc.col]
        a, b = c[Qc.row[m]], c[Qc.col[m]]
        f = pi[Qc.row[m]] * Qc.data[m]
        off = a != b
        K = len(cls)
        J = np.zeros((K, K))
        np.add.at(J, (a[off], b[off]), f[off])
        asym = np.abs(J - J.T)
        i, j = np.unravel_index(asym.argmax(), asym.shape)
        gaps = []
        for k in (i, j):
            members = np.where(c == k)[0]
            others = np.where(keep & (c != k))[0]
            gaps.append(min(np.abs(B[others] - B[x]).min() for x in members))
        sa = [(int(N[x]), int(R[x])) for x in np.where(c == i)[0]][:3]
        sb = [(int(N[x]), int(R[x])) for x in np.where(c == j)[0]][:3]
        print(f"NMAX={NMAX} alpha={alpha:>6}: {K} classes of {keep.sum()} states; "
              f"max |J_ab - J_ba| = {asym.max():.3e} (flux {J[i, j]:.3e} vs {J[j, i]:.3e}, "
              f"{100 * asym.max() / max(J[i, j], J[j, i]):.0f}% of the larger); classes {sa} <-> {sb}; "
              f"sizes {np.sum(c == i)},{np.sum(c == j)}; min score gap {gaps[0]:.2e}, {gaps[1]:.2e}")
