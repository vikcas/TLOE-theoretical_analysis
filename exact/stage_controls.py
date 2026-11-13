"""Corollary 2 and Lemma 2: reversibility of (N, n_j) and block interchangeability in K-stage tandems (Table 2)."""
import numpy as np
from kstage import build, stationary, joint
lam = 1.0
# (rates, 1-based stage j)
cases = [((2.0, 1.6, 2.0), 2), ((2.0, 2.0, 2.0), 2), ((2.0, 1.6, 1.8), 2), ((3.0, 1.3, 2.0), 2), ((2.0, 1.6, 2.05), 2),
         ((2.0, 1.6, 1.6, 2.0), 2), ((2.0, 1.6, 1.6, 2.0), 3), ((2.0, 1.6, 1.8, 2.0, 1.6), 3), ((2.0, 1.7, 1.8, 1.7, 2.0), 3)]
for mus, j in cases:
    NCs = {3: (18, 24), 4: (14, 17), 5: (11, 13)}[len(mus)]
    for NC in NCs:
        Q, st = build(lam, mus, NC); pi = stationary(Q); N = st.sum(1)
        lab = np.c_[N, st[:, j - 1]]
        out = [np.abs((J := joint(Q, pi, lab, t)) - J.T).max() for t in (0.5, 1.0, 2.0)]
        print(f"K={len(mus)} mu={mus} (N,n{j}) NC={NC}: max|J-J^T| t=0.5,1,2: " + ", ".join(f"{x:.1e}" for x in out), flush=True)
def J2(mus, j, NC, t=1.0):
    Q, st = build(lam, mus, NC); pi = stationary(Q); N = st.sum(1)
    return joint(Q, pi, np.c_[N, st[:, j-1]], t, pmin=0)   # all labels, same ordering for both chains
# 1) block-permutation invariance of the law of (N, n_j)
for a, b, j in [((2.0, 1.6, 1.8, 2.2), (1.6, 2.0, 1.8, 2.2), 3), ((2.0, 1.6, 1.8, 2.2, 1.7), (2.0, 1.6, 1.8, 1.7, 2.2), 3),
                ((2.0, 1.6, 1.8, 2.2), (1.6, 1.8, 2.0, 2.2), 3)]:
    NC = 13 if len(a) == 5 else 15
    d = np.abs(J2(a, j, NC) - J2(b, j, NC)).max()
    print(f"law of (N,n{j}): {a} vs {b}: max diff {d:.1e}", flush=True)
# 2) middle stage, multisets equal vs different (K=5)
for mus in [(2.0, 1.6, 1.8, 1.6, 2.0), (1.6, 2.0, 1.8, 2.0, 1.6), (2.0, 1.6, 1.8, 2.0, 1.7), (2.0, 1.6, 1.8, 2.2, 1.4)]:
    out = []
    for NC in (11, 13):
        J = J2(mus, 3, NC); out.append(np.abs(J - J.T).max())
    print(f"K=5 mu={mus} (N,n3): max|J-J^T| NC=11,13: {out[0]:.1e}, {out[1]:.1e}", flush=True)
