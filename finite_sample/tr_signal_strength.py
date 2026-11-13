import numpy as np
from scipy.sparse.linalg import expm_multiply
from exact import reference, K1
tg = np.linspace(0, 60, 241)
for K2 in (1, 2, 3):
    r = reference(K2, tg); Q, pi, N = r['Q'], r['pi'], r['N']; Ni = N.astype(int)
    M = 13
    D = np.array([(Ni == k).astype(float) for k in range(M-1)] + [(Ni >= M-1).astype(float)])
    p = D @ pi
    best = []
    for t in (0.5, 1, 2, 4, 8, 16):
        J = np.array([expm_multiply(Q.T*t, pi*d) for d in D]) @ D.T
        S = (J - J.T) / np.sqrt(np.outer(p, p))
        a, b = np.unravel_index(np.abs(S).argmax(), S.shape)
        best.append((t, S[a, b], a, b))
    bias = np.abs(r['R_fdt']-r['R_exact']).max()/r['chi']
    print(f"K2={K2}: tau_int={r['tau_int']:.2f}  E[N]={r['meanN']:.2f}  FDT bias (sup/chi)={bias:.3f}  static bias={(r['chi']-r['varN'])/r['chi']:.3f}")
    print("   max standardized indicator asym:", ", ".join(f"t={t}: {s:+.4f} (N={a},{b})" for t, s, a, b in best))
