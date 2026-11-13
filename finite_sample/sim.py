"""Exact simulation of the tandem (models J, B) by uniformization, vectorized over replicates.
Telemetry = total WIP N sampled every dt (as a monitoring system would). Initial state ~ pi (exact)."""
import numpy as np
from exact import reference, lam, mu1, mu2, K1

def simulate(K2, T, dt, R, seed):
    rng = np.random.default_rng(seed)
    ref = reference(K2, np.array([0.0, 1.0]))
    pi, n1v, n2v = ref['pi'], ref['n1'].astype(np.int64), ref['n2'].astype(np.int64)
    s0 = rng.choice(len(pi), size=R, p=pi)
    n1, n2 = n1v[s0].copy(), n2v[s0].copy()
    Lam = lam + mu1 + mu2
    p_arr, p_s1 = lam / Lam, (lam + mu1) / Lam
    nsteps = int(round(T / dt))
    out = np.empty((R, nsteps + 1), dtype=np.int16); out[:, 0] = n1 + n2
    cap2 = K2 if K2 < K1 else 10**9
    for k in range(1, nsteps + 1):
        m = rng.poisson(Lam * dt, size=R)
        for j in range(m.max()):
            act = m > j
            u = rng.random(R)
            arr = act & (u < p_arr)
            s1 = act & (u >= p_arr) & (u < p_s1) & (n1 > 0) & (n2 < cap2)
            s2 = act & (u >= p_s1) & (n2 > 0)
            n1 += arr; n1 -= s1; n2 += s1; n2 -= s2
        out[:, k] = n1 + n2
    return out
