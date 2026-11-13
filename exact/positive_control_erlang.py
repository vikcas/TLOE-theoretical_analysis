"""Positive control for weber_conditional_check.py: one stage has Erlang-2 service (same mean),
so the series is not ./M/1 and Weber's interchangeability need not hold. Fixed arrival times, empty start."""
import numpy as np, itertools
from scipy.linalg import expm
arr = [0.0, 0.15, 0.4, 0.9, 1.0, 1.7]; times = [0.5, 1.2, 2.0, 3.1]; M = len(arr)

def stage_states(kind):  # (n, phase) ; phase 0 if idle or exponential
    return [(0, 0)] + [(n, ph) for n in range(1, M + 1) for ph in ((0,) if kind == 'M' else (0, 1))]

def build(kinds, mus):
    S1, S2 = stage_states(kinds[0]), stage_states(kinds[1])
    st = [(s1, s2) for s1 in S1 for s2 in S2 if s1[0] + s2[0] <= M]; ix = {s: i for i, s in enumerate(st)}
    Q = np.zeros((len(st),) * 2)
    def completions(s, kind, mu):  # list of (new_state_of_stage, rate, departs?)
        n, ph = s
        if n == 0: return []
        if kind == 'M': return [((n - 1, 0), mu, True)]
        r = 2 * mu
        if ph == 0: return [((n, 1), r, False)]
        return [((n - 1, 0), r, True)]
    for (s1, s2), i in ix.items():
        for ns1, r, dep in completions(s1, kinds[0], mus[0]):
            ns2 = s2
            if dep: ns2 = (s2[0] + 1, s2[1]) if s2[0] > 0 else (1, 0)
            Q[i, ix[(ns1, ns2)]] += r
        for ns2, r, dep in completions(s2, kinds[1], mus[1]):
            Q[i, ix[(s1, ns2)]] += r
    Q -= np.diag(Q.sum(1))
    def arrive(p):
        q = np.zeros_like(p)
        for (s1, s2), i in ix.items():
            if p[i]:
                n1 = (s1[0] + 1, s1[1]) if s1[0] > 0 else (1, 0); q[ix[(n1, s2)]] += p[i]
        return q
    Ntot = np.array([s1[0] + s2[0] for s1, s2 in st]); p0 = np.zeros(len(st)); p0[ix[((0, 0), (0, 0))]] = 1
    return Q, arrive, Ntot, p0

def joint_N(kinds, mus):
    Q, arrive, Ntot, p = build(kinds, mus)
    def evolve(v, t0, t1):
        t = t0
        for a in arr:
            if t0 < a <= t1: v = arrive(v @ expm(Q * (a - t))); t = a
        return v @ expm(Q * (t1 - t))
    br = {(): p}; t = 0.0
    for tk in times:
        new = {}
        for k, v in br.items():
            v = evolve(v, t, tk)
            for n in range(M + 1):
                w = v * (Ntot == n)
                if w.sum() > 1e-300: new[k + (n,)] = w
        br = new; t = tk
    return {k: v.sum() for k, v in br.items()}

def cmp(a, b):
    ks = set(a) | set(b); return max(abs(a.get(k, 0) - b.get(k, 0)) for k in ks)
print(f"sanity (M,M) swapped: {cmp(joint_N(('M','M'),(2.0,1.6)), joint_N(('M','M'),(1.6,2.0))):.1e}")
print(f"positive control (E2,M) vs (M,E2): {cmp(joint_N(('E2','M'),(2.0,1.6)), joint_N(('M','E2'),(1.6,2.0))):.1e}")
