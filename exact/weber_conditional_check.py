"""Exact check of the form of Weber's interchangeability needed by the lemma:
for FIXED (deterministic) arrival times, the joint law of the departure process
(equivalently of N(t) = A(t) - D(t) at several times) of two ./M/1 queues in series
does not depend on the order of the servers. Initially empty system."""
import numpy as np
from scipy.linalg import expm
import itertools

def gen(m1, m2, M):
    # states (n1,n2), n1+n2 <= M ; no external arrivals in this generator
    st = [(a, b) for a in range(M + 1) for b in range(M + 1 - a)]
    ix = {s: i for i, s in enumerate(st)}
    Q = np.zeros((len(st), len(st)))
    for (a, b), i in ix.items():
        if a > 0: Q[i, ix[(a - 1, b + 1)]] += m1
        if b > 0: Q[i, ix[(a, b - 1)]] += m2
    Q -= np.diag(Q.sum(1))
    return st, ix, Q

def arrive(p, st, ix):
    q = np.zeros_like(p)
    for (a, b), i in ix.items():
        if p[i]: q[ix[(a + 1, b)]] += p[i]
    return q

def evolve(p, t0, t1, arr, st, ix, Q):
    """propagate row vector p from t0 to t1, applying arrivals in (t0, t1]"""
    t = t0
    for a in arr:
        if t0 < a <= t1:
            p = p @ expm(Q * (a - t)); p = arrive(p, st, ix); t = a
    return p @ expm(Q * (t1 - t))

def joint_N(m1, m2, arr, times):
    M = len(arr); st, ix, Q = gen(m1, m2, M)
    Ntot = np.array([a + b for a, b in st])
    p = np.zeros(len(st)); p[ix[(0, 0)]] = 1.0
    # joint law of (N(t1),...,N(tk)) by branching on N at each time
    branches = {(): p}; t = 0.0
    for tk in times:
        new = {}
        for key, v in branches.items():
            v = evolve(v, t, tk, arr, st, ix, Q)
            for n in range(M + 1):
                w = v * (Ntot == n)
                if w.sum() > 1e-300: new[key + (n,)] = w
        branches = new; t = tk
    return {k: v.sum() for k, v in branches.items()}

arr = [0.0, 0.15, 0.4, 0.9, 1.0, 1.7]
times = [0.5, 1.2, 2.0, 3.1]
for (m1, m2) in [(2.0, 1.6), (0.7, 3.0)]:
    J12 = joint_N(m1, m2, arr, times); J21 = joint_N(m2, m1, arr, times)
    keys = set(J12) | set(J21)
    d = max(abs(J12.get(k, 0) - J21.get(k, 0)) for k in keys)
    print(f"mu=({m1},{m2}) vs ({m2},{m1}): {len(keys)} atoms, max |diff| = {d:.1e}, total mass {sum(J12.values()):.12f}")
# Positive control (Erlang-2 stage, interchangeability must fail): see positive_control_erlang.py
