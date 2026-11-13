"""T-F cost estimate (same design as tf_power.py) for model A (autoscaler, alpha = 0.1)."""
import numpy as np
from scipy.sparse.linalg import expm_multiply
import autoscaler as A
A.lam0, A.U, A.L = 1.2, 8, 1; alpha = 0.1; h = 1e-5
Q, N, _ = A.build(A.lam0, alpha); pi = A.stationary(Q); Ni = N.astype(int)
r = A.analyse(alpha, verbose=False); tau = r['tau']; tg = np.linspace(0, 4*tau, 401)
pp, pm = A.stationary(A.build(A.lam0*np.exp(h), alpha)[0]), A.stationary(A.build(A.lam0*np.exp(-h), alpha)[0])
marg = lambda p: np.bincount(Ni, weights=p); BY = (np.log(marg(pp)) - np.log(marg(pm))) / (2*h); BYx = BY[Ni] - pi @ BY[Ni]
E = lambda B: expm_multiply(Q.T, pi*B, start=0, stop=tg[-1], num=len(tg), endpoint=True) @ N - (pi @ N)*(pi @ B)
traj = lambda Q_: expm_multiply(Q_.T, pi, start=0, stop=tg[-1], num=len(tg), endpoint=True) @ N
Rex = (traj(A.build(A.lam0*np.exp(h), alpha)[0]) - traj(A.build(A.lam0*np.exp(-h), alpha)[0])) / (2*h)
CNN = E(N); sd = np.sqrt(np.maximum(2*(CNN[0] - CNN), 0))
for cname, B in [('B = N (hypothesised)', N), ('B = B^Y (estimated)', BYx)]:
    C = E(B); bias = Rex - (C[0] - C)
    for d in (0.1, 0.2):
        with np.errstate(divide='ignore', invalid='ignore'): m = ((1.96 + 0.84)*sd/(d*np.abs(bias)))**2
        i = np.nanargmin(m[1:]) + 1
        print(f"A {cname:22s} delta={d}: lag {tg[i]/tau:.2f} tau, |bias| {abs(bias[i]):.3f}, sd {sd[i]:.2f}, m = {m[i]:,.0f}, telemetry ~ {m[i]*(tg[i]+3*tau)/tau:,.0f} tau_int")
