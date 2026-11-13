"""Order-of-magnitude sample size for a dynamic T-F test, for the two conjugates:
 A: hypothesised B = N;  B: estimated (top-level) B^Y = d ln pi^N / d ln lambda.
m = ((1.96+0.84) sd / (delta |bias(t)|))^2, sd = sqrt(2(Var N - C_{N,N}(t))), optimised over t in (0, 4 tau_int]."""
import numpy as np
from scipy.sparse.linalg import expm_multiply
from exact import build, stationary, reference, lam, mu1, mu2
h = 1e-5
for K2, lab in [(3, 'B3'), (2, 'B2')]:
    tau = reference(K2, np.array([0., 1.]))['tau_int']; tg = np.linspace(0, 4 * tau, 401)
    Q, n1, n2 = build(lam, mu1, mu2, K2); pi = stationary(Q); N = n1 + n2; Ni = N.astype(int)
    marg = lambda p: np.bincount(Ni, weights=p)
    BY = (np.log(marg(stationary(build(lam*np.exp(h), mu1, mu2, K2)[0]))) - np.log(marg(stationary(build(lam*np.exp(-h), mu1, mu2, K2)[0])))) / (2*h)
    BYx = BY[Ni] - pi @ BY[Ni]
    E = lambda B: expm_multiply(Q.T, pi * B, start=0, stop=tg[-1], num=len(tg), endpoint=True) @ N - (pi @ N) * (pi @ B)
    traj = lambda Q_: expm_multiply(Q_.T, pi, start=0, stop=tg[-1], num=len(tg), endpoint=True) @ N
    Rex = (traj(build(lam*np.exp(h), mu1, mu2, K2)[0]) - traj(build(lam*np.exp(-h), mu1, mu2, K2)[0])) / (2*h)
    CNN = E(N); sd = np.sqrt(np.maximum(2 * (CNN[0] - CNN), 0))
    for cname, B in [('B = N (hypothesised)', N), ('B = B^Y (estimated)', BYx)]:
        C = E(B); bias = Rex - (C[0] - C)
        for d in (0.1, 0.2):
            with np.errstate(divide='ignore', invalid='ignore'): m = ((1.96 + 0.84) * sd / (d * np.abs(bias)))**2
            i = np.nanargmin(m[1:]) + 1
            print(f"{lab} {cname:22s} delta={d}: lag {tg[i]/tau:.2f} tau, |bias| {abs(bias[i]):.3f}, sd {sd[i]:.2f}, m = {m[i]:,.0f}, telemetry ~ {m[i]*(tg[i]+3*tau)/tau:,.0f} tau_int")
