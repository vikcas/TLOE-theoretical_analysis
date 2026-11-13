"""Evidence figure: (a) model J, response of n2 to a load step: exact vs FDT predictions with three conjugates;
(b) model B (blocking), response of N: exact vs FDT prediction with B = N."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm_multiply
from exact import build, stationary, lam, mu1, mu2, K1
h = 1e-5; tg = np.linspace(0, 40, 161)
def curves(K2, A_name):
    Q, n1, n2 = build(lam, mu1, mu2, K2); pi = stationary(Q); N = n1 + n2
    A = {'n2': n2, 'N': N}[A_name]
    traj = lambda Q_: expm_multiply(Q_.T, pi, start=0, stop=tg[-1], num=len(tg), endpoint=True) @ A
    ex = (traj(build(lam*np.exp(h), mu1, mu2, K2)[0]) - traj(build(lam*np.exp(-h), mu1, mu2, K2)[0])) / (2*h)
    C = lambda B: expm_multiply(Q.T, pi*B, start=0, stop=tg[-1], num=len(tg), endpoint=True) @ A - (pi@A)*(pi@B)
    st = lambda B: C(B)[0] - C(B)
    return ex, {'N': st(N), 'n2': st(n2), 'n1': st(n1)}
ink, ink2, grid = '#0b0b0b', '#52514e', '#e4e3df'
c1, c2, c3 = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': ink2, 'axes.labelcolor': ink, 'xtick.color': ink2,
                     'ytick.color': ink2, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
ex, p = curves(K1, 'n2'); a = ax[0]
a.plot(tg, ex, color=ink, lw=2.6, label='exact')
a.plot(tg, p['N'], color=c1, lw=2, ls=(0, (4, 2)), label='FDT, B = N (global)')
a.plot(tg, p['n2'], color=c2, lw=2, label='FDT, B = n₂ (per-stage)')
a.plot(tg, p['n1'], color=c3, lw=2, label='FDT, B = n₁')
a.axhline(0, color=grid, lw=1, zorder=0)
a.set_title('(a) Jackson tandem: response of n₂ to δ ln λ', fontsize=9, loc='left', color=ink)
a.set_xlabel('time after the step'); a.set_ylabel('response per unit δ ln λ'); a.legend(frameon=False, fontsize=7.5, loc='center right')
ex, p = curves(3, 'N'); a = ax[1]
exJ, pJ = curves(K1, 'N')
a.plot(tg, ex, color=ink, lw=2.6, label='exact, blocking')
a.plot(tg, p['N'], color=c1, lw=2, ls=(0, (4, 2)), label='FDT, B = N, blocking')
a.plot(tg, exJ, color=ink2, lw=1.2, ls=':', label='exact = FDT, Jackson')
a.set_title('(b) Blocking (buffer 3): response of N to δ ln λ', fontsize=9, loc='left', color=ink)
a.set_xlabel('time after the step'); a.legend(frameon=False, fontsize=7.5, loc='lower right')
for a in ax:
    a.grid(axis='y', color=grid, lw=0.6); a.set_xlim(0, 40)
    for tv in (1, 4): a.axvline(tv, color=ink2, lw=0.8, ls=(0, (1, 2)), zorder=0)
    a.text(1, 1.0, ' t=1', transform=a.get_xaxis_transform(), fontsize=7, color=ink2, va='top'); a.text(4, 1.0, ' t=4', transform=a.get_xaxis_transform(), fontsize=7, color=ink2, va='top')
out = sys.argv[1] if len(sys.argv) > 1 else 'fig_evidence'
fig.savefig(out + '.pdf'); fig.savefig(out + '.png', dpi=200)
