"""Evidence figure for the memoria, with labels for computer scientists.
Same computation as repo/exact/fig_evidence.py; only labels and layout change.
Usage: python -I fig_memoria.py <path-to-exact.py> <output-stem>"""
import sys, importlib.util, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm_multiply
spec = importlib.util.spec_from_file_location("exact", sys.argv[1]); ex_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex_mod)
build, stationary, lam, mu1, mu2, K1 = ex_mod.build, ex_mod.stationary, ex_mod.lam, ex_mod.mu1, ex_mod.mu2, ex_mod.K1
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
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
ex, p = curves(K1, 'n2'); a = ax[0]
i1 = int(np.argmin(abs(tg-1)))
print(f"(a) stage 2 at t=1: true {ex[i1]:.3f}, from total {p['N'][i1]:.3f}, from stage-2 queue {p['n2'][i1]:.3f}, from stage-1 queue {p['n1'][i1]:.3f}; final true {ex[-1]:.3f}")
a.plot(tg, ex, color=ink, lw=2.6, label='true response')
a.plot(tg, p['N'], color=c1, lw=2, ls=(0, (4, 2)), label='predicted from total requests')
a.plot(tg, p['n2'], color=c2, lw=2, label='predicted from stage-2 queue')
a.plot(tg, p['n1'], color=c3, lw=2, label='predicted from stage-1 queue')
a.axhline(0, color=grid, lw=1, zorder=0)
a.set_title('(a) Two services in a chain', fontsize=9, loc='left', color=ink)
a.set_ylabel('extra requests at stage 2')
a.legend(frameon=False, fontsize=7.5, loc='center right')
ex, p = curves(3, 'N'); a = ax[1]
exJ, pJ = curves(K1, 'N')
print(f"(b) total at t=1: true {ex[i1]:.3f}, predicted {p['N'][i1]:.3f}; at t=40: true {ex[-1]:.3f}, predicted {p['N'][-1]:.3f}")
a.plot(tg, ex, color=ink, lw=2.6, label='true response, with blocking')
a.plot(tg, p['N'], color=c1, lw=2, ls=(0, (4, 2)), label='predicted, with blocking')
a.plot(tg, exJ, color=ink2, lw=1.2, ls=':', label='without blocking (true = predicted)')
a.set_title('(b) Same chain with a buffer of 3 at stage 2', fontsize=9, loc='left', color=ink)
a.set_ylabel('extra requests in the chain')
a.legend(frameon=False, fontsize=7.5, loc='lower right')
for a in ax:
    a.grid(axis='y', color=grid, lw=0.6); a.set_xlim(0, 40)
    a.set_xlabel('time after the load increase')
out = sys.argv[2]
fig.savefig(out + '.png', dpi=220); fig.savefig(out + '.pdf')
