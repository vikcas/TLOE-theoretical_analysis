"""Tables for §7 from study_<model>.json: medians/p90 with bootstrap CIs, rejection rates with Wilson 95% CIs."""
import json, numpy as np, sys
rng = np.random.default_rng(0)
def wilson(x, n, z=1.96):
    p = x / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, max(0, c - h), min(1, c + h)
def boot(v, f, B=2000):
    v = np.asarray(v); s = [f(v[rng.integers(0, len(v), len(v))]) for _ in range(B)]; return f(v), np.quantile(s, .025), np.quantile(s, .975)
fmtp = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
fmte = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
pre = sys.argv[1] if len(sys.argv) > 1 else '../results/study_'
R = {m: json.load(open(f'{pre}{m}.json')) for m in ('J', 'B3', 'B2', 'A')}
KS = ['10', '30', '100', '300', '1000', '3000', '10000']
print("model tau bias lags block n:", {m: (round(R[m]['tau'], 2), round(R[m]['bias'], 3), R[m]['lags'], R[m]['block'], len(R[m]['by_k']['10']['z'])) for m in R})
print("\n### y: median [95% CI] and p90 [95% CI]")
for m in R:
    for k in KS:
        e = R[m]['by_k'][k]['ey']; print(f"{m:3s} k={k:>5}: median {fmte(boot(e, np.median))}  p90 {fmte(boot(e, lambda v: np.quantile(v, .9)))}")
print("\n### T-R: rejection rate [Wilson 95%], oracle power [Wilson], baseline cross-moment rejection [Wilson]; mean bootstrap threshold; J empirical q95")
for k in KS:
    zJ = np.array(R['J']['by_k'][k]['z']); thr = np.quantile(zJ, 0.95)
    for m in R:
        d = R[m]['by_k'][k]; z, q, zc, qc = map(np.array, (d['z'], d['q'], d['zc'], d['qc'])); n = len(z)
        rej = wilson((z > q).sum(), n); orc = wilson((z > thr).sum(), n); cm = wilson((zc > qc).sum(), n)
        print(f"k={k:>5} {m:3s}: TR {fmtp(rej)} | oracle {fmtp(orc)} | CM {fmtp(cm)} | mean q {q[np.isfinite(q)].mean():.2f} | J q95 {thr:.2f} | min nused {min(d['nused'])}")
for m in ('J', 'B3'):
    if 'sens' not in R[m]: continue
    print(f"\n### sensitivity ({m}, 200 replicates): rejection rate [Wilson]")
    row = []
    for k in ('300', '1000', '3000'):
        d = R[m]['by_k'][k]; z, q = np.array(d['z'][:200]), np.array(d['q'][:200]); row.append(f"k={k}: {fmtp(wilson((z > q).sum(), 200))}")
    print("main    " + " | ".join(row))
    for name, out in R[m]['sens'].items():
        row = []
        for k in ('300', '1000', '3000'):
            a = np.array(out[k]); row.append(f"k={k}: {fmtp(wilson((a[:,0] > a[:,1]).sum(), len(a)))}")
        print(f"{name:7s} " + " | ".join(row))
