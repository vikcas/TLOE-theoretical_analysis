"""Finite-sample study (revised). Usage: python3 study2.py MODEL [NCHUNK=5] [SENS=0]
Per replicate and window k*tau_int: prediction error y (B = N), T-R max-|z| statistic and its bootstrap 95% threshold,
and a baseline cross-moment test (standardised N and N^2) with the same bootstrap. Components with fewer than MINC events
or a degenerate bootstrap s.e. are excluded (fixes the k = 10 degeneracy of the first version)."""
import numpy as np, json, sys, time
from models import MODELS
DT = 0.5; NBOOT = 300; RCH = 200; MINC = 10; SEFLOOR = 1e-12
KS = [10, 30, 100, 300, 1000, 3000, 10000]
SEEDS = {'J': 1000, 'B3': 3000, 'B2': 2000, 'A': 4000}

def acov(x, maxlag):
    x = x - x.mean(); n = len(x); f = np.fft.rfft(x, 2 * n)
    return np.fft.irfft(f * np.conj(f))[:maxlag + 1] / n

def boot_maxz(A, nblk, n, rng, extra_ok=None):
    """A: nblk x m matrix of per-block sums of the antisymmetric statistics. Returns (max|z|, bootstrap 95% threshold, #used)."""
    Dhat = A.sum(0) / n
    W = rng.multinomial(nblk, np.full(nblk, 1.0 / nblk), size=NBOOT).astype(float)
    Dst = (W @ A) / n; se = Dst.std(0, ddof=1)
    ok = se > SEFLOOR
    if extra_ok is not None: ok &= extra_ok
    if not ok.any(): return 0.0, np.inf, 0
    z = np.abs(Dhat[ok] / se[ok]).max()
    zst = np.abs((Dst[:, ok] - Dhat[ok]) / se[ok]).max(1)
    return float(z), float(np.quantile(zst, 0.95)), int(ok.sum())

def tr_test(x, lags, blk, rng, nb=13):
    b = np.minimum(x, nb - 1).astype(np.int64); L = max(lags)
    n = (len(b) - L) // blk * blk; nblk = n // blk
    if nblk < 2: return 0.0, np.inf, 0
    iu = np.triu_indices(nb, 1); blocks = np.repeat(np.arange(nblk), blk)
    As, cnts = [], []
    for lag in lags:
        c = b[:n] * nb + b[lag:lag + n]
        cnt = np.bincount(blocks * nb * nb + c, minlength=nblk * nb * nb).reshape(nblk, nb, nb)
        As.append((cnt - cnt.transpose(0, 2, 1))[:, iu[0], iu[1]])
        tot = cnt.sum(0); cnts.append((tot + tot.T)[iu])
    return boot_maxz(np.concatenate(As, 1).astype(float), nblk, n, rng, np.concatenate(cnts) >= MINC)

def cm_test(x, lags, blk, rng):
    """baseline: antisymmetric cross-moment E[u(N_0) v(N_t)] - E[v(N_0) u(N_t)], u, v = standardised N, N^2"""
    x = x.astype(float); L = max(lags); n = (len(x) - L) // blk * blk; nblk = n // blk
    if nblk < 2: return 0.0, np.inf, 0
    u = (x - x.mean()) / (x.std() + 1e-300); w = x ** 2; v = (w - w.mean()) / (w.std() + 1e-300)
    cols = [(u[:n] * v[lag:lag + n] - v[:n] * u[lag:lag + n]).reshape(nblk, blk).sum(1) for lag in lags]
    return boot_maxz(np.stack(cols, 1), nblk, n, rng)

def run(model, nchunk, sens):
    ref, sim = MODELS[model]
    tau = ref(np.array([0.0, 1.0]))['tau']; maxlag = int(np.ceil(3 * tau / DT))
    r = ref(np.arange(maxlag + 1) * DT); Rex, chi = r['R_exact'], r['chi']
    bias = float(np.abs(r['R_fdt'] - Rex).max() / abs(chi))
    lags = sorted({max(1, int(round(f * tau / DT))) for f in (1/16, 1/8, 1/4)})
    blk = int(np.ceil(2 * tau / DT)) + max(lags)
    res = dict(model=model, tau=tau, chi=chi, bias=bias, lags=[l * DT for l in lags], block=blk * DT, by_k={str(k): dict(ey=[], z=[], q=[], nused=[], zc=[], qc=[]) for k in KS})
    if sens: res['sens'] = {}
    for ch in range(nchunk):
        t0 = time.time(); X = sim(KS[-1] * tau, DT, RCH, SEEDS[model] + ch); rng = np.random.default_rng(SEEDS[model] + 100 + ch)
        for k in KS:
            nk = int(round(k * tau / DT)) + 1; d = res['by_k'][str(k)]
            for i in range(RCH):
                x = X[i, :nk]
                c = acov(x.astype(float), maxlag); d['ey'].append(float(np.abs((c[0] - c) - Rex).max() / abs(chi)))
                z, q, nu = tr_test(x, lags, blk, rng); d['z'].append(z); d['q'].append(q); d['nused'].append(nu)
                zc, qc, _ = cm_test(x, lags, blk, rng); d['zc'].append(zc); d['qc'].append(qc)
        if sens and ch == 0:
            for name, kw in [('dt2', dict(sub=4)), ('dt8', dict(sub=16)), ('bins7', dict(nb=7)), ('bins21', dict(nb=21)), ('block6', dict(bm=6))]:
                out = {}
                for k in (300, 1000, 3000):
                    nk = int(round(k * tau / DT)) + 1; sub = kw.get('sub', 1); dtk = DT * sub
                    lg = sorted({max(1, int(round(f * tau / dtk))) for f in (1/16, 1/8, 1/4)})
                    bk = int(np.ceil(kw.get('bm', 2) * tau / dtk)) + max(lg)
                    zz = [tr_test(X[i, :nk:sub], lg, bk, rng, kw.get('nb', 13))[:2] for i in range(RCH)]
                    out[str(k)] = [list(t) for t in zz]
                res['sens'][name] = out
        print(f"{model}: chunk {ch+1}/{nchunk} done ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(f'../results/study_{model}.json', 'w'))
    return res

if __name__ == "__main__":
    m = sys.argv[1]; nchunk = int(sys.argv[2]) if len(sys.argv) > 2 else 5; sens = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    run(m, nchunk, sens)
