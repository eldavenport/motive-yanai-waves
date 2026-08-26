"""Correlation statistics for the shear vs. momentum-flux-divergence analysis.

Two flavours of correlation are needed (see the plan / notebook):

  depth-structure  -- sample dimension is DEPTH. Correlate a time-mean divergence
                      profile against a shear profile down the water column. Depth
                      levels are strongly autocorrelated, so a naive p-value is
                      invalid; `depth_corr` returns Pearson & Spearman r plus a
                      first-pass surrogate p from phase-randomised profiles that
                      preserve the vertical autocorrelation.

  temporal         -- sample dimension is TIME, at each depth. Band-passed series
                      are strongly autocorrelated, so `temporal_corr` returns r
                      with an effective-DOF (lag-1) estimate and a provisional p.

NOTE: the choice of significance test is deliberately left OPEN. The p-values here
are a first-pass guide only; the appropriate test will be decided after seeing the
correlation results. The r values are the primary output.
"""

import numpy as np
from scipy.stats import spearmanr, t as tdist


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _finite_pair(a, b):
    """Flatten to matching finite pairs; returns (a2, b2) 1-D float64."""
    a = np.asarray(a, dtype='f8').ravel()
    b = np.asarray(b, dtype='f8').ravel()
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def lag1_autocorr(x):
    """Lag-1 autocorrelation of a 1-D series (finite values), 0 if undefined."""
    x = np.asarray(x, dtype='f8')
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom == 0:
        return 0.0
    return float(np.sum(x[1:] * x[:-1]) / denom)


# --------------------------------------------------------------------------- #
# depth-structure correlation
# --------------------------------------------------------------------------- #
def _phase_randomize(x, rng):
    """Phase-randomised surrogate of a 1-D series (preserves the power spectrum,
    hence the autocorrelation). Real-valued output, same length as x."""
    n = x.size
    X = np.fft.rfft(x - x.mean())
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=X.shape))
    phases[0] = 1.0
    if n % 2 == 0:                      # keep Nyquist real
        phases[-1] = 1.0
    surr = np.fft.irfft(np.abs(X) * phases, n=n)
    return surr + x.mean()


def depth_corr(prof_a, prof_b, n_surro=2000, seed=0):
    """Depth-structure correlation between two vertical profiles.

    Returns dict with Pearson r, Spearman r, and a first-pass two-sided surrogate
    p (fraction of phase-randomised surrogates of `prof_a` whose |Pearson r| with
    `prof_b` meets or exceeds the observed |r|). n is the number of finite depth
    pairs used. Surrogate p is provisional -- significance methodology TBD.
    """
    a, b = _finite_pair(prof_a, prof_b)
    out = {'n': int(a.size), 'r_pearson': np.nan, 'r_spearman': np.nan,
           'p_surrogate': np.nan}
    if a.size < 4 or a.std() == 0 or b.std() == 0:
        return out
    r = float(np.corrcoef(a, b)[0, 1])
    out['r_pearson'] = r
    out['r_spearman'] = float(spearmanr(a, b).statistic)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_surro):
        sa = _phase_randomize(a, rng)
        if sa.std() == 0:
            continue
        rs = np.corrcoef(sa, b)[0, 1]
        if abs(rs) >= abs(r):
            hits += 1
    out['p_surrogate'] = (hits + 1) / (n_surro + 1)
    return out


# --------------------------------------------------------------------------- #
# temporal correlation
# --------------------------------------------------------------------------- #
def temporal_corr(a, b):
    """Temporal correlation between two 1-D time series with effective DOF.

    Returns dict: r (Pearson), n (finite pairs), n_eff (effective sample size from
    the lag-1 autocorrelations, Neff = n*(1-a1*b1)/(1+a1*b1)), and a provisional
    two-sided p from a t-distribution with n_eff-2 dof. p is a first-pass guide;
    significance methodology is TBD.
    """
    a, b = _finite_pair(a, b)
    out = {'r': np.nan, 'n': int(a.size), 'n_eff': np.nan, 'p': np.nan}
    if a.size < 4 or a.std() == 0 or b.std() == 0:
        return out
    r = float(np.corrcoef(a, b)[0, 1])
    out['r'] = r
    a1, b1 = lag1_autocorr(a), lag1_autocorr(b)
    factor = (1 - a1 * b1) / (1 + a1 * b1)
    n_eff = max(3.0, a.size * min(1.0, max(factor, 1e-3)))
    out['n_eff'] = float(n_eff)
    if abs(r) < 1.0:
        tstat = r * np.sqrt((n_eff - 2) / (1 - r * r))
        out['p'] = float(2 * tdist.sf(abs(tstat), df=n_eff - 2))
    else:
        out['p'] = 0.0
    return out


def temporal_corr_profile(A, B, taxis=0):
    """Temporal correlation as a function of depth.

    A, B have time on `taxis` and depth on the last axis. Returns dict of profiles
    {'r','n_eff','p'} (one value per depth level) from `temporal_corr`.
    """
    A = np.moveaxis(np.asarray(A, dtype='f8'), taxis, 0)
    B = np.moveaxis(np.asarray(B, dtype='f8'), taxis, 0)
    nz = A.shape[-1]
    r = np.full(nz, np.nan)
    neff = np.full(nz, np.nan)
    p = np.full(nz, np.nan)
    for k in range(nz):
        res = temporal_corr(A[..., k], B[..., k])
        r[k], neff[k], p[k] = res['r'], res['n_eff'], res['p']
    return {'r': r, 'n_eff': neff, 'p': p}
