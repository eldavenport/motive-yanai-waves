"""Depth-frequency power spectra of the mooring velocities.

For each MOTIVE mooring (A 0.5N,140W; B 1.75N,138W; C 3N,140W) and each model a
per-depth periodogram of the *total* velocity (window mean + unfiltered anomaly,
the repo's "total") is stacked into a depth x frequency map: y = depth, x =
frequency in cycles per year, color = power spectral density on a log scale
(white = low, blue = high). Two columns per figure: U (left), V (right). A few
log-spaced contour lines help read the levels; the 15-40 day Yanai band is marked.

One figure per mooring (mooring_<name>_spectrum_depth_freq.png -> spectra/).
Reuses the mooring cache + perturbation construction from plot_moorings.

Also (tpose24 only) a six-panel depth-frequency spectra+cospectra figure per
mooring and per treatment (mooring_<name>_cospectra_depth_freq_<treat>.png ->
spectra/): top row = auto-spectra of u',v',w' (log Blues); bottom row = the
signed co-spectra u'w',v'w',u'v' (Re of the cross-spectrum, RdBu_r, integrates
over frequency to <x'y'>). Each panel self-scales with clean ticks. `treat` is
'unfiltered' (total anomaly) or 'yanai' (15-40 day band-pass anomaly).

Record-length caveat: tpose24 is ~6 months and tpose6 ~10 months, so the
low-frequency (near-annual) end is coarse/under-resolved -- these resolve the
intraseasonal / Yanai band well, not the annual cycle.
"""

import os
import numpy as np
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm

import plot_moorings as pm
import wave_filter as wf

ZMIN, ZMAX = 300.0, 1400.0    # plotted depth range (m, positive down)
XMAX_CPY = 60.0               # frequency axis limit (cycles per year)
N_CONTOURS = 6                # log-spaced PSD contour lines overlaid
N_FILL = 100                  # log-spaced filled-contour levels (near-continuous)
N_DECADES = 4.0               # color range spans this many decades below the max
DAYS_PER_YEAR = 365.25


def depth_freq_psd(field, dt_days):
    """Per-depth periodogram of (time, depth) `field`.

    Returns f_cpy (nfreq,) in cycles/year and Pxx (nfreq, depth). DC (f=0) is
    dropped so the log axes/colour behave. All-NaN depth columns stay NaN.
    """
    f_cpd, Pxx = scipy.signal.periodogram(np.nan_to_num(field), fs=1.0 / dt_days,
                                          detrend='linear', axis=0)
    dead = ~np.isfinite(field).any(0)          # below-bathymetry columns
    Pxx[:, dead] = np.nan
    return f_cpd[1:] * DAYS_PER_YEAR, Pxx[1:]


def depth_freq_cospectrum(x, y, dt_days):
    """Per-depth cospectrum of two (time, depth) fields x, y.

    The cospectrum is the real part of the cross-spectral density (the quadrature
    spectrum, Im, is dropped); it is *signed* and integrates over frequency to the
    covariance <x'y'>. A full-length boxcar window (nperseg = record length) makes
    csd(x, x) identical to periodogram(x), so the auto- and co-spectra share one
    estimator. Returns f_cpy (nfreq,) and Co (nfreq, depth); DC dropped, dead
    columns NaN.
    """
    n = x.shape[0]
    f_cpd, Pxy = scipy.signal.csd(np.nan_to_num(x), np.nan_to_num(y),
                                  fs=1.0 / dt_days, detrend='linear', axis=0,
                                  window='boxcar', nperseg=n, noverlap=0)
    dead = ~(np.isfinite(x).any(0) & np.isfinite(y).any(0))
    Co = np.real(Pxy)
    Co[:, dead] = np.nan
    return f_cpd[1:] * DAYS_PER_YEAR, Co[1:]


def make_fig(M, p, dt_days):
    Z = M['Z']
    zmask = (-Z >= ZMIN) & (-Z <= ZMAX)
    zz = Z[zmask]
    name = M['names'][p]

    spec = {}
    for C, c in (('U', 'u'), ('V', 'v')):
        tot = M['mean'][C][p][None, :] + M['treat']['unfiltered'][c][:, p, :]
        f_cpy, Pxx = depth_freq_psd(tot, dt_days)
        spec[C] = Pxx[:, zmask]
    xsel = f_cpy <= XMAX_CPY

    pos = np.concatenate([spec[C][xsel][np.isfinite(spec[C][xsel])
                                        & (spec[C][xsel] > 0)] for C in spec])
    vmax = np.nanpercentile(pos, 99.5)
    vmin = vmax / 10 ** N_DECADES
    norm = LogNorm(vmin=vmin, vmax=vmax)
    fill_levels = np.logspace(np.log10(vmin), np.log10(vmax), N_FILL)
    clines = np.logspace(np.log10(vmin), np.log10(vmax), N_CONTOURS)
    yb = [DAYS_PER_YEAR / wf.YANAI_MAX_DAYS, DAYS_PER_YEAR / wf.YANAI_MIN_DAYS]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True,
                             layout='constrained')
    for ax, C in zip(axes, ('U', 'V')):
        S = np.clip(spec[C][xsel], vmin, vmax)
        pcm = ax.contourf(f_cpy[xsel], zz, S.T, levels=fill_levels, cmap='Blues',
                          norm=norm, extend='both')
        ax.contour(f_cpy[xsel], zz, S.T, levels=clines, colors='0.3',
                   linewidths=0.5)
        ax.axvspan(yb[0], yb[1], color='orange', alpha=0.12, zorder=0)
        for xb in yb:
            ax.axvline(xb, color='orange', lw=0.8, ls='--')
        ax.set_xlabel('frequency (cycles yr$^{-1}$)')
        ax.set_title(f'{C}', fontsize=11)
        ax.set_xlim(0, XMAX_CPY)
    axes[0].set_ylabel('depth (m)')
    axes[0].set_ylim(-ZMAX, -ZMIN)
    cb = fig.colorbar(pcm, ax=list(axes), pad=0.01, shrink=0.9)
    decades = 10.0 ** np.arange(np.ceil(np.log10(vmin)), np.floor(np.log10(vmax)) + 1)
    cb.set_ticks(decades)
    cb.ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    cb.set_label('PSD (m$^2$ s$^{-2}$ / cpy)')
    fig.suptitle(f'Velocity power spectra — {pm._moor_label(name)} '
                 f'(15-40 day Yanai band shaded)')
    fname = pm.fig_path(M['fig_dir'], f'mooring_{name}_spectrum_depth_freq.png')
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


def _yanai_band_cpy():
    """[low, high] frequency edges (cycles/yr) of the 15-40 day Yanai band."""
    return [DAYS_PER_YEAR / wf.YANAI_MAX_DAYS, DAYS_PER_YEAR / wf.YANAI_MIN_DAYS]


def _mark_yanai(ax, yb):
    ax.axvspan(yb[0], yb[1], color='orange', alpha=0.12, zorder=0)
    for xb in yb:
        ax.axvline(xb, color='orange', lw=0.8, ls='--')


def _nice_vmax(v):
    """Round v up to the nearest 1/2/5 x 10^n (clean colorbar limit)."""
    if not (v > 0):
        return 1.0
    exp = np.floor(np.log10(v))
    base = v / 10 ** exp
    nb = 1 if base <= 1 else 2 if base <= 2 else 5 if base <= 5 else 10
    return nb * 10.0 ** exp


def _auto_panel(fig, ax, f, zz, S):
    """Filled log-PSD panel (positive auto-spectrum), own Blues scale + decades.

    Each panel self-scales (w' is orders of magnitude weaker than u',v', so a
    shared scale would wash it out). Color spans N_DECADES below the 99.5th pct.
    """
    vmax = _nice_vmax(np.nanpercentile(S[np.isfinite(S) & (S > 0)], 99.5))
    vmin = vmax / 10 ** N_DECADES
    norm = LogNorm(vmin=vmin, vmax=vmax)
    Sc = np.clip(S, vmin, vmax).T
    levels = np.logspace(np.log10(vmin), np.log10(vmax), N_FILL)
    pcm = ax.contourf(f, zz, Sc, levels=levels, cmap='Blues', norm=norm,
                      extend='both')
    clines = np.logspace(np.log10(vmin), np.log10(vmax), N_CONTOURS)
    ax.contour(f, zz, Sc, levels=clines, colors='0.3', linewidths=0.5)
    cb = fig.colorbar(pcm, ax=ax, pad=0.02, shrink=0.9)
    # ticks only at integer powers of ten (LogFormatterMathtext labels no others)
    decades = 10.0 ** np.arange(np.ceil(np.log10(vmin)), np.floor(np.log10(vmax)) + 1)
    cb.set_ticks(decades)
    cb.ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    return cb


def _co_panel(fig, ax, f, zz, Co):
    """Filled signed-cospectrum panel, own symmetric RdBu_r scale, clean ticks.

    Self-scales (u'w',v'w' are far weaker than u'v'). The limit is rounded to a
    1/2/5 decade and the power of ten folded into the colorbar label, so the five
    ticks read as clean mantissas (as the mooring figures do).
    """
    v = _nice_vmax(np.nanpercentile(np.abs(Co[np.isfinite(Co)]), 99.0))
    Cc = np.clip(Co, -v, v).T
    levels = np.linspace(-v, v, N_FILL + 1)
    pcm = ax.contourf(f, zz, Cc, levels=levels, cmap='RdBu_r', extend='both')
    clines = np.linspace(-v, v, N_CONTOURS + 1)          # includes 0
    ax.contour(f, zz, Cc, levels=clines, colors='0.3', linewidths=0.5)
    ticks = np.linspace(-v, v, 5)
    cb = fig.colorbar(pcm, ax=ax, pad=0.02, shrink=0.9, ticks=ticks)
    exp = int(np.floor(np.log10(v)))
    if exp < -2 or exp >= 3:
        s = 10.0 ** exp
        cb.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda t, _, s=s: f'{t/s:g}'))
        cb.set_label(f'$\\times10^{{{exp}}}$', fontsize=8)
    else:
        cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda t, _: f'{t:g}'))
    return cb


def make_cospectra_fig(M, p, dt_days, treat):
    """Six-panel depth-frequency spectra/cospectra for one mooring & treatment.

    Top row: auto-spectra of u', v', w' (log Blues, shared scale). Bottom row:
    co-spectra u'w', v'w', u'v' (signed RdBu_r, shared symmetric scale; red =
    in-phase/positive covariance at that frequency, integrates to <x'y'>).
    y = depth, x = frequency (cycles/yr). `treat` is 'unfiltered' (total anomaly)
    or 'yanai' (15-40 day band-pass anomaly). -> spectra/
    """
    Z = M['Z']
    zmask = (-Z >= ZMIN) & (-Z <= ZMAX)
    zz = Z[zmask]
    name = M['names'][p]
    d = M['treat'][treat]
    u, v, w = d['u'][:, p, :], d['v'][:, p, :], d['w'][:, p, :]

    f_cpy, _ = depth_freq_psd(u, dt_days)
    xsel = f_cpy <= XMAX_CPY
    f = f_cpy[xsel]

    autos = {}
    for lab, fld in (("u'", u), ("v'", v), ("w'", w)):
        _, P = depth_freq_psd(fld, dt_days)
        autos[lab] = P[xsel][:, zmask]
    cos = {}
    for lab, a, b in ((r"u'w'", u, w), (r"v'w'", v, w), (r"u'v'", u, v)):
        _, C = depth_freq_cospectrum(a, b, dt_days)
        cos[lab] = C[xsel][:, zmask]

    yb = _yanai_band_cpy()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True, sharey=True,
                             layout='constrained')
    for ax, (lab, S) in zip(axes[0], autos.items()):
        _auto_panel(fig, ax, f, zz, S)
        ax.set_title(f'${lab}$ spectrum', fontsize=11)
    for ax, (lab, C) in zip(axes[1], cos.items()):
        _co_panel(fig, ax, f, zz, C)
        ax.set_title(f"${lab}$ cospectrum", fontsize=11)
    for ax in axes.ravel():
        _mark_yanai(ax, yb)
        ax.set_xlim(0, XMAX_CPY)
    for ax in axes[1]:
        ax.set_xlabel('frequency (cycles yr$^{-1}$)')
    for ax in axes[:, 0]:
        ax.set_ylabel('depth (m)')
    axes[0, 0].set_ylim(-ZMAX, -ZMIN)
    lbl = {'unfiltered': 'total anomaly', 'yanai': '15-40 day band'}[treat]
    fig.suptitle(f'Velocity spectra & cospectra — {pm._moor_label(name)} — '
                 f'{lbl} (Yanai band shaded)')
    fname = pm.fig_path(M['fig_dir'],
                        f'mooring_{name}_cospectra_depth_freq_{treat}.png')
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


def main(models=None):
    import importlib
    for model in (models or ['tpose24', 'tpose6']):
        cfg = pm.MODELS[model]
        if not os.path.exists(f"{importlib.import_module(cfg['io']).CACHE_DIR}/"
                              f"{cfg['cache']}"):
            print(f'SKIP {model}: cache not found ({cfg["cache"]})')
            continue
        os.makedirs(cfg['fig_dir'], exist_ok=True)
        M = pm.load_model(cfg)
        dt_days = float(np.median(np.diff(M['time'])) / np.timedelta64(1, 'D'))
        for p in range(len(M['names'])):
            print('wrote', make_fig(M, p, dt_days))
            # cospectra: tpose24 only for now (short records elsewhere)
            if model == 'tpose24':
                for treat in ('unfiltered', 'yanai'):
                    print('wrote', make_cospectra_fig(M, p, dt_days, treat))


def _selfcheck():
    """A pure sinusoid must peak at its own frequency in the right cpy bin."""
    dt_days = 1.0
    t = np.arange(2048)
    period_days = 20.0
    x = np.sin(2 * np.pi * t / period_days)[:, None] * np.ones((1, 3))
    x[:, 2] = np.nan                                   # dead column stays NaN
    f_cpy, Pxx = depth_freq_psd(x, dt_days)
    peak_cpy = f_cpy[np.argmax(Pxx[:, 0])]
    assert abs(peak_cpy - DAYS_PER_YEAR / period_days) < f_cpy[0], peak_cpy
    assert np.all(np.isnan(Pxx[:, 2]))

    # cospectrum: in-phase -> positive & == auto-PSD; quadrature -> ~0.
    y_inphase = (2.0 * np.sin(2 * np.pi * t / period_days))[:, None] * np.ones((1, 3))
    y_quad = (2.0 * np.cos(2 * np.pi * t / period_days))[:, None] * np.ones((1, 3))
    y_inphase[:, 2] = np.nan
    _, Co_x = depth_freq_cospectrum(x, x, dt_days)       # == periodogram(x)
    k = np.argmax(Pxx[:, 0])
    assert np.allclose(Co_x[k, 0], Pxx[k, 0]), (Co_x[k, 0], Pxx[k, 0])
    _, Co_in = depth_freq_cospectrum(x, y_inphase, dt_days)
    _, Co_q = depth_freq_cospectrum(x, y_quad, dt_days)
    assert Co_in[k, 0] > 0 and abs(Co_q[k, 0]) < 0.05 * Co_in[k, 0], \
        (Co_in[k, 0], Co_q[k, 0])
    assert np.all(np.isnan(Co_in[:, 2]))
    print('spectra self-check OK  (peak at %.2f cpy, expected %.2f; '
          'cospectrum in-phase=%.3g quad=%.3g)'
          % (peak_cpy, DAYS_PER_YEAR / period_days, Co_in[k, 0], Co_q[k, 0]))


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['test']:
        _selfcheck()
    else:
        main(sys.argv[1:] or None)
