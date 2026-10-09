"""Depth-frequency power spectra of the mooring velocities.

For each MOTIVE mooring (A 0.5N,140W; B 1.75N,138W; C 3N,140W) and each model a
per-depth periodogram of the *total* velocity (window mean + unfiltered anomaly,
the repo's "total") is stacked into a depth x frequency map: y = depth, x =
frequency in cycles per year, color = power spectral density on a log scale
(white = low, blue = high). Two columns per figure: U (left), V (right). A few
log-spaced contour lines help read the levels; the 15-40 day Yanai band is marked.

One figure per mooring (mooring_<name>_spectrum_depth_freq.png -> spectra/).
Reuses the mooring cache + perturbation construction from plot_moorings.

Record-length caveat: tpose24 is ~3 months and tpose6 ~10 months, so the
low-frequency (near-annual) end is coarse/under-resolved -- these resolve the
intraseasonal / Yanai band well, not the annual cycle. The noVel 5-year run is
the one to use for a true cycles-per-year spectrum (not drawn here by request).
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

ZMIN, ZMAX = 0.0, 1500.0      # plotted depth range (m, positive down)
XMAX_CPY = 60.0               # frequency axis limit (cycles per year)
N_CONTOURS = 6                # log-spaced PSD contour lines overlaid
N_FILL = 20                   # log-spaced filled-contour levels
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
    print('spectra self-check OK  (peak at %.2f cpy, expected %.2f)'
          % (peak_cpy, DAYS_PER_YEAR / period_days))


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['test']:
        _selfcheck()
    else:
        main(sys.argv[1:] or None)
