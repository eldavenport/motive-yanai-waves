"""Mooring-simulation depth-time plots at the three MOTIVE mooring points.

For each mooring (A 0.5N,140W; B 1.75N,138W; C 3N,140W) and each model this
draws two figures of stacked depth-time (Hovmoller) panels:

  figure "uw":  u'w' (top), u' (middle), w' (bottom)
  figure "vw":  v'w' (top), v' (middle), w' (bottom)

y-axis = depth, x-axis = time. Each figure is drawn twice in the frequency
treatment -- **unfiltered** (perturbation = anomaly from the analysis-window
mean, i.e. the repo's "total") and **yanai** (15-40 day band-pass, the same
filtering used elsewhere in this repo) -- and twice in depth range (0-800 m and
300-1500 m). So 2 comps x 2 treatments x 2 depth ranges = 8 figures per mooring.

The perturbation velocities are formed exactly as in the shear/divergence
notebooks: TPOSE24 drops the first SPINUP_DAYS before filtering and centers WVEL
from faces; TPOSE6 uses the whole record with WVEL already centered in its cache.
Both apply the 20-day band-pass edge trim to define a common analysis window.

Reads the caches from build_mooring_cache.py / build_mooring_cache_t6.py.
Figures -> figures/ (TPOSE24) and figures/tpose6/ (TPOSE6).
"""

import os
import importlib
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# number of filled-contour levels for the depth-time panels
N_LEVELS = 100

import wave_filter as wf
import fluxes as fx
import shear as sh

REPO = '/home/edavenport/analysis/motive-yanai-waves'

MODELS = {
    'tpose24': dict(io='tpose24_io', cache='yanai_mooring_dt60.nc',
                    meanuv='yanai_meanUV_maps_dt60.nc',
                    fig_dir=f'{REPO}/figures', spinup=True, w_on_faces=True),
    'tpose6':  dict(io='tpose6_io', cache='yanai_mooring_tp6.nc',
                    meanuv='yanai_meanUV_maps_tp6.nc',
                    fig_dir=f'{REPO}/figures/tpose6', spinup=False,
                    w_on_faces=False),
}

# depth ranges (m, positive down) requested for the panels
DEPTH_RANGES = [(0, 800, '0-800m'), (300, 1500, '300-1500m')]

# fixed symmetric color limits for the velocity panels (m s^-1), per treatment;
# the u'w'/v'w' product panels keep per-panel 98th-percentile auto-scaling.
FIXED_CLIM = {
    'unfiltered': {'u': 0.75, 'v': 0.75, 'w': 1e-3},
    'yanai':      {'u': 0.75, 'v': 0.75, 'w': 1e-4},
}

# depth range (m, positive down) for the mean-profile figures -- cropped below
# the EUC / thermocline so the deep-jet structure sets the scale, not the EUC.
PROFILE_ZMIN = 300.0
PROFILE_ZMAX = 1500.0

# per-mooring line colors for the overlaid profile figures (A, B, C)
MOORING_COLORS = ['black', 'blue', 'red']

# figure layouts: label + which field key each row shows
FIGS = {
    'uw': [(r"$u'w'$ (m$^2$ s$^{-2}$)", 'uw'),
           (r"$u'$ (m s$^{-1}$)", 'u'),
           (r"$w'$ (m s$^{-1}$)", 'w')],
    'vw': [(r"$v'w'$ (m$^2$ s$^{-2}$)", 'vw'),
           (r"$v'$ (m s$^{-1}$)", 'v'),
           (r"$w'$ (m s$^{-1}$)", 'w')],
}


def load_model(cfg):
    """Load a mooring cache and build the unfiltered/yanai perturbations."""
    io = importlib.import_module(cfg['io'])
    ds = xr.open_dataset(f"{io.CACHE_DIR}/{cfg['cache']}")
    Z = ds['depth'].values
    U, V, W = ds['UVEL'].values, ds['VVEL'].values, ds['WVEL'].values
    times = ds['time'].values
    names = [str(x) for x in ds['loc'].values]
    lat, lon = ds['lat'].values, ds['lon'].values
    ds.close()

    if cfg['spinup']:                                # drop spin-up BEFORE filtering
        cut = (np.datetime64(io.REF_DATE)
               + np.timedelta64(int(io.SPINUP_DAYS * 86400), 's'))
        keep = times > cut
        times, U, V, W = times[keep], U[keep], V[keep], W[keep]
        dt_days = io.OUT_STEP_SEC / 86400.0
    else:
        dt_days = 1.0
    Wc = fx.w_to_center(W, zaxis=-1) if cfg['w_on_faces'] else W

    etm = io.edge_trim_mask(times)                   # drop band-pass edge days
    Ut, Vt, Wt = U[etm], V[etm], Wc[etm]
    unfilt = {'u': Ut - np.nanmean(Ut, 0, keepdims=True),
              'v': Vt - np.nanmean(Vt, 0, keepdims=True),
              'w': Wt - np.nanmean(Wt, 0, keepdims=True)}
    yanai = {'u': wf.bandpass(U, axis=0, dt_days=dt_days)[etm],
             'v': wf.bandpass(V, axis=0, dt_days=dt_days)[etm],
             'w': wf.bandpass(Wc, axis=0, dt_days=dt_days)[etm]}

    mean = {'U': np.nanmean(Ut, 0), 'V': np.nanmean(Vt, 0)}   # (loc, depth)
    return dict(Z=Z, time=times[etm], names=names, lat=lat, lon=lon,
                fig_dir=cfg['fig_dir'], mean=mean,
                treat={'unfiltered': unfilt, 'yanai': yanai})


def _moor_label(name):
    return f"{name[0]} ({name.split('_', 1)[1].replace('_', ' ')})"


def make_profile_fig(M, comp, subset=None, tag=''):
    """Three-column mean-profile figure for velocity component `comp` ('U'/'V').

    col1: time-mean velocity <comp>; col2: mean shear d<comp>/dz (dotted, top
    axis) and the vertical Reynolds stress -<comp'w'> (solid, bottom axis);
    col3: the shear production -<comp'w'> d<comp>/dz. The moorings in `subset`
    (indices into M['names']; default all) are overlaid in their fixed colors.
    Unfiltered (total) perturbations only.
    """
    vk = comp.lower()                                 # 'u' or 'v'
    Z = M['Z']
    zmask = (-Z >= PROFILE_ZMIN) & (-Z <= PROFILE_ZMAX)
    zz = Z[zmask]
    d = M['treat']['unfiltered']
    subset = range(len(M['names'])) if subset is None else subset

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 6.5), sharey=True)
    ax1t = ax1.twiny()
    for p in subset:
        name = M['names'][p]
        c = MOORING_COLORS[p]
        vel = M['mean'][comp][p]                      # (depth,)
        S = sh.vertical_shear(vel, Z)                 # d<comp>/dz
        F = -np.nanmean(d[vk][:, p, :] * d['w'][:, p, :], 0)   # -<comp'w'>
        P = F * S                                     # shear production
        ax0.plot(vel[zmask], zz, c, label=_moor_label(name))
        ax1.plot(F[zmask], zz, c, ls='-')             # solid: stress
        ax1t.plot(S[zmask], zz, c, ls=':')            # dotted: shear
        ax2.plot(P[zmask], zz, c)

    for ax in (ax0, ax1, ax2):
        ax.axvline(0, color='0.6', lw=0.5)
    ax0.set_title(f'mean $\\langle {comp}\\rangle$', fontsize=10)
    ax0.set_xlabel(f'$\\langle {comp}\\rangle$ (m s$^{{-1}}$)')
    ax0.set_ylabel('depth (m)')
    ax0.legend(fontsize=8, loc='lower left')
    ax1.set_title(f"$-\\langle {vk}'w'\\rangle$ (solid) & "
                  f"$\\partial_z\\langle {comp}\\rangle$ (dotted)", fontsize=10)
    ax1.set_xlabel(f"$-\\langle {vk}'w'\\rangle$ (m$^2$ s$^{{-2}}$)")
    ax1t.set_xlabel(f"$\\partial_z\\langle {comp}\\rangle$ (s$^{{-1}}$)")
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax1t.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax2.set_title(f"production $-\\langle {vk}'w'\\rangle\\,"
                  f"\\partial_z\\langle {comp}\\rangle$", fontsize=10)
    ax2.set_xlabel('production (m$^2$ s$^{-3}$)')
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax0.set_ylim(-PROFILE_ZMAX, -PROFILE_ZMIN)
    fig.suptitle(f'Mooring mean profiles — {comp} — unfiltered '
                 f'({PROFILE_ZMIN:.0f}-{PROFILE_ZMAX:.0f} m)', y=0.99)
    fig.tight_layout()
    fname = f"{M['fig_dir']}/mooring_profiles_{comp}_unfiltered{tag}.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


def restrict_unfiltered(M, t0, t1):
    """Re-reference the unfiltered perturbations and mean flow to [t0, t1].

    The cached `unfiltered` perturbation is an anomaly from the FULL analysis
    window; here we restrict to the sub-window t0..t1 and re-subtract the
    sub-window mean (so the covariance is taken over exactly this window). The
    stored per-mooring mean velocity is likewise shifted by the sub-window mean
    of the anomaly, so `mean[comp]` is the time-mean over t0..t1. No raw
    velocities are needed -- everything follows from the stored anomaly + mean.
    """
    tmask = (M['time'] >= t0) & (M['time'] <= t1)
    d = M['treat']['unfiltered']
    unfilt, mean = {}, {}
    for vk in ('u', 'v', 'w'):
        p = d[vk][tmask]
        unfilt[vk] = p - np.nanmean(p, 0, keepdims=True)
    for comp in ('U', 'V'):
        mean[comp] = M['mean'][comp] + np.nanmean(d[comp.lower()][tmask], 0)
    return dict(mean=mean, unfilt=unfilt, n=int(tmask.sum()))


def make_profile_compare(M, comp, subset=(0, 2)):
    """Overlay A & C mean profiles from tpose24 and tpose6 over the tpose24 window.

    Four columns, each its own x-axis: mean velocity <comp>, mean shear
    d<comp>/dz, vertical Reynolds stress -<comp'w'>, and shear production
    -<comp'w'> d<comp>/dz. Both models are restricted to the tpose24 record so
    the comparison isolates dynamics from record length. Mooring is encoded by
    color (A black, C red); model by line style (tpose24 solid, tpose6 dashed).
    No titles. Unfiltered (total) perturbations only.
    """
    from matplotlib.lines import Line2D
    M24 = M['tpose24']
    t0, t1 = M24['time'][0], M24['time'][-1]
    vk = comp.lower()
    # model -> line style: tpose24 solid, tpose6 dashed (mooring is the color)
    styles = {'tpose24': dict(ls='-', lw=2.2),
              'tpose6':  dict(ls='--', lw=1.3)}
    ns = {}

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(14, 6.5),
                                             sharey=True)
    for mname in ('tpose24', 'tpose6'):
        Mi = M[mname]
        r = restrict_unfiltered(Mi, t0, t1)
        ns[mname] = r['n']
        Z = Mi['Z']
        zmask = (-Z >= PROFILE_ZMIN) & (-Z <= PROFILE_ZMAX)
        zz = Z[zmask]
        st = styles[mname]
        for p in subset:
            c = MOORING_COLORS[p]
            vel = r['mean'][comp][p]
            S = sh.vertical_shear(vel, Z)
            F = -np.nanmean(r['unfilt'][vk][:, p, :] * r['unfilt']['w'][:, p, :], 0)
            P = F * S
            ax0.plot(vel[zmask], zz, color=c, **st)
            ax1.plot(S[zmask], zz, color=c, **st)
            ax2.plot(F[zmask], zz, color=c, **st)
            ax3.plot(P[zmask], zz, color=c, **st)

    for ax in (ax0, ax1, ax2, ax3):
        ax.axvline(0, color='0.6', lw=0.5)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax0.set_xlabel(f'$\\langle {comp}\\rangle$ (m s$^{{-1}}$)')
    ax1.set_xlabel(f"$\\partial_z\\langle {comp}\\rangle$ (s$^{{-1}}$)")
    ax2.set_xlabel(f"$-\\langle {vk}'w'\\rangle$ (m$^2$ s$^{{-2}}$)")
    ax3.set_xlabel('production (m$^2$ s$^{-3}$)')
    ax0.set_ylabel('depth (m)')
    # two legends: mooring by color, model by line style
    moor_h = [Line2D([], [], color=MOORING_COLORS[p], lw=2) for p in subset]
    moor_l = [_moor_label(M24['names'][p]) for p in subset]
    model_h = [Line2D([], [], color='0.3', **styles[m])
               for m in ('tpose24', 'tpose6')]
    model_l = [f"tpose24 (n={ns['tpose24']})", f"tpose6 (n={ns['tpose6']})"]
    leg1 = ax0.legend(moor_h, moor_l, fontsize=8, loc='lower left',
                      title='mooring')
    ax0.add_artist(leg1)
    ax0.legend(model_h, model_l, fontsize=8, loc='lower right', title='model')
    ax0.set_ylim(-PROFILE_ZMAX, -PROFILE_ZMIN)
    fig.tight_layout()
    fname = f'{REPO}/figures/mooring_profiles_{comp}_compare_t24window.png'
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


def load_mean_gradients(cfg, mlon, mlat):
    """Horizontal gradients of the time-mean flow at each mooring point.

    Reads the mean-U/V map cache (from the shear analysis), takes the horizontal
    derivatives d<V>/dy, d<U>/dy, d<V>/dx on the map grid (spherical metrics),
    and samples the nearest grid column at each mooring. Returns (Zmv, grads)
    where grads[p] = {'dVdy','dUdy','dVdx'} each (depth,).
    """
    io = importlib.import_module(cfg['io'])
    mv = xr.open_dataset(f"{io.CACHE_DIR}/{cfg['meanuv']}")
    lon, lat = mv['lon'].values, mv['lat'].values
    Um, Vm = mv['Umean'].values, mv['Vmean'].values   # (depth, y, x)
    Zmv = mv['depth'].values
    mv.close()

    R, d2r = 6371e3, np.pi / 180.0
    dy = R * d2r * float(lat[1] - lat[0])              # uniform meridional spacing
    dx = R * np.cos(d2r * lat) * d2r * float(lon[1] - lon[0])   # (y,) zonal spacing
    dVdy = np.gradient(Vm, axis=1) / dy
    dUdy = np.gradient(Um, axis=1) / dy
    dVdx = np.gradient(Vm, axis=2) / dx[None, :, None]

    grads = []
    for la, lo in zip(mlat, mlon):
        iy = int(np.abs(lat - la).argmin())
        ix = int(np.abs(lon - lo).argmin())
        grads.append({'dVdy': dVdy[:, iy, ix], 'dUdy': dUdy[:, iy, ix],
                      'dVdx': dVdx[:, iy, ix]})
    return Zmv, grads


def make_hprod_fig(M, grads, subset=None, tag=''):
    """Horizontal shear-production profiles at the moorings (unfiltered).

    Three columns: -<v'v'> d<V>/dy, -<u'v'> d<U>/dy, -<u'v'> d<V>/dx. Horizontal
    Reynolds stresses come from the mooring perturbations; the mean-flow
    gradients come from the mean-U/V map cache (see load_mean_gradients).
    """
    Z = M['Z']
    zmask = (-Z >= PROFILE_ZMIN) & (-Z <= PROFILE_ZMAX)
    zz = Z[zmask]
    d = M['treat']['unfiltered']
    subset = range(len(M['names'])) if subset is None else subset

    cols = [(r"$-\langle v'v'\rangle\,\partial_y\langle V\rangle$", 'vv', 'dVdy'),
            (r"$-\langle u'v'\rangle\,\partial_y\langle U\rangle$", 'uv', 'dUdy'),
            (r"$-\langle u'v'\rangle\,\partial_x\langle V\rangle$", 'uv', 'dVdx')]

    fig, axes = plt.subplots(1, 3, figsize=(12, 6.5), sharey=True)
    for p in subset:
        name = M['names'][p]
        c = MOORING_COLORS[p]
        up, vp = d['u'][:, p, :], d['v'][:, p, :]
        stress = {'vv': np.nanmean(vp * vp, 0), 'uv': np.nanmean(up * vp, 0)}
        for ax, (_, sk, gk) in zip(axes, cols):
            P = -stress[sk] * grads[p][gk]            # production (m^2 s^-3)
            ax.plot(P[zmask], zz, c, label=_moor_label(name))

    for ax, (title, _, _) in zip(axes, cols):
        ax.axvline(0, color='0.6', lw=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('production (m$^2$ s$^{-3}$)')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    axes[0].set_ylabel('depth (m)')
    axes[0].legend(fontsize=8, loc='lower left')
    axes[0].set_ylim(-PROFILE_ZMAX, -PROFILE_ZMIN)
    fig.suptitle(f'Mooring horizontal shear production — unfiltered '
                 f'({PROFILE_ZMIN:.0f}-{PROFILE_ZMAX:.0f} m)', y=0.99)
    fig.tight_layout()
    fname = f"{M['fig_dir']}/mooring_hprod_unfiltered{tag}.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


def _movavg(F, win):
    """Centered NaN-aware moving average of (time, depth) `F` along time.

    `win` is the window in samples; win<=1 returns F unchanged. Below-bathymetry
    depths are all-NaN columns and stay NaN; edges shrink the window.
    """
    if win <= 1:
        return F
    return pd.DataFrame(F).rolling(win, center=True, min_periods=1).mean().values


def _autoscale(F):
    """Symmetric 98th-percentile color limit of |F| (as make_fig uses)."""
    fin = np.abs(F[np.isfinite(F)])
    v = np.nanpercentile(fin, 98) if fin.size else 1.0
    return v if v > 0 else 1.0


def _contourf(ax, xt, zz, F, v):
    """Draw one RdBu_r depth-time filled-contour field with +/- v limits."""
    levels = np.linspace(-v, v, N_LEVELS + 1)        # 100 filled contour bands
    cs = ax.contourf(xt, zz, F, levels=levels, cmap='RdBu_r', extend='both')
    ax.xaxis_date()
    return cs


def _add_colorbar(fig, cs, ax, v):
    """Attach a colorbar to `ax` (an axis or list of axes) with +/- v ticks.

    The power of ten is folded into a side label so the ScalarFormatter offset
    text ("1e-X") can't overlap the panel.
    """
    cb = fig.colorbar(cs, ax=ax, pad=0.01, shrink=0.9,
                      ticks=np.linspace(-v, v, 9))
    exp = int(np.floor(np.log10(v))) if v > 0 else 0
    if exp < -2 or exp >= 3:
        s = 10.0 ** exp
        cb.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda t, _, s=s: f'{t/s:.1f}'))
        cb.set_label(f'$\\times10^{{{exp}}}$', fontsize=8)
    else:
        cb.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda t, _: f'{t:g}'))
    return cb


def _contour_panel(fig, ax, xt, zz, F, rlabel, v=None):
    """Draw one depth-time filled-contour panel with its own colorbar.

    `F` is (nz, n_time). If `v` is None the panel auto-scales to the symmetric
    98th percentile of |F| (as make_fig does for the product panels); otherwise
    `v` sets the fixed +/- color limit.
    """
    if v is None:
        v = _autoscale(F)
    cs = _contourf(ax, xt, zz, F, v)
    ax.set_ylabel(rlabel + '\ndepth (m)', fontsize=9)
    _add_colorbar(fig, cs, ax, v)
    return cs


def make_fig(M, p, comp, treat, zr):
    d = M['treat'][treat]
    up, vp, wp = d['u'][:, p, :], d['v'][:, p, :], d['w'][:, p, :]
    fields = {'u': up, 'v': vp, 'w': wp, 'uw': up * wp, 'vw': vp * wp}

    zmin, zmax, zlab = zr
    zmask = (-M['Z'] >= zmin) & (-M['Z'] <= zmax)
    zz = M['Z'][zmask]
    name = M['names'][p]
    label = name.split('_', 1)[1].replace('_', ' ')

    clim = FIXED_CLIM[treat]
    xt = mdates.date2num(M['time'])                  # contourf needs numeric x
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    for ax, (rlabel, key) in zip(axes, FIGS[comp]):
        F = fields[key][:, zmask].T                  # (nz_sel, n_time)
        _contour_panel(fig, ax, xt, zz, F, rlabel, v=clim.get(key))
    axes[-1].set_xlabel('time')
    fig.suptitle(f'Mooring {name[0]} ({label}) — {treat} — {comp} '
                 f'({zlab})', y=0.997)
    fig.autofmt_xdate()
    fig.tight_layout()
    fname = f"{M['fig_dir']}/mooring_{name}_{comp}_{treat}_{zlab}.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


# horizontal shear-production columns: label, stress key, mean-gradient key
HPROD_COLS = [
    (r"$-\langle v'v'\rangle\,\partial_y\langle V\rangle$ (m$^2$ s$^{-3}$)",
     'vv', 'dVdy'),
    (r"$-\langle u'v'\rangle\,\partial_y\langle U\rangle$ (m$^2$ s$^{-3}$)",
     'uv', 'dUdy'),
    (r"$-\langle u'v'\rangle\,\partial_x\langle V\rangle$ (m$^2$ s$^{-3}$)",
     'uv', 'dVdx')]


def _profile_rlabels(comp):
    """Row labels for the vertical-profile hovmoller (matches make_profile_fig)."""
    vk = comp.lower()
    return [f'$\\langle {comp}\\rangle$ (m s$^{{-1}}$)',
            f'$\\partial_z\\langle {comp}\\rangle$ (s$^{{-1}}$)',
            f"$-\\langle {vk}'w'\\rangle$ (m$^2$ s$^{{-2}}$)",
            f"production $-\\langle {vk}'w'\\rangle\\,"
            f"\\partial_z\\langle {comp}\\rangle$ (m$^2$ s$^{{-3}}$)"]


def _profile_fields(M, p, comp, win=1):
    """The four (time, depth) profile-hovmoller fields for mooring p.

    Velocity is the raw total <comp> = mean + anomaly and is NEVER averaged (the
    background flow is shown at full resolution). Shear = MA(d(vel)/dz), flux =
    MA(-comp'w'), and production = MA(shear) * MA(flux) all use a centered
    `win`-sample moving average -- production uses the moving-average shear
    (matching the shear panel), not the mean shear. Unfiltered perturbations.
    """
    vk = comp.lower()
    Z = M['Z']
    d = M['treat']['unfiltered']
    vel_t = M['mean'][comp][p][None, :] + d[vk][:, p, :]        # raw total vel
    shear_ma = _movavg(sh.vertical_shear(vel_t, Z, zaxis=-1), win)
    flux_ma = _movavg(-(d[vk][:, p, :] * d['w'][:, p, :]), win)
    prod = shear_ma * flux_ma                                  # MA shear * MA flux
    return [vel_t, shear_ma, flux_ma, prod]


def _hprod_field(M, grads, p, sk, gk):
    """One horizontal-production (time, depth) field: -<sk>(z,t) * mean grad gk."""
    d = M['treat']['unfiltered']
    up, vp = d['u'][:, p, :], d['v'][:, p, :]
    stress = -(up * vp) if sk == 'uv' else -(vp * vp)
    return stress * grads[p][gk][None, :]


def make_profile_hovmoller(M, p, comp, zr, win=1, out_dir=None, avg_tag=''):
    """Depth-time contour version of make_profile_fig for one mooring/component.

    Four stacked depth-time panels whose time-averages reproduce the mean-profile
    figure (see _profile_fields). Unfiltered perturbations. `win` (samples)
    applies a centered moving average along time.
    """
    Z = M['Z']
    zmin, zmax, zlab = zr
    zmask = (-Z >= zmin) & (-Z <= zmax)
    zz = Z[zmask]
    name = M['names'][p]
    label = name.split('_', 1)[1].replace('_', ' ')

    fields = _profile_fields(M, p, comp, win)
    rlabels = _profile_rlabels(comp)
    xt = mdates.date2num(M['time'])
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for ax, rlabel, F in zip(axes, rlabels, fields):
        _contour_panel(fig, ax, xt, zz, F[:, zmask].T, rlabel)
    axes[-1].set_xlabel('time')
    fig.suptitle(f'Mooring {name[0]} ({label}) — unfiltered — {comp} '
                 f'profile ({zlab}){avg_tag}', y=0.997)
    fig.autofmt_xdate()
    fig.tight_layout()
    fname = (f"{out_dir or M['fig_dir']}/"
             f"mooring_{name}_{comp}_profile_hovmoller_{zlab}.png")
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


def make_hprod_hovmoller(M, grads, p, zr, win=1, out_dir=None, avg_tag=''):
    """Depth-time contour version of make_hprod_fig for one mooring.

    Three stacked depth-time panels of horizontal (barotropic) shear production
    (see HPROD_COLS / _hprod_field); each panel's time average reproduces the
    mean-profile figure. Unfiltered perturbations. `win` (samples) applies a
    centered moving average along time.
    """
    Z = M['Z']
    zmin, zmax, zlab = zr
    zmask = (-Z >= zmin) & (-Z <= zmax)
    zz = Z[zmask]
    name = M['names'][p]
    label = name.split('_', 1)[1].replace('_', ' ')

    xt = mdates.date2num(M['time'])
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    for ax, (rlabel, sk, gk) in zip(axes, HPROD_COLS):
        F = _movavg(_hprod_field(M, grads, p, sk, gk), win)
        _contour_panel(fig, ax, xt, zz, F[:, zmask].T, rlabel)
    axes[-1].set_xlabel('time')
    fig.suptitle(f'Mooring {name[0]} ({label}) — unfiltered — horizontal '
                 f'production ({zlab}){avg_tag}', y=0.997)
    fig.autofmt_xdate()
    fig.tight_layout()
    fname = (f"{out_dir or M['fig_dir']}/"
             f"mooring_{name}_hprod_hovmoller_{zlab}.png")
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


def _hovmoller_grid(M, zr, rlabels, fields_by_mooring, subset, figsize):
    """Draw a rows x 2 depth-time contour grid, columns = moorings in `subset`.

    `fields_by_mooring[p]` is the list of (time, depth) row fields for mooring p.
    Color limits are shared per row across the two moorings (98th percentile over
    both), so the columns are directly comparable; one colorbar per row.
    """
    Z = M['Z']
    zmin, zmax, zlab = zr
    zmask = (-Z >= zmin) & (-Z <= zmax)
    zz = Z[zmask]
    xt = mdates.date2num(M['time'])
    nrows = len(rlabels)
    # constrained_layout (not tight_layout) so the per-row colorbar that spans
    # both columns lands at the far right instead of between them.
    fig, axes = plt.subplots(nrows, 2, figsize=figsize, sharex=True, sharey=True,
                             layout='constrained')
    for r in range(nrows):
        panels = [fields_by_mooring[p][r][:, zmask] for p in subset]  # (time, nz)
        v = _autoscale(np.concatenate([P.ravel() for P in panels]))   # shared
        for cix, P in enumerate(panels):
            cs = _contourf(axes[r, cix], xt, zz, P.T, v)
        axes[r, 0].set_ylabel(rlabels[r] + '\ndepth (m)', fontsize=9)
        _add_colorbar(fig, cs, list(axes[r]), v)
    for cix, p in enumerate(subset):
        axes[0, cix].set_title(_moor_label(M['names'][p]), fontsize=11)
        axes[-1, cix].set_xlabel('time')
        for lbl in axes[-1, cix].get_xticklabels():   # rotate dates (no autofmt)
            lbl.set_rotation(30)
            lbl.set_horizontalalignment('right')
    axes[0, 0].set_ylim(-zmax, -zmin)
    return fig, zlab


def make_profile_hovmoller_AC(M, comp, zr, subset=(0, 2), win=1, out_dir=None,
                              avg_tag=''):
    """A & C side-by-side (2-column) depth-time version of make_profile_fig.

    Rows = the four profile quantities (see _profile_fields); columns = moorings
    A and C. Color limits shared per row so the two columns are comparable.
    `win` (samples) applies a centered moving average along time.
    """
    subset = list(subset)
    fields = {p: _profile_fields(M, p, comp, win) for p in subset}
    fig, zlab = _hovmoller_grid(M, zr, _profile_rlabels(comp), fields, subset,
                                figsize=(13, 11))
    fig.suptitle(f'Moorings A & C — unfiltered — {comp} profile '
                 f'({zlab}){avg_tag}')
    fname = (f"{out_dir or M['fig_dir']}/"
             f"mooring_AC_{comp}_profile_hovmoller_{zlab}.png")
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


def make_hprod_hovmoller_AC(M, grads, zr, subset=(0, 2), win=1, out_dir=None,
                            avg_tag=''):
    """A & C side-by-side (2-column) depth-time version of make_hprod_fig.

    Rows = the three horizontal-production terms (HPROD_COLS); columns = moorings
    A and C. Color limits shared per row so the two columns are comparable.
    `win` (samples) applies a centered moving average along time.
    """
    subset = list(subset)
    fields = {p: [_movavg(_hprod_field(M, grads, p, sk, gk), win)
                  for _, sk, gk in HPROD_COLS] for p in subset}
    rlabels = [lab for lab, _, _ in HPROD_COLS]
    fig, zlab = _hovmoller_grid(M, zr, rlabels, fields, subset, figsize=(13, 9))
    fig.suptitle(f'Moorings A & C — unfiltered — horizontal production '
                 f'({zlab}){avg_tag}')
    fname = (f"{out_dir or M['fig_dir']}/"
             f"mooring_AC_hprod_hovmoller_{zlab}.png")
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


def make_compare_all(loaded):
    """Cross-model A&C profile comparison over the tpose24 window (both comps)."""
    M = {}
    for m in ('tpose24', 'tpose6'):
        if m in loaded:
            M[m] = loaded[m]
        else:
            cfg = MODELS[m]
            path = (f"{importlib.import_module(cfg['io']).CACHE_DIR}/"
                    f"{cfg['cache']}")
            if not os.path.exists(path):
                print(f'SKIP compare: {m} cache missing ({cfg["cache"]})')
                return
            M[m] = load_model(cfg)
    for comp in ('U', 'V'):
        print('wrote', make_profile_compare(M, comp))


def main(models=None):
    loaded = {}
    for model in (models or list(MODELS)):
        cfg = MODELS[model]
        if not os.path.exists(f"{importlib.import_module(cfg['io']).CACHE_DIR}/"
                              f"{cfg['cache']}"):
            print(f'SKIP {model}: cache not found ({cfg["cache"]})')
            continue
        os.makedirs(cfg['fig_dir'], exist_ok=True)
        M = load_model(cfg)
        loaded[model] = M
        print(f'{model}: window {str(M["time"][0])[:10]} .. '
              f'{str(M["time"][-1])[:10]} ({M["time"].size} steps)')
        n = 0
        for p in range(len(M['names'])):
            for comp in ('uw', 'vw'):
                for treat in ('unfiltered', 'yanai'):
                    for zr in DEPTH_RANGES:
                        make_fig(M, p, comp, treat, zr)
                        n += 1
        for comp in ('U', 'V'):                       # mean-profile figures
            make_profile_fig(M, comp)                  # all moorings
            make_profile_fig(M, comp, subset=[0, 2], tag='_AC')  # A & C only
            n += 2
        # horizontal shear-production profiles (needs mean-U/V map cache)
        _, grads = load_mean_gradients(cfg, M['lon'], M['lat'])
        make_hprod_fig(M, grads)
        make_hprod_fig(M, grads, subset=[0, 2], tag='_AC')
        n += 2
        # depth-time contour counterparts of the mean-profile figures, drawn raw
        # and at three moving-average windows (subfolders). dt from the record.
        prof_range = (PROFILE_ZMIN, PROFILE_ZMAX, '300-1500m')
        dt_days = float(np.median(np.diff(M['time'])) / np.timedelta64(1, 'D'))

        def hovmoller_set(win=1, out_dir=None, avg_tag=''):
            k = 0
            for p in range(len(M['names'])):
                for comp in ('U', 'V'):
                    make_profile_hovmoller(M, p, comp, prof_range, win, out_dir,
                                           avg_tag)
                    k += 1
                make_hprod_hovmoller(M, grads, p, prof_range, win, out_dir, avg_tag)
                k += 1
            for comp in ('U', 'V'):                   # A & C side-by-side
                make_profile_hovmoller_AC(M, comp, prof_range, win=win,
                                          out_dir=out_dir, avg_tag=avg_tag)
                k += 1
            make_hprod_hovmoller_AC(M, grads, prof_range, win=win,
                                    out_dir=out_dir, avg_tag=avg_tag)
            return k + 1

        n += hovmoller_set()                          # raw, top level
        for days in (14, 25, 40):
            win = max(1, round(days / dt_days))
            od = f"{cfg['fig_dir']}/{days}day_mov_avg"
            os.makedirs(od, exist_ok=True)
            n += hovmoller_set(win, od, f' — {days}-day avg')
        print(f'{model}: wrote {n} figures to {cfg["fig_dir"]}')

    make_compare_all(loaded)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:] or None)
