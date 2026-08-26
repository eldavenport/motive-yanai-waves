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
        if key in clim:
            v = clim[key]
        else:                                        # products: auto-scale
            fin = np.abs(F[np.isfinite(F)])
            v = np.nanpercentile(fin, 98) if fin.size else 1.0
            v = v if v > 0 else 1.0
        levels = np.linspace(-v, v, N_LEVELS + 1)    # 100 filled contour bands
        cs = ax.contourf(xt, zz, F, levels=levels, cmap='RdBu_r', extend='both')
        ax.xaxis_date()
        ax.set_ylabel(rlabel + '\ndepth (m)', fontsize=9)
        cb = fig.colorbar(cs, ax=ax, pad=0.01, shrink=0.9,
                          ticks=np.linspace(-v, v, 9))
        # avoid the ScalarFormatter offset text (the "1e-X") that overlaps the
        # panel: fold the power of ten into a side label instead
        exp = int(np.floor(np.log10(v))) if v > 0 else 0
        if exp < -2 or exp >= 3:
            s = 10.0 ** exp
            cb.ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda t, _, s=s: f'{t/s:.1f}'))
            cb.set_label(f'$\\times10^{{{exp}}}$', fontsize=8)
        else:
            cb.ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda t, _: f'{t:g}'))
    axes[-1].set_xlabel('time')
    fig.suptitle(f'Mooring {name[0]} ({label}) — {treat} — {comp} '
                 f'({zlab})', y=0.997)
    fig.autofmt_xdate()
    fig.tight_layout()
    fname = f"{M['fig_dir']}/mooring_{name}_{comp}_{treat}_{zlab}.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


def main(models=None):
    for model in (models or list(MODELS)):
        cfg = MODELS[model]
        if not os.path.exists(f"{importlib.import_module(cfg['io']).CACHE_DIR}/"
                              f"{cfg['cache']}"):
            print(f'SKIP {model}: cache not found ({cfg["cache"]})')
            continue
        os.makedirs(cfg['fig_dir'], exist_ok=True)
        M = load_model(cfg)
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
        print(f'{model}: wrote {n} figures to {cfg["fig_dir"]}')


if __name__ == '__main__':
    import sys
    main(sys.argv[1:] or None)
