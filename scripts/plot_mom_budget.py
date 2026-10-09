"""Meridional convergence of zonal momentum flux: -d<u'v'>/dy (lat-depth section).

Science question: is there an eddy influx of zonal momentum onto / away from the
equator? The term in the zonal-momentum budget is the meridional convergence of
the zonal-meridional Reynolds stress,

    -d<u'v'>/dy      (m s^-2),

positive = eddies deposit eastward momentum at that latitude (influx/convergence),
negative = they remove it (export/divergence). (This is the meridional analogue
of the vertical -d<u'w'>/dz term; d<u'w'>/dy is *not* a budget term.)

<u'v'> is the time-mean 15-40 day band covariance already in the cov3d cache
(component 'uv', dims component,depth,y,x). For each requested longitude we take
the lat-depth slice and differentiate in y with spherical metrics (reused from
plot_moorings.load_mean_gradients). Two panels: <u'v'> (context) and its
meridional convergence. -> momentum_budget/

Needs yanai_flux_cov3d_*.nc (build_map_cache[/_t6].py).
"""

import os
import importlib
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plot_moorings as pm

# cov3d cache per model (component-stacked time-mean covariances)
COV3D = {'tpose24': 'yanai_flux_cov3d_dt60.nc', 'tpose6': 'yanai_flux_cov3d_tp6.nc'}
# longitudes to section (deg W); only those inside the model domain are drawn
LONS_W = {'tpose24': [140], 'tpose6': [170, 140, 110]}
LAT_LIM = 5.0            # +/- latitude shown (deg)
# depth windows (zmin, zmax, filename tag). The colour limit auto-scales to the
# plotted band, so the 100-1000 m view gets smaller limits that bring out the
# (much weaker) deep structure hidden under the surface-intensified signal.
ZRANGES = [(0.0, 1500.0, ''), (100.0, 1000.0, '_100-1000m')]
R_EARTH, D2R = 6371e3, np.pi / 180.0


def merid_convergence(uv, lat):
    """-d(uv)/dy on the lat axis (axis=1) with spherical metric. uv: (depth, lat)."""
    dy = R_EARTH * D2R * float(lat[1] - lat[0])        # uniform meridional spacing
    return -np.gradient(uv, axis=1) / dy


# cache of TOTAL-anomaly <u'v'> sections (all frequencies), from build_uv_section.py
TOTAL = {'tpose24': 'yanai_uv_section_total_dt60.nc',
         'tpose6': 'yanai_uv_section_total_tp6.nc'}
BAND_LBL = {'yanai': 'time-mean 15-40 day band',
            'total': 'time-mean total anomaly (all freqs)'}


def uv_band(ds, lon_w):
    """<u'v'>(depth, lat) at the nearest lon from the band-passed cov3d cache."""
    ix = int(np.abs(ds.lon.values - (360.0 - lon_w)).argmin())
    return ds.lat.values, ds.depth.values, \
        ds['cov'].sel(component='uv').values[:, :, ix]


def uv_total(ds, lon_w):
    """<u'v'>(depth, lat) for lon_w from the total-anomaly section cache."""
    sub = ds.sel(lon_w=lon_w)
    return ds.lat.values, ds.depth.values, sub['uv'].values


def make_fig(cfg, lon_w, lat, depth, uv, zmin, zmax, ztag, treat):
    conv = merid_convergence(uv, lat)
    zmask = (-depth >= zmin) & (-depth <= zmax)
    lmask = np.abs(lat) <= LAT_LIM
    zz, yy = depth[zmask], lat[lmask]
    panels = [('$\\langle u\'v\'\\rangle$ (m$^2$ s$^{-2}$)', uv),
              ("$-\\partial\\langle u'v'\\rangle/\\partial y$ "
               '(m s$^{-2}$)  [+ = influx]', conv)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True,
                             layout='constrained')
    for ax, (title, F) in zip(axes, panels):
        Fs = F[np.ix_(zmask, lmask)]
        v = pm._autoscale(Fs)
        pcm = ax.pcolormesh(yy, zz, Fs, cmap='RdBu_r', vmin=-v, vmax=v,
                            shading='auto')
        ax.axvline(0, color='0.3', lw=0.8)
        ax.set_xlabel('latitude (deg)')
        ax.set_title(title, fontsize=10)
        fig.colorbar(pcm, ax=ax, pad=0.01, shrink=0.9)
    axes[0].set_ylabel('depth (m)')
    axes[0].set_ylim(-zmax, -zmin)
    fig.suptitle(f'Meridional eddy flux of zonal momentum — {lon_w}W '
                 f'({BAND_LBL[treat]}; {zmin:.0f}-{zmax:.0f} m)')
    fname = pm.fig_path(cfg['fig_dir'],
                        f'uv_merid_convergence_{lon_w}W_{treat}_lat_depth{ztag}.png')
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


def main(models=None):
    for model in (models or ['tpose24', 'tpose6']):
        cfg = pm.MODELS[model]
        io = importlib.import_module(cfg['io'])
        cov_path = f"{io.CACHE_DIR}/{COV3D[model]}"
        if not os.path.exists(cov_path):
            print(f'SKIP {model}: cov3d cache not found ({COV3D[model]})')
            continue
        os.makedirs(cfg['fig_dir'], exist_ok=True)
        ds_band = xr.open_dataset(cov_path)
        lo, hi = 360 - ds_band.lon.values.max(), 360 - ds_band.lon.values.min()

        sec_path = f"{io.CACHE_DIR}/{TOTAL[model]}"
        ds_tot = xr.open_dataset(sec_path) if os.path.exists(sec_path) else None
        if ds_tot is None:
            print(f'  (no total-anomaly section cache; run build_uv_section.py '
                  f'{model} for the total version)')

        for lon_w in LONS_W[model]:
            if not lo <= lon_w <= hi:
                print(f'SKIP {model} {lon_w}W: outside domain')
                continue
            sources = [('yanai', uv_band(ds_band, lon_w))]
            if ds_tot is not None and lon_w in ds_tot.lon_w.values:
                sources.append(('total', uv_total(ds_tot, lon_w)))
            for treat, (lat, depth, uv) in sources:
                for zmin, zmax, ztag in ZRANGES:
                    print('wrote', make_fig(cfg, lon_w, lat, depth, uv,
                                            zmin, zmax, ztag, treat))
        ds_band.close()
        if ds_tot is not None:
            ds_tot.close()


def _selfcheck():
    """Monotonically northward-increasing <u'v'> must give uniform divergence (<0)."""
    lat = np.linspace(-5, 5, 101)
    uv = np.tile(lat, (3, 1))                           # (depth, lat), d/dy > 0
    conv = merid_convergence(uv, lat)
    assert np.all(conv < 0), conv.max()
    # sinusoid: -d/dy sin(ky) = -k cos(ky); zero crossing sign check at y=0
    k = 2 * np.pi / 4.0
    uvs = np.tile(np.sin(k * lat), (2, 1))
    c0 = merid_convergence(uvs, lat)[0, np.argmin(np.abs(lat))]
    assert c0 < 0, c0                                   # -k cos(0) = -k < 0
    print('mom-budget self-check OK')


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['test']:
        _selfcheck()
    else:
        main(sys.argv[1:] or None)
