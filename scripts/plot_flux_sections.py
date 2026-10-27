"""Depth-section mean & variance of the momentum fluxes u'w', v'w', u'v'.

Two section orientations:
  lon-depth at 0.5 N (nearest latitude to SEC_LAT)
  lat-depth at 140 W

For each model (tpose24, tpose6) and each band (total, yanai) a 2x3 figure is
drawn:
  rows = time-mean <x'y'>  and  time-variance Var_t[x'y']  of the flux
  cols = u'w', v'w', u'v'

Perturbations follow the mooring/section convention: spin-up dropped before
filtering (tpose24), 20-day band-pass edge-trim removed; `total` = anomaly from
the window mean (all frequencies), `yanai` = 15-40 day band-pass.

The two section planes are streamed directly -- one equatorial lat-row and one
140 W lon-column over all timesteps -- the same cheap single-slice reads that
build_uv_section uses, so there is no full-domain (~600 GB) load. WVEL is moved
to cell centers before forming the fluxes.

Figures -> figures/flux_sections (tpose24), figures/tpose6/flux_sections (tpose6).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from concurrent.futures import ThreadPoolExecutor

import tpose24_io as io24
import tpose6_io as io6
import wave_filter as wf
import fluxes as fx

REPO = '/home/edavenport/analysis/motive-yanai-waves'
SEC_LON_W = 140.0           # lon (deg W) for the lat-depth section
SEC_LAT = 0.5               # lat (deg N) for the lon-depth section
SEC_ZMAX = 2500.0           # depth limit (m) for the sections
N_LEVELS = 100
FLUX_COLS = [('uw', r"$u'w'$"), ('vw', r"$v'w'$"), ('uv', r"$u'v'$")]


def _flux_mean_var(U, V, W, land, band, interior, dt_days):
    """Time-mean and time-variance of u'w', v'w', u'v' on a (time, nz, ns) plane.

    `land` (nz, ns) is NaN-masked. `interior` indexes the analysis window along
    time (slice for tpose24, boolean mask for tpose6). `total` perturbations are
    the anomaly from the window mean; `yanai` is the 15-40 day band-pass (filtered
    on the full record, then restricted to the interior). Returns (mean, var),
    each a dict keyed uw/vw/uv of (nz, ns) arrays.
    """
    U, V, W = (A.astype('f8').copy() for A in (U, V, W))
    for A in (U, V, W):
        A[:, land] = np.nan
    if band == 'total':
        U, V, W = U[interior], V[interior], W[interior]
        up = U - np.nanmean(U, 0, keepdims=True)
        vp = V - np.nanmean(V, 0, keepdims=True)
        wp = W - np.nanmean(W, 0, keepdims=True)
    else:
        up = wf.bandpass(U, axis=0, dt_days=dt_days)[interior]
        vp = wf.bandpass(V, axis=0, dt_days=dt_days)[interior]
        wp = wf.bandpass(W, axis=0, dt_days=dt_days)[interior]
    prod = {'uw': up * wp, 'vw': vp * wp, 'uv': up * vp}
    mean = {k: np.nanmean(p, 0) for k, p in prod.items()}
    var = {k: np.nanvar(p, 0) for k, p in prod.items()}
    return mean, var


def stream_t24():
    """Stream the equator lat-row and 140 W lon-column over the tpose24 record."""
    g = io24.load_grid()
    lon, lat, Z, hFacC = g['lon'], g['lat'], g['Z'], g['hFacC']
    iters = io24.iters_after_spinup()
    nt, k = len(iters), io24.edge_trim_samples()
    jeq = io24.nearest_idx(lat, SEC_LAT)
    ix = io24.nearest_idx(lon, 360.0 - SEC_LON_W)
    rows = {f: np.empty((nt, io24.NZ, io24.NX), 'f4') for f in ('U', 'V', 'W')}
    cols = {f: np.empty((nt, io24.NZ, io24.NY), 'f4') for f in ('U', 'V', 'W')}

    def read_one(t):
        it = int(iters[t])
        for f, fld in (('U', 'UVEL'), ('V', 'VVEL'), ('W', 'WVEL')):
            rows[f][t] = io24.read_field_subdomain(it, fld, jeq, jeq + 1,
                                                   0, io24.NX)[:, 0, :]
            cols[f][t] = io24.read_field_subdomain(it, fld, 0, io24.NY,
                                                   ix, ix + 1)[:, :, 0]
        return t

    with ThreadPoolExecutor(max_workers=12) as ex:
        for n, _ in enumerate(ex.map(read_one, range(nt))):
            if n % 200 == 0 or n == nt - 1:
                print(f'  t24 read {n+1}/{nt}', flush=True)
    for d in (rows, cols):                                  # WVEL faces -> centers
        d['W'] = fx.w_to_center(d['W'], zaxis=1)
    interior = slice(k, nt - k)
    dt_days = io24.OUT_STEP_SEC / 86400.0
    row = dict(U=rows['U'], V=rows['V'], W=rows['W'], Z=Z, x=lon,
               xlabel='lon (°E)', land=(hFacC[:, jeq, :] == 0.0))
    col = dict(U=cols['U'], V=cols['V'], W=cols['W'], Z=Z, x=lat,
               xlabel='lat (°N)', land=(hFacC[:, :, ix] == 0.0))
    return row, col, interior, dt_days


def stream_t6():
    """Stream the equator lat-row and 140 W lon-column over the tpose6 record."""
    g = io6.load_grid()
    lon, lat, Z, hFacC = g['lon'], g['lat'], g['Z'], g['hFacC']
    etm = io6.edge_trim_mask(io6.time_index())
    jeq = io6.nearest_idx(lat, SEC_LAT)
    ix = io6.nearest_idx(lon, 360.0 - SEC_LON_W)

    def read_plane(iy, ix_, horiz_axis, target_len):
        """Stream the three velocities on one section plane, co-located on centers.

        WVEL is averaged from Zl faces to Z centers; the C-grid velocity whose
        along-section dim lands on the staggered face grid (VVEL on YG, UVEL on XG)
        is one longer than the tracer grid, so it is averaged to tracer centers.
        """
        out = {}
        for f, fld in (('U', 'UVEL'), ('V', 'VVEL'), ('W', 'WVEL')):
            a = io6.load_var(fld, iy=iy, ix=ix_).values     # (time, z, y, x)
            a = np.take(a, 0, axis=horiz_axis)              # drop singleton dim
            if fld == 'WVEL':                               # 66 Zl faces -> 65 Z
                a = fx.w_faces_to_centers(a, zaxis=1)
            if a.shape[-1] == target_len + 1:               # face grid -> centers
                a = 0.5 * (a[..., :-1] + a[..., 1:])
            out[f] = a
        return out                                          # (time, nz, ns)

    rows = read_plane(slice(jeq, jeq + 1), None, 2, len(lon))   # 0.5N, all lon
    cols = read_plane(None, slice(ix, ix + 1), 3, len(lat))     # 140 W, all lat
    row = dict(Z=Z, x=lon, xlabel='lon (°E)',
               land=(hFacC[:, jeq, :] == 0.0), **rows)
    col = dict(Z=Z, x=lat, xlabel='lat (°N)',
               land=(hFacC[:, :, ix] == 0.0), **cols)
    return row, col, etm, 1.0


def _fold_pow10(cb, v):
    """Fold the power-of-ten into the colorbar label (as plot_moorings does)."""
    exp = int(np.floor(np.log10(v))) if v > 0 else 0
    if exp < -2 or exp >= 3:
        s = 10.0 ** exp
        cb.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda t, _, s=s: f'{t/s:.1f}'))
        cb.set_label(f'$\\times10^{{{exp}}}$', fontsize=8)


def _cf(ax, x, z, F, cmap, levels, v):
    cs = ax.contourf(x, z, F, levels=levels, cmap=cmap, extend='both')
    return cs


# depth ranges (zmin, zmax, zlab) per band; '' zlab = full column, no name suffix.
# The yanai band also gets a 300-1400 m zoom whose colorbars rescale to the deep
# signal (otherwise the strong surface/equatorial flux swamps the deep fluxes).
SEC_RANGES = {
    'total': [(0.0, SEC_ZMAX, '')],
    'yanai': [(0.0, SEC_ZMAX, ''), (300.0, 1400.0, '300-1400m')],
}


def _draw_section(mean, var, Z, x, xlabel, zr, band, model, orient, out_dir):
    """Draw one 2x3 mean/variance section figure over the depth range `zr`.

    Colorbars auto-scale to the plotted depth range (98th pct), so a deep zoom
    rescales to the deep fluxes/variance rather than the near-surface signal.
    """
    zmin, zmax, zlab = zr
    zmask = (-Z >= zmin) & (-Z <= zmax)
    zz = Z[zmask]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True, sharey=True,
                             layout='constrained')
    for cix, (key, lab) in enumerate(FLUX_COLS):
        m = mean[key][zmask]                               # (nz_sel, ns)
        va = var[key][zmask]
        vm = np.nanpercentile(np.abs(m[np.isfinite(m)]), 98) if np.isfinite(m).any() else 1.0
        vm = vm if vm > 0 else 1.0
        cs0 = _cf(axes[0, cix], x, zz, m, 'RdBu_r',
                  np.linspace(-vm, vm, N_LEVELS + 1), vm)
        fin = va[np.isfinite(va)]
        vv = np.nanpercentile(fin, 98) if fin.size else 1.0
        vv = vv if vv > 0 else 1.0
        cs1 = _cf(axes[1, cix], x, zz, va, 'viridis',
                  np.linspace(0, vv, N_LEVELS + 1), vv)
        axes[0, cix].set_title(lab, fontsize=12)
        cb0 = fig.colorbar(cs0, ax=axes[0, cix], pad=0.01, shrink=0.9)
        _fold_pow10(cb0, vm)
        cb1 = fig.colorbar(cs1, ax=axes[1, cix], pad=0.01, shrink=0.9)
        _fold_pow10(cb1, vv)
        axes[1, cix].set_xlabel(xlabel)
    axes[0, 0].set_ylabel('mean $\\langle x\'y\'\\rangle$\ndepth (m)', fontsize=10)
    axes[1, 0].set_ylabel('variance Var$_t[x\'y\']$\ndepth (m)', fontsize=10)
    axes[0, 0].set_ylim(-zmax, -zmin)
    zttl = f' — {zlab}' if zlab else ''
    fig.suptitle(f'{model} — momentum-flux {orient.replace("_", " ")} — '
                 f'{band} — mean (top) & variance (bottom){zttl}')
    os.makedirs(out_dir, exist_ok=True)
    suffix = f'_{zlab}' if zlab else ''
    fname = os.path.join(out_dir, f'flux_section_{orient}_{band}{suffix}.png')
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print('wrote', fname, flush=True)
    return fname


def plot_section(sec, band, model, orient, out_dir):
    """Mean/variance section figure(s) for one plane, one per SEC_RANGES[band]."""
    mean, var = _flux_mean_var(sec['U'], sec['V'], sec['W'], sec['land'],
                               band, sec['interior'], sec['dt_days'])
    for zr in SEC_RANGES[band]:
        _draw_section(mean, var, sec['Z'], sec['x'], sec['xlabel'],
                      zr, band, model, orient, out_dir)


MODELS = {
    'tpose24': dict(stream=stream_t24, fig_dir=f'{REPO}/figures/flux_sections'),
    'tpose6': dict(stream=stream_t6, fig_dir=f'{REPO}/figures/tpose6/flux_sections'),
}
ORIENTS = ['lon_depth_0.5N', 'lat_depth_140W']


def main(models=None):
    for model in (models or list(MODELS)):
        cfg = MODELS[model]
        print(f'=== {model} ===', flush=True)
        row, col, interior, dt_days = cfg['stream']()
        for sec, orient in ((row, ORIENTS[0]), (col, ORIENTS[1])):
            sec['interior'], sec['dt_days'] = interior, dt_days
            for band in ('total', 'yanai'):
                plot_section(sec, band, model, orient, cfg['fig_dir'])


def _selfcheck():
    """Variance >= 0 and a known-product mean recovered on a toy plane."""
    rng = np.random.default_rng(0)
    nt, nz, ns = 200, 4, 5
    U = rng.standard_normal((nt, nz, ns))
    W = rng.standard_normal((nt, nz, ns))
    V = rng.standard_normal((nt, nz, ns))
    land = np.zeros((nz, ns), bool)
    land[0, 0] = True
    mean, var = _flux_mean_var(U, V, W, land, 'total', slice(None), 0.125)
    assert np.all(var['uw'][~land] >= 0)
    assert np.isnan(mean['uw'][0, 0]) and np.isnan(var['uw'][0, 0])
    # independent unit-normal u',w': <u'w'> ~ 0, Var[u'w'] ~ 1
    assert abs(np.nanmean(mean['uw'])) < 0.2
    assert abs(np.nanmean(var['uw']) - 1.0) < 0.3
    print('selfcheck OK')


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['test']:
        _selfcheck()
    else:
        main(sys.argv[1:] or None)
