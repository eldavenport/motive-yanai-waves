"""Cache THETA/SALT columns at the three mooring points, for dynamical vertical
modes (N^2-based) in plot_deep_jets.py.

Companion to the build_mooring_cache*.py scripts -- same three columns, same grid,
but the two tracers instead of U/V/W. The full time series is stored (small), so
both the time-mean N^2 modes and the time-evolving N^2 modes can be built from one
cache. Below-seafloor cells are left as read (0 for mds runs, NaN for TPOSE6 Vel);
plot_deep_jets masks via hFacC from the mooring cache.

Run:  python build_strat_cache.py [tpose24|tpose6|tpose6_noVel_5year ...]
(TPOSE6 / 5-year read whole fields off NFS -- run those detached.)

Output: <CACHE_DIR>/yanai_strat_<tag>.nc  (tag mirrors the mooring cache name)
"""

import os
import sys
import time
import importlib
import numpy as np
import xarray as xr
from concurrent.futures import ThreadPoolExecutor

import plot_moorings as pm

FIELDS = ['THETA', 'SALT']
MOORINGS = [('A_0.5N_140W', 0.5, 220.0),
            ('B_1.75N_138W', 1.75, 222.0),
            ('C_3N_140W', 3.0, 220.0)]


def _read_tpose24(io, ii, jj):
    iters = io.diag_iters()
    times = io.iter_times(iters)
    data = {f: np.full((len(iters), len(ii), io.NZ), np.nan, 'f4') for f in FIELDS}

    def read_one(t):
        return t, io.read_columns(int(iters[t]), FIELDS, ii, jj)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for t, cols in ex.map(read_one, range(len(iters))):
            for f in FIELDS:
                data[f][t] = cols[f]
    return times, data


def _read_xarray(io, ii, jj, open_da):
    """THETA/SALT columns via an xarray DataArray getter `open_da(var) -> da`."""
    iy, ix = np.unique(jj), np.unique(ii)
    py = np.array([int(np.where(iy == j)[0][0]) for j in jj])
    px = np.array([int(np.where(ix == i)[0][0]) for i in ii])
    times, data = None, {}
    for var in FIELDS:
        da = open_da(var, iy, ix)
        ydim, xdim = io._hdim(da)
        zdim = [d for d in da.dims if d not in ('time', ydim, xdim)][0]
        da = da.transpose('time', zdim, ydim, xdim)
        cols = np.asarray(da.values)[:, :, py, px]        # (time, nz, n_p)
        data[var] = np.moveaxis(cols, -1, 1).astype('f4')  # (time, n_p, nz)
        times = da['time'].values
    return times, data


def build(model):
    cfg = pm.MODELS[model]
    io = importlib.import_module(cfg['io'])
    grid = io.load_grid()
    lon, lat, Z = grid['lon'], grid['lat'], grid['Z']
    names = [m[0] for m in MOORINGS]
    ii = np.array([io.nearest_idx(lon, m[2]) for m in MOORINGS])
    jj = np.array([io.nearest_idx(lat, m[1]) for m in MOORINGS])
    print(model, 'moorings:', [f'{n}->{lon[i]:.2f}E,{lat[j]:.2f}N'
                               for n, i, j in zip(names, ii, jj)], flush=True)

    t0 = time.time()
    if model == 'tpose24':
        times, data = _read_tpose24(io, ii, jj)
    elif model == 'tpose6':
        times, data = _read_xarray(io, ii, jj,
                                   lambda v, iy, ix: io.load_var(v, iy=iy, ix=ix))
        _ds = None
    else:  # tpose6_noVel_5year
        ds = io.open_5year()
        times, data = _read_xarray(io, ii, jj, lambda v, iy, ix: ds[v].isel(
            {io._hdim(ds[v])[0]: iy, io._hdim(ds[v])[1]: ix}))
    print(f'  read {FIELDS} in {time.time()-t0:.0f}s', flush=True)

    out = xr.Dataset(
        {f: (['time', 'loc', 'depth'], data[f]) for f in FIELDS},
        coords={'time': times, 'loc': names, 'depth': Z},
        attrs={'run': model, 'note': 'mean/time-evolving N^2 modes in plot_deep_jets'})
    fname = os.path.join(io.CACHE_DIR, cfg['cache'].replace('mooring', 'strat'))
    out.to_netcdf(fname)
    print('saved', fname, f'({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    for m in (sys.argv[1:] or list(pm.MODELS)):
        build(m)
