"""Cache full-depth UVEL/VVEL/WVEL columns at the three mooring-simulation points
for TPOSE6 (daily NetCDF).

TPOSE6 replication of build_mooring_cache.py. The daily per-variable/per-month
NetCDFs are chunked whole-field-per-timestep, so extracting three columns still
decompresses the full fields (slow on NFS) -- run this detached. Only a small
lat/lon sub-block is read per month (via tpose6_io.load_var iy/ix), then the
three points are extracted from it. WVEL is stored on 66 Zl faces and is averaged
to the 65 cell centers here (matching the profile cache), so the plotting script
sees U/V/W all co-located on the cell-center Z grid.

Mooring points (deg):  A 0.5N,140W   B 1.75N,138W   C 3N,140W

Output: <CACHE_DIR>/yanai_mooring_tp6.nc
"""

import os
import time
import numpy as np
import xarray as xr

import tpose6_io as io
import fluxes as fx

FIELDS = ['UVEL', 'VVEL', 'WVEL']
OUT = os.path.join(io.CACHE_DIR, 'yanai_mooring_tp6.nc')

MOORINGS = [('A_0.5N_140W', 0.5, 220.0),
            ('B_1.75N_138W', 1.75, 222.0),
            ('C_3N_140W', 3.0, 220.0)]


def main():
    grid = io.load_grid()
    lon, lat, Z = grid['lon'], grid['lat'], grid['Z']

    names = [m[0] for m in MOORINGS]
    ii = np.array([io.nearest_idx(lon, m[2]) for m in MOORINGS])
    jj = np.array([io.nearest_idx(lat, m[1]) for m in MOORINGS])
    iy, ix = np.unique(jj), np.unique(ii)          # sub-block read from each file
    py = np.array([int(np.where(iy == j)[0][0]) for j in jj])  # point -> sub-block row
    px = np.array([int(np.where(ix == i)[0][0]) for i in ii])
    print('moorings:', [f'{n} -> lon={lon[i]:.3f}E lat={lat[j]:.3f}'
                        for n, i, j in zip(names, ii, jj)])

    times = io.time_index()
    n_t, n_p = len(times), len(names)
    print(f'timesteps: {n_t}   fields: {FIELDS}')

    data = {}
    t0 = time.time()
    for var in FIELDS:
        da = io.load_var(var, iy=iy, ix=ix)        # (time, z, y_sub, x_sub) any order
        ydim, xdim = io._hdim(da)
        zdim = [d for d in da.dims if d not in ('time', ydim, xdim)][0]
        da = da.transpose('time', zdim, ydim, xdim)
        cols = np.asarray(da.values)[:, :, py, px]  # (time, nz_or_66, n_p)
        if var == 'WVEL':                           # 66 Zl faces -> 65 Z centers
            cols = fx.w_faces_to_centers(cols, zaxis=1)
        data[var] = np.moveaxis(cols, -1, 1).astype('f4')   # (time, n_p, nz)
        print(f'  read {var} {data[var].shape}  ({time.time()-t0:.0f}s)', flush=True)

    hFacC_cols = grid['hFacC'][:, jj, ii].T          # (n_p, nz)

    ds = xr.Dataset(
        {f: (['time', 'loc', 'depth'], data[f]) for f in FIELDS} |
        {'hFacC': (['loc', 'depth'], hFacC_cols.astype('f4'))},
        coords={'time': times, 'loc': names, 'depth': Z,
                'lat': ('loc', lat[jj]), 'lon': ('loc', lon[ii]),
                'drF': ('depth', grid['drF'])},
        attrs={'run': io.DATA_DIR,
               'note': 'raw daily mooring columns; WVEL already averaged to cell '
                       'centers; band-pass / anomaly done in the plotting script'})
    ds.to_netcdf(OUT)
    print('saved', OUT, f'({time.time()-t0:.0f}s total)')


if __name__ == '__main__':
    main()
