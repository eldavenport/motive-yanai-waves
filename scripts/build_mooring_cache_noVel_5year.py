"""Cache full-depth UVEL/VVEL/WVEL columns at the three mooring points for the
5-year TPOSE6 "noVel" run (raw mds via tpose6_noVel_5year_io.open_5year).

Replication of build_mooring_cache_t6.py for the mds-binary 5-year run. A small
lat/lon sub-block spanning the moorings is read for the whole 1827-day record and
the three columns are extracted from it. WVEL is on 66 Zl faces here (U/V/T on 66
Z centers), so it is averaged to the 66 centers with fx.w_to_center (equal-length
Zl->Z, unlike the TPOSE-Vel run's 66->65 w_faces_to_centers). Stored centered, so
the plotting script sees U/V/W co-located (w_on_faces=False).

Mooring points (deg):  A 0.5N,140W   B 1.75N,138W   C 3N,140W

Output: <CACHE_DIR>/yanai_mooring_tp6_noVel_5yr.nc   (run detached: many mds reads)
"""

import os
import time
import numpy as np
import xarray as xr

import tpose6_noVel_5year_io as io
import fluxes as fx

FIELDS = ['UVEL', 'VVEL', 'WVEL']
OUT = os.path.join(io.CACHE_DIR, 'yanai_mooring_tp6_noVel_5yr.nc')

MOORINGS = [('A_0.5N_140W', 0.5, 220.0),
            ('B_1.75N_138W', 1.75, 222.0),
            ('C_3N_140W', 3.0, 220.0)]


def main():
    os.makedirs(io.CACHE_DIR, exist_ok=True)
    grid = io.load_grid()
    lon, lat, Z = grid['lon'], grid['lat'], grid['Z']

    names = [m[0] for m in MOORINGS]
    ii = np.array([io.nearest_idx(lon, m[2]) for m in MOORINGS])
    jj = np.array([io.nearest_idx(lat, m[1]) for m in MOORINGS])
    iy, ix = np.unique(jj), np.unique(ii)              # sub-block read from ds
    py = np.array([int(np.where(iy == j)[0][0]) for j in jj])   # point -> sub row
    px = np.array([int(np.where(ix == i)[0][0]) for i in ii])
    print('moorings:', [f'{n} -> lon={lon[i]:.3f}E lat={lat[j]:.3f}'
                        for n, i, j in zip(names, ii, jj)])

    times = io.time_index().values
    print(f'timesteps: {len(times)}   fields: {FIELDS}', flush=True)

    ds = io.open_5year()
    data = {}
    t0 = time.time()
    for var in FIELDS:
        da = ds[var]
        ydim, xdim = io._hdim(da)
        zdim = [d for d in da.dims if d not in ('time', ydim, xdim)][0]
        sub = da.transpose('time', zdim, ydim, xdim).isel(
            {ydim: iy, xdim: ix}).load()             # (time, nz, y_sub, x_sub)
        cols = np.asarray(sub.values)[:, :, py, px]  # (time, nz, n_p)
        if var == 'WVEL':                            # 66 Zl faces -> 66 Z centers
            cols = fx.w_to_center(cols, zaxis=1)
        data[var] = np.moveaxis(cols, -1, 1).astype('f4')   # (time, n_p, nz)
        print(f'  read {var} {data[var].shape}  ({time.time()-t0:.0f}s)', flush=True)

    hFacC_cols = grid['hFacC'][:, jj, ii].T          # (n_p, nz)

    out = xr.Dataset(
        {f: (['time', 'loc', 'depth'], data[f]) for f in FIELDS} |
        {'hFacC': (['loc', 'depth'], hFacC_cols.astype('f4'))},
        coords={'time': times, 'loc': names, 'depth': Z,
                'lat': ('loc', lat[jj]), 'lon': ('loc', lon[ii]),
                'drF': ('depth', grid['drF'])},
        attrs={'run': '5-year TPOSE6 noVel (diag_state mds, 2012-2016)',
               'note': 'raw daily mooring columns; WVEL averaged Zl(66)->Z(66) '
                       'centers; band-pass / anomaly done in the plotting script'})
    out.to_netcdf(OUT)
    print('saved', OUT, f'({time.time()-t0:.0f}s total)')


if __name__ == '__main__':
    main()
