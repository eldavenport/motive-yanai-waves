"""Cache full-depth UVEL/VVEL/WVEL columns at the three mooring-simulation points.

Companion to build_profile_cache.py, but at the MOTIVE mooring locations rather
than the 140W profile line. One pass over all diag_state files: at each of the
three mooring points the full 138-level columns of U/V/W are read for every
3-hourly timestep. Raw (unfiltered) columns are saved -- WVEL is left on its top
faces (Zl); the plotting script centers it and does the band-pass / anomaly.

Mooring points (deg):  A 0.5N,140W   B 1.75N,138W   C 3N,140W
All three are inside the TPOSE24 nested domain (~150.6W-129.4W, 5.5S-10.5N).

Output: <CACHE_DIR>/yanai_mooring_dt60.nc
"""

import os
import time
import numpy as np
import xarray as xr
from concurrent.futures import ThreadPoolExecutor

import tpose24_io as io

FIELDS = ['UVEL', 'VVEL', 'WVEL']
OUT = os.path.join(io.CACHE_DIR, 'yanai_mooring_dt60.nc')
N_WORKERS = 8

# (name, lat degN, lon degE); names carry the label used in figure filenames
MOORINGS = [('A_0.5N_140W', 0.5, 220.0),
            ('B_1.75N_138W', 1.75, 222.0),
            ('C_3N_140W', 3.0, 220.0)]


def main():
    grid = io.load_grid()
    lon, lat, Z = grid['lon'], grid['lat'], grid['Z']

    names = [m[0] for m in MOORINGS]
    ii = np.array([io.nearest_idx(lon, m[2]) for m in MOORINGS])
    jj = np.array([io.nearest_idx(lat, m[1]) for m in MOORINGS])

    iters = io.diag_iters()
    times = io.iter_times(iters)
    n_t, n_p, nz = len(iters), len(names), io.NZ
    print('moorings:', [f'{n} -> lon={lon[i]:.3f}E lat={lat[j]:.3f}'
                        for n, i, j in zip(names, ii, jj)])
    print(f'timesteps: {n_t}   fields: {FIELDS}')

    data = {f: np.full((n_t, n_p, nz), np.nan, np.float32) for f in FIELDS}

    def read_one(t):
        return t, io.read_columns(int(iters[t]), FIELDS, ii, jj)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for n, (t, cols) in enumerate(ex.map(read_one, range(n_t))):
            for f in FIELDS:
                data[f][t] = cols[f]
            if n % 100 == 0 or n == n_t - 1:
                print(f'  {n+1}/{n_t}  ({time.time()-t0:.0f}s)', flush=True)

    hFacC_cols = grid['hFacC'][:, jj, ii].T          # (n_p, nz)

    ds = xr.Dataset(
        {f: (['time', 'loc', 'depth'], data[f]) for f in FIELDS} |
        {'hFacC': (['loc', 'depth'], hFacC_cols.astype('f4'))},
        coords={'time': times, 'loc': names, 'depth': Z,
                'lat': ('loc', lat[jj]), 'lon': ('loc', lon[ii]),
                'drF': ('depth', grid['drF'])},
        attrs={'run': io.RUN_DIR,
               'note': 'raw 3-hourly mooring columns; WVEL on Zl faces, '
                       'centered + band-passed in the plotting script'})
    ds.to_netcdf(OUT)
    print('saved', OUT, f'({time.time()-t0:.0f}s total)')


if __name__ == '__main__':
    main()
