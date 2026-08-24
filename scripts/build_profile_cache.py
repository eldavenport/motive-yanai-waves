"""Cache full-depth UVEL/VVEL/WVEL/PHIHYD columns at the profile locations.

One pass over all diag_state files: at each of the 5 profile points
(140W; 1S,0N,1N,2N,3N) the full 138-level columns of the four fields are read
for every 3-hourly timestep. Raw (unfiltered) columns are saved so band-pass
filtering and flux computation can be redone interactively in the notebook.

Output: <CACHE_DIR>/yanai_profiles_dt60.nc
"""

import os
import time
import numpy as np
import xarray as xr
from concurrent.futures import ThreadPoolExecutor

import tpose24_io as io

FIELDS = ['UVEL', 'VVEL', 'WVEL', 'PHIHYD']
OUT = os.path.join(io.CACHE_DIR, 'yanai_profiles_dt60.nc')
N_WORKERS = 8


def main():
    grid = io.load_grid()
    lon, lat, Z = grid['lon'], grid['lat'], grid['Z']

    loc_names = list(io.PROFILE_LAT)                      # 1S,0N,1N,2N,3N
    lon_val = io.PROFILE_LON['140W']
    i_idx = io.nearest_idx(lon, lon_val)
    j_idx = np.array([io.nearest_idx(lat, io.PROFILE_LAT[n]) for n in loc_names])
    ii = np.full(len(loc_names), i_idx)

    iters = io.diag_iters()
    times = io.iter_times(iters)
    n_t, n_p, nz = len(iters), len(loc_names), io.NZ
    print(f'points: {loc_names} at lon={lon[i_idx]:.3f}E '
          f'lat={[round(float(lat[j]),3) for j in j_idx]}')
    print(f'timesteps: {n_t}   fields: {FIELDS}')

    data = {f: np.full((n_t, n_p, nz), np.nan, np.float32) for f in FIELDS}

    def read_one(t):
        cols = io.read_columns(int(iters[t]), FIELDS, ii, j_idx)  # field->(n_p,nz)
        return t, cols

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for n, (t, cols) in enumerate(ex.map(read_one, range(n_t))):
            for f in FIELDS:
                data[f][t] = cols[f]
            if n % 100 == 0 or n == n_t - 1:
                el = time.time() - t0
                print(f'  {n+1}/{n_t}  ({el:.0f}s)', flush=True)

    # partial-cell column info at each point (for depth weighting later)
    hFacC_cols = grid['hFacC'][:, j_idx, i_idx].T          # (n_p, nz)

    ds = xr.Dataset(
        {f: (['time', 'loc', 'depth'], data[f]) for f in FIELDS} |
        {'hFacC': (['loc', 'depth'], hFacC_cols.astype('f4'))},
        coords={'time': times, 'loc': loc_names, 'depth': Z,
                'lat': ('loc', lat[j_idx]),
                'drF': ('depth', grid['drF'])},
        attrs={'lon': float(lon[i_idx]), 'run': io.RUN_DIR,
               'note': 'raw 3-hourly columns; band-pass/flux done in notebook'})
    ds.to_netcdf(OUT)
    print('saved', OUT, f'({time.time()-t0:.0f}s total)')


if __name__ == '__main__':
    main()
