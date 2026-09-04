"""Cache the time-mean U/V maps over the map sub-domain (175W-105W, 10S-10N) for
the 5-year TPOSE6 "noVel" run, so plot_moorings.load_mean_gradients can form the
horizontal mean-flow gradients used by the barotropic-production hovmollers.

Trimmed version of build_map_cache_t6.py: only the mean-U/V pass is kept (no flux
covariances/maps). The time mean is taken over the same edge-trimmed window the
mooring analysis uses (20-day band-pass trim at each end; no spin-up drop). Land
(hFacC == 0) is set to NaN. Reads stream lazily over the sub-domain via dask.

Output: <CACHE_DIR>/yanai_meanUV_maps_tp6_noVel_5yr.nc   (run detached)
"""

import os
import time
import numpy as np
import xarray as xr

import tpose6_noVel_5year_io as io
import shear as sh

OUT = os.path.join(io.CACHE_DIR, 'yanai_meanUV_maps_tp6_noVel_5yr.nc')


def main():
    os.makedirs(io.CACHE_DIR, exist_ok=True)
    t0 = time.time()
    grid = io.load_grid()
    Z, drF = grid['Z'], grid['drF']
    iy_a, ix_a = io.map_subset_idx(grid)
    iy = slice(int(iy_a[0]), int(iy_a[-1]) + 1)
    ix = slice(int(ix_a[0]), int(ix_a[-1]) + 1)
    lon, lat = grid['lon'][ix], grid['lat'][iy]
    land = grid['hFacC'][:, iy, ix] == 0.0
    print(f'sub-domain ny={lat.size} nx={lon.size}', flush=True)

    ds = io.open_5year()
    times = np.asarray(ds['time'].values)
    etm = io.edge_trim_mask(times)
    print(f'edge-trim keeps {int(etm.sum())}/{times.size} steps', flush=True)

    def tmean(var, ydim, xdim):
        da = ds[var].isel(time=etm)
        da = da.isel({ydim: iy, xdim: ix})
        m = np.asarray(da.mean('time').transpose(
            [d for d in da.dims if d != 'time'][0], ydim, xdim).values, dtype='f4')
        m[land] = np.nan
        return m

    print('computing mean U/V', flush=True)
    Umean = tmean('UVEL', 'YC', 'XG')                 # UVEL on YC/XG
    Vmean = tmean('VVEL', 'YG', 'XC')                 # VVEL on YG/XC
    Sx0 = sh.vertical_shear(Umean, Z, zaxis=0)
    Sy0 = sh.vertical_shear(Vmean, Z, zaxis=0)

    xr.Dataset(
        {'Umean': (['depth', 'y', 'x'], Umean),
         'Vmean': (['depth', 'y', 'x'], Vmean),
         'Sx0': (['depth', 'y', 'x'], Sx0.astype('f4')),
         'Sy0': (['depth', 'y', 'x'], Sy0.astype('f4'))},
        coords={'depth': Z, 'lon': ('x', lon), 'lat': ('y', lat),
                'drF': ('depth', drF)},
        attrs={'units_vel': 'm s-1', 'units_shear': 's-1',
               'window': 'edge-trim removed, no spin-up drop',
               'edge_trim_days': io.EDGE_TRIM_DAYS,
               'run': '5-year TPOSE6 noVel (diag_state mds, 2012-2016)',
               'map_lon': str(io.MAP_LON), 'map_lat': str(io.MAP_LAT),
               'note': 'time-mean raw UVEL/VVEL and background vertical shear'}
    ).to_netcdf(OUT)
    print('saved', OUT, f'({time.time()-t0:.0f}s total)')


if __name__ == '__main__':
    main()
