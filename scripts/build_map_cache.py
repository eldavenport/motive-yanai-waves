"""Cache depth/time-compressed 2D maps of Yanai-band momentum & energy fluxes.

Full-domain, single parallel pass over all diag_state files:
  1. read UVEL, VVEL, WVEL, PHIHYD for every 3-hourly timestep into memory
  2. mask land (hFacC == 0), band-pass each field to 15-40 day periods in time
  3. move WVEL to cell centers; form p' = rho0 * PHIHYD'
  4. time-mean the 6 momentum-flux and 3 energy-flux products -> 3D covariances
  5. depth-integrate each covariance over the layers 0-350, 350-700, 700-1500,
     1500m-bottom, and full column -> 2D lat/lon maps

Outputs (in CACHE_DIR):
  yanai_flux_cov3d_dt60.nc   time-mean 3D covariances (component, depth, y, x)
  yanai_flux_maps_dt60.nc    layer-integrated 2D maps (component, layer, y, x)
"""

import os
import time
import numpy as np
import xarray as xr
from concurrent.futures import ThreadPoolExecutor

import tpose24_io as io
import wave_filter as wf
import fluxes as fx

FIELDS = ['UVEL', 'VVEL', 'WVEL', 'PHIHYD']
N_READERS = 24
COV3D_OUT = os.path.join(io.CACHE_DIR, 'yanai_flux_cov3d_dt60.nc')
MAPS_OUT  = os.path.join(io.CACHE_DIR, 'yanai_flux_maps_dt60.nc')


def load_all(iters):
    """Read the four fields for all iterations into (nt,nz,ny,nx) float32 arrays."""
    nt = len(iters)
    arrs = {f: np.empty((nt, io.NZ, io.NY, io.NX), np.float32) for f in FIELDS}

    def read_one(t):
        mm = np.memmap(io._path(int(iters[t])), dtype=io.DT_DTYPE, mode='r')
        for f in FIELDS:
            b = io.FIELD_IDX[f] * io._FIELD
            arrs[f][t] = np.asarray(mm[b:b + io._FIELD],
                                    dtype=np.float32).reshape(io.NZ, io.NY, io.NX)
        return t

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_READERS) as ex:
        for n, _ in enumerate(ex.map(read_one, range(nt))):
            if n % 100 == 0 or n == nt - 1:
                print(f'  read {n+1}/{nt}  ({time.time()-t0:.0f}s)', flush=True)
    return arrs


def main():
    grid = io.load_grid()
    Z, drF, hFacC = grid['Z'], grid['drF'], grid['hFacC']
    land = hFacC == 0.0                                   # (nz,ny,nx)

    # drop the first SPINUP_DAYS immediately: only post-spin-up timesteps are
    # loaded, so the spin-up never enters the band-pass filter.
    iters = io.iters_after_spinup()
    k = io.edge_trim_samples()                            # samples to trim each end
    print(f'timesteps: {len(iters)}  (spin-up {io.SPINUP_DAYS} d dropped before '
          f'loading; then {io.EDGE_TRIM_DAYS} d = {k} samples trimmed each end '
          f'after filtering)   fields: {FIELDS}')

    arrs = load_all(iters)

    # mask land to NaN, then band-pass each field to the Yanai band in place
    print('masking land + band-passing', flush=True)
    for f in FIELDS:
        arrs[f][:, land] = np.nan
        t0 = time.time()
        wf.bandpass_inplace(arrs[f])
        print(f'  filtered {f}  ({time.time()-t0:.0f}s)', flush=True)

    # trim band-pass edge contamination at both ends (views, no copy)
    sl = slice(k, len(iters) - k)
    up, vp = arrs['UVEL'][sl], arrs['VVEL'][sl]
    wp = fx.w_to_center(arrs['WVEL'][sl], zaxis=1)         # faces -> centers
    pp = fx.RHO0 * arrs['PHIHYD'][sl]                      # pressure perturbation

    # time-mean products (einsum reduction avoids a full-size product temporary)
    def tmean(a, b):
        return (np.einsum('tzyx,tzyx->zyx', a, b) / a.shape[0]).astype(np.float32)

    print('computing time-mean covariances', flush=True)
    cov = {
        'uu': tmean(up, up), 'vv': tmean(vp, vp), 'ww': tmean(wp, wp),
        'uv': tmean(up, vp), 'uw': tmean(up, wp), 'vw': tmean(vp, wp),
        'pu': tmean(pp, up), 'pv': tmean(pp, vp), 'pw': tmean(pp, wp),
    }
    mom = fx.MOM_COMPONENTS            # kinematic momentum flux (m^2 s^-2)
    ene = fx.ENE_COMPONENTS            # energy flux rho0*p'u'_i (W m^-2)

    # save 3D time-mean covariances
    comps = mom + ene
    cov3d = np.stack([cov[c] for c in comps])             # (ncomp, nz, ny, nx)
    ds3d = xr.Dataset(
        {'cov': (['component', 'depth', 'y', 'x'], cov3d)},
        coords={'component': comps, 'depth': Z,
                'lon': ('x', grid['lon']), 'lat': ('y', grid['lat']),
                'drF': ('depth', drF)},
        attrs={'momentum_units': 'm2 s-2', 'energy_units': 'W m-2',
               'band_days': f'{wf.YANAI_MIN_DAYS}-{wf.YANAI_MAX_DAYS}',
               'spinup_days': io.SPINUP_DAYS, 'edge_trim_days': io.EDGE_TRIM_DAYS,
               'run': io.RUN_DIR})
    ds3d.to_netcdf(COV3D_OUT)
    print('saved', COV3D_OUT, flush=True)

    # depth-integrate each covariance over each layer -> 2D maps
    masks = fx.layer_masks(Z)                             # label -> (nz,) bool
    layers = list(masks)
    maps = np.full((len(comps), len(layers), io.NY, io.NX), np.nan, np.float32)
    for ci, c in enumerate(comps):
        for li, lab in enumerate(layers):
            maps[ci, li] = fx.layer_integral(cov[c], drF, hFacC, masks[lab])

    ds_maps = xr.Dataset(
        {'flux_int': (['component', 'layer', 'y', 'x'], maps)},
        coords={'component': comps, 'layer': layers,
                'lon': ('x', grid['lon']), 'lat': ('y', grid['lat'])},
        attrs={'momentum_units': 'm3 s-2 (int of m2 s-2 over dz)',
               'energy_units': 'W m-1 (int of W m-2 over dz)',
               'band_days': f'{wf.YANAI_MIN_DAYS}-{wf.YANAI_MAX_DAYS}',
               'spinup_days': io.SPINUP_DAYS, 'edge_trim_days': io.EDGE_TRIM_DAYS,
               'layer_bounds_m': str(fx.LAYER_BOUNDS), 'run': io.RUN_DIR})
    ds_maps.to_netcdf(MAPS_OUT)
    print('saved', MAPS_OUT, flush=True)


if __name__ == '__main__':
    main()
