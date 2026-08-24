"""Cache depth/time-compressed 2D maps of Yanai-band momentum & energy fluxes
for TPOSE6 over the map sub-domain (175W-105W, 10S-10N).

Single pass over the sub-domain:
  1. read THETA, SALT over the sub-domain for the whole daily record and rebuild
     PHIHYD = p_hyd/rhonil via JMD95 (eos_jmd95.py); free THETA, SALT
  2. read UVEL, VVEL, WVEL; average WVEL from 66 Zl faces to 65 Z centers
  3. mask land (hFacC == 0), band-pass each field to 15-40 day periods in time
     (daily sampling), then drop the 20-day edge-trim interval at each end
  4. time-mean the 6 momentum-flux and 3 energy-flux products -> 3D covariances
  5. depth-integrate over the layers 0-350, 350-700, 700-1500, 1500m-bottom, and
     full column -> 2D lat/lon maps

No spin-up drop (the run has none); the whole record is used, only the filter
edge-trim is removed. rho0 for the energy flux is rhonil = 1035.

Outputs (in CACHE_DIR):
  yanai_flux_cov3d_tp6.nc   time-mean 3D covariances (component, depth, y, x)
  yanai_flux_maps_tp6.nc    layer-integrated 2D maps (component, layer, y, x)
"""

import os
import time
import numpy as np
import xarray as xr

import tpose6_io as io
import wave_filter as wf
import fluxes as fx
import eos_jmd95 as eos

fx.RHO0 = io.RHONIL                       # energy flux uses rhonil = 1035


def load_subset(var, iy, ix):
    """Sub-domain array (nt, nz, ny, nx) float32 for one variable."""
    return io.load_var(var, iy=iy, ix=ix).values.astype(np.float32)


def main():
    t0 = time.time()
    grid = io.load_grid()
    Z, drF = grid['Z'], grid['drF']
    iy_a, ix_a = io.map_subset_idx(grid)                   # contiguous index arrays
    iy = slice(int(iy_a[0]), int(iy_a[-1]) + 1)            # -> slices (fast reads)
    ix = slice(int(ix_a[0]), int(ix_a[-1]) + 1)
    hFacC = grid['hFacC'][:, iy, ix]                       # (nz, ny, nx)
    land = hFacC == 0.0
    lon = grid['lon'][ix]; lat = grid['lat'][iy]
    ny, nx = lat.size, lon.size
    times = io.time_index().values
    etm = io.edge_trim_mask(times)                         # keep interior
    print(f'sub-domain ny={ny} nx={nx}  nt={times.size} '
          f'(edge-trim keeps {int(etm.sum())})', flush=True)

    # 1. THETA/SALT -> PHIHYD, then free T/S
    print('reading THETA/SALT -> PHIHYD', flush=True)
    th = load_subset('THETA', iy, ix)
    sa = load_subset('SALT', iy, ix)
    th[:, land] = np.nan; sa[:, land] = np.nan
    phi = eos.phihyd_from_ts(th, sa, Z, drF, zaxis=1).astype(np.float32)
    del th, sa

    # 2. velocities; WVEL faces -> centers
    print('reading UVEL/VVEL/WVEL', flush=True)
    up = load_subset('UVEL', iy, ix)
    vp = load_subset('VVEL', iy, ix)
    wc = fx.w_faces_to_centers(load_subset('WVEL', iy, ix), zaxis=1).astype(np.float32)

    # 3. mask land, band-pass in time (daily), each field in place
    print('masking land + band-passing (15-40 d, daily)', flush=True)
    for name, arr in (('UVEL', up), ('VVEL', vp), ('WVEL', wc), ('PHIHYD', phi)):
        arr[:, land] = np.nan
        ts = time.time()
        wf.bandpass_inplace(arr, dt_days=io.DT_DAYS)
        print(f'  filtered {name}  ({time.time()-ts:.0f}s)', flush=True)

    # edge trim (band-pass transients) -- views, no copy
    up, vp, wc, pp = up[etm], vp[etm], wc[etm], fx.RHO0 * phi[etm]

    def tmean(a, b):
        return (np.einsum('tzyx,tzyx->zyx', a, b) / a.shape[0]).astype(np.float32)

    print('computing time-mean covariances', flush=True)
    cov = {'uu': tmean(up, up), 'vv': tmean(vp, vp), 'ww': tmean(wc, wc),
           'uv': tmean(up, vp), 'uw': tmean(up, wc), 'vw': tmean(vp, wc),
           'pu': tmean(pp, up), 'pv': tmean(pp, vp), 'pw': tmean(pp, wc)}
    comps = fx.MOM_COMPONENTS + fx.ENE_COMPONENTS

    attrs = {'momentum_units': 'm2 s-2', 'energy_units': 'W m-2',
             'band_days': f'{wf.YANAI_MIN_DAYS}-{wf.YANAI_MAX_DAYS}',
             'dt_days': io.DT_DAYS, 'edge_trim_days': io.EDGE_TRIM_DAYS,
             'spinup_days': 0.0, 'rhonil': io.RHONIL, 'eos': io.EOS_TYPE,
             'map_lon': str(io.MAP_LON), 'map_lat': str(io.MAP_LAT),
             'run': io.DATA_DIR}

    cov3d = np.stack([cov[c] for c in comps])
    xr.Dataset(
        {'cov': (['component', 'depth', 'y', 'x'], cov3d)},
        coords={'component': comps, 'depth': Z,
                'lon': ('x', lon), 'lat': ('y', lat), 'drF': ('depth', drF)},
        attrs=attrs).to_netcdf(os.path.join(io.CACHE_DIR, 'yanai_flux_cov3d_tp6.nc'))
    print('saved cov3d', flush=True)

    # 5. partial-cell-weighted depth AVERAGE over layers -> 2D maps
    # (averaged, not integrated, so the unequal-thickness layers are comparable)
    masks = fx.layer_masks(Z)
    layers = list(masks)
    maps = np.full((len(comps), len(layers), ny, nx), np.nan, np.float32)
    for ci, c in enumerate(comps):
        for li, lab in enumerate(layers):
            maps[ci, li] = fx.layer_average(cov[c], drF, hFacC, masks[lab])

    map_attrs = dict(attrs)
    map_attrs.update({'momentum_units': 'm2 s-2 (depth avg)',
                      'energy_units': 'W m-2 (depth avg)',
                      'reduction': 'partial-cell-weighted depth average',
                      'layer_bounds_m': str(fx.LAYER_BOUNDS)})
    xr.Dataset(
        {'flux_avg': (['component', 'layer', 'y', 'x'], maps)},
        coords={'component': comps, 'layer': layers,
                'lon': ('x', lon), 'lat': ('y', lat)},
        attrs=map_attrs).to_netcdf(os.path.join(io.CACHE_DIR, 'yanai_flux_maps_tp6.nc'))
    print('saved maps', f'({time.time()-t0:.0f}s total)', flush=True)


if __name__ == '__main__':
    main()
