"""Regenerate the TPOSE6 layer maps as depth AVERAGES from the cached 3D
covariances -- no raw data re-read needed.

Reads yanai_flux_cov3d_tp6.nc (time-mean 3D covariances over the map sub-domain)
and the sub-domain hFacC, and writes yanai_flux_maps_tp6.nc with partial-cell-
weighted depth *averages* per layer (native flux units), replacing the earlier
depth-integrated maps. Matches build_map_cache_t6.py's map step exactly.
"""

import os
import numpy as np
import xarray as xr

import tpose6_io as io
import fluxes as fx


def main():
    cov3d = xr.open_dataset(os.path.join(io.CACHE_DIR, 'yanai_flux_cov3d_tp6.nc'))
    comps = [str(c) for c in cov3d['component'].values]
    Z = cov3d['depth'].values
    drF = cov3d['drF'].values
    lon = cov3d['lon'].values
    lat = cov3d['lat'].values

    grid = io.load_grid()
    iy_a, ix_a = io.map_subset_idx(grid)
    iy = slice(int(iy_a[0]), int(iy_a[-1]) + 1)
    ix = slice(int(ix_a[0]), int(ix_a[-1]) + 1)
    hFacC = grid['hFacC'][:, iy, ix]                       # (nz, ny, nx)
    assert hFacC.shape[1:] == (lat.size, lon.size), 'subdomain mismatch'

    cov = cov3d['cov'].values                              # (comp, depth, y, x)
    masks = fx.layer_masks(Z)
    layers = list(masks)
    maps = np.full((len(comps), len(layers), lat.size, lon.size), np.nan, np.float32)
    for ci in range(len(comps)):
        for li, lab in enumerate(layers):
            maps[ci, li] = fx.layer_average(cov[ci], drF, hFacC, masks[lab])

    attrs = dict(cov3d.attrs)
    attrs.update({'momentum_units': 'm2 s-2 (depth avg)',
                  'energy_units': 'W m-2 (depth avg)',
                  'reduction': 'partial-cell-weighted depth average',
                  'layer_bounds_m': str(fx.LAYER_BOUNDS)})
    out = os.path.join(io.CACHE_DIR, 'yanai_flux_maps_tp6.nc')
    xr.Dataset(
        {'flux_avg': (['component', 'layer', 'y', 'x'], maps)},
        coords={'component': comps, 'layer': layers,
                'lon': ('x', lon), 'lat': ('y', lat)},
        attrs=attrs).to_netcdf(out)
    print('wrote', out)
    print('layers:', layers)
    for ci, c in enumerate(comps):
        a = maps[ci]
        print(f'  {c}: finite%', round(100*np.isfinite(a).mean(), 1),
              'medAbs', f'{np.nanmedian(np.abs(a)):.3e}')


if __name__ == '__main__':
    main()
