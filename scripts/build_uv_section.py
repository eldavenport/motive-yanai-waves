"""Cache the TOTAL-anomaly zonal-momentum Reynolds stress <u'v'>(depth, lat).

Companion to the band-passed cov3d caches: here u' = u - time_mean(u) is the
TOTAL anomaly (all frequencies), not the 15-40 day band. Only the few longitude
sections that plot_mom_budget needs are computed, so this stays cheap -- one lon
column is streamed at a time (low memory) over the same analysis window as the
band-passed covariances (spin-up dropped + band-pass edge-trim removed, so the
total and band-passed fig-3 views use identical time samples).

Outputs (CACHE_DIR):
  yanai_uv_section_total_dt60.nc   (tpose24)   uv(lon_w, depth, lat)
  yanai_uv_section_total_tp6.nc    (tpose6)
"""

import os
import numpy as np
import xarray as xr
from concurrent.futures import ThreadPoolExecutor

import tpose24_io as io24
import tpose6_io as io6

LONS_W = {'tpose24': [140], 'tpose6': [170, 140, 110]}     # deg W sections
OUT = {'tpose24': 'yanai_uv_section_total_dt60.nc',
       'tpose6': 'yanai_uv_section_total_tp6.nc'}


def _cov_total(u, v, land):
    """Total-anomaly <u'v'> over time. u,v: (time, nz, ny); land: (nz, ny) bool."""
    u, v = u.astype('f8'), v.astype('f8')
    u[:, land] = np.nan
    v[:, land] = np.nan
    up = u - np.nanmean(u, 0, keepdims=True)
    vp = v - np.nanmean(v, 0, keepdims=True)
    return np.nanmean(up * vp, 0).astype('f4')             # (nz, ny)


def section_t24(lons_w):
    grid = io24.load_grid()
    lon, lat, Z, hFacC = grid['lon'], grid['lat'], grid['Z'], grid['hFacC']
    iters = io24.iters_after_spinup()
    nt, k = len(iters), io24.edge_trim_samples()
    ixs = {lw: io24.nearest_idx(lon, 360.0 - lw) for lw in lons_w}
    U = {lw: np.empty((nt, io24.NZ, io24.NY), 'f4') for lw in lons_w}
    V = {lw: np.empty((nt, io24.NZ, io24.NY), 'f4') for lw in lons_w}

    def read_one(t):
        it = int(iters[t])
        for lw, ix in ixs.items():
            U[lw][t] = io24.read_field_subdomain(it, 'UVEL', 0, io24.NY, ix, ix + 1)[:, :, 0]
            V[lw][t] = io24.read_field_subdomain(it, 'VVEL', 0, io24.NY, ix, ix + 1)[:, :, 0]
        return t

    with ThreadPoolExecutor(max_workers=12) as ex:
        for n, _ in enumerate(ex.map(read_one, range(nt))):
            if n % 100 == 0 or n == nt - 1:
                print(f'  t24 read {n+1}/{nt}', flush=True)
    sl = slice(k, nt - k)                                  # match cov3d window
    out = {lw: _cov_total(U[lw][sl], V[lw][sl], hFacC[:, :, ix] == 0.0)
           for lw, ix in ixs.items()}
    return Z, lat, out


def section_t6(lons_w):
    grid = io6.load_grid()
    lon, Z, hFacC = grid['lon'], grid['Z'], grid['hFacC']
    iy_a, _ = io6.map_subset_idx(grid)
    iy = slice(int(iy_a[0]), int(iy_a[-1]) + 1)
    lat = grid['lat'][iy]
    etm = io6.edge_trim_mask(io6.time_index())
    out = {}
    for lw in lons_w:
        ix = io6.nearest_idx(lon, 360.0 - lw)
        u = io6.load_var('UVEL', iy=iy, ix=slice(ix, ix + 1)).values[..., 0]
        v = io6.load_var('VVEL', iy=iy, ix=slice(ix, ix + 1)).values[..., 0]
        out[lw] = _cov_total(u[etm], v[etm], hFacC[:, iy, ix] == 0.0)
        print(f'  t6 {lw}W done', flush=True)
    return Z, lat, out


def build(model):
    sec = section_t24 if model == 'tpose24' else section_t6
    cache = io24.CACHE_DIR if model == 'tpose24' else io6.CACHE_DIR
    Z, lat, out = sec(LONS_W[model])
    lons = list(out)
    arr = np.stack([out[lw] for lw in lons])               # (nlon, nz, ny)
    ds = xr.Dataset(
        {'uv': (['lon_w', 'depth', 'lat'], arr)},
        coords={'lon_w': lons, 'depth': Z, 'lat': lat},
        attrs={'units': 'm2 s-2', 'model': model,
               'note': "total-anomaly <u'v'> (u - time mean, all frequencies); "
                       'window matches band-passed cov3d (spin-up + edge-trim removed)'})
    path = os.path.join(cache, OUT[model])
    ds.to_netcdf(path)
    print('saved', path, flush=True)


def main(models=None):
    for model in (models or ['tpose24', 'tpose6']):
        print(f'=== {model} ===', flush=True)
        build(model)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:] or None)
