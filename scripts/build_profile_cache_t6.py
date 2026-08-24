"""Cache full-depth U/V/W/PHIHYD columns at the TPOSE6 profile locations.

At each of the 3 longitudes (170W, 140W, 110W) x 5 latitudes (1S,0N,1N,2N,3N)
the full 65-level daily columns of UVEL, VVEL, WVEL and the reconstructed
PHIHYD (from THETA/SALT via JMD95, see eos_jmd95.py) are read for the whole
record. WVEL is averaged from its 66 Zl faces to the 65 Z centers. Raw
(unfiltered) columns are saved so the band-pass and flux computation can be
redone interactively in the notebook, exactly as for TPOSE24.

Below-seafloor points (hFacC == 0) are set to NaN. Output:
  <CACHE_DIR>/yanai_profiles_tp6.nc   dims (time, lon, lat, depth)
"""

import os
import time
import numpy as np
import xarray as xr

import tpose6_io as io
import fluxes as fx
import eos_jmd95 as eos


def main():
    t0 = time.time()
    grid = io.load_grid()
    lon, lat, Z, drF = grid['lon'], grid['lat'], grid['Z'], grid['drF']

    lon_names = list(io.PROFILE_LON)                       # 170W,140W,110W
    lat_names = list(io.PROFILE_LAT)                       # 1S,0N,1N,2N,3N
    i_idx = [io.nearest_idx(lon, io.PROFILE_LON[n]) for n in lon_names]
    j_idx = [io.nearest_idx(lat, io.PROFILE_LAT[n]) for n in lat_names]
    print('lon points:', [(n, round(float(lon[i]), 2)) for n, i in zip(lon_names, i_idx)])
    print('lat points:', [(n, round(float(lat[j]), 2)) for n, j in zip(lat_names, j_idx)])

    # Read a contiguous bounding slab (fast; fancy per-point indexing forces
    # netCDF to decompress whole chunks), then pick the exact points inside it.
    i0, i1 = min(i_idx), max(i_idx)
    j0, j1 = min(j_idx), max(j_idx)
    ii_rel = [i - i0 for i in i_idx]
    jj_rel = [j - j0 for j in j_idx]

    def read(var):
        v = io.load_var(var, iy=slice(j0, j1 + 1),
                        ix=slice(i0, i1 + 1)).values.astype('f8')  # (t,Z,lat,lon)
        v = v[:, :, jj_rel, :][:, :, :, ii_rel]  # -> (t, Z, lat5, lon3)
        v = np.moveaxis(v, 1, -1)                # (t, lat, lon, depth)
        return np.moveaxis(v, 1, 2)              # (t, lon, lat, depth)

    th = read('THETA'); sa = read('SALT')
    up = read('UVEL'); vp = read('VVEL')
    wc = fx.w_faces_to_centers(read('WVEL'), zaxis=-1)      # 66 faces -> 65 centers

    # below-seafloor mask (lon, lat, depth) from hFacC
    hf = np.moveaxis(grid['hFacC'][:, j_idx][:, :, i_idx], 0, -1)   # (lat,lon,depth)
    hf = np.moveaxis(hf, 0, 1)                                       # (lon,lat,depth)
    land = hf == 0.0
    for a in (th, sa, up, vp, wc):
        a[:, land] = np.nan

    # reconstruct PHIHYD = p_hyd/rhonil (m^2 s^-2) per column from THETA/SALT
    phi = eos.phihyd_from_ts(th, sa, Z, drF, zaxis=-1)     # (time, lon, lat, depth)

    ds = xr.Dataset(
        {'UVEL':   (['time', 'lon', 'lat', 'depth'], up.astype('f4')),
         'VVEL':   (['time', 'lon', 'lat', 'depth'], vp.astype('f4')),
         'WVEL':   (['time', 'lon', 'lat', 'depth'], wc.astype('f4')),
         'PHIHYD': (['time', 'lon', 'lat', 'depth'], phi.astype('f4')),
         'hFacC':  (['lon', 'lat', 'depth'], hf.astype('f4'))},
        coords={'time': io.time_index(), 'lon': lon_names, 'lat': lat_names,
                'depth': Z, 'drF': ('depth', drF),
                'lon_deg': ('lon', np.array([lon[i] for i in i_idx])),
                'lat_deg': ('lat', np.array([lat[j] for j in j_idx]))},
        attrs={'run': io.DATA_DIR, 'rhonil': io.RHONIL, 'eos': io.EOS_TYPE,
               'note': 'raw daily columns; band-pass/flux done in notebook. '
                       'WVEL averaged to Z centers. PHIHYD reconstructed from '
                       'THETA/SALT (JMD95Z). No spin-up drop.'})
    out = os.path.join(io.CACHE_DIR, 'yanai_profiles_tp6.nc')
    ds.to_netcdf(out)
    print('saved', out, f'({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
