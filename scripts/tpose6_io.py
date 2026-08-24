"""IO helpers for the TPOSE6 TPOSE-Vel run (per-variable, per-month NetCDF).

Fields THETA, SALT, UVEL, VVEL, WVEL, ETAN live in
/data/SO3/edavenport/tpose6/TPOSE-Vel/ as one NetCDF per variable per month,
daily output, grid 65(z) x 240(y) x 1128(x) (WVEL on 66 Zl faces). The run uses
eosType='JMD95Z', rhonil=1035; there is NO PHIHYD diagnostic, so the hydrostatic
pressure potential is rebuilt from THETA/SALT (see eos_jmd95.py).

This mirrors the TPOSE24 analysis (see tpose24_io.py) but for daily NetCDF and
the wider tropical-Pacific domain. Differences, per user direction:
- NO spin-up drop: the whole record (2012-09-01 .. 2013-06-30, 303 daily steps)
  is used.
- The 20-day band-pass edge trim IS kept (it removes filtfilt edge transients,
  a filter artifact independent of spin-up).
- Profiles at 170W, 140W and 110W (all now inside the domain) x 1S,0N,1N,2N,3N.
- Maps are restricted to 175W-105W, 10S-10N.

Horizontal C-grid stagger (U on XG, V on YG, W/T on XC/YC) is neglected: fields
are read at matching integer indices and treated as co-located, as in TPOSE24.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr

# --- run configuration --------------------------------------------------------
DATA_DIR  = '/data/SO3/edavenport/tpose6/TPOSE-Vel'
CACHE_DIR = '/data/SO3/edavenport/tpose6/cache'
RHONIL    = 1035.0                       # rhonil from the TPOSE6 data namelist
EOS_TYPE  = 'JMD95Z'

# chronological order of the monthly files (filenames are not date-sortable)
MONTHS = ['sep2012', 'oct2012', 'nov2012', 'dec2012', 'jan2013',
          'feb2013', 'mar2013', 'apr2013', 'may2013', 'jun2013']

DT_DAYS = 1.0                            # daily output
EDGE_TRIM_DAYS = 20.0                    # band-pass edge contamination trimmed
                                         # at each end (kept; filter artifact)
# NO spin-up drop for TPOSE6 (the run has none) -- whole record is used.

# map sub-domain (deg E / deg N): 175W-105W, 10S-10N
MAP_LON = (185.0, 255.0)                 # 175W .. 105W in degE
MAP_LAT = (-10.0, 10.0)

# --- profile locations --------------------------------------------------------
PROFILE_LON = {'170W': 190.0, '140W': 220.0, '110W': 250.0}   # degE
PROFILE_LAT = {'1S': -1.0, '0N': 0.0, '1N': 1.0, '2N': 2.0, '3N': 3.0}

FIELDS_3D = ['THETA', 'SALT', 'UVEL', 'VVEL', 'WVEL']


def var_files(var):
    """Chronologically-ordered file paths for one variable."""
    return [os.path.join(DATA_DIR, f'tpose_vel_{var}_{m}_daily_20Sto20N.nc')
            for m in MONTHS]


def _hdim(da):
    """(y_dim_name, x_dim_name) for a data array on any C-grid position."""
    ydim = 'YG' if 'YG' in da.dims else 'YC'
    xdim = 'XG' if 'XG' in da.dims else 'XC'
    return ydim, xdim


def load_var(var, iy=None, ix=None):
    """Concatenate one variable over all months into a single DataArray.

    iy, ix : optional integer index slices (applied on the variable's own
    horizontal dims) to read only a sub-block. Non-dimension coordinates
    (iter, rA, hFacC, ...) are dropped so months concatenate cleanly.
    """
    das = []
    for f in var_files(var):
        ds = xr.open_dataset(f)
        da = ds[var].reset_coords(drop=True)
        if iy is not None or ix is not None:
            ydim, xdim = _hdim(da)
            sel = {}
            if iy is not None:
                sel[ydim] = iy
            if ix is not None:
                sel[xdim] = ix
            da = da.isel(**sel)
        das.append(da.load())
        ds.close()
    return xr.concat(das, dim='time')


def load_grid():
    """1D coordinates and vertical metrics for the full domain.

    Returns dict:
      lon (nx), lat (ny)   cell-center longitude / latitude (deg)
      lon_g, lat_g         u-/v-point longitude (XG) / latitude (YG)
      Z (nz), Zl (nz+1)    cell-center / top-face depth (m, negative down)
      drF (nz)             cell thickness (m, positive)
      hFacC (nz, ny, nx)   partial-cell fraction (0 = land)
    """
    t = xr.open_dataset(var_files('THETA')[0])
    w = xr.open_dataset(var_files('WVEL')[0])
    u = xr.open_dataset(var_files('UVEL')[0])
    v = xr.open_dataset(var_files('VVEL')[0])
    grid = dict(
        lon=np.asarray(t.XC.values), lat=np.asarray(t.YC.values),
        lon_g=np.asarray(u.XG.values), lat_g=np.asarray(v.YG.values),
        Z=np.asarray(t.Z.values), Zl=np.asarray(w.Zl.values),
        drF=np.asarray(t.drF.values),
        hFacC=np.asarray(t.hFacC.values))
    for ds in (t, w, u, v):
        ds.close()
    return grid


def time_index():
    """DatetimeIndex over the whole daily record (all months concatenated)."""
    ts = []
    for f in var_files('THETA'):
        ds = xr.open_dataset(f)
        ts.append(ds.time.values)
        ds.close()
    return pd.DatetimeIndex(np.concatenate(ts))


def edge_trim_mask(times, trim_days=EDGE_TRIM_DAYS):
    """Boolean mask keeping times at least `trim_days` inside both record ends.

    Applied to the *filtered* signal to drop band-pass edge contamination near
    each boundary (filtfilt start/end transients).
    """
    t = np.asarray(times)
    lo = t[0] + np.timedelta64(int(trim_days * 86400), 's')
    hi = t[-1] - np.timedelta64(int(trim_days * 86400), 's')
    return (t >= lo) & (t <= hi)


def nearest_idx(coord1d, target):
    """Index of the grid point nearest `target` in a 1D coordinate array."""
    return int(np.abs(np.asarray(coord1d) - target).argmin())


def map_subset_idx(grid):
    """Integer index arrays (iy, ix) into the full grid for the map sub-domain."""
    lon, lat = grid['lon'], grid['lat']
    ix = np.where((lon >= MAP_LON[0]) & (lon <= MAP_LON[1]))[0]
    iy = np.where((lat >= MAP_LAT[0]) & (lat <= MAP_LAT[1]))[0]
    return iy, ix
