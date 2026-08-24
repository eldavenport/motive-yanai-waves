"""IO helpers for the TPOSE24 dt=60 run (oct2012_3mo_dt60_AB3).

The MITgcm diag_state output is raw big-endian float32 with the field order
    THETA, SALT, UVEL, VVEL, WVEL, PHIHYD, DRHODR
packed as (nz, ny, nx) per field per iteration file. Grid files (XC, YC, RC,
DRF, hFacC, ...) live in the same run directory and are shared across all
TPOSE24 runs.

Field reads use direct big-endian memmap offsets (fast, low memory); grid
coordinates are taken from a light 2-iteration xmitgcm load so the spherical
grid is not reconstructed by hand.
"""

import os
import re
import glob
import numpy as np
import pandas as pd

# --- run configuration --------------------------------------------------------
RUN_DIR   = '/data/SO3/edavenport/tpose24/oct2012_3mo_dt60_AB3'
GRID_DIR  = RUN_DIR                       # grid files shared, live in the run dir
CACHE_DIR = '/data/SO3/edavenport/tpose24/cache'
REF_DATE  = '2012-10-01'
DELTA_T   = 60.0                          # model time step (s), dt=60 run
OUT_STEP_SEC = 10800.0                    # diag_state output interval (3 h)
SPINUP_DAYS = 8.0                         # initial model days never loaded (spin-up)
EDGE_TRIM_DAYS = 20.0                      # filtered-signal days dropped at each end
                                          # (band-pass edge contamination, ~half band)

NZ, NY, NX = 138, 384, 512
DT_DTYPE   = '>f4'                        # big-endian float32
MISSING    = -999.0

# record index of each field in diag_state
FIELD_IDX = {'THETA': 0, 'SALT': 1, 'UVEL': 2, 'VVEL': 3,
             'WVEL': 4, 'PHIHYD': 5, 'DRHODR': 6}

_SLICE = NY * NX                         # elements in one horizontal level
_FIELD = NZ * _SLICE                     # elements in one full 3D field


def diag_iters():
    """Sorted iteration numbers with a diag_state file present."""
    its = [int(re.search(r'\.(\d+)\.data$', f).group(1))
           for f in glob.glob(os.path.join(RUN_DIR, 'diag_state.*.data'))]
    return np.array(sorted(its))


def iter_times(iters):
    """DatetimeIndex for the given iteration numbers."""
    return pd.DatetimeIndex(
        [pd.Timestamp(REF_DATE) + pd.Timedelta(seconds=int(i) * DELTA_T)
         for i in iters])


def iters_after_spinup(iters=None, spinup_days=SPINUP_DAYS):
    """Iteration numbers strictly later than `spinup_days` into the run."""
    iters = diag_iters() if iters is None else np.asarray(iters)
    return iters[iters * DELTA_T > spinup_days * 86400.0]


def edge_trim_samples(trim_days=EDGE_TRIM_DAYS, out_step_sec=OUT_STEP_SEC):
    """Number of output samples spanning `trim_days` (to drop at each end)."""
    return int(round(trim_days * 86400.0 / out_step_sec))


def edge_trim_mask(times, trim_days=EDGE_TRIM_DAYS):
    """Boolean mask keeping times at least `trim_days` inside both record ends.

    Applied to the *filtered* signal to drop band-pass edge contamination that
    lives within roughly one filter length of each boundary.
    """
    t = np.asarray(times)
    lo = t[0]  + np.timedelta64(int(trim_days * 86400), 's')
    hi = t[-1] - np.timedelta64(int(trim_days * 86400), 's')
    return (t >= lo) & (t <= hi)


def _path(it):
    return os.path.join(RUN_DIR, f'diag_state.{it:010d}.data')


def read_column(it, field, i, j):
    """Full-depth column (nz,) of `field` at grid point (i, j), NaN-masked."""
    base = FIELD_IDX[field] * _FIELD
    mm = np.memmap(_path(it), dtype=DT_DTYPE, mode='r')
    idx = base + (np.arange(NZ) * NY + j) * NX + i
    col = np.array(mm[idx]).astype('f8')
    col[col == MISSING] = np.nan
    return col


def read_columns(it, fields, ii, jj):
    """Columns for several fields at several (i, j) points from one file.

    Returns dict field -> array (n_pts, nz). One memmap open per file so all
    requested fields/points come from a single pass over that iteration.
    """
    ii = np.asarray(ii); jj = np.asarray(jj)
    mm = np.memmap(_path(it), dtype=DT_DTYPE, mode='r')
    out = {}
    for f in fields:
        base = FIELD_IDX[f] * _FIELD
        cols = np.empty((len(ii), NZ), dtype='f8')
        for p in range(len(ii)):
            idx = base + (np.arange(NZ) * NY + jj[p]) * NX + ii[p]
            cols[p] = mm[idx]
        cols[cols == MISSING] = np.nan
        out[f] = cols
    return out


def read_field(it, field):
    """Full 3D field (nz, ny, nx) for one iteration, NaN-masked."""
    base = FIELD_IDX[field] * _FIELD
    fld = np.fromfile(_path(it), dtype=DT_DTYPE, count=_FIELD,
                      offset=base * 4).astype('f8').reshape(NZ, NY, NX)
    fld[fld == MISSING] = np.nan
    return fld


def read_field_subdomain(it, field, y0, y1, x0, x1):
    """Sub-block (nz, y1-y0, x1-x0) of `field`, NaN-masked.

    Reads each depth level's [y0:y1, x0:x1] window to avoid pulling the whole
    field into memory when only an equatorial band / longitude range is needed.
    """
    base = FIELD_IDX[field] * _FIELD
    mm = np.memmap(_path(it), dtype=DT_DTYPE, mode='r')
    ny_s, nx_s = y1 - y0, x1 - x0
    out = np.empty((NZ, ny_s, nx_s), dtype='f8')
    for k in range(NZ):
        lev = mm[base + k * _SLICE: base + (k + 1) * _SLICE].reshape(NY, NX)
        out[k] = lev[y0:y1, x0:x1]
    out[out == MISSING] = np.nan
    return out


def load_grid():
    """1D grid coordinates and vertical metrics via a light xmitgcm load.

    Returns dict with:
      lon (nx), lat (ny)           cell-center longitude/latitude (deg)
      Z (nz)                       cell-center depth (m, negative down)
      Zl (nz)                      upper-face depth (m, negative down)
      drF (nz)                     cell thickness (m, positive)
      hFacC (nz, ny, nx)           partial-cell fraction (0 = land)
    """
    from xmitgcm import open_mdsdataset
    its = diag_iters()[:2].tolist()
    ds = open_mdsdataset(RUN_DIR, grid_dir=GRID_DIR, iters=its,
                         prefix=['diag_state'], ref_date=REF_DATE,
                         delta_t=DELTA_T)
    xc = np.asarray(ds.XC.values); yc = np.asarray(ds.YC.values)
    lon = xc[0, :] if xc.ndim == 2 else xc
    lat = yc[:, 0] if yc.ndim == 2 else yc
    return dict(
        lon=lon, lat=lat,
        Z=np.asarray(ds.Z.values), Zl=np.asarray(ds.Zl.values),
        drF=np.asarray(ds.drF.values), hFacC=np.asarray(ds.hFacC.values))


def nearest_idx(coord1d, target):
    """Index of the grid point nearest `target` in a 1D coordinate array."""
    return int(np.abs(np.asarray(coord1d) - target).argmin())


# --- profile locations --------------------------------------------------------
# Only 140W is inside this nested domain (~150.6W-129.4W); 170W/110W requested
# earlier fall outside, so a single longitude is used (user-confirmed).
PROFILE_LON = {'140W': 220.0}                     # deg E
PROFILE_LAT = {'1S': -1.0, '0N': 0.0, '1N': 1.0,  # deg N
               '2N': 2.0, '3N': 3.0}
