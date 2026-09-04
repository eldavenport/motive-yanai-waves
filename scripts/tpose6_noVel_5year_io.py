"""IO helpers for the 5-year TPOSE6 "noVel" run (raw MITgcm mds binaries).

This is a DIFFERENT TPOSE6 run than the repo's TPOSE-Vel one (tpose6_io.py):
it does NOT assimilate velocity and spans 2012-2016. Fields THETA, SALT, UVEL,
VVEL, WVEL, DRHODR live in the `diag_state` diagnostic (daily) as mds binaries:

  2012:      /data/SO6/TPOSE_diags/tpose6/<month>/diags[...]/
  2013-2016: /data/SO3/averdy/TPOSE6/<month>/diags_daily/
  grid:      /data/SO6/TPOSE_diags/tpose6/grid_6/

Cleaned rewrite of EqMixRemix_Processing/Python/open_tpose.py::tpose2012to2016.
Bugs fixed relative to that reference:
  - `nov2012/diags_new/` no longer exists -> use `nov2012/diags/`.
  - Per-segment itPerFile taken from disk (48 vs 72 vary by run), not guessed.
  - Uniform offset handling: each 2-month assimilation run has a ~10-day
    adjustment transient, so every segment except the very first (jan2012) skips
    its first 10 days; the previous segment extends ~10 days to cover them. This
    matches the *dated* tpose2012to2014 variant (the one that yields clean,
    contiguous years) rather than the undated tpose2012to2016 (which special-
    cased jan2013/jan2014 with an off-by-one no-skip, double-covering the seam).
  - Real daily time coordinate assigned (the reference omitted ref_date/delta_t).

Coverage is derived from the "first-kept date" chain: jan2012 -> 2012-01-01, and
every later segment -> the 10th of its (odd) month. Segment keep-counts are the
differences between consecutive first-kept dates, so the union is exactly the
contiguous daily record 2012-01-01 .. 2016-12-31 (1827 days). No spin-up drop
(the analysis window is the whole record); the 20-day band-pass edge trim is kept.

Analysis conventions (domain, profiles, edge trim) mirror tpose6_io.py.
"""

import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from xmitgcm import open_mdsdataset

# xmitgcm emits one timedelta-decoding FutureWarning per open; silence the noise.
warnings.filterwarnings('ignore', message='.*decode.*timedelta.*')

# --- run configuration --------------------------------------------------------
GRID_DIR  = '/data/SO6/TPOSE_diags/tpose6/grid_6'
CACHE_DIR = '/data/SO3/edavenport/tpose6_noVel_5year/cache'
RHONIL    = 1035.0
EOS_TYPE  = 'JMD95Z'

DT_DAYS = 1.0                            # daily output
EDGE_TRIM_DAYS = 20.0                    # band-pass edge contamination trimmed
# NO spin-up drop -- whole record is used.

# map sub-domain (deg E / deg N): 175W-105W, 10S-10N  (matches tpose6_io)
MAP_LON = (185.0, 255.0)
MAP_LAT = (-10.0, 10.0)

# profile locations (matches tpose6_io)
PROFILE_LON = {'170W': 190.0, '140W': 220.0, '110W': 250.0}
PROFILE_LAT = {'1S': -1.0, '0N': 0.0, '1N': 1.0, '2N': 2.0, '3N': 3.0}

PREFIX = 'diag_state'                    # holds THETA SALT UVEL VVEL WVEL DRHODR
START_DATE = '2012-01-01'

_SO6 = '/data/SO6/TPOSE_diags/tpose6'
_SO3 = '/data/SO3/averdy/TPOSE6'

# ordered 2-month assimilation runs: (parent, folder, itPerFile).  itPerFile is
# the diag_state iteration step per day, read from disk (48 = 30-min timestep,
# 72 = 20-min timestep).
_RUNS = [
    (_SO6, 'jan2012/diags',             72),
    (_SO6, 'mar2012/diags',             72),
    (_SO6, 'may2012/diags',             72),
    (_SO6, 'jul2012/diags',             72),
    (_SO6, 'sep2012/diags_iter7_daily', 48),
    (_SO6, 'nov2012/diags',             72),
    (_SO3, 'jan2013/diags_daily',       72),
    (_SO3, 'mar2013/diags_daily',       72),
    (_SO3, 'may2013/diags_daily',       72),
    (_SO3, 'jul2013/diags_daily',       72),
    (_SO3, 'sep2013/diags_daily',       72),
    (_SO3, 'nov2013/diags_daily',       72),
    (_SO3, 'jan2014/diags_daily',       48),
    (_SO3, 'mar2014/diags_daily',       48),
    (_SO3, 'may2014/diags_daily',       72),
    (_SO3, 'jul2014/diags_daily',       72),
    (_SO3, 'sep2014/diags_daily',       72),
    (_SO3, 'nov2014/diags_daily',       72),
    (_SO3, 'jan2015/diags_daily',       72),
    (_SO3, 'mar2015/diags_daily',       48),
    (_SO3, 'may2015/diags_daily',       72),
    (_SO3, 'jul2015/diags_daily',       72),
    (_SO3, 'sep2015/diags_daily',       72),
    (_SO3, 'nov2015/diags_daily',       72),
    (_SO3, 'jan2016/diags_daily',       72),
    (_SO3, 'mar2016/diags_daily',       72),
    (_SO3, 'may2016/diags_daily',       72),
    (_SO3, 'jul2016/diags_daily',       72),
    (_SO3, 'sep2016/diags_daily',       72),
    (_SO3, 'nov2016/diags_daily',       72),
]

END_DATE = '2016-12-31'
OFFSET_DAYS = 10                         # spin-up transient skipped per run


def _first_kept(folder):
    """Contiguity anchor: 10th of the run's month, except the first run (Jan 1)."""
    mon = {'jan': 1, 'mar': 3, 'may': 5, 'jul': 7, 'sep': 9, 'nov': 11}
    name = folder.split('/')[0]                       # e.g. 'mar2013'
    m, y = mon[name[:3]], int(name[3:7])
    if (y, m) == (2012, 1):
        return pd.Timestamp('2012-01-01')
    return pd.Timestamp(year=y, month=m, day=10)


def segments():
    """Resolved segment table: dicts with data_dir, itPerFile, iters, n_days.

    keep-count = days to the next segment's first-kept date (last segment runs
    to END_DATE inclusive); start day = 1 for the first run, else OFFSET_DAYS+1.
    """
    firsts = [_first_kept(f) for _, f, _ in _RUNS]
    nexts = firsts[1:] + [pd.Timestamp(END_DATE) + pd.Timedelta(days=1)]
    segs = []
    for (parent, folder, ipf), d0, d1 in zip(_RUNS, firsts, nexts):
        keep = int((d1 - d0).days)
        start_day = 1 if d0 == pd.Timestamp(START_DATE) else OFFSET_DAYS + 1
        iters = [(start_day - 1 + d) * ipf + ipf for d in range(keep)]
        segs.append(dict(data_dir=os.path.join(parent, folder), itPerFile=ipf,
                         iters=iters, n_days=keep, first=d0))
    return segs


def date_index():
    """Clean daily DatetimeIndex 2012-01-01 .. 2016-12-31 (1827 days)."""
    return pd.date_range(START_DATE, END_DATE, freq='D')


# tpose6_io-compatible alias (build scripts call io.time_index())
def time_index():
    return date_index()


def open_5year(prefix=PREFIX, **kw):
    """Open the whole 2012-2016 daily record as one lazy Dataset.

    Concatenates the per-run mds datasets and assigns the clean daily time axis
    (the raw runs restart their iteration counters, so the mds `time` values are
    per-run and are replaced wholesale). Asserts the 1827-day total.
    """
    segs = segments()
    parts = []
    for s in segs:
        ds = open_mdsdataset(data_dir=s['data_dir'], grid_dir=GRID_DIR,
                             iters=s['iters'], prefix=[prefix], **kw)
        parts.append(ds)
    ds = xr.concat(parts, dim='time')
    dates = date_index()
    assert ds.sizes['time'] == len(dates), (
        f"got {ds.sizes['time']} timesteps, expected {len(dates)}")
    return ds.assign_coords(time=dates.values)


def load_grid():
    """1D coordinates and vertical metrics (same dict shape as tpose6_io)."""
    s = segments()[0]
    g = open_mdsdataset(data_dir=s['data_dir'], grid_dir=GRID_DIR,
                        iters=[s['iters'][0]], prefix=[PREFIX])
    grid = dict(
        lon=np.asarray(g.XC.values), lat=np.asarray(g.YC.values),
        lon_g=np.asarray(g.XG.values), lat_g=np.asarray(g.YG.values),
        Z=np.asarray(g.Z.values), Zl=np.asarray(g.Zl.values),
        drF=np.asarray(g.drF.values),
        hFacC=np.asarray(g.hFacC.values))
    g.close()
    return grid


def _hdim(da):
    """(y_dim_name, x_dim_name) for a data array on any C-grid position."""
    ydim = 'YG' if 'YG' in da.dims else 'YC'
    xdim = 'XG' if 'XG' in da.dims else 'XC'
    return ydim, xdim


def edge_trim_mask(times, trim_days=EDGE_TRIM_DAYS):
    """Boolean mask keeping times at least `trim_days` inside both record ends."""
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


if __name__ == '__main__':
    # self-check: segment table is contiguous daily and totals 1827 days
    segs = segments()
    total = sum(s['n_days'] for s in segs)
    print(f'{len(segs)} segments, {total} days (expect 1827)')
    assert total == 1827, total
    prev_last = None
    for s in segs:
        first, last = s['first'], s['first'] + pd.Timedelta(days=s['n_days'] - 1)
        if prev_last is not None:
            assert first == prev_last + pd.Timedelta(days=1), \
                f'seam gap/overlap before {s["data_dir"]}: {prev_last} -> {first}'
        prev_last = last
    print(f'contiguous {segs[0]["first"].date()} .. {prev_last.date()}  OK')
