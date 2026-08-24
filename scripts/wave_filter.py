"""Temporal band-pass filtering for the Yanai-wave band.

The Yanai band is defined here (per this analysis) as 15-40 day periods. The
diag_state output is 3-hourly, so the sampling interval is fixed at 3 h. A
zero-phase Butterworth band-pass (filtfilt) is applied along the time axis;
zero-phase filtering avoids shifting wave phase, which matters for the
covariances (u'w', p'w', ...) computed from the filtered fields.

Assumptions:
- Uniform 3-hourly sampling (diag_state is written every 10800 s).
- Filtering is applied along the first axis (time) unless `axis` is given.
- NaNs (land / below seafloor) are handled by filtering only finite columns;
  all-finite requirement per column, columns with any NaN are returned NaN.
"""

import numpy as np
from scipy.signal import butter, filtfilt

# band definition (days) and fixed sampling interval (days)
YANAI_MIN_DAYS = 15.0
YANAI_MAX_DAYS = 40.0
DT_DAYS = 10800.0 / 86400.0        # 3-hourly output = 0.125 day


def bandpass_sos(period_min_days=YANAI_MIN_DAYS,
                 period_max_days=YANAI_MAX_DAYS,
                 dt_days=DT_DAYS, order=4):
    """Butterworth band-pass coefficients (b, a) for the given period band."""
    fs = 1.0 / dt_days                      # samples per day
    f_lo = 1.0 / period_max_days            # low-frequency edge (long period)
    f_hi = 1.0 / period_min_days            # high-frequency edge (short period)
    nyq = fs / 2.0
    return butter(order, [f_lo / nyq, f_hi / nyq], btype='band')


def bandpass(x, axis=0, **kw):
    """Zero-phase band-pass filter `x` along `axis`.

    Columns containing any NaN along `axis` are returned as NaN (land/seafloor).
    """
    b, a = bandpass_sos(**kw)
    x = np.asarray(x, dtype='f8')
    x = np.moveaxis(x, axis, 0)
    nt = x.shape[0]
    flat = x.reshape(nt, -1)
    out = np.full_like(flat, np.nan)
    good = np.all(np.isfinite(flat), axis=0)
    if good.any():
        out[:, good] = filtfilt(b, a, flat[:, good], axis=0)
    out = out.reshape(x.shape)
    return np.moveaxis(out, 0, axis)


def bandpass_inplace(arr, chunk_cols=2_000_000, **kw):
    """Band-pass a large float32 array (nt, ...) in place, low peak memory.

    Time is axis 0. Columns are filtered in blocks so only one block is upcast
    to float64 at a time (avoids doubling a multi-GB array). NaN columns
    (land/seafloor) are left as NaN. Modifies `arr` and returns it.
    """
    b, a = bandpass_sos(**kw)
    nt = arr.shape[0]
    flat = arr.reshape(nt, -1)           # view; no copy for C-contiguous arr
    ncols = flat.shape[1]
    for c0 in range(0, ncols, chunk_cols):
        blk = flat[:, c0:c0 + chunk_cols]
        good = np.all(np.isfinite(blk), axis=0)
        if good.any():
            filt = filtfilt(b, a, blk[:, good].astype('f8'), axis=0)
            blk[:, good] = filt.astype(arr.dtype)
        blk[:, ~good] = np.nan
    return arr
