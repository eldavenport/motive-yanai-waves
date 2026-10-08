"""Deep-jet depth-time plots: zonal velocity split into low vs high vertical modes.

For each of the three MOTIVE moorings (A 0.5N,140W; B 1.75N,138W; C 3N,140W) the
total zonal velocity profile U(z,t) is decomposed into *dynamical* vertical normal
modes -- the flat-bottom, rigid-lid baroclinic (pressure/horizontal-velocity) modes
p_n(z) of the local buoyancy frequency N^2(z). The reconstruction is split into
"low" modes (n=1-10) and "high" modes (n=11-20); the barotropic n=0 (depth mean)
is dropped. Because the N^2-weighted modes stretch the thermocline, the EUC now
projects onto the low modes (unlike flat cosines) and the deep jets onto the high.

One figure per model (mooring_deep_jets_Umodes_*.png), modes from the TIME-MEAN
N^2(z): 2 rows x 3 columns (columns = moorings, top = low modes, bottom = high),
y = depth (100-1000 m), x = time. Color scale shared across the three moorings per
row: +/-1 m/s (low), +/-0.3 m/s (high), cmo.balance, 100 filled-contour levels.

Assumptions (flagged):
  - N^2 from JMD95 in-situ density, parcels bracketed at the interface pressure to
    remove compressibility; floored at 1e-7 s^-2 so modes stay real through the
    mixed layer / weak inversions.
  - Flat-bottom, rigid-lid modes over the full wet column to the true bottom H.
Needs the stratification cache from build_strat_cache.py.
"""

import os
import importlib
import numpy as np
import xarray as xr
import cmocean
import scipy.linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import plot_moorings as pm
import eos_jmd95 as eos

NMODES = 20                       # highest mode number kept
ZMIN, ZMAX = 100.0, 1000.0        # plotted depth range (m, positive down)
N_LEVELS = 100                    # filled-contour levels
N2_FLOOR = 1e-7                   # s^-2, keeps the eigenproblem stably stratified
CLIM = {'low': 1.0, 'high': 0.3}  # shared symmetric color limits (m/s)
CTICK = {'low': 0.25, 'high': 0.1}   # colorbar tick step per band


def _hfac_drf(cfg):
    """Static (loc, depth) wet fraction and cell thickness from the mooring cache."""
    io = importlib.import_module(cfg['io'])
    ds = xr.open_dataset(f"{io.CACHE_DIR}/{cfg['cache']}")
    hfac, drF = ds['hFacC'].values, ds['drF'].values
    ds.close()
    return hfac, drF


def _load_strat(cfg, times):
    """Window-aligned THETA, SALT (time, loc, depth) at the mooring columns."""
    io = importlib.import_module(cfg['io'])
    f = os.path.join(io.CACHE_DIR, cfg['cache'].replace('mooring', 'strat'))
    if not os.path.exists(f):
        return None
    ds = xr.open_dataset(f).sel(time=times)
    th, sa = ds['THETA'].values, ds['SALT'].values
    ds.close()
    return th, sa


def n2_interfaces(theta, salt, zc):
    """Buoyancy frequency N^2 at the interfaces between wet cell centers.

    theta, salt: (nwet,) at centers; zc: center depth (m, positive down). Parcels
    are compared at the common interface pressure (adiabatic N^2), floored positive.
    """
    di = 0.5 * (zc[:-1] + zc[1:])                      # interface depth (m)
    p = eos.RHONIL * eos.GRAV * di / 1.0e4             # dbar
    ru = eos.densjmd95(salt[:-1], theta[:-1], p)       # upper parcel at p
    rl = eos.densjmd95(salt[1:], theta[1:], p)         # lower parcel at p
    n2 = (eos.GRAV / eos.RHONIL) * (rl - ru) / (zc[1:] - zc[:-1])
    return np.maximum(n2, N2_FLOOR)


def pmodes(n2i, zc, dz):
    """Flat-bottom baroclinic pressure modes from N^2 at interfaces.

    Solves d/dz((1/N^2) dp/dz) = -(1/c^2) p with Neumann BC (no normal flow at
    surface/bottom) as the symmetric generalized eigenproblem K p = (1/c^2) M p,
    M = diag(cell thickness). Returns V (nwet, NMODES+1), M-orthonormal, ordered
    by phase speed (mode 0 = barotropic), and M for the projection.
    """
    nz = len(zc)
    a = 1.0 / (n2i * (zc[1:] - zc[:-1]))               # interface coupling (nz-1,)
    K = np.zeros((nz, nz))
    i = np.arange(nz - 1)
    K[i, i] += a; K[i + 1, i + 1] += a
    K[i, i + 1] -= a; K[i + 1, i] -= a
    Mm = np.diag(dz)
    _, V = scipy.linalg.eigh(K, Mm)                    # ascending 1/c^2  ->  V^T M V = I
    return V[:, :NMODES + 1], Mm


def _recon(U, V, Mm):
    """Low- (modes 1-10) and high-mode (11-20) reconstructions of U (time, nwet)."""
    a = V.T @ (Mm @ U.T)                               # coeffs (NMODES+1, time); V^T M V = I
    return (V[:, 1:11] @ a[1:11]).T, (V[:, 11:21] @ a[11:21]).T


def _place(low_w, high_w, shape, wet):
    out = []
    for rw in (low_w, high_w):
        rec = np.full(shape, np.nan)
        rec[:, wet] = rw
        out.append(rec)
    return out


def split_mean(U, theta, salt, Z, drF, hfac):
    """Low/high reconstructions using modes from the TIME-MEAN N^2 profile."""
    wet = hfac > 0
    zc, dz = -Z[wet], (drF * hfac)[wet]
    n2 = n2_interfaces(np.nanmean(theta[:, wet], 0), np.nanmean(salt[:, wet], 0), zc)
    V, Mm = pmodes(n2, zc, dz)
    return _place(*_recon(U[:, wet], V, Mm), U.shape, wet)


def make_fig(M, cfg, strat, splitter, tag, basis_label):
    hfac, drF = _hfac_drf(cfg)
    Z, names = M['Z'], M['names']
    th, sa = strat
    xt = mdates.date2num(M['time'])                    # contourf needs numeric x
    zmask = (-Z >= ZMIN) & (-Z <= ZMAX)
    zz = Z[zmask]

    rec = {}                                           # (band, mooring) -> (time, depth)
    for p in range(len(names)):
        Utot = M['mean']['U'][p][None, :] + M['treat']['unfiltered']['u'][:, p, :]
        low, high = splitter(Utot, th[:, p, :], sa[:, p, :], Z, drF, hfac[p])
        rec['low', p], rec['high', p] = low, high

    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), sharex=True, sharey=True,
                             layout='constrained')
    for ri, (band, modes) in enumerate([('low', '1-10'), ('high', '11-20')]):
        v = CLIM[band]
        levels = np.linspace(-v, v, N_LEVELS + 1)
        for p in range(len(names)):
            cs = axes[ri, p].contourf(xt, zz, rec[band, p][:, zmask].T,
                                      levels=levels, cmap=cmocean.cm.balance,
                                      extend='both')
            axes[ri, p].xaxis_date()
        cb = fig.colorbar(cs, ax=list(axes[ri]), pad=0.01, shrink=0.9,
                          ticks=np.arange(-v, v + CTICK[band] / 2, CTICK[band]))
        cb.set_label('m s$^{-1}$')
        axes[ri, 0].set_ylabel(f'{band} modes ({modes})\ndepth (m)', fontsize=10)
    for p in range(len(names)):
        axes[0, p].set_title(pm._moor_label(names[p]), fontsize=11)
        axes[-1, p].set_xlabel('time')
        for lbl in axes[-1, p].get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_horizontalalignment('right')
    axes[0, 0].set_ylim(-ZMAX, -ZMIN)
    fig.suptitle(f'Equatorial deep jets — zonal velocity {basis_label} '
                 f'({ZMIN:.0f}-{ZMAX:.0f} m)')
    fname = f"{M['fig_dir']}/mooring_deep_jets_{tag}_{ZMIN:.0f}-{ZMAX:.0f}m.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


VARIANTS = [('Umodes', split_mean, 'vertical modes (mean N²)')]


def main(models=None):
    for model in (models or list(pm.MODELS)):
        cfg = pm.MODELS[model]
        path = f"{importlib.import_module(cfg['io']).CACHE_DIR}/{cfg['cache']}"
        if not os.path.exists(path):
            print(f'SKIP {model}: mooring cache not found ({cfg["cache"]})')
            continue
        os.makedirs(cfg['fig_dir'], exist_ok=True)
        M = pm.load_model(cfg)
        strat = _load_strat(cfg, M['time'])
        if strat is None:
            print(f'SKIP {model}: strat cache missing (run build_strat_cache.py)')
            continue
        for tag, splitter, label in VARIANTS:
            print('wrote', make_fig(M, cfg, strat, splitter, tag, label))


def _selfcheck():
    """pmodes on constant N^2 must give c_n = N H/(n pi) and cosine structure."""
    N, H, nz = 3e-3, 1000.0, 200
    dz = np.full(nz, H / nz)
    zc = np.cumsum(dz) - dz / 2
    n2i = np.full(nz - 1, N ** 2)
    V, Mm = pmodes(n2i, zc, dz)
    a = 1.0 / (n2i * (zc[1:] - zc[:-1]))
    K = np.zeros((nz, nz)); i = np.arange(nz - 1)
    K[i, i] += a; K[i + 1, i + 1] += a; K[i, i + 1] -= a; K[i + 1, i] -= a
    for n in range(1, 6):
        p = V[:, n]
        c = 1 / np.sqrt((p @ K @ p) / (p @ Mm @ p))
        assert abs(c - N * H / (n * np.pi)) < 1e-3, (n, c)
        assert abs(np.corrcoef(p, np.cos(n * np.pi * zc / H))[0, 1]) > 0.999
    print('pmodes self-check OK')


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['test']:
        _selfcheck()
    else:
        main(sys.argv[1:] or None)
