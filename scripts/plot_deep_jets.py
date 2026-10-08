"""Deep-jet depth-time plots: zonal velocity split into low vs high vertical modes.

For each of the three MOTIVE moorings (A 0.5N,140W; B 1.75N,138W; C 3N,140W) the
total zonal velocity profile U(z,t) is decomposed into flat-bottom vertical normal
modes cos(n*pi*z/H), n=0..20, fit over the full wet water column at each time (a
cell-thickness-weighted least-squares Galerkin projection). The reconstruction is
then split into "low" modes (n=1-10) and "high" modes (n=11-20); the barotropic
n=0 (depth mean) is dropped from both. Equatorial deep jets are the stacked
alternating-sign structure that lives in the high modes.

One figure per model: 2 rows x 3 columns (columns = moorings, top row = low modes,
bottom row = high modes), y = depth (100-1000 m), x = time. Color scale is shared
across the three moorings per row: +/-1 m/s (low), +/-0.3 m/s (high), cmo.balance.

ponytail: flat-bottom (constant-N^2) cosine modes, not true dynamical modes -- the
mooring caches carry no THETA/SALT to build N^2(z). Fine for separating vertical
scales; swap in Sturm-Liouville modes if a stratification profile is ever cached.
"""

import os
import importlib
import numpy as np
import xarray as xr
import cmocean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plot_moorings as pm

NMODES = 20                       # highest mode number in the fit
LOW = slice(1, 11)                # modes 1-10
HIGH = slice(11, 21)              # modes 11-20
ZMIN, ZMAX = 100.0, 1000.0        # plotted depth range (m, positive down)
CLIM = {'low': 1.0, 'high': 0.3}  # shared symmetric color limits (m/s)


def _hfac_drf(cfg):
    """Static (loc, depth) wet fraction and cell thickness from the mooring cache."""
    io = importlib.import_module(cfg['io'])
    ds = xr.open_dataset(f"{io.CACHE_DIR}/{cfg['cache']}")
    hfac, drF = ds['hFacC'].values, ds['drF'].values
    ds.close()
    return hfac, drF


def mode_split(U, Z, drF, hfac):
    """Low- and high-mode reconstructions of U(time, depth) for one mooring.

    U(z) is interpolated onto a uniform 5-m grid (where the cos(n*pi*z/H) modes
    are orthogonal, so the projection is well conditioned -- a direct least-squares
    fit on the native stretched grid blows up for n up to 20), projected onto
    modes n=0..NMODES, then the modes 1-10 (low) and 11-20 (high) reconstructions
    are mapped back onto the native depths. Returns (U_low, U_high) each
    (time, depth) with below-bottom cells NaN; the barotropic n=0 is dropped.
    """
    wet = hfac > 0
    zc = -Z[wet]                                  # cell-center depth, positive down
    H = float((drF * hfac)[wet].sum())             # water-column depth
    zu = np.linspace(0.0, H, max(200, int(H / 5)))
    Uw = U[:, wet]

    Uu = np.array([np.interp(zu, zc, Uw[t]) for t in range(U.shape[0])])  # (time, nz)
    n = np.arange(NMODES + 1)
    Phi = np.cos(np.pi * np.outer(zu, n) / H)      # (nz, NMODES+1)
    w = np.full(zu.size, zu[1] - zu[0]); w[0] = w[-1] = w[0] / 2   # trapezoid weights
    cn = np.where(n == 0, 1.0, 2.0) / H
    a = cn[None, :] * ((Uu * w) @ Phi)             # mode coeffs (time, NMODES+1)

    out = []
    for sl in (LOW, HIGH):
        recu = a[:, sl] @ Phi[:, sl].T             # reconstruct on uniform grid
        rec = np.full(U.shape, np.nan)
        rec[:, wet] = np.array([np.interp(zc, zu, recu[t])
                                for t in range(U.shape[0])])
        out.append(rec)
    return out


def eof_split(U, Z, drF, hfac):
    """Statistical-mode (EOF/PCA) analog of mode_split.

    U(time, depth) over the wet column is decomposed by SVD into empirical
    orthogonal functions, ranked by variance (cells weighted by sqrt of thickness
    so coarse deep cells don't dominate the variance inner product). The field is
    NOT time-centered, so EOF 1 carries the mean + dominant pattern -- keeping the
    quasi-steady deep-jet structure, parallel to the dynamical-mode figure.
    Low = EOFs 1-10, high = EOFs 11-20. Returns (U_low, U_high) each (time, depth),
    below-bottom cells NaN.
    """
    wet = hfac > 0
    sw = np.sqrt((drF * hfac)[wet])                # sqrt cell thickness
    Uc, S, Vt = np.linalg.svd(U[:, wet] * sw[None, :], full_matrices=False)

    out = []
    for a, b in ((0, 10), (10, 20)):
        recw = (Uc[:, a:b] * S[a:b]) @ Vt[a:b]     # reconstruct weighted field
        rec = np.full(U.shape, np.nan)
        rec[:, wet] = recw / sw[None, :]
        out.append(rec)
    return out


# decomposition variants: tag -> (splitter, basis label for the title)
VARIANTS = {
    'Umodes': (mode_split, 'vertical modes'),
    'Ueof':   (eof_split,  'statistical modes (EOF)'),
}


def make_fig(M, cfg, split_fn, tag, basis_label):
    hfac, drF = _hfac_drf(cfg)
    Z, t, names = M['Z'], M['time'], M['names']
    zmask = (-Z >= ZMIN) & (-Z <= ZMAX)
    zz = Z[zmask]

    # (band, mooring) -> (time, depth) reconstruction
    rec = {}
    for p in range(len(names)):
        Utot = M['mean']['U'][p][None, :] + M['treat']['unfiltered']['u'][:, p, :]
        low, high = split_fn(Utot, Z, drF, hfac[p])
        rec['low', p], rec['high', p] = low, high

    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), sharex=True, sharey=True,
                             layout='constrained')
    rows = [('low', '1-10'), ('high', '11-20')]
    for ri, (band, modes) in enumerate(rows):
        v = CLIM[band]
        for p in range(len(names)):
            pc = axes[ri, p].pcolormesh(t, zz, rec[band, p][:, zmask].T,
                                        cmap=cmocean.cm.balance, vmin=-v, vmax=v,
                                        shading='nearest')
        cb = fig.colorbar(pc, ax=list(axes[ri]), pad=0.01, shrink=0.9)
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


def main(models=None):
    for model in (models or list(pm.MODELS)):
        cfg = pm.MODELS[model]
        path = (f"{importlib.import_module(cfg['io']).CACHE_DIR}/{cfg['cache']}")
        if not os.path.exists(path):
            print(f'SKIP {model}: cache not found ({cfg["cache"]})')
            continue
        os.makedirs(cfg['fig_dir'], exist_ok=True)
        M = pm.load_model(cfg)
        for tag, (split_fn, basis_label) in VARIANTS.items():
            print('wrote', make_fig(M, cfg, split_fn, tag, basis_label))


if __name__ == '__main__':
    import sys
    main(sys.argv[1:] or None)
