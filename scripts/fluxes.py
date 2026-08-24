"""Wave momentum- and energy-flux quantities from band-passed perturbations.

Given band-passed velocity perturbations (u', v', w') and the band-passed
pressure-potential perturbation (phi' = PHIHYD'), this module forms:

  momentum flux (Reynolds-stress) covariances, kinematic units m^2 s^-2:
      u'u', v'v', w'w', u'v', u'w', v'w'
  energy flux (pressure work), units W m^-2 = kg s^-3:
      F = rho0 * phi' * (u', v', w')
  vertical convergence of the vertical momentum fluxes, units m s^-2:
      -d/dz (u'w'),  -d/dz (v'w')

Conventions / assumptions:
- PHIHYD is the hydrostatic pressure potential anomaly p'/rho0 (m^2 s^-2);
  pressure perturbation p' = RHO0 * PHIHYD. Energy flux uses p' * u'_i.
- The C-grid half-cell horizontal stagger between U, V, W, PHIHYD points is
  neglected (fields treated as co-located at their stored indices). The
  vertical stagger of WVEL is handled: WVEL sits on cell top faces (Zl) and is
  averaged to cell centers (Z) before forming w-products, so all covariances
  and the vertical derivative live on the cell-center Z grid.
- Z is irregularly spaced; the vertical derivative uses np.gradient(., Z) which
  accounts for non-uniform spacing. Z is negative downward (z positive up), so
  convergence = -d(flux)/dZ.
"""

import numpy as np

RHO0 = 1027.0        # rhonil from the run's data namelist (kg m^-3)

# the six momentum-flux (velocity covariance) components
MOM_COMPONENTS = ['uu', 'vv', 'ww', 'uv', 'uw', 'vw']
# the three energy-flux (pressure work) components
ENE_COMPONENTS = ['pu', 'pv', 'pw']


def w_to_center(w, zaxis=-1):
    """Average WVEL from cell top faces (Zl) to cell centers (Z).

    w_center[k] = 0.5*(w[k] + w[k+1]); deepest center reuses its top face.
    """
    w = np.moveaxis(np.asarray(w, dtype='f8'), zaxis, -1)
    wc = np.empty_like(w)
    wc[..., :-1] = 0.5 * (w[..., :-1] + w[..., 1:])
    wc[..., -1] = w[..., -1]
    return np.moveaxis(wc, -1, zaxis)


def w_faces_to_centers(w, zaxis=-1):
    """Average WVEL from nz+1 cell top faces (Zl) to nz cell centers (Z).

    For TPOSE6, WVEL is stored on 66 Zl faces while T/U/V live on 65 Z centers;
    center k = 0.5*(face[k] + face[k+1]). Returns an array with the depth axis
    reduced by one. (Distinct from `w_to_center`, which maps equal-length Zl->Z.)
    """
    w = np.moveaxis(np.asarray(w, dtype='f8'), zaxis, -1)
    wc = 0.5 * (w[..., :-1] + w[..., 1:])
    return np.moveaxis(wc, -1, zaxis)


def momentum_fluxes(up, vp, wp):
    """Instantaneous kinematic momentum-flux components from perturbations.

    Inputs are band-passed perturbations of matching shape (w already on
    cell centers). Returns dict component -> product array (same shape).
    """
    return {'uu': up * up, 'vv': vp * vp, 'ww': wp * wp,
            'uv': up * vp, 'uw': up * wp, 'vw': vp * wp}


def energy_fluxes(phip, up, vp, wp):
    """Instantaneous pressure-work energy-flux components (W m^-2).

    F_i = rho0 * phi' * u'_i. Returns dict {'pu','pv','pw'}.
    """
    p = RHO0 * phip
    return {'pu': p * up, 'pv': p * vp, 'pw': p * wp}


def vertical_convergence(flux, Z, zaxis=-1):
    """Vertical convergence -d(flux)/dZ on the irregular Z grid (Z<0 down)."""
    dFdz = np.gradient(np.asarray(flux, dtype='f8'), np.asarray(Z), axis=zaxis)
    return -dFdz


def layer_integral(field, drF, hFacC, layer_mask, zaxis=0):
    """Depth integral of `field` over a layer, weighting by partial cells.

    field, hFacC broadcast over (nz, ...); drF is (nz,); layer_mask is a (nz,)
    boolean selecting the layer's levels. Integral = sum_k field*drF*hFacC over
    the selected levels (units field * m). NaNs are treated as zero thickness.
    """
    field = np.moveaxis(np.asarray(field, dtype='f8'), zaxis, 0)
    hf = np.moveaxis(np.asarray(hFacC, dtype='f8'), zaxis, 0)
    dz = (drF[:, None, None] if field.ndim == 3 else drF.reshape(
        [-1] + [1] * (field.ndim - 1)))
    thick = dz * hf
    w = np.where(np.isfinite(field), thick, 0.0)
    f = np.where(np.isfinite(field), field, 0.0)
    sel = layer_mask.reshape([-1] + [1] * (field.ndim - 1))
    return np.sum(np.where(sel, f * w, 0.0), axis=0)


def layer_thickness(field, drF, hFacC, layer_mask, zaxis=0):
    """Wet thickness (m) of a layer where `field` is defined (finite).

    Sum of drF*hFacC over the layer's selected levels, counting only cells where
    `field` is finite (so land/below-seafloor cells contribute no thickness).
    Matches the weighting used by `layer_integral`, so integral/thickness is the
    partial-cell-weighted depth average of `field` over the layer.
    """
    field = np.moveaxis(np.asarray(field, dtype='f8'), zaxis, 0)
    hf = np.moveaxis(np.asarray(hFacC, dtype='f8'), zaxis, 0)
    dz = (drF[:, None, None] if field.ndim == 3 else drF.reshape(
        [-1] + [1] * (field.ndim - 1)))
    w = np.where(np.isfinite(field), dz * hf, 0.0)
    sel = layer_mask.reshape([-1] + [1] * (field.ndim - 1))
    return np.sum(np.where(sel, w, 0.0), axis=0)


def layer_average(field, drF, hFacC, layer_mask, zaxis=0):
    """Partial-cell-weighted depth average of `field` over a layer.

    = layer_integral / layer_thickness, in the native units of `field`. Columns
    with no wet cells in the layer (thickness 0) are returned as NaN. Removes the
    layer-thickness bias of `layer_integral`, so layers of unequal thickness are
    comparable on one color scale.
    """
    integral = layer_integral(field, drF, hFacC, layer_mask, zaxis)
    H = layer_thickness(field, drF, hFacC, layer_mask, zaxis)
    with np.errstate(invalid='ignore', divide='ignore'):
        avg = integral / H
    avg[H == 0] = np.nan
    return avg


# depth-layer boundaries (m, positive down) for the map integrals
LAYER_BOUNDS = [(0, 350), (350, 700), (700, 1500), (1500, None)]


def layer_masks(Z):
    """Boolean (nz,) masks for each LAYER_BOUNDS entry plus a full-column mask.

    Z is cell-center depth (m, negative down). Returns dict label -> mask.
    None as a lower bound means "to the bottom".
    """
    depth = -np.asarray(Z)          # positive down
    masks = {}
    for top, bot in LAYER_BOUNDS:
        lbl = f'{top}-{bot}m' if bot is not None else f'{top}m-bottom'
        hi = np.inf if bot is None else bot
        masks[lbl] = (depth >= top) & (depth < hi)
    masks['full'] = np.ones_like(depth, dtype=bool)
    return masks
