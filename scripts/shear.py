"""Vertical shear of the horizontal velocity, on the irregular cell-center grid.

Companion to fluxes.py. Given horizontal velocity (U or V) on the cell-center
depth grid Z, the vertical shear is the derivative with respect to height z
(positive up). Because Z is stored negative-downward and *increasing* upward
(Z[0] nearest the surface), d/dz == d/dZ, so the shear is simply
np.gradient(vel, Z) -- the same finite-difference operator fluxes.vertical_convergence
uses, but WITHOUT the sign flip (convergence = -d(flux)/dZ, shear = +d(vel)/dZ).

Definitions used downstream:
  zonal shear      Sx = dU/dz
  meridional shear Sy = dV/dz
  total shear      |S| = sqrt(Sx**2 + Sy**2)

The background (mean) and band-matched (fluctuating) shear variants are formed by
the caller (apply `vertical_shear` to the time-mean velocity or to the velocity
perturbations, respectively); keeping these two operators thin lets the sign/grid
convention live in exactly one place.
"""

import numpy as np


def vertical_shear(vel, Z, zaxis=-1):
    """Vertical shear d(vel)/dz on the irregular Z grid (Z<0 down, z up).

    Z increases upward (toward 0 at the surface), so d/dz == d/dZ and no sign
    flip is needed. np.gradient handles the non-uniform spacing. NaNs (land /
    below seafloor) propagate to their neighbours, as for fluxes.vertical_convergence.
    """
    return np.gradient(np.asarray(vel, dtype='f8'), np.asarray(Z), axis=zaxis)


def shear_magnitude(Sx, Sy):
    """Total vertical-shear magnitude |S| = sqrt(Sx**2 + Sy**2)."""
    return np.sqrt(np.asarray(Sx, dtype='f8') ** 2 + np.asarray(Sy, dtype='f8') ** 2)
