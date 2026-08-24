"""JMD95 equation of state and hydrostatic pressure-potential reconstruction.

The TPOSE6 TPOSE-Vel output has no PHIHYD diagnostic, so the hydrostatic
pressure-potential anomaly used for the energy-flux (pressure work) must be
rebuilt from THETA and SALT. The run uses `eosType = 'JMD95Z'` (Jackett &
McDougall 1995, with pressure taken from the reference depth rather than the
prognostic pressure), rhonil = 1035 kg m^-3.

`densjmd95` ports the MITgcm densjmd95.F polynomial (in-situ density from
potential temperature, practical salinity and pressure in dbar). `phihyd_from_ts`
integrates the hydrostatic relation to give phi_hyd = p_hyd / rhonil (m^2 s^-2),
matching MITgcm's PHIHYD diagnostic (hydrostatic part only; the free-surface /
barotropic loading g*eta is NOT included, consistent with the TPOSE24 PHIHYD).

Because the analysis band-passes phi in time, any time-constant reference
(reference density profile, PHrefC) cancels and is irrelevant to the resulting
phi'. The JMD95Z depth-based reference pressure is used so density does not
depend on the (unavailable) prognostic pressure.
"""

import numpy as np

RHONIL = 1035.0      # rhonil from the TPOSE6 data namelist (kg m^-3)
GRAV = 9.81          # gravity (m s^-2), MITgcm default


def densjmd95(s, theta, p):
    """In-situ density (kg m^-3) from JMD95.

    s     : practical salinity (PSU)
    theta : potential temperature (deg C)
    p     : pressure (dbar)
    Array inputs broadcast together. Port of MITgcm pkg/model densjmd95.F.
    """
    s = np.asarray(s, dtype='f8')
    t = np.asarray(theta, dtype='f8')
    # JMD95 secant-bulk-modulus polynomial expects pressure in bar (1 bar = 10 dbar)
    p = np.asarray(p, dtype='f8') / 10.0

    # coefficients for density of fresh water at p = 0
    eosJMDCFw = [999.842594, 6.793952e-02, -9.095290e-03,
                 1.001685e-04, -1.120083e-06, 6.536332e-09]
    # coefficients for density of sea water at p = 0
    eosJMDCSw = [8.244930e-01, -4.089900e-03, 7.643800e-05,
                 -8.246700e-07, 5.387500e-09,
                 -5.724660e-03, 1.022700e-04, -1.654600e-06,
                 4.831400e-04]

    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    s = np.maximum(s, 0.0)          # guard against tiny negative salinities
    s3o2 = s * np.sqrt(s)

    rho_fw = (eosJMDCFw[0] + eosJMDCFw[1] * t + eosJMDCFw[2] * t2
              + eosJMDCFw[3] * t3 + eosJMDCFw[4] * t4 + eosJMDCFw[5] * t4 * t)

    dens_p0 = (rho_fw
               + s * (eosJMDCSw[0] + eosJMDCSw[1] * t + eosJMDCSw[2] * t2
                      + eosJMDCSw[3] * t3 + eosJMDCSw[4] * t4)
               + s3o2 * (eosJMDCSw[5] + eosJMDCSw[6] * t + eosJMDCSw[7] * t2)
               + eosJMDCSw[8] * s * s)

    # secant bulk modulus K(S,theta,p)
    # pure water terms
    eosJMDCKFw = [1.965933e+04, 1.444304e+02, -1.706103e+00,
                  9.648704e-03, -4.190253e-05]
    # salt water terms
    eosJMDCKSw = [5.284855e+01, -3.101089e-01, 6.283263e-03,
                  -5.084188e-05, 3.886640e-01, 9.085835e-03, -4.619924e-04]
    # pressure terms
    eosJMDCKP = [3.186519e+00, 2.212276e-02, -2.984642e-04, 1.956415e-06,
                 6.704388e-03, -1.847318e-04, 2.059331e-07, 1.480266e-05,
                 2.102898e-04, -1.202016e-05, 1.394680e-07, -2.040237e-06,
                 6.128773e-08, 6.207323e-10]

    bulk_fw = (eosJMDCKFw[0] + eosJMDCKFw[1] * t + eosJMDCKFw[2] * t2
               + eosJMDCKFw[3] * t3 + eosJMDCKFw[4] * t4)
    bulk_sw = (bulk_fw
               + s * (eosJMDCKSw[0] + eosJMDCKSw[1] * t + eosJMDCKSw[2] * t2
                      + eosJMDCKSw[3] * t3)
               + s3o2 * (eosJMDCKSw[4] + eosJMDCKSw[5] * t + eosJMDCKSw[6] * t2))
    bulk = (bulk_sw
            + p * (eosJMDCKP[0] + eosJMDCKP[1] * t + eosJMDCKP[2] * t2
                   + eosJMDCKP[3] * t3)
            + p * s * (eosJMDCKP[4] + eosJMDCKP[5] * t + eosJMDCKP[6] * t2)
            + p * s3o2 * eosJMDCKP[7]
            + p * p * (eosJMDCKP[8] + eosJMDCKP[9] * t + eosJMDCKP[10] * t2)
            + p * p * s * (eosJMDCKP[11] + eosJMDCKP[12] * t + eosJMDCKP[13] * t2))

    return dens_p0 / (1.0 - p / bulk)


def pref_dbar(Z, rhonil=RHONIL, grav=GRAV):
    """Reference (depth-based) pressure in dbar at cell-center depths Z (m, <0).

    JMD95Z uses pressure from the reference depth: p = rhonil*g*|z| (Pa) / 1e4.
    """
    depth = -np.asarray(Z, dtype='f8')          # positive down (m)
    return rhonil * grav * depth / 1.0e4        # Pa -> dbar


def phihyd_from_ts(theta, salt, Z, drF, rhonil=RHONIL, grav=GRAV, zaxis=0):
    """Hydrostatic pressure-potential anomaly phi = p_hyd/rhonil (m^2 s^-2).

    theta, salt : arrays with a depth axis `zaxis` (levels ordered surface->deep,
                  matching Z and drF). Land/missing may be NaN.
    Z           : cell-center depth (m, negative down), length nz.
    drF         : cell thickness (m, positive), length nz.

    In-situ density is computed with JMD95 at the depth-reference pressure, then
    integrated hydrostatically from the surface:
        phi(z_k) = (g/rhonil) * [ sum_{m<k} rho_m drF_m + 0.5 rho_k drF_k ].
    Only relative (time-varying, band-passed) values are used downstream, so the
    absence of an explicit density reference / surface term does not matter.
    """
    theta = np.moveaxis(np.asarray(theta, dtype='f8'), zaxis, 0)
    salt = np.moveaxis(np.asarray(salt, dtype='f8'), zaxis, 0)
    nz = theta.shape[0]
    p = pref_dbar(Z, rhonil, grav)              # (nz,)
    shp = [nz] + [1] * (theta.ndim - 1)
    rho = densjmd95(salt, theta, p.reshape(shp))
    dz = np.asarray(drF, dtype='f8').reshape(shp)

    rdz = rho * dz                              # rho * thickness per level
    # cumulative mass above each cell center: sum of full cells above + half self
    cum_above = np.cumsum(rdz, axis=0) - rdz    # exclusive prefix sum
    phi = (grav / rhonil) * (cum_above + 0.5 * rdz)
    phi = np.where(np.isfinite(rho), phi, np.nan)
    return np.moveaxis(phi, 0, zaxis)
