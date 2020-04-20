import astropy.units as u
import numpy as np

from aia import get_prepped_aia_maps

import linalg


@u.quantity_input(temps=u.K,
                  K=u.cm**5 / u.s / u.pix,
                  data=1 / u.pix / u.s,
                  data_err=1 / u.pix / u.s,
                  xi0=u.cm**-5,
                  L=u.cm**5)
def dem_guess(temps, K, data, data_err, xi0, L, alpha):
    """
    Produce a refined guess for the DEM.

    This follows eqs (6) and (7) from HK12.

    Parameters
    ----------
    temps : astropy.quantity.Quantity
        Temperatures
    K : astropy.quantity.Quantity
        Temperature response function.
    data : astropy.quantity.Quantity
        Data.
    dat_err : astropy.quantity.Qauntity
        Data errors.
    xi0 : astropy.quantity.Quantity
        Initial DEM guess
    L : astropy.quantity.Quantity
        Constraint matrix.
    alpha : float
        Regularisation parameter.

    Returns
    -------
    dem : astropy.quantity.Quantity
        DEM estimate.
    """
    Ktilde = (K.T / data_err).T
    sva, svb, u, v, w = linalg.gsvd(Ktilde, L)
    lamb = linalg.inv_reg_param(sva, svb, u, w, xi0, data, K, data_err, alpha)
    return lamb
