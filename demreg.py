import logging

import astropy.units as u
import numpy as np

import linalg

logger = logging.getLogger(__name__)


@u.quantity_input(temps=u.K)
def calc_dem(temps, obs):
    """
    Calculate the DEM.

    Parameters
    ----------
    temps : astropy.quantity.Quantity
        Temperatures at which to calculate the DEM.
    obs : dict
        A dictionary of Observation objects.

    Returns
    -------
    dem : astropy.quantity.Quantity
        DEM estimate.
    """
    ntemps = temps.size
    nobs = len(obs)
    # Construct temperature response matrix
    K = np.zeros((nobs, ntemps)) * u.cm**5 / u.s / u.pix
    # Fill up the K array
    for i, key in enumerate(obs):
        K[i, :] = maps[key].temp_response.sample(temps)
        data[i] = maps[key].intensity.data[pixel[0], pixel[1]]
    # First estimate
    xi0 = np.zeros(ntemps) * data.unit / K.unit
    # First constraint matrix. The exact value of this doesn't matter, but we
    # set the order of magnitude to try and avoid close to zero or close to one
    # floating point computations when calculating the generalised SVD.
    L = np.identity(ntemps) * np.mean(K) / np.mean(data)
    alpha = 10
    guess = dem_guess(temps, K, data, data_err, xi0, L, alpha)

    # Only take positive values within a certain range of the maximum value,
    # and set the rest to small postiive values. Note: this is done in the IDL
    # implementation, but I'm not sure why and it doesn't seem to be referenced
    # in the paper.
    reltol = 1e-4
    thresh = reltol * np.max(guess)
    guess[guess < thresh] = reltol * np.max(guess)

    # Take a second guess
    L = np.identity(ntemps) * np.mean(K) / np.mean(data)
    alpha = 1
    guess2 = dem_guess(temps, K, data, data_err, guess, L, alpha)

    return guess, guess2


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
    lamb, guess = inv_reg_param(xi0, data, K, data_err, alpha, L)
    return guess


def inv_reg_param(xi0, data, K, err, reg_tweak, L):
    """
    Compute the regularisation parameter.

    Parameters
    ----------
    sva:
        vector, generalised singular values
    svb:
        vector, generalised singular values
    u:
        matrix, GSVD matrix
    w:
        matrix, GSVD matrix
    data:
        vector, containing data (eg dn)
    err:
        vector, uncertanty on data (same units and dimension)
    reg_tweak:
        scalar, parameter to adjusting regularization (chisq)

    Returns
    -------
    opt:
        regularization parameter
    dem_guess:
        DEM guess using the given parameter
    """
    Ktilde = (K.T / err).T
    sva, svb, u, v, w = linalg.gsvd(Ktilde, L)

    data_tilde = data / err
    # Number of mu values to try out
    nmu = 20
    ntemps = w.shape[0]
    nobs = w.shape[1]

    # Store each term in the sum for each mu value
    xis = np.zeros((ntemps, nobs, nmu)) * xi0.unit

    phis = sva / svb
    minx = np.min(phis)
    maxx = np.max(phis)
    # Choose a range of test values between the min/max values of phi
    mus = np.geomspace(minx**2 * 1e-3, maxx**2, nmu)

    print(phis / sva**2)

    # Loop over the terms in the sum (one for each observation)
    for i in range(nobs):
        phi = phis[i]
        alpha = sva[i]
        term1 = np.dot(data_tilde, u[:, i]) * w[:, i] / alpha
        # Loop over candiate mu values
        for j, mu in enumerate(mus):
            # term2 = mu * xi0 / alpha**2
            factor = (phi ** 2) / (phi**2 + mu)
            xis[:, i, j] = factor * (term1)# + term2)
    # Sum over the observations
    dem_guess = np.sum(xis, axis=1)
    # print(f'DEM guess is {dem_guess}')
    # Forward model the DEM guess to an intensity guess
    intensity_guess = (K @ dem_guess).T
    # print(f'Intensity guess is {intensity_guess}')
    terms = (intensity_guess - data[0, :]) / err
    norm = np.sum(terms**2, axis=1)
    # print(f'Norm is {norm}')
    tomin = norm - reg_tweak * data.size
    muidx = np.argmin(np.abs(tomin))
    if muidx == 0:
        logger.warn('Selected smallest value of mu')
    return mus[muidx], dem_guess[:, muidx]
