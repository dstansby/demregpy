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
        Temperature bins within which to calculated the DEM.
    obs : dict
        A dictionary of Observation objects.

    Returns
    -------
    dem : astropy.quantity.Quantity
        DEM estimate.
    """
    ntemps = temps.size - 1
    nobs = len(obs)
    # Construct temperature response matrix
    K = np.zeros((nobs, ntemps)) * u.cm**5 / u.s / u.pix
    data = np.zeros(nobs) / u.s / u.pix
    data = [1.5015837, 1.7489849, 29.46018, 78.363713, 90.357413, 3.440197] / u.s / u.pix
    data_err = data.copy()
    # Fill up the K array
    #
    # Set pixel, REMOVE ME EVENTUALLY
    pixel = np.array([2310, 1800])
    for i, key in enumerate(obs):
        tresp = obs[key].temp_response.sample(temps)
        # Take average at the bin ends
        K[i, :] = (tresp[1:] + tresp[:-1]) / 2
        # data[i] = obs[key].intensity.data[pixel[0], pixel[1]] / u.s / u.pix
        data_err[i] = data[i] * 0.2    # REMOVE THIS ASSUMPTION EVENTUALLY
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

    guess2 = guess.copy()
    for i in range(0):
        # Take a second guess
        L = np.identity(ntemps) * np.mean(K) * np.mean(guess2) / guess2 / np.mean(data)
        alpha = 1
        guess2 = dem_guess(temps, K, data, data_err, guess2, L, alpha)

    intensity_guess = K @ guess2
    # Convert from EM to DEM
    guess /= np.diff(temps)
    guess2 /= np.diff(temps)
    return guess, guess2, intensity_guess


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
    # print(w.shape)
    '''import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for i in range(w.shape[1]):
        ax.plot(np.geomspace(0.1, 100, 100) * 1e6, w[:, i], label=str(i))
    ax.set_xscale('log')
    ax.legend()'''

    data_tilde = data / err
    # Number of mu values to try out
    nmu = 200
    ntemps = w.shape[0]
    nobs = w.shape[1]

    # Store each term in the sum for each mu value
    xis = np.zeros((ntemps, nobs, nmu)) * xi0.unit

    phis = sva / svb
    minx = np.min(phis)
    maxx = np.max(phis)
    # Choose a range of test values between the min/max values of phi
    mus = np.geomspace(minx**2 * 1e-2, maxx**2, nmu)

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
    terms = (intensity_guess - data) / err
    norm = np.sum(terms**2, axis=1)
    # print(f'Norm is {norm}')
    tomin = np.abs(norm - reg_tweak * data.size)
    # if force_positive:
    # tomin[np.sum(dem_guess < 0, axis=0) > 0] = np.nan
    muidx = np.nanargmin(np.abs(tomin))
    mu = mus[muidx]
    '''import matplotlib.pyplot as plt
    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].plot(mus, np.sum(dem_guess < 0, axis=0))
    axs[1].plot(mus, tomin)
    axs[0].set_xscale('log')
    for phi in phis**2:
        axs[0].axvline(phi, color='k')'''

    if muidx == 0:
        logger.warn('Selected smallest value of mu')
    return mus[muidx], dem_guess[:, muidx]
