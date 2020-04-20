import logging

import numpy as np
import astropy.units as u

logger = logging.getLogger(__name__)


def gsvd(A, B):
    # Generalised Singular Value Decomposition
    AB1 = A @ np.linalg.inv(B)
    u, s, vt = np.linalg.svd(AB1, full_matrices=False)

    beta = 1 / np.sqrt(1 + s**2)
    alpha = s * beta
    if np.any(np.isclose(alpha, 1, rtol=0, atol=1e-8)):
        logger.warn('Some singular value values very close to 1')
    if np.any(np.isclose(beta, 1, rtol=0, atol=1e-8)):
        logger.warn('Some singular value values very close to 1')

    wT = np.linalg.inv(np.diag(alpha)) @ u.T @ A
    wT /= wT.unit **2
    return alpha, beta, u, vt.T, wT.T


def inv_reg_param(sva, svb, u, w, xi0, data, K, err, reg_tweak):
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
    """
    data_tilde = data / err
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

    # Loop over the terms in the sum (one for each observation)
    for i in range(nobs):
        phi = phis[i]
        alpha = sva[i]
        term1 = np.dot(data_tilde, u[:, i]) * w[:, i] / alpha
        # Loop over candiate mu values
        for j, mu in enumerate(mus):
            term2 = mu * xi0 / alpha**2
            factor = (phi ** 2) / (phi**2 + mu)
            xis[:, i, j] = factor * (term1 + term2)
    dem_guess = np.sum(xis, axis=1)
    intensity_guess = (K @ dem_guess).T
    terms = (intensity_guess - data[0, :]) / err
    norm = np.sum(terms**2, axis=1)
    tomin = norm - reg_tweak * data.size
    muidx = np.argmin(np.abs(tomin))
    if muidx == 0:
        logger.warn('Selected smallest value of mu')
    return mus[muidx], dem_guess[:, muidx]
