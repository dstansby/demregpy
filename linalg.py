import numpy as np
import astropy.units as u


def gsvd(A, B):
    # Generalised Singular Value Decomposition
    AB1 = A @ np.linalg.inv(B)
    u, s, vt = np.linalg.svd(AB1, full_matrices=False)

    beta = 1 / np.sqrt(1 + s**2)
    alpha = s * beta

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
    return mus[np.argmin(np.abs(tomin))]


@u.quantity_input(temps=u.K,
                  K=u.cm**5 / u.s / u.pix,
                  xi0=u.cm**-5,
                  L=u.cm**5)
def dem_guess(temps, K, xi0, L, alpha):
    """
    Produce a refined guess for the DEM.

    This follows eqs (6) and (7) from HK12.

    Parameters
    ----------
    temps : astropy.quantity.Quantity
        Temperatures
    K : astropy.quantity.Quantity
        Temperature response function.
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
    pass