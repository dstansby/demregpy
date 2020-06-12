import logging

import numpy as np
from numpy.linalg import inv
import astropy.units as u

logger = logging.getLogger(__name__)


def gsvd(A, B):
    # Generalised Singular Value Decomposition
    AB1 = A @ inv(B)
    # u is square and unitary
    u, s, vt = np.linalg.svd(AB1, full_matrices=True)

    beta = 1 / np.sqrt(1 + s**2)
    alpha = s * beta
    if np.any(np.isclose(alpha, 1, rtol=0, atol=1e-8)):
        logger.warn('Some singular value values very close to 1')
    if np.any(np.isclose(beta, 1, rtol=0, atol=1e-8)):
        logger.warn('Some singular value values very close to 1')

    wT = inv(np.diag(alpha)) @ u.T @ A
    w = np.linalg.pinv(wT.value) / wT.unit
    return alpha, beta, u, vt.T, w
