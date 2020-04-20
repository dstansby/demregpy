import astropy.units as u
import numpy as np

from aia import get_prepped_aia_maps
from linalg import gsvd

maps = get_prepped_aia_maps()
temps = [1e6, 1.2e6] * u.K
ntemps = temps.size

# Get reponse functions at the given temps
maps

L = np.identity(ntemps)
alpha = 10
xi0 = np.zeros(ntemps)

K = ()

# Work out first inversion
gsvd(g, L)
print(xi0)
