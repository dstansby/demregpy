"""
Functions for fetching and prepping AIA maps.
"""
from collections import OrderedDict
import logging
from pathlib import Path

from aiapy.calibrate import (update_pointing, fix_observer_location,
                             correct_degradation, register)
from astropy import units as u
from datetime import timedelta
import numpy as np
from sunpy.net import Fido, attrs
from sunpy.map import Map
from sunpy.time import parse_time

from obs import TemperatureResponse, Observation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

wlen_ints = [94, 131, 171, 211, 193, 304, 335]
wlens = wlen_ints * u.angstrom

l1_map_dir = Path('/Users/dstansby/sunpy/data/dem')
l15_map_dir = Path('/Users/dstansby/sunpy/data/dem_prepped')


@u.quantity_input(wlen=u.angstrom)
def get_aia_map(wlen, dtime):
    """
    Get AIA map at wavelength *wlen* observed closest to *dtime*.
    """
    if wlen not in wlens:
        raise ValueError(f'Wavelength {wlen} not in list of '
                         f'allowed wavelengths {wlens}')
    stime = parse_time(dtime)
    etime = stime + timedelta(hours=1)
    q = Fido.search(
        attrs.Time(stime, etime, stime),
        attrs.Instrument('AIA'),
        attrs.Wavelength(wlen),
    )
    file = Fido.fetch(q)
    return Map(file)


def get_aia_maps(dtime):
    """
    Get the set of AIA maps observed closest to *dtime*.
    """
    return [get_aia_map(wlen, dtime) for wlen in wlens]


def prep(smap):
    """
    Prep an AIA map. This fixes the observer location and pointing,
    corrects for channel degredation, and registers the map to level 1.5.
    """
    logger.info(f'Prepping AIA {int(smap.wavelength.to_value(u.Angstrom))}\t'
                f'{smap.date}')
    smap = fix_observer_location(smap)
    smap = update_pointing(smap)
    smap = correct_degradation(smap)
    smap = register(smap)
    # Downsample from 64 to 32 bit
    data = smap.data.astype(np.float32)
    smap = Map(data, smap.meta)
    return smap


def get_response_function(wlen):
    """
    Get an AIA temperature response function.
    """
    import scipy.io
    dat = scipy.io.readsav('aia_resp15.dat')
    band = f'a{wlen}'
    Te = 10**(dat['tresp'][band][0].logte[0]) * u.K
    resp = dat['tresp'][band][0].tresp[0] * u.cm**5 * u.s**-1 * u.pix**-1
    return TemperatureResponse(Te, resp)


def get_prepped_aia_maps():
    maps = OrderedDict()
    for wlen in wlen_ints:
        f = list(l1_map_dir.glob(f'*{wlen}*.fits'))[0]
        prepped_f = l15_map_dir / f.name
        if not prepped_f.exists():
            m = Map(f)
            m = prep(m)
            m.save(prepped_f)
        else:
            m = Map(prepped_f)
        response = get_response_function(wlen)
        # Set the data errors to be zero for now
        maps[wlen] = Observation(m, response)
    return maps


if __name__ == '__main__':
    resps = get_aia_maps('2012-06-03 00:00:00')
