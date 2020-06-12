"""
Code for handling observations.
"""
import astropy.units as u
import numpy as np


class Observation:
    """
    A combination of an intensity measurement, the error on that intensity,
    measurement and the temperature response function for the measurement.

    Parameters
    ----------
    intensity : astropy.NDData
        Intensities. Must have uncertainties set.
    temp_response : TemperatureResponse
        Temperature response function.
    """
    # TODO: use astropy quantity input here
    def __init__(self, intensity, temp_response):
        self.intensity = intensity
        self.temp_response = temp_response


class TemperatureResponse:
    """
    A discretely sampled temperature response function.

    Parameters
    ----------
    temps : astropy.units.Quantity
        Temperatures.
    responses : astropy.units.Quantity
        Response function.
    """
    @u.quantity_input(temps=u.K, responses=u.cm**5 * u.s**-1 * u.pix**-1)
    def __init__(self, temps, responses):
        if temps.shape != responses.shape:
            raise ValueError('Shape of temps must be the same as '
                             'shape of responses')
        self.temps = temps
        self.responses = responses

    def __repr__(self):
        return f'TemperatureResponse, T={self.temps}, response={self.responses}'

    @u.quantity_input(temps=u.K)
    def sample(self, temps):
        """
        Sample the temperature repsonse function at temperatures *temps*.

        Uses linear interpolation.
        """
        resps = np.interp(temps, self.temps, self.responses,
                          left=np.nan, right=np.nan)
        return resps
