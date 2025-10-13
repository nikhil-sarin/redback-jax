# This is mostly from mosfit/photospheres but rewritten considerably.

from typing import Union
import jax.numpy as jnp
from jax import jit

from redback_jax.utils.citation_wrapper import citation_wrapper
from redback_jax.constants import *


@citation_wrapper("https://ui.adsabs.harvard.edu/abs/2017ApJ...851L..21V/abstract")
@jit
def compute_temperature_floor(
    time: jnp.ndarray,
    luminosity: jnp.ndarray,
    vej: jnp.ndarray,
    temperature_floor: Union[float, int],
) -> tuple:
    """
    Compute the photosphere temperature and radius with a floor temperature
    and effective blackbody otherwise.

    :param time: source frame time in days
    :type time: numpy.ndarray
    :param luminosity: luminosity
    :type luminosity: numpy.ndarray
    :param vej: ejecta velocity in km/s
    :type vej: numpy.ndarray
    :param temperature_floor: floor temperature in kelvin
    :type temperature_floor: Union[float, int]

    :return: photosphere temperature and radius
    """
    STEF_CONSTANT = 4 * jnp.pi * sigma_sb
    RADIUS_CONSTANT = km_cgs * day_to_s

    radius_squared = (RADIUS_CONSTANT * vej * time) ** 2
    rec_radius_squared = luminosity / (STEF_CONSTANT * temperature_floor ** 4)
    mask = radius_squared <= rec_radius_squared

    r_photosphere = jnp.where(
        mask,
        radius_squared ** 0.5,
        rec_radius_squared ** 0.5
    )
    photosphere_temperature = jnp.where(
        mask,
        (luminosity / (STEF_CONSTANT * radius_squared)) ** 0.25,
        temperature_floor,
    )

    return photosphere_temperature, r_photosphere
