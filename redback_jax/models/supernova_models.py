"""
JAX-friendly classes for supernova modeling.
"""

import jax.numpy as jnp

from redback_jax.utils.citation_wrapper import citation_wrapper
from redback_jax.constants import *


@citation_wrapper("1994ApJS...92..527N")
def _nickelcobalt_engine(time, f_nickel, mej):
    """Compute the bolometric luminosity from nickel and cobalt decay.

    :param time: time in days
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param kwargs: None
    :return: bolometric_luminosity
    """
    ni56_lum = 6.45e43
    co56_lum = 1.45e43
    ni56_life = 8.8  # days
    co56_life = 111.3  # days
    nickel_mass = f_nickel * mej
    lbol = nickel_mass * (ni56_lum * jnp.exp(-time/ni56_life) + co56_lum * jnp.exp(-time/co56_life))
    return lbol
