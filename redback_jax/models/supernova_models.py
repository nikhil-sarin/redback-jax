"""
JAX-friendly classes for supernova modeling.
"""

from jax import jit
import jax.numpy as jnp

from redback_jax.utils.citation_wrapper import citation_wrapper
from redback_jax.constants import *
from redback_jax.interaction_processes import diffusion_convert_luminosity


@citation_wrapper("1994ApJS...92..527N")
@jit
def _nickelcobalt_engine(time, f_nickel, mej):
    """Compute the bolometric luminosity from nickel and cobalt decay.

    :param time: time in days
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param kwargs: None
    :return: bolometric_luminosity (in erg/s)
    """
    ni56_lum = 6.45e43
    co56_lum = 1.45e43
    ni56_life = 8.8  # days
    co56_life = 111.3  # days
    nickel_mass = f_nickel * mej
    lbol = nickel_mass * (ni56_lum * jnp.exp(-time/ni56_life) + co56_lum * jnp.exp(-time/co56_life))
    return lbol


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
def arnett_bolometric(
    time,
    f_nickel,
    mej,
    interaction_process=diffusion_convert_luminosity,
    dense_resolution=1000,
    **kwargs,
):
    """
    :param time: time in days
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param kwargs: Must be all the kwargs required by the specific interaction_process
    :param interaction_process: Function pointer to the interaction process to apply.
        Default is diffusion_convert_luminosity. Output can be None to return raw luminosity.
    :param dense_resolution: Number of points to use in the dense time array if 
        interaction_process is not None
    :param kwargs: Must be all the kwargs required by the specific interaction_process.
    :return: bolometric_luminosity in erg/s
    """
    lbol = _nickelcobalt_engine(time=time, f_nickel=f_nickel, mej=mej)
    if interaction_process is not None:
        dense_times = jnp.linspace(0, time[-1]+100, dense_resolution)
        dense_lbols = _nickelcobalt_engine(time=dense_times, f_nickel=f_nickel, mej=mej)
        _, new_luminosity = interaction_process(
            time=time,
            dense_times=dense_times,
            luminosity=dense_lbols,
            mej=mej,
            **kwargs,
        )
        lbol = new_luminosity
    return lbol
