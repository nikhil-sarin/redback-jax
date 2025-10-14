"""
JAX-friendly classes for supernova modeling.
"""

from jax import jit
import jax.numpy as jnp

from redback_jax.utils.citation_wrapper import citation_wrapper
from redback_jax.constants import *
from redback_jax.interaction_processes import diffusion_convert_luminosity


def calc_kcorrected_properties(frequency, redshift, time):
    """
    Perform k-correction

    :param frequency: observer frame frequency
    :param redshift: source redshift
    :param time: observer frame time
    :return: k-corrected frequency and source frame time
    """
    time = time / (1 + redshift)
    frequency = frequency * (1 + redshift)
    return frequency, time


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
    *,
    interaction_process="diffusion",
    vej=None,
    kappa=None,
    kappa_gamma=None,
    dense_resolution=1000,
    **kwargs,
):
    """
    :param time: time in days
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param kappa: opacity (required if interaction_process is not None)
    :param kappa_gamma: gamma-ray opacity (required if interaction_process is not None)
    :param vej: ejecta velocity in km/s (required if interaction_process is not None)
    :param dense_resolution: Number of points to use in the dense time array if 
        interaction_process is not None
    :param interaction_process: Function pointer to the interaction process to apply.
        Default is diffusion_convert_luminosity. Output can be None to return raw luminosity.

    :param kwargs: Must be all the kwargs required by the specific interaction_process.
    :return: bolometric_luminosity in erg/s
    """
    lbol = _nickelcobalt_engine(time=time, f_nickel=f_nickel, mej=mej)
    if interaction_process == "diffusion":
        dense_times = jnp.linspace(0, time[-1]+100, dense_resolution)
        dense_lbols = _nickelcobalt_engine(time=dense_times, f_nickel=f_nickel, mej=mej)
        _, new_luminosity = diffusion_convert_luminosity(
            time=time,
            dense_times=dense_times,
            luminosity=dense_lbols,
            mej=mej,
            kappa=kappa,
            kappa_gamma=kappa_gamma,
            vej=vej,
        )
        lbol = new_luminosity
    return lbol
