"""
JAX-friendly classes for supernova modeling.
"""

from jax import jit
import jax.numpy as jnp

from wcosmo import wcosmo

from redback_jax.utils.citation_wrapper import citation_wrapper
from redback_jax.constants import *
from redback_jax.conversions import calc_kcorrected_properties, lambda_to_nu
from redback_jax.interaction_processes import diffusion_convert_luminosity
from redback_jax.photosphere import compute_temperature_floor


def blackbody_to_flux_density(temperature, r_photosphere, dl, frequency):
    """
    A general blackbody_to_flux_density formula

    :param temperature: effective temperature in kelvin
    :param r_photosphere: photosphere radius in cm
    :param dl: luminosity_distance in cm
    :param frequency: frequency to calculate in Hz - Must be same length as time array or a single number.
                      In source frame
    :return: flux_density in erg/s/Hz/cm^2
    """
    radius = r_photosphere
    temperature = temperature
    planck = cc.h.cgs.value
    speed_of_light = cc.c.cgs.value
    boltzmann_constant = cc.k_B.cgs.value
    num = 2 * jnp.pi * planck * frequency ** 3 * radius ** 2
    denom = dl ** 2 * speed_of_light ** 2
    frac = 1. / (jnp.expm1((planck * frequency) / (boltzmann_constant * temperature)))
    flux_density = num / denom * frac
    return flux_density


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
    :param interaction_process: Name of the interaction process to apply (as a string).
        Default is diffusion. Output can be None to return raw luminosity.
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
