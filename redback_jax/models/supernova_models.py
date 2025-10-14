"""
JAX-friendly classes for supernova modeling.
"""

import unxt as uu

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
    apply_diffusion=True,
    vej=None,
    kappa=None,
    kappa_gamma=None,
    dense_resolution=1000,
):
    """
    :param time: time in days
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param kappa: opacity (required if apply_diffusion is True)
    :param kappa_gamma: gamma-ray opacity (required if apply_diffusion is True)
    :param vej: ejecta velocity in km/s (required if apply_diffusion is True)
    :param dense_resolution: Number of points to use in the dense time array if 
        apply_diffusion is True
    :param apply_diffusion: True if diffusion interaction process should be applied.
    :return: bolometric_luminosity in erg/s
    """
    lbol = _nickelcobalt_engine(time=time, f_nickel=f_nickel, mej=mej)
    if apply_diffusion:
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

# We compile the arnett_bolometric function with JIT, specifying static arguments.
# Changing these will require recompilation.
arnett_bolometric_jit = jit(arnett_bolometric, static_argnames=('apply_diffusion', 'dense_resolution'))


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
def arnett_model(
    time,
    f_nickel,
    mej,
    *,
    redshift=0.0,
    cosmo_H0=None,
    cosmo_Om0=None,
    apply_diffusion=True,
    vej=None,
    kappa=None,
    kappa_gamma=None,
    temperature_floor=None,
    output_format="magnitude",
):
    """
    A version of the arnett model where SED has time-evolving spectral features.

    :param time: time in days
    :param redshift: source redshift
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param redshift: source redshift
    :param cosmo_H0: Hubble constant to use for luminosity distance calculation.
    :param cosmo_Om0: Matter density to use for luminosity distance calculation.
    :param apply_diffusion: True if diffusion interaction process should be applied.
    :param kappa: opacity (required if apply_diffusion is True)
    :param kappa_gamma: gamma-ray opacity (required if apply_diffusion is True)
    :param vej: ejecta velocity in km/s (required if apply_diffusion is True)
    :param temperature_floor: Floor temperature in kelvin (required if photosphere is temperature_floor)
    :param output_format: 'magnitude', 'spectra', 'flux'
    :param bands: Required if output_format is 'magnitude' or 'flux'.

    :return: set by output format - 'magnitude', 'spectra', 'flux'
    """
    dl = wcosmo.luminosity_distance(redshift, cosmo_H0, cosmo_Om0)
    
    time_obs = time
    lambda_observer_frame = jnp.geomspace(100, 60000, 100)
    time_temp = jnp.geomspace(0.1, 3000, 3000)  # in days
    time_observer_frame = time_temp * (1. + redshift)
    frequency, time = calc_kcorrected_properties(
        frequency=lambda_to_nu(lambda_observer_frame),
        redshift=redshift,
        time=time_observer_frame
    )

    lbol = arnett_bolometric_jit(
        time=time,
        f_nickel=f_nickel,
        mej=mej,
        apply_diffusion=apply_diffusion,
        vej=vej,
        kappa=kappa,
        kappa_gamma=kappa_gamma,
    )

    # Use a temperature floor photosphere model.
    photo_temp, r_photo = compute_temperature_floor(
        time=time,
        luminosity=lbol,
        vej=vej,
        temperature_floor=temperature_floor,
    )

    # Use the blackbody SED.
    fmjy = blackbody_to_flux_density(
        temperature=photo_temp,
        r_photosphere=r_photo,
        dl=dl,
        frequency=frequency[:, None],
    ).T

    #equivalencies = uu.spectral_density(frequency * uu.Hz, wav=lambda_observer_frame * uu.Angstrom)
    #spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom, equivalencies=equivalencies)
    #return spectra
