"""
JAX-friendly classes for supernova modeling.
"""

from jax import jit
import jax.numpy as jnp
from wcosmo import wcosmo

from collections import namedtuple


from redback_jax.utils.citation_wrapper import citation_wrapper
from redback_jax.constants import *
from redback_jax.conversions import calc_kcorrected_properties, lambda_to_nu
from redback_jax.interaction_processes import diffusion_convert_luminosity
from redback_jax.models.sed_features import NO_SED_FEATURES, apply_sed_feature
from redback_jax.photosphere import compute_temperature_floor


# Planck18 cosmology defaults
PLANCK18_H0 = 67.66  # km/s/Mpc
PLANCK18_OM0 = 0.3111


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


@jit
def _compute_mass_and_nickel(
        vmin,
        esn,
        mej,
        f_nickel,
        f_mixing,
        vmax,
        delta=0.0,
        n=12.0,
    ):
    """
    Compute the mass and nickel distributions following a broken power-law
    density profile inspired by Matzner & McKee (1999)

    :param vmin: minimum velocity in km/s
    :param esn: supernova explosion energy in foe
    :param mej: total ejecta mass in solar masses
    :param f_nickel: fraction of nickel mass
    :param f_mixing: fraction of nickel mass that is mixed
    :param vmax: maximum velocity in km/s
    :param delta: inner density profile exponent (actual mass dist is 2 - delta)
    :param n: outer density profile exponent (actual mass dist is 2 - n)
    :return: vel in km/s, v_m in cm/s, m_array in solar masses, ni_array in solar masses (total nickel mass is f_nickel*mej)
    """
    # Create velocity grid in km/s and convert to cm/s. The grid must be a fixed
    # size for JAX, so we use 200 points (previously mass_len parameter).
    vel = jnp.geomspace(vmin, vmax, 200) # km/s
    v_m = vel * km_cgs # cgs

    # Define a break velocity; use shock speed from Matzner & McKee (1999).
    num = 2 * (5 - delta) * (n - 5) * esn * 1e51
    denom = (3 - delta)*(n - 3) * mej * solar_mass
    v_break = jnp.sqrt(num/denom) / km_cgs

    # For a uniform grid, determine the velocity spacing.
    dv = vel[1] - vel[0]

    # Compute the unnormalized mass distribution using vectorized operations.
    # For the inner part: (v/v_break)^(2 - delta)
    # For the outer part: (v/v_break)^(2 - n)
    m_array = jnp.where(
        vel <= v_break,
        (vel / v_break)**(2.0 - delta),
        (vel / v_break)**(2.0 - n),
    )
    # Multiply by the bin width.
    m_array = m_array * dv
    # Normalize the mass array so that the summed mass equals mej.
    total_mass = jnp.sum(m_array)
    m_array = mej * m_array / total_mass

    # --- Compute the nickel distribution ---
    # Total nickel mass.
    ni_mass = f_nickel * mej
    # Only the inner fraction of the shells receives nickel.
    limiting_index = jnp.floor(200 * f_mixing).astype(int)
    limiting_index = jnp.maximum(limiting_index, 1)
    limiting_mask = jnp.arange(200) < limiting_index

    # Using the same inner profile for the nickel weight,
    # zeroing out the outer shells.
    ni_array = jnp.where(
        vel <= v_break,
        (vel / v_break)**(2.0 - delta),
        (vel / v_break)**(2.0 - n),
    )
    ni_array = jnp.where(limiting_mask, ni_array, 0.0)
    ni_array = ni_mass * ni_array / jnp.sum(ni_array)

    return vel, v_m, m_array, ni_array


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
@jit
def arnett_bolometric(
    time,
    f_nickel,
    mej,
    *,
    vej=None,
    kappa=None,
    kappa_gamma=None,
):
    """
    Compute the bolometric luminosity using the Arnett model.

    :param time: time in days
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param kappa: opacity (required)
    :param kappa_gamma: gamma-ray opacity (required)
    :param vej: ejecta velocity in km/s (required)
    :return: bolometric_luminosity in erg/s
    """
    lbol = _nickelcobalt_engine(time=time, f_nickel=f_nickel, mej=mej)

    # Perform the diffusion interaction process.
    dense_times = jnp.linspace(0, time[-1]+100, 1000)
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


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
@jit
def arnett_with_features_lum_dist(
    f_nickel,
    mej,
    *,
    redshift=0.0,
    lum_dist=None,
    vej=None,
    kappa=None,
    kappa_gamma=None,
    temperature_floor=None,
    features=NO_SED_FEATURES,
):
    """
    A version of the arnett model where SED has time-evolving spectral features.

    :param redshift: source redshift
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param redshift: source redshift
    :param lum_dist: luminosity distance in cm
    :param kappa: opacity (required)
    :param kappa_gamma: gamma-ray opacity (required)
    :param vej: ejecta velocity in km/s (required)
    :param temperature_floor: Floor temperature in kelvin (required if photosphere is temperature_floor)
    :param features: SEDFeatures object with the spectral features to add.

    :return: A named tuple of three arrays: time (in days), lambdas (in Angstrom), and spectra
    """
    lambda_observer_frame = jnp.geomspace(100, 60000, 100)
    time_temp = jnp.geomspace(0.1, 3000, 3000)  # in days
    time_observer_frame = time_temp * (1. + redshift)
    frequency, time = calc_kcorrected_properties(
        frequency=lambda_to_nu(lambda_observer_frame),
        redshift=redshift,
        time=time_observer_frame
    )

    lbol = arnett_bolometric(
        time=time,
        f_nickel=f_nickel,
        mej=mej,
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

    # Use the blackbody spectral flux density. Result is in erg/s/Hz/cm^2.
    spectral_flux_density = blackbody_to_flux_density(
        temperature=photo_temp,
        r_photosphere=r_photo,
        dl=lum_dist,
        frequency=frequency[:, None],
    ).T

    # Apply any spectral features (this is a no-op if not features are given).
    spectral_flux_density = apply_sed_feature(features, spectral_flux_density, frequency, time)

    # Convert erg/s/Hz/cm^2 to erg/cm^2/s/Angstrom using 2.998e18 for the
    # speed of light in Angstrom/s. We define this numerically to be JAX-friendly.
    spectra = spectral_flux_density * 2.998e18 / (lambda_observer_frame[None, :] ** 2)
    return namedtuple('output', ['time', 'lambdas', 'spectra'])(
        time=time_observer_frame,
        lambdas=lambda_observer_frame,
        spectra=spectra
    )


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
@jit
def arnett_with_features_cosmology(
    f_nickel,
    mej,
    *,
    redshift=0.0,
    cosmo_H0=PLANCK18_H0,
    cosmo_Om0=PLANCK18_OM0,
    vej=None,
    kappa=None,
    kappa_gamma=None,
    temperature_floor=None,
    features=NO_SED_FEATURES,
):
    """
    A version of the arnett model where SED has time-evolving spectral features.

    :param redshift: source redshift
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param redshift: source redshift
    :param cosmo_H0: Hubble constant to use for luminosity distance calculation.
    :param cosmo_Om0: Matter density to use for luminosity distance calculation.
    :param kappa: opacity (required)
    :param kappa_gamma: gamma-ray opacity (required)
    :param vej: ejecta velocity in km/s (required)
    :param temperature_floor: Floor temperature in kelvin (required if photosphere is temperature_floor)
    :param features: SEDFeatures object with the spectral features to add.

    :return: A named tuple of three arrays: time (in days), lambdas (in Angstrom), and spectra
    """
    # Wcosmo returns in Mpc (though it is marked as km/s), so we need to
    # correct the units and convert to cm.
    dl = wcosmo.luminosity_distance(redshift, cosmo_H0, cosmo_Om0).value * Mpc_to_cm
    return arnett_with_features_lum_dist(
        f_nickel=f_nickel,
        mej=mej,
        redshift=redshift,
        lum_dist=dl,
        vej=vej,
        kappa=kappa,
        kappa_gamma=kappa_gamma,
        temperature_floor=temperature_floor,
        features=features,
    )
