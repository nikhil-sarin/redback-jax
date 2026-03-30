# Photosphere models for transient light-curve models.
# Works in log10 space for float32 safety throughout.

import math as _math

from typing import Union
import jax.numpy as jnp
from jax import jit

from redback_jax.utils.citation_wrapper import citation_wrapper

# Physical constants as Python floats (avoids astropy float64 promotion)
_SIGMA_SB = 5.6704e-5   # erg/s/cm^2/K^4
_KM_CGS   = 1.0e5       # cm/km
_DAY_TO_S = 86400.0     # s/day

_LOG10_4PI_SIGMASB = _math.log10(4.0 * _math.pi * _SIGMA_SB)
_LOG10_RADIUS_CONST = _math.log10(_KM_CGS * _DAY_TO_S)   # log10(cm/km * s/day)


@citation_wrapper("https://ui.adsabs.harvard.edu/abs/2017ApJ...851L..21V/abstract")
@jit
def compute_temperature_floor(
    time: jnp.ndarray,
    luminosity: jnp.ndarray,
    vej: jnp.ndarray,
    temperature_floor: Union[float, int],
) -> tuple:
    """
    Compute the photosphere temperature and radius with a floor temperature.
    NOTE: This function accepts LINEAR luminosity (erg/s) and returns linear values.
    For float32-safe spectra computations, use compute_temperature_floor_log10 instead.

    :param time: source frame time in days
    :param luminosity: luminosity in erg/s
    :param vej: ejecta velocity in km/s
    :param temperature_floor: floor temperature in kelvin
    :return: (photosphere_temperature, r_photosphere)
    """
    _stef    = 4.0 * _math.pi * _SIGMA_SB
    _r_const = _KM_CGS * _DAY_TO_S

    radius_squared     = (_r_const * vej * time) ** 2
    rec_radius_squared = luminosity / (_stef * temperature_floor ** 4)
    mask = radius_squared <= rec_radius_squared

    r_photosphere = jnp.where(mask, radius_squared ** 0.5, rec_radius_squared ** 0.5)
    photosphere_temperature = jnp.where(
        mask,
        (luminosity / (_stef * radius_squared)) ** 0.25,
        temperature_floor,
    )
    return photosphere_temperature, r_photosphere


@jit
def compute_temperature_floor_log10(
    time: jnp.ndarray,
    log10_luminosity: jnp.ndarray,
    vej: jnp.ndarray,
    temperature_floor: Union[float, int],
) -> tuple:
    """
    Float32-safe photosphere with temperature floor, working in log10 space.

    :param time: source frame time in days
    :param log10_luminosity: log10 of luminosity in erg/s
    :param vej: ejecta velocity in km/s
    :param temperature_floor: floor temperature in kelvin
    :return: (photosphere_temperature_K, log10_r_photosphere_cm)
             Temperature is linear (O(1e3-1e5) K, fits float32).
             Radius is log10 (avoids float32 overflow of ~1e15 cm).
    """
    fp = time.dtype
    _r_const = jnp.array(_KM_CGS * _DAY_TO_S, dtype=fp)

    # log10(r_ejecta) = log10(vej * km_cgs * day_s * time)
    log10_r_ej = (jnp.log10(jnp.maximum(vej, jnp.array(1e-10, dtype=fp)))
                  + jnp.array(_math.log10(_KM_CGS * _DAY_TO_S), dtype=fp)
                  + jnp.log10(jnp.maximum(time, jnp.array(1e-10, dtype=fp))))

    # log10(T_bb) = 0.25 * (log10_L - log10(4*pi*sigma_sb) - 2*log10_r_ej)
    log10_T_bb = 0.25 * (log10_luminosity
                         - jnp.array(_LOG10_4PI_SIGMASB, dtype=fp)
                         - 2.0 * log10_r_ej)

    T_floor_f     = jnp.asarray(temperature_floor, dtype=fp)
    log10_T_floor = jnp.log10(jnp.maximum(T_floor_f, jnp.array(1.0, dtype=fp)))

    # When T_bb >= T_floor: r = r_ej,  T = T_bb
    # When T_bb <  T_floor: r grows until T = T_floor
    log10_r_rec = 0.5 * (log10_luminosity
                         - jnp.array(_LOG10_4PI_SIGMASB, dtype=fp)
                         - 4.0 * log10_T_floor)

    below_floor = log10_T_bb < log10_T_floor

    log10_r_ph = jnp.where(below_floor, log10_r_rec, log10_r_ej)
    T_ph       = jnp.where(below_floor,
                            T_floor_f,
                            jnp.power(jnp.array(10.0, dtype=fp), log10_T_bb))

    return T_ph, log10_r_ph
