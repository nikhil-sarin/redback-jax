"""
Generic factory for converting any bolometric model into a spectra model.

Usage::

    from redback_jax.models.spectra_model import make_spectra_model
    from redback_jax.models import magnetar_powered_bolometric

    magnetar_powered_spectra = make_spectra_model(magnetar_powered_bolometric)

    out = magnetar_powered_spectra(
        redshift=0.1,
        lum_dist=dl_cm,
        vej=10000.0,
        temperature_floor=3000.0,
        # remaining kwargs forwarded verbatim to the bolometric function:
        p0=2.0, bp=1.0, mass_ns=1.4, theta_pb=0.3,
        mej=1.0, kappa=0.1, kappa_gamma=10.0,
    )

All operations stay in log10 / temperature space for float32 safety.
The bolometric function is called and its output (linear erg/s) is converted
to log10 immediately. If the bolometric function overflows float32 the returned
spectra will contain NaN/Inf, but for typical parameter ranges float32 is safe.
"""

import math as _math
from collections import namedtuple

import jax
import jax.numpy as jnp
from jax import jit

from redback_jax.conversions import calc_kcorrected_properties, lambda_to_nu
from redback_jax.models.sed_features import NO_SED_FEATURES, apply_sed_feature
from redback_jax.photosphere import compute_temperature_floor_log10

# Physical constants as Python floats
_H      = 6.626e-27   # erg s
_C      = 2.998e10    # cm/s
_KB     = 1.381e-16   # erg/K
_C_ANG  = 2.998e18    # Angstrom/s  (speed of light)

_LOG10_2PI_H  = _math.log10(2.0 * _math.pi * _H)
_LOG10_C2     = _math.log10(_C ** 2)
_LOG10_H_OVER_KB = _math.log10(_H / _KB)


def make_spectra_model(bolometric_fn):
    """
    Wrap a bolometric model function to produce a full spectra model.

    The returned function has signature::

        spectra_model(redshift, lum_dist, vej, temperature_floor,
                      features=NO_SED_FEATURES, **bolometric_kwargs)
        -> namedtuple(time, lambdas, spectra)

    Parameters
    ----------
    bolometric_fn : callable
        Any function ``f(time_days, **kwargs) -> log10_lbol`` (log10 erg/s).
        ``time_days`` must be its first positional argument.

    Returns
    -------
    callable
        A spectra model with the same photosphere/SED pipeline.
    """
    import inspect as _inspect
    _bolo_accepts_vej = 'vej' in _inspect.signature(bolometric_fn).parameters

    def spectra_model(redshift, lum_dist, vej, temperature_floor,
                      features=NO_SED_FEATURES, **bolometric_kwargs):
        return _spectra_model_impl(
            bolometric_fn,
            redshift, lum_dist, vej, temperature_floor,
            features, bolometric_kwargs, _bolo_accepts_vej,
        )

    spectra_model.__doc__ = (
        f"Spectra model wrapping ``{bolometric_fn.__name__}``.\\n\\n"
        "Args:\\n"
        "    redshift: source redshift\\n"
        "    lum_dist: luminosity distance in cm\\n"
        "    vej: ejecta velocity in km/s (photosphere)\\n"
        "    temperature_floor: floor temperature in K\\n"
        "    features: SEDFeatures (default NO_SED_FEATURES)\\n"
        "    **bolometric_kwargs: forwarded to the bolometric function\\n\\n"
        "Returns:\\n"
        "    namedtuple with fields ``time`` (days), ``lambdas`` (Angstrom), "
        "``spectra`` (erg/s/cm^2/Angstrom)\\n"
    )
    spectra_model.__name__ = bolometric_fn.__name__ + "_spectra"
    return spectra_model


def _spectra_model_impl(bolometric_fn, redshift, lum_dist, vej,
                         temperature_floor, features, bolometric_kwargs,
                         bolo_accepts_vej=False):
    """Inner implementation — log10-space SED pipeline for float32 safety."""
    lambda_observer_frame = jnp.geomspace(100.0, 60000.0, 100)
    time_temp = jnp.geomspace(0.1, 3000.0, 3000)          # days
    time_observer_frame = time_temp * (1.0 + redshift)

    frequency, time = calc_kcorrected_properties(
        frequency=lambda_to_nu(lambda_observer_frame),
        redshift=redshift,
        time=time_observer_frame,
    )

    # Bolometric luminosity in log10 erg/s (returned directly by all bolometric fns)
    # If the bolometric function also accepts vej (e.g. arnett_bolometric uses it
    # for diffusion), forward it — unless the caller already supplied it explicitly.
    if bolo_accepts_vej:
        bolometric_kwargs = {'vej': vej, **bolometric_kwargs}
    log10_lbol = bolometric_fn(time, **bolometric_kwargs)

    # Temperature and log10(radius) — both float32-safe
    T_ph, log10_r_ph = compute_temperature_floor_log10(
        time=time,
        log10_luminosity=log10_lbol,
        vej=vej,
        temperature_floor=temperature_floor,
    )

    # Blackbody flux density in log10 space:
    # F_nu = 2*pi*h*nu^3 * R^2 / (dl^2 * c^2) / expm1(h*nu / kB*T)
    # log10(F_nu) = log10(2pi*h) + 3*log10(nu) + 2*log10_r_ph
    #              - 2*log10(dl) - 2*log10(c) - log10(expm1(h*nu/(kB*T)))
    fp     = time.dtype
    nu     = frequency.astype(fp)                  # (Nfreq,)
    dl     = jnp.asarray(lum_dist, dtype=fp)
    log10_dl = jnp.log10(jnp.maximum(dl, jnp.array(1.0, dtype=fp)))

    # x = h*nu / (kB * T)  — O(1) quantity, safe in float32
    # T_ph: (Ntime,),  nu: (Nfreq,)
    x = (_H / _KB) * nu[None, :] / jnp.maximum(T_ph[:, None], jnp.array(1.0, dtype=fp))
    x = jnp.clip(x, jnp.array(1e-10, dtype=fp), jnp.array(80.0, dtype=fp))

    log10_Fnu = (jnp.array(_LOG10_2PI_H, dtype=fp)
                 + 3.0 * jnp.log10(nu[None, :])
                 + 2.0 * log10_r_ph[:, None]
                 - 2.0 * log10_dl
                 - jnp.array(_LOG10_C2, dtype=fp)
                 - jnp.log10(jnp.expm1(x)))    # (Ntime, Nfreq)

    spectral_flux_density = jnp.power(jnp.array(10.0, dtype=fp), log10_Fnu)  # erg/s/Hz/cm^2

    spectral_flux_density = apply_sed_feature(
        features, spectral_flux_density, frequency, time)

    # Convert erg/s/Hz/cm^2 → erg/s/cm^2/Angstrom, then correct for bandwidth stretching
    lam = lambda_observer_frame.astype(fp)
    spectra = spectral_flux_density * jnp.array(_C_ANG, dtype=fp) / (lam[None, :] ** 2)
    spectra = spectra * jnp.asarray(1.0 + redshift, dtype=fp)

    return namedtuple('output', ['time', 'lambdas', 'spectra'])(
        time=time_observer_frame,
        lambdas=lambda_observer_frame,
        spectra=spectra,
    )
