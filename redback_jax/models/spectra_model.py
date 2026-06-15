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

import jax.numpy as jnp

from redback_jax.conversions import calc_kcorrected_properties, lambda_to_nu
from redback_jax.models.sed_features import NO_SED_FEATURES, apply_sed_feature
from redback_jax.photosphere import compute_temperature_floor_log10
from jax_supernovae.utils import bandflux_integration

# Physical constants as Python floats
_H      = 6.626e-27   # erg s
_C      = 2.998e10    # cm/s
_KB     = 1.381e-16   # erg/K
_C_ANG  = 2.998e18    # Angstrom/s  (speed of light)

_LOG10_2PI_H  = _math.log10(2.0 * _math.pi * _H)
_LOG10_C2     = _math.log10(_C ** 2)
_LOG10_H_OVER_KB = _math.log10(_H / _KB)

_DEFAULT_LAMBDA_OBSERVER_FRAME = jnp.geomspace(100.0, 60000.0, 100)
_DEFAULT_TIME_SOURCE_FRAME = jnp.geomspace(0.1, 3000.0, 800)


def _build_spectra_grids(redshift, time_observer_frame_grid=None, lambda_observer_frame_grid=None):
    """Return observer-frame grids plus source-frame time/frequency equivalents."""
    lambda_observer_frame = (
        jnp.asarray(lambda_observer_frame_grid)
        if lambda_observer_frame_grid is not None
        else _DEFAULT_LAMBDA_OBSERVER_FRAME
    )
    time_observer_frame = (
        jnp.asarray(time_observer_frame_grid)
        if time_observer_frame_grid is not None
        else _DEFAULT_TIME_SOURCE_FRAME * (1.0 + redshift)
    )
    frequency, time = calc_kcorrected_properties(
        frequency=lambda_to_nu(lambda_observer_frame),
        redshift=redshift,
        time=time_observer_frame,
    )
    return lambda_observer_frame, time_observer_frame, frequency, time


def _blackbody_flux_density_frequency(time, frequency, log10_lbol, vej, temperature_floor, lum_dist):
    """Return observer-frame F_nu on a source-frame time / source-frame frequency grid."""
    T_ph, log10_r_ph = compute_temperature_floor_log10(
        time=time,
        log10_luminosity=log10_lbol,
        vej=vej,
        temperature_floor=temperature_floor,
    )

    fp = time.dtype
    nu = frequency.astype(fp)
    dl = jnp.asarray(lum_dist, dtype=fp)
    log10_dl = jnp.log10(jnp.maximum(dl, jnp.array(1.0, dtype=fp)))
    x = (_H / _KB) * nu[None, :] / jnp.maximum(T_ph[:, None], jnp.array(1.0, dtype=fp))
    x = jnp.clip(x, jnp.array(1e-10, dtype=fp), jnp.array(80.0, dtype=fp))

    log10_Fnu = (jnp.array(_LOG10_2PI_H, dtype=fp)
                 + 3.0 * jnp.log10(nu[None, :])
                 + 2.0 * log10_r_ph[:, None]
                 - 2.0 * log10_dl
                 - jnp.array(_LOG10_C2, dtype=fp)
                 - jnp.log10(jnp.expm1(x)))
    return jnp.power(jnp.array(10.0, dtype=fp), log10_Fnu)


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
                      features=NO_SED_FEATURES,
                      _time_observer_frame_grid=None,
                      _lambda_observer_frame_grid=None,
                      **bolometric_kwargs):
        return _spectra_model_impl(
            bolometric_fn,
            redshift, lum_dist, vej, temperature_floor,
            features, bolometric_kwargs, _bolo_accepts_vej,
            time_observer_frame_grid=_time_observer_frame_grid,
            lambda_observer_frame_grid=_lambda_observer_frame_grid,
        )

    def direct_photometry_model(
        *,
        obs_source_time,
        obs_band_idx,
        bridges,
        redshift,
        lum_dist,
        vej,
        temperature_floor,
        features=NO_SED_FEATURES,
        **bolometric_kwargs,
    ):
        return _direct_photometry_impl(
            bolometric_fn,
            obs_source_time=obs_source_time,
            obs_band_idx=obs_band_idx,
            bridges=bridges,
            redshift=redshift,
            lum_dist=lum_dist,
            vej=vej,
            temperature_floor=temperature_floor,
            features=features,
            bolometric_kwargs=bolometric_kwargs,
            bolo_accepts_vej=_bolo_accepts_vej,
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
    spectra_model._redback_jax_bolometric_fn = bolometric_fn
    spectra_model._redback_jax_supports_custom_grids = True
    spectra_model._redback_jax_direct_photometry = direct_photometry_model
    return spectra_model


def _spectra_model_impl(bolometric_fn, redshift, lum_dist, vej,
                         temperature_floor, features, bolometric_kwargs,
                         bolo_accepts_vej=False,
                         time_observer_frame_grid=None,
                         lambda_observer_frame_grid=None):
    """Inner implementation — log10-space SED pipeline for float32 safety."""
    lambda_observer_frame, time_observer_frame, frequency, time = _build_spectra_grids(
        redshift,
        time_observer_frame_grid=time_observer_frame_grid,
        lambda_observer_frame_grid=lambda_observer_frame_grid,
    )

    # Bolometric luminosity in log10 erg/s (returned directly by all bolometric fns)
    # If the bolometric function also accepts vej (e.g. arnett_bolometric uses it
    # for diffusion), forward it — unless the caller already supplied it explicitly.
    if bolo_accepts_vej:
        bolometric_kwargs = {'vej': vej, **bolometric_kwargs}
    log10_lbol = bolometric_fn(time, **bolometric_kwargs)

    spectral_flux_density = _blackbody_flux_density_frequency(
        time=time,
        frequency=frequency,
        log10_lbol=log10_lbol,
        vej=vej,
        temperature_floor=temperature_floor,
        lum_dist=lum_dist,
    )

    spectral_flux_density = apply_sed_feature(
        features, spectral_flux_density, frequency, time)

    # Convert erg/s/Hz/cm^2 → erg/s/cm^2/Angstrom, then correct for bandwidth stretching
    fp = time.dtype
    lam = lambda_observer_frame.astype(fp)
    spectra = spectral_flux_density * jnp.array(_C_ANG, dtype=fp) / (lam[None, :] ** 2)
    spectra = spectra * jnp.asarray(1.0 + redshift, dtype=fp)

    return namedtuple('output', ['time', 'lambdas', 'spectra'])(
        time=time_observer_frame,
        lambdas=lambda_observer_frame,
        spectra=spectra,
    )


def _direct_photometry_impl(bolometric_fn, *, obs_source_time, obs_band_idx, bridges,
                            redshift, lum_dist, vej, temperature_floor, features,
                            bolometric_kwargs, bolo_accepts_vej=False):
    """Inference-only fast path: integrate blackbody flux directly through bandpasses."""
    if bolo_accepts_vej:
        bolometric_kwargs = {'vej': vej, **bolometric_kwargs}

    log10_lbol = bolometric_fn(obs_source_time, **bolometric_kwargs)
    fp = obs_source_time.dtype
    band_indices = obs_band_idx.astype(jnp.int32)
    redshift_f = jnp.asarray(1.0 + redshift, dtype=fp)

    T_ph, log10_r_ph = compute_temperature_floor_log10(
        time=obs_source_time,
        log10_luminosity=log10_lbol,
        vej=vej,
        temperature_floor=temperature_floor,
    )

    def _one_band(bridge):
        lam = jnp.asarray(bridge['wave'], dtype=fp)
        trans = jnp.asarray(bridge['trans'], dtype=fp)
        dwave = jnp.asarray(bridge['dwave'], dtype=fp)
        source_frequency = lambda_to_nu(lam) * redshift_f

        x = (_H / _KB) * source_frequency[None, :] / jnp.maximum(
            T_ph[:, None], jnp.array(1.0, dtype=fp)
        )
        x = jnp.clip(x, jnp.array(1e-10, dtype=fp), jnp.array(80.0, dtype=fp))
        dl = jnp.asarray(lum_dist, dtype=fp)
        log10_dl = jnp.log10(jnp.maximum(dl, jnp.array(1.0, dtype=fp)))
        log10_Fnu = (jnp.array(_LOG10_2PI_H, dtype=fp)
                     + 3.0 * jnp.log10(source_frequency[None, :])
                     + 2.0 * log10_r_ph[:, None]
                     - 2.0 * log10_dl
                     - jnp.array(_LOG10_C2, dtype=fp)
                     - jnp.log10(jnp.expm1(x)))
        spectral_flux_density = jnp.power(jnp.array(10.0, dtype=fp), log10_Fnu)
        spectral_flux_density = apply_sed_feature(
            features, spectral_flux_density, source_frequency, obs_source_time)
        spectra = spectral_flux_density * jnp.array(_C_ANG, dtype=fp) / (lam[None, :] ** 2)
        spectra = spectra * redshift_f
        bandflux = bandflux_integration(lam, trans, spectra, dwave)
        zpbandflux = jnp.asarray(bridge['zpbandflux_ab'], dtype=fp)
        return bandflux / zpbandflux

    flux_by_band = jnp.stack([_one_band(bridge) for bridge in bridges], axis=1)
    return flux_by_band[jnp.arange(len(obs_source_time)), band_indices]
