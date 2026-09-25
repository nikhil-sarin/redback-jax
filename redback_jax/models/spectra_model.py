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
_DEFAULT_TIME_SOURCE_FRAME = jnp.geomspace(0.1, 250.0, 150)
_DEFAULT_GRID_BANDPASS_POINTS = 64


def _build_spectra_grids(redshift, time_observer_frame_grid=None, lambda_observer_frame_grid=None):
    """Return observer-frame grids plus source-frame time/frequency equivalents."""
    fp = jnp.asarray(redshift).dtype
    lambda_observer_frame = (
        jnp.asarray(lambda_observer_frame_grid, dtype=fp)
        if lambda_observer_frame_grid is not None
        else _DEFAULT_LAMBDA_OBSERVER_FRAME.astype(fp)
    )
    time_observer_frame = (
        jnp.asarray(time_observer_frame_grid, dtype=fp)
        if time_observer_frame_grid is not None
        else _DEFAULT_TIME_SOURCE_FRAME.astype(fp) * (jnp.asarray(1.0, dtype=fp) + jnp.asarray(redshift, dtype=fp))
    )
    frequency, time = calc_kcorrected_properties(
        frequency=lambda_to_nu(lambda_observer_frame),
        redshift=redshift,
        time=time_observer_frame,
    )
    return lambda_observer_frame, time_observer_frame, frequency, time


def _stack_bandpass_bridges(bridges, fp, max_points=None):
    """Return padded bandpass arrays for per-observation quadrature."""
    def _resample_bridge(bridge):
        wave = jnp.asarray(bridge['wave'], dtype=fp)
        trans = jnp.asarray(bridge['trans'], dtype=fp)
        if max_points is None or len(wave) <= max_points:
            dwave = jnp.asarray(bridge['dwave'], dtype=fp)
            return wave, trans, dwave
        new_wave = jnp.linspace(wave[0], wave[-1], int(max_points), dtype=fp)
        new_trans = jnp.interp(new_wave, wave, trans)
        new_dwave = (new_wave[-1] - new_wave[0]) / jnp.asarray(
            max(int(max_points) - 1, 1), dtype=fp
        )
        return new_wave, new_trans, new_dwave

    resampled = [_resample_bridge(bridge) for bridge in bridges]
    max_wlen = max(len(wave) for wave, _, _ in resampled)

    def _pad_edge(a):
        a = jnp.asarray(a, dtype=fp)
        return jnp.pad(a, (0, max_wlen - len(a)), mode='edge') if len(a) < max_wlen else a

    def _pad_zero(a):
        a = jnp.asarray(a, dtype=fp)
        return jnp.pad(a, (0, max_wlen - len(a))) if len(a) < max_wlen else a

    waves = jnp.stack([_pad_edge(wave) for wave, _, _ in resampled])
    trans = jnp.stack([_pad_zero(trans_i) for _, trans_i, _ in resampled])
    dwave = jnp.stack([dwave_i for _, _, dwave_i in resampled])
    zpbf = jnp.asarray([float(bridge['zpbandflux_ab']) for bridge in bridges], dtype=fp)
    return waves, trans, dwave, zpbf


def _blackbody_bandflux_from_state(
    obs_source_time,
    obs_band_idx,
    bridges,
    redshift,
    lum_dist,
    temperature,
    log10_r_ph,
    features=NO_SED_FEATURES,
    max_bandpass_points=None,
):
    """Integrate a blackbody photosphere through observed bandpasses."""
    fp = obs_source_time.dtype
    waves, trans, dwaves, zpbf = _stack_bandpass_bridges(
        bridges, fp, max_points=max_bandpass_points
    )
    band_indices = obs_band_idx.astype(jnp.int32)
    redshift_f = jnp.asarray(1.0 + redshift, dtype=fp)
    dl = jnp.asarray(lum_dist, dtype=fp)
    log10_dl = jnp.log10(jnp.maximum(dl, jnp.array(1.0, dtype=fp)))

    def _one_obs(t_i, band_idx, temp_i, log10_r_i):
        lam = waves[band_idx]
        response = trans[band_idx]
        source_frequency = lambda_to_nu(lam) * redshift_f
        x = (_H / _KB) * source_frequency / jnp.maximum(temp_i, jnp.array(1.0, dtype=fp))
        x = jnp.clip(x, jnp.array(1e-10, dtype=fp), jnp.array(80.0, dtype=fp))
        log10_Fnu = (jnp.array(_LOG10_2PI_H, dtype=fp)
                     + 3.0 * jnp.log10(source_frequency)
                     + 2.0 * log10_r_i
                     - 2.0 * log10_dl
                     - jnp.array(_LOG10_C2, dtype=fp)
                     - jnp.log10(jnp.expm1(x)))
        spectral_flux_density = jnp.power(jnp.array(10.0, dtype=fp), log10_Fnu)
        spectral_flux_density = apply_sed_feature(
            features, spectral_flux_density[None, :], source_frequency, t_i[None]
        )[0]
        spectra = spectral_flux_density * jnp.array(_C_ANG, dtype=fp) / (lam ** 2)
        spectra = spectra * redshift_f
        bandflux = jnp.sum(lam * response * spectra) * dwaves[band_idx] / jnp.asarray(
            6.626e-27 * 2.998e18, dtype=fp
        )
        return bandflux / zpbf[band_idx]

    return jax.vmap(_one_obs)(obs_source_time, band_indices, temperature, log10_r_ph)


def _blackbody_bandflux_grid_from_state(
    source_time_grid,
    bridges,
    redshift,
    lum_dist,
    temperature_grid,
    log10_r_grid,
    features=NO_SED_FEATURES,
    max_bandpass_points=None,
):
    """Return AB-normalised band fluxes on ``(n_time, n_band)`` grid."""
    fp = source_time_grid.dtype
    waves, trans, dwaves, zpbf = _stack_bandpass_bridges(
        bridges, fp, max_points=max_bandpass_points
    )
    redshift_f = jnp.asarray(1.0 + redshift, dtype=fp)
    dl = jnp.asarray(lum_dist, dtype=fp)
    log10_dl = jnp.log10(jnp.maximum(dl, jnp.array(1.0, dtype=fp)))
    hc = jnp.asarray(6.626e-27 * 2.998e18, dtype=fp)

    def _one_band(lam, response, dwave, zpbandflux):
        source_frequency = lambda_to_nu(lam) * redshift_f
        x = (_H / _KB) * source_frequency[None, :] / jnp.maximum(
            temperature_grid[:, None], jnp.array(1.0, dtype=fp)
        )
        x = jnp.clip(x, jnp.array(1e-10, dtype=fp), jnp.array(80.0, dtype=fp))
        log10_Fnu = (jnp.array(_LOG10_2PI_H, dtype=fp)
                     + 3.0 * jnp.log10(source_frequency[None, :])
                     + 2.0 * log10_r_grid[:, None]
                     - 2.0 * log10_dl
                     - jnp.array(_LOG10_C2, dtype=fp)
                     - jnp.log10(jnp.expm1(x)))
        spectral_flux_density = jnp.power(jnp.array(10.0, dtype=fp), log10_Fnu)
        spectral_flux_density = apply_sed_feature(
            features, spectral_flux_density, source_frequency, source_time_grid
        )
        spectra = spectral_flux_density * jnp.array(_C_ANG, dtype=fp) / (lam[None, :] ** 2)
        spectra = spectra * redshift_f
        bandflux = jnp.sum(lam[None, :] * response[None, :] * spectra, axis=1) * dwave / hc
        return bandflux / zpbandflux

    return jax.vmap(_one_band, in_axes=(0, 0, 0, 0), out_axes=1)(
        waves, trans, dwaves, zpbf
    )


def _interp_bandflux_grid(obs_source_time, obs_band_idx, source_time_grid, flux_grid):
    """Interpolate a ``(n_time, n_band)`` flux grid to observed ``(time, band)``."""
    band_indices = obs_band_idx.astype(jnp.int32)

    def _one_obs(t_i, band_i):
        return jnp.interp(t_i, source_time_grid, flux_grid[:, band_i])

    return jax.vmap(_one_obs)(obs_source_time, band_indices)


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

    def grid_photometry_model(
        *,
        obs_source_time,
        obs_band_idx,
        bridges,
        redshift,
        lum_dist,
        vej,
        temperature_floor,
        features=NO_SED_FEATURES,
        _time_observer_frame_grid=None,
        **bolometric_kwargs,
    ):
        return _grid_photometry_impl(
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
            time_observer_frame_grid=_time_observer_frame_grid,
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
    spectra_model._redback_jax_grid_photometry = grid_photometry_model
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


# ---------------------------------------------------------------------------
# CutoffBlackbody spectra factory  (for models where vej comes from the ODE)
# ---------------------------------------------------------------------------

def _cutoff_blackbody_freq_grid(
    frequency,          # (N_freq,) source-frame Hz
    log10_lbol,         # (N_time,) log10 erg/s
    vej_kms,            # (N_time,) km/s
    time,               # (N_time,) source-frame days
    temperature_floor,  # scalar K
    lum_dist,           # scalar cm
    cutoff_wavelength_ang,   # scalar Å
    alpha_uv,                # scalar, UV power-law index
):
    """CutoffBlackbody F_ν on a source-frame time × source-frame frequency grid.

    Returns F_ν in erg/s/cm²/Hz, shape (N_time, N_freq).
    Uses vmap over time steps; calls cutoff_blackbody_flux_density per step.
    """
    from redback_jax.sed import (
        _4PI as _CBD_4PI,
        _C_CM as _CBD_C_CM,
        _FLUX_CONST as _CBD_FLUX_CONST,
        _X_CONST as _CBD_X_CONST,
        cutoff_blackbody_log10_norm as _cbd_log10_norm,
    )

    T_ph, log10_r_ph = compute_temperature_floor_log10(
        time=time,
        log10_luminosity=log10_lbol,
        vej=vej_kms,
        temperature_floor=temperature_floor,
    )
    fp = log10_lbol.dtype
    r_ph = jnp.power(jnp.array(10.0, dtype=fp), log10_r_ph)   # (N_time,) cm
    freq = frequency.astype(fp)                                 # (N_freq,)
    dl   = jnp.asarray(lum_dist, dtype=fp)
    lc   = jnp.asarray(cutoff_wavelength_ang, dtype=fp)
    alp  = jnp.asarray(alpha_uv, dtype=fp)
    log10_norm = _cbd_log10_norm(log10_lbol, T_ph, r_ph, lc, alp)

    c_cm = jnp.asarray(_CBD_C_CM, dtype=fp)
    lam_cm = c_cm / freq[None, :]
    x = jnp.asarray(_CBD_X_CONST, dtype=fp) / (lam_cm * T_ph[:, None])
    x = jnp.clip(x, jnp.asarray(1e-10, dtype=fp), jnp.asarray(500.0, dtype=fp))

    lc_cm = lc * jnp.asarray(1e-8, dtype=fp)
    alpha = jnp.clip(alp, jnp.asarray(0.0, dtype=fp), jnp.asarray(3.99, dtype=fp))
    log10_r = log10_r_ph[:, None]
    log10_lam_cm = jnp.log10(lam_cm)
    log10_lc = jnp.log10(lc_cm)
    log10_planck_r2 = jnp.where(
        lam_cm < lc_cm,
        (
            jnp.asarray(2.0, dtype=fp) * log10_r
            - alpha * log10_lc
            - (jnp.asarray(5.0, dtype=fp) - alpha) * log10_lam_cm
        ),
        jnp.asarray(2.0, dtype=fp) * log10_r - jnp.asarray(5.0, dtype=fp) * log10_lam_cm,
    )
    log10_sed = (
        jnp.asarray(_math.log10(_CBD_FLUX_CONST), dtype=fp)
        + log10_planck_r2
        - jnp.log10(jnp.expm1(x))
        + log10_norm[:, None]
    )
    log10_F_nu = (
        log10_sed
        + jnp.log10(lam_cm * jnp.asarray(1e8, dtype=fp))
        - jnp.log10(freq[None, :])
        - jnp.asarray(_math.log10(_CBD_4PI), dtype=fp)
        - jnp.asarray(2.0, dtype=fp) * jnp.log10(dl)
    )
    return jnp.power(jnp.asarray(10.0, dtype=fp), log10_F_nu)


def make_cutoff_spectra_model(bolometric_and_vej_fn,
                               default_cutoff_wavelength=3000.0,
                               default_alpha_uv=1.0):
    """Wrap a (log10_lbol, vej_kms) bolometric function into a CutoffBlackbody spectra model.

    Unlike ``make_spectra_model``, this factory:
    - Expects the bolometric function to return ``(log10_lbol, vej_kms)`` (vej
      is derived from the ODE, not a free parameter).
    - Uses a CutoffBlackbody SED with user-controllable ``cutoff_wavelength``
      and ``alpha_uv`` that can be placed in the inference prior.

    Parameters
    ----------
    bolometric_and_vej_fn : callable
        ``f(time_days, **kwargs) -> (log10_lbol, vej_kms)``.
    default_cutoff_wavelength : float, Å  (default 3000)
    default_alpha_uv : float  (default 1.0)

    Returns
    -------
    spectra_model : callable
        ``spectra_model(redshift, lum_dist, temperature_floor,
                        cutoff_wavelength=..., alpha_uv=...,
                        features=NO_SED_FEATURES, **bolometric_kwargs)``
        returning ``namedtuple(time, lambdas, spectra)``.
    """

    def spectra_model(redshift, lum_dist, temperature_floor,
                      cutoff_wavelength=default_cutoff_wavelength,
                      alpha_uv=default_alpha_uv,
                      features=NO_SED_FEATURES,
                      _time_observer_frame_grid=None,
                      _lambda_observer_frame_grid=None,
                      **bolometric_kwargs):
        return _cutoff_spectra_model_impl(
            bolometric_and_vej_fn,
            redshift, lum_dist, temperature_floor,
            cutoff_wavelength, alpha_uv,
            features, bolometric_kwargs,
            time_observer_frame_grid=_time_observer_frame_grid,
            lambda_observer_frame_grid=_lambda_observer_frame_grid,
        )

    def grid_photometry_model(
        *,
        obs_source_time,
        obs_band_idx,
        bridges,
        redshift,
        lum_dist,
        temperature_floor,
        cutoff_wavelength=default_cutoff_wavelength,
        alpha_uv=default_alpha_uv,
        features=NO_SED_FEATURES,
        _time_observer_frame_grid=None,
        **bolometric_kwargs,
    ):
        return _cutoff_grid_photometry_impl(
            bolometric_and_vej_fn,
            obs_source_time=obs_source_time,
            obs_band_idx=obs_band_idx,
            bridges=bridges,
            redshift=redshift,
            lum_dist=lum_dist,
            temperature_floor=temperature_floor,
            cutoff_wavelength=cutoff_wavelength,
            alpha_uv=alpha_uv,
            features=features,
            bolometric_kwargs=bolometric_kwargs,
            time_observer_frame_grid=_time_observer_frame_grid,
        )

    spectra_model.__doc__ = (
        f"CutoffBlackbody spectra model wrapping ``{bolometric_and_vej_fn.__name__}``.\n\n"
        "Args:\n"
        "    redshift: source redshift\n"
        "    lum_dist: luminosity distance in cm\n"
        "    temperature_floor: floor temperature in K\n"
        "    cutoff_wavelength: UV cutoff wavelength in Å (default 3000, can be inferred)\n"
        "    alpha_uv: UV power-law suppression index (default 1.0, can be inferred)\n"
        "    features: SEDFeatures (default NO_SED_FEATURES)\n"
        "    **bolometric_kwargs: forwarded to the bolometric+vej function\n\n"
        "Returns:\n"
        "    namedtuple with fields ``time`` (days), ``lambdas`` (Angstrom), "
        "``spectra`` (erg/s/cm^2/Angstrom)\n"
    )
    spectra_model.__name__ = bolometric_and_vej_fn.__name__ + "_spectra"
    spectra_model._redback_jax_bolometric_fn = bolometric_and_vej_fn
    spectra_model._redback_jax_supports_custom_grids = True
    spectra_model._redback_jax_grid_photometry = grid_photometry_model
    return spectra_model


def _cutoff_spectra_model_impl(bolometric_and_vej_fn, redshift, lum_dist,
                                temperature_floor, cutoff_wavelength, alpha_uv,
                                features, bolometric_kwargs,
                                time_observer_frame_grid=None,
                                lambda_observer_frame_grid=None):
    """CutoffBlackbody spectra pipeline: grids → ODE (log10_lbol, vej) → SED → F_λ namedtuple."""
    lambda_observer_frame, time_observer_frame, frequency, time = _build_spectra_grids(
        redshift,
        time_observer_frame_grid=time_observer_frame_grid,
        lambda_observer_frame_grid=lambda_observer_frame_grid,
    )

    log10_lbol, vej_kms = bolometric_and_vej_fn(time, **bolometric_kwargs)

    spectral_flux_density = _cutoff_blackbody_freq_grid(
        frequency=frequency,
        log10_lbol=log10_lbol,
        vej_kms=vej_kms,
        time=time,
        temperature_floor=temperature_floor,
        lum_dist=lum_dist,
        cutoff_wavelength_ang=cutoff_wavelength,
        alpha_uv=alpha_uv,
    )

    spectral_flux_density = apply_sed_feature(
        features, spectral_flux_density, frequency, time)

    fp = time.dtype
    lam = lambda_observer_frame.astype(fp)
    spectra = spectral_flux_density * jnp.array(_C_ANG, dtype=fp) / (lam[None, :] ** 2)
    spectra = spectra * jnp.asarray(1.0 + redshift, dtype=fp)

    return namedtuple('output', ['time', 'lambdas', 'spectra'])(
        time=time_observer_frame,
        lambdas=lambda_observer_frame,
        spectra=spectra,
    )


def _cutoff_bandflux_from_state(
    obs_source_time,
    obs_band_idx,
    bridges,
    redshift,
    lum_dist,
    log10_lbol,
    temperature,
    log10_r_ph,
    cutoff_wavelength,
    alpha_uv,
    max_bandpass_points=None,
):
    """Integrate a cutoff-blackbody photosphere through observed bandpasses."""
    from redback_jax.sed import cutoff_blackbody_flux_density as _cbd_fd

    fp = obs_source_time.dtype
    waves, trans, dwaves, zpbf = _stack_bandpass_bridges(
        bridges, fp, max_points=max_bandpass_points
    )
    band_indices = obs_band_idx.astype(jnp.int32)
    z_f = jnp.asarray(redshift, dtype=fp)
    one_pz = jnp.asarray(1.0, dtype=fp) + z_f
    dl_f = jnp.asarray(lum_dist, dtype=fp)
    cw_f = jnp.asarray(cutoff_wavelength, dtype=fp)
    alpha_f = jnp.asarray(alpha_uv, dtype=fp)
    c_ang = jnp.asarray(_C_ANG, dtype=fp)
    hc = jnp.asarray(6.626e-27 * 2.998e18, dtype=fp)

    def _one_obs(band_idx, log10_l_i, temp_i, log10_r_i):
        lam = waves[band_idx]
        response = trans[band_idx]
        lam_src = lam / one_pz
        freq_src = jnp.asarray(_C, dtype=fp) / (lam_src * jnp.asarray(1e-8, dtype=fp))
        lbol_i = jnp.power(jnp.asarray(10.0, dtype=fp), log10_l_i)
        r_i = jnp.power(jnp.asarray(10.0, dtype=fp), log10_r_i)
        f_mjy = _cbd_fd(
            freq_src,
            jnp.broadcast_to(lbol_i, lam.shape),
            jnp.broadcast_to(temp_i, lam.shape),
            jnp.broadcast_to(r_i, lam.shape),
            dl_f,
            cw_f,
            alpha_f,
        )
        f_lam = f_mjy * jnp.asarray(1e-26, dtype=fp) * c_ang / lam ** 2 * one_pz
        bandflux = jnp.sum(lam * response * f_lam) * dwaves[band_idx] / hc
        return bandflux / zpbf[band_idx]

    return jax.vmap(_one_obs)(band_indices, log10_lbol, temperature, log10_r_ph)


def _cutoff_bandflux_grid_from_state(
    source_time_grid,
    bridges,
    redshift,
    lum_dist,
    log10_lbol_grid,
    temperature_grid,
    log10_r_grid,
    cutoff_wavelength,
    alpha_uv,
    max_bandpass_points=None,
):
    """Return cutoff-blackbody AB-normalised band fluxes on ``(n_time, n_band)``."""
    from redback_jax.sed import (
        _4PI as _CBD_4PI,
        _C_CM as _CBD_C_CM,
        _FLUX_CONST as _CBD_FLUX_CONST,
        _X_CONST as _CBD_X_CONST,
        cutoff_blackbody_log10_norm as _cbd_log10_norm,
    )

    fp = source_time_grid.dtype
    waves, trans, dwaves, zpbf = _stack_bandpass_bridges(
        bridges, fp, max_points=max_bandpass_points
    )
    z_f = jnp.asarray(redshift, dtype=fp)
    one_pz = jnp.asarray(1.0, dtype=fp) + z_f
    dl_f = jnp.asarray(lum_dist, dtype=fp)
    cw_f = jnp.asarray(cutoff_wavelength, dtype=fp)
    alpha_f = jnp.asarray(alpha_uv, dtype=fp)
    c_ang = jnp.asarray(_C_ANG, dtype=fp)
    hc = jnp.asarray(6.626e-27 * 2.998e18, dtype=fp)
    r_grid = jnp.power(jnp.asarray(10.0, dtype=fp), log10_r_grid)
    log10_norm_grid = _cbd_log10_norm(
        log10_lbol_grid, temperature_grid, r_grid, cw_f, alpha_f
    )

    lam = waves[None, :, :]
    response = trans[None, :, :]
    lam_src_cm = (lam / one_pz) * jnp.asarray(1e-8, dtype=fp)
    freq_src = jnp.asarray(_CBD_C_CM, dtype=fp) / lam_src_cm
    x = jnp.asarray(_CBD_X_CONST, dtype=fp) / (
        lam_src_cm * temperature_grid[:, None, None]
    )
    x = jnp.clip(x, jnp.asarray(1e-10, dtype=fp), jnp.asarray(500.0, dtype=fp))

    lc = cw_f * jnp.asarray(1e-8, dtype=fp)
    alpha = jnp.clip(alpha_f, jnp.asarray(0.0, dtype=fp), jnp.asarray(3.99, dtype=fp))
    log10_r = log10_r_grid[:, None, None]
    log10_lam_cm = jnp.log10(lam_src_cm)
    log10_lc = jnp.log10(lc)
    log10_planck_r2 = jnp.where(
        lam_src_cm < lc,
        (
            jnp.asarray(2.0, dtype=fp) * log10_r
            - alpha * log10_lc
            - (jnp.asarray(5.0, dtype=fp) - alpha) * log10_lam_cm
        ),
        jnp.asarray(2.0, dtype=fp) * log10_r - jnp.asarray(5.0, dtype=fp) * log10_lam_cm,
    )
    log10_sed = (
        jnp.asarray(_math.log10(_CBD_FLUX_CONST), dtype=fp)
        + log10_planck_r2
        - jnp.log10(jnp.expm1(x))
        + log10_norm_grid[:, None, None]
    )
    log10_f_mjy = (
        log10_sed
        + jnp.log10(lam_src_cm * jnp.asarray(1e8, dtype=fp))
        - jnp.log10(freq_src)
        - jnp.asarray(_math.log10(_CBD_4PI), dtype=fp)
        - jnp.asarray(2.0, dtype=fp) * jnp.log10(dl_f)
        + jnp.asarray(26.0, dtype=fp)
    )
    log10_f_lam = (
        log10_f_mjy
        - jnp.asarray(26.0, dtype=fp)
        + jnp.log10(c_ang)
        - jnp.asarray(2.0, dtype=fp) * jnp.log10(lam)
        + jnp.log10(one_pz)
    )
    f_lam = jnp.power(jnp.asarray(10.0, dtype=fp), log10_f_lam)
    bandflux = jnp.sum(lam * response * f_lam, axis=2) * dwaves[None, :] / hc
    return bandflux / zpbf[None, :]


def _cutoff_grid_photometry_impl(
    bolometric_and_vej_fn,
    *,
    obs_source_time,
    obs_band_idx,
    bridges,
    redshift,
    lum_dist,
    temperature_floor,
    cutoff_wavelength,
    alpha_uv,
    features,
    bolometric_kwargs,
    time_observer_frame_grid=None,
):
    """Cutoff-blackbody photometry path using a bolometric/photosphere grid."""
    fp = obs_source_time.dtype
    if time_observer_frame_grid is None:
        source_time_grid = _DEFAULT_TIME_SOURCE_FRAME.astype(fp)
    else:
        source_time_grid = (
            jnp.asarray(time_observer_frame_grid, dtype=fp)
            / jnp.asarray(1.0 + redshift, dtype=fp)
        )

    log10_lbol_grid, vej_grid = bolometric_and_vej_fn(source_time_grid, **bolometric_kwargs)
    T_grid, log10_r_grid = compute_temperature_floor_log10(
        time=source_time_grid,
        log10_luminosity=log10_lbol_grid,
        vej=vej_grid,
        temperature_floor=temperature_floor,
    )
    valid = obs_source_time >= source_time_grid[0]
    flux_grid = _cutoff_bandflux_grid_from_state(
        source_time_grid=source_time_grid,
        bridges=bridges,
        redshift=redshift,
        lum_dist=lum_dist,
        log10_lbol_grid=log10_lbol_grid,
        temperature_grid=T_grid,
        log10_r_grid=log10_r_grid,
        cutoff_wavelength=cutoff_wavelength,
        alpha_uv=alpha_uv,
        max_bandpass_points=_DEFAULT_GRID_BANDPASS_POINTS,
    )
    norm_fluxes = _interp_bandflux_grid(
        obs_source_time, obs_band_idx, source_time_grid, flux_grid
    )
    return jnp.where(valid, norm_fluxes, 0.0)


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


def _grid_photometry_impl(
    bolometric_fn,
    *,
    obs_source_time,
    obs_band_idx,
    bridges,
    redshift,
    lum_dist,
    vej,
    temperature_floor,
    features,
    bolometric_kwargs,
    bolo_accepts_vej=False,
    time_observer_frame_grid=None,
):
    """Photometry path using a bolometric/photosphere grid but no spectra cube."""
    if bolo_accepts_vej:
        bolometric_kwargs = {'vej': vej, **bolometric_kwargs}

    fp = obs_source_time.dtype
    if time_observer_frame_grid is None:
        source_time_grid = _DEFAULT_TIME_SOURCE_FRAME.astype(fp)
    else:
        source_time_grid = (
            jnp.asarray(time_observer_frame_grid, dtype=fp)
            / jnp.asarray(1.0 + redshift, dtype=fp)
        )

    log10_lbol_grid = bolometric_fn(source_time_grid, **bolometric_kwargs)
    T_grid, log10_r_grid = compute_temperature_floor_log10(
        time=source_time_grid,
        log10_luminosity=log10_lbol_grid,
        vej=vej,
        temperature_floor=temperature_floor,
    )

    valid = obs_source_time >= source_time_grid[0]
    flux_grid = _blackbody_bandflux_grid_from_state(
        source_time_grid=source_time_grid,
        bridges=bridges,
        redshift=redshift,
        lum_dist=lum_dist,
        temperature_grid=T_grid,
        log10_r_grid=log10_r_grid,
        features=features,
        max_bandpass_points=_DEFAULT_GRID_BANDPASS_POINTS,
    )
    norm_fluxes = _interp_bandflux_grid(
        obs_source_time, obs_band_idx, source_time_grid, flux_grid
    )
    return jnp.where(valid, norm_fluxes, 0.0)
