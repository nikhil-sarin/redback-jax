"""
JAX-friendly classes for supernova modeling.
"""

import math as _math
import os as _os
from collections import namedtuple

import numpy as _np
import jax
from jax import jit, lax
import jax.numpy as jnp
from functools import partial
from scipy.interpolate import RegularGridInterpolator as _RGI
from wcosmo import wcosmo

from redback_jax.utils.citation_wrapper import citation_wrapper
from redback_jax.utils.cosmology import PLANCK18_H0, PLANCK18_OM0, MPC_TO_CM
from redback_jax.conversions import calc_kcorrected_properties, lambda_to_nu
from redback_jax.interaction_processes import (
    _compute_diffusion_constants,
    diffusion_convert_luminosity,
    csm_diffusion_convert_luminosity,
)
from redback_jax.models.sed_features import NO_SED_FEATURES, apply_sed_feature
from redback_jax.photosphere import compute_temperature_floor_log10
from redback_jax.sed import cutoff_blackbody_flux_density

# Enable float64 — the general magnetar ODE state spans ~16 orders of magnitude.
jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Physical constants — Python floats (not astropy, avoids float64 promotion)
# ---------------------------------------------------------------------------
_SOLAR_MASS    = 1.989e33    # g
_SPEED_OF_LIGHT = 2.998e10   # cm/s
_KM_CGS        = 1.0e5       # cm/km
_DAY_TO_S      = 86400.0     # s/day
_AU_CGS        = 1.496e13    # cm/AU
_MPC_TO_CM     = MPC_TO_CM   # cm/Mpc (from redback_jax.utils.cosmology)

# Log10 of key constants (float32-safe pre-computation)
_LOG10_MSUN    = _math.log10(_SOLAR_MASS)
_LOG10_CCGS    = _math.log10(_SPEED_OF_LIGHT)
_LOG10_KM_CGS  = _math.log10(_KM_CGS)

# Magnetar log10 constants
# erot = 2.6e52 * (mass_ns/1.4)^1.5 * p0^-2   [erg]
# tp   = 1.3e5  * bp^-2 * p0^2 * (mass_ns/1.4)^1.5 / sin^2(theta_pb)  [s]
_LOG10_EROT_COEFF = _math.log10(2.6e52)
_LOG10_TP_COEFF   = _math.log10(1.3e5)
_LOG10_2_FLOAT    = _math.log10(2.0)
_ARNETT_HALF_TIMESTEPS = 50
_ARNETT_MIN_LOG_SPACING = -3.0
_ARNETT_START_DAY = 0.01

# ---------------------------------------------------------------------------
# General magnetar ODE constants (aliases onto the names used above + new values)
# ---------------------------------------------------------------------------
_C      = _SPEED_OF_LIGHT   # cm/s  (alias)
_MSUN   = _SOLAR_MASS       # g     (alias)
_DAY    = _DAY_TO_S         # s/day (alias)
_MP     = 1.673e-24         # proton mass, g
_ARAD   = 7.566e-15         # radiation constant a = 4σ/c, erg/cm³/K⁴
_NI56_LUM  = 6.45e43        # Ni-56 specific luminosity, erg/s per M_sun
_CO56_LUM  = 1.45e43        # Co-56 specific luminosity, erg/s per M_sun
_NI56_LIFE = 8.8   * _DAY_TO_S   # Ni-56 e-folding time, s
_CO56_LIFE = 111.3 * _DAY_TO_S   # Co-56 e-folding time, s
_R0_CM  = 1.0e11            # initial ejecta radius, cm
_N_ISM  = 1.0e-5            # ISM number density, cm⁻³


# ---------------------------------------------------------------------------
# CSM table — loaded once at import time as static numpy arrays.
# Columns: eta, nn, Bf, Br, AA  (300 rows = 10 eta × 30 nn)
# ---------------------------------------------------------------------------
_CSM_TABLE_PATH = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)),
                                'tables', 'csm_table.txt')
_csm_eta_raw, _csm_nn_raw, _csm_bf_raw, _csm_br_raw, _csm_aa_raw = _np.loadtxt(
    _CSM_TABLE_PATH, delimiter=',', unpack=True)
_csm_eta_unique = _np.unique(_csm_eta_raw)   # 10 values  0–2
_csm_nn_unique  = _np.unique(_csm_nn_raw)    # 30 values  6–14
# Redback reshape: (10,30).T → (30,10); grid axes are (nn, eta)
_csm_AA_grid = _np.reshape(_csm_aa_raw, (10, 30)).T   # (30, 10)
_csm_Bf_grid = _np.reshape(_csm_bf_raw, (10, 30)).T   # (30, 10)
_csm_Br_grid = _np.reshape(_csm_br_raw, (10, 30)).T   # (30, 10)
_csm_AA_interp = _RGI((_csm_nn_unique, _csm_eta_unique), _csm_AA_grid,
                       bounds_error=False, fill_value=None)
_csm_Bf_interp = _RGI((_csm_nn_unique, _csm_eta_unique), _csm_Bf_grid,
                       bounds_error=False, fill_value=None)
_csm_Br_interp = _RGI((_csm_nn_unique, _csm_eta_unique), _csm_Br_grid,
                       bounds_error=False, fill_value=None)



def blackbody_to_flux_density(temperature, r_photosphere, dl, frequency):
    """
    A general blackbody_to_flux_density formula

    :param temperature: effective temperature in kelvin
    :param r_photosphere: photosphere radius in cm
    :param dl: luminosity_distance in cm
    :param frequency: frequency to calculate in Hz
    :return: flux_density in erg/s/Hz/cm^2
    """
    # Use Python float constants to avoid astropy float64 promotion
    _h   = 6.626e-27   # erg s
    _c   = 2.998e10    # cm/s
    _kB  = 1.381e-16   # erg/K
    num  = 2.0 * jnp.pi * _h * frequency ** 3 * r_photosphere ** 2
    denom = dl ** 2 * _c ** 2
    frac  = 1.0 / jnp.expm1((_h * frequency) / (_kB * temperature))
    return num / denom * frac


@jit
def _nickelcobalt_log10_engine(time, f_nickel, mej):
    """Ni/Co decay engine — returns log10(L) in erg/s (float32-safe).

    :param time: time in days
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :return: log10 of bolometric luminosity in erg/s
    """
    _log10_ni = _math.log10(6.45e43)
    _log10_co = _math.log10(1.45e43)
    ni56_life = 8.8    # days
    co56_life = 111.3  # days
    fp = time.dtype
    log10_mni = jnp.log10(jnp.maximum(f_nickel * mej, jnp.array(1e-30, dtype=fp)))
    log10_a = jnp.array(_log10_ni, dtype=fp) + (-time / ni56_life) * jnp.array(_math.log10(_math.e), dtype=fp)
    log10_b = jnp.array(_log10_co, dtype=fp) + (-time / co56_life) * jnp.array(_math.log10(_math.e), dtype=fp)
    log10_max = jnp.maximum(log10_a, log10_b)
    log10_sum = log10_max + jnp.log10(
        jnp.power(jnp.array(10.0, dtype=fp), log10_a - log10_max)
        + jnp.power(jnp.array(10.0, dtype=fp), log10_b - log10_max))
    return log10_mni + log10_sum


@jit
def _redback_arnett_quadrature_nodes(tau_diff_days, max_time_days):
    """Build the same adaptive log-mirror nodes used by redback's diffusion path."""
    fp = tau_diff_days.dtype
    min_ratio = jnp.maximum(
        tau_diff_days / jnp.maximum(max_time_days, jnp.array(1e-30, dtype=fp)),
        jnp.array(1e-30, dtype=fp),
    )
    log_min = jnp.log10(min_ratio) + jnp.array(_ARNETT_MIN_LOG_SPACING, dtype=fp)
    lsp = jnp.power(
        jnp.array(10.0, dtype=fp),
        jnp.linspace(log_min, jnp.array(0.0, dtype=fp), _ARNETT_HALF_TIMESTEPS),
    )
    return jnp.sort(jnp.concatenate((lsp, 1.0 - lsp)))


@jit
def _diffused_nickelcobalt_log10_luminosity(time, f_nickel, mej, *, vej, kappa, kappa_gamma):
    """
    Specialized Arnett diffusion that evaluates the Ni/Co engine directly at the
    redback quadrature points instead of interpolating a dense precomputed grid.
    """
    fp = time.dtype
    eval_time = jnp.maximum(time, jnp.array(_ARNETT_START_DAY, dtype=fp))
    tb = jnp.array(_ARNETT_START_DAY, dtype=fp)
    dense_end = eval_time[-1] + jnp.array(100.0, dtype=fp)

    log10_td, log10_A = _compute_diffusion_constants(
        jnp.log10(jnp.maximum(jnp.asarray(kappa, dtype=fp), jnp.array(1e-30, dtype=fp))),
        jnp.log10(jnp.maximum(jnp.asarray(kappa_gamma, dtype=fp), jnp.array(1e-30, dtype=fp))),
        jnp.log10(jnp.maximum(jnp.asarray(mej, dtype=fp), jnp.array(1e-30, dtype=fp))),
        jnp.log10(jnp.maximum(jnp.asarray(vej, dtype=fp), jnp.array(1e-30, dtype=fp))),
    )
    tau_diff = jnp.power(jnp.array(10.0, dtype=fp), log10_td)
    trap_coeff = jnp.power(jnp.array(10.0, dtype=fp), log10_A)
    quad_nodes = _redback_arnett_quadrature_nodes(tau_diff, dense_end)

    int_times = jnp.clip(tb + (eval_time[:, None] - tb) * quad_nodes[None, :], tb, dense_end)
    log10_engine = _nickelcobalt_log10_engine(int_times, f_nickel, mej)
    log10_scale = _nickelcobalt_log10_engine(jnp.array([tb], dtype=fp), f_nickel, mej)[0]
    engine_n = jnp.power(jnp.array(10.0, dtype=fp), log10_engine - log10_scale)

    exponent = jnp.clip((int_times ** 2 - eval_time[:, None] ** 2) / tau_diff ** 2, -80.0, 0.0)
    integrand = engine_n * int_times * jnp.exp(exponent)
    integral = jnp.trapezoid(integrand, int_times, axis=1)
    trap_factor = -jnp.expm1(
        -trap_coeff / jnp.maximum(eval_time ** 2, jnp.array(1e-30, dtype=fp))
    )
    lum_n = jnp.maximum(2.0 / tau_diff ** 2 * integral * trap_factor, 0.0)
    return jnp.log10(jnp.maximum(lum_n, jnp.array(1e-30, dtype=fp))) + log10_scale


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
@jit
def arnett_bolometric(time, f_nickel, mej, *, vej=None, kappa=None, kappa_gamma=None):
    """
    Bolometric Arnett (1982) light curve with Ni/Co decay engine + diffusion.

    :param time: time in days
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param kappa: opacity in cm^2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm^2/g (required)
    :param vej: ejecta velocity in km/s (required)
    :return: log10 of bolometric luminosity in erg/s
    """
    return _diffused_nickelcobalt_log10_luminosity(
        time, f_nickel, mej, vej=vej, kappa=kappa, kappa_gamma=kappa_gamma)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
@jit
def arnett_with_features_lum_dist(
    f_nickel, mej, *, redshift=0.0, lum_dist=None,
    vej=None, kappa=None, kappa_gamma=None,
    temperature_floor=None, features=NO_SED_FEATURES,
):
    """
    Arnett model with spectra — SED has time-evolving spectral features.

    :param redshift: source redshift
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param lum_dist: luminosity distance in cm
    :param kappa: opacity in cm^2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm^2/g (required)
    :param vej: ejecta velocity in km/s (required)
    :param temperature_floor: floor temperature in K
    :param features: SEDFeatures object
    :return: namedtuple(time, lambdas, spectra)
    """
    lambda_observer_frame = jnp.geomspace(100.0, 60000.0, 100)
    time_temp = jnp.geomspace(0.1, 3000.0, 3000)  # days
    time_observer_frame = time_temp * (1.0 + redshift)
    frequency, time = calc_kcorrected_properties(
        frequency=lambda_to_nu(lambda_observer_frame),
        redshift=redshift,
        time=time_observer_frame,
    )

    log10_lbol = _diffused_nickelcobalt_log10_luminosity(
        time, f_nickel, mej, vej=vej, kappa=kappa, kappa_gamma=kappa_gamma)

    T_ph, log10_r_ph = compute_temperature_floor_log10(
        time=time, log10_luminosity=log10_lbol,
        vej=vej, temperature_floor=temperature_floor)

    fp = time.dtype
    nu = frequency.astype(fp)
    dl = jnp.asarray(lum_dist, dtype=fp)
    log10_dl = jnp.log10(jnp.maximum(dl, jnp.array(1.0, dtype=fp)))

    _H   = 6.626e-27; _KB = 1.381e-16; _C = 2.998e10
    x = (_H / _KB) * nu[None, :] / jnp.maximum(T_ph[:, None], jnp.array(1.0, dtype=fp))
    x = jnp.clip(x, jnp.array(1e-10, dtype=fp), jnp.array(80.0, dtype=fp))
    log10_2pi_h = jnp.array(_math.log10(2.0 * _math.pi * _H), dtype=fp)
    log10_c2    = jnp.array(_math.log10(_C ** 2), dtype=fp)

    log10_Fnu = (log10_2pi_h
                 + 3.0 * jnp.log10(nu[None, :])
                 + 2.0 * log10_r_ph[:, None]
                 - 2.0 * log10_dl
                 - log10_c2
                 - jnp.log10(jnp.expm1(x)))

    spectral_flux_density = jnp.power(jnp.array(10.0, dtype=fp), log10_Fnu)

    spectral_flux_density = apply_sed_feature(
        features, spectral_flux_density, frequency, time)

    lam    = lambda_observer_frame.astype(fp)
    spectra = spectral_flux_density * jnp.array(2.998e18, dtype=fp) / (lam[None, :] ** 2)
    spectra = spectra * jnp.asarray(1.0 + redshift, dtype=fp)
    return namedtuple('output', ['time', 'lambdas', 'spectra'])(
        time=time_observer_frame, lambdas=lambda_observer_frame, spectra=spectra)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
def arnett_with_features_cosmology(
    f_nickel, mej, *, redshift=0.0, cosmo_H0=PLANCK18_H0, cosmo_Om0=PLANCK18_OM0,
    vej=None, kappa=None, kappa_gamma=None,
    temperature_floor=None, features=NO_SED_FEATURES,
):
    """
    Arnett model with cosmological luminosity distance calculation.

    :param redshift: source redshift
    :param f_nickel: fraction of nickel mass
    :param mej: total ejecta mass in solar masses
    :param cosmo_H0: Hubble constant (km/s/Mpc)
    :param cosmo_Om0: matter density parameter
    :param kappa: opacity in cm^2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm^2/g (required)
    :param vej: ejecta velocity in km/s (required)
    :param temperature_floor: floor temperature in K
    :param features: SEDFeatures object
    :return: namedtuple(time, lambdas, spectra)
    """
    dl = wcosmo.luminosity_distance(redshift, cosmo_H0, cosmo_Om0).value * _MPC_TO_CM
    return arnett_with_features_lum_dist(
        f_nickel=f_nickel, mej=mej, redshift=redshift, lum_dist=dl,
        vej=vej, kappa=kappa, kappa_gamma=kappa_gamma,
        temperature_floor=temperature_floor, features=features)


# ---------------------------------------------------------------------------
# Magnetar-powered supernova
# Reference: Kasen & Bildsten 2010, Inserra et al. 2013, Yu et al. 2017
# ---------------------------------------------------------------------------

@jit
def _magnetar_log10_lbol(time_days, log10_p0_ms, log10_bp, mass_ns, theta_pb):
    """
    Dipole spin-down log10 luminosity (float32-safe). Returns log10(L) in erg/s.

    :param time_days: source-frame time in days
    :param log10_p0_ms: log10 of initial spin period in milliseconds
    :param log10_bp: log10 of polar B-field in units of 10^14 G
    :param mass_ns: NS mass in solar masses
    :param theta_pb: spin–B-field angle in radians
    :return: log10 of luminosity in erg/s
    """
    t_s = time_days * _DAY_TO_S

    log10_mass_ratio = jnp.log10(mass_ns / 1.4)
    log10_erot = (_LOG10_EROT_COEFF + 1.5 * log10_mass_ratio - 2.0 * log10_p0_ms)
    log10_tp = (_LOG10_TP_COEFF - 2.0 * log10_bp + 2.0 * log10_p0_ms
                + 1.5 * log10_mass_ratio
                - jnp.log10(jnp.maximum(jnp.sin(theta_pb) ** 2, 1e-10)))
    tp = jnp.power(10.0, log10_tp)

    log10_L = (_LOG10_2_FLOAT + log10_erot - log10_tp
               - 2.0 * jnp.log10(1.0 + 2.0 * t_s / tp))
    return log10_L


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2006ApJ...648L..51S/abstract')
@jit
def _basic_magnetar_engine(time_days, p0, bp, mass_ns, theta_pb):
    """
    Dipole spin-down — returns log10(L) in erg/s.

    :param time_days: source-frame time in days
    :param p0: initial spin period in milliseconds
    :param bp: polar B-field in units of 10^14 G
    :param mass_ns: NS mass in solar masses
    :param theta_pb: spin–B-field angle in radians
    :return: log10 of luminosity in erg/s
    """
    return _magnetar_log10_lbol(
        time_days,
        jnp.log10(jnp.maximum(p0, 1e-10)),
        jnp.log10(jnp.maximum(bp, 1e-10)),
        mass_ns, theta_pb)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017ApJ...850...55N/abstract')
@jit
def magnetar_powered_bolometric(time, p0, bp, mass_ns, theta_pb,
                                 mej, kappa, kappa_gamma, vej):
    """
    Bolometric light curve of a magnetar-powered supernova (Arnett diffusion).

    :param time: source-frame time in days
    :param p0: initial spin period in milliseconds
    :param bp: polar B-field in units of 10^14 G
    :param mass_ns: NS mass in solar masses
    :param theta_pb: spin–B-field angle in radians
    :param mej: ejecta mass in solar masses
    :param kappa: optical opacity in cm^2/g
    :param kappa_gamma: gamma-ray opacity in cm^2/g
    :param vej: ejecta velocity in km/s
    :return: log10 of bolometric luminosity in erg/s
    """
    dense_times = jnp.linspace(0.01, time[-1] + 100.0, 1000)
    log10_p0 = jnp.log10(jnp.maximum(p0, 1e-10))
    log10_bp = jnp.log10(jnp.maximum(bp, 1e-10))
    log10_dense_lbols = _magnetar_log10_lbol(dense_times, log10_p0, log10_bp, mass_ns, theta_pb)
    _, log10_lbol = diffusion_convert_luminosity(
        time=time, dense_times=dense_times, log10_luminosity=log10_dense_lbols,
        kappa=kappa, kappa_gamma=kappa_gamma, mej=mej, vej=vej)
    return log10_lbol


# ---------------------------------------------------------------------------
# CSM interaction
# Reference: Chevalier & Fransson 1994, Chatzopoulos et al. 2013,
#            Villar et al. 2017, Jacobson-Galan et al. 2020
# ---------------------------------------------------------------------------

def _get_csm_coefficients(nn, eta):
    """
    Lookup AA, Bf, Br from the pre-loaded CSM table via scipy interpolation.
    Runs *outside* JIT (called with concrete Python/numpy scalars).
    """
    pt = _np.array([[nn, eta]])
    AA = float(_csm_AA_interp(pt)[0])
    Bf = float(_csm_Bf_interp(pt)[0])
    Br = float(_csm_Br_interp(pt)[0])
    return AA, Bf, Br


@jit
def _csm_engine(time, mej, csm_mass, vej, eta, rho, kappa, r0, nn, AA, Bf, Br,
                delta, efficiency):
    """
    JAX CSM interaction engine (Chevalier 1982 forward/reverse shocks).
    nn, AA, Bf, Br are concrete floats — passed in from the static table lookup.
    Uses log10 arithmetic for Esn and g_n to stay float32-safe.

    :param time: source-frame time in days
    :param mej: ejecta mass in solar masses
    :param csm_mass: CSM mass in solar masses
    :param vej: ejecta velocity in km/s
    :param eta: CSM density profile exponent
    :param rho: CSM density amplitude in g/cm^3
    :param kappa: opacity in cm^2/g
    :param r0: inner CSM radius in AU
    :param nn: ejecta density power-law slope (concrete float)
    :param AA, Bf, Br: CSM shock coefficients (concrete floats from table)
    :param delta: inner ejecta density slope
    :param efficiency: kinetic-to-luminosity conversion efficiency
    :return: (lbol, r_photosphere, mass_csm_threshold)
    """
    mej_g      = mej * _SOLAR_MASS
    csm_mass_g = csm_mass * _SOLAR_MASS
    r0_cm      = r0 * _AU_CGS
    vej_cms    = vej * _KM_CGS

    # Esn = 3 * vej^2 * mej / 10  [erg] — computed in log10 to avoid overflow
    log10_Esn  = (jnp.log10(3.0 / 10.0) + 2.0 * jnp.log10(vej_cms)
                  + jnp.log10(mej_g))
    Esn        = jnp.power(10.0, log10_Esn)

    ti  = 1.0  # seconds offset

    qq         = rho * r0_cm ** eta
    radius_csm = ((3.0 - eta) / (4.0 * jnp.pi * qq) * csm_mass_g
                  + r0_cm ** (3.0 - eta)) ** (1.0 / (3.0 - eta))
    r_photosphere = jnp.abs(
        (-2.0 * (1.0 - eta) / (3.0 * kappa * qq)
         + radius_csm ** (1.0 - eta)) ** (1.0 / (1.0 - eta)))
    mass_csm_threshold = jnp.abs(
        4.0 * jnp.pi * qq / (3.0 - eta)
        * (r_photosphere ** (3.0 - eta) - r0_cm ** (3.0 - eta)))

    # g_n in log10 to avoid overflow
    # g_n = 1/(4pi*(nn-delta)) * [2*(5-delta)*(nn-5)*Esn]^((nn-3)/2) / [(3-delta)*(nn-3)*mej_g]^((nn-5)/2)
    log10_g_n = (- jnp.log10(4.0 * jnp.pi * (nn - delta))
                 + ((nn - 3.0) / 2.0) * jnp.log10(jnp.maximum(
                     2.0 * (5.0 - delta) * (nn - 5.0) * Esn, 1e-300))
                 - ((nn - 5.0) / 2.0) * jnp.log10(
                     (3.0 - delta) * (nn - 3.0) * mej_g))
    g_n = jnp.power(10.0, log10_g_n)

    t_FS = (jnp.abs(
        (3.0 - eta) * qq ** ((3.0 - nn) / (nn - eta))
        * (AA * g_n) ** ((eta - 3.0) / (nn - eta))
        / (4.0 * jnp.pi * Bf ** (3.0 - eta))
    ) ** ((nn - eta) / ((nn - 3.0) * (3.0 - eta)))
            * mass_csm_threshold ** ((nn - eta) / ((nn - 3.0) * (3.0 - eta))))

    t_RS = (vej_cms / (Br * (AA * g_n / qq) ** (1.0 / (nn - eta)))
            * (1.0 - (3.0 - nn) * mej_g
               / (4.0 * jnp.pi * vej_cms ** (3.0 - nn) * g_n))
            ** (1.0 / (3.0 - nn))) ** ((nn - eta) / (eta - 3.0))

    t_s     = time * _DAY_TO_S + ti
    exp_FS  = (2.0 * nn + 6.0 * eta - nn * eta - 15.0) / (nn - eta)

    lbol_FS = (2.0 * jnp.pi / (nn - eta) ** 3
               * g_n ** ((5.0 - eta) / (nn - eta))
               * qq ** ((nn - 5.0) / (nn - eta))
               * (nn - 3.0) ** 2 * (nn - 5.0)
               * Bf ** (5.0 - eta)
               * AA ** ((5.0 - eta) / (nn - eta))
               * t_s ** exp_FS)

    lbol_RS = (2.0 * jnp.pi
               * (AA * g_n / qq) ** ((5.0 - nn) / (nn - eta))
               * Br ** (5.0 - nn) * g_n
               * ((3.0 - eta) / (nn - eta)) ** 3
               * t_s ** exp_FS)

    lbol_FS = jnp.where(t_FS - t_s > 0, lbol_FS, 0.0)
    lbol_RS = jnp.where(t_RS - t_s > 0, lbol_RS, 0.0)
    lbol    = efficiency * (lbol_FS + lbol_RS)

    return lbol, r_photosphere, mass_csm_threshold


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract')
@jit
def magnetar_nickel_bolometric(time, f_nickel, mej, p0, bp, mass_ns, theta_pb,
                                kappa, kappa_gamma, vej):
    """
    Bolometric light curve powered by both a magnetar and Ni/Co radioactive decay
    (Arnett diffusion). The two luminosity sources are added before diffusion.

    Reference: Gomez et al. 2018 (https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract)

    :param time: source-frame time in days
    :param f_nickel: nickel mass fraction (M_Ni = f_nickel * mej)
    :param mej: total ejecta mass in solar masses
    :param p0: initial spin period in milliseconds
    :param bp: polar B-field in units of 10^14 G
    :param mass_ns: NS mass in solar masses
    :param theta_pb: spin–B-field angle in radians
    :param kappa: optical opacity in cm^2/g
    :param kappa_gamma: gamma-ray opacity in cm^2/g
    :param vej: ejecta velocity in km/s
    :return: log10 of bolometric luminosity in erg/s
    """
    dense_times = jnp.linspace(0.01, time[-1] + 100.0, 1000)

    # Ni/Co decay engine in log10 space
    log10_nickel = _nickelcobalt_log10_engine(dense_times, f_nickel, mej)

    # Magnetar spin-down engine in log10 space
    log10_p0 = jnp.log10(jnp.maximum(p0, jnp.array(1e-10, dtype=time.dtype)))
    log10_bp = jnp.log10(jnp.maximum(bp, jnp.array(1e-10, dtype=time.dtype)))
    log10_mag = _magnetar_log10_lbol(dense_times, log10_p0, log10_bp, mass_ns, theta_pb)

    # Add the two luminosity sources in log10 space (logsumexp-style, float32-safe)
    fp = time.dtype
    log10_max = jnp.maximum(log10_nickel, log10_mag)
    log10_combined = log10_max + jnp.log10(
        jnp.power(jnp.array(10.0, dtype=fp), log10_nickel - log10_max)
        + jnp.power(jnp.array(10.0, dtype=fp), log10_mag - log10_max))

    _, log10_lbol = diffusion_convert_luminosity(
        time=time, dense_times=dense_times, log10_luminosity=log10_combined,
        kappa=kappa, kappa_gamma=kappa_gamma, mej=mej, vej=vej)
    return log10_lbol


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2013ApJ...773...76C/abstract,'
                  'https://ui.adsabs.harvard.edu/abs/2017ApJ...849...70V/abstract,'
                  'https://ui.adsabs.harvard.edu/abs/2020RNAAS...4...16J/abstract')
def csm_interaction_bolometric(time, mej, csm_mass, vej, eta, rho, kappa, r0,
                                nn=12, delta=1, efficiency=0.5):
    """
    Bolometric CSM-interaction light curve (Chevalier 1982 shocks + diffusion).

    :param time: source-frame time in days
    :param mej: ejecta mass in solar masses
    :param csm_mass: CSM mass in solar masses
    :param vej: ejecta velocity in km/s
    :param eta: CSM density profile exponent
    :param rho: CSM density amplitude in g/cm^3
    :param kappa: opacity in cm^2/g
    :param r0: inner CSM radius in AU
    :param nn: ejecta density power-law slope (default 12)
    :param delta: inner ejecta density slope (default 1)
    :param efficiency: kinetic-to-luminosity efficiency (default 0.5)
    :return: log10 of bolometric luminosity in erg/s
    """
    AA, Bf, Br = _get_csm_coefficients(nn, eta)
    dense_times_jnp = jnp.linspace(0.1, time[-1] + 100.0, 1000)
    _nn, _AA, _Bf, _Br, _delta, _eff = float(nn), AA, Bf, Br, float(delta), float(efficiency)

    @jit
    def _engine_and_diffuse(time, dense_times):
        dense_lbols, r_phot, mass_csm_thresh = _csm_engine(
            dense_times, mej, csm_mass, vej, eta, rho, kappa, r0,
            _nn, _AA, _Bf, _Br, _delta, _eff)
        log10_dense = jnp.log10(jnp.maximum(dense_lbols, jnp.array(1e-30, dtype=dense_lbols.dtype)))
        return csm_diffusion_convert_luminosity(
            time=time, dense_times=dense_times, log10_luminosity=log10_dense,
            kappa=kappa, r_photosphere=r_phot, mass_csm_threshold=mass_csm_thresh)

    return _engine_and_diffuse(time, dense_times_jnp)


# ===========================================================================
# General magnetar-driven supernova (relativistic ODE, float64)
# Translated from redback's _ejecta_dynamics_and_interaction + magnetar_only
# + TemperatureFloor + CutoffBlackbody (Sarin+22, Omand&Sarin+24, Nicholl+17)
# ===========================================================================

def _make_scan_step(mej_g, kappa, kappa_gamma, f_nickel, fp):
    """Return a scan-compatible step function closed over the physical parameters.

    All parameters are JAX arrays of dtype *fp* (float64).  The closure lets
    ``jax.lax.scan`` treat them as compile-time constants within a JIT trace.
    """
    c        = jnp.array(_C,      dtype=fp)
    m_p      = jnp.array(_MP,     dtype=fp)
    a_rad    = jnp.array(_ARAD,   dtype=fp)
    msun     = jnp.array(_MSUN,   dtype=fp)
    ni56_lum  = jnp.array(_NI56_LUM,  dtype=fp)
    co56_lum  = jnp.array(_CO56_LUM,  dtype=fp)
    ni56_life = jnp.array(_NI56_LIFE, dtype=fp)
    co56_life = jnp.array(_CO56_LIFE, dtype=fp)
    n_ism    = jnp.array(_N_ISM,  dtype=fp)
    pi       = jnp.array(jnp.pi,  dtype=fp)
    one      = jnp.array(1.0,     dtype=fp)
    zero     = jnp.array(0.0,     dtype=fp)
    tiny     = jnp.array(1e-15,   dtype=fp)

    # nickel mass in solar masses
    nickel_msun = f_nickel * mej_g / msun

    def scan_step(carry, xs):
        """One explicit-Euler step, replicating _ejecta_dynamics_and_interaction.

        The carry order is:
          (gamma, r, V, E, prev_dgamma_dt, prev_drdt, prev_dV_dt, prev_dE_dt)

        xs = (t_i, mag_lum_i, dt_i)

        Convention (matching redback):
          - beta and doppler are computed from the *old* gamma (pre-update).
          - All spatial/energy state is then Euler-updated.
          - Thermalisation efficiency uses the *updated* gamma (= vej after step).
          - drdt uses old beta; dgamma uses updated gamma in denominator.
        """
        gamma, r, V, E, prev_dgamma, prev_drdt, prev_dV, prev_dE = carry
        t_i, mag_i, dt_i = xs

        # ── 1. Old beta and doppler (from pre-update gamma) ──────────────────
        # Clamp to 1e-16 (not zero) so sqrt gradient stays finite near γ=1.
        beta_old   = jnp.sqrt(jnp.maximum(one - one / gamma ** 2, jnp.array(1e-16, dtype=fp)))
        dop_old    = one / (gamma * jnp.maximum(one - beta_old, tiny))

        # ── 2. Euler update of state ──────────────────────────────────────────
        gamma = gamma + prev_dgamma * dt_i
        r     = r     + prev_drdt   * dt_i
        V     = V     + prev_dV     * dt_i
        E     = E     + prev_dE     * dt_i

        # Safety clamps (keep state physical)
        gamma = jnp.maximum(gamma, one + tiny)
        r     = jnp.maximum(r,     jnp.array(1e8,  dtype=fp))
        V     = jnp.maximum(V,     jnp.array(1e24, dtype=fp))
        E     = jnp.maximum(E,     jnp.array(1e30, dtype=fp))

        # ── 3. Physics on updated state (beta/doppler still old) ─────────────
        swept_mass        = (jnp.array(4.0 / 3.0, dtype=fp)) * pi * r ** 3 * n_ism * m_p
        comoving_pressure = E / (jnp.array(3.0, dtype=fp) * V)
        t_comov           = dop_old * t_i                       # comoving time, s

        # Ni/Co decay luminosity (comoving frame)
        L_ni = nickel_msun * (
            ni56_lum * jnp.exp(jnp.maximum(-t_comov / ni56_life, jnp.array(-700.0, dtype=fp)))
            + co56_lum * jnp.exp(jnp.maximum(-t_comov / co56_life, jnp.array(-700.0, dtype=fp)))
        )

        # Optical depth
        tau = kappa * (mej_g / V) * (r / gamma)

        # Emitted luminosity and temperature (branched on optical depth)
        r_ov_g    = r / gamma
        tau_safe  = jnp.maximum(tau, tiny)
        L_thin    = E * c / r_ov_g
        L_thick   = E * c / (tau_safe * r_ov_g)
        L_emit    = jnp.where(tau <= one, L_thin, L_thick)

        T_thin    = (E / (a_rad * V)) ** jnp.array(0.25, dtype=fp)
        T_thick   = (E / (a_rad * V * tau_safe)) ** jnp.array(0.25, dtype=fp)
        T_comov   = jnp.where(tau <= one, T_thin, T_thick)

        L_obs     = L_emit * dop_old ** 2          # Doppler boost to observer frame

        # Thermalisation efficiency (uses *updated* gamma → new vej)
        vej_new   = jnp.sqrt(jnp.maximum(one - one / gamma ** 2, zero)) * c
        vej_safe  = jnp.maximum(vej_new, jnp.array(1e5, dtype=fp))
        prefactor = (jnp.array(3.0, dtype=fp) * kappa_gamma * mej_g
                     / (jnp.array(4.0, dtype=fp) * pi * vej_safe ** 2))
        t_safe    = jnp.maximum(t_i, one)
        eta_th    = one - jnp.exp(-prefactor / t_safe ** 2)

        # ── 4. New derivatives ────────────────────────────────────────────────
        beta_safe = jnp.maximum(beta_old, tiny)
        one_mb    = jnp.maximum(one - beta_old, tiny)
        drdt      = beta_safe * c / one_mb

        dM_sw_dt   = jnp.array(4.0, dtype=fp) * pi * r ** 2 * n_ism * m_p * drdt
        dvdt_c     = jnp.array(4.0, dtype=fp) * pi * r ** 2 * beta_safe * c   # comoving dV/dt

        dE_tot_dt  = eta_th * mag_i + dop_old ** 2 * (L_ni - L_emit)
        dE_com_dt  = (eta_th * dop_old ** (-2) * mag_i
                      + L_ni - L_emit
                      - comoving_pressure * dvdt_c)
        dV_com_dt  = dvdt_c * dop_old
        dE_int_dt  = dE_com_dt * dop_old

        denom      = (mej_g * c ** 2 + E
                      + jnp.array(2.0, dtype=fp) * gamma * swept_mass * c ** 2)
        denom      = jnp.maximum(jnp.abs(denom), jnp.array(1e30, dtype=fp))
        dgamma_dt  = ((dE_tot_dt
                       - gamma * dop_old * dE_com_dt
                       - (gamma ** 2 - one) * c ** 2 * dM_sw_dt)
                      / denom)

        new_carry  = (gamma, r, V, E, dgamma_dt, drdt, dV_com_dt, dE_int_dt)
        outputs    = (L_obs, gamma, r, T_comov, dop_old, tau, eta_th)
        return new_carry, outputs

    return scan_step


def _run_magnetar_ode(
    time,
    mej,
    E_sn,
    kappa,
    l0,
    tau_sd,
    nn,
    kappa_gamma,
    f_nickel=0.0,
    n_grid=2000,
):
    """Run the ODE scan and return (log10_lbol, vej_kms) at the requested times.

    Both arrays have the same shape and dtype as *time*.  This function is the
    shared engine used by the single-output and full-output wrappers below.

    Parameters
    ----------
    time : array_like, days (source frame)
    Returns
    -------
    log10_lbol : ndarray  — log10(L_bol) in erg/s
    vej_kms    : ndarray  — ejecta velocity in km/s (from ODE Lorentz factor)
    """
    fp_out = jnp.asarray(time).dtype
    fp     = jnp.float64

    # Dense log-spaced time grid (source frame, seconds)
    time_s = jnp.geomspace(
        jnp.array(1.0, dtype=fp),
        jnp.array(1.0e8, dtype=fp),
        n_grid,
    )

    # Magnetar spin-down luminosity: L = l0 · (1 + t/τ)^((1+n)/(1-n))
    l0_f     = jnp.asarray(l0,     dtype=fp)
    tau_f    = jnp.asarray(tau_sd, dtype=fp)
    nn_f     = jnp.asarray(nn,     dtype=fp)
    exp_mag  = (jnp.array(1.0, dtype=fp) + nn_f) / (jnp.array(1.0, dtype=fp) - nn_f)
    mag_lum  = l0_f * (jnp.array(1.0, dtype=fp) + time_s / tau_f) ** exp_mag

    # Initial conditions
    mej_g  = jnp.asarray(mej,   dtype=fp) * jnp.array(_MSUN, dtype=fp)
    E_sn_f = jnp.asarray(E_sn,  dtype=fp)
    c_f    = jnp.array(_C,      dtype=fp)
    beta0  = jnp.sqrt(E_sn_f / (jnp.array(0.5, dtype=fp) * mej_g)) / c_f
    beta0  = jnp.minimum(beta0, jnp.array(0.9999, dtype=fp))
    gamma0 = jnp.array(1.0, dtype=fp) / jnp.sqrt(jnp.array(1.0, dtype=fp) - beta0 ** 2)
    E0     = jnp.array(0.5, dtype=fp) * beta0 ** 2 * mej_g * c_f ** 2
    r0     = jnp.array(_R0_CM, dtype=fp)
    V0     = jnp.array(4.0 / 3.0, dtype=fp) * jnp.array(jnp.pi, dtype=fp) * r0 ** 3

    # Time deltas: dt[0]=0 so first Euler step is a no-op
    dt = jnp.concatenate([
        jnp.zeros(1, dtype=fp),
        jnp.diff(time_s),
    ])

    # Build scan function (closes over physical parameters)
    kappa_f   = jnp.asarray(kappa,       dtype=fp)
    kg_f      = jnp.asarray(kappa_gamma, dtype=fp)
    fni_f     = jnp.asarray(f_nickel,    dtype=fp)
    scan_step = _make_scan_step(mej_g, kappa_f, kg_f, fni_f, fp)

    # Run ODE via lax.scan
    carry0 = (
        gamma0, r0, V0, E0,
        jnp.array(0.0, dtype=fp),   # dgamma_dt
        jnp.array(0.0, dtype=fp),   # drdt
        jnp.array(0.0, dtype=fp),   # dV_dt
        jnp.array(0.0, dtype=fp),   # dE_dt
    )
    _, (lbol, gamma_grid, _, _, _, _, _) = lax.scan(
        scan_step, carry0, (time_s, mag_lum, dt)
    )

    # Interpolate to requested times
    time_s_req = jnp.asarray(time, dtype=fp) * jnp.array(_DAY, dtype=fp)

    lbol_safe  = jnp.maximum(lbol, jnp.array(1e25, dtype=fp))
    log10_out  = jnp.interp(time_s_req, time_s, jnp.log10(lbol_safe))

    # vej from Lorentz factor: β = sqrt(1 - 1/γ²), vej = β·c [km/s]
    beta_grid   = jnp.sqrt(
        jnp.maximum(
            jnp.array(1.0, dtype=fp) - jnp.array(1.0, dtype=fp) / gamma_grid ** 2,
            jnp.array(0.0, dtype=fp),
        )
    )
    vej_kms_grid = beta_grid * c_f / jnp.array(1e5, dtype=fp)
    vej_kms_out  = jnp.interp(time_s_req, time_s, vej_kms_grid)

    return log10_out.astype(fp_out), vej_kms_out.astype(fp_out)


def _magnetar_impl(
    time,
    mej,
    E_sn,
    kappa,
    l0,
    tau_sd,
    nn,
    kappa_gamma,
    f_nickel=0.0,
    n_grid=2000,
):
    """Core ODE — returns log10(L_bol).  No JIT, safe for jax.vmap."""
    log10_out, _ = _run_magnetar_ode(
        time, mej, E_sn, kappa, l0, tau_sd, nn, kappa_gamma, f_nickel, n_grid
    )
    return log10_out


@citation_wrapper(
    'https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.4949S/abstract,'
    'https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.6455O/abstract'
)
@partial(jit, static_argnames=['solver', 'n_grid'])
def general_magnetar_driven_supernova_bolometric(
    time,
    mej,
    E_sn,
    kappa,
    l0,
    tau_sd,
    nn,
    kappa_gamma,
    f_nickel=0.0,
    solver='diffrax',
    rtol=1e-5,
    atol=1e-8,
    n_grid=2000,
):
    """Bolometric light curve of a general magnetar-driven supernova.

    Translated from redback's ``general_magnetar_driven_supernova_bolometric``
    (Sarin et al. 2022) into JAX, enabling JIT compilation and gradients.

    Parameters
    ----------
    time : array_like, days
        Source-frame times at which to evaluate the model.
    mej : float, M_sun
        Ejecta mass.
    E_sn : float, erg
        Explosion kinetic energy.
    kappa : float, cm²/g
        Optical opacity.
    l0 : float, erg/s
        Initial magnetar spin-down luminosity.
    tau_sd : float, s
        Magnetar spin-down timescale.
    nn : float
        Magnetar braking index (3 = dipole).
    kappa_gamma : float, cm²/g
        Gamma-ray opacity for thermalisation efficiency.
    f_nickel : float, optional
        Ni-56 mass fraction of ejecta.  Default 0.
    solver : str, optional
        ODE backend.  ``'diffrax'`` (default) uses the adaptive Tsit5
        integrator (~380× faster than redback).  ``'euler'`` uses the
        fixed-step Euler scan (n_grid points).
    rtol, atol : float, optional
        Tolerances for the diffrax solver (ignored when solver='euler').
    n_grid : int, optional
        Grid points for the Euler scan (ignored when solver='diffrax').
        Static — each unique value triggers a separate compilation.

    Returns
    -------
    jnp.ndarray
        ``log10(L_bol)`` in erg/s evaluated at each element of *time*.
    """
    if solver == 'diffrax':
        log10_lbol, _ = _run_magnetar_ode_diffrax(
            time, mej, E_sn, kappa, l0, tau_sd, nn, kappa_gamma, f_nickel, rtol, atol
        )
    elif solver == 'euler':
        log10_lbol, _ = _run_magnetar_ode(
            time, mej, E_sn, kappa, l0, tau_sd, nn, kappa_gamma, f_nickel, n_grid
        )
    else:
        raise ValueError(f"solver must be 'diffrax' or 'euler', got {solver!r}")
    return log10_lbol


@citation_wrapper(
    'https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.4949S/abstract,'
    'https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.6455O/abstract'
)
@partial(jit, static_argnames=['solver', 'n_grid'])
def general_magnetar_driven_supernova_bolometric_and_vej(
    time,
    mej,
    E_sn,
    kappa,
    l0,
    tau_sd,
    nn,
    kappa_gamma,
    f_nickel=0.0,
    solver='diffrax',
    rtol=1e-5,
    atol=1e-8,
    n_grid=2000,
):
    """Bolometric light curve + time-varying ejecta velocity.

    Returns the same ODE outputs as
    ``general_magnetar_driven_supernova_bolometric`` but also exposes the
    ejecta velocity v_ej(t) = β(t)·c derived from the ODE Lorentz factor.
    Pass ``vej_kms`` to ``compute_temperature_floor_log10`` for a
    time-varying photosphere (consistent with the redback reference model).

    Parameters
    ----------
    (same as ``general_magnetar_driven_supernova_bolometric``)

    Returns
    -------
    log10_lbol : jnp.ndarray  — log10(L_bol) in erg/s, shape (T,)
    vej_kms    : jnp.ndarray  — ejecta velocity in km/s, shape (T,)
                 Both arrays have the same dtype as *time*.
    """
    if solver == 'diffrax':
        return _run_magnetar_ode_diffrax(
            time, mej, E_sn, kappa, l0, tau_sd, nn, kappa_gamma, f_nickel, rtol, atol
        )
    elif solver == 'euler':
        return _run_magnetar_ode(
            time, mej, E_sn, kappa, l0, tau_sd, nn, kappa_gamma, f_nickel, n_grid
        )
    else:
        raise ValueError(f"solver must be 'diffrax' or 'euler', got {solver!r}")


@partial(jit, static_argnames=['n_grid'])
def general_magnetar_driven_supernova_bolometric_batched(
    time,
    mej,
    E_sn,
    kappa,
    l0,
    tau_sd,
    nn,
    kappa_gamma,
    f_nickel=None,
    n_grid=2000,
):
    """Evaluate B parameter samples simultaneously via ``jax.vmap``.

    Runs ``general_magnetar_driven_supernova_bolometric`` on B independent
    parameter vectors in a single JIT-compiled kernel.  On GPU this maps each
    ODE trajectory to a separate set of CUDA cores; on CPU it benefits from
    better cache reuse compared to sequential calls.

    Parameters
    ----------
    time : array_like, shape (T,)
        Shared source-frame times in days (broadcast over all B samples).
    mej, E_sn, kappa, l0, tau_sd, nn, kappa_gamma : array_like, shape (B,)
        Physical parameters — one scalar value per sample.
    f_nickel : array_like, shape (B,), optional
        Ni-56 mass fractions.  Defaults to zeros for all samples if None.
    n_grid : int, optional
        ODE grid points (static — triggers recompile on change).  Default 2000.

    Returns
    -------
    jnp.ndarray, shape (B, T)
        ``log10(L_bol)`` in erg/s — row i corresponds to sample i.
    """
    mej_arr = jnp.asarray(mej, dtype=jnp.float64)
    f_ni    = (jnp.zeros_like(mej_arr)
               if f_nickel is None
               else jnp.asarray(f_nickel, dtype=jnp.float64))
    return jax.vmap(
        lambda m, e, k, l, t, n, kg, fn: _magnetar_impl(
            time, m, e, k, l, t, n, kg, fn, n_grid
        )
    )(
        mej_arr,
        jnp.asarray(E_sn,        dtype=jnp.float64),
        jnp.asarray(kappa,       dtype=jnp.float64),
        jnp.asarray(l0,          dtype=jnp.float64),
        jnp.asarray(tau_sd,      dtype=jnp.float64),
        jnp.asarray(nn,          dtype=jnp.float64),
        jnp.asarray(kappa_gamma, dtype=jnp.float64),
        f_ni,
    )


def _magnetar_vf_diffrax(t, y, args):
    """diffrax-compatible vector field for the magnetar-driven ejecta ODE.

    dy/dt = f(t, y, args)  where  y = [gamma, r, V, E].

    Identical physics to ``_make_scan_step`` but expressed in standard ODE
    form (Doppler factor evaluated at the *current* state rather than the
    previous-step state — the difference is O(dt) and vanishes for the small
    steps used by the adaptive solver).
    """
    gamma, r, V, E = y[0], y[1], y[2], y[3]
    mej_g, kappa, kappa_gamma, f_nickel, l0, tau_sd, nn = args

    fp = jnp.float64

    c       = jnp.array(_C,    dtype=fp)
    m_p     = jnp.array(_MP,   dtype=fp)
    a_rad   = jnp.array(_ARAD, dtype=fp)
    n_ism   = jnp.array(_N_ISM, dtype=fp)
    pi      = jnp.array(jnp.pi, dtype=fp)
    one     = jnp.array(1.0,   dtype=fp)
    zero    = jnp.array(0.0,   dtype=fp)
    tiny    = jnp.array(1e-15, dtype=fp)
    msun    = jnp.array(_MSUN, dtype=fp)

    ni56_lum  = jnp.array(_NI56_LUM,  dtype=fp)
    co56_lum  = jnp.array(_CO56_LUM,  dtype=fp)
    ni56_life = jnp.array(_NI56_LIFE, dtype=fp)
    co56_life = jnp.array(_CO56_LIFE, dtype=fp)

    nickel_msun = f_nickel * mej_g / msun

    # Safety clamps on state
    gamma = jnp.maximum(gamma, one + tiny)
    r     = jnp.maximum(r,     jnp.array(1e8,  dtype=fp))
    V     = jnp.maximum(V,     jnp.array(1e24, dtype=fp))
    E     = jnp.maximum(E,     jnp.array(1e30, dtype=fp))

    # Current beta and Doppler factor — clamp to 1e-16 so sqrt gradient stays
    # finite when γ ≈ 1 (non-relativistic limit).
    beta_sq = jnp.maximum(one - one / gamma ** 2, jnp.array(1e-16, dtype=fp))
    beta    = jnp.sqrt(beta_sq)
    dop     = one / (gamma * jnp.maximum(one - beta, tiny))

    # Magnetar spin-down luminosity (evaluated analytically at time t)
    exp_mag = (one + nn) / (one - nn)
    mag_lum = l0 * (one + t / tau_sd) ** exp_mag

    # Comoving quantities
    swept_mass        = jnp.array(4.0/3.0, dtype=fp) * pi * r**3 * n_ism * m_p
    comoving_pressure = E / (jnp.array(3.0, dtype=fp) * V)
    t_comov           = dop * t

    # Ni/Co decay luminosity (comoving frame)
    L_ni = nickel_msun * (
        ni56_lum * jnp.exp(jnp.maximum(-t_comov / ni56_life, jnp.array(-700.0, dtype=fp)))
        + co56_lum * jnp.exp(jnp.maximum(-t_comov / co56_life, jnp.array(-700.0, dtype=fp)))
    )

    # Optical depth and emitted luminosity
    tau        = kappa * (mej_g / V) * (r / gamma)
    r_ov_g     = r / gamma
    tau_safe   = jnp.maximum(tau, tiny)
    L_thin     = E * c / r_ov_g
    L_thick    = E * c / (tau_safe * r_ov_g)
    L_emit     = jnp.where(tau <= one, L_thin, L_thick)

    # Thermalisation efficiency
    vej_new   = beta * c
    vej_safe  = jnp.maximum(vej_new, jnp.array(1e5, dtype=fp))
    prefactor = (jnp.array(3.0, dtype=fp) * kappa_gamma * mej_g
                 / (jnp.array(4.0, dtype=fp) * pi * vej_safe**2))
    t_safe    = jnp.maximum(t, one)
    eta_th    = one - jnp.exp(-prefactor / t_safe**2)

    # Derivatives
    beta_safe = jnp.maximum(beta, tiny)
    one_mb    = jnp.maximum(one - beta, tiny)
    drdt      = beta_safe * c / one_mb

    dM_sw_dt  = jnp.array(4.0, dtype=fp) * pi * r**2 * n_ism * m_p * drdt
    dvdt_c    = jnp.array(4.0, dtype=fp) * pi * r**2 * beta_safe * c

    dE_tot_dt = eta_th * mag_lum + dop**2 * (L_ni - L_emit)
    dE_com_dt = (eta_th * dop**(-2) * mag_lum
                 + L_ni - L_emit
                 - comoving_pressure * dvdt_c)
    dV_dt     = dvdt_c * dop
    dE_dt     = dE_com_dt * dop

    denom     = (mej_g * c**2 + E
                 + jnp.array(2.0, dtype=fp) * gamma * swept_mass * c**2)
    denom     = jnp.maximum(jnp.abs(denom), jnp.array(1e30, dtype=fp))
    dgamma_dt = ((dE_tot_dt
                  - gamma * dop * dE_com_dt
                  - (gamma**2 - one) * c**2 * dM_sw_dt)
                 / denom)

    return jnp.array([dgamma_dt, drdt, dV_dt, dE_dt], dtype=fp)


def _run_magnetar_ode_diffrax(
    time,
    mej,
    E_sn,
    kappa,
    l0,
    tau_sd,
    nn,
    kappa_gamma,
    f_nickel=0.0,
    rtol=1e-5,
    atol=1e-8,
):
    """Run the ejecta ODE with diffrax Tsit5 (adaptive step-size).

    Returns ``(log10_lbol, vej_kms)`` at the requested times — identical
    output contract to ``_run_magnetar_ode`` but uses an adaptive
    4th/5th-order RK solver instead of fixed-step Euler.

    Parameters
    ----------
    time : array_like, source-frame days
    rtol, atol : float
        Relative and absolute tolerances for the PID step-size controller.
    """
    from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController

    fp     = jnp.float64
    fp_out = jnp.asarray(time).dtype

    time_s_req = jnp.asarray(time, dtype=fp) * jnp.array(_DAY, dtype=fp)

    # Sort times for diffrax SaveAt (requires ascending order)
    sort_idx    = jnp.argsort(time_s_req)
    unsort_idx  = jnp.argsort(sort_idx)
    time_s_sort = time_s_req[sort_idx]

    # Initial conditions (same as _run_magnetar_ode)
    mej_g  = jnp.asarray(mej,   dtype=fp) * jnp.array(_MSUN, dtype=fp)
    E_sn_f = jnp.asarray(E_sn,  dtype=fp)
    c_f    = jnp.array(_C,      dtype=fp)
    beta0  = jnp.sqrt(E_sn_f / (jnp.array(0.5, dtype=fp) * mej_g)) / c_f
    beta0  = jnp.minimum(beta0, jnp.array(0.9999, dtype=fp))
    gamma0 = jnp.array(1.0, dtype=fp) / jnp.sqrt(jnp.array(1.0, dtype=fp) - beta0**2)
    E0     = jnp.array(0.5, dtype=fp) * beta0**2 * mej_g * c_f**2
    r0     = jnp.array(_R0_CM, dtype=fp)
    V0     = jnp.array(4.0/3.0, dtype=fp) * jnp.array(jnp.pi, dtype=fp) * r0**3
    y0     = jnp.array([gamma0, r0, V0, E0], dtype=fp)

    args = (
        mej_g,
        jnp.asarray(kappa,       dtype=fp),
        jnp.asarray(kappa_gamma, dtype=fp),
        jnp.asarray(f_nickel,    dtype=fp),
        jnp.asarray(l0,          dtype=fp),
        jnp.asarray(tau_sd,      dtype=fp),
        jnp.asarray(nn,          dtype=fp),
    )

    # Set t0 dynamically so SaveAt times are never before t0 (diffrax raises
    # ValueError if any save time falls outside [t0, t1]).
    t0 = jnp.minimum(jnp.array(1.0, dtype=fp),
                     time_s_sort[0] * jnp.array(0.9, dtype=fp))
    t1 = jnp.maximum(
        time_s_sort[-1] * jnp.array(1.001, dtype=fp),
        t0 + jnp.array(1.0, dtype=fp),
    )

    solution = diffeqsolve(
        ODETerm(_magnetar_vf_diffrax),
        Tsit5(),
        t0=t0,
        t1=t1,
        dt0=jnp.array(10.0, dtype=fp),
        y0=y0,
        args=args,
        saveat=SaveAt(ts=time_s_sort),
        stepsize_controller=PIDController(rtol=rtol, atol=atol),
        max_steps=262144,
        throw=False,
    )

    # Extract saved state (shape: N_save × 4)
    gamma_out = solution.ys[:, 0]
    r_out     = solution.ys[:, 1]
    V_out     = solution.ys[:, 2]
    E_out     = solution.ys[:, 3]

    # Derive L_obs and v_ej from saved state
    beta_out  = jnp.sqrt(
        jnp.maximum(jnp.array(1.0, dtype=fp) - jnp.array(1.0, dtype=fp) / gamma_out**2,
                    jnp.array(0.0, dtype=fp))
    )
    dop_out   = jnp.array(1.0, dtype=fp) / (
        gamma_out * jnp.maximum(jnp.array(1.0, dtype=fp) - beta_out, jnp.array(1e-15, dtype=fp))
    )
    tau_out   = (jnp.asarray(kappa, dtype=fp)
                 * (mej_g / V_out)
                 * (r_out / gamma_out))
    r_ov_g    = r_out / gamma_out
    tau_safe  = jnp.maximum(tau_out, jnp.array(1e-15, dtype=fp))
    L_thin    = E_out * c_f / r_ov_g
    L_thick   = E_out * c_f / (tau_safe * r_ov_g)
    L_emit    = jnp.where(tau_out <= jnp.array(1.0, dtype=fp), L_thin, L_thick)
    L_obs     = L_emit * dop_out**2

    lbol_safe     = jnp.maximum(L_obs, jnp.array(1e25, dtype=fp))
    log10_lbol    = jnp.log10(lbol_safe)
    vej_kms_out   = beta_out * c_f / jnp.array(1e5, dtype=fp)

    # Unsort to restore original time ordering
    log10_lbol  = log10_lbol[unsort_idx]
    vej_kms_out = vej_kms_out[unsort_idx]

    return log10_lbol.astype(fp_out), vej_kms_out.astype(fp_out)


@citation_wrapper(
    'https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.4949S/abstract,'
    'https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.6455O/abstract'
)
@jit
def general_magnetar_driven_supernova_bolometric_diffrax(
    time,
    mej,
    E_sn,
    kappa,
    l0,
    tau_sd,
    nn,
    kappa_gamma,
    f_nickel=0.0,
    rtol=1e-5,
    atol=1e-8,
):
    """Bolometric light curve using the diffrax Tsit5 adaptive ODE solver.

    Functionally equivalent to ``general_magnetar_driven_supernova_bolometric``
    (with ``solver='diffrax'``) but exposed as a standalone function for
    backward compatibility.  Typically 3–10× faster than the Euler scan at
    the same accuracy once compiled.

    Parameters
    ----------
    time : array_like, source-frame days
    (other params same as general_magnetar_driven_supernova_bolometric)
    rtol : float, default 1e-5
    atol : float, default 1e-8

    Returns
    -------
    log10_lbol : jnp.ndarray  — log10(L_bol) in erg/s
    """
    log10_lbol, _ = _run_magnetar_ode_diffrax(
        time, mej, E_sn, kappa, l0, tau_sd, nn, kappa_gamma, f_nickel, rtol, atol
    )
    return log10_lbol


@citation_wrapper(
    'https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.4949S/abstract,'
    'https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.6455O/abstract,'
    'https://ui.adsabs.harvard.edu/abs/2017ApJ...850...55N/abstract,'
    'https://ui.adsabs.harvard.edu/abs/2017ApJ...851L..21V/abstract'
)
@partial(jit, static_argnames=['solver', 'n_grid'])
def general_magnetar_driven_supernova(
    time,
    frequency,
    mej,
    E_sn,
    kappa,
    l0,
    tau_sd,
    nn,
    kappa_gamma,
    temperature_floor,
    luminosity_distance,
    redshift,
    cutoff_wavelength=3000.0,
    f_nickel=0.0,
    solver='diffrax',
    rtol=1e-5,
    atol=1e-8,
    n_grid=2000,
    alpha_uv=1.0,
):
    """Full multiband general magnetar-driven supernova model.

    1:1 JAX translation of redback's ``general_magnetar_driven_supernova``
    with ``output_format='flux_density'``, using the ``TemperatureFloor``
    photosphere and ``CutoffBlackbody`` SED.

    Pipeline
    --------
    1. K-correction:  freq_src = freq_obs·(1+z),  t_src = t_obs/(1+z)
    2. ODE:           (log10_lbol, vej_kms) from chosen backend
    3. Photosphere:   (T_ph, log10_r_ph) from TemperatureFloor
    4. SED:           F_mjy from CutoffBlackbody
    5. Return:        F_mjy · (1+z)

    Parameters
    ----------
    time : (N,) observer-frame days
    frequency : (N,) observer-frame Hz
    mej : float, M_sun
    E_sn : float, erg
    kappa : float, cm²/g
    l0 : float, erg/s
    tau_sd : float, s
    nn : float
    kappa_gamma : float, cm²/g
    temperature_floor : float, K
    luminosity_distance : float, cm
    redshift : float
    cutoff_wavelength : float, Å  (default 3000)
    f_nickel : float  (default 0)
    solver : str, optional
        ``'diffrax'`` (default, adaptive Tsit5) or ``'euler'`` (fixed-step).
    rtol, atol : float, optional
        diffrax tolerances (ignored when solver='euler').
    n_grid : int, optional
        Euler grid points (ignored when solver='diffrax').
    alpha_uv : float, default 1.0
        UV power-law suppression index for CutoffBlackbody SED.
        λ < λ_c contributes Planck × (λ/λ_c)^alpha_uv.  Valid range [0, 4).

    Returns
    -------
    F_mjy : (N,) mJy  — observer-frame flux density
    """
    fp = jnp.float64

    # 1. K-correction
    freq  = jnp.asarray(frequency, dtype=fp)
    t_obs = jnp.asarray(time,      dtype=fp)
    z     = jnp.asarray(redshift,  dtype=fp)

    freq_src = freq  * (jnp.array(1.0, dtype=fp) + z)
    time_src = t_obs / (jnp.array(1.0, dtype=fp) + z)

    # 2. ODE
    if solver == 'diffrax':
        log10_lbol, vej_kms = _run_magnetar_ode_diffrax(
            time_src, mej, E_sn, kappa, l0, tau_sd, nn, kappa_gamma, f_nickel, rtol, atol
        )
    elif solver == 'euler':
        log10_lbol, vej_kms = _run_magnetar_ode(
            time_src, mej, E_sn, kappa, l0, tau_sd, nn, kappa_gamma, f_nickel, n_grid
        )
    else:
        raise ValueError(f"solver must be 'diffrax' or 'euler', got {solver!r}")

    # 3. TemperatureFloor photosphere
    T_ph, log10_r_ph = compute_temperature_floor_log10(
        time_src, log10_lbol, vej_kms, temperature_floor
    )
    r_ph  = jnp.power(jnp.array(10.0, dtype=fp), log10_r_ph)
    lbol  = jnp.power(jnp.array(10.0, dtype=fp), log10_lbol)

    # 4. CutoffBlackbody SED
    F_mjy = cutoff_blackbody_flux_density(
        freq_src, lbol, T_ph, r_ph,
        jnp.asarray(luminosity_distance, dtype=fp),
        cutoff_wavelength,
        alpha_uv,
    )

    # 5. Observer-frame correction
    return F_mjy * (jnp.array(1.0, dtype=fp) + z)


@citation_wrapper(
    'https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.4949S/abstract,'
    'https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.6455O/abstract,'
    'https://ui.adsabs.harvard.edu/abs/2017ApJ...850...55N/abstract,'
    'https://ui.adsabs.harvard.edu/abs/2017ApJ...851L..21V/abstract'
)
@jit
def general_magnetar_driven_supernova_diffrax(
    time,
    frequency,
    mej,
    E_sn,
    kappa,
    l0,
    tau_sd,
    nn,
    kappa_gamma,
    temperature_floor,
    luminosity_distance,
    redshift,
    cutoff_wavelength=3000.0,
    f_nickel=0.0,
    rtol=1e-5,
    atol=1e-8,
    alpha_uv=1.0,
):
    """Full multiband model using the diffrax Tsit5 adaptive ODE solver.

    Standalone backward-compatible variant of ``general_magnetar_driven_supernova``
    that always uses the diffrax backend.  Equivalent to calling
    ``general_magnetar_driven_supernova(..., solver='diffrax')``.

    Parameters
    ----------
    (all params same as general_magnetar_driven_supernova except n_grid)
    rtol : float, default 1e-5
    atol : float, default 1e-8
    alpha_uv : float, default 1.0

    Returns
    -------
    F_mjy : (N,) mJy
    """
    fp = jnp.float64

    freq  = jnp.asarray(frequency, dtype=fp)
    t_obs = jnp.asarray(time,      dtype=fp)
    z     = jnp.asarray(redshift,  dtype=fp)

    freq_src = freq  * (jnp.array(1.0, dtype=fp) + z)
    time_src = t_obs / (jnp.array(1.0, dtype=fp) + z)

    log10_lbol, vej_kms = _run_magnetar_ode_diffrax(
        time_src, mej, E_sn, kappa, l0, tau_sd, nn, kappa_gamma, f_nickel, rtol, atol
    )

    T_ph, log10_r_ph = compute_temperature_floor_log10(
        time_src, log10_lbol, vej_kms, temperature_floor
    )
    r_ph = jnp.power(jnp.array(10.0, dtype=fp), log10_r_ph)
    lbol = jnp.power(jnp.array(10.0, dtype=fp), log10_lbol)

    F_mjy = cutoff_blackbody_flux_density(
        freq_src, lbol, T_ph, r_ph,
        jnp.asarray(luminosity_distance, dtype=fp),
        cutoff_wavelength,
        alpha_uv,
    )

    return F_mjy * (jnp.array(1.0, dtype=fp) + z)


@jit
def general_magnetar_driven_supernova_bolometric_and_vej_diffrax(
    time,
    mej,
    E_sn,
    kappa,
    l0,
    tau_sd,
    nn,
    kappa_gamma,
    f_nickel=0.0,
    rtol=1e-5,
    atol=1e-8,
):
    """Return (log10_lbol, vej_kms) from the diffrax Tsit5 adaptive ODE.

    Counterpart to ``general_magnetar_driven_supernova_bolometric_and_vej``
    using the diffrax backend.  Required internally by
    ``general_magnetar_supernova_spectra_diffrax`` (the spectra factory needs
    vej derived from the ODE, not as a free parameter).

    Parameters
    ----------
    time : array_like, source-frame days
    (other params same as general_magnetar_driven_supernova_bolometric_diffrax)

    Returns
    -------
    log10_lbol : jnp.ndarray  — log10(L_bol) in erg/s
    vej_kms : jnp.ndarray  — ejecta velocity in km/s
    """
    return _run_magnetar_ode_diffrax(
        time, mej, E_sn, kappa, l0, tau_sd, nn, kappa_gamma, f_nickel, rtol, atol
    )
