"""
JAX-friendly classes for supernova modeling.
"""

import math as _math
import os as _os
from collections import namedtuple

import numpy as _np
from jax import jit
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator as _RGI
from wcosmo import wcosmo

from redback_jax.utils.citation_wrapper import citation_wrapper
from redback_jax.utils.cosmology import PLANCK18_H0, PLANCK18_OM0, MPC_TO_CM
from redback_jax.conversions import calc_kcorrected_properties, lambda_to_nu
from redback_jax.interaction_processes import (
    diffusion_convert_luminosity,
    csm_diffusion_convert_luminosity,
)
from redback_jax.models.sed_features import NO_SED_FEATURES, apply_sed_feature
from redback_jax.photosphere import compute_temperature_floor_log10

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
    dense_times = jnp.linspace(0.01, time[-1] + 100.0, 1000)
    log10_dense_lbols = _nickelcobalt_log10_engine(dense_times, f_nickel, mej)
    _, log10_lum = diffusion_convert_luminosity(
        time=time, dense_times=dense_times, log10_luminosity=log10_dense_lbols,
        mej=mej, kappa=kappa, kappa_gamma=kappa_gamma, vej=vej)
    return log10_lum


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

    # Compute log10(lbol) directly in log10 space (float32-safe)
    dense_times = jnp.linspace(0.01, time[-1] + 100.0, 1000)
    log10_ni = _math.log10(6.45e43)
    log10_co = _math.log10(1.45e43)
    ni56_life = 8.8; co56_life = 111.3
    log10_a = log10_ni + (-dense_times / ni56_life) * _math.log10(_math.e)
    log10_b = log10_co + (-dense_times / co56_life) * _math.log10(_math.e)
    log10_max_ab = jnp.maximum(log10_a, log10_b)
    log10_sum    = log10_max_ab + jnp.log10(
        jnp.power(10.0, log10_a - log10_max_ab)
        + jnp.power(10.0, log10_b - log10_max_ab))
    log10_mni = jnp.log10(jnp.maximum(f_nickel * mej, 1e-30))
    log10_dense = log10_mni + log10_sum

    _, log10_lbol = diffusion_convert_luminosity(
        time=time, dense_times=dense_times, log10_luminosity=log10_dense,
        mej=mej, kappa=kappa, kappa_gamma=kappa_gamma, vej=vej)

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
