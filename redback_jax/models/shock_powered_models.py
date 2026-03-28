"""
JAX-friendly shock-powered transient models.

References:
    Piro 2021 (shock cooling): https://ui.adsabs.harvard.edu/abs/2021ApJ...909..209P/abstract
    Piro & Kollmeier 2018 (shocked cocoon): https://ui.adsabs.harvard.edu/abs/2018ApJ...855..103P/abstract
"""

import math as _math

import jax.numpy as jnp
from jax import jit

from redback_jax.utils.citation_wrapper import citation_wrapper

# Physical constants as Python floats (avoids astropy float64 promotion)
_SOLAR_MASS     = 1.989e33   # g
_SPEED_OF_LIGHT = 2.998e10   # cm/s
_KM_CGS         = 1.0e5      # cm/km
_DAY_TO_S       = 86400.0    # s/day

# Diffusion constant for shocked cocoon (Python float, no JAX promotion):
# diff_const = Msun / (4*pi * c * km_cgs)
_DIFF_CONST = _SOLAR_MASS / (4.0 * _math.pi * _SPEED_OF_LIGHT * _KM_CGS)
_LOG10_DIFF_CONST = _math.log10(_DIFF_CONST)
_LOG10_SOLAR_MASS = _math.log10(_SOLAR_MASS)
_LOG10_KM_CGS     = _math.log10(_KM_CGS)
_LOG10_DAY_TO_S   = _math.log10(_DAY_TO_S)
_LOG10_C_CGS      = _math.log10(_SPEED_OF_LIGHT)
_LOG10_E          = _math.log10(_math.e)


# ---------------------------------------------------------------------------
# Shock cooling (Piro 2021)
# ---------------------------------------------------------------------------

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2021ApJ...909..209P/abstract')
@jit
def shock_cooling_bolometric(time, log10_mass, log10_radius, log10_energy):
    """
    Bolometric shock-cooling light curve following Piro (2021), n=10, delta=1.1.

    All large quantities (mass, radius, energy) are passed as log10 to stay
    float32-safe throughout; all intermediate computations stay in log10 space.

    :param time: source-frame time in days
    :param log10_mass: log10 envelope mass in solar masses
    :param log10_radius: log10 envelope radius in cm
    :param log10_energy: log10 explosion energy in erg
    :return: log10 of bolometric luminosity in erg/s
    """
    fp = time.dtype
    n     = 10.0
    delta = 1.1
    kappa = 0.2  # cm^2/g

    # kk_pow is a small Python float — no overflow risk
    kk_pow = (n - 3.0) * (3.0 - delta) / (4.0 * _math.pi * (n - delta))

    # log10(mass in grams)
    log10_mass_g = log10_mass + jnp.array(_LOG10_SOLAR_MASS, dtype=fp)

    # log10(vt):  vt^2 = coeff * E/m  =>  log10(vt) = 0.5*(log10(coeff) + log10_E - log10_m)
    _vt_coeff = (n - 5.0) * (5.0 - delta) / ((n - 3.0) * (3.0 - delta)) * 2.0
    log10_vt = 0.5 * (jnp.array(_math.log10(_vt_coeff), dtype=fp) + log10_energy - log10_mass_g)

    # log10(td in days):  td^2 = coeff * m / (vt * c)
    _td_const = _math.log10(3.0 * kappa * kk_pow / ((n - 1.0) * _SPEED_OF_LIGHT)) - _LOG10_DAY_TO_S
    log10_td = 0.5 * (jnp.array(_td_const, dtype=fp) + log10_mass_g - log10_vt)
    td = jnp.power(jnp.array(10.0, dtype=fp), log10_td)  # days (< 1e5 days, safe float32)

    t = jnp.maximum(time, jnp.array(1.0 / _DAY_TO_S, dtype=fp))

    # log10_prefactor = log10(pi*(n-1)/(3*(n-5)) * c / kappa) + log10_R + 2*log10_vt
    # All three terms can be >> 38 in log10, but we stay in log10
    _pf_const = _math.log10(_math.pi * abs(n - 1.0) * _SPEED_OF_LIGHT
                             / (3.0 * abs(n - 5.0) * kappa))
    log10_prefactor = (jnp.array(_pf_const, dtype=fp)
                       + log10_radius
                       + 2.0 * log10_vt)

    log10_t  = jnp.log10(jnp.maximum(t,  jnp.array(1e-30, dtype=fp)))
    log10_td_val = jnp.log10(jnp.maximum(td, jnp.array(1e-30, dtype=fp)))

    # Pre-peak: lbol = prefactor * (td/t)^(4/(n-2))
    log10_lbol_pre = log10_prefactor + jnp.array(4.0 / (n - 2.0), dtype=fp) * (log10_td_val - log10_t)

    # Post-peak: lbol = prefactor * exp(-0.5*(t^2/td^2 - 1))
    exponent = jnp.clip(-0.5 * (t ** 2 / jnp.maximum(td, jnp.array(1e-30, dtype=fp)) ** 2 - 1.0),
                         jnp.array(-87.0, dtype=fp), jnp.array(87.0, dtype=fp))
    log10_lbol_post = log10_prefactor + exponent * jnp.array(_LOG10_E, dtype=fp)

    return jnp.where(t < td, log10_lbol_pre, log10_lbol_post)


# ---------------------------------------------------------------------------
# Shocked cocoon (Piro & Kollmeier 2018) — full log10-space rewrite
# ---------------------------------------------------------------------------

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2018ApJ...855..103P/abstract')
@jit
def shocked_cocoon_bolometric(time, mej, vej, eta, tshock,
                               shocked_fraction, cos_theta_cocoon, kappa):
    """
    Bolometric light curve of a shocked jet cocoon (Piro & Kollmeier 2018).
    All large intermediate quantities computed in log10 space for float32 safety.

    :param time: source-frame time in days
    :param mej: ejecta mass in solar masses
    :param vej: ejecta velocity in units of c (speed of light)
    :param eta: ejecta density power-law slope
    :param tshock: shock time in seconds
    :param shocked_fraction: fraction of ejecta mass that is shocked
    :param cos_theta_cocoon: cosine of cocoon opening half-angle
    :param kappa: gray opacity in cm^2/g
    :return: log10 of bolometric luminosity in erg/s
    """
    fp = time.dtype

    # Cast scalar inputs to fp
    mej_f   = jnp.asarray(mej,             dtype=fp)
    vej_f   = jnp.asarray(vej,             dtype=fp)
    eta_f   = jnp.asarray(eta,             dtype=fp)
    ts_f    = jnp.asarray(tshock,          dtype=fp)
    sf_f    = jnp.asarray(shocked_fraction, dtype=fp)
    ctc_f   = jnp.asarray(cos_theta_cocoon, dtype=fp)
    kappa_f = jnp.asarray(kappa,           dtype=fp)

    # vej_kms = vej * c / km_cgs  (km/s)
    # log10(vej_kms) = log10(vej) + log10(c/km_cgs)
    _log10_c_over_km = _LOG10_C_CGS - _LOG10_KM_CGS
    log10_vej_kms = jnp.log10(jnp.maximum(vej_f, jnp.array(1e-30, dtype=fp))) + jnp.array(_log10_c_over_km, dtype=fp)

    # shocked_mass = mej * shocked_fraction  (solar masses)
    log10_shocked_mass = (jnp.log10(jnp.maximum(mej_f,  jnp.array(1e-30, dtype=fp)))
                          + jnp.log10(jnp.maximum(sf_f,  jnp.array(1e-30, dtype=fp))))

    # tau_diff^2 = _DIFF_CONST * kappa * shocked_mass / vej_kms  (days^2 after /day_s^2)
    # log10(tau_diff) = 0.5*(log10_DIFF_CONST + log10_kappa + log10_shocked_mass - log10_vej_kms - 2*log10_day_s)
    log10_tau_diff = 0.5 * (jnp.array(_LOG10_DIFF_CONST, dtype=fp)
                             + jnp.log10(jnp.maximum(kappa_f, jnp.array(1e-30, dtype=fp)))
                             + log10_shocked_mass
                             - log10_vej_kms
                             - jnp.array(2.0 * _LOG10_DAY_TO_S, dtype=fp))
    tau_diff = jnp.power(jnp.array(10.0, dtype=fp), log10_tau_diff)  # days

    # t_thin = sqrt(c_kms / vej_kms) * tau_diff
    # log10(t_thin) = 0.5*(log10_c_over_km - log10_vej_kms) + log10_tau_diff
    log10_t_thin = (0.5 * (jnp.array(_log10_c_over_km, dtype=fp) - log10_vej_kms)
                    + log10_tau_diff)
    t_thin = jnp.power(jnp.array(10.0, dtype=fp), log10_t_thin)  # days

    # rshock = tshock * c  (cm)
    # log10(rshock) = log10(tshock) + log10(c)
    log10_rshock = (jnp.log10(jnp.maximum(ts_f, jnp.array(1e-30, dtype=fp)))
                    + jnp.array(_LOG10_C_CGS, dtype=fp))

    # theta = arccos(cos_theta_cocoon)
    theta = jnp.arccos(jnp.clip(ctc_f, jnp.array(-1.0, dtype=fp), jnp.array(1.0, dtype=fp)))

    # l0 = (theta^2/2)^(1/3) * shocked_mass_g * vej_cm_s * rshock / tau_diff_s^2
    # Compute log10(l0) to avoid overflow
    # log10((theta^2/2)^(1/3)) = (2*log10(theta) - log10(2)) / 3
    log10_theta_fac = (2.0 * jnp.log10(jnp.maximum(theta, jnp.array(1e-10, dtype=fp)))
                       - jnp.array(_math.log10(2.0), dtype=fp)) / 3.0

    # shocked_mass in grams: log10 = log10_shocked_mass + log10_SOLAR_MASS
    log10_shocked_mass_g = log10_shocked_mass + jnp.array(_LOG10_SOLAR_MASS, dtype=fp)

    # vej_kms in cm/s: log10 = log10_vej_kms + log10_KM_CGS
    log10_vej_cms = log10_vej_kms + jnp.array(_LOG10_KM_CGS, dtype=fp)

    # tau_diff in seconds: log10 = log10_tau_diff + log10_DAY_TO_S
    log10_tau_diff_s = log10_tau_diff + jnp.array(_LOG10_DAY_TO_S, dtype=fp)

    log10_l0 = (log10_theta_fac
                + log10_shocked_mass_g
                + log10_vej_cms
                + log10_rshock
                - 2.0 * log10_tau_diff_s)

    # lbol = l0 * (t/tau_diff)^(-4/(eta+2)) * (1 + tanh(t_thin - t)) / 2
    # log10_lbol = log10_l0 + (-4/(eta+2)) * (log10_t - log10_tau_diff)
    #              + log10((1 + tanh(t_thin - t))/2)   -- this last factor is < 1, safe in linear
    t_safe = jnp.maximum(time, jnp.array(1e-30, dtype=fp))
    log10_t = jnp.log10(t_safe)
    power_exp = -4.0 / (eta_f + 2.0)
    log10_lbol = log10_l0 + power_exp * (log10_t - log10_tau_diff)

    # Apply thin-shell tapering in log10 space: log10(lbol * taper) = log10_lbol + log10(taper)
    # taper = (1 + tanh(t_thin - t)) / 2 is in (0, 1], safe to log10
    taper = (1.0 + jnp.tanh(t_thin - time)) / 2.0
    log10_taper = jnp.log10(jnp.maximum(taper, jnp.array(1e-30, dtype=fp)))
    return log10_lbol + log10_taper
