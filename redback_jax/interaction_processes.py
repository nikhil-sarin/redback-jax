# Interaction processes for transient light-curve models.
# All diffusion functions accept log10(luminosity) inputs and return
# log10(luminosity) outputs, keeping float32 intermediates O(1).

import math as _math

import jax
import jax.numpy as jnp
from jax import jit

from redback_jax.utils.citation_wrapper import citation_wrapper

# ---------------------------------------------------------------------------
# Pre-computed log10 constants (Python floats — avoids astropy float64 at trace time)
# ---------------------------------------------------------------------------
_LOG10_MSUN     = _math.log10(1.989e33)      # log10(g per Msun)
_LOG10_CCGS     = _math.log10(2.998e10)      # log10(cm/s)
_LOG10_KM_CGS   = _math.log10(1.0e5)         # log10(cm/km)
_LOG10_DAYS2SEC = _math.log10(86400.0)        # log10(s/day)
_LOG10_4PI      = _math.log10(4.0 * _math.pi)
_LOG10_2        = _math.log10(2.0)
_LOG10_3        = _math.log10(3.0)
_LOG10_13p7     = _math.log10(13.7)

# Barnes & Kasen 2016 bilinear interpolation tables (Table 1 of Barnes et al. 2016)
_BK16_MASS_GRID = jnp.array([1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
_BK16_VEJ_GRID  = jnp.array([0.1, 0.2, 0.3, 0.4])
_BK16_A_GRID = jnp.array([[2.01, 4.52, 8.16, 16.3],
                           [0.81, 1.9,  3.2,  5.0],
                           [0.56, 1.31, 2.19, 3.0],
                           [0.27, 0.55, 0.95, 2.0],
                           [0.20, 0.39, 0.65, 0.9]])
_BK16_B_GRID = jnp.array([[0.28, 0.62, 1.19, 2.4],
                           [0.19, 0.28, 0.45, 0.65],
                           [0.17, 0.21, 0.31, 0.45],
                           [0.10, 0.13, 0.15, 0.17],
                           [0.06, 0.11, 0.12, 0.12]])
_BK16_D_GRID = jnp.array([[1.12, 1.39, 1.52, 1.65],
                           [0.86, 1.21, 1.39, 1.5],
                           [0.74, 1.13, 1.32, 1.4],
                           [0.60, 0.90, 1.13, 1.25],
                           [0.63, 0.79, 1.04, 1.5]])

# Pre-computed log-mirrored quadrature nodes (module-level — static shapes for JIT)
import numpy as _np
def _build_log_mirror_quad(n_half=50, minimum_log_spacing=-3):
    lsp = _np.logspace(minimum_log_spacing, 0, n_half)
    xm  = _np.unique(_np.concatenate((lsp, 1.0 - lsp)))
    return jnp.array(xm, dtype=jnp.float32)

_ARNETT_QUAD_NODES = _build_log_mirror_quad(n_half=50)
_CSM_QUAD_NODES    = _build_log_mirror_quad(n_half=250)


# ---------------------------------------------------------------------------
# Diffusion timescale helpers
# ---------------------------------------------------------------------------

@jit
def _compute_diffusion_constants(log10_kappa, log10_kappa_gamma,
                                  log10_mej_msun, log10_vej_kms):
    """
    Compute Arnett diffusion timescale constants in log10 space (float32-safe).

    :param log10_kappa: log10 of opacity in cm^2/g
    :param log10_kappa_gamma: log10 of gamma-ray opacity in cm^2/g
    :param log10_mej_msun: log10 of ejecta mass in solar masses
    :param log10_vej_kms: log10 of ejecta velocity in km/s
    :return: (log10_td_days, log10_A_trap_days2)
    """
    log10_mej_g = log10_mej_msun + _LOG10_MSUN

    # tau_diff = sqrt(2 * kappa * mej_g / (13.7 * c * vej_cms))
    log10_td_sec = 0.5 * (_LOG10_2 + log10_kappa + log10_mej_g
                          - _LOG10_13p7 - _LOG10_CCGS
                          - log10_vej_kms - _LOG10_KM_CGS)
    log10_td_days = log10_td_sec - _LOG10_DAYS2SEC

    # A_trap = 3 * kappa_gamma * mej_g / (4*pi * vej_cms^2)
    log10_A_sec2 = (_LOG10_3 + log10_kappa_gamma + log10_mej_g
                    - _LOG10_4PI - 2.0 * (log10_vej_kms + _LOG10_KM_CGS))
    log10_A_trap = log10_A_sec2 - 2.0 * _LOG10_DAYS2SEC

    return log10_td_days, log10_A_trap


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
def diffusion_convert_luminosity(time, dense_times, log10_luminosity,
                                  kappa, kappa_gamma, mej, vej):
    """
    Arnett (1982) photon diffusion integral in log10 space (float32-safe).

    :param time: source-frame time in days (evaluation points)
    :param dense_times: dense time grid in days (must match log10_luminosity)
    :param log10_luminosity: log10 of engine luminosity on dense_times (erg/s)
    :param kappa: opacity in cm^2/g
    :param kappa_gamma: gamma-ray opacity in cm^2/g
    :param mej: ejecta mass in solar masses
    :param vej: ejecta velocity in km/s
    :return: (tau_diff_days, log10_new_luminosity) — both scalars/arrays in log10 erg/s
    """
    # Cast all inputs to time's dtype (float32 when x64 is disabled)
    fp           = time.dtype
    kappa_f      = jnp.asarray(kappa,       dtype=fp)
    kappa_gamma_f= jnp.asarray(kappa_gamma, dtype=fp)
    mej_f        = jnp.asarray(mej,         dtype=fp)
    vej_f        = jnp.asarray(vej,         dtype=fp)
    log10_luminosity = jnp.asarray(log10_luminosity, dtype=fp)
    dense_times  = jnp.asarray(dense_times, dtype=fp)

    log10_td, log10_A = _compute_diffusion_constants(
        jnp.log10(jnp.maximum(kappa_f,       jnp.array(1e-30, dtype=fp))),
        jnp.log10(jnp.maximum(kappa_gamma_f, jnp.array(1e-30, dtype=fp))),
        jnp.log10(jnp.maximum(mej_f,         jnp.array(1e-30, dtype=fp))),
        jnp.log10(jnp.maximum(vej_f,         jnp.array(1e-30, dtype=fp))),
    )
    td     = jnp.power(jnp.array(10.0, dtype=fp), log10_td)   # days
    A_trap = jnp.power(jnp.array(10.0, dtype=fp), log10_A)    # days^2

    log10_L_scale = jnp.maximum(jnp.max(log10_luminosity), jnp.array(30.0, dtype=fp))
    L_engine_n    = jnp.power(jnp.array(10.0, dtype=fp), log10_luminosity - log10_L_scale)

    def _integrate_one(t_e):
        int_t    = t_e * _ARNETT_QUAD_NODES                        # (N,)
        L_at_t   = jnp.interp(int_t, dense_times, L_engine_n)
        exponent = jnp.clip((int_t ** 2 - t_e ** 2) / td ** 2, -80.0, 0.0)
        integrand = L_at_t * int_t * jnp.exp(exponent)
        integral  = jnp.trapezoid(integrand, int_t)
        trap_factor = -jnp.expm1(-A_trap / jnp.maximum(t_e ** 2, 1e-30))
        return jnp.maximum(2.0 / td ** 2 * integral * trap_factor, 0.0)

    L_obs_n     = jax.vmap(_integrate_one)(time)
    log10_L_obs = jnp.log10(jnp.maximum(L_obs_n, jnp.array(1e-30, dtype=fp))) + log10_L_scale

    return td, log10_L_obs


@jit
def barnes_kasen_16_thermalisation(mej, vej):
    """
    Bilinear interpolation of the Barnes & Kasen (2016) r-process thermalisation
    efficiency table (Table 1 of Barnes et al. 2016).

    :param mej: ejecta mass in solar masses
    :param vej: ejecta velocity as a fraction of the speed of light
    :return: (av, bv, dv) thermalisation efficiency parameters
    """
    mej_c = jnp.clip(mej, _BK16_MASS_GRID[0], _BK16_MASS_GRID[-1])
    vej_c = jnp.clip(vej, _BK16_VEJ_GRID[0], _BK16_VEJ_GRID[-1])

    im = jnp.searchsorted(_BK16_MASS_GRID, mej_c, side='right') - 1
    im = jnp.clip(im, 0, len(_BK16_MASS_GRID) - 2)
    m0, m1 = _BK16_MASS_GRID[im], _BK16_MASS_GRID[im + 1]
    tm = (mej_c - m0) / (m1 - m0)

    iv = jnp.searchsorted(_BK16_VEJ_GRID, vej_c, side='right') - 1
    iv = jnp.clip(iv, 0, len(_BK16_VEJ_GRID) - 2)
    v0, v1 = _BK16_VEJ_GRID[iv], _BK16_VEJ_GRID[iv + 1]
    tv = (vej_c - v0) / (v1 - v0)

    def bilinear(grid):
        f00 = grid[im, iv];     f01 = grid[im, iv + 1]
        f10 = grid[im + 1, iv]; f11 = grid[im + 1, iv + 1]
        return (f00 * (1 - tm) * (1 - tv) + f01 * (1 - tm) * tv
                + f10 * tm * (1 - tv) + f11 * tm * tv)

    return bilinear(_BK16_A_GRID), bilinear(_BK16_B_GRID), bilinear(_BK16_D_GRID)


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2013ApJ...773...76C/abstract')
def csm_diffusion_convert_luminosity(time, dense_times, log10_luminosity,
                                      kappa, r_photosphere, mass_csm_threshold):
    """
    CSM photon diffusion integral in log10 space (float32-safe).

    :param time: source-frame time in days (evaluation points)
    :param dense_times: dense time grid in days
    :param log10_luminosity: log10 of engine luminosity on dense_times (erg/s)
    :param kappa: opacity in cm^2/g
    :param r_photosphere: photosphere radius in cm
    :param mass_csm_threshold: mass of optically thick CSM in grams
    :return: log10 of diffused luminosity at time (erg/s)
    """
    fp           = time.dtype
    log10_luminosity = jnp.asarray(log10_luminosity, dtype=fp)
    dense_times  = jnp.asarray(dense_times, dtype=fp)
    kappa_f      = jnp.asarray(kappa, dtype=fp)
    r_phot_f     = jnp.asarray(r_photosphere, dtype=fp)
    mass_csm_f   = jnp.asarray(mass_csm_threshold, dtype=fp)

    beta_csm = jnp.array(4.0 * _math.pi ** 3 / 9.0, dtype=fp)
    t0 = (kappa_f * mass_csm_f
          / (beta_csm * jnp.array(2.998e10, dtype=fp) * r_phot_f)
          / jnp.array(86400.0, dtype=fp))   # days
    t0 = jnp.maximum(t0, jnp.array(1e-30, dtype=fp))

    log10_L_scale = jnp.maximum(jnp.max(log10_luminosity), jnp.array(30.0, dtype=fp))
    L_n           = jnp.power(jnp.array(10.0, dtype=fp), log10_luminosity - log10_L_scale)

    def _csm_integrate_one(t_e):
        int_t    = t_e * _CSM_QUAD_NODES.astype(fp)
        L_at_t   = jnp.interp(int_t, dense_times, L_n)
        integrand = L_at_t * jnp.exp((int_t - t_e) / t0)
        return jnp.maximum(jnp.trapezoid(integrand, int_t) / t0, jnp.array(0.0, dtype=fp))

    L_diff_n    = jax.vmap(_csm_integrate_one)(time)
    log10_L_obs = jnp.log10(jnp.maximum(L_diff_n, jnp.array(1e-30, dtype=fp))) + log10_L_scale
    return log10_L_obs
