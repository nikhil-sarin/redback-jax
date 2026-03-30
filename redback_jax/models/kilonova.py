"""
JAX-based kilonova light-curve models.

References:
    Metzger 2017: https://ui.adsabs.harvard.edu/abs/2017LRR....20....3M/abstract
    Barnes & Kasen 2013/2016: thermalisation efficiency
    Kasen & Bildsten 2010, Yu et al. 2017: magnetar-boosted kilonova
"""

import math as _math

import jax
import jax.numpy as jnp
from jax import jit

from redback_jax.utils.citation_wrapper import citation_wrapper
from redback_jax.interaction_processes import barnes_kasen_16_thermalisation

# Physical constants as Python floats (avoids astropy float64 promotion)
_SOLAR_MASS     = 1.989e33   # g
_SPEED_OF_LIGHT = 2.998e10   # cm/s
_DAY_TO_S       = 86400.0    # s/day

# Magnetar log10 constants
_LOG10_EROT_COEFF = _math.log10(2.6e52)
_LOG10_TP_COEFF   = _math.log10(1.3e5)
_LOG10_2_FLOAT    = _math.log10(2.0)
_LOG10_MSUN       = _math.log10(_SOLAR_MASS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@jit
def _electron_fraction_from_kappa(kappa):
    """Approximate electron fraction from gray opacity."""
    return jnp.where(kappa < 1.0, 0.4,
           jnp.where(kappa < 5.0, 0.35,
           jnp.where(kappa < 20.0, 0.25, 0.1)))


@jit
def _rprocess_heating_rate(t_days, e_th):
    """r-process heating rate per unit mass (erg/s/g)."""
    t_sec  = t_days * _DAY_TO_S
    t0     = 1.3    # seconds
    sig    = 0.11   # seconds
    edotr_late  = 2.1e10 * e_th * jnp.maximum(t_days, 1e-10) ** (-1.3)
    edotr_early = (4.0e18
                   * jnp.power(0.5 - jnp.arctan((t_sec - t0) / sig) / jnp.pi, 1.3)
                   * e_th)
    return jnp.where(t_sec > t0, edotr_late, edotr_early)


@jit
def _magnetar_log10_lbol(t_sec, log10_p0_ms, log10_bp, mass_ns, theta_pb):
    """
    Dipole spin-down: returns log10(L) in erg/s (float32-safe).

    :param t_sec: time in seconds
    :param log10_p0_ms: log10 of spin period in milliseconds
    :param log10_bp: log10 of B-field in units of 10^14 G
    :param mass_ns: NS mass in solar masses
    :param theta_pb: spin–B-field angle in radians
    :return: log10 of luminosity in erg/s
    """
    log10_mass_ratio = jnp.log10(mass_ns / 1.4)
    log10_erot = (_LOG10_EROT_COEFF + 1.5 * log10_mass_ratio - 2.0 * log10_p0_ms)
    log10_tp   = (_LOG10_TP_COEFF - 2.0 * log10_bp + 2.0 * log10_p0_ms
                  + 1.5 * log10_mass_ratio
                  - jnp.log10(jnp.maximum(jnp.sin(theta_pb) ** 2, 1e-10)))
    tp = jnp.power(10.0, log10_tp)

    log10_L = (_LOG10_2_FLOAT + log10_erot - log10_tp
               - 2.0 * jnp.log10(1.0 + t_sec / tp))
    return log10_L


# ---------------------------------------------------------------------------
# Metzger kilonova (200-shell ODE via jax.lax.scan)
# Reference: Metzger 2017, redback _metzger_kilonova_model
# ---------------------------------------------------------------------------

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2017LRR....20....3M/abstract')
def metzger_kilonova_bolometric(time, mej, vej, beta, kappa,
                                vmax=0.7, neutron_precursor=True):
    """
    Bolometric kilonova light curve (Metzger 2017) with 200 shells and
    Barnes+16 thermalisation, solved via a sequential Euler ODE with
    jax.lax.scan.

    :param time: source-frame time in days (must be strictly increasing, ≥2 points)
    :param mej: ejecta mass in solar masses
    :param vej: minimum ejecta velocity in units of c
    :param beta: velocity power-law slope (M ∝ v^{-beta})
    :param kappa: gray opacity in cm^2/g
    :param vmax: maximum ejecta velocity in units of c (default 0.7)
    :param neutron_precursor: include neutron precursor emission (default True)
    :return: log10 of bolometric luminosity in erg/s
    """
    return _metzger_kilonova_scan(
        time, mej, vej, beta, kappa, vmax, neutron_precursor)


def _metzger_kilonova_scan(time, mej, vej, beta, kappa, vmax, neutron_precursor):
    mass_len = 200
    fp = time.dtype

    t_sec = time * _DAY_TO_S
    dt    = jnp.diff(t_sec)           # (T-1,)

    av, bv, dv = barnes_kasen_16_thermalisation(mej, vej)
    e_th = 0.36 * (jnp.exp(-av * time)
                   + jnp.log1p(2.0 * bv * jnp.maximum(time, 1e-10) ** dv)
                   / (2.0 * bv * jnp.maximum(time, 1e-10) ** dv))

    edotr = _rprocess_heating_rate(time, e_th)   # (T,) erg/s/g

    vel    = jnp.linspace(vej, vmax, mass_len)
    m_arr  = mej * (vel / vej) ** (-beta)         # solar masses, (S,)
    v_m    = vel * _SPEED_OF_LIGHT                 # cm/s, (S,)
    dm     = jnp.abs(jnp.diff(m_arr))             # (S-1,)

    tau_neutron = 900.0  # seconds

    if neutron_precursor:
        Ye           = _electron_fraction_from_kappa(kappa)
        # Metzger 2014 eq. 7: transition mass m_n ~ 1e-4 Msun separates
        # neutron-dominated outer layers (m << m_n) from r-process inner layers
        neutron_mass = 1e-4 * _SOLAR_MASS
        Xn0 = ((1.0 - 2.0 * Ye) * 2.0 * jnp.arctan(neutron_mass
               / (m_arr * _SOLAR_MASS)) / jnp.pi)
        Xr  = 1.0 - Xn0

    # E0 in Msun*(cm/s)^2  (safe: ~1e17, fits float32)
    E0 = 0.5 * m_arr * v_m ** 2   # (S,)

    # Normalize by E_scale to keep intermediates O(1) throughout scan
    # E_scale chosen as max(E0) in erg: E0_erg = E0 * solar_mass
    # log10(E_scale) = log10(max(E0)) + LOG10_MSUN
    log10_E_scale = jnp.maximum(
        jnp.log10(jnp.maximum(jnp.max(E0), jnp.array(1e-30, dtype=fp))) + _LOG10_MSUN,
        jnp.array(30.0, dtype=fp))
    E_scale = jnp.power(jnp.array(10.0, dtype=fp), log10_E_scale)  # erg

    # msun_per_E = solar_mass / E_scale  (for converting Msun*(cm/s)^2 → normalized)
    msun_per_E = jnp.power(jnp.array(10.0, dtype=fp),
                            jnp.array(_LOG10_MSUN, dtype=fp) - log10_E_scale)

    E0_n = E0 * msun_per_E   # normalized: O(1)

    def _step(E_n, inputs):
        t_i, dt_i, edotr_i = inputs

        if neutron_precursor:
            Xn_t    = Xn0 * jnp.exp(-t_i / tau_neutron)
            # Metzger 2014 eq.: ėn = 3.2e14 * Xn (linear, not quadratic)
            edotn   = 3.2e14 * Xn_t
            # kappa_n: e-scattering from protons produced by neutron decay
            # (fraction that was Xn0 but has since decayed = Xn0 - Xn(t))
            kappa_n = 0.4 * (1.0 - Xn_t - Xr)
            kap     = kappa_n + kappa * Xr
        else:
            edotn = jnp.zeros(mass_len)
            kap   = kappa * jnp.ones(mass_len)

        # Diffusion timescale per shell
        td_v = (kap[:-1] * m_arr[:-1] * _SOLAR_MASS * 3.0
                / (4.0 * jnp.pi * v_m[:-1] * _SPEED_OF_LIGHT * t_i * beta))

        # lum_n = E_n / (td_v + t_i*v/c)  [normalized lum, erg/s / E_scale]
        lum_n = E_n[:-1] / (td_v + t_i * v_m[:-1] / _SPEED_OF_LIGHT)

        # heat in normalized units: edotr [erg/s/g] * dm [Msun] * msun [g/Msun] / E_scale
        heat_n = (edotr_i + edotn[:-1]) * dm * msun_per_E

        E_new_inner = E_n[:-1] + (heat_n - E_n[:-1] / t_i - lum_n) * dt_i
        E_new = jnp.concatenate([jnp.maximum(E_new_inner, jnp.array(0.0, dtype=fp)), E_n[-1:]])

        # L_total = sum(lum_n * dm * msun_per_E * E_scale) = sum(lum_n * dm) * solar_mass
        # = sum(lum * dm) * solar_mass where lum = lum_n * E_scale
        # Normalized: L_n_total = sum(lum_n * dm)  (to convert: * solar_mass / msun_per_E = E_scale)
        # But we can just store lum_n_total and multiply by E_scale at the end
        L_n_total = jnp.sum(lum_n * dm)  # in 1/s (normalized by E_scale)

        tau = (m_arr[:-1] * _SOLAR_MASS * kap[:-1]
               / (4.0 * jnp.pi * (t_i * v_m[:-1]) ** 2))
        tau_full = jnp.concatenate([tau, tau[-1:]])
        pig  = jnp.argmin(jnp.abs(tau_full - 1.0))
        R_ph = v_m[pig] * t_i

        return E_new, (L_n_total, R_ph)

    _, (L_n_arr, R_arr) = jax.lax.scan(
        _step, E0_n, (t_sec[:-1], dt, edotr[:-1]))

    # log10(L) = log10(L_n) + log10_E_scale  (no exponentiation, no float32 overflow)
    log10_L_n = jnp.log10(jnp.maximum(L_n_arr, jnp.array(1e-30, dtype=fp)))
    log10_L = log10_L_n + log10_E_scale
    log10_L = jnp.concatenate([log10_L, log10_L[-1:]])

    return log10_L


# ---------------------------------------------------------------------------
# Magnetar-boosted kilonova (200-shell ODE + dipole spin-down injection)
# Reference: Yu et al. 2013
# ---------------------------------------------------------------------------

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2013ApJ...776L..40Y/abstract')
def magnetar_boosted_kilonova_bolometric(time, mej, vej, beta, kappa,
                                         p0, bp, mass_ns, theta_pb,
                                         thermalisation_efficiency=1.0,
                                         vmax=0.7, neutron_precursor=True):
    """
    Bolometric kilonova light curve with magnetar spin-down energy injection.

    :param time: source-frame time in days (strictly increasing, ≥2 points)
    :param mej: ejecta mass in solar masses
    :param vej: minimum ejecta velocity in units of c
    :param beta: velocity power-law slope
    :param kappa: gray opacity in cm^2/g
    :param p0: initial spin period in milliseconds
    :param bp: polar B-field in units of 10^14 G
    :param mass_ns: neutron star mass in solar masses
    :param theta_pb: angle between spin and B-field axes in radians
    :param thermalisation_efficiency: fraction of magnetar luminosity thermalised (default 1.0)
    :param vmax: maximum ejecta velocity in units of c (default 0.7)
    :param neutron_precursor: include neutron precursor emission (default True)
    :return: log10 of bolometric luminosity in erg/s
    """
    return _magnetar_kilonova_scan(
        time, mej, vej, beta, kappa,
        p0, bp, mass_ns, theta_pb,
        thermalisation_efficiency, vmax, neutron_precursor)


def _magnetar_kilonova_scan(time, mej, vej, beta, kappa,
                             p0, bp, mass_ns, theta_pb,
                             th_eff, vmax, neutron_precursor):
    mass_len = 200
    fp = time.dtype

    t_sec = time * _DAY_TO_S
    dt    = jnp.diff(t_sec)

    av, bv, dv = barnes_kasen_16_thermalisation(mej, vej)
    e_th = 0.36 * (jnp.exp(-av * time)
                   + jnp.log1p(2.0 * bv * jnp.maximum(time, 1e-10) ** dv)
                   / (2.0 * bv * jnp.maximum(time, 1e-10) ** dv))

    edotr = _rprocess_heating_rate(time, e_th)

    # Magnetar luminosity in log10 (float32-safe)
    log10_p0 = jnp.log10(jnp.maximum(p0, jnp.array(1e-10, dtype=fp)))
    log10_bp = jnp.log10(jnp.maximum(bp, jnp.array(1e-10, dtype=fp)))
    log10_L_mag = _magnetar_log10_lbol(t_sec, log10_p0, log10_bp, mass_ns, theta_pb)

    vel   = jnp.linspace(vej, vmax, mass_len)
    m_arr = mej * (vel / vej) ** (-beta)
    v_m   = vel * _SPEED_OF_LIGHT
    dm    = jnp.abs(jnp.diff(m_arr))

    tau_neutron = 900.0

    if neutron_precursor:
        Ye           = _electron_fraction_from_kappa(kappa)
        neutron_mass = 1e-4 * _SOLAR_MASS
        Xn0 = ((1.0 - 2.0 * Ye)
               * 2.0 * jnp.arctan(neutron_mass / (m_arr * _SOLAR_MASS)) / jnp.pi)
        Xr  = 1.0 - Xn0

    E0 = 0.5 * m_arr * v_m ** 2   # Msun*(cm/s)^2, safe in float32

    # Use log10_L_mag[0] as energy scale (magnetar dominates)
    log10_E_scale = jnp.maximum(log10_L_mag[0], jnp.array(30.0, dtype=fp))
    E_scale       = jnp.power(jnp.array(10.0, dtype=fp), log10_E_scale)
    msun_per_E    = jnp.power(jnp.array(10.0, dtype=fp),
                               jnp.array(_LOG10_MSUN, dtype=fp) - log10_E_scale)

    E0_n = E0 * msun_per_E

    def _step(E_n, inputs):
        t_i, dt_i, edotr_i, log10_L_mag_i = inputs

        if neutron_precursor:
            Xn_t    = Xn0 * jnp.exp(-t_i / tau_neutron)
            edotn   = 3.2e14 * Xn_t
            kappa_n = 0.4 * (1.0 - Xn_t - Xr)
            kap     = kappa_n + kappa * Xr
        else:
            edotn = jnp.zeros(mass_len)
            kap   = kappa * jnp.ones(mass_len)

        td_v = (kap[:-1] * m_arr[:-1] * _SOLAR_MASS * 3.0
                / (4.0 * jnp.pi * v_m[:-1] * _SPEED_OF_LIGHT * t_i * beta))

        lum_n = E_n[:-1] / (td_v + t_i * v_m[:-1] / _SPEED_OF_LIGHT)
        heat_n = (edotr_i + edotn[:-1]) * dm * msun_per_E

        # Magnetar injection (normalized): L_mag / E_scale = 10^(log10_L_mag - log10_E_scale)
        L_mag_n = jnp.power(jnp.array(10.0, dtype=fp),
                             log10_L_mag_i - log10_E_scale)
        mag_n = jnp.concatenate([
            jnp.array([th_eff * L_mag_n], dtype=fp),
            jnp.zeros(mass_len - 2, dtype=fp),
        ])

        E_new_inner = E_n[:-1] + (heat_n + mag_n - E_n[:-1] / t_i - lum_n) * dt_i
        E_new = jnp.concatenate([jnp.maximum(E_new_inner, jnp.array(0.0, dtype=fp)), E_n[-1:]])

        L_n_total = jnp.sum(lum_n * dm)

        tau = (m_arr[:-1] * _SOLAR_MASS * kap[:-1]
               / (4.0 * jnp.pi * (t_i * v_m[:-1]) ** 2))
        tau_full = jnp.concatenate([tau, tau[-1:]])
        pig  = jnp.argmin(jnp.abs(tau_full - 1.0))
        R_ph = v_m[pig] * t_i

        return E_new, (L_n_total, R_ph)

    _, (L_n_arr, _) = jax.lax.scan(
        _step, E0_n, (t_sec[:-1], dt, edotr[:-1], log10_L_mag[:-1]))

    log10_L_n = jnp.log10(jnp.maximum(L_n_arr, jnp.array(1e-30, dtype=fp)))
    log10_L = log10_L_n + log10_E_scale
    log10_L = jnp.concatenate([log10_L, log10_L[-1:]])
    return log10_L
