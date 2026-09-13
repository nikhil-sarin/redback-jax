"""
JAX-friendly tidal disruption event (TDE) analytical light-curve models.

Reference:
    Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/tde_models.py
"""

import math as _math
from pathlib import Path as _Path

import jax.numpy as jnp
import numpy as _np
from jax import vmap
from jax import jit

from redback_jax.utils.citation_wrapper import citation_wrapper
from redback_jax.interaction_processes import diffusion_convert_luminosity

_DAY_TO_S = 86400.0
_C_CGS = 2.99792458e10
_G_CGS = 6.67430e-8
_SOLAR_MASS = 1.988409870698051e33
_KAPPA_T = 0.2 * (1.0 + 0.74)
_LOG10_LN_E = _math.log10(_math.e)
_LOG10_EDDINGTON_COEFFICIENT = _math.log10(
    4.0 * _math.pi * _G_CGS * _SOLAR_MASS * _C_CGS / _KAPPA_T)


def _load_guillochon_fallback_tables(n_phase=512):
    """Load and rectangularise the Guillochon et al. (2013) fallback tables.

    Redback constructs ragged interpolation objects on every model call.  Here
    each pre/post-peak branch is resampled onto a common phase coordinate once
    at import time.  The resulting fixed-shape arrays can be interpolated in
    ``bb`` and stellar mass inside JIT-compiled JAX code.
    """
    table_root = (_Path(__file__).resolve().parents[1] / "tables" /
                  "guillochon_tde_data")
    phase = _np.linspace(0.0, 1.0, n_phase)
    mh_base_g = 1.0e6 * _SOLAR_MASS

    tables = {}
    for gamma in ("4-3", "5-3"):
        beta_values = []
        log_time_branches = []
        log_dmdt_branches = []
        for filename in sorted((table_root / gamma).glob("*.dat"),
                               key=lambda path: float(path.stem)):
            energy, dmde = _np.loadtxt(filename)
            bound = energy < 0.0
            energy = energy[bound]
            dmde = dmde[bound]
            dedt = ((-2.0 * energy) ** 2.5 /
                    (6.0 * _np.pi * _G_CGS * mh_base_g))
            log_time = _np.log10(
                (2.0 * _np.pi * _G_CGS * mh_base_g) *
                (-2.0 * energy) ** -1.5)
            log_dmdt = _np.log10(dmde * dedt)
            peak = int(_np.argmax(log_dmdt))

            time_pair = (log_time[:peak], log_time[peak:])
            dmdt_pair = (log_dmdt[:peak], log_dmdt[peak:])
            resampled_time = []
            resampled_dmdt = []
            for branch_time, branch_dmdt in zip(time_pair, dmdt_pair):
                branch_phase = ((branch_time - branch_time[0]) /
                                (branch_time[-1] - branch_time[0]))
                branch_phase[0] = 0.0
                branch_phase[-1] = 1.0
                resampled_time.append(_np.interp(phase, branch_phase, branch_time))
                resampled_dmdt.append(_np.interp(phase, branch_phase, branch_dmdt))

            beta_values.append(float(filename.stem))
            log_time_branches.append(resampled_time)
            log_dmdt_branches.append(resampled_dmdt)

        tables[gamma] = (
            jnp.asarray(beta_values),
            jnp.asarray(_np.asarray(log_time_branches)),
            jnp.asarray(_np.asarray(log_dmdt_branches)),
        )
    return tables


_FALLBACK_TABLES = _load_guillochon_fallback_tables()


def _interp_beta_table(beta, gamma):
    """Interpolate a rectangular fallback table along its beta axis."""
    beta_grid, log_time, log_dmdt = _FALLBACK_TABLES[gamma]
    upper = jnp.clip(jnp.searchsorted(beta_grid, beta, side="right"),
                     1, beta_grid.size - 1)
    lower = upper - 1
    fraction = ((beta - beta_grid[lower]) /
                (beta_grid[upper] - beta_grid[lower]))
    fraction = jnp.clip(fraction, 0.0, 1.0)
    time_out = log_time[lower] + fraction * (log_time[upper] - log_time[lower])
    dmdt_out = log_dmdt[lower] + fraction * (log_dmdt[upper] - log_dmdt[lower])
    return time_out, dmdt_out


def _stellar_radius_tout(mstar):
    """Tout et al. main-sequence mass-radius relation used by MOSFiT."""
    mass = jnp.maximum(mstar, 0.1)
    log_z = _math.log10(0.0134 / 0.02)
    theta = (1.71535900 + 0.62246212 * log_z - 0.92557761 * log_z**2
             - 1.16996966 * log_z**3 - 0.30631491 * log_z**4)
    ll = (6.59778800 - 0.42450044 * log_z - 12.13339427 * log_z**2
          - 10.73509484 * log_z**3 - 2.51487077 * log_z**4)
    kpa = (10.08855000 - 7.11727086 * log_z - 31.67119479 * log_z**2
           - 24.24848322 * log_z**3 - 5.33608972 * log_z**4)
    lbda = (1.01249500 + 0.32699690 * log_z - 0.00923418 * log_z**2
            - 0.03876858 * log_z**3 - 0.00412750 * log_z**4)
    mu = (0.07490166 + 0.02410413 * log_z + 0.07233664 * log_z**2
          + 0.03040467 * log_z**3 + 0.00197741 * log_z**4)
    nu = 0.01077422
    eps = (3.08223400 + 0.94472050 * log_z - 2.15200882 * log_z**2
           - 2.49219496 * log_z**3 - 0.63848738 * log_z**4)
    oo = (17.84778000 - 7.45345690 * log_z - 48.9606685 * log_z**2
          - 40.05386135 * log_z**3 - 9.09331816 * log_z**4)
    pi = (0.00022582 - 0.00186899 * log_z + 0.00388783 * log_z**2
          + 0.00142402 * log_z**3 - 0.00007671 * log_z**4)
    numerator = (theta * mass**2.5 + ll * mass**6.5 + kpa * mass**11
                 + lbda * mass**19 + mu * mass**19.5)
    denominator = (nu + eps * mass**2 + oo * mass**8.5 + mass**18.5
                   + pi * mass**19.5)
    return numerator / denominator


@jit
def _tde_fallback_engine_log10(times, mbh6, mstar, bb, eta, leddlimit):
    """JAX/MOSFiT fallback engine, returning log10 luminosity."""
    beta43 = jnp.where(bb < 1.0, 0.6 + 1.25 * bb,
                       1.85 + 2.15 * (bb - 1.0))
    beta53 = jnp.where(bb < 1.0, 0.5 + 0.4 * bb,
                       0.9 + 1.6 * (bb - 1.0))
    time43, dmdt43 = _interp_beta_table(beta43, "4-3")
    time53, dmdt53 = _interp_beta_table(beta53, "5-3")

    low_transition = jnp.clip((1.0 - mstar) / 0.7, 0.0, 1.0)
    high_transition = jnp.clip((mstar - 15.0) / 7.0, 0.0, 1.0)
    gamma_fraction = jnp.where(mstar < 1.0, low_transition,
                               jnp.where(mstar > 15.0, high_transition, 0.0))
    log_time = time43 + gamma_fraction * (time53 - time43)
    log_dmdt = dmdt43 + gamma_fraction * (dmdt53 - dmdt43)

    sim_time_s = jnp.power(10.0, log_time.reshape(-1))
    log_dmdt = log_dmdt.reshape(-1)
    rstar = _stellar_radius_tout(mstar)
    mh = mbh6 * 1.0e6
    time_scale = jnp.sqrt(mh / 1.0e6) / mstar * rstar**1.5
    dmdt_scale = jnp.sqrt(1.0e6 / mh) * mstar**2 / rstar**1.5
    sim_time_days = sim_time_s * time_scale / _DAY_TO_S
    sim_time_days = sim_time_days - sim_time_days[0]
    log_dmdt = log_dmdt + jnp.log10(dmdt_scale)

    # The pre-peak and post-peak branches meet monotonically after flattening.
    log_rate = jnp.interp(times, sim_time_days, log_dmdt,
                          left=-jnp.inf, right=-jnp.inf)
    log_uncapped = jnp.log10(eta) + log_rate + 2.0 * _math.log10(_C_CGS)
    # Evaluate in log space: the linear product exceeds float32 (~1e44).
    log_edd = _LOG10_EDDINGTON_COEFFICIENT + jnp.log10(mh)
    log_cap = jnp.log10(leddlimit) + log_edd
    maximum = jnp.maximum(log_uncapped, log_cap)
    log_sum = maximum + jnp.log10(
        jnp.power(10.0, log_uncapped - maximum) +
        jnp.power(10.0, log_cap - maximum))
    return log_uncapped + log_cap - log_sum


@jit
def _viscous_convert_log10(time, dense_times, log10_luminosity, tvisc):
    """Apply Redback's exponential viscous convolution in log10 space."""
    half_steps = 500
    minimum_log_spacing = -3.0
    lsp = jnp.logspace(
        jnp.log10(tvisc / dense_times[-1]) + minimum_log_spacing,
        0.0, half_steps)
    fraction = jnp.sort(jnp.concatenate((lsp, 1.0 - lsp)))

    def convolve_one(eval_time):
        integration_time = jnp.clip(eval_time * fraction, 0.0, dense_times[-1])
        log_lum = jnp.interp(integration_time, dense_times, log10_luminosity,
                             left=-jnp.inf, right=-jnp.inf)
        log_integrand = (log_lum +
                         (integration_time - eval_time) / tvisc * _LOG10_LN_E)
        scale = jnp.max(jnp.where(jnp.isfinite(log_integrand), log_integrand,
                                  -jnp.inf))
        scaled = jnp.where(jnp.isfinite(log_integrand),
                           jnp.power(10.0, log_integrand - scale), 0.0)
        integral = jnp.trapezoid(scaled, integration_time) / tvisc
        return scale + jnp.log10(jnp.maximum(integral, 1e-30))

    result = vmap(convolve_one)(time)
    return jnp.where(time >= 0.0, result, -jnp.inf)


@jit
def _analytic_fallback_log10(time, log10_l0, t_0_turn):
    """
    log10 of t^{-5/3} fallback luminosity with flat plateau below t_0_turn.

    :param time: source-frame time in days
    :param log10_l0: log10 of bolometric luminosity at 1 second in erg/s
    :param t_0_turn: turn-on time in days
    :return: log10 of bolometric luminosity in erg/s
    """
    t_eff = jnp.maximum(time, t_0_turn)
    log10_L = log10_l0 - (5.0 / 3.0) * jnp.log10(t_eff * _DAY_TO_S)
    return log10_L


@citation_wrapper('redback')
@jit
def tde_analytical_bolometric(time, log10_l0, t_0_turn, mej, vej, kappa, kappa_gamma):
    """
    Bolometric TDE light curve: t^{-5/3} fallback engine + Arnett diffusion.

    :param time: source-frame time in days
    :param log10_l0: log10 of bolometric luminosity at 1 second in erg/s
    :param t_0_turn: turn-on time in days
    :param mej: ejecta mass in solar masses
    :param vej: ejecta velocity in km/s
    :param kappa: optical opacity in cm^2/g
    :param kappa_gamma: gamma-ray opacity in cm^2/g
    :return: log10 of bolometric luminosity in erg/s
    """
    dense_times = jnp.linspace(0.01, time[-1] + 100.0, 1000)
    log10_dense = _analytic_fallback_log10(dense_times, log10_l0, t_0_turn)
    _, log10_lbol = diffusion_convert_luminosity(
        time=time, dense_times=dense_times, log10_luminosity=log10_dense,
        kappa=kappa, kappa_gamma=kappa_gamma, mej=mej, vej=vej)
    return log10_lbol


@citation_wrapper(
    'https://ui.adsabs.harvard.edu/abs/2019ApJ...872..151M/abstract, '
    'https://ui.adsabs.harvard.edu/abs/2013ApJ...767...25G/abstract, '
    'https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract'
)
@jit
def tde_fallback_bolometric(time, mbh6, mstar, tvisc, bb, eta, leddlimit):
    """MOSFiT/Guillochon TDE fallback model with viscous processing.

    This is the JAX counterpart of Redback's ``tde_fallback_bolometric``.
    It interpolates the Guillochon et al. (2013) hydrodynamic fallback tables,
    applies the MOSFiT stellar-mass and black-hole-mass scalings and Eddington
    cap, then performs Redback's exponential viscous convolution.

    :param time: source-frame time after first fallback in days
    :param mbh6: black-hole mass in units of 10^6 solar masses
    :param mstar: stellar mass in solar masses
    :param tvisc: viscous timescale in days (strictly positive)
    :param bb: dimensionless impact-parameter mapping, in the range [0, 2]
    :param eta: radiative efficiency (strictly positive)
    :param leddlimit: multiplicative Eddington-limit factor (strictly positive)
    :return: log10 of bolometric luminosity in erg/s
    """
    dense_times = jnp.geomspace(1.0e-5, time[-1] + 100.0, 1000)
    log10_fallback = _tde_fallback_engine_log10(
        dense_times, mbh6, mstar, bb, eta, leddlimit)
    return _viscous_convert_log10(time, dense_times, log10_fallback, tvisc)
