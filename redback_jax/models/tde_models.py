"""
JAX-friendly tidal disruption event (TDE) analytical light-curve models.

Reference:
    Redback: https://github.com/nikhil-sarin/redback/blob/master/redback/transient_models/tde_models.py
"""

import math as _math

import jax.numpy as jnp
from jax import jit

from redback_jax.utils.citation_wrapper import citation_wrapper
from redback_jax.interaction_processes import diffusion_convert_luminosity

_DAY_TO_S = 86400.0


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
