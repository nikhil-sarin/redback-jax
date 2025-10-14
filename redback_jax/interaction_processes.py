# This is mostly from mosfit/transforms but rewritten to be modular

import jax.numpy as jnp
from jax import jit
from jax.scipy.integrate import trapezoid

from redback_jax.utils.citation_wrapper import citation_wrapper
from redback_jax.constants import *


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1982ApJ...253..785A/abstract')
@jit
def diffusion_convert_luminosity(time, dense_times, luminosity, kappa, kappa_gamma, mej, vej):
    """
    :param time: source frame time in days
    :param dense_times: dense time array in days
    :param dense_luminosity: luminosity
    :param kappa: opacity
    :param kappa_gamma: gamma-ray opacity
    :param mej: ejecta mass in solar masses
    :param vej: ejecta velocity in km/s
    :return: tau_diffusion, new_luminosity accounting for the interaction process at the time values
    """
    timesteps = 100
    minimum_log_spacing = -3
    diffusion_constant = 2.0 * solar_mass / (13.7 * speed_of_light * km_cgs)
    trapping_constant = 3.0 * solar_mass / (4*jnp.pi * km_cgs ** 2)

    tau_diff = jnp.sqrt(diffusion_constant * kappa * mej / vej) / day_to_s
    trap_coeff = (trapping_constant * kappa_gamma * mej / (vej ** 2)) / day_to_s ** 2

    min_te = jnp.min(dense_times)
    tb = jnp.maximum(0.0, min_te)

    # Clip time to valid range, but leave the shape intact.
    clipped_time = jnp.clip(time, tb, dense_times[-1])

    # Create logarithmically spaced integration points.
    num = timesteps // 2
    lsp = jnp.logspace(jnp.log10(tau_diff / dense_times[-1]) + minimum_log_spacing, 0, num)
    xm = jnp.concatenate([lsp, 1 - lsp])
    xm = jnp.sort(xm)

    # Vectorized computation for all time points with shape = (n_times, n_integration_points)
    time_expanded = clipped_time.reshape(-1, 1)  # (n_times, 1)
    xm_expanded = xm.reshape(1, -1)  # (1, n_integration_points)

    # Compute the integration times
    int_times = jnp.clip(tb + (time_expanded - tb) * xm_expanded, tb, dense_times[-1])
    int_te2s = clipped_time ** 2  # Shape: (n_times,)

    # Interpolate luminosity at integration points
    int_lums = jnp.interp(int_times, dense_times, luminosity)

    # Compute integrand
    int_args = int_lums * int_times * jnp.exp((int_times ** 2 - int_te2s.reshape(-1, 1)) / tau_diff**2)
    int_args = jnp.nan_to_num(int_args, nan=0.0, posinf=0.0, neginf=0.0)

    # Integrate along integration points axis
    new_lums = trapezoid(int_args, int_times, axis=1)
    new_lums *= -2.0 * jnp.expm1(-trap_coeff / int_te2s) / tau_diff**2

    return tau_diff, new_lums
