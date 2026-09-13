"""Central-engine histories for non-impulsive afterglow dynamics."""

import jax.numpy as jnp
from jax import jit


@jit
def constant_engine_cumulative_log10(time_seconds, parameters):
    """Cumulative energy for constant power over a finite duration.

    Parameters are ``(log10_total_energy_erg, log10_duration_seconds)``.
    """
    log10_total_energy, log10_duration = parameters
    duration = jnp.power(10.0, log10_duration)
    fraction = jnp.clip(time_seconds / duration, 0.0, 1.0)
    return log10_total_energy + jnp.log10(fraction)


@jit
def fallback_engine_cumulative_log10(time_seconds, parameters):
    """Cumulative energy for constant power followed by ``t^-5/3``.

    Parameters are ``(log10_total_energy_erg, log10_break_time_seconds)``.
    The history is normalized so its integral to infinite time is the supplied
    total energy.
    """
    log10_total_energy, log10_break_time = parameters
    break_time = jnp.power(10.0, log10_break_time)
    ratio = jnp.maximum(time_seconds / break_time, 0.0)
    before_break = 0.4 * ratio
    after_break = 1.0 - 0.6 * jnp.power(jnp.maximum(ratio, 1.0), -2.0 / 3.0)
    fraction = jnp.where(ratio <= 1.0, before_break, after_break)
    return log10_total_energy + jnp.log10(fraction)


@jit
def tabulated_engine_cumulative_log10(time_seconds, parameters):
    """Interpolate cumulative energy from a fixed log-time table.

    Parameters are ``(grid_log10_time_seconds, grid_log10_energy_erg)``.
    """
    grid_log10_time, grid_log10_energy = parameters
    minimum_time = jnp.power(10.0, grid_log10_time[0])
    safe_time = jnp.maximum(time_seconds, minimum_time)
    interpolated = jnp.interp(jnp.log10(safe_time), grid_log10_time, grid_log10_energy)
    return jnp.where(time_seconds > 0.0, interpolated, -jnp.inf)
