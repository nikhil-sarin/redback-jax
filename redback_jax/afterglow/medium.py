"""Ambient-medium primitives shared by afterglow dynamics backends."""

import jax.numpy as jnp
from jax import jit
from jax.scipy.special import logsumexp


@jit
def power_law_density(radius, density_reference, radius_reference, index):
    """Number density ``n(R) = n_ref (R / R_ref)**(-index)`` in cm^-3."""
    radius = jnp.asarray(radius)
    return density_reference * jnp.power(radius / radius_reference, -index)


@jit
def swept_mass_derivative(radius, number_density, solid_angle, proton_mass):
    """Evaluate ``dm/dR`` for a cold hydrogen ambient medium."""
    return solid_angle * radius**2 * number_density * proton_mass


@jit
def power_law_log_density(
    log10_radius, log10_density_reference, log10_radius_reference, index
):
    """Log-density form suitable for arbitrary-medium dynamics."""
    return log10_density_reference - index * (log10_radius - log10_radius_reference)


@jit
def smoothly_broken_power_law_log_density(
    log10_radius,
    log10_density_break,
    log10_radius_break,
    inner_index,
    outer_index,
    smoothness=5.0,
):
    """A positive, continuous CSM profile normalized at the break radius."""
    coordinate = log10_radius - log10_radius_break
    terms = jnp.stack(
        (smoothness * inner_index * coordinate, smoothness * outer_index * coordinate)
    )
    log10_sum = logsumexp(terms * jnp.log(10.0), axis=0) / jnp.log(10.0)
    return log10_density_break + jnp.log10(2.0) / smoothness - log10_sum / smoothness


@jit
def tabulated_log_density(log10_radius, grid_log10_radius, grid_log10_density):
    """Interpolate a tabulated radial CSM profile in log-log space."""
    return jnp.interp(log10_radius, grid_log10_radius, grid_log10_density)
