"""Ambient-medium primitives shared by afterglow dynamics backends."""

import jax.numpy as jnp
from jax import jit


@jit
def power_law_density(radius, density_reference, radius_reference, index):
    """Number density ``n(R) = n_ref (R / R_ref)**(-index)`` in cm^-3."""
    radius = jnp.asarray(radius)
    return density_reference * jnp.power(radius / radius_reference, -index)


@jit
def swept_mass_derivative(radius, number_density, solid_angle, proton_mass):
    """Evaluate ``dm/dR`` for a cold hydrogen ambient medium."""
    return solid_angle * radius**2 * number_density * proton_mass
