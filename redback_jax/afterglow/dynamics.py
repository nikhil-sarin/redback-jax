"""Blast-wave dynamics for the native Redback afterglow family."""

import math
from functools import partial

import jax.numpy as jnp
from jax import jit, lax

from redback_jax.constants import proton_mass, speed_of_light

_LOG10_FOUR_PI = math.log10(4.0 * math.pi)
_LOG10_PROTON_MASS = math.log10(proton_mass)
_LOG10_C_SQUARED = 2.0 * math.log10(speed_of_light)
# Native Redback adds 1e-15 to a float64 Lorentz factor; at Gamma ~= 1 this
# rounds to five float64 ULPs. Preserve that effective increment in Gamma-1.
_NATIVE_GAMMA_INCREMENT = 5.0 * 2.220446049250313e-16


def _adiabatic_index(gamma_minus_one):
    gamma_squared_minus_one = gamma_minus_one * (2.0 + gamma_minus_one)
    root = jnp.sqrt(jnp.maximum(gamma_squared_minus_one, 0.0))
    temperature = (
        root
        * (root + 1.07 * gamma_squared_minus_one)
        / (3.0 * (1.0 + root + 1.07 * gamma_squared_minus_one))
    )
    z = temperature / (0.24 + temperature)
    return (
        (
            ((((1.07136 * z - 2.39332) * z + 2.32513) * z - 0.96583) * z + 0.18203) * z
            - 1.21937
        )
        * z
        + 5.0
    ) / 3.0


def _rk_increment(
    g_hat, log10_mass, gamma_minus_one, log10_ejecta_mass, factor, thermal
):
    mass_ratio = jnp.power(10.0, log10_mass - log10_ejecta_mass)
    gamma = 1.0 + gamma_minus_one
    gamma_squared_minus_one = gamma_minus_one * (2.0 + gamma_minus_one)
    numerator = mass_ratio * (
        g_hat * gamma_squared_minus_one
        - (g_hat - 1.0) * gamma_squared_minus_one / gamma
    )
    denominator = 1.0 + mass_ratio * (
        thermal
        + (1.0 - thermal) * (2.0 * g_hat * gamma - (g_hat - 1.0) * (1.0 + gamma**-2))
    )
    return factor * numerator / denominator


@partial(jit, static_argnames=("steps",))
def legacy_impulsive_dynamics(
    gamma_initial,
    log10_energy,
    log10_density,
    density_index=0.0,
    thermal_fraction=0.0,
    steps=250,
):
    """Reproduce Redback's impulsive RK4 evolution without float32 overflow.

    Energy is supplied as ``log10(erg)`` and density as ``log10(cm^-3)``.
    The returned swept-up mass is also logarithmic.  ``density_index`` retains
    the native convention ``n(R) = n0 R**(-k)``.
    """
    gamma_initial = jnp.atleast_1d(gamma_initial)
    log10_energy = jnp.broadcast_to(log10_energy, gamma_initial.shape)
    gamma_minus_one_initial = gamma_initial - 1.0
    log10_ejecta_mass = log10_energy - jnp.log10(gamma_initial) - _LOG10_C_SQUARED

    radial_power = 3.0 - density_index
    log10_mass_initial = (
        _LOG10_FOUR_PI
        - jnp.log10(radial_power)
        + log10_density
        + radial_power * 10.0
        + _LOG10_PROTON_MASS
    )
    log10_mass_initial = jnp.broadcast_to(log10_mass_initial, gamma_initial.shape)
    step_size = radial_power * (24.0 - 10.0) / steps
    factor = -step_size * math.log(10.0)

    def step(carry, _):
        gamma_minus_one, log10_mass = carry
        g_hat = _adiabatic_index(gamma_minus_one)
        first = _rk_increment(
            g_hat,
            log10_mass,
            gamma_minus_one,
            log10_ejecta_mass,
            factor,
            thermal_fraction,
        )
        second = _rk_increment(
            g_hat,
            log10_mass + 0.5 * step_size,
            gamma_minus_one + 0.5 * first,
            log10_ejecta_mass,
            factor,
            thermal_fraction,
        )
        third = _rk_increment(
            g_hat,
            log10_mass + 0.5 * step_size,
            gamma_minus_one + 0.5 * second,
            log10_ejecta_mass,
            factor,
            thermal_fraction,
        )
        fourth = _rk_increment(
            g_hat,
            log10_mass + step_size,
            gamma_minus_one + third,
            log10_ejecta_mass,
            factor,
            thermal_fraction,
        )
        next_gamma_minus_one = (
            gamma_minus_one
            + (first + 2.0 * (second + third) + fourth) / 6.0
            + _NATIVE_GAMMA_INCREMENT
        )
        output = (1.0 + gamma_minus_one, gamma_minus_one, log10_mass, g_hat)
        return (next_gamma_minus_one, log10_mass + step_size), output

    _, history = lax.scan(
        step, (gamma_minus_one_initial, log10_mass_initial), None, length=steps
    )
    gamma, gamma_minus_one, log10_mass, g_hat = history
    return gamma.T, gamma_minus_one.T, log10_mass.T, g_hat.T
