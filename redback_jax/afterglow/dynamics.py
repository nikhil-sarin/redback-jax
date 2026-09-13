"""Blast-wave dynamics for the native Redback afterglow family."""

import math
from functools import partial

import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.special import logsumexp

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


def _log10_add(log10_a, log10_b):
    maximum = jnp.maximum(log10_a, log10_b)
    return maximum + jnp.log10(
        jnp.power(10.0, log10_a - maximum) + jnp.power(10.0, log10_b - maximum)
    )


@partial(jit, static_argnames=("steps",))
def legacy_refreshed_dynamics(
    gamma_initial,
    gamma_injection,
    log10_energy_initial,
    log10_energy_maximum,
    injection_index,
    log10_density,
    density_index=0.0,
    thermal_fraction=0.0,
    steps=250,
):
    """Reproduce Redback's refreshed-shell dynamics in scaled variables."""
    gamma_initial = jnp.atleast_1d(gamma_initial)
    log10_energy_initial = jnp.broadcast_to(log10_energy_initial, gamma_initial.shape)
    log10_energy_maximum = jnp.broadcast_to(log10_energy_maximum, gamma_initial.shape)
    gamma_minus_one_initial = gamma_initial - 1.0
    log10_ejecta_mass = (
        log10_energy_initial - jnp.log10(gamma_initial) - _LOG10_C_SQUARED
    )
    radial_power = 3.0 - density_index
    # The native refreshed implementation uses 4 pi / 3 for both k=0 and k=2.
    log10_mass_initial = (
        _LOG10_FOUR_PI
        - math.log10(3.0)
        + log10_density
        + radial_power * 10.0
        + _LOG10_PROTON_MASS
    )
    log10_mass_initial = jnp.broadcast_to(log10_mass_initial, gamma_initial.shape)
    step_size = radial_power * (24.0 - 10.0) / steps
    factor = -step_size * math.log(10.0)
    beta_injection = jnp.sqrt(gamma_injection**2 - 1.0) / gamma_injection

    def step(carry, _):
        gamma_minus_one, log10_mass, log10_ejecta, log10_energy = carry
        g_hat = _adiabatic_index(gamma_minus_one)
        first = _rk_increment(
            g_hat, log10_mass, gamma_minus_one, log10_ejecta, factor, thermal_fraction
        )
        second = _rk_increment(
            g_hat,
            log10_mass + 0.5 * step_size,
            gamma_minus_one + 0.5 * first,
            log10_ejecta,
            factor,
            thermal_fraction,
        )
        third = _rk_increment(
            g_hat,
            log10_mass + 0.5 * step_size,
            gamma_minus_one + 0.5 * second,
            log10_ejecta,
            factor,
            thermal_fraction,
        )
        fourth = _rk_increment(
            g_hat,
            log10_mass + step_size,
            gamma_minus_one + third,
            log10_ejecta,
            factor,
            thermal_fraction,
        )
        next_u = gamma_minus_one + (first + 2.0 * (second + third) + fourth) / 6.0
        next_gamma = 1.0 + next_u
        beta_next = jnp.sqrt(next_u * (2.0 + next_u)) / next_gamma
        candidate_energy = log10_energy_initial - injection_index * jnp.log10(
            beta_next / beta_injection
        )
        next_energy = jnp.minimum(candidate_energy, log10_energy_maximum)
        inject = next_gamma <= gamma_injection
        next_energy = jnp.where(inject, next_energy, log10_energy)
        energy_fraction = jnp.clip(
            1.0 - jnp.power(10.0, log10_energy - next_energy), 0.0, 1.0
        )
        log10_delta_energy = next_energy + jnp.log10(energy_fraction)
        log10_added_mass = (
            log10_delta_energy - jnp.log10(1.0 + gamma_minus_one) - _LOG10_C_SQUARED
        )
        next_ejecta = jnp.where(
            inject & (next_energy > log10_energy),
            _log10_add(log10_ejecta, log10_added_mass),
            log10_ejecta,
        )
        output = (1.0 + gamma_minus_one, gamma_minus_one, log10_mass, g_hat)
        return (next_u, log10_mass + step_size, next_ejecta, next_energy), output

    initial = (
        gamma_minus_one_initial,
        log10_mass_initial,
        log10_ejecta_mass,
        log10_energy_initial,
    )
    _, history = lax.scan(step, initial, None, length=steps)
    gamma, gamma_minus_one, log10_mass, g_hat = history
    return gamma.T, gamma_minus_one.T, log10_mass.T, g_hat.T


@partial(jit, static_argnames=("density_function", "steps"))
def arbitrary_csm_impulsive_dynamics(
    gamma_initial,
    log10_energy,
    density_parameters,
    log10_swept_mass_initial,
    density_function,
    thermal_fraction=0.0,
    log10_radius_minimum=10.0,
    log10_radius_maximum=24.0,
    steps=250,
):
    """Evolve an impulsive blast wave through an arbitrary radial CSM.

    ``density_function(log10_radius, density_parameters)`` must return
    ``log10(number density / cm^-3)``. The initial enclosed swept mass is in
    grams. A static callable keeps the function JIT-compatible while its
    parameter PyTree remains differentiable.
    """
    gamma_initial = jnp.atleast_1d(gamma_initial)
    log10_energy = jnp.broadcast_to(log10_energy, gamma_initial.shape)
    gamma_minus_one_initial = gamma_initial - 1.0
    log10_ejecta_mass = log10_energy - jnp.log10(gamma_initial) - _LOG10_C_SQUARED

    radius_edges = jnp.linspace(log10_radius_minimum, log10_radius_maximum, steps + 1)
    radius_midpoints = 0.5 * (radius_edges[:-1] + radius_edges[1:])
    log10_density_edges = density_function(radius_edges, density_parameters)
    log10_density_midpoints = density_function(radius_midpoints, density_parameters)
    radial_step = (log10_radius_maximum - log10_radius_minimum) / steps
    integration_constant = (
        _LOG10_FOUR_PI
        + _LOG10_PROTON_MASS
        + math.log10(math.log(10.0))
        + math.log10(radial_step / 6.0)
    )
    edge_integrand = log10_density_edges + 3.0 * radius_edges
    stacked_integrands = jnp.stack(
        (
            edge_integrand[:-1],
            log10_density_midpoints + 3.0 * radius_midpoints + math.log10(4.0),
            edge_integrand[1:],
        )
    )
    log10_shell_mass = integration_constant + logsumexp(
        stacked_integrands * math.log(10.0), axis=0
    ) / math.log(10.0)

    def accumulate(log10_mass, log10_increment):
        next_mass = _log10_add(log10_mass, log10_increment)
        return next_mass, log10_mass

    _, log10_mass_grid = lax.scan(
        accumulate, log10_swept_mass_initial, log10_shell_mass
    )
    mass_steps = jnp.diff(
        jnp.concatenate(
            (
                log10_mass_grid,
                jnp.asarray([_log10_add(log10_mass_grid[-1], log10_shell_mass[-1])]),
            )
        )
    )
    factor_steps = -mass_steps * math.log(10.0)

    def dynamical_step(gamma_minus_one, inputs):
        log10_mass, mass_step, factor = inputs
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
            log10_mass + 0.5 * mass_step,
            gamma_minus_one + 0.5 * first,
            log10_ejecta_mass,
            factor,
            thermal_fraction,
        )
        third = _rk_increment(
            g_hat,
            log10_mass + 0.5 * mass_step,
            gamma_minus_one + 0.5 * second,
            log10_ejecta_mass,
            factor,
            thermal_fraction,
        )
        fourth = _rk_increment(
            g_hat,
            log10_mass + mass_step,
            gamma_minus_one + third,
            log10_ejecta_mass,
            factor,
            thermal_fraction,
        )
        next_u = (
            gamma_minus_one
            + (first + 2.0 * (second + third) + fourth) / 6.0
            + _NATIVE_GAMMA_INCREMENT
        )
        return next_u, (1.0 + gamma_minus_one, gamma_minus_one, g_hat)

    _, history = lax.scan(
        dynamical_step,
        gamma_minus_one_initial,
        (log10_mass_grid, mass_steps, factor_steps),
    )
    gamma, gamma_minus_one, g_hat = history
    radius = jnp.power(10.0, radius_edges[:-1])
    return (
        gamma.T,
        gamma_minus_one.T,
        jnp.broadcast_to(log10_mass_grid, gamma.T.shape),
        g_hat.T,
        radius,
        log10_density_edges[:-1],
    )
