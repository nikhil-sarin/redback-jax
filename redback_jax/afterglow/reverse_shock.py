"""Finite-width forward/reverse-shock dynamics.

The equations follow the four-region mechanical model used by VegasAfterglow,
specialized to an unmagnetized upstream ejecta shell. Evolution is performed
on a fixed logarithmic-radius grid using dimensionless masses, energies, and
shell widths. This keeps the solver stable with JAX's default float32 mode.
"""

from __future__ import annotations

import math
from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
from jax import jit, lax, tree_util

from redback_jax.constants import proton_mass, speed_of_light

from .dynamics import _adiabatic_index

_LOG10_FOUR_PI_PROTON_MASS = math.log10(4.0 * math.pi * proton_mass)
_LOG10_C = math.log10(speed_of_light)
_LOG10_C2 = 2.0 * _LOG10_C


class ReverseShockDynamics(NamedTuple):
    """Radial history of a coupled unmagnetized shock pair."""

    bulk_gamma: jnp.ndarray
    relative_gamma: jnp.ndarray
    radius: jnp.ndarray
    observer_time: jnp.ndarray
    comoving_time: jnp.ndarray
    log10_forward_mass: jnp.ndarray
    log10_reverse_mass: jnp.ndarray
    log10_ejecta_mass: jnp.ndarray
    log10_forward_internal_energy: jnp.ndarray
    log10_reverse_internal_energy: jnp.ndarray
    log10_unshocked_width: jnp.ndarray
    log10_shocked_width: jnp.ndarray
    crossing_fraction: jnp.ndarray


def _smoothstep(edge0, edge1, value):
    fraction = jnp.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return fraction**2 * (3.0 - 2.0 * fraction)


def _four_velocity(gamma):
    return jnp.sqrt(jnp.maximum((gamma - 1.0) * (gamma + 1.0), 0.0))


def _beta(gamma):
    return _four_velocity(gamma) / gamma


def _relative_gamma(gamma_upstream, gamma_downstream):
    """Cancellation-safe relative Lorentz factor for co-linear velocities."""
    product_u = _four_velocity(gamma_upstream) * _four_velocity(gamma_downstream)
    difference = gamma_upstream - gamma_downstream
    denominator = gamma_upstream * gamma_downstream - 1.0 + product_u
    return 1.0 + difference**2 / _positive(denominator)


def _thermal_excess(relative_gamma):
    """Regularize the exactly cold state for finite autodiff derivatives."""
    excess = relative_gamma - 1.0
    floor = jnp.asarray(1.0e-12, dtype=relative_gamma.dtype)
    return jnp.where(excess > floor, excess, lax.stop_gradient(floor))


def _sound_speed_over_c(relative_gamma):
    excess = _thermal_excess(relative_gamma)
    adiabatic_index = _adiabatic_index(excess)
    numerator = adiabatic_index * (adiabatic_index - 1.0) * excess
    denominator = 1.0 + adiabatic_index * excess
    # A strictly positive floor avoids the undefined derivative of sqrt(x) at
    # the initially unshocked Gamma_rel == 1 state.
    ratio = jnp.maximum(numerator / denominator, jnp.finfo(relative_gamma.dtype).tiny)
    return jnp.sqrt(ratio)


def _effective_gamma(adiabatic_index, gamma):
    return (adiabatic_index * gamma**2 - adiabatic_index + 1.0) / gamma


def _effective_gamma_derivative(adiabatic_index, gamma):
    return (adiabatic_index * gamma**2 + adiabatic_index - 1.0) / gamma**2


def _positive(value):
    return jnp.maximum(value, jnp.finfo(jnp.result_type(value, 1.0)).tiny)


@partial(jit, static_argnames=("density_function", "steps"))
def unmagnetized_reverse_shock_dynamics(
    gamma_initial,
    log10_energy,
    engine_duration,
    density_parameters,
    log10_swept_mass_initial,
    density_function,
    log10_radius_minimum=10.0,
    log10_radius_maximum=24.0,
    steps=512,
):
    """Evolve a finite ejecta shell and its forward/reverse shock pair.

    Masses are evolved in units of the total ejecta mass, internal energies in
    units of its rest energy, and widths as fractions of radius. Parameters use
    isotropic-equivalent energy and swept mass, a uniform rescaling of the
    per-solid-angle VegasAfterglow equations. ``density_function`` follows the
    afterglow density contract and returns ``log10(number density / cm^-3)``.
    """
    gamma4 = jnp.atleast_1d(gamma_initial)
    log10_energy = jnp.broadcast_to(log10_energy, gamma4.shape)
    duration = jnp.broadcast_to(engine_duration, gamma4.shape)
    log10_ejecta_mass = log10_energy - jnp.log10(gamma4) - _LOG10_C2
    u4 = _four_velocity(gamma4)

    log10_radius_edges = jnp.linspace(
        jnp.asarray(log10_radius_minimum, dtype=gamma4.dtype),
        jnp.asarray(log10_radius_maximum, dtype=gamma4.dtype),
        steps + 1,
    )
    log_radius_edges = log10_radius_edges * math.log(10.0)
    radius_edges = jnp.power(10.0, log10_radius_edges)
    radius0_over_c = jnp.power(10.0, log10_radius_minimum - _LOG10_C)
    observer_time0 = radius0_over_c / _positive(u4 * (gamma4 + u4))
    comoving_time0 = radius0_over_c / _positive(u4)
    injected_time0 = jnp.minimum(observer_time0, duration)
    mu4 = injected_time0 / duration

    sound4 = _sound_speed_over_c(gamma4)
    width4 = (
        u4 * injected_time0 / radius0_over_c
        + jnp.maximum(observer_time0 - duration, 0.0) * gamma4 * sound4 / radius0_over_c
    )
    mu2 = jnp.power(10.0, log10_swept_mass_initial - log10_ejecta_mass)
    gamma = jnp.clip(gamma4 / (1.0 + mu2), 1.0, gamma4)
    q2 = jnp.maximum((gamma - 1.0) * mu2, 0.0)
    gamma34 = _relative_gamma(gamma4, gamma)
    seed_fraction = 1.0e-8
    width3 = width4 * seed_fraction
    mu3 = jnp.minimum(mu4 * (4.0 * gamma34) * seed_fraction, mu4)
    q3 = jnp.maximum((gamma34 - 1.0) * mu3, 0.0)

    initial_state = (
        gamma,
        width4,
        width3,
        mu2,
        mu3,
        q2,
        q3,
        observer_time0,
        comoving_time0,
        mu4,
    )

    gamma_floor = 1.0 + 4.0 * jnp.finfo(gamma4.dtype).eps

    def project(state):
        gamma_value, w4, w3, m2, m3, e2, e3, time, proper_time, m4 = state
        m4 = _positive(m4)
        return (
            jnp.clip(gamma_value, gamma_floor, gamma4),
            _positive(w4),
            _positive(w3),
            _positive(m2),
            jnp.clip(m3, 0.0, m4),
            jnp.maximum(e2, 0.0),
            jnp.maximum(e3, 0.0),
            _positive(time),
            _positive(proper_time),
            m4,
        )

    def derivatives(state, log10_radius):
        state = project(state)
        (
            gamma_value,
            w4,
            w3,
            m2,
            m3,
            q2_value,
            q3_value,
            observer_time,
            _comoving_time,
            m4,
        ) = state
        bulk_u = _positive(_four_velocity(gamma_value))
        radial_factor = bulk_u * (gamma_value + bulk_u)
        radius_over_c = jnp.power(10.0, log10_radius - _LOG10_C)
        dt_dlnr = radius_over_c / radial_factor
        comoving_factor = gamma_value + bulk_u

        log10_density = jnp.asarray(
            density_function(log10_radius, density_parameters), dtype=gamma4.dtype
        )
        dm2_dlnr = jnp.power(
            10.0,
            _LOG10_FOUR_PI_PROTON_MASS
            + log10_density
            + 3.0 * log10_radius
            - log10_ejecta_mass,
        )

        injection_weight = _smoothstep(1.5 * duration, 0.5 * duration, observer_time)
        dm4_dlnr = injection_weight * dt_dlnr / duration

        gamma34_value = _relative_gamma(gamma4, gamma_value)
        compression = 4.0 * gamma34_value
        sound3 = _sound_speed_over_c(gamma34_value)
        dx4_dlnr_over_r = (
            injection_weight * u4 + (1.0 - injection_weight) * sound4 * comoving_factor
        ) / radial_factor
        dw4_dlnr = dx4_dlnr_over_r - w4

        remaining = jnp.maximum(m4 - m3, 0.0)
        crossing_weight = injection_weight + (1.0 - injection_weight) * remaining / m4
        beta3 = _beta(gamma_value)
        penetration = gamma_value * compression / gamma4 - 1.0
        crossing_speed = jnp.abs(
            (gamma4 - gamma_value)
            * (gamma4 + gamma_value)
            * (1.0 + beta3)
            * gamma_value
            / _positive(gamma4**2 * (_beta(gamma4) + beta3) * penetration)
        )
        dx3_dlnr_over_r = (
            crossing_weight * crossing_speed
            + (1.0 - crossing_weight) * sound3 * comoving_factor
        ) / radial_factor
        dw3_dlnr = dx3_dlnr_over_r - w3

        effective_mass = injection_weight * m4 + (1.0 - injection_weight) * remaining
        dm3_raw = effective_mass * compression / w4 * dx3_dlnr_over_r
        cap_weight = _smoothstep(0.0, 1.0, m3 / m4)
        capped_dm3 = (1.0 - cap_weight) * dm3_raw + cap_weight * jnp.minimum(
            dm3_raw, dm4_dlnr
        )
        dm3_dlnr = jnp.where(injection_weight > 1.0e-6, capped_dm3, dm3_raw)
        dm3_dlnr = jnp.where(
            (remaining <= 0.0) & (injection_weight < 1.0e-6), 0.0, dm3_dlnr
        )

        adiabatic2 = _adiabatic_index(gamma_value - 1.0)
        adiabatic3 = _adiabatic_index(_thermal_excess(gamma34_value))
        dlog_volume2 = 3.0 + dw4_dlnr / w4
        dlog_volume3 = 3.0 + dw3_dlnr / w3
        dq2_dlnr = (gamma_value - 1.0) * dm2_dlnr - (
            adiabatic2 - 1.0
        ) * dlog_volume2 * q2_value
        dq3_dlnr = (gamma34_value - 1.0) * dm3_dlnr - (
            adiabatic3 - 1.0
        ) * dlog_volume3 * q3_value

        effective2 = _effective_gamma(adiabatic2, gamma_value)
        effective3 = _effective_gamma(adiabatic3, gamma_value)
        derivative2 = _effective_gamma_derivative(adiabatic2, gamma_value)
        derivative3 = _effective_gamma_derivative(adiabatic3, gamma_value)
        numerator = (
            (gamma_value - 1.0) * dm2_dlnr
            + (gamma_value - gamma4) * dm3_dlnr
            + effective2 * dq2_dlnr
            + effective3 * dq3_dlnr
        )
        denominator = m2 + m3 + derivative2 * q2_value + derivative3 * q3_value
        dgamma_dlnr = -numerator / _positive(denominator)

        return (
            dgamma_dlnr,
            dw4_dlnr,
            dw3_dlnr,
            dm2_dlnr,
            dm3_dlnr,
            dq2_dlnr,
            dq3_dlnr,
            dt_dlnr,
            comoving_factor * dt_dlnr,
            dm4_dlnr,
        )

    def add_scaled(state, derivative, scale):
        return tree_util.tree_map(lambda x, dx: x + scale * dx, state, derivative)

    def rk4_step(state, inputs):
        log10_radius, step_size = inputs
        first = derivatives(state, log10_radius)
        log10_half_step = log10_radius + 0.5 * step_size / math.log(10.0)
        second = derivatives(add_scaled(state, first, 0.5 * step_size), log10_half_step)
        third = derivatives(add_scaled(state, second, 0.5 * step_size), log10_half_step)
        fourth = derivatives(
            add_scaled(state, third, step_size),
            log10_radius + step_size / math.log(10.0),
        )
        increment = tree_util.tree_map(
            lambda a, b, c, d: (a + 2.0 * b + 2.0 * c + d) / 6.0,
            first,
            second,
            third,
            fourth,
        )
        next_state = project(add_scaled(state, increment, step_size))
        return next_state, project(state)

    step_sizes = jnp.diff(log_radius_edges)
    _, history = lax.scan(
        rk4_step, initial_state, (log10_radius_edges[:-1], step_sizes)
    )
    (
        gamma_history,
        width4_history,
        width3_history,
        mass2_history,
        mass3_history,
        internal2_history,
        internal3_history,
        observer_time_history,
        comoving_time_history,
        mass4_history,
    ) = history
    gamma34_history = _relative_gamma(gamma4[None, :], gamma_history)
    crossing_fraction = jnp.clip(mass3_history / mass4_history, 0.0, 1.0)

    def transpose(value):
        return jnp.swapaxes(value, 0, 1)

    radius_history = jnp.broadcast_to(
        radius_edges[:-1], (gamma4.size, radius_edges.size - 1)
    )
    log10_radius_history = jnp.broadcast_to(
        log10_radius_edges[:-1], (gamma4.size, steps)
    )
    log10_mass_scale = log10_ejecta_mass[:, None]
    log10_energy_scale = log10_mass_scale + _LOG10_C2
    return ReverseShockDynamics(
        transpose(gamma_history),
        transpose(gamma34_history),
        radius_history,
        transpose(observer_time_history),
        transpose(comoving_time_history),
        jnp.log10(_positive(transpose(mass2_history))) + log10_mass_scale,
        jnp.log10(_positive(transpose(mass3_history))) + log10_mass_scale,
        jnp.log10(_positive(transpose(mass4_history))) + log10_mass_scale,
        jnp.log10(_positive(transpose(internal2_history))) + log10_energy_scale,
        jnp.log10(_positive(transpose(internal3_history))) + log10_energy_scale,
        jnp.log10(_positive(transpose(width4_history))) + log10_radius_history,
        jnp.log10(_positive(transpose(width3_history))) + log10_radius_history,
        transpose(crossing_fraction),
    )
