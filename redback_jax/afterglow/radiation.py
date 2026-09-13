"""Overflow-safe synchrotron radiation for native Redback afterglows."""

import math
from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.special import logsumexp

from redback_jax.constants import (
    electron_mass,
    proton_mass,
    qe,
    sigma_T,
    speed_of_light,
)

_FOUR_PI = 4.0 * math.pi


class ShockState(NamedTuple):
    beta: jnp.ndarray
    log10_electrons: jnp.ndarray
    solid_angle: jnp.ndarray
    radius: jnp.ndarray
    log10_magnetic_field: jnp.ndarray
    log10_gamma_minimum: jnp.ndarray
    log10_nu_minimum_prime: jnp.ndarray
    log10_peak_power: jnp.ndarray
    log10_electron_energy: jnp.ndarray


class ObserverState(NamedTuple):
    log10_blackbody_factor: jnp.ndarray
    log10_peak_flux: jnp.ndarray
    log10_nu_cooling: jnp.ndarray
    log10_nu_minimum: jnp.ndarray
    observer_time: jnp.ndarray


@partial(jit, static_argnames=("expansion", "resolution"))
def forward_shock_state(
    gamma,
    gamma_minus_one,
    log10_swept_mass,
    electron_index,
    spectral_peak,
    peak_flux_factor,
    epsilon_b,
    epsilon_e,
    log10_density,
    density_index,
    theta,
    adiabatic_index,
    azimuth_step,
    latitude_step,
    accelerated_fraction=1.0,
    expansion=False,
    expansion_index=1.0,
    resolution=50,
    radius_override=None,
    log10_local_density_override=None,
):
    """Calculate shock and synchrotron state for one polar jet ring."""
    beta = jnp.sqrt(gamma_minus_one * (2.0 + gamma_minus_one)) / (1.0 + gamma_minus_one)
    log10_electrons = log10_swept_mass - math.log10(proton_mass)

    sound_speed = speed_of_light * jnp.sqrt(
        adiabatic_index
        * (adiabatic_index - 1.0)
        * gamma_minus_one
        / (1.0 + adiabatic_index * gamma_minus_one)
    )
    expansion_angle = jnp.arcsin(
        jnp.clip(sound_speed / (speed_of_light * jnp.sqrt(gamma**2 - 1.0)), 0.0, 1.0)
    )
    angular_growth = expansion_angle / gamma ** (expansion_index + 1.0)
    base_solid_angle = azimuth_step * (
        jnp.cos(theta - 0.5 * latitude_step) - jnp.cos(theta + 0.5 * latitude_step)
    )
    expanded_solid_angle = azimuth_step * (
        jnp.cos(theta - 0.5 * latitude_step)
        - jnp.cos(theta + 0.5 * latitude_step + angular_growth / resolution)
    )
    solid_angle = jnp.where(expansion, expanded_solid_angle, base_solid_angle)

    if radius_override is None:
        log10_radius_base = (
            jnp.log10(3.0 - density_index)
            + log10_electrons
            - math.log10(_FOUR_PI)
            - log10_density
        ) / (3.0 - density_index)
        radius_base = jnp.power(10.0, log10_radius_base)
        relative_expansion = (
            (1.0 - jnp.cos(latitude_step + angular_growth[0]))
            / (1.0 - jnp.cos(latitude_step + angular_growth))
        ) ** (0.5 / (3.0 - density_index))
        radius_increment = jnp.diff(radius_base, prepend=0.0)
        radius = jnp.cumsum(
            jnp.where(
                expansion, radius_increment * relative_expansion, radius_increment
            )
        )
        log10_local_density = log10_density - density_index * jnp.log10(radius)
    else:
        radius = radius_override
        log10_local_density = log10_local_density_override
    shock_factor = (
        (adiabatic_index * gamma + 1.0) / (adiabatic_index - 1.0)
    ) * gamma_minus_one
    log10_magnetic_field = 0.5 * (
        math.log10(8.0 * math.pi)
        + jnp.log10(epsilon_b)
        + log10_local_density
        + math.log10(proton_mass)
        + 2.0 * math.log10(speed_of_light)
        + jnp.log10(shock_factor)
    )
    log10_gamma_maximum = 0.5 * (
        math.log10(1.5 * _FOUR_PI * qe / sigma_T) - log10_magnetic_field
    )
    log10_base_gamma = (
        jnp.log10(epsilon_e / accelerated_fraction)
        + jnp.log10(gamma_minus_one)
        + math.log10(proton_mass / electron_mass)
    )

    def above_two(_):
        return log10_base_gamma + jnp.log10(
            (electron_index - 2.0) / (electron_index - 1.0)
        )

    def equal_two(_):
        coefficient = 1.0 / (math.log(10.0) * (log10_gamma_maximum - log10_base_gamma))
        return log10_base_gamma + jnp.log10(coefficient)

    def below_two(_):
        value = (
            jnp.log10((2.0 - electron_index) / (electron_index - 1.0))
            + log10_base_gamma
            + (electron_index - 2.0) * log10_gamma_maximum
        ) / (electron_index - 1.0)
        return value

    log10_gamma_minimum = lax.cond(
        electron_index > 2.0,
        above_two,
        lambda _: lax.cond(electron_index == 2.0, equal_two, below_two, None),
        None,
    )
    log10_nu_minimum_prime = (
        jnp.log10(
            3.0 * spectral_peak * qe / (_FOUR_PI * electron_mass * speed_of_light)
        )
        + 2.0 * log10_gamma_minimum
        + log10_magnetic_field
    )
    log10_peak_power = (
        jnp.log10(accelerated_fraction * peak_flux_factor)
        + math.log10(electron_mass * speed_of_light**2 * sigma_T / (3.0 * qe))
        + log10_magnetic_field
    )
    log10_electron_energy = log10_gamma_minimum + math.log10(
        electron_mass * speed_of_light**2
    )
    return ShockState(
        beta,
        log10_electrons,
        solid_angle,
        radius,
        log10_magnetic_field,
        log10_gamma_minimum,
        log10_nu_minimum_prime,
        log10_peak_power,
        log10_electron_energy,
    )


@jit
def observer_state(
    log10_distance,
    initial_solid_angle,
    azimuth_step,
    observer_angle_value,
    gamma,
    shock,
):
    """Transform a forward-shock state to one observer-facing angular patch."""
    cosine = jnp.cos(observer_angle_value)
    radius_increment = jnp.diff(shock.radius, prepend=0.0)
    observer_time = jnp.cumsum(
        radius_increment * (1.0 / shock.beta - cosine) / speed_of_light
    )
    on_axis_time = jnp.cumsum(
        radius_increment * (1.0 / shock.beta - 1.0) / speed_of_light
    )
    log10_doppler = -jnp.log10(gamma * (1.0 - shock.beta * cosine))
    log10_gamma_cooling = (
        math.log10(6.0 * math.pi * electron_mass * speed_of_light / sigma_T)
        - jnp.log10(gamma)
        - 2.0 * shock.log10_magnetic_field
        - jnp.log10(on_axis_time)
    )
    log10_nu_cooling = (
        log10_doppler
        + math.log10(0.286 * 3.0 * qe / (_FOUR_PI * electron_mass * speed_of_light))
        + 2.0 * log10_gamma_cooling
        + shock.log10_magnetic_field
    )
    log10_nu_minimum = log10_doppler + shock.log10_nu_minimum_prime
    log10_peak_flux = (
        jnp.log10(initial_solid_angle)
        + shock.log10_electrons
        - math.log10(_FOUR_PI)
        + shock.log10_peak_power
        + 3.0 * log10_doppler
        - math.log10(_FOUR_PI)
        - 2.0 * log10_distance
    )
    visible_solid_angle = jnp.maximum(initial_solid_angle, shock.solid_angle)
    cosine_edge = 1.0 - visible_solid_angle / azimuth_step
    log10_blackbody_factor = (
        jnp.log10(2.0 * visible_solid_angle * cosine_edge)
        + log10_doppler
        + shock.log10_electron_energy
        + 2.0 * jnp.log10(shock.radius)
        - 2.0 * math.log10(speed_of_light)
        - 2.0 * log10_distance
    )
    return ObserverState(
        log10_blackbody_factor,
        log10_peak_flux,
        log10_nu_cooling,
        log10_nu_minimum,
        observer_time,
    )


def _synchrotron_branches(observer, log10_frequency, electron_index):
    log_nuc = observer.log10_nu_cooling
    log_num = observer.log10_nu_minimum
    fast = log_nuc < log_num
    fast_low = observer.log10_peak_flux + (log10_frequency - log_nuc) / 3.0
    fast_mid = observer.log10_peak_flux - 0.5 * (log10_frequency - log_nuc)
    fast_high = (
        observer.log10_peak_flux
        - 0.5 * (log_num - log_nuc)
        - 0.5 * electron_index * (log10_frequency - log_num)
    )
    slow_low = observer.log10_peak_flux + (log10_frequency - log_num) / 3.0
    slow_mid = observer.log10_peak_flux - 0.5 * (electron_index - 1.0) * (
        log10_frequency - log_num
    )
    slow_high = (
        observer.log10_peak_flux
        - 0.5 * (electron_index - 1.0) * (log_nuc - log_num)
        - 0.5 * electron_index * (log10_frequency - log_nuc)
    )
    return fast, (fast_low, fast_mid, fast_high), (slow_low, slow_mid, slow_high)


@jit
def optically_thin_synchrotron_log_flux(observer, log10_frequency, electron_index):
    """Return the native sharp synchrotron spectrum without self-absorption."""
    fast, fast_branches, slow_branches = _synchrotron_branches(
        observer, log10_frequency, electron_index
    )
    fast_low, fast_mid, fast_high = fast_branches
    slow_low, slow_mid, slow_high = slow_branches
    log_nuc = observer.log10_nu_cooling
    log_num = observer.log10_nu_minimum
    fast_flux = jnp.where(
        log10_frequency < log_nuc,
        fast_low,
        jnp.where(log10_frequency < log_num, fast_mid, fast_high),
    )
    slow_flux = jnp.where(
        log10_frequency < log_num,
        slow_low,
        jnp.where(log10_frequency < log_nuc, slow_mid, slow_high),
    )
    return jnp.where(fast, fast_flux, slow_flux)


@jit
def synchrotron_log_flux(observer, log10_frequency, electron_index):
    """Return log10 flux density including native Redback self-absorption."""
    thin_flux = optically_thin_synchrotron_log_flux(
        observer, log10_frequency, electron_index
    )
    log_num = observer.log10_nu_minimum
    optically_thick = (
        observer.log10_blackbody_factor
        + 2.0 * log10_frequency
        + jnp.maximum(0.0, 0.5 * (log10_frequency - log_num))
    )
    return jnp.minimum(thin_flux, optically_thick)


def _smooth_minimum_log10(values, smoothness):
    scale = -smoothness * math.log(10.0)
    return logsumexp(scale * jnp.stack(values), axis=0) / scale


@jit
def smooth_synchrotron_log_flux(observer, log10_frequency, electron_index, smoothness):
    """Return a differentiably joined synchrotron spectrum with absorption.

    ``smoothness`` controls all spectral joins. Values around 5--20 approach
    the native sharp broken power law, while smaller values produce broader
    transitions. This is a generic smooth-envelope prescription rather than a
    calibrated Granot--Sari spectrum.
    """
    fast, fast_branches, slow_branches = _synchrotron_branches(
        observer, log10_frequency, electron_index
    )
    fast_flux = _smooth_minimum_log10(fast_branches, smoothness)
    slow_flux = _smooth_minimum_log10(slow_branches, smoothness)
    thin_flux = jnp.where(fast, fast_flux, slow_flux)
    optically_thick = (
        observer.log10_blackbody_factor
        + 2.0 * log10_frequency
        + jnp.maximum(0.0, 0.5 * (log10_frequency - observer.log10_nu_minimum))
    )
    return _smooth_minimum_log10((thin_flux, optically_thick), smoothness)


@jit
def legacy_radiation_prescription(
    observer, log10_frequency, electron_index, parameters
):
    """Callable adapter for the exact native Redback radiation prescription."""
    del parameters
    return synchrotron_log_flux(observer, log10_frequency, electron_index)


@jit
def optically_thin_radiation_prescription(
    observer, log10_frequency, electron_index, parameters
):
    """Callable adapter for synchrotron emission without self-absorption."""
    del parameters
    return optically_thin_synchrotron_log_flux(
        observer, log10_frequency, electron_index
    )


@jit
def smooth_synchrotron_radiation_prescription(
    observer, log10_frequency, electron_index, parameters
):
    """Callable adapter for smooth synchrotron; parameters are ``(s,)``."""
    (smoothness,) = parameters
    return smooth_synchrotron_log_flux(
        observer, log10_frequency, electron_index, smoothness
    )
