"""Composed light-curve engine for the native Redback afterglow family."""

import math
from functools import partial

import jax.numpy as jnp
from jax import jit, tree_util, vmap
from jax.scipy.special import logsumexp

from redback_jax.constants import day_to_s

from .dynamics import (
    arbitrary_csm_impulsive_dynamics,
    legacy_impulsive_dynamics,
    legacy_refreshed_dynamics,
    powered_thin_shell_dynamics,
)
from .geometry import angular_mesh, observer_angle
from .radiation import forward_shock_state, observer_state, synchrotron_log_flux
from .structure import jet_structure


def _legacy_power_law_profile(log10_radius, parameters):
    log10_coefficient, density_index = parameters
    return log10_coefficient - density_index * log10_radius


_P_GRID = jnp.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.7, 3.0, 3.2, 3.4])
_SPECTRAL_PEAK = jnp.array(
    [3.0, 1.4, 1.1, 0.86, 0.725, 0.637, 0.579, 0.520, 0.487, 0.451, 0.434, 0.420]
)
_PEAK_FLUX = jnp.array(
    [0.41, 0.44, 0.48, 0.53, 0.56, 0.59, 0.612, 0.630, 0.641, 0.659, 0.660, 0.675]
)


def _log10_linear_interpolate(x, xp, log10_yp):
    """Linear interpolation of positive values represented in log10 space."""
    upper = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, xp.size - 1)
    lower = upper - 1
    weight = jnp.clip((x - xp[lower]) / (xp[upper] - xp[lower]), 0.0, 1.0)
    low = log10_yp[lower]
    high = log10_yp[upper]
    scale = jnp.maximum(low, high)
    mixed = scale + jnp.log10(
        (1.0 - weight) * jnp.power(10.0, low - scale)
        + weight * jnp.power(10.0, high - scale)
    )
    return jnp.where(
        x <= xp[0], log10_yp[0], jnp.where(x >= xp[-1], log10_yp[-1], mixed)
    )


@partial(
    jit,
    static_argnames=(
        "structure_kind",
        "resolution",
        "steps",
        "expansion",
        "refreshed",
        "density_function",
        "engine_function",
    ),
)
def native_afterglow_flux_density(
    time,
    frequency,
    redshift,
    theta_observer,
    log10_energy,
    theta_core,
    theta_jet,
    log10_density,
    electron_index,
    log10_epsilon_e,
    log10_epsilon_b,
    gamma_initial,
    accelerated_fraction,
    log10_luminosity_distance,
    structure_kind="tophat",
    structure_energy=0.01,
    structure_gamma=0.5,
    density_index=0.0,
    expansion=True,
    expansion_index=1.0,
    resolution=50,
    steps=250,
    refreshed=False,
    gamma_injection=2.0,
    energy_factor=1.0,
    injection_index=0.0,
    density_function=None,
    density_parameters=None,
    log10_swept_mass_initial=None,
    engine_function=None,
    engine_parameters=None,
    gamma_engine=1000.0,
):
    """Evaluate a native Redback-style afterglow in mJy.

    ``time`` is observer-frame days and ``frequency`` is observer-frame Hz.
    The luminosity distance is supplied as ``log10(cm)`` to remain float32-safe.
    """
    time = jnp.atleast_1d(time)
    frequency = jnp.broadcast_to(frequency, time.shape)
    solid_angle, theta, phi = angular_mesh(theta_jet, resolution=resolution)
    gamma_ring, energy_fraction = jet_structure(
        theta,
        gamma_initial,
        1.0,
        theta_core,
        theta_jet,
        structure_kind,
        structure_energy,
        structure_gamma,
    )
    ring_log10_energy = log10_energy + jnp.log10(energy_fraction)
    legacy_log10_density = jnp.where(
        density_index == 2.0, log10_density + math.log10(3.0e35), log10_density
    )
    radius_override = None
    local_density_override = None
    if engine_function is not None:
        if refreshed:
            raise ValueError("continuous and refreshed injection cannot be combined")
        active_density_function = (
            _legacy_power_law_profile if density_function is None else density_function
        )
        active_density_parameters = (
            (legacy_log10_density, density_index)
            if density_function is None
            else density_parameters
        )
        if log10_swept_mass_initial is None:
            if density_function is None:
                initial_mass = (
                    math.log10(4.0 * math.pi)
                    - jnp.log10(3.0 - density_index)
                    + legacy_log10_density
                    + (3.0 - density_index) * 10.0
                    + math.log10(1.67262192369e-24)
                )
            else:
                density_at_minimum = density_function(10.0, density_parameters)
                initial_mass = (
                    math.log10(4.0 * math.pi / 3.0)
                    + math.log10(1.67262192369e-24)
                    + density_at_minimum
                    + 30.0
                )
        else:
            initial_mass = log10_swept_mass_initial
        (
            gamma,
            gamma_minus_one,
            log10_mass,
            adiabatic_index,
            radius_override,
            local_density_override,
            _,
            _,
        ) = powered_thin_shell_dynamics(
            gamma_ring,
            ring_log10_energy,
            active_density_parameters,
            initial_mass,
            active_density_function,
            engine_parameters,
            engine_function,
            log10_engine_scale=jnp.log10(energy_fraction),
            gamma_engine=gamma_engine,
            steps=steps,
        )
    elif density_function is not None:
        if refreshed:
            raise ValueError(
                "arbitrary CSM is not yet supported with refreshed dynamics"
            )
        density_at_minimum = density_function(10.0, density_parameters)
        default_initial_mass = (
            math.log10(4.0 * math.pi / 3.0)
            + math.log10(1.67262192369e-24)
            + density_at_minimum
            + 30.0
        )
        initial_mass = (
            default_initial_mass
            if log10_swept_mass_initial is None
            else log10_swept_mass_initial
        )
        (
            gamma,
            gamma_minus_one,
            log10_mass,
            adiabatic_index,
            radius_override,
            local_density_override,
        ) = arbitrary_csm_impulsive_dynamics(
            gamma_ring,
            ring_log10_energy,
            density_parameters,
            initial_mass,
            density_function,
            steps=steps,
        )
    elif refreshed:
        ring_log10_energy_maximum = (
            jnp.log10(energy_factor) + log10_energy + 2.0 * jnp.log10(energy_fraction)
        )
        gamma, gamma_minus_one, log10_mass, adiabatic_index = legacy_refreshed_dynamics(
            gamma_ring,
            gamma_injection,
            ring_log10_energy,
            ring_log10_energy_maximum,
            injection_index,
            legacy_log10_density,
            density_index=density_index,
            steps=steps,
        )
    else:
        gamma, gamma_minus_one, log10_mass, adiabatic_index = legacy_impulsive_dynamics(
            gamma_ring,
            ring_log10_energy,
            legacy_log10_density,
            density_index=density_index,
            steps=steps,
        )

    spectral_peak = jnp.interp(electron_index, _P_GRID, _SPECTRAL_PEAK)
    peak_flux_factor = jnp.interp(electron_index, _P_GRID, _PEAK_FLUX)
    azimuth_step = 2.0 * jnp.pi / resolution
    latitude_step = theta_jet / resolution
    shocks = vmap(
        lambda g, u, lm, th, gh: forward_shock_state(
            g,
            u,
            lm,
            electron_index,
            spectral_peak,
            peak_flux_factor,
            jnp.power(10.0, log10_epsilon_b),
            jnp.power(10.0, log10_epsilon_e),
            legacy_log10_density,
            density_index,
            th,
            gh,
            azimuth_step,
            latitude_step,
            accelerated_fraction,
            expansion,
            expansion_index,
            resolution,
            radius_override,
            local_density_override,
        )
    )(gamma, gamma_minus_one, log10_mass, theta, adiabatic_index)

    patch_shocks = tree_util.tree_map(
        lambda value: jnp.repeat(value, resolution, axis=0), shocks
    )
    patch_gamma = jnp.repeat(gamma, resolution, axis=0)
    patch_angles = observer_angle(phi, theta, theta_observer)
    observers = vmap(observer_state, in_axes=(None, 0, None, 0, 0, 0))(
        log10_luminosity_distance,
        solid_angle,
        azimuth_step,
        patch_angles,
        patch_gamma,
        patch_shocks,
    )

    def one_observation(observer_time_days, observer_frequency):
        source_frequency = observer_frequency * (1.0 + redshift)
        patch_log_flux = vmap(
            lambda state: synchrotron_log_flux(
                state, jnp.log10(source_frequency), electron_index
            )
        )(observers)
        source_time = observer_time_days * day_to_s / (1.0 + redshift)
        interpolated = vmap(
            lambda times, flux: _log10_linear_interpolate(source_time, times, flux)
        )(observers.observer_time, patch_log_flux)
        log10_total = logsumexp(interpolated * math.log(10.0)) / math.log(10.0)
        return jnp.power(10.0, log10_total + 26.0 + jnp.log10(1.0 + redshift))

    return vmap(one_observation)(time, frequency)
