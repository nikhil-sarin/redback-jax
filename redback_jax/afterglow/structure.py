"""Native Redback jet angular structures."""

from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.scipy.special import erf


@partial(jit, static_argnames=("kind",))
def jet_structure(
    theta,
    gamma_core,
    energy_core,
    theta_core,
    theta_jet,
    kind,
    structure_energy=0.01,
    structure_gamma=0.5,
):
    """Return Lorentz factor and energy for a native Redback jet structure."""
    floor = 1.0 + 1.0e-12

    if kind == "tophat":
        edge = jnp.minimum(theta_core, theta_jet)
        factor = 0.5 * erf(-(theta - edge) * 1000.0) + 0.5
        return (gamma_core - 1.0) * factor + floor, energy_core * factor

    if kind == "gaussian":
        factor = jnp.exp(-0.5 * (theta / theta_core) ** 2)
        return (gamma_core - 1.0) * factor + floor, energy_core * factor

    if kind == "powerlaw":
        ratio = jnp.where(theta >= theta_core, theta_core / theta, 1.0)
        gamma = (gamma_core - 1.0) * ratio**structure_gamma + floor
        energy = energy_core * ratio**structure_energy
        return gamma, energy

    if kind == "alternative_powerlaw":
        factor = jnp.sqrt(1.0 + (theta / theta_core) ** 2)
        gamma = (gamma_core - 1.0) * factor ** (-structure_gamma) + floor
        energy = energy_core * factor ** (-structure_energy)
        return gamma, energy

    if kind == "two_component":
        outside = theta > theta_core
        gamma = jnp.where(outside, structure_gamma, gamma_core)
        energy = jnp.where(outside, energy_core * structure_energy, energy_core)
        return gamma, energy

    if kind == "double_gaussian":
        core = jnp.exp(-0.5 * (theta / theta_core) ** 2)
        wing = jnp.exp(-0.5 * (theta / theta_jet) ** 2)
        numerator = (1.0 - structure_energy) * core + structure_energy * wing
        denominator = (1.0 - structure_energy / structure_gamma) * core + (
            structure_energy / structure_gamma
        ) * wing
        factor = jnp.where(denominator > 0.0, numerator / denominator, 0.0)
        gamma = jnp.where(
            jnp.isfinite(factor), (gamma_core - 1.0) * factor + floor, structure_gamma
        )
        return gamma, energy_core * numerator

    raise ValueError(f"Unknown jet structure: {kind}")
