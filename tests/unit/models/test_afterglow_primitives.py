"""Tests for the composable native-Redback afterglow primitives."""

import jax.numpy as jnp
import numpy as np

from redback_jax.afterglow import (
    angular_mesh,
    jet_structure,
    observer_angle,
    power_law_density,
    swept_mass_derivative,
)
from redback_jax.constants import proton_mass


def test_angular_mesh_has_expected_size_and_solid_angle():
    solid_angle, theta, phi = angular_mesh(0.4, resolution=16)
    assert solid_angle.shape == (16**2,)
    assert theta.shape == phi.shape == (16,)
    np.testing.assert_allclose(solid_angle.sum(), 2 * np.pi * (1 - np.cos(0.4)),
                               rtol=1e-6)


def test_observer_angle_on_axis_repeats_latitudes():
    _, theta, phi = angular_mesh(0.4, resolution=8)
    angles = observer_angle(phi, theta, 0.0).reshape(8, 8)
    np.testing.assert_allclose(angles, np.repeat(np.asarray(theta)[:, None], 8, axis=1),
                               atol=2e-6)


def test_all_native_structures_are_finite():
    theta = jnp.linspace(0.01, 0.5, 20)
    kinds = ("tophat", "gaussian", "powerlaw", "alternative_powerlaw",
             "two_component", "double_gaussian")
    for kind in kinds:
        gamma, energy = jet_structure(
            theta, 100.0, 1.0, 0.1, 0.4, kind,
            structure_energy=0.01, structure_gamma=5.0)
        assert gamma.shape == energy.shape == theta.shape
        assert bool(jnp.all(jnp.isfinite(gamma)))
        assert bool(jnp.all(jnp.isfinite(energy)))
        assert bool(jnp.all(gamma >= 1.0))
        assert bool(jnp.all(energy >= 0.0))


def test_general_power_law_medium_and_swept_mass():
    radius = jnp.array([1.0e16, 2.0e16])
    density = power_law_density(radius, 10.0, 1.0e16, 1.5)
    np.testing.assert_allclose(density, [10.0, 10.0 * 2.0**-1.5])
    derivative = swept_mass_derivative(radius, density, 0.1, proton_mass)
    assert bool(jnp.all(derivative > 0.0))
