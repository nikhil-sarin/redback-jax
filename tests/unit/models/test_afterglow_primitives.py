"""Tests for the composable native-Redback afterglow primitives."""

import jax.numpy as jnp
import numpy as np

from redback_jax.afterglow import (
    angular_mesh,
    jet_structure,
    legacy_impulsive_dynamics,
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


def test_legacy_dynamics_matches_native_redback_fixture():
    gamma, log10_mass, adiabatic_index = legacy_impulsive_dynamics(
        jnp.array([100.0, 30.0]), jnp.array([52.0, 50.0]), 0.0, steps=64)
    indices = jnp.array([0, 16, 32, 48, 63])
    expected_gamma = np.array([
        [100.0, 99.99999997347332, 29.20047522664234, 1.00000024243586, 1.0],
        [30.0, 29.99999992884352, 3.52505486784157, 1.00000000444406, 1.0],
    ])
    expected_log10_mass = np.array(
        [[6.84548669967853, 17.34548669967854, 27.84548669967854,
          38.34548669967854, 48.18923669967854]] * 2)
    expected_adiabatic_index = np.array([
        [1.33336797614664, 1.33336797614666, 1.33388535370789,
         1.66666639293234, 1.66666666666666],
        [1.33385565022863, 1.33385565023117, 1.36110806617093,
         1.66666666164909, 1.66666666666666],
    ])
    np.testing.assert_allclose(gamma[:, indices], expected_gamma, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(log10_mass[:, indices], expected_log10_mass, rtol=2e-6)
    np.testing.assert_allclose(adiabatic_index[:, indices], expected_adiabatic_index,
                               rtol=2e-6)
