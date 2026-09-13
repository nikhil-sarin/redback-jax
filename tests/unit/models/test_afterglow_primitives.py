"""Tests for the composable native-Redback afterglow primitives."""

import jax.numpy as jnp
import numpy as np

from redback_jax.afterglow import (
    angular_mesh,
    jet_structure,
    legacy_impulsive_dynamics,
    observer_angle,
    observer_state,
    power_law_density,
    swept_mass_derivative,
    forward_shock_state,
    synchrotron_log_flux,
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
    gamma, gamma_minus_one, log10_mass, adiabatic_index = legacy_impulsive_dynamics(
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
    assert bool(jnp.all(gamma_minus_one > 0.0))


def test_synchrotron_pipeline_matches_native_redback_fixture():
    gamma, gamma_minus_one, log10_mass, adiabatic_index = legacy_impulsive_dynamics(
        jnp.array([100.0]), jnp.array([52.0]), 0.0, steps=64)
    solid_angle, theta, phi = angular_mesh(0.4, resolution=16)
    shock = forward_shock_state(
        gamma[0], gamma_minus_one[0], log10_mass[0], 2.2, 0.579, 0.612, 0.01, 0.1,
        0.0, 0.0, theta[0], adiabatic_index[0], 2 * jnp.pi / 16,
        0.4 / 16, expansion=False, resolution=16)
    observed = observer_state(28.0, solid_angle[0], 2 * jnp.pi / 16,
                              0.1, gamma[0], shock)
    log10_flux = synchrotron_log_flux(observed, 9.0, 2.2)
    indices = jnp.array([0, 16, 32, 48, 63])

    expected_radius = np.array(
        [1e10, 3.1622776601683746e13, 1e17, 3.162277660168373e20,
         6.042963902381311e23])
    expected_magnetic_field = np.array(
        [3.8823333655498216, 3.882333364518495, 1.1290886985345698,
         1.9141157541353547e-5, 1.9212599146843117e-9])
    expected_peak_flux = np.array(
        [2.275757932523929e-52, 7.196578473673764e-42,
         1.965836062698856e-30, 4.567084217905911e-27,
         3.1923046972748673e-21])
    expected_observer_time = np.array(
        [0.001683110572595627, 5.322462963325076, 17671.424078218985,
         8.748192555398361e12, 1.9118792102454513e20])
    expected_flux = np.array(
        [4.094492471322591e-54, 1.294792208078938e-43,
         8.459652440984453e-32, 1.2732466486735169e-37,
         8.996039391362757e-44])

    np.testing.assert_allclose(shock.radius[indices], expected_radius, rtol=3e-5)
    np.testing.assert_allclose(
        shock.log10_magnetic_field[indices], np.log10(expected_magnetic_field),
        rtol=0.0, atol=0.01)
    np.testing.assert_allclose(
        observed.log10_peak_flux[indices], np.log10(expected_peak_flux),
        rtol=0.0, atol=0.015)
    np.testing.assert_allclose(jnp.log10(observed.observer_time[indices]),
                               np.log10(expected_observer_time), rtol=0.0, atol=0.015)
    np.testing.assert_allclose(log10_flux[indices], np.log10(expected_flux),
                               rtol=0.0, atol=0.05)
