"""Tests for the composable native-Redback afterglow primitives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from redback_jax.afterglow import (
    angular_mesh,
    angular_patch_coordinates,
    arbitrary_csm_impulsive_dynamics,
    constant_engine_cumulative_log10,
    fallback_engine_cumulative_log10,
    forward_shock_state,
    jet_structure,
    legacy_impulsive_dynamics,
    legacy_radiation_prescription,
    legacy_refreshed_dynamics,
    native_afterglow_flux_density,
    observer_angle,
    observer_angle_patches,
    observer_state,
    optically_thin_radiation_prescription,
    power_law_density,
    power_law_log_density,
    powered_thin_shell_dynamics,
    smoothly_broken_power_law_log_density,
    smooth_synchrotron_radiation_prescription,
    swept_mass_derivative,
    synchrotron_log_flux,
    tabulated_engine_cumulative_log10,
    tabulated_log_density,
)
from redback_jax.constants import proton_mass
from redback_jax.models import (
    MODEL_REGISTRY,
    alternativepowerlaw_redback,
    alternativepowerlaw_redback_refreshed,
    doublegaussian_redback,
    doublegaussian_redback_refreshed,
    gaussian_redback,
    gaussian_redback_refreshed,
    powerlaw_redback,
    powerlaw_redback_refreshed,
    tophat_redback,
    tophat_redback_refreshed,
    twocomponent_redback,
    twocomponent_redback_refreshed,
)


def _power_law_profile(log10_radius, parameters):
    return power_law_log_density(log10_radius, *parameters)


def _no_engine(time_seconds, parameters):
    del parameters
    return jnp.full_like(time_seconds, -jnp.inf)


def _custom_gaussian_structure(
    theta, phi, gamma_core, theta_core, theta_jet, parameters
):
    del phi, theta_jet, parameters
    factor = jnp.exp(-0.5 * (theta / theta_core) ** 2)
    return (gamma_core - 1.0) * factor + 1.0 + 1.0e-12, factor


def _azimuthal_gaussian_structure(
    theta, phi, gamma_core, theta_core, theta_jet, parameters
):
    del theta_jet
    asymmetry, phi_peak = parameters
    radial = jnp.exp(-0.5 * (theta / theta_core) ** 2)
    energy = radial * (1.0 + asymmetry * jnp.cos(phi - phi_peak))
    gamma = (gamma_core - 1.0) * radial + 1.0 + 1.0e-12
    return gamma, energy


def _flexible_lightcurve_kwargs():
    return dict(
        time=jnp.array([1.0, 10.0, 100.0]),
        frequency=3.0e9,
        redshift=0.01,
        theta_observer=0.18,
        log10_energy=52.0,
        theta_core=0.1,
        theta_jet=0.4,
        log10_density=0.0,
        electron_index=2.2,
        log10_epsilon_e=-1.0,
        log10_epsilon_b=-2.0,
        gamma_initial=100.0,
        accelerated_fraction=1.0,
        log10_luminosity_distance=jnp.log10(1.3776657447116507e26),
        structure_kind="gaussian",
        expansion=False,
        resolution=6,
        steps=96,
    )


def test_angular_mesh_has_expected_size_and_solid_angle():
    solid_angle, theta, phi = angular_mesh(0.4, resolution=16)
    assert solid_angle.shape == (16**2,)
    assert theta.shape == phi.shape == (16,)
    np.testing.assert_allclose(
        solid_angle.sum(), 2 * np.pi * (1 - np.cos(0.4)), rtol=1e-6
    )


def test_observer_angle_on_axis_repeats_latitudes():
    _, theta, phi = angular_mesh(0.4, resolution=8)
    angles = observer_angle(phi, theta, 0.0).reshape(8, 8)
    np.testing.assert_allclose(
        angles, np.repeat(np.asarray(theta)[:, None], 8, axis=1), atol=2e-6
    )


def test_flat_angular_patch_geometry_matches_tensor_mesh():
    _, theta, phi = angular_mesh(0.4, resolution=8)
    patch_theta, patch_phi = angular_patch_coordinates(theta, phi)
    tensor_angles = observer_angle(phi, theta, 0.2, 0.7)
    flat_angles = observer_angle_patches(patch_phi, patch_theta, 0.2, 0.7)
    np.testing.assert_allclose(flat_angles, tensor_angles)


def test_all_native_structures_are_finite():
    theta = jnp.linspace(0.01, 0.5, 20)
    kinds = (
        "tophat",
        "gaussian",
        "powerlaw",
        "alternative_powerlaw",
        "two_component",
        "double_gaussian",
    )
    for kind in kinds:
        gamma, energy = jet_structure(
            theta,
            100.0,
            1.0,
            0.1,
            0.4,
            kind,
            structure_energy=0.01,
            structure_gamma=5.0,
        )
        assert gamma.shape == energy.shape == theta.shape
        assert bool(jnp.all(jnp.isfinite(gamma)))
        assert bool(jnp.all(jnp.isfinite(energy)))
        assert bool(jnp.all(gamma >= 1.0))
        assert bool(jnp.all(energy >= 0.0))


def test_arbitrary_structure_recovers_builtin_gaussian():
    common = _flexible_lightcurve_kwargs()
    builtin = native_afterglow_flux_density(**common)
    custom = native_afterglow_flux_density(
        **common,
        structure_function=_custom_gaussian_structure,
        structure_parameters=(),
    )
    np.testing.assert_allclose(custom, builtin, rtol=2e-10, atol=0.0)


def test_non_axisymmetric_structure_responds_to_observer_azimuth():
    common = _flexible_lightcurve_kwargs()
    parameters = (0.8, 0.0)
    aligned = native_afterglow_flux_density(
        **common,
        structure_function=_azimuthal_gaussian_structure,
        structure_parameters=parameters,
        phi_observer=0.0,
    )
    opposite = native_afterglow_flux_density(
        **common,
        structure_function=_azimuthal_gaussian_structure,
        structure_parameters=parameters,
        phi_observer=jnp.pi,
    )
    assert float(aligned.sum()) > float(opposite.sum())

    gradient = jax.grad(
        lambda asymmetry: native_afterglow_flux_density(
            **common,
            structure_function=_azimuthal_gaussian_structure,
            structure_parameters=(asymmetry, 0.0),
            phi_observer=0.3,
        ).sum()
    )(0.4)
    assert bool(jnp.isfinite(gradient))


def test_pluggable_radiation_prescriptions_are_finite_and_differentiable():
    common = _flexible_lightcurve_kwargs()
    native = native_afterglow_flux_density(**common)
    adapted = native_afterglow_flux_density(
        **common,
        radiation_function=legacy_radiation_prescription,
        radiation_parameters=(),
    )
    np.testing.assert_allclose(adapted, native, rtol=0.0, atol=0.0)

    optically_thin = native_afterglow_flux_density(
        **common,
        radiation_function=optically_thin_radiation_prescription,
        radiation_parameters=(),
    )
    assert bool(jnp.all(optically_thin >= native))

    smooth_flux = lambda smoothness: native_afterglow_flux_density(
        **common,
        radiation_function=smooth_synchrotron_radiation_prescription,
        radiation_parameters=(smoothness,),
    ).sum()
    assert bool(jnp.isfinite(smooth_flux(8.0)))
    assert bool(jnp.isfinite(jax.grad(smooth_flux)(8.0)))


def test_general_power_law_medium_and_swept_mass():
    radius = jnp.array([1.0e16, 2.0e16])
    density = power_law_density(radius, 10.0, 1.0e16, 1.5)
    np.testing.assert_allclose(density, [10.0, 10.0 * 2.0**-1.5])
    derivative = swept_mass_derivative(radius, density, 0.1, proton_mass)
    assert bool(jnp.all(derivative > 0.0))


def test_flexible_medium_profiles_are_normalized_and_differentiable():
    radii = jnp.array([15.0, 16.0, 17.0])
    smooth = smoothly_broken_power_law_log_density(radii, 3.0, 16.0, 0.5, 2.0, 5.0)
    assert smooth[1] == pytest.approx(3.0)
    assert smooth[0] > smooth[1] > smooth[2]
    tabulated = tabulated_log_density(radii, radii, jnp.array([4.0, 3.0, 1.0]))
    np.testing.assert_allclose(tabulated, [4.0, 3.0, 1.0])
    derivative = jax.grad(
        lambda index: smoothly_broken_power_law_log_density(
            17.0, 3.0, 16.0, 0.5, index, 5.0
        )
    )(2.0)
    assert bool(jnp.isfinite(derivative))


@pytest.mark.parametrize("density_index", [0.0, 1.0, 2.0])
def test_arbitrary_csm_solver_recovers_power_law_legacy_limit(density_index):
    log10_density = 0.0
    initial_mass = (
        np.log10(4.0 * np.pi / (3.0 - density_index))
        + log10_density
        + (3.0 - density_index) * 10.0
        + np.log10(proton_mass)
    )
    generalized = arbitrary_csm_impulsive_dynamics(
        jnp.array([100.0]),
        jnp.array([52.0]),
        (log10_density, 0.0, density_index),
        initial_mass,
        _power_law_profile,
        steps=128,
    )
    legacy = legacy_impulsive_dynamics(
        jnp.array([100.0]),
        jnp.array([52.0]),
        log10_density,
        density_index=density_index,
        steps=128,
    )
    indices = jnp.array([0, 32, 64, 96, 127])
    np.testing.assert_allclose(
        generalized[0][:, indices], legacy[0][:, indices], rtol=2e-3, atol=2e-5
    )
    np.testing.assert_allclose(
        generalized[2][:, indices], legacy[2][:, indices], rtol=0.0, atol=2e-3
    )


def test_arbitrary_csm_lightcurve_recovers_uniform_medium():
    common = dict(
        time=jnp.array([1.0, 10.0, 100.0]),
        frequency=1.0e9,
        redshift=0.01,
        theta_observer=0.05,
        log10_energy=52.0,
        theta_core=0.2,
        theta_jet=0.2,
        log10_density=0.0,
        electron_index=2.2,
        log10_epsilon_e=-1.0,
        log10_epsilon_b=-2.0,
        gamma_initial=100.0,
        accelerated_fraction=1.0,
        log10_luminosity_distance=jnp.log10(1.3776657447116507e26),
        structure_kind="tophat",
        expansion=False,
        resolution=8,
        steps=128,
    )
    legacy = native_afterglow_flux_density(**common)
    generalized = native_afterglow_flux_density(
        **common,
        density_function=_power_law_profile,
        density_parameters=(0.0, 0.0, 0.0),
    )
    np.testing.assert_allclose(generalized, legacy, rtol=0.015)


def _broken_profile(log10_radius, parameters):
    return smoothly_broken_power_law_log_density(log10_radius, *parameters)


def test_broken_csm_lightcurve_is_finite_and_differentiable():
    def total_flux(outer_index):
        return native_afterglow_flux_density(
            time=jnp.array([1.0, 10.0]),
            frequency=3.0e9,
            redshift=0.01,
            theta_observer=0.05,
            log10_energy=52.0,
            theta_core=0.2,
            theta_jet=0.2,
            log10_density=0.0,
            electron_index=2.2,
            log10_epsilon_e=-1.0,
            log10_epsilon_b=-2.0,
            gamma_initial=100.0,
            accelerated_fraction=1.0,
            log10_luminosity_distance=jnp.log10(1.3776657447116507e26),
            structure_kind="tophat",
            expansion=False,
            resolution=6,
            steps=96,
            density_function=_broken_profile,
            density_parameters=(2.0, 17.0, 0.5, outer_index, 5.0),
        ).sum()

    assert bool(jnp.isfinite(total_flux(2.0)))
    assert bool(jnp.isfinite(jax.grad(total_flux)(2.0)))


def test_engine_cumulative_energy_normalization():
    total_energy = 52.0
    duration = 5.0
    constant = constant_engine_cumulative_log10(
        jnp.array([0.0, 0.5e5, 1.0e5, 2.0e5]), (total_energy, duration)
    )
    assert bool(jnp.isneginf(constant[0]))
    np.testing.assert_allclose(constant[1:], total_energy + np.log10([0.5, 1.0, 1.0]))
    fallback = fallback_engine_cumulative_log10(
        jnp.array([0.0, 1.0e5, 1.0e8]), (total_energy, duration)
    )
    assert bool(jnp.isneginf(fallback[0]))
    assert fallback[1] == pytest.approx(total_energy + np.log10(0.4))
    assert fallback[2] < total_energy
    assert fallback[2] > total_energy - 0.01
    tabulated = tabulated_engine_cumulative_log10(
        jnp.array([0.0, 10.0, 100.0]), (jnp.array([1.0, 2.0]), jnp.array([50.0, 52.0]))
    )
    assert bool(jnp.isneginf(tabulated[0]))
    np.testing.assert_allclose(tabulated[1:], [50.0, 52.0])


def test_powered_thin_shell_without_engine_recovers_impulsive_dynamics():
    initial_mass = np.log10(4.0 * np.pi / 3.0) + 30.0 + np.log10(proton_mass)
    generalized = arbitrary_csm_impulsive_dynamics(
        jnp.array([100.0]),
        jnp.array([52.0]),
        (0.0, 0.0, 0.0),
        initial_mass,
        _power_law_profile,
        steps=96,
    )
    powered = powered_thin_shell_dynamics(
        jnp.array([100.0]),
        jnp.array([52.0]),
        (0.0, 0.0, 0.0),
        initial_mass,
        _power_law_profile,
        (),
        _no_engine,
        steps=96,
    )
    indices = jnp.array([0, 24, 48, 72, 95])
    np.testing.assert_allclose(
        powered[0][:, indices], generalized[0][:, indices], rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(powered[7], 52.0, atol=1e-6)


def test_fallback_power_keeps_the_late_shell_faster():
    initial_mass = np.log10(4.0 * np.pi / 3.0) + 30.0 + np.log10(proton_mass)
    no_engine = powered_thin_shell_dynamics(
        jnp.array([30.0]),
        jnp.array([51.0]),
        (0.0, 0.0, 0.0),
        initial_mass,
        _power_law_profile,
        (),
        _no_engine,
        steps=128,
    )
    fallback = powered_thin_shell_dynamics(
        jnp.array([30.0]),
        jnp.array([51.0]),
        (0.0, 0.0, 0.0),
        initial_mass,
        _power_law_profile,
        (52.0, np.log10(30.0 * 86400.0)),
        fallback_engine_cumulative_log10,
        steps=128,
    )
    assert fallback[1][0, 96] > no_engine[1][0, 96]
    assert fallback[7][0, -1] > 51.9


def test_powered_redback_lightcurve_is_delayed_relative_to_impulse():
    common = dict(
        time=jnp.array([1.0, 10.0, 100.0]),
        frequency=3.0e9,
        redshift=0.01,
        theta_observer=0.05,
        theta_core=0.2,
        theta_jet=0.2,
        log10_density=0.0,
        electron_index=2.2,
        log10_epsilon_e=-1.0,
        log10_epsilon_b=-2.0,
        gamma_initial=30.0,
        accelerated_fraction=1.0,
        log10_luminosity_distance=jnp.log10(1.3776657447116507e26),
        structure_kind="tophat",
        expansion=False,
        resolution=6,
        steps=128,
    )
    impulsive = native_afterglow_flux_density(log10_energy=np.log10(1.1e52), **common)
    powered = native_afterglow_flux_density(
        log10_energy=51.0,
        engine_function=fallback_engine_cumulative_log10,
        engine_parameters=(52.0, np.log10(30.0 * 86400.0)),
        **common,
    )
    assert bool(jnp.all(jnp.isfinite(powered)))
    assert bool(jnp.all(powered < impulsive))


def test_powered_public_wrapper_has_engine_timescale_gradient():
    def total_flux(log10_break_time):
        return tophat_redback(
            jnp.array([1.0, 10.0]),
            0.01,
            0.05,
            51.0,
            0.2,
            0.0,
            2.2,
            -1.0,
            -2.0,
            30.0,
            1.0,
            frequency=3.0e9,
            output_format="flux_density",
            expansion=False,
            res=6,
            steps=96,
            engine_function=fallback_engine_cumulative_log10,
            engine_parameters=(52.0, log10_break_time),
        ).sum()

    gradient = jax.grad(total_flux)(np.log10(30.0 * 86400.0))
    assert bool(jnp.isfinite(gradient))


def test_legacy_dynamics_matches_native_redback_fixture():
    gamma, gamma_minus_one, log10_mass, adiabatic_index = legacy_impulsive_dynamics(
        jnp.array([100.0, 30.0]), jnp.array([52.0, 50.0]), 0.0, steps=64
    )
    indices = jnp.array([0, 16, 32, 48, 63])
    expected_gamma = np.array(
        [
            [100.0, 99.99999997347332, 29.20047522664234, 1.00000024243586, 1.0],
            [30.0, 29.99999992884352, 3.52505486784157, 1.00000000444406, 1.0],
        ]
    )
    expected_log10_mass = np.array(
        [
            [
                6.84548669967853,
                17.34548669967854,
                27.84548669967854,
                38.34548669967854,
                48.18923669967854,
            ]
        ]
        * 2
    )
    expected_adiabatic_index = np.array(
        [
            [
                1.33336797614664,
                1.33336797614666,
                1.33388535370789,
                1.66666639293234,
                1.66666666666666,
            ],
            [
                1.33385565022863,
                1.33385565023117,
                1.36110806617093,
                1.66666666164909,
                1.66666666666666,
            ],
        ]
    )
    np.testing.assert_allclose(gamma[:, indices], expected_gamma, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(log10_mass[:, indices], expected_log10_mass, rtol=2e-6)
    np.testing.assert_allclose(
        adiabatic_index[:, indices], expected_adiabatic_index, rtol=2e-6
    )
    assert bool(jnp.all(gamma_minus_one > 0.0))


def test_legacy_refreshed_dynamics_matches_native_redback_fixture():
    gamma, _, log10_mass, adiabatic_index = legacy_refreshed_dynamics(
        jnp.array([100.0]),
        10.0,
        jnp.array([52.0]),
        jnp.array([np.log10(5e52)]),
        3.0,
        0.0,
        steps=64,
    )
    indices = jnp.array([0, 16, 32, 48, 63])
    expected_gamma = np.array(
        [
            [
                100.0,
                99.99999997347332,
                29.200475226642375,
                1.0000005418068327,
                1.0000000000000018,
            ]
        ]
    )
    expected_log10_mass = np.array(
        [
            [
                6.845486699678535,
                17.345486699678535,
                27.845486699678535,
                38.34548669967854,
                48.18923669967854,
            ]
        ]
    )
    expected_adiabatic_index = np.array(
        [
            [
                1.3333679761466402,
                1.3333679761466641,
                1.3338853537078947,
                1.666666054898741,
                1.6666666666666645,
            ]
        ]
    )
    np.testing.assert_allclose(gamma[:, indices], expected_gamma, rtol=3e-5, atol=2e-6)
    np.testing.assert_allclose(log10_mass[:, indices], expected_log10_mass, rtol=2e-6)
    np.testing.assert_allclose(
        adiabatic_index[:, indices], expected_adiabatic_index, rtol=3e-5
    )


def test_synchrotron_pipeline_matches_native_redback_fixture():
    gamma, gamma_minus_one, log10_mass, adiabatic_index = legacy_impulsive_dynamics(
        jnp.array([100.0]), jnp.array([52.0]), 0.0, steps=64
    )
    solid_angle, theta, phi = angular_mesh(0.4, resolution=16)
    shock = forward_shock_state(
        gamma[0],
        gamma_minus_one[0],
        log10_mass[0],
        2.2,
        0.579,
        0.612,
        0.01,
        0.1,
        0.0,
        0.0,
        theta[0],
        adiabatic_index[0],
        2 * jnp.pi / 16,
        0.4 / 16,
        expansion=False,
        resolution=16,
    )
    observed = observer_state(
        28.0, solid_angle[0], 2 * jnp.pi / 16, 0.1, gamma[0], shock
    )
    log10_flux = synchrotron_log_flux(observed, 9.0, 2.2)
    indices = jnp.array([0, 16, 32, 48, 63])

    expected_radius = np.array(
        [1e10, 3.1622776601683746e13, 1e17, 3.162277660168373e20, 6.042963902381311e23]
    )
    expected_magnetic_field = np.array(
        [
            3.8823333655498216,
            3.882333364518495,
            1.1290886985345698,
            1.9141157541353547e-5,
            1.9212599146843117e-9,
        ]
    )
    expected_peak_flux = np.array(
        [
            2.275757932523929e-52,
            7.196578473673764e-42,
            1.965836062698856e-30,
            4.567084217905911e-27,
            3.1923046972748673e-21,
        ]
    )
    expected_observer_time = np.array(
        [
            0.001683110572595627,
            5.322462963325076,
            17671.424078218985,
            8.748192555398361e12,
            1.9118792102454513e20,
        ]
    )
    expected_flux = np.array(
        [
            4.094492471322591e-54,
            1.294792208078938e-43,
            8.459652440984453e-32,
            1.2732466486735169e-37,
            8.996039391362757e-44,
        ]
    )

    np.testing.assert_allclose(shock.radius[indices], expected_radius, rtol=3e-5)
    np.testing.assert_allclose(
        shock.log10_magnetic_field[indices],
        np.log10(expected_magnetic_field),
        rtol=0.0,
        atol=0.01,
    )
    np.testing.assert_allclose(
        observed.log10_peak_flux[indices],
        np.log10(expected_peak_flux),
        rtol=0.0,
        atol=0.015,
    )
    np.testing.assert_allclose(
        jnp.log10(observed.observer_time[indices]),
        np.log10(expected_observer_time),
        rtol=0.0,
        atol=0.015,
    )
    np.testing.assert_allclose(
        log10_flux[indices], np.log10(expected_flux), rtol=0.0, atol=0.05
    )


def test_composed_tophat_lightcurve_matches_native_redback_fixture():
    result = native_afterglow_flux_density(
        time=jnp.array([1.0, 10.0, 100.0]),
        frequency=1.0e9,
        redshift=0.01,
        theta_observer=0.05,
        log10_energy=52.0,
        theta_core=0.2,
        theta_jet=0.2,
        log10_density=0.0,
        electron_index=2.2,
        log10_epsilon_e=-1.0,
        log10_epsilon_b=-2.0,
        gamma_initial=100.0,
        accelerated_fraction=1.0,
        log10_luminosity_distance=np.log10(1.3776657447116507e26),
        structure_kind="tophat",
        expansion=False,
        resolution=8,
        steps=64,
    )
    expected_mjy = np.array([108.08590549, 120.30437189, 195.31447198])
    np.testing.assert_allclose(result, expected_mjy, rtol=0.015)


def test_public_tophat_wrapper_and_registry():
    result = tophat_redback(
        jnp.array([1.0, 10.0, 100.0]),
        0.01,
        0.05,
        52.0,
        0.2,
        0.0,
        2.2,
        -1.0,
        -2.0,
        100.0,
        1.0,
        frequency=1.0e9,
        output_format="flux_density",
        expansion=False,
        res=8,
        steps=64,
    )
    expected_mjy = np.array([108.08590549, 120.30437189, 195.31447198])
    np.testing.assert_allclose(result, expected_mjy, rtol=0.02)
    assert MODEL_REGISTRY["tophat_redback"] is tophat_redback
    for name in (
        "gaussian_redback",
        "twocomponent_redback",
        "powerlaw_redback",
        "alternativepowerlaw_redback",
        "doublegaussian_redback",
    ):
        assert name in MODEL_REGISTRY


@pytest.mark.parametrize(
    ("model", "expected_mjy"),
    [
        (gaussian_redback, [76.75951234, 77.22319264, 79.90381173]),
        (twocomponent_redback, [60.67329546, 50.68955938, 66.58012673]),
        (powerlaw_redback, [94.22838090, 94.27717169, 113.34948292]),
        (alternativepowerlaw_redback, [56.75649553, 58.53103025, 52.94765781]),
        (doublegaussian_redback, [86.57037117, 97.23753895, 111.65166607]),
    ],
)
def test_structured_public_wrappers_match_native_redback(model, expected_mjy):
    result = model(
        jnp.array([1.0, 10.0, 100.0]),
        0.01,
        0.05,
        52.0,
        0.1,
        0.3,
        0.0,
        2.2,
        -1.0,
        -2.0,
        100.0,
        1.0,
        frequency=1.0e9,
        output_format="flux_density",
        expansion=False,
        res=8,
        steps=64,
    )
    np.testing.assert_allclose(result, expected_mjy, rtol=0.025)


@pytest.mark.parametrize(
    ("density_index", "expansion", "expected_mjy"),
    [
        (0.0, True, [295.10217435, 1730.31589495, 29.74114130]),
        (2.0, False, [84.91510751, 1783.73853997, 19.31117045]),
        (2.0, True, [82.70605294, 1686.73641157, 7.32435741]),
    ],
)
def test_wind_and_expansion_match_native_redback(
    density_index, expansion, expected_mjy
):
    result = tophat_redback(
        jnp.array([1.0, 10.0, 100.0]),
        0.01,
        0.05,
        52.0,
        0.2,
        -1.0,
        2.2,
        -1.0,
        -2.0,
        100.0,
        1.0,
        frequency=jnp.array([1.0e9, 3.0e9, 1.0e10]),
        output_format="flux_density",
        k=density_index,
        expansion=expansion,
        res=8,
        steps=64,
    )
    np.testing.assert_allclose(result, expected_mjy, rtol=0.04)


def test_composed_afterglow_has_finite_energy_gradient():
    def total_flux(log10_energy):
        return native_afterglow_flux_density(
            time=jnp.array([1.0, 10.0]),
            frequency=1.0e9,
            redshift=0.01,
            theta_observer=0.05,
            log10_energy=log10_energy,
            theta_core=0.2,
            theta_jet=0.2,
            log10_density=0.0,
            electron_index=2.2,
            log10_epsilon_e=-1.0,
            log10_epsilon_b=-2.0,
            gamma_initial=100.0,
            accelerated_fraction=1.0,
            log10_luminosity_distance=jnp.log10(1.3776657447116507e26),
            structure_kind="tophat",
            expansion=False,
            resolution=6,
            steps=48,
        ).sum()

    assert bool(jnp.isfinite(jax.grad(total_flux)(52.0)))


@pytest.mark.parametrize(
    ("model", "expected_mjy", "extra"),
    [
        (tophat_redback_refreshed, [108.08590549, 120.30437189, 195.31447198], {}),
        (gaussian_redback_refreshed, [105.22857326, 161.12478750, 234.11269432], {}),
        (
            twocomponent_redback_refreshed,
            [105.67508074, 139.66274614, 284.61251216],
            {"ss": 0.5},
        ),
        (powerlaw_redback_refreshed, [115.31731711, 190.76498019, 313.01541507], {}),
        (
            alternativepowerlaw_redback_refreshed,
            [91.46102832, 119.78539057, 161.21755063],
            {},
        ),
        (
            doublegaussian_redback_refreshed,
            [106.69755457, 167.38942391, 235.17995467],
            {},
        ),
    ],
)
def test_refreshed_public_wrappers_match_native_redback(model, expected_mjy, extra):
    common = (jnp.array([1.0, 10.0, 100.0]), 0.01, 0.05, 52.0)
    tail = (0.0, 2.2, -1.0, -2.0, 100.0, 1.0)
    if model is tophat_redback_refreshed:
        args = common + (0.2, 2.0, 10.0, 3.0) + tail
    else:
        args = common + (0.2, 0.3, 2.0, 10.0, 3.0) + tail
    result = model(
        *args,
        frequency=1.0e9,
        output_format="flux_density",
        expansion=False,
        res=8,
        steps=64,
        **extra,
    )
    np.testing.assert_allclose(result, expected_mjy, rtol=0.04)
    assert MODEL_REGISTRY[model.__name__] is model
