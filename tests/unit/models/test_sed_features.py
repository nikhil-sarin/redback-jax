"""
Tests for the sed_features module in redback_jax.models.
"""
import jax.numpy as jnp

import pytest
from redback_jax.models.sed_features import apply_sed_feature, SEDFeatures, NO_SED_FEATURES


class TestSEDFeaturesModule:
    def test_create_sed_features(self):
        """Test creating an SEDFeatures object."""
        feature = SEDFeatures(
            rest_wavelengths=jnp.array([5000.0, 6000.0]),
            sigmas=jnp.array([100.0, 200.0]),
            amplitudes=jnp.array([1.0, 0.5]),
            t_starts=jnp.array([0.0, 50.0]),
            t_ends=jnp.array([100.0, 150.0]),
            rise_times=jnp.array([10.0, 10.0]),
            fall_times=jnp.array([10.0, 10.0]),
        )
        assert jnp.allclose(feature.rest_wavelengths, jnp.array([5000.0, 6000.0]))
        assert jnp.allclose(feature.sigmas, jnp.array([100.0, 200.0]))
        assert jnp.allclose(feature.amplitudes, jnp.array([1.0, 0.5]))
        assert jnp.allclose(feature.t_starts, jnp.array([0.0, 50.0]))
        assert jnp.allclose(feature.t_ends, jnp.array([100.0, 150.0]))
        assert jnp.allclose(feature.rise_times, jnp.array([10.0, 10.0]))
        assert jnp.allclose(feature.fall_times, jnp.array([10.0, 10.0]))

        # We can create a feature from with singleton values.
        feature = SEDFeatures(
            rest_wavelengths=jnp.array([5000.0, 6000.0]),
            sigmas=1000.0,
            amplitudes=jnp.array([1.0, 0.5]),
            t_starts=jnp.array([0.0, 50.0]),
            t_ends=jnp.array([100.0, 150.0]),
            rise_times=2.0 * 24.0 * 3600.0,
            fall_times=9.0 * 24.0 * 3600.0,
        )
        assert jnp.allclose(feature.rest_wavelengths, jnp.array([5000.0, 6000.0]))
        assert jnp.allclose(feature.sigmas, jnp.array([1000.0]))
        assert jnp.allclose(feature.amplitudes, jnp.array([1.0, 0.5]))
        assert jnp.allclose(feature.t_starts, jnp.array([0.0, 50.0]))
        assert jnp.allclose(feature.t_ends, jnp.array([100.0, 150.0]))
        assert jnp.allclose(feature.rise_times, jnp.array([2.0 * 24.0 * 3600.0]))
        assert jnp.allclose(feature.fall_times, jnp.array([9.0 * 24.0 * 3600.0]))

    def test_sed_features_from_list(self):
        """Test creating an SEDFeatures object from a list of feature dictionaries."""
        feature_list = [
            {
                "rest_wavelength": 5000.0,
                "sigma": 100.0,
                "amplitude": 1.0,
                "t_start": 0.0,
                "t_end": 100.0,
                "rise_time": 10.0,
                "fall_time": 10.0,
            },
            {
                "rest_wavelength": 6000.0,
                "sigma": 200.0,
                "amplitude": 0.5,
                "t_start": 50.0,
                "t_end": 150.0,
                # Use default rise and fall times.
            },
        ]
        feature = SEDFeatures.from_feature_list(feature_list)
        assert jnp.allclose(feature.rest_wavelengths, jnp.array([5000.0, 6000.0]))
        assert jnp.allclose(feature.sigmas, jnp.array([100.0, 200.0]))
        assert jnp.allclose(feature.amplitudes, jnp.array([1.0, 0.5]))
        assert jnp.allclose(feature.t_starts, jnp.array([0.0, 50.0]))
        assert jnp.allclose(feature.t_ends, jnp.array([100.0, 150.0]))
        assert jnp.allclose(feature.rise_times, jnp.array([10.0, 2.0 * 24.0 * 3600.0]))
        assert jnp.allclose(feature.fall_times, jnp.array([10.0, 5.0 * 24.0 * 3600.0]))

    def test_calculate_smooth_evolution(self):
        """Test the calculate_smooth_evolution method."""
        feature = SEDFeatures(
            rest_wavelengths=jnp.array([5000.0, 6000.0]),
            sigmas=jnp.array([100.0, 100.0]),
            amplitudes=jnp.array([1.0, 1.0]),
            t_starts=jnp.array([10.0, 20.0]),
            t_ends=jnp.array([100.0, 80.0]),
            rise_times=jnp.array([10.0, 10.0]),
            fall_times=jnp.array([20.0, 20.0]),
        )

        # Evaluate the feature contributions at 
        time = jnp.array([-10.0, 0.0, 10.0, 15.0, 20.0, 50.0, 75.0, 95.0, 100.0, 110.0])
        evolution = feature.calculate_smooth_evolution(time)
        expected = jnp.array(
            [
                [0.0, 0.0, 0.00247262, 0.5, 1.0, 1.0, 1.0, 0.04742587, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.00247262, 1.0, 0.04742587, 0.0, 0.0, 0.0],
            ]
        )
        assert jnp.allclose(evolution, expected)

    def test_apply_sed_no_features(self):
        """Test that applying NO_SED_FEATURES returns the original flux."""
        flux = jnp.array([1.0, 2.0, 3.0])
        time = jnp.array([0.0, 1.0, 2.0])
        frequency = jnp.array([1e14, 2e14, 3e14])

        result = apply_sed_feature(NO_SED_FEATURES, flux, frequency, time)
        assert jnp.allclose(result, flux)

    def test_apply_sed_features(self):
        """Test that applying features modifies the flux."""
        feature = SEDFeatures(
            rest_wavelengths=jnp.array([5000.0]),
            sigmas=jnp.array([100.0]),
            amplitudes=jnp.array([10.0]),
            t_starts=jnp.array([0.0]),
            t_ends=jnp.array([10000.0]),
        )
        flux = jnp.ones((100, 50))
        time = jnp.linspace(0, 20000, 100)
        frequency = jnp.linspace(3e14, 6e14, 50)

        result = apply_sed_feature(feature, flux, frequency, time)
        assert not jnp.allclose(result, flux)
        assert jnp.all(result >= flux)
        assert jnp.unique(result).shape[0] > 1  # We see multiple results.
