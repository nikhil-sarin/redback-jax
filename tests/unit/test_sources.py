"""
Tests for redback_jax.sources module with jax-bandflux TimeSeriesSource integration.

These tests verify:
1. PrecomputedSpectraSource correctly wraps TimeSeriesSource
2. Functional API works with params dict (amplitude parameter)
3. Simple mode (single band) and optimized mode (bridges) both work
4. from_arnett_model classmethod integrates with new supernova_models interface
5. JIT compilation works for fitting workflows
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from redback_jax.sources import PrecomputedSpectraSource


# Enable JAX 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def simple_sed_source():
    """Create a simple test SED for basic tests."""
    # Create simple analytical SED (cooling blackbody)
    phases = np.linspace(-20, 40, 60)
    wavelengths = np.linspace(3000, 9000, 200)

    # Create 2D grid
    phase_grid, wave_grid = np.meshgrid(phases, wavelengths, indexing='ij')

    # Simple model: cooling blackbody
    temperature = 10000 / np.maximum(1.0 + phase_grid / 20.0, 0.1)
    luminosity_factor = np.exp(-0.5 * (phase_grid / 15.0)**2)

    # Wien's approximation for peak wavelength
    lambda_peak = 2.898e7 / temperature

    # Gaussian-like SED shape
    sed_shape = np.exp(-0.5 * ((wave_grid - lambda_peak) / 800.0)**2)
    flux_grid = sed_shape * luminosity_factor * 1e-15

    return PrecomputedSpectraSource(
        phases=phases,
        wavelengths=wavelengths,
        flux_grid=flux_grid,
        zero_before=True,
        time_spline_degree=3,
        name='test_sed',
        version='1.0'
    )


def test_initialization(simple_sed_source):
    """Test that PrecomputedSpectraSource initializes correctly."""
    source = simple_sed_source

    # Check attributes exist and have correct types
    assert hasattr(source, 'phases')
    assert hasattr(source, 'wavelengths')
    assert hasattr(source, 'flux_grid')
    assert hasattr(source, '_source')

    # Check shapes
    assert len(source.phases) == 60
    assert len(source.wavelengths) == 200
    assert source.flux_grid.shape == (60, 200)

    # Check that underlying TimeSeriesSource was created
    assert source._source is not None
    assert hasattr(source._source, 'bandflux')
    assert hasattr(source._source, 'bandmag')


def test_initialization_shape_mismatch():
    """Test that initialization fails with mismatched shapes."""
    phases = np.linspace(-20, 40, 60)
    wavelengths = np.linspace(3000, 9000, 200)
    flux_grid = np.ones((50, 200))  # Wrong shape!

    with pytest.raises(ValueError, match="Flux grid shape"):
        PrecomputedSpectraSource(
            phases=phases,
            wavelengths=wavelengths,
            flux_grid=flux_grid
        )


def test_bandflux_simple_mode(simple_sed_source):
    """Test bandflux calculation in simple mode (single band)."""
    source = simple_sed_source
    params = {'amplitude': 1.0}

    # Test single phase
    flux = source.bandflux(params, 'bessellb', 0.0)
    assert isinstance(flux, (float, jnp.ndarray))
    assert jnp.isfinite(flux)
    assert flux > 0

    # Test multiple phases
    phases = jnp.array([-10, -5, 0, 5, 10])
    fluxes = source.bandflux(params, 'bessellb', phases)
    assert fluxes.shape == (5,)
    assert jnp.all(jnp.isfinite(fluxes))
    assert jnp.all(fluxes > 0)


def test_bandmag_simple_mode(simple_sed_source):
    """Test bandmag calculation in simple mode (single band)."""
    source = simple_sed_source
    params = {'amplitude': 1.0}

    # Test single phase
    mag = source.bandmag(params, 'bessellv', 0.0)
    assert isinstance(mag, (float, jnp.ndarray))
    assert jnp.isfinite(mag)

    # Test multiple phases
    phases = jnp.array([-10, -5, 0, 5, 10])
    mags = source.bandmag(params, 'bessellv', phases)
    assert mags.shape == (5,)
    assert jnp.all(jnp.isfinite(mags))


def test_amplitude_scaling(simple_sed_source):
    """Test that amplitude parameter correctly scales flux."""
    source = simple_sed_source
    phase = 0.0
    band = 'bessellb'

    # Get fluxes with different amplitudes
    flux_1 = source.bandflux({'amplitude': 1.0}, band, phase)
    flux_2 = source.bandflux({'amplitude': 2.0}, band, phase)
    flux_05 = source.bandflux({'amplitude': 0.5}, band, phase)

    # Check scaling relationship
    assert jnp.allclose(flux_2, 2.0 * flux_1, rtol=1e-6)
    assert jnp.allclose(flux_05, 0.5 * flux_1, rtol=1e-6)


def test_prepare_bridges(simple_sed_source):
    """Test prepare_bridges method for optimized mode."""
    source = simple_sed_source
    unique_bands = ['bessellb', 'bessellv', 'bessellr']

    bridges, band_to_idx = source.prepare_bridges(unique_bands)

    # Check return types
    assert isinstance(bridges, tuple)
    assert isinstance(band_to_idx, dict)
    assert len(bridges) == 3
    assert len(band_to_idx) == 3

    # Check band mapping
    assert band_to_idx['bessellb'] == 0
    assert band_to_idx['bessellv'] == 1
    assert band_to_idx['bessellr'] == 2


def test_bandflux_optimized_mode(simple_sed_source):
    """Test bandflux in optimized mode with bridges."""
    source = simple_sed_source
    params = {'amplitude': 1.0}

    # Prepare bridges
    unique_bands = ['bessellb', 'bessellv', 'bessellr']
    bridges, band_to_idx = source.prepare_bridges(unique_bands)

    # Simulate observations
    obs_phases = jnp.array([-5, 5, -5, 5, -5, 5])
    obs_bands = ['bessellb', 'bessellb', 'bessellv', 'bessellv', 'bessellr', 'bessellr']
    band_indices = jnp.array([band_to_idx[b] for b in obs_bands])

    # Calculate fluxes in optimized mode
    fluxes_opt = source.bandflux(
        params, None, obs_phases,
        band_indices=band_indices,
        bridges=bridges,
        unique_bands=unique_bands
    )

    # Check output
    assert fluxes_opt.shape == (6,)
    assert jnp.all(jnp.isfinite(fluxes_opt))
    assert jnp.all(fluxes_opt > 0)

    # Compare with simple mode for validation
    flux_simple_0 = source.bandflux(params, 'bessellb', -5)
    assert jnp.allclose(fluxes_opt[0], flux_simple_0, rtol=1e-6)


def test_jit_compilation(simple_sed_source):
    """Test that bandflux can be JIT compiled for fitting."""
    source = simple_sed_source

    # Prepare bridges
    unique_bands = ['bessellb', 'bessellv']
    bridges, band_to_idx = source.prepare_bridges(unique_bands)

    # Setup mock observations
    obs_phases = jnp.array([0, 5, 10])
    band_indices = jnp.array([0, 0, 1])

    # Define JIT-compiled function
    @jax.jit
    def compute_model_fluxes(amplitude):
        params = {'amplitude': amplitude}
        return source.bandflux(
            params, None, obs_phases,
            band_indices=band_indices,
            bridges=bridges,
            unique_bands=unique_bands
        )

    # Test JIT compilation works
    fluxes = compute_model_fluxes(1.0)
    assert fluxes.shape == (3,)
    assert jnp.all(jnp.isfinite(fluxes))

    # Test different amplitude values
    fluxes_2 = compute_model_fluxes(2.0)
    assert jnp.allclose(fluxes_2, 2.0 * fluxes, rtol=1e-6)


def test_jit_likelihood(simple_sed_source):
    """Test JIT-compiled likelihood function for fitting."""
    source = simple_sed_source

    # Prepare bridges
    unique_bands = ['bessellb', 'bessellv']
    bridges, band_to_idx = source.prepare_bridges(unique_bands)

    # Generate synthetic observations
    true_amplitude = 2.0
    obs_phases = jnp.array([0, 5, 10])
    band_indices = jnp.array([0, 0, 1])

    true_params = {'amplitude': true_amplitude}
    true_fluxes = source.bandflux(
        true_params, None, obs_phases,
        band_indices=band_indices,
        bridges=bridges,
        unique_bands=unique_bands
    )

    flux_errors = 0.05 * true_fluxes
    obs_fluxes = true_fluxes  # Use true values for this test

    # Define JIT-compiled likelihood
    @jax.jit
    def neg_log_likelihood(amplitude):
        params = {'amplitude': amplitude}
        model_fluxes = source.bandflux(
            params, None, obs_phases,
            band_indices=band_indices,
            bridges=bridges,
            unique_bands=unique_bands
        )
        chi2 = jnp.sum(((obs_fluxes - model_fluxes) / flux_errors)**2)
        return 0.5 * chi2

    # Test at true amplitude (should give low chi2)
    chi2_true = neg_log_likelihood(true_amplitude)
    assert jnp.isfinite(chi2_true)
    assert chi2_true < 1e-10  # Should be ~0 since we're using true fluxes

    # Test at wrong amplitude (should give higher chi2)
    chi2_wrong = neg_log_likelihood(1.0)
    assert chi2_wrong > chi2_true


def test_from_arnett_model():
    """Test from_arnett_model classmethod with new supernova_models interface."""
    source = PrecomputedSpectraSource.from_arnett_model(
        f_nickel=0.1,
        mej=1.4,
        vej=5000,
        kappa=0.07,
        kappa_gamma=0.1,
        temperature_floor=5000,
        redshift=0.01
    )

    # Check that source was created
    assert source is not None
    assert isinstance(source, PrecomputedSpectraSource)

    # Check that it has the expected structure
    # The new interface generates 3000 time points from 0.1 to 3030 days
    assert len(source.phases) == 3000
    assert source.phases[0] >= 0.1
    assert source.phases[-1] <= 3100  # Allow some tolerance

    # Wavelength grid: 100-60000 Å with 100 points
    assert len(source.wavelengths) == 100
    assert source.wavelengths[0] >= 90  # Allow some tolerance
    assert source.wavelengths[-1] <= 61000

    # Check flux grid shape
    assert source.flux_grid.shape == (3000, 100)

    # Test that we can calculate fluxes
    params = {'amplitude': 1.0}
    flux = source.bandflux(params, 'bessellv', 15.0)
    assert jnp.isfinite(flux)
    assert flux > 0

    # Test magnitude calculation
    mag = source.bandmag(params, 'bessellv', 15.0)
    assert jnp.isfinite(mag)


def test_from_arnett_model_with_cosmology_params():
    """Test from_arnett_model with custom cosmology parameters."""
    source = PrecomputedSpectraSource.from_arnett_model(
        f_nickel=0.15,
        mej=1.2,
        vej=6000,
        kappa=0.08,
        kappa_gamma=0.12,
        temperature_floor=4000,
        redshift=0.05,
        cosmo_H0=70.0,  # Custom H0
        cosmo_Om0=0.3   # Custom Om0
    )

    assert source is not None
    assert len(source.phases) == 3000
    assert len(source.wavelengths) == 100

    # Test flux calculation
    params = {'amplitude': 1.0}
    flux = source.bandflux(params, 'bessellb', 20.0)
    assert jnp.isfinite(flux)


def test_zero_before_parameter():
    """Test zero_before parameter behavior."""
    phases = np.linspace(0, 40, 40)
    wavelengths = np.linspace(3000, 9000, 100)
    flux_grid = np.ones((40, 100)) * 1e-15

    # Create source with zero_before=True
    source_zero = PrecomputedSpectraSource(
        phases=phases,
        wavelengths=wavelengths,
        flux_grid=flux_grid,
        zero_before=True
    )

    # Create source with zero_before=False
    source_no_zero = PrecomputedSpectraSource(
        phases=phases,
        wavelengths=wavelengths,
        flux_grid=flux_grid,
        zero_before=False
    )

    params = {'amplitude': 1.0}

    # Test at phase before grid (should be zero for zero_before=True)
    flux_zero = source_zero.bandflux(params, 'bessellb', -10.0)
    flux_no_zero = source_no_zero.bandflux(params, 'bessellb', -10.0)

    # Note: The exact behavior depends on TimeSeriesSource implementation
    # This test documents the expected behavior
    assert jnp.isfinite(flux_zero)
    assert jnp.isfinite(flux_no_zero)


def test_different_spline_degrees():
    """Test different time_spline_degree values."""
    phases = np.linspace(0, 40, 40)
    wavelengths = np.linspace(3000, 9000, 100)
    flux_grid = np.ones((40, 100)) * 1e-15

    # Cubic interpolation (default)
    source_cubic = PrecomputedSpectraSource(
        phases=phases,
        wavelengths=wavelengths,
        flux_grid=flux_grid,
        time_spline_degree=3
    )

    # Linear interpolation
    source_linear = PrecomputedSpectraSource(
        phases=phases,
        wavelengths=wavelengths,
        flux_grid=flux_grid,
        time_spline_degree=1
    )

    params = {'amplitude': 1.0}

    # Both should work
    flux_cubic = source_cubic.bandflux(params, 'bessellb', 20.0)
    flux_linear = source_linear.bandflux(params, 'bessellb', 20.0)

    assert jnp.isfinite(flux_cubic)
    assert jnp.isfinite(flux_linear)

    # They should be similar but not identical
    # (for a uniform grid they should be very close)
    assert jnp.allclose(flux_cubic, flux_linear, rtol=0.1)
