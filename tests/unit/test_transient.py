"""
Tests for redback_jax.transient module.
"""
import pytest
import numpy as np
import jax.numpy as jnp
import tempfile
import os
from redback_jax.transient import Transient, Spectrum


class TestSpectrum:
    """Test cases for Spectrum class."""
    
    def test_spectrum_creation(self):
        """Test basic spectrum creation."""
        wavelength = np.linspace(4000, 7000, 100)
        flux = np.random.random(100)
        flux_err = 0.1 * flux
        
        spectrum = Spectrum(wavelength, flux, flux_err, name="test_spectrum")
        
        assert len(spectrum.wavelength) == 100
        assert len(spectrum.flux_density) == 100
        assert len(spectrum.flux_density_err) == 100
        assert spectrum.name == "test_spectrum"
        assert spectrum.data_mode == "spectrum"
    
    def test_spectrum_jax_arrays(self):
        """Test spectrum with JAX arrays."""
        wavelength = jnp.linspace(4000, 7000, 50)
        flux = jnp.array(np.random.random(50))
        
        spectrum = Spectrum(wavelength, flux)
        
        assert isinstance(spectrum.wavelength, jnp.ndarray)
        assert isinstance(spectrum.flux_density, jnp.ndarray)
    
    def test_spectrum_validation(self):
        """Test spectrum validation."""
        wavelength = np.linspace(4000, 7000, 100)
        flux = np.random.random(50)  # Wrong size
        
        with pytest.raises(ValueError, match="must have the same length"):
            Spectrum(wavelength, flux)


class TestTransient:
    """Test cases for Transient class."""
    
    def test_transient_creation(self):
        """Test basic transient creation."""
        time = np.linspace(0, 10, 20)
        flux = np.random.random(20)
        flux_err = 0.1 * flux
        
        transient = Transient(
            time=time, 
            y=flux, 
            y_err=flux_err,
            data_mode='flux',
            name='test_transient'
        )
        
        assert len(transient.time) == 20
        assert len(transient.y) == 20
        assert transient.data_mode == 'flux'
        assert transient.name == 'test_transient'
    
    def test_transient_jax_arrays(self):
        """Test transient with JAX arrays."""
        time = jnp.linspace(0, 10, 15)
        magnitude = jnp.array(np.random.uniform(15, 20, 15))
        
        transient = Transient(
            time=time, 
            y=magnitude, 
            data_mode='magnitude'
        )
        
        assert isinstance(transient.time, jnp.ndarray)
        assert isinstance(transient.y, jnp.ndarray)
    
    def test_transient_validation(self):
        """Test transient validation."""
        time = np.linspace(0, 10, 20)
        flux = np.random.random(15)  # Wrong size
        
        with pytest.raises(ValueError, match="must have the same length"):
            Transient(time=time, y=flux)
        
        # Test invalid data mode
        with pytest.raises(ValueError, match="data_mode must be one of"):
            Transient(time=time, y=time, data_mode='invalid_mode')
    
    def test_empty_transient(self):
        """Test creating empty transient."""
        transient = Transient(name='empty', data_mode='flux')
        
        assert transient.time is None
        assert transient.y is None
        assert transient.name == 'empty'
    
    def test_from_data_file(self):
        """Test loading data from file."""
        # Create temporary data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("time,flux,flux_err,band\n")
            f.write("1.0,1.5,0.1,g\n")
            f.write("2.0,1.2,0.15,g\n")
            f.write("3.0,0.8,0.12,r\n")
            f.write("4.0,0.6,0.1,r\n")
            temp_filename = f.name
        
        try:
            transient = Transient.from_data_file(
                temp_filename, 
                data_mode='flux',
                name='file_test'
            )
            
            assert len(transient.time) == 4
            assert len(transient.y) == 4
            assert transient.data_mode == 'flux'
            assert transient.name == 'file_test'
            assert transient.bands is not None
            
        finally:
            os.unlink(temp_filename)
    
    def test_properties(self):
        """Test transient properties."""
        time = np.linspace(0, 10, 10)
        magnitude = np.random.uniform(15, 20, 10)
        
        transient = Transient(time=time, y=magnitude, data_mode='magnitude')
        
        assert "Time" in transient.xlabel
        assert "Magnitude" in transient.ylabel
        
        # Test flux data mode
        flux_transient = Transient(time=time, y=magnitude, data_mode='flux')
        assert "Flux" in flux_transient.ylabel
    
    def test_repr(self):
        """Test string representation."""
        time = np.linspace(0, 5, 10)
        flux = np.random.random(10)
        
        transient = Transient(time=time, y=flux, name='repr_test', data_mode='flux')
        repr_str = repr(transient)
        
        assert "repr_test" in repr_str
        assert "flux" in repr_str
        assert "n_points=10" in repr_str


class TestIntegration:
    """Integration tests for transient functionality."""
    
    def test_full_workflow(self):
        """Test a complete workflow from data creation to analysis."""
        # Generate synthetic lightcurve data
        time = np.linspace(0, 20, 50)
        # Simple exponential decay
        flux = 10 * np.exp(-time/5) + 0.1 * np.random.randn(50)
        flux_err = 0.1 * np.abs(flux) + 0.05
        
        # Create transient
        transient = Transient(
            time=time,
            y=flux, 
            y_err=flux_err,
            data_mode='flux',
            name='synthetic_transient',
            redshift=0.1
        )
        
        # Verify basic properties
        assert transient.redshift == 0.1
        assert len(transient.time) == 50
        
        # Test data access
        peak_flux = float(jnp.max(transient.y))
        assert peak_flux > 0
        
        # Test that we can work with the data using JAX
        mean_time = float(jnp.mean(transient.time))
        assert 0 <= mean_time <= 20
    
    def test_multi_band_data(self):
        """Test handling multi-band photometric data."""
        time = np.repeat(np.linspace(0, 10, 10), 3)  # 3 bands, 10 times each
        bands = np.tile(['g', 'r', 'i'], 10)
        
        # Different flux levels for different bands
        base_flux = np.tile(np.linspace(1, 0.1, 10), 3)
        band_offsets = {'g': 0, 'r': -0.2, 'i': -0.5}
        flux = base_flux + np.array([band_offsets[b] for b in bands])
        
        transient = Transient(
            time=time,
            y=flux,
            data_mode='flux',
            bands=bands,
            name='multi_band_test'
        )
        
        assert len(transient.time) == 30
        assert len(transient.bands) == 30
        assert set(transient.bands) == {'g', 'r', 'i'}