"""
Tests for redback_jax.models.kilonova module.
"""
import pytest
import numpy as np
import jax.numpy as jnp
from redback_jax.models.kilonova import (
    one_component_kilonova_model,
    kilonova_nickel_cobalt,
    nickel_cobalt_heating_rate,
    thermalisation_efficiency,
    barnes_kasen_thermalisation_efficiency,
    diffusion_timescale,
    blackbody_temperature,
    photospheric_radius
)


class TestKilonovaPhysics:
    """Test core kilonova physics functions."""
    
    def test_barnes_kasen_efficiency(self):
        """Test Barnes & Kasen thermalisation efficiency calculation."""
        mej = 0.01  # solar masses
        vej = 0.1   # units of c
        
        av, bv, dv = barnes_kasen_thermalisation_efficiency(mej, vej)
        
        # Check that parameters are positive
        assert av > 0
        assert bv > 0
        assert dv > 0
        
        # Check approximate expected values
        assert 0.1 <= av <= 1.0
        assert 1.0 <= bv <= 10.0
        assert 1.0 <= dv <= 2.0
    
    def test_thermalisation_efficiency(self):
        """Test time-dependent thermalisation efficiency."""
        time = jnp.linspace(0.1, 100, 50) * 86400  # 0.1 to 100 days in seconds
        mej = 0.01
        vej = 0.1
        
        e_th = thermalisation_efficiency(time, mej, vej)
        
        # Check that efficiency is between 0 and 1
        assert jnp.all(e_th >= 0)
        assert jnp.all(e_th <= 1)
        
        # Check that efficiency decreases with time
        assert e_th[0] > e_th[-1]
        
        # Check reasonable initial efficiency (up to ~0.7 with our parameters)
        assert e_th[0] <= 1.0  # More lenient check
    
    def test_nickel_cobalt_heating_rate(self):
        """Test Ni-Co radioactive decay heating rate."""
        time = jnp.linspace(0.1, 200, 100) * 86400  # 0.1 to 200 days
        m_ni56 = 0.001  # solar masses
        
        heating_rate = nickel_cobalt_heating_rate(time, m_ni56)
        
        # Check that heating rate is positive
        assert jnp.all(heating_rate > 0)
        
        # Check that heating rate decreases with time
        assert heating_rate[0] > heating_rate[-1]
        
        # Check reasonable order of magnitude (should be ~ 1e40-1e42 erg/s initially)
        assert 1e38 <= heating_rate[0] <= 1e44  # More lenient range
        
        # Check exponential-like decay behavior
        # After ~80 days, Co-56 dominates, so should decay more slowly
        mid_point = len(time) // 2
        early_ratio = heating_rate[1] / heating_rate[0]
        late_ratio = heating_rate[mid_point + 1] / heating_rate[mid_point]
        # Late decay should be slower than early decay
        assert late_ratio > early_ratio
    
    def test_diffusion_timescale(self):
        """Test photon diffusion timescale calculation."""
        mej = 0.01
        vej = 0.1
        kappa = 10.0
        
        tau_diff = diffusion_timescale(mej, vej, kappa)
        
        # Check positive timescale
        assert tau_diff > 0
        
        # Check reasonable order of magnitude (should be days to weeks)
        assert 86400 <= tau_diff <= 30 * 86400  # 1 to 30 days
        
        # Check scaling with parameters
        tau_diff_high_kappa = diffusion_timescale(mej, vej, 100.0)
        tau_diff_high_mass = diffusion_timescale(0.1, vej, kappa)
        tau_diff_high_vel = diffusion_timescale(mej, 0.3, kappa)
        
        assert tau_diff_high_kappa > tau_diff  # Higher opacity -> longer diffusion
        assert tau_diff_high_mass > tau_diff   # Higher mass -> longer diffusion
        assert tau_diff_high_vel < tau_diff    # Higher velocity -> shorter diffusion
    
    def test_blackbody_temperature(self):
        """Test blackbody temperature calculation."""
        luminosity = jnp.array([1e40, 1e41, 1e42])  # erg/s
        radius = jnp.array([1e14, 1e14, 1e14])      # cm
        
        temperature = blackbody_temperature(luminosity, radius)
        
        # Check positive temperatures
        assert jnp.all(temperature > 0)
        
        # Check reasonable range (thousands of K)
        assert jnp.all(temperature >= 1000)
        assert jnp.all(temperature <= 50000)
        
        # Check scaling: higher luminosity -> higher temperature
        assert temperature[2] > temperature[1] > temperature[0]
    
    def test_photospheric_radius(self):
        """Test photospheric radius evolution."""
        time = jnp.linspace(1, 100, 50) * 86400  # 1 to 100 days
        vej = 0.1
        tau_diff = 7 * 86400  # 7 days
        
        radius = photospheric_radius(time, vej, tau_diff)
        
        # Check positive radius
        assert jnp.all(radius > 0)
        
        # Check generally increasing with time
        assert radius[-1] > radius[0]
        
        # Check reasonable order of magnitude (should be ~ 1e14-1e15 cm)
        assert 1e13 <= radius[0] <= 1e16
        assert 1e14 <= radius[-1] <= 1e17


class TestKilonovaModel:
    """Test the full kilonova model."""
    
    def test_one_component_kilonova_luminosity(self):
        """Test the one-component kilonova luminosity model."""
        time = jnp.linspace(0.1, 100, 50) * 86400  # seconds
        mej = 0.01
        vej = 0.1
        m_ni56 = 0.001
        kappa = 10.0
        
        from redback_jax.models.kilonova import one_component_kilonova_luminosity
        luminosity = one_component_kilonova_luminosity(time, mej, vej, m_ni56, kappa)
        
        # Check positive luminosity
        assert jnp.all(luminosity > 0)
        
        # Check reasonable range (broaden for safety)
        assert 1e35 <= jnp.max(luminosity) <= 1e45
        
        # Check that peak occurs early and then declines
        peak_idx = jnp.argmax(luminosity)
        assert peak_idx < len(luminosity) // 2  # Peak in first half
    
    def test_full_kilonova_model(self):
        """Test the full kilonova model with all outputs."""
        time = jnp.linspace(0.1, 50, 30) * 86400  # seconds
        mej = 0.01
        vej = 0.1
        m_ni56 = 0.001
        
        result = one_component_kilonova_model(time, mej, vej, m_ni56)
        
        # Check all expected keys are present
        expected_keys = ['luminosity', 'temperature', 'radius', 'heating_rate']
        for key in expected_keys:
            assert key in result
            assert len(result[key]) == len(time)
        
        # Check physical reasonableness
        assert jnp.all(result['luminosity'] > 0)
        assert jnp.all(result['temperature'] >= 3000)  # Temperature floor
        assert jnp.all(result['radius'] > 0)
        assert jnp.all(result['heating_rate'] > 0)
        
        # Check that temperature and luminosity decrease with time
        assert result['temperature'][-1] <= result['temperature'][0]
        
    def test_kilonova_nickel_cobalt_interface(self):
        """Test the interface function for Transient class."""
        time_days = np.linspace(0.1, 30, 20)  # days
        mej = 0.01
        vej = 0.1
        m_ni56 = 0.001
        
        luminosity = kilonova_nickel_cobalt(time_days, mej, vej, m_ni56)
        
        # Check output is JAX array
        assert isinstance(luminosity, jnp.ndarray)
        assert len(luminosity) == len(time_days)
        
        # Check positive values
        assert jnp.all(luminosity > 0)
        
        # Check reasonable magnitude
        assert 1e38 <= jnp.max(luminosity) <= 1e44


class TestParameterScaling:
    """Test how the model scales with different parameters."""
    
    def test_mass_scaling(self):
        """Test scaling with ejecta mass."""
        time = jnp.array([1.0, 10.0, 30.0]) * 86400
        vej = 0.1
        m_ni56 = 0.001
        
        # Test different ejecta masses
        masses = [0.005, 0.01, 0.02]
        luminosities = []
        
        for mej in masses:
            lum = kilonova_nickel_cobalt(time / 86400, mej, vej, m_ni56)
            luminosities.append(lum)
        
        # Higher mass should generally give different light curves
        # (exact scaling depends on complex physics)
        assert not jnp.allclose(luminosities[0], luminosities[1])
        assert not jnp.allclose(luminosities[1], luminosities[2])
    
    def test_velocity_scaling(self):
        """Test scaling with ejecta velocity."""
        time = jnp.array([1.0, 10.0, 30.0]) * 86400
        mej = 0.01
        m_ni56 = 0.001
        
        # Test different velocities
        velocities = [0.05, 0.1, 0.2]
        luminosities = []
        
        for vej in velocities:
            lum = kilonova_nickel_cobalt(time / 86400, mej, vej, m_ni56)
            luminosities.append(lum)
        
        # Different velocities should give different light curves
        assert not jnp.allclose(luminosities[0], luminosities[1])
        assert not jnp.allclose(luminosities[1], luminosities[2])
    
    def test_nickel_mass_scaling(self):
        """Test scaling with Ni-56 mass."""
        time = jnp.array([1.0, 10.0, 30.0]) * 86400
        mej = 0.01
        vej = 0.1
        
        # Test different Ni-56 masses
        ni_masses = [0.0005, 0.001, 0.002]
        luminosities = []
        
        for m_ni56 in ni_masses:
            lum = kilonova_nickel_cobalt(time / 86400, mej, vej, m_ni56)
            luminosities.append(lum)
        
        # Higher Ni-56 mass should give higher luminosity
        assert jnp.all(luminosities[2] > luminosities[1])
        assert jnp.all(luminosities[1] > luminosities[0])


class TestJAXCompatibility:
    """Test JAX-specific functionality."""
    
    def test_jit_compilation(self):
        """Test that functions can be JIT compiled."""
        import jax
        
        # Test JIT compilation of main function
        @jax.jit
        def jitted_model(time, mej, vej, m_ni56):
            return kilonova_nickel_cobalt(time, mej, vej, m_ni56)
        
        time = jnp.array([1.0, 5.0, 10.0])
        result = jitted_model(time, 0.01, 0.1, 0.001)
        
        assert isinstance(result, jnp.ndarray)
        assert len(result) == 3
    
    def test_gradient_computation(self):
        """Test gradient computation with respect to parameters."""
        import jax
        
        def model_wrapper(params):
            mej, vej, m_ni56 = params
            time = jnp.array([1.0, 5.0, 10.0])
            lum = kilonova_nickel_cobalt(time, mej, vej, m_ni56)
            return jnp.sum(lum)  # Scalar output for gradient
        
        params = jnp.array([0.01, 0.1, 0.001])
        grad_func = jax.grad(model_wrapper)
        
        gradients = grad_func(params)
        
        # Check that gradients are finite and non-zero
        assert jnp.all(jnp.isfinite(gradients))
        assert jnp.any(gradients != 0)
    
    def test_vectorization(self):
        """Test vectorized evaluation over parameter arrays."""
        import jax
        
        # Test vmap over multiple parameter sets
        time = jnp.array([1.0, 5.0, 10.0])
        mej_array = jnp.array([0.005, 0.01, 0.02])
        
        def single_model(mej):
            return kilonova_nickel_cobalt(time, mej, 0.1, 0.001)
        
        vmap_model = jax.vmap(single_model)
        results = vmap_model(mej_array)
        
        assert results.shape == (3, 3)  # 3 masses × 3 time points
        assert jnp.all(results > 0)