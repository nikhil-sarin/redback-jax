import jax.numpy as jnp
import unittest

from redback_jax.photosphere import compute_temperature_floor


class TestTemperatureFloor(unittest.TestCase):

    def test_calculate_photosphere_properties(self):
        time = jnp.array([1, 2, 3])
        luminosity = jnp.array([1, 2, 3]) * 2e17
        vej = jnp.array([1, 2, 3])
        temperature_floor = 1

        temperatures, r_photosphere = compute_temperature_floor(time, luminosity, vej, temperature_floor)
        expected_temperatures = jnp.array([1.39250008, 1., 1.])
        expected_r_photosphere = jnp.array([8640000000.0, 2.36929531e10, 29017822836.350456])
        self.assertTrue(jnp.allclose(expected_temperatures, temperatures))
        self.assertTrue(jnp.allclose(expected_r_photosphere, r_photosphere))
