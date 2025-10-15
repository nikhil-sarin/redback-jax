import jax.numpy as jnp

from redback_jax.interaction_processes import diffusion_convert_luminosity


def test_diffusion_convert_luminosity():
    times = jnp.array([1.0, 2.5, 10.0, 20.0])
    dense_times = jnp.arange(0.0, 200.0, 0.1)
    dense_luminosity = jnp.sin(dense_times / 20.0) * 1e42 + 1e42
    kappa = 0.1
    kappa_gamma = 0.03
    mej = 1.0
    vej = 1e4

    tau_diff, new_luminosity = diffusion_convert_luminosity(times, dense_times, dense_luminosity, kappa, kappa_gamma, mej, vej)
    assert jnp.allclose(tau_diff, 11.388947093970085)

    expected_luminosity = jnp.array([7.93656336e+39, 5.09847164e+40, 7.25889745e+41, 1.64496713e+42])
    assert jnp.allclose(new_luminosity, expected_luminosity, rtol=1e-4)
