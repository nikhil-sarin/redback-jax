import jax.numpy as jnp
import numpy as np

from redback_jax.interaction_processes import diffusion_convert_luminosity


def test_diffusion_convert_luminosity():
    times = jnp.array([1.0, 2.5, 10.0, 20.0], dtype=jnp.float32)
    dense_times = jnp.arange(0.0, 200.0, 0.1, dtype=jnp.float32)
    # Function takes log10_luminosity — compute in float64 then cast to float32
    log10_luminosity = jnp.array(
        np.log10(np.sin(np.arange(0.0, 200.0, 0.1) / 20.0) + 1.0) + 42.0,
        dtype=jnp.float32,
    )
    kappa = 0.1
    kappa_gamma = 0.03
    mej = 1.0
    vej = 1e4

    tau_diff, log10_new_luminosity = diffusion_convert_luminosity(
        times, dense_times, log10_luminosity, kappa, kappa_gamma, mej, vej)

    # tau_diff is a scalar in days
    assert not jnp.isnan(tau_diff) and not jnp.isinf(tau_diff)
    assert float(tau_diff) > 0.0

    # log10_new_luminosity should be in physical range with no nan/inf
    assert log10_new_luminosity.shape == times.shape
    assert not jnp.any(jnp.isnan(log10_new_luminosity))
    assert not jnp.any(jnp.isinf(log10_new_luminosity))
    assert jnp.all(log10_new_luminosity > 35.0) and jnp.all(log10_new_luminosity < 50.0)

    # Output should be monotonically increasing over this rising light curve
    assert jnp.all(jnp.diff(log10_new_luminosity) > 0)
