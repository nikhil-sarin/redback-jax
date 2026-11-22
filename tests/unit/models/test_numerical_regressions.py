"""Independent numerical references and sampling-grid regressions."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from redback_jax.models import tophat_redback
from redback_jax.models.kilonova import (
    metzger_kilonova_bolometric, magnetar_boosted_kilonova_bolometric,
)
from redback_jax.models.supernova_models import _run_magnetar_ode_diffrax


def test_kilonova_native_converged_reference_and_sparse_epochs():
    # Redback 1.21.0 _metzger_kilonova_model on geomspace(0.01, 7e6,
    # 200000) seconds, interpolated in linear luminosity at these epochs.
    epochs = jnp.array([0.5, 1., 2., 3., 5., 7., 10.])
    reference = np.array([41.4905, 41.4689, 41.4001, 41.3168,
                          41.1344, 40.9450, 40.6701])
    params = dict(mej=0.05, vej=0.1, beta=3., kappa=3.)
    sparse = metzger_kilonova_bolometric(epochs, **params)
    np.testing.assert_allclose(sparse, reference, atol=0.002, rtol=0)
    dense = jnp.sort(jnp.concatenate([epochs, jnp.geomspace(0.1, 10., 150)]))
    output = metzger_kilonova_bolometric(dense, **params)
    np.testing.assert_allclose(jnp.interp(epochs, dense, output), sparse, atol=1e-9)


def test_magnetar_kilonova_late_energy_budget():
    time = jnp.array([100., 300., 1000.])
    params = dict(mej=0.05, vej=0.1, beta=3., kappa=3.,
                  p0=1., bp=1., mass_ns=2.2, theta_pb=np.pi/2)
    actual = magnetar_boosted_kilonova_bolometric(time, **params)
    # Independent dipole spin-down expression; optically thin ejecta radiate
    # the deposited power. This detects a spurious shell-mass suppression.
    erot = 2.6e52 * (2.2/1.4)**1.5
    tp = 1.3e5 * (2.2/1.4)**1.5
    engine = 2 * erot / tp / (1 + np.asarray(time)*86400/tp)**2
    np.testing.assert_allclose(actual, np.log10(engine), atol=0.1, rtol=0)


@pytest.mark.parametrize("angle", [0.05, 0.3])
def test_afterglow_time_step_convergence(angle):
    params = dict(redshift=0.1, thv=angle, loge0=52.5, thc=0.1,
                  logn0=-1., p=2.3, logepse=-1., logepsb=-2., g0=300.,
                  xiN=1., frequency=6e9, res=16)
    epochs = jnp.array([0.1, 1., 10., 30., 100., 300.])
    reference = tophat_redback(epochs, steps=16000, **params)
    actual = tophat_redback(epochs, **params)
    np.testing.assert_allclose(actual, reference, rtol=0.01)


def test_magnetar_step_limit_raises():
    with pytest.raises(Exception, match="maximum number of solver steps"):
        result = _run_magnetar_ode_diffrax(
            jnp.array([1., 100.]), 1., 51., 0.1, 45., 1e6, 3., 1.,
            max_steps=1,
        )
        jax.block_until_ready(result)


def test_observer_quadrature_matches_analytic_integral():
    from redback_jax.afterglow.radiation import ShockState, observer_state
    from redback_jax.constants import speed_of_light
    radius = jnp.array([1., 2., 4., 8.]) * 1e15
    # Linear 1/beta has an exact trapezoidal integral. Initial unresolved
    # interval is coasting with the first sampled beta.
    inverse_beta = 1.1 + radius / 1e16
    zeros = jnp.zeros_like(radius)
    shock = ShockState(1/inverse_beta, zeros, jnp.ones_like(radius)*0.01,
                       radius, zeros, zeros, zeros, zeros, zeros)
    angle = 0.2
    result = observer_state(27., 0.01, 1., angle, jnp.ones_like(radius)*2., shock)
    r = np.asarray(radius)
    expected = (r[0]*(float(inverse_beta[0])-np.cos(angle))
                + (1.1-np.cos(angle))*(r-r[0])
                + (r*r-r[0]*r[0])/(2e16)) / speed_of_light
    np.testing.assert_allclose(result.observer_time, expected, rtol=1e-12)


def test_batched_magnetar_matches_tight_adaptive_reference():
    from redback_jax.models import general_magnetar_driven_supernova_bolometric_batched
    time = jnp.geomspace(1., 300., 30)
    values = (1., 51., 0.1, 45., 1e6, 3., 1., 0.1)
    reference = _run_magnetar_ode_diffrax(time, *values, rtol=1e-11, atol=1e-14)[0]
    actual = general_magnetar_driven_supernova_bolometric_batched(
        time, *[jnp.array([v]) for v in values]
    )[0]
    np.testing.assert_allclose(actual, reference, atol=0.003, rtol=0)


def test_spreading_angular_convergence():
    params = dict(redshift=0.1, thv=0.05, loge0=52.5, thc=0.1,
                  logn0=-1., p=2.3, logepse=-1., logepsb=-2., g0=300.,
                  xiN=1., frequency=6e9, steps=1000)
    epochs = jnp.array([30., 100., 300.])
    reference = tophat_redback(epochs, res=400, **params)
    for resolution, tolerance in [(16, 0.012), (50, 0.005), (200, 0.0003)]:
        actual = tophat_redback(epochs, res=resolution, **params)
        np.testing.assert_allclose(actual, reference, rtol=tolerance)
