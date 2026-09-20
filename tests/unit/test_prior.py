"""Tests for redback_jax.inference.prior distributions, constraints and Prior."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from redback_jax.inference.prior import (
    Constraint,
    Gaussian,
    LogUniform,
    Prior,
    Uniform,
    _Distribution,
    make_csm_photosphere_constraint,
    make_csm_photosphere_constraint_log,
)

KEY = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def test_base_distribution_is_abstract():
    d = _Distribution("x")
    with pytest.raises(NotImplementedError):
        d.sample(KEY)
    with pytest.raises(NotImplementedError):
        d.log_prob(jnp.array(0.0))
    with pytest.raises(NotImplementedError):
        d.low
    with pytest.raises(NotImplementedError):
        d.high


def test_uniform_log_prob_and_support():
    p = Uniform(0.05, 0.2, name="f")
    assert (p.low, p.high) == (0.05, 0.2)
    np.testing.assert_allclose(float(p.log_prob(jnp.array(0.1))), -math.log(0.15), rtol=1e-6)
    assert float(p.log_prob(jnp.array(0.5))) == -np.inf
    assert float(p.log_prob(jnp.array(0.0))) == -np.inf
    assert "Uniform" in repr(p) and "'f'" in repr(p)


def test_uniform_samples_within_bounds():
    p = Uniform(-2.0, 3.0, name="x")
    draws = np.asarray(jax.vmap(p.sample)(jax.random.split(KEY, 500)))
    assert draws.min() >= -2.0 and draws.max() <= 3.0


def test_loguniform_requires_positive_minimum():
    with pytest.raises(ValueError):
        LogUniform(0.0, 1.0, name="x")
    with pytest.raises(ValueError):
        LogUniform(-1.0, 1.0, name="x")


def test_loguniform_normalised_and_bounded():
    lo, hi = 1e-2, 1e2
    p = LogUniform(lo, hi, name="k")
    assert (p.low, p.high) == (lo, hi)

    # Integrate exp(log_prob) over log-spaced grid: should be 1.
    x = jnp.geomspace(lo, hi, 20001)
    density = jnp.exp(jax.vmap(p.log_prob)(x))
    integral = float(jnp.trapezoid(density, x))
    np.testing.assert_allclose(integral, 1.0, rtol=1e-3)

    assert float(p.log_prob(jnp.array(1e3))) == -np.inf
    assert float(p.log_prob(jnp.array(1e-3))) == -np.inf
    assert "LogUniform" in repr(p)

    draws = np.asarray(jax.vmap(p.sample)(jax.random.split(KEY, 500)))
    assert draws.min() >= lo * 0.999 and draws.max() <= hi * 1.001
    # log-uniform: the median of log10 draws sits near the centre of the range
    assert abs(np.median(np.log10(draws))) < 0.4


def test_gaussian_log_prob_and_truncation():
    g = Gaussian(1.0, 2.0, name="g")
    np.testing.assert_allclose(
        float(g.log_prob(jnp.array(1.0))), -math.log(2.0 * math.sqrt(2 * math.pi)), rtol=1e-6,
    )
    assert np.isinf(g.low) and np.isinf(g.high)
    assert "Gaussian" in repr(g)

    trunc = Gaussian(0.0, 1.0, name="t", minimum=-1.0, maximum=1.0)
    assert (trunc.low, trunc.high) == (-1.0, 1.0)
    assert float(trunc.log_prob(jnp.array(2.0))) == -np.inf
    assert np.isfinite(float(trunc.log_prob(jnp.array(0.5))))

    draws = np.asarray(jax.vmap(g.sample)(jax.random.split(KEY, 4000)))
    assert abs(draws.mean() - 1.0) < 0.15
    assert abs(draws.std() - 2.0) < 0.15


# ---------------------------------------------------------------------------
# Constraint
# ---------------------------------------------------------------------------

def test_constraint_barrier_shape_and_gradient():
    c = Constraint(lambda d: d["x"], scale=1.0, name="x_positive")
    satisfied = float(c.log_prob({"x": jnp.array(5.0)}))
    violated = float(c.log_prob({"x": jnp.array(-5.0)}))
    assert abs(satisfied) < 1e-6
    assert violated < -50.0
    assert "x_positive" in repr(c)

    grad = jax.grad(lambda x: c.log_prob({"x": x}))(jnp.array(-0.01))
    assert np.isfinite(float(grad)) and float(grad) > 0  # pushes back to feasible


def _csm_params(csm_mass, rho, eta=2.0, r0=50.0):
    return dict(
        r0=jnp.float32(r0),
        csm_mass=jnp.float32(csm_mass),
        rho=jnp.float32(rho),
        eta=jnp.float32(eta),
    )


def _to_log_params(p):
    return dict(
        log_r0=jnp.log(p["r0"]),
        csm_mass=p["csm_mass"],
        log_rho=jnp.log(p["rho"]),
        eta=p["eta"],
    )


def test_csm_photosphere_constraint_satisfied_and_violated():
    c = make_csm_photosphere_constraint(fixed_kappa=0.1, scale=0.1)
    ok = _csm_params(csm_mass=1.0, rho=1e-14)
    bad = _csm_params(csm_mass=0.01, rho=1e-12)

    assert float(c.fn(ok)) > 0.5
    assert float(c.fn(bad)) < 0.0
    assert abs(float(c.log_prob(ok))) < 1e-3
    assert float(c.log_prob(bad)) < -10.0
    assert "csm_photosphere" in repr(c)


def test_csm_photosphere_log_variant_matches_linear():
    lin = make_csm_photosphere_constraint(fixed_kappa=0.1, scale=0.1)
    log = make_csm_photosphere_constraint_log(fixed_kappa=0.1, scale=0.1)
    for csm_mass, rho in [(1.0, 1e-14), (0.01, 1e-12), (10.0, 1e-10), (50.0, 1e-8)]:
        p = _csm_params(csm_mass, rho)
        np.testing.assert_allclose(
            float(log.log_prob(_to_log_params(p))),
            float(lin.log_prob(p)),
            rtol=1e-2, atol=1e-3,
        )


def test_csm_photosphere_constraint_eta_near_one_is_finite():
    """The eta≈1 singularity guard must keep value and gradient finite."""
    c = make_csm_photosphere_constraint()
    p = _csm_params(csm_mass=1.0, rho=1e-13, eta=1.0)
    assert np.isfinite(float(c.log_prob(p)))
    g = jax.grad(lambda rho: c.log_prob({**p, "rho": rho}))(p["rho"])
    assert np.isfinite(float(g))


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------

@pytest.fixture
def prior():
    return Prior([
        Uniform(0.0, 1.0, name="a"),
        LogUniform(1.0, 100.0, name="b"),
        Gaussian(0.0, 1.0, name="c", minimum=-3.0, maximum=3.0),
    ])


def test_prior_bookkeeping(prior):
    assert prior.names == ["a", "b", "c"]
    assert prior.n_params == len(prior) == 3
    np.testing.assert_allclose(np.asarray(prior._lows), [0.0, 1.0, -3.0])
    np.testing.assert_allclose(np.asarray(prior._highs), [1.0, 100.0, 3.0])
    text = repr(prior)
    assert "Uniform" in text and "LogUniform" in text and "Gaussian" in text


def test_prior_sampling(prior):
    one = prior.sample(KEY)
    assert set(one) == {"a", "b", "c"}

    many = prior.sample_n(KEY, 200)
    assert many.shape == (200, 3)
    assert np.all(np.asarray(many[:, 0]) >= 0) and np.all(np.asarray(many[:, 0]) <= 1)
    assert np.all(np.asarray(many[:, 1]) >= 0.99) and np.all(np.asarray(many[:, 1]) <= 101)
    # all in-support draws should have finite prior density
    lp = jax.vmap(prior.log_prob)(many)
    assert np.all(np.isfinite(np.asarray(lp)))


def test_prior_log_prob_is_sum_and_out_of_support_is_neg_inf(prior):
    x = jnp.array([0.5, 10.0, 0.0])
    expected = sum(float(d.log_prob(x[i])) for i, d in enumerate(prior.distributions))
    np.testing.assert_allclose(float(prior.log_prob(x)), expected, rtol=1e-6)
    np.testing.assert_allclose(float(prior.log_prob_fn()(x)), expected, rtol=1e-6)

    outside = jnp.array([1.5, 10.0, 0.0])
    assert float(prior.log_prob(outside)) == -np.inf


def test_prior_param_dict_roundtrip(prior):
    vec = jnp.array([0.25, 5.0, -1.0])
    d = prior.params_to_dict(vec)
    assert set(d) == {"a", "b", "c"}
    np.testing.assert_allclose(np.asarray(prior.dict_to_params(d)), np.asarray(vec))


def test_prior_with_constraint_adds_barrier_and_is_jit_safe():
    c = Constraint(lambda d: d["x"] - 0.5, scale=0.1, name="x_above_half")
    p = Prior([Uniform(0.0, 1.0, name="x"), c])

    assert p.n_params == 1 and p.names == ["x"]
    assert "x_above_half" in repr(p)

    hi, lo = jnp.array([0.9]), jnp.array([0.1])
    assert float(p.log_prob(lo)) < float(p.log_prob(hi)) - 5.0

    fn = jax.jit(p.log_prob_fn())
    np.testing.assert_allclose(float(fn(hi)), float(p.log_prob(hi)), rtol=1e-6)
    np.testing.assert_allclose(float(fn(lo)), float(p.log_prob(lo)), rtol=1e-6)
