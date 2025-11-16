"""
Tests for redback_jax.inference.sampler module with BlackJAX integration.

These tests verify:
1. Sampler utilities (prior transform, likelihood creation)
2. Nested sampling functionality
3. MCMC sampling functionality
4. High-level fit_transient API
5. Result processing utilities
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from redback_jax.inference import (
    SamplerResult,
    create_uniform_prior,
    create_gaussian_likelihood,
    run_nested_sampling,
    summarize_result,
)


# Enable JAX 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


def test_inference_imports():
    """Test that inference module imports correctly."""
    from redback_jax import inference
    assert hasattr(inference, 'run_nested_sampling')
    assert hasattr(inference, 'run_mcmc')
    assert hasattr(inference, 'fit_transient')
    assert hasattr(inference, 'SamplerResult')


def test_create_uniform_prior():
    """Test uniform prior creation."""
    prior_bounds = {
        'a': (0.0, 10.0),
        'b': (-1.0, 1.0),
        'c': (100.0, 200.0)
    }

    prior_fn = create_uniform_prior(prior_bounds)

    # Test at lower bounds (u = 0)
    params_low = prior_fn(jnp.array([0.0, 0.0, 0.0]))
    assert jnp.isclose(params_low['a'], 0.0)
    assert jnp.isclose(params_low['b'], -1.0)
    assert jnp.isclose(params_low['c'], 100.0)

    # Test at upper bounds (u = 1)
    params_high = prior_fn(jnp.array([1.0, 1.0, 1.0]))
    assert jnp.isclose(params_high['a'], 10.0)
    assert jnp.isclose(params_high['b'], 1.0)
    assert jnp.isclose(params_high['c'], 200.0)

    # Test at midpoint (u = 0.5)
    params_mid = prior_fn(jnp.array([0.5, 0.5, 0.5]))
    assert jnp.isclose(params_mid['a'], 5.0)
    assert jnp.isclose(params_mid['b'], 0.0)
    assert jnp.isclose(params_mid['c'], 150.0)


def test_create_gaussian_likelihood():
    """Test Gaussian likelihood creation."""
    # Simple linear model
    def model_fn(params):
        return params['m'] * jnp.array([1, 2, 3]) + params['b']

    observed_data = jnp.array([2.1, 4.0, 6.05])
    errors = jnp.array([0.1, 0.1, 0.1])

    likelihood_fn = create_gaussian_likelihood(model_fn, observed_data, errors)

    # Test at approximately correct parameters
    params = {'m': 2.0, 'b': 0.0}
    log_like = likelihood_fn(params)
    assert jnp.isfinite(log_like)

    # Likelihood should be higher for better fit
    params_bad = {'m': 1.0, 'b': 0.0}
    log_like_bad = likelihood_fn(params_bad)
    assert log_like > log_like_bad


def test_create_gaussian_likelihood_jit():
    """Test that likelihood function is JIT-compilable."""
    def model_fn(params):
        return params['a'] * jnp.ones(10)

    observed_data = jnp.ones(10)
    errors = 0.1 * jnp.ones(10)

    likelihood_fn = create_gaussian_likelihood(model_fn, observed_data, errors)

    # JIT compile and run
    jit_likelihood = jax.jit(likelihood_fn)
    params = {'a': 1.0}

    log_like = jit_likelihood(params)
    assert jnp.isfinite(log_like)

    # Run again to ensure compilation works
    log_like_2 = jit_likelihood({'a': 1.1})
    assert log_like > log_like_2  # Perfect fit should have higher likelihood


@pytest.mark.slow
def test_run_nested_sampling_simple():
    """Test importance sampling on a simple Gaussian problem.

    This is a simple test that the sampler runs without errors.
    Tests the importance sampling approach for evidence estimation.
    """
    # True parameters
    true_a = 2.0
    true_b = 3.0

    # Generate synthetic data
    x = jnp.linspace(0, 10, 20)
    y_true = true_a * x + true_b
    noise = 0.1
    y_obs = y_true + noise * jax.random.normal(jax.random.PRNGKey(123), shape=y_true.shape)
    y_err = noise * jnp.ones_like(y_obs)

    # Model function
    def model_fn(params):
        return params['a'] * x + params['b']

    # Prior bounds (wide enough)
    prior_bounds = {
        'a': (0.0, 5.0),
        'b': (0.0, 10.0)
    }

    # Create likelihood
    likelihood_fn = create_gaussian_likelihood(model_fn, y_obs, y_err)

    # Run importance sampling (with reduced settings for speed)
    result = run_nested_sampling(
        likelihood_fn,
        prior_bounds,
        n_particles=100,  # Small number for test speed
        num_mcmc_steps=10,
        max_iterations=100,
        rng_key=jax.random.PRNGKey(42),
        verbose=False
    )

    # Check result structure
    assert isinstance(result, SamplerResult)
    assert 'a' in result.samples
    assert 'b' in result.samples
    assert len(result.samples['a']) > 0
    assert len(result.samples['b']) > 0
    assert jnp.isfinite(result.log_evidence)

    # Check that samples are within prior bounds
    assert jnp.all(result.samples['a'] >= 0.0)
    assert jnp.all(result.samples['a'] <= 5.0)
    assert jnp.all(result.samples['b'] >= 0.0)
    assert jnp.all(result.samples['b'] <= 10.0)

    # Check metadata
    assert result.metadata['param_names'] == ['a', 'b']
    assert result.metadata['n_particles'] == 100


def test_summarize_result():
    """Test result summarization."""
    # Create mock result
    n_samples = 1000
    samples = {
        'a': jnp.array(np.random.normal(2.0, 0.1, n_samples)),
        'b': jnp.array(np.random.normal(3.0, 0.2, n_samples))
    }

    result = SamplerResult(
        samples=samples,
        log_likelihoods=jnp.zeros(n_samples),
        log_weights=jnp.zeros(n_samples),
        log_evidence=0.0,
        log_evidence_error=0.1,
        metadata={
            'param_names': ['a', 'b'],
            'prior_bounds': {'a': (0, 10), 'b': (0, 10)},
        }
    )

    summary = summarize_result(result)

    # Check structure
    assert 'a' in summary
    assert 'b' in summary
    assert 'mean' in summary['a']
    assert 'std' in summary['a']
    assert 'median' in summary['a']

    # Check approximate values
    assert jnp.isclose(summary['a']['mean'], 2.0, atol=0.1)
    assert jnp.isclose(summary['b']['mean'], 3.0, atol=0.1)
    assert jnp.isclose(summary['a']['std'], 0.1, atol=0.05)
    assert jnp.isclose(summary['b']['std'], 0.2, atol=0.05)


def test_sampler_result_structure():
    """Test SamplerResult named tuple structure."""
    samples = {'a': jnp.array([1, 2, 3])}
    log_likes = jnp.array([-1, -2, -3])
    log_weights = jnp.array([0, 0, 0])

    result = SamplerResult(
        samples=samples,
        log_likelihoods=log_likes,
        log_weights=log_weights,
        log_evidence=-10.0,
        log_evidence_error=0.5,
        metadata={'test': 'data'}
    )

    # Check access
    assert result.samples == samples
    assert jnp.array_equal(result.log_likelihoods, log_likes)
    assert result.log_evidence == -10.0
    assert result.metadata['test'] == 'data'


def test_likelihood_with_source():
    """Test likelihood function with PrecomputedSpectraSource."""
    from redback_jax.sources import PrecomputedSpectraSource

    # Create a simple source
    phases = np.linspace(0, 40, 40)
    wavelengths = np.linspace(3000, 9000, 100)
    flux_grid = np.ones((40, 100)) * 1e-15

    source = PrecomputedSpectraSource(
        phases=phases,
        wavelengths=wavelengths,
        flux_grid=flux_grid
    )

    # Prepare for bandflux calculation
    unique_bands = ['g', 'r']
    bridges, band_to_idx = source.prepare_bridges(unique_bands)

    obs_phases = jnp.array([10.0, 20.0, 30.0])
    band_indices = jnp.array([0, 1, 0])

    # Model function
    def model_fn(params):
        return source.bandflux(
            params, None, obs_phases,
            band_indices=band_indices,
            bridges=bridges,
            unique_bands=unique_bands
        )

    # Synthetic observations
    true_params = {'amplitude': 2.0}
    obs_fluxes = model_fn(true_params)
    obs_errors = 0.05 * obs_fluxes

    # Create likelihood
    likelihood_fn = create_gaussian_likelihood(model_fn, obs_fluxes, obs_errors)

    # Test likelihood at true params (should be high)
    log_like_true = likelihood_fn(true_params)

    # Test likelihood at wrong params (should be lower)
    log_like_wrong = likelihood_fn({'amplitude': 1.0})

    assert log_like_true > log_like_wrong


def test_prior_transform_vectorization():
    """Test that prior transform works with JAX operations."""
    prior_bounds = {
        'x': (0.0, 1.0),
        'y': (0.0, 2.0)
    }

    prior_fn = create_uniform_prior(prior_bounds)

    # Test with vmap
    u_samples = jax.random.uniform(jax.random.PRNGKey(0), shape=(10, 2))

    # Map over samples
    def transform_single(u):
        params = prior_fn(u)
        return jnp.array([params['x'], params['y']])

    transformed = jax.vmap(transform_single)(u_samples)
    assert transformed.shape == (10, 2)

    # Check bounds
    assert jnp.all(transformed[:, 0] >= 0.0)
    assert jnp.all(transformed[:, 0] <= 1.0)
    assert jnp.all(transformed[:, 1] >= 0.0)
    assert jnp.all(transformed[:, 1] <= 2.0)


def test_likelihood_gradient():
    """Test that gradients can be computed through likelihood."""
    def model_fn(params):
        return params['a'] * jnp.array([1.0, 2.0, 3.0])

    obs_data = jnp.array([2.0, 4.0, 6.0])
    errors = jnp.array([0.1, 0.1, 0.1])

    likelihood_fn = create_gaussian_likelihood(model_fn, obs_data, errors)

    # Compute gradient
    grad_fn = jax.grad(lambda a: likelihood_fn({'a': a}))
    grad_at_true = grad_fn(2.0)

    # At true value, gradient should be near zero
    assert jnp.abs(grad_at_true) < 0.1


def test_multiple_parameter_inference():
    """Test inference with multiple parameters."""
    prior_bounds = {
        'a': (0.0, 10.0),
        'b': (-5.0, 5.0),
        'c': (1.0, 100.0)
    }

    prior_fn = create_uniform_prior(prior_bounds)

    # Check parameter order is preserved
    u = jnp.array([0.5, 0.5, 0.5])
    params = prior_fn(u)

    assert jnp.isclose(params['a'], 5.0)
    assert jnp.isclose(params['b'], 0.0)
    assert jnp.isclose(params['c'], 50.5)
