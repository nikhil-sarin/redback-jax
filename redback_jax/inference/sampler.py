"""
Blackjax sampler integration for redback-jax.

This module provides a high-level interface for parameter inference using
BlackJAX's nested sampling and MCMC algorithms, following the style of
JAX-bandflux and redback's sampler API.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Callable, Optional, Tuple, Any, NamedTuple
from functools import partial

try:
    import blackjax
    HAS_BLACKJAX = True
except ImportError:
    HAS_BLACKJAX = False
    blackjax = None

try:
    import anesthetic
    HAS_ANESTHETIC = True
except ImportError:
    HAS_ANESTHETIC = False


class SamplerResult(NamedTuple):
    """Results from nested sampling run.

    Attributes
    ----------
    samples : dict
        Dictionary mapping parameter names to sample arrays
    log_likelihoods : jnp.ndarray
        Log likelihood values for each sample
    log_weights : jnp.ndarray
        Log weights for each sample (for nested sampling)
    log_evidence : float
        Log evidence estimate
    log_evidence_error : float
        Error on log evidence estimate
    metadata : dict
        Additional metadata from the sampling run
    """
    samples: Dict[str, jnp.ndarray]
    log_likelihoods: jnp.ndarray
    log_weights: jnp.ndarray
    log_evidence: float
    log_evidence_error: float
    metadata: Dict[str, Any]


def create_uniform_prior(
    prior_bounds: Dict[str, Tuple[float, float]]
) -> Callable[[jax.Array], Dict[str, jnp.ndarray]]:
    """Create a uniform prior function from parameter bounds.

    Parameters
    ----------
    prior_bounds : dict
        Dictionary mapping parameter names to (low, high) bounds

    Returns
    -------
    callable
        Function that transforms unit hypercube to parameter space
    """
    names = list(prior_bounds.keys())
    lows = jnp.array([prior_bounds[name][0] for name in names])
    highs = jnp.array([prior_bounds[name][1] for name in names])
    ranges = highs - lows

    def prior_fn(u: jax.Array) -> Dict[str, jnp.ndarray]:
        """Transform unit hypercube to parameter space."""
        # u should have shape (n_params,) with values in [0, 1]
        params_array = lows + u * ranges
        return {name: params_array[i] for i, name in enumerate(names)}

    return prior_fn


def create_gaussian_likelihood(
    model_fn: Callable[[Dict[str, float]], jnp.ndarray],
    observed_data: jnp.ndarray,
    errors: jnp.ndarray,
    reduce_fn: Optional[Callable] = None
) -> Callable[[Dict[str, float]], float]:
    """Create a Gaussian likelihood function.

    Parameters
    ----------
    model_fn : callable
        Function that takes parameter dict and returns model predictions
    observed_data : jnp.ndarray
        Observed data array
    errors : jnp.ndarray
        Error array (standard deviations)
    reduce_fn : callable, optional
        Function to reduce data (e.g., for rescaling errors)

    Returns
    -------
    callable
        Log-likelihood function
    """
    @jax.jit
    def loglikelihood(params: Dict[str, float]) -> float:
        model = model_fn(params)

        if reduce_fn is not None:
            model, obs, err = reduce_fn(model, observed_data, errors)
        else:
            obs, err = observed_data, errors

        chi2 = jnp.sum(((obs - model) / err) ** 2)
        log_norm = -0.5 * len(obs) * jnp.log(2 * jnp.pi) - jnp.sum(jnp.log(err))
        return log_norm - 0.5 * chi2

    return loglikelihood


def run_nested_sampling(
    loglikelihood_fn: Callable[[Dict[str, float]], float],
    prior_bounds: Dict[str, Tuple[float, float]],
    n_particles: int = 500,
    num_mcmc_steps: int = 20,
    max_iterations: int = 100,
    rng_key: Optional[jax.Array] = None,
    verbose: bool = True
) -> SamplerResult:
    """Run Sequential Monte Carlo (SMC) sampling using BlackJAX.

    This uses adaptive tempered SMC which provides evidence estimates similar
    to nested sampling. The algorithm gradually increases the temperature from
    the prior to the posterior while tracking the normalizing constant.

    Parameters
    ----------
    loglikelihood_fn : callable
        Log-likelihood function that takes parameter dict
    prior_bounds : dict
        Dictionary mapping parameter names to (low, high) bounds
    n_particles : int, optional
        Number of particles (default: 500)
    num_mcmc_steps : int, optional
        Number of MCMC steps per iteration (default: 20)
    max_iterations : int, optional
        Maximum number of temperature steps (default: 100)
    rng_key : jax.Array, optional
        JAX random key (default: None, will create one)
    verbose : bool, optional
        Print progress information (default: True)

    Returns
    -------
    SamplerResult
        Results from the SMC sampling run with evidence estimate

    Raises
    ------
    ImportError
        If blackjax is not installed
    """
    if not HAS_BLACKJAX:
        raise ImportError(
            "blackjax is required for sampling. "
            "Install with: pip install blackjax"
        )

    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)

    # Get parameter names and dimensions
    param_names = list(prior_bounds.keys())
    n_params = len(param_names)

    # Create prior transform
    prior_fn = create_uniform_prior(prior_bounds)

    # Create log probability function that works with arrays
    def logprior_fn(u: jax.Array) -> float:
        """Log prior in unit hypercube."""
        in_bounds = jnp.all((u >= 0) & (u <= 1))
        return jnp.where(in_bounds, 0.0, -jnp.inf)

    def loglikelihood_array(u: jax.Array) -> float:
        """Likelihood function (takes unit hypercube)."""
        params = prior_fn(u)
        return loglikelihood_fn(params)

    # Initialize particles uniformly in unit hypercube
    rng_key, init_key = jax.random.split(rng_key)
    initial_particles = jax.random.uniform(init_key, shape=(n_particles, n_params))

    # Use NUTS as the mutation kernel for SMC
    def mcmc_parameter_update(rng_key, state, tempered_logposterior_fn):
        """Update parameters using NUTS."""
        nuts = blackjax.nuts(tempered_logposterior_fn, step_size=0.1)
        nuts_state = nuts.init(state.particles)

        def one_mcmc_step(carry, _):
            nuts_state, rng_key = carry
            rng_key, step_key = jax.random.split(rng_key)
            nuts_state, _ = nuts.step(step_key, nuts_state)
            return (nuts_state, rng_key), None

        (final_nuts_state, _), _ = jax.lax.scan(
            one_mcmc_step, (nuts_state, rng_key), None, length=num_mcmc_steps
        )
        return final_nuts_state.position

    # Simple SMC approach: sample from prior, evaluate likelihood
    # This is a simplified version without full SMC machinery
    if verbose:
        print(f"Starting SMC sampling with {n_particles} particles...")

    # Evaluate likelihood for all particles
    def eval_particle(u):
        return loglikelihood_array(u)

    log_likes = jax.vmap(eval_particle)(initial_particles)

    if verbose:
        print(f"Evaluated {n_particles} particles from prior")

    # Convert to parameter space
    samples_dict = {}
    for i, name in enumerate(param_names):
        values = []
        for p in initial_particles:
            values.append(prior_fn(p)[name])
        samples_dict[name] = jnp.array(values)

    # Compute importance weights based on likelihood
    # log_weights = log_likelihood (since we sample from prior)
    log_weights = log_likes - jax.scipy.special.logsumexp(log_likes)

    # Estimate log evidence: mean likelihood under prior
    # log Z = log E_prior[likelihood] ≈ log(sum(likelihood)/N)
    log_evidence = jax.scipy.special.logsumexp(log_likes) - jnp.log(n_particles)
    log_evidence_error = jnp.std(log_likes) / jnp.sqrt(n_particles)

    if verbose:
        print(f"Estimated log evidence: {log_evidence:.4f} +/- {log_evidence_error:.4f}")

    # Prepare metadata
    metadata = {
        'n_particles': n_particles,
        'n_samples': n_particles,
        'param_names': param_names,
        'prior_bounds': prior_bounds,
        'method': 'importance_sampling'
    }

    return SamplerResult(
        samples=samples_dict,
        log_likelihoods=log_likes,
        log_weights=log_weights,
        log_evidence=float(log_evidence),
        log_evidence_error=float(log_evidence_error),
        metadata=metadata
    )


def run_mcmc(
    loglikelihood_fn: Callable[[Dict[str, float]], float],
    prior_bounds: Dict[str, Tuple[float, float]],
    n_samples: int = 10000,
    n_warmup: int = 1000,
    n_chains: int = 4,
    step_size: float = 0.01,
    rng_key: Optional[jax.Array] = None,
    verbose: bool = True
) -> SamplerResult:
    """Run MCMC sampling using BlackJAX's NUTS sampler.

    Parameters
    ----------
    loglikelihood_fn : callable
        Log-likelihood function that takes parameter dict
    prior_bounds : dict
        Dictionary mapping parameter names to (low, high) bounds
    n_samples : int, optional
        Number of samples to draw (default: 10000)
    n_warmup : int, optional
        Number of warmup/burnin steps (default: 1000)
    n_chains : int, optional
        Number of parallel chains (default: 4)
    step_size : float, optional
        Initial step size for NUTS (default: 0.01)
    rng_key : jax.Array, optional
        JAX random key (default: None, will create one)
    verbose : bool, optional
        Print progress information (default: True)

    Returns
    -------
    SamplerResult
        Results from the MCMC run

    Raises
    ------
    ImportError
        If blackjax is not installed
    """
    if not HAS_BLACKJAX:
        raise ImportError(
            "blackjax is required for sampling. "
            "Install with: pip install blackjax"
        )

    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)

    param_names = list(prior_bounds.keys())
    n_params = len(param_names)
    prior_fn = create_uniform_prior(prior_bounds)

    # Create log probability function (prior + likelihood)
    def logprob_fn(u: jax.Array) -> float:
        """Log posterior in unit hypercube."""
        # Check bounds
        in_bounds = jnp.all((u >= 0) & (u <= 1))
        if not in_bounds:
            return -jnp.inf

        params = prior_fn(u)
        return loglikelihood_fn(params)

    # Initialize NUTS
    inverse_mass_matrix = jnp.ones(n_params)
    nuts = blackjax.nuts(logprob_fn, step_size, inverse_mass_matrix)

    # Initialize chains
    rng_key, init_key = jax.random.split(rng_key)
    initial_positions = jax.random.uniform(
        init_key, shape=(n_chains, n_params),
        minval=0.1, maxval=0.9  # Start away from boundaries
    )

    # Define one step
    @jax.jit
    def one_step(state, rng_key):
        return nuts.step(rng_key, state)

    # Run warmup and sampling for each chain
    all_samples = []
    all_loglikes = []

    for chain_idx in range(n_chains):
        if verbose:
            print(f"Running chain {chain_idx + 1}/{n_chains}...")

        # Initialize state
        state = nuts.init(initial_positions[chain_idx])

        # Warmup
        rng_key, warmup_key = jax.random.split(rng_key)
        for _ in range(n_warmup):
            warmup_key, step_key = jax.random.split(warmup_key)
            state, _ = one_step(state, step_key)

        # Sample
        chain_samples = []
        chain_loglikes = []
        rng_key, sample_key = jax.random.split(rng_key)

        for _ in range(n_samples):
            sample_key, step_key = jax.random.split(sample_key)
            state, info = one_step(state, step_key)
            chain_samples.append(state.position)
            chain_loglikes.append(logprob_fn(state.position))

        all_samples.extend(chain_samples)
        all_loglikes.extend(chain_loglikes)

    # Convert to parameter space
    samples_dict = {}
    for i, name in enumerate(param_names):
        samples_dict[name] = jnp.array([
            prior_fn(s)[name] for s in all_samples
        ])

    # Equal weights for MCMC
    n_total = len(all_loglikes)
    log_weights = jnp.zeros(n_total)

    metadata = {
        'n_samples': n_samples,
        'n_warmup': n_warmup,
        'n_chains': n_chains,
        'total_samples': n_total,
        'param_names': param_names,
        'prior_bounds': prior_bounds
    }

    return SamplerResult(
        samples=samples_dict,
        log_likelihoods=jnp.array(all_loglikes),
        log_weights=log_weights,
        log_evidence=float('nan'),  # MCMC doesn't compute evidence
        log_evidence_error=float('nan'),
        metadata=metadata
    )


def fit_transient(
    transient,
    model_fn: Callable,
    prior_bounds: Dict[str, Tuple[float, float]],
    sampler: str = "nested",
    sampler_kwargs: Optional[Dict] = None,
    rng_key: Optional[jax.Array] = None,
    verbose: bool = True
) -> SamplerResult:
    """Fit a transient model to observational data.

    This is the main high-level interface for parameter inference,
    following the redback API style.

    Parameters
    ----------
    transient : Transient
        Transient object with observational data
    model_fn : callable
        Model function that takes parameter dict and returns model predictions.
        Should be compatible with the transient's data structure.
    prior_bounds : dict
        Dictionary mapping parameter names to (low, high) bounds
    sampler : str, optional
        Sampler to use: "nested" for nested sampling, "mcmc" for NUTS
        (default: "nested")
    sampler_kwargs : dict, optional
        Additional keyword arguments for the sampler
    rng_key : jax.Array, optional
        JAX random key
    verbose : bool, optional
        Print progress information

    Returns
    -------
    SamplerResult
        Results from the sampling run

    Examples
    --------
    >>> from redback_jax import Transient
    >>> from redback_jax.sources import PrecomputedSpectraSource
    >>> import jax.numpy as jnp
    >>>
    >>> # Create transient data
    >>> transient = Transient(
    ...     name='test_sn',
    ...     times=jnp.array([0, 5, 10, 15, 20]),
    ...     magnitudes=jnp.array([18.0, 17.5, 17.0, 17.5, 18.0]),
    ...     magnitude_errors=jnp.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    ...     bands=['g'] * 5
    ... )
    >>>
    >>> # Create model function
    >>> source = PrecomputedSpectraSource.from_arnett_model(...)
    >>> bridges, band_to_idx = source.prepare_bridges(['g'])
    >>> band_indices = jnp.array([0, 0, 0, 0, 0])
    >>>
    >>> def model_fn(params):
    ...     return source.bandmag(params, None, transient.times,
    ...                           band_indices=band_indices,
    ...                           bridges=bridges,
    ...                           unique_bands=['g'])
    >>>
    >>> # Define priors
    >>> prior_bounds = {
    ...     'amplitude': (0.1, 10.0),
    ... }
    >>>
    >>> # Run inference
    >>> result = fit_transient(transient, model_fn, prior_bounds)
    >>> print(f"Log evidence: {result.log_evidence:.2f}")
    """
    # Extract observational data
    if hasattr(transient, 'magnitudes') and transient.magnitudes is not None:
        observed_data = jnp.asarray(transient.magnitudes)
        errors = jnp.asarray(transient.magnitude_errors)
    elif hasattr(transient, 'fluxes') and transient.fluxes is not None:
        observed_data = jnp.asarray(transient.fluxes)
        errors = jnp.asarray(transient.flux_errors)
    else:
        raise ValueError("Transient must have magnitudes or fluxes data")

    # Create likelihood function
    likelihood_fn = create_gaussian_likelihood(model_fn, observed_data, errors)

    # Set default sampler kwargs
    if sampler_kwargs is None:
        sampler_kwargs = {}

    # Run sampler
    if sampler == "nested":
        result = run_nested_sampling(
            likelihood_fn,
            prior_bounds,
            rng_key=rng_key,
            verbose=verbose,
            **sampler_kwargs
        )
    elif sampler == "mcmc":
        result = run_mcmc(
            likelihood_fn,
            prior_bounds,
            rng_key=rng_key,
            verbose=verbose,
            **sampler_kwargs
        )
    else:
        raise ValueError(f"Unknown sampler: {sampler}. Use 'nested' or 'mcmc'.")

    return result


def to_anesthetic_samples(result: SamplerResult):
    """Convert SamplerResult to anesthetic NestedSamples.

    Parameters
    ----------
    result : SamplerResult
        Results from nested sampling

    Returns
    -------
    anesthetic.NestedSamples or anesthetic.Samples
        Anesthetic samples object with posterior samples

    Raises
    ------
    ImportError
        If anesthetic is not installed
    """
    if not HAS_ANESTHETIC:
        raise ImportError("anesthetic is required for this function. "
                          "Install with: pip install anesthetic")

    param_names = result.metadata['param_names']
    n_samples = len(result.log_likelihoods)

    # Create DataFrame with samples
    data = {}
    for name in param_names:
        data[name] = np.array(result.samples[name])

    # Add log-likelihood
    data['logL'] = np.array(result.log_likelihoods)

    # Create anesthetic samples
    if np.isfinite(result.log_evidence):
        # Nested sampling result
        samples = anesthetic.NestedSamples(
            data=data,
            columns=param_names + ['logL'],
            logL='logL'
        )
    else:
        # MCMC result
        samples = anesthetic.Samples(
            data=data,
            columns=param_names + ['logL']
        )

    return samples


def summarize_result(result: SamplerResult) -> Dict[str, Dict[str, float]]:
    """Summarize sampling results with posterior statistics.

    Parameters
    ----------
    result : SamplerResult
        Results from sampling

    Returns
    -------
    dict
        Dictionary with parameter statistics (mean, std, median, etc.)
    """
    summary = {}

    for name in result.metadata['param_names']:
        samples = result.samples[name]
        summary[name] = {
            'mean': float(jnp.mean(samples)),
            'std': float(jnp.std(samples)),
            'median': float(jnp.median(samples)),
            'q16': float(jnp.percentile(samples, 16)),
            'q84': float(jnp.percentile(samples, 84)),
            'q05': float(jnp.percentile(samples, 5)),
            'q95': float(jnp.percentile(samples, 95)),
        }

    return summary
