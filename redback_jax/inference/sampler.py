"""
Blackjax sampler integration for redback-jax.

This module provides a high-level interface for parameter inference using
BlackJAX's nested sampling and MCMC algorithms, following the style of
JAX-bandflux and redback's sampler API.
"""

import warnings
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

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
    reduce_fn: Optional[Callable] = None,
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
    verbose: bool = True,
    target_ess: float = 0.5,
    step_size: float = 0.1,
) -> SamplerResult:
    """Run adaptive tempered Sequential Monte Carlo (SMC) using BlackJAX.

    Particles start at the prior and are annealed to the posterior over a
    sequence of temperatures chosen adaptively to hold the effective sample
    size near ``target_ess * n_particles``; each step mutates the particles
    with NUTS. The log evidence is the sum of the per-step log-likelihood
    increments.

    Sampling happens in an unconstrained space: each parameter is mapped to the
    real line with a logit transform, under which a uniform prior on the
    parameter is exactly a standard logistic prior. This keeps the target
    smooth for NUTS instead of the hard -inf walls of a bounded uniform.

    Parameters
    ----------
    loglikelihood_fn : callable
        Log-likelihood function that takes parameter dict
    prior_bounds : dict
        Dictionary mapping parameter names to (low, high) bounds
    n_particles : int, optional
        Number of particles (default: 500)
    num_mcmc_steps : int, optional
        NUTS steps used to mutate particles per temperature (default: 20)
    max_iterations : int, optional
        Maximum number of temperature steps (default: 100)
    rng_key : jax.Array, optional
        JAX random key (default: None, will create one)
    verbose : bool, optional
        Print progress information (default: True)
    target_ess : float, optional
        Effective sample size to hold, as a fraction of n_particles
        (default: 0.5)
    step_size : float, optional
        NUTS step size in the unconstrained space (default: 0.1)

    Returns
    -------
    SamplerResult
        Equally-weighted posterior samples and the evidence estimate.

    Raises
    ------
    ImportError
        If blackjax is not installed
    """
    if not HAS_BLACKJAX:
        raise ImportError(
            "blackjax is required for sampling. " "Install with: pip install blackjax"
        )

    from blackjax.smc import resampling

    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)

    param_names = list(prior_bounds.keys())
    n_params = len(param_names)
    lows = jnp.array([prior_bounds[name][0] for name in param_names])
    highs = jnp.array([prior_bounds[name][1] for name in param_names])

    def to_params(z: jax.Array) -> jax.Array:
        """Unconstrained z -> bounded parameter vector."""
        return lows + jax.nn.sigmoid(z) * (highs - lows)

    def logprior_fn(z: jax.Array) -> float:
        """Standard logistic prior: the logit-transform Jacobian of a uniform."""
        return jnp.sum(jax.nn.log_sigmoid(z) + jax.nn.log_sigmoid(-z))

    def loglikelihood_array(z: jax.Array) -> float:
        theta = to_params(z)
        return loglikelihood_fn({name: theta[i] for i, name in enumerate(param_names)})

    # Draw the prior directly: uniform in the parameter <=> standard logistic in z.
    rng_key, init_key = jax.random.split(rng_key)
    initial_particles = jax.random.logistic(init_key, shape=(n_particles, n_params))

    smc = blackjax.adaptive_tempered_smc(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_array,
        mcmc_step_fn=blackjax.nuts.build_kernel(),
        mcmc_init_fn=blackjax.nuts.init,
        # blackjax treats a leading dim of 1 as "shared across all particles".
        mcmc_parameters={
            "step_size": jnp.array([step_size]),
            "inverse_mass_matrix": jnp.ones((1, n_params)),
        },
        resampling_fn=resampling.systematic,
        target_ess=target_ess,
        num_mcmc_steps=num_mcmc_steps,
    )

    state = smc.init(initial_particles)
    step = jax.jit(smc.step)
    batch_loglike = jax.jit(jax.vmap(loglikelihood_array))

    if verbose:
        print(
            f"Starting adaptive tempered SMC: {n_particles} particles, "
            f"{num_mcmc_steps} NUTS steps/temperature"
        )

    log_evidence = 0.0
    # Per-step relative variance of the reweighting, accumulated for the
    # evidence error (see below).
    rel_var = 0.0
    n_steps = 0
    while state.tempering_param < 1.0 and n_steps < max_iterations:
        # Incremental weights for this step use the likelihood at the current
        # particles, so evaluate before the state is mutated.
        loglike_now = batch_loglike(state.particles)
        lam_prev = state.tempering_param

        rng_key, step_key = jax.random.split(rng_key)
        state, info = step(step_key, state)
        log_evidence += float(info.log_likelihood_increment)
        n_steps += 1

        d_lam = float(state.tempering_param) - float(lam_prev)
        logw = d_lam * loglike_now
        logw = logw - jax.scipy.special.logsumexp(logw)
        ess = float(1.0 / jnp.sum(jnp.exp(2.0 * logw)))
        rel_var += 1.0 / ess - 1.0 / n_particles

        if verbose:
            print(
                f"  step {n_steps}: lambda={float(state.tempering_param):.4f} "  # noqa: E231,E501
                f"ess={ess:.1f} logZ={log_evidence:.3f}"  # noqa: E231
            )

    if state.tempering_param < 1.0:
        temp = float(state.tempering_param)
        msg = (
            f"SMC stopped at temperature {temp} after {max_iterations} "
            "iterations before reaching the posterior. "
            "Raise max_iterations or n_particles."
        )
        warnings.warn(
            msg,
            RuntimeWarning,
            stacklevel=2,
        )

    # Var(log Z) ~ sum of the per-step relative variances of the reweighting
    # (delta method, treating the steps as independent) -- an approximation.
    log_evidence_error = float(jnp.sqrt(rel_var))

    # Particles are equally weighted after resampling, so these are posterior
    # samples directly -- no importance weights left to apply.
    final_params = jax.vmap(to_params)(state.particles)
    samples_dict = {name: final_params[:, i] for i, name in enumerate(param_names)}
    log_likes = batch_loglike(state.particles)
    log_weights = jnp.full((n_particles,), -jnp.log(n_particles))

    if verbose:
        print(
            f"Estimated log evidence: {log_evidence:.4f} +/- {log_evidence_error:.4f}"  # noqa: E231,E501
        )

    metadata = {
        "n_particles": n_particles,
        "n_samples": n_particles,
        "param_names": param_names,
        "prior_bounds": prior_bounds,
        "method": "adaptive_tempered_smc",
        "n_tempering_steps": n_steps,
        "target_ess": target_ess,
        "final_temperature": float(state.tempering_param),
    }

    return SamplerResult(
        samples=samples_dict,
        log_likelihoods=log_likes,
        log_weights=log_weights,
        log_evidence=float(log_evidence),
        log_evidence_error=log_evidence_error,
        metadata=metadata,
    )


def run_mcmc(
    loglikelihood_fn: Callable[[Dict[str, float]], float],
    prior_bounds: Dict[str, Tuple[float, float]],
    n_samples: int = 10000,
    n_warmup: int = 1000,
    n_chains: int = 4,
    step_size: float = 0.01,
    rng_key: Optional[jax.Array] = None,
    verbose: bool = True,
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
            "blackjax is required for sampling. " "Install with: pip install blackjax"
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
        init_key,
        shape=(n_chains, n_params),
        minval=0.1,
        maxval=0.9,  # Start away from boundaries
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
        samples_dict[name] = jnp.array([prior_fn(s)[name] for s in all_samples])

    # Equal weights for MCMC
    n_total = len(all_loglikes)
    log_weights = jnp.zeros(n_total)

    metadata = {
        "n_samples": n_samples,
        "n_warmup": n_warmup,
        "n_chains": n_chains,
        "total_samples": n_total,
        "param_names": param_names,
        "prior_bounds": prior_bounds,
    }

    return SamplerResult(
        samples=samples_dict,
        log_likelihoods=jnp.array(all_loglikes),
        log_weights=log_weights,
        log_evidence=float("nan"),  # MCMC doesn't compute evidence
        log_evidence_error=float("nan"),
        metadata=metadata,
    )


def fit_transient(
    transient,
    model_fn: Callable,
    prior_bounds: Dict[str, Tuple[float, float]],
    sampler: str = "nested",
    sampler_kwargs: Optional[Dict] = None,
    rng_key: Optional[jax.Array] = None,
    verbose: bool = True,
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
    if hasattr(transient, "magnitudes") and transient.magnitudes is not None:
        observed_data = jnp.asarray(transient.magnitudes)
        errors = jnp.asarray(transient.magnitude_errors)
    elif hasattr(transient, "fluxes") and transient.fluxes is not None:
        observed_data = jnp.asarray(transient.fluxes)
        errors = jnp.asarray(transient.flux_errors)
    elif hasattr(transient, "flux_densities") and transient.flux_densities is not None:
        observed_data = jnp.asarray(transient.flux_densities)
        errors = jnp.asarray(transient.flux_density_errors)
    else:
        raise ValueError(
            "Transient must have magnitudes, fluxes, or flux_densities data"
        )

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
            **sampler_kwargs,
        )
    elif sampler == "mcmc":
        result = run_mcmc(
            likelihood_fn,
            prior_bounds,
            rng_key=rng_key,
            verbose=verbose,
            **sampler_kwargs,
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
        raise ImportError(
            "anesthetic is required for this function. "
            "Install with: pip install anesthetic"
        )

    param_names = result.metadata["param_names"]

    # Create DataFrame with samples
    data = {}
    for name in param_names:
        data[name] = np.array(result.samples[name])

    # Add log-likelihood
    data["logL"] = np.array(result.log_likelihoods)

    # Create anesthetic samples
    if np.isfinite(result.log_evidence):
        # Nested sampling result
        samples = anesthetic.NestedSamples(
            data=data, columns=param_names + ["logL"], logL="logL"
        )
    else:
        # MCMC result
        samples = anesthetic.Samples(data=data, columns=param_names + ["logL"])

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

    for name in result.metadata["param_names"]:
        samples = result.samples[name]
        summary[name] = {
            "mean": float(jnp.mean(samples)),
            "std": float(jnp.std(samples)),
            "median": float(jnp.median(samples)),
            "q16": float(jnp.percentile(samples, 16)),
            "q84": float(jnp.percentile(samples, 84)),
            "q05": float(jnp.percentile(samples, 5)),
            "q95": float(jnp.percentile(samples, 95)),
        }

    return summary
