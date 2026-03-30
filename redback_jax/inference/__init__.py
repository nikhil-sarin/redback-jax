"""
JAX-based Bayesian inference tools using BlackJAX.

Clean API
---------
    from redback_jax.inference import Prior, Uniform, LogUniform, Gaussian
    from redback_jax.inference import SpectralLikelihood, NestedSampler, NSResult

Legacy API (kept for backward compatibility)
--------------------------------------------
    from redback_jax.inference import (
        SamplerResult, create_uniform_prior, create_gaussian_likelihood,
        run_nested_sampling, run_mcmc, fit_transient, HAS_BLACKJAX,
    )
"""

# ------------------------------------------------------------------
# Clean API
# ------------------------------------------------------------------
from .prior import (
    Uniform,
    LogUniform,
    Gaussian,
    Prior,
)

from .likelihood import Likelihood

from .nested_sampler import (
    NestedSampler,
    NSResult,
    HAS_BLACKJAX,
)

from .mcmc_sampler import (
    MCMCSampler,
    MCMCResult,
)

# ------------------------------------------------------------------
# Legacy API
# ------------------------------------------------------------------
from .sampler import (
    SamplerResult,
    create_uniform_prior,
    create_gaussian_likelihood,
    run_nested_sampling,
    run_mcmc,
    fit_transient,
    to_anesthetic_samples,
    summarize_result,
)

__all__ = [
    # Clean API
    'Uniform',
    'LogUniform',
    'Gaussian',
    'Prior',
    'Likelihood',
    'NestedSampler',
    'NSResult',
    'MCMCSampler',
    'MCMCResult',
    # Shared
    'HAS_BLACKJAX',
    # Legacy API
    'SamplerResult',
    'create_uniform_prior',
    'create_gaussian_likelihood',
    'run_nested_sampling',
    'run_mcmc',
    'fit_transient',
    'to_anesthetic_samples',
    'summarize_result',
]
