"""
JAX-based Bayesian inference tools using BlackJAX.
"""

from .sampler import (
    SamplerResult,
    create_uniform_prior,
    create_gaussian_likelihood,
    run_nested_sampling,
    run_mcmc,
    fit_transient,
    to_anesthetic_samples,
    summarize_result,
    HAS_BLACKJAX,
)

__all__ = [
    'SamplerResult',
    'create_uniform_prior',
    'create_gaussian_likelihood',
    'run_nested_sampling',
    'run_mcmc',
    'fit_transient',
    'to_anesthetic_samples',
    'summarize_result',
    'HAS_BLACKJAX',
]
