"""
MCMC sampler for redback-jax using BlackJAX NUTS.

Usage::

    from redback_jax.inference import Prior, Uniform, Likelihood, MCMCSampler
    import jax

    prior = Prior([
        Uniform(58580, 58620, name='t0'),
        Uniform(0.05,  0.20,  name='f_nickel'),
        Uniform(0.8,   2.0,   name='mej'),
        Uniform(3000,  8000,  name='vej'),
    ])

    likelihood = Likelihood(
        model='arnett_spectra',
        transient=transient,
        fixed_params=fixed,
    )

    sampler = MCMCSampler(likelihood, prior, n_warmup=500, n_samples=2000)
    result  = sampler.run(jax.random.PRNGKey(0))
    result.summary()
"""

import jax
import jax.numpy as jnp
import numpy as np

try:
    import blackjax
    HAS_BLACKJAX = True
except ImportError:
    HAS_BLACKJAX = False

try:
    import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class MCMCResult:
    """Container for MCMC results.

    Attributes
    ----------
    samples : dict
        Posterior samples as ``{name: jnp.ndarray}`` — shape ``(n_chains * n_samples,)``.
    samples_per_chain : dict
        Samples per chain as ``{name: jnp.ndarray}`` — shape ``(n_chains, n_samples)``.
    param_names : list of str
        Ordered parameter names.
    n_chains : int
        Number of chains.
    n_samples : int
        Number of post-warmup samples per chain.
    """

    def __init__(self, samples_per_chain, param_names):
        self.param_names      = param_names
        self.n_chains, self.n_samples = next(iter(samples_per_chain.values())).shape
        self.samples_per_chain = samples_per_chain
        self.samples = {
            name: arr.reshape(-1)
            for name, arr in samples_per_chain.items()
        }

    def summary(self):
        """Print a parameter summary table."""
        print(f"\n{'Param':<14} {'Mean':>12} {'Std':>10} {'q16':>10} {'q84':>10}")
        print("-" * 58)
        for name in self.param_names:
            s   = self.samples[name]
            mu  = float(jnp.mean(s))
            std = float(jnp.std(s))
            q16 = float(jnp.percentile(s, 16))
            q84 = float(jnp.percentile(s, 84))
            print(f"{name:<14} {mu:>12.4f} {std:>10.4f} {q16:>10.4f} {q84:>10.4f}")

    def __repr__(self) -> str:
        n = next(iter(self.samples.values())).shape[0]
        return f"MCMCResult(n_chains={self.n_chains}, n_samples_per_chain={self.n_samples}, total={n})"


class MCMCSampler:
    """BlackJAX NUTS sampler with a clean redback-style interface.

    The log-posterior is ``log_likelihood + log_prior``, evaluated in the
    original parameter space.  A reflected boundary is used to keep samples
    inside the prior support.

    Parameters
    ----------
    likelihood : Likelihood
        A :class:`~redback_jax.inference.Likelihood` instance.
    prior : Prior
        Composite prior object.
    n_warmup : int, optional
        Number of warmup (adaptation) steps per chain (default 500).
    n_samples : int, optional
        Number of post-warmup samples per chain (default 2000).
    n_chains : int, optional
        Number of independent chains (default 4).
    step_size : float, optional
        Initial NUTS step size (default 0.05).
    verbose : bool, optional
        Show a progress bar (default True).

    Examples
    --------
    >>> sampler = MCMCSampler(likelihood, prior, n_warmup=500, n_samples=2000)
    >>> result  = sampler.run(jax.random.PRNGKey(0))
    >>> result.summary()
    """

    def __init__(
        self,
        likelihood,
        prior,
        n_warmup: int = 500,
        n_samples: int = 2000,
        n_chains: int = 4,
        step_size: float = 0.05,
        verbose: bool = True,
    ):
        if not HAS_BLACKJAX:
            raise ImportError(
                "blackjax is required for MCMC sampling.\n"
                "Install with: pip install blackjax"
            )

        self.likelihood = likelihood
        self.prior      = prior
        self.n_warmup   = n_warmup
        self.n_samples  = n_samples
        self.n_chains   = n_chains
        self.step_size  = step_size
        self.verbose    = verbose

        log_prior_fn = prior.log_prob_fn()
        log_like_fn  = likelihood._make_log_likelihood(prior)

        # Log-posterior: returns -inf outside prior support automatically
        # because log_prior_fn returns -inf there.
        def _log_posterior(params):
            return log_like_fn(params) + log_prior_fn(params)

        self._log_posterior = _log_posterior

    # ------------------------------------------------------------------

    def run(self, key: jax.Array) -> MCMCResult:
        """Run MCMC.

        Parameters
        ----------
        key : jax.Array
            JAX random key.

        Returns
        -------
        MCMCResult
        """
        inverse_mass_matrix = jnp.ones(self.prior.n_params)
        nuts = blackjax.nuts(self._log_posterior, self.step_size, inverse_mass_matrix)

        # Draw initial positions from the prior
        key, init_key = jax.random.split(key)
        init_positions = self.prior.sample_n(init_key, self.n_chains)  # (n_chains, n_params)

        @jax.jit
        def one_step(state, step_key):
            new_state, info = nuts.step(step_key, state)
            return new_state, new_state.position

        if self.verbose:
            print(f"MCMC: {self.n_chains} chains, {self.n_warmup} warmup + "
                  f"{self.n_samples} samples, device: {jax.devices()[0]}")

        all_chains = []
        for chain_idx in range(self.n_chains):
            key, chain_key = jax.random.split(key)

            state = nuts.init(init_positions[chain_idx])

            # Warmup
            warmup_keys = jax.random.split(chain_key, self.n_warmup)
            for wk in warmup_keys:
                state, _ = one_step(state, wk)

            # Sample via lax.scan for efficiency
            key, sample_key = jax.random.split(key)
            sample_keys = jax.random.split(sample_key, self.n_samples)
            final_state, positions = jax.lax.scan(
                lambda s, k: one_step(s, k),
                state,
                sample_keys,
            )
            all_chains.append(positions)  # (n_samples, n_params)

            if self.verbose:
                print(f"  Chain {chain_idx + 1}/{self.n_chains} done")

        # all_chains: list of (n_samples, n_params) → stack to (n_chains, n_samples, n_params)
        stacked = jnp.stack(all_chains, axis=0)  # (n_chains, n_samples, n_params)

        samples_per_chain = {
            name: stacked[:, :, i]
            for i, name in enumerate(self.prior.names)
        }

        return MCMCResult(samples_per_chain, self.prior.names)
