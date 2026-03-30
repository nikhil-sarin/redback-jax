"""
High-level nested sampler for redback-jax.

Usage::

    from redback_jax.inference import Prior, Uniform, SpectralLikelihood, NestedSampler
    import jax

    prior = Prior([
        Uniform(58580, 58620, name='t0'),
        Uniform(0.05,  0.20,  name='f_nickel'),
        Uniform(0.8,   2.0,   name='mej'),
        Uniform(3000,  8000,  name='vej'),
    ])

    likelihood = SpectralLikelihood(
        model='arnett_spectra',
        transient=transient,
        fixed_params={
            'redshift':          0.01,
            'lum_dist':          dl_cm,
            'temperature_floor': 5000.0,
            'kappa':             0.07,
            'kappa_gamma':       0.1,
        },
        bridge_params={'vej': 5000.0},  # needed when vej is a free param
    )

    sampler = NestedSampler(
        likelihood,
        prior,
        outdir   = 'results/',
        n_live   = 125,
        n_delete = 20,
        num_mcmc_steps_multiplier = 5,
    )
    result = sampler.run(jax.random.PRNGKey(0))

    # Corner plot (requires anesthetic)
    sampler.plot_corner(result, truth={'t0': 58600.0, 'f_nickel': 0.1})
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

try:
    import blackjax
    from blackjax.ns.utils import log_weights as _bj_log_weights
    HAS_BLACKJAX = True
except ImportError:
    HAS_BLACKJAX = False

try:
    import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from jax_supernovae.utils import save_chains_dead_birth as _save_chains
    HAS_JSN_UTILS = True
except ImportError:
    HAS_JSN_UTILS = False


class NSResult:
    """Container for nested sampling results.

    Attributes
    ----------
    logZ : float
        Log evidence estimate.
    samples : dict
        Posterior samples as ``{name: jnp.ndarray}``.
    dead : object
        Raw dead-point pytree from BlackJAX (for expert use).
    log_weights : jnp.ndarray
        Log importance weights (shape ``(n_dead,)``).
    param_names : list of str
        Ordered parameter names.
    """

    def __init__(self, logZ, samples, dead, log_weights, param_names):
        self.logZ        = logZ
        self.samples     = samples
        self.dead        = dead
        self.log_weights = log_weights
        self.param_names = param_names

    def summary(self):
        """Print a parameter summary table."""
        print(f"\n{'Param':<14} {'Mean':>12} {'Std':>10} {'q16':>10} {'q84':>10}")
        print("-" * 58)
        for name in self.param_names:
            s  = self.samples[name]
            w  = jnp.exp(self.log_weights - jax.scipy.special.logsumexp(self.log_weights))
            mu = float(jnp.sum(w * s))
            sq = float(jnp.sum(w * (s - mu) ** 2)) ** 0.5
            q16 = float(jnp.percentile(s, 16))
            q84 = float(jnp.percentile(s, 84))
            print(f"{name:<14} {mu:>12.4f} {sq:>10.4f} {q16:>10.4f} {q84:>10.4f}")

    def __repr__(self) -> str:
        return f"NSResult(logZ={self.logZ:.2f}, n_samples={len(next(iter(self.samples.values())))})"


class NestedSampler:
    """BlackJAX nested sampler with a clean redback-style interface.

    Parameters
    ----------
    likelihood : Likelihood
        A :class:`~redback_jax.inference.Likelihood` instance.
    prior : Prior
        Composite prior object.
    outdir : str, optional
        Directory for output files.  Created if it does not exist.
        Set to ``None`` to disable file output.
    n_live : int, optional
        Number of live points (default 125).
    n_delete : int, optional
        Number of points to remove per iteration (default 20).
    num_mcmc_steps_multiplier : int, optional
        MCMC steps per iteration = ``n_params × multiplier`` (default 5).
    termination_dlogz : float, optional
        Stop when ``logZ_live - logZ < termination_dlogz`` (default -3).
    verbose : bool, optional
        Show a tqdm progress bar (default True).

    Examples
    --------
    >>> sampler = NestedSampler(likelihood, prior, outdir='results/')
    >>> result  = sampler.run(jax.random.PRNGKey(42))
    >>> result.summary()
    """

    def __init__(
        self,
        likelihood,
        prior,
        outdir: str = 'results/',
        n_live: int = 125,
        n_delete: int = 20,
        num_mcmc_steps_multiplier: int = 5,
        termination_dlogz: float = -3.0,
        verbose: bool = True,
    ):
        if not HAS_BLACKJAX:
            raise ImportError(
                "blackjax is required for nested sampling.\n"
                "Install with: pip install git+https://github.com/handley-lab/blackjax@proposal"
            )

        self.likelihood   = likelihood
        self.prior        = prior
        self.outdir       = outdir
        self.n_live       = n_live
        self.n_delete     = n_delete
        self.n_mcmc_steps = prior.n_params * num_mcmc_steps_multiplier
        self.term_dlogz   = termination_dlogz
        self.verbose      = verbose

        # Build JAX-traceable prior and likelihood functions
        self._log_prior_fn = prior.log_prob_fn()
        self._log_like_fn  = likelihood._make_log_likelihood(prior)

        # BlackJAX NS algorithm
        self._algo = blackjax.nss(
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_like_fn,
            num_inner_steps=self.n_mcmc_steps,
            num_delete=self.n_delete,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, key: jax.Array) -> NSResult:
        """Run nested sampling.

        Parameters
        ----------
        key : jax.Array
            JAX random key.

        Returns
        -------
        NSResult
            Posterior samples and evidence estimate.
        """
        # Draw initial live points from the prior
        key, init_key = jax.random.split(key)
        initial_particles = self.prior.sample_n(init_key, self.n_live)  # (n_live, n_params)
        state = self._algo.init(initial_particles)

        if self.verbose:
            print(f"Nested sampling: {self.n_live} live points, "
                  f"{self.n_mcmc_steps} MCMC steps/iter, "
                  f"device: {jax.devices()[0]}")

        dead = []
        if self.verbose and HAS_TQDM:
            pbar = _tqdm.tqdm(desc="Dead points", unit=" pts")
        else:
            pbar = None

        while state.logZ_live - state.logZ > self.term_dlogz:
            key, subkey = jax.random.split(key)
            state, dead_point = self._algo.step(subkey, state)
            dead.append(dead_point)
            if pbar is not None:
                pbar.update(self.n_delete)

        if pbar is not None:
            pbar.close()

        if self.verbose:
            print(f"\nlogZ = {state.logZ:.2f}")

        # Collate dead points
        dead_all = jax.tree.map(lambda *args: jnp.concatenate(args), *dead)

        # Compute log-weights and evidence
        logw    = _bj_log_weights(key, dead_all)
        logZs   = jax.scipy.special.logsumexp(logw, axis=0)
        logZ    = float(logZs.mean()) if logZs.ndim > 0 else float(logZs)

        if self.verbose:
            if hasattr(logZs, 'std'):
                print(f"log Z = {logZ:.2f} ± {float(logZs.std()):.2f}")
            else:
                print(f"log Z = {logZ:.2f}")

        # Convert dead-point particle array to per-parameter samples
        # BlackJAX stores particles in dead_all.particles — shape (n_dead, n_params)
        particles = dead_all.particles   # (n_dead, n_params)
        samples = {
            name: particles[:, i]
            for i, name in enumerate(self.prior.names)
        }

        # Save chains
        if self.outdir is not None and HAS_JSN_UTILS:
            os.makedirs(self.outdir, exist_ok=True)
            chains_dir = os.path.join(self.outdir, 'chains')
            os.makedirs(chains_dir, exist_ok=True)
            try:
                _save_chains(dead_all, self.prior.names, root=chains_dir + '/chains')
                if self.verbose:
                    print(f"Chains saved to {chains_dir}/")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: could not save chains: {e}")

        return NSResult(
            logZ=logZ,
            samples=samples,
            dead=dead_all,
            log_weights=logw,
            param_names=self.prior.names,
        )

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def plot_corner(self, result: NSResult, truth: dict = None,
                    filename: str = None, **kwargs):
        """Make a corner plot using anesthetic.

        Parameters
        ----------
        result : NSResult
            Output of :meth:`run`.
        truth : dict, optional
            True parameter values to mark on the plot.
        filename : str, optional
            Path to save the figure.  Defaults to ``{outdir}/corner.png``.
        """
        try:
            from anesthetic import read_chains, make_2d_axes
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("anesthetic and matplotlib are required for corner plots.\n"
                              "pip install anesthetic matplotlib")

        chains_root = None
        if self.outdir is not None:
            chains_root = os.path.join(self.outdir, 'chains', 'chains')

        if chains_root is not None and os.path.exists(chains_root + '.txt'):
            samples = read_chains(chains_root, columns=self.prior.names)
        else:
            # Fall back: build NestedSamples from raw arrays
            from anesthetic import NestedSamples
            data = {n: np.array(result.samples[n]) for n in self.prior.names}
            data['logL'] = np.array(result.dead.logL)
            data['logL_birth'] = np.array(result.dead.logL_birth)
            samples = NestedSamples(
                data=data,
                logL='logL',
                logL_birth='logL_birth',
                columns=self.prior.names,
            )

        fig, axes = make_2d_axes(self.prior.names,
                                  figsize=(3 * self.prior.n_params,) * 2,
                                  facecolor='w')
        samples.plot_2d(axes, alpha=0.9, label='posterior', **kwargs)

        if truth is not None:
            for i, name in enumerate(self.prior.names):
                if name not in truth:
                    continue
                tv = truth[name]
                axes.iloc[i, i].axvline(tv, color='red', linestyle='--', linewidth=2)
                for j in range(i):
                    axes.iloc[i, j].axhline(tv, color='red', linestyle='--',
                                             linewidth=1, alpha=0.5)
                    if self.prior.names[j] in truth:
                        axes.iloc[i, j].axvline(truth[self.prior.names[j]],
                                                  color='red', linestyle='--',
                                                  linewidth=1, alpha=0.5)

        plt.suptitle('Posterior', y=1.02)
        plt.tight_layout()

        if filename is None and self.outdir is not None:
            filename = os.path.join(self.outdir, 'corner.png')

        if filename is not None:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Corner plot saved to {filename}")

        return fig, axes
