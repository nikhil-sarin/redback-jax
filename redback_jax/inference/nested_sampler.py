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
from jax import tree_util

try:
    import blackjax  # noqa: F401
    from blackjax.ns.utils import finalise as _bj_finalise
    from blackjax.ns.utils import log_weights as _bj_log_weights

    try:
        # Newer BlackJAX NS API: state.sampler_state, dead.logL
        from blackjax.ns.adaptive import nss as _nss

        _NS_API = "adaptive"
    except ImportError:
        # handley-lab ``nested_sampling`` fork: state.integrator, dead.particles.*
        from blackjax.ns.nss import as_top_level_api as _nss

        _NS_API = "fork"

    HAS_BLACKJAX = True
    _BLACKJAX_NS_IMPORT_ERROR = None
except ImportError as _e:
    HAS_BLACKJAX = False
    _BLACKJAX_NS_IMPORT_ERROR = _e


# ---------------------------------------------------------------------------
# BlackJAX NS API compatibility
#
# Two incompatible layouts exist in the wild:
#   * newer API:  state.sampler_state.logZ,  dead.particles = positions array,
#                 dead.logL / dead.logL_birth
#   * fork API:   _ns_integrator(state).logZ,     dead.particles.position,
#                 dead.particles.loglikelihood / .loglikelihood_birth
# These helpers hide the difference so the samplers work with either.
# ---------------------------------------------------------------------------

def _build_nss(logprior_fn, loglikelihood_fn, n_mcmc_steps, n_delete,
               max_shrinkage=25, max_steps=10):
    """Construct the NS algorithm with whichever BlackJAX signature is installed.

    ``max_shrinkage`` / ``max_steps`` are only understood by the fork API and
    are ignored otherwise.
    """
    if _NS_API == "adaptive":
        return _nss(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            num_mcmc_steps=n_mcmc_steps,
            n_delete=n_delete,
        )
    return _nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        num_inner_steps=n_mcmc_steps,
        num_delete=n_delete,
        max_shrinkage=max_shrinkage,
        max_steps=max_steps,
    )


def _ns_integrator(state):
    """Return the object holding ``logZ`` / ``logZ_live``."""
    if hasattr(state, "sampler_state"):
        return state.sampler_state
    return state.integrator


def _dead_positions(dead):
    """(n_points, n_params) positions of the dead points."""
    particles = dead.particles
    return particles.position if hasattr(particles, "position") else particles


def _dead_logL(dead):
    if hasattr(dead, "logL"):
        return dead.logL
    return dead.particles.loglikelihood


def _dead_logL_birth(dead):
    if hasattr(dead, "logL_birth"):
        return dead.logL_birth
    return dead.particles.loglikelihood_birth


try:
    import tqdm as _tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from jax_supernovae.utils import (  # noqa: F401
        save_chains_dead_birth as _save_chains,
    )

    HAS_JSN_UTILS = True
except ImportError:
    HAS_JSN_UTILS = False

from redback_jax.inference.likelihood import make_batched_log_likelihood


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
        self.logZ = logZ
        self.samples = samples
        self.dead = dead
        self.log_weights = log_weights
        self.param_names = param_names

    def summary(self):
        """Print a parameter summary table."""
        header = f"\n{'Param':<14} {'Mean':>12} {'Std':>10} {'q16':>10} {'q84':>10}"  # noqa: E231,E501
        print(header)
        print("-" * 58)
        for name in self.param_names:
            s = self.samples[name]
            w = jnp.exp(
                self.log_weights - jax.scipy.special.logsumexp(self.log_weights)
            )
            mu = float(jnp.sum(w * s))
            sq = float(jnp.sum(w * (s - mu) ** 2)) ** 0.5
            q16 = float(jnp.percentile(s, 16))
            q84 = float(jnp.percentile(s, 84))
            row = f"{name:<14} {mu:>12.4f} {sq:>10.4f} {q16:>10.4f} {q84:>10.4f}"  # noqa: E231,E501
            print(row)

    def __repr__(self) -> str:
        n = len(self.log_weights) if self.log_weights is not None else 0
        return f"NSResult(logZ={self.logZ:.2f}, n_samples={n})"  # noqa: E231


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
        outdir: str = "results/",
        n_live: int = 125,
        n_delete: int = 20,
        num_mcmc_steps_multiplier: int = 5,
        termination_dlogz: float = -3.0,
        max_shrinkage: int = 25,
        max_steps: int = 10,
        verbose: bool = True,
    ):
        if not HAS_BLACKJAX:
            raise ImportError(
                "blackjax nested-sampling API unavailable "
                f"({_BLACKJAX_NS_IMPORT_ERROR}).\n"
                "Install the handley-lab fork: pip install "
                "git+https://github.com/handley-lab/blackjax@proposal\n"
                "(Or use run_nested_sampling, which runs on mainline blackjax.)"
            )

        self.likelihood = likelihood
        self.prior = prior
        self.outdir = outdir
        self.n_live = n_live
        self.n_delete = n_delete
        self.n_mcmc_steps = prior.n_params * num_mcmc_steps_multiplier
        self.term_dlogz = termination_dlogz
        self.verbose = verbose

        # Build JAX-traceable prior and likelihood functions
        self._log_prior_fn = prior.log_prob_fn()
        self._log_like_fn = likelihood._make_log_likelihood(prior)

        # BlackJAX NS algorithm
        self._algo = _build_nss(
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_like_fn,
            n_mcmc_steps=self.n_mcmc_steps,
            n_delete=self.n_delete,
            max_shrinkage=max_shrinkage,
            max_steps=max_steps,
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
        initial_particles = self.prior.sample_n(
            init_key, self.n_live
        )  # (n_live, n_params)
        state = self._algo.init(initial_particles)

        if self.verbose:
            print(
                f"Nested sampling: {self.n_live} live points, "
                f"{self.n_mcmc_steps} MCMC steps/iter, "
                f"device: {jax.devices()[0]}"
            )

        # JIT the kernel step for GPU performance.
        step = jax.jit(self._algo.step)

        dead = []
        if self.verbose and HAS_TQDM:
            pbar = _tqdm.tqdm(desc="Dead points", unit=" pts")
        else:
            pbar = None

        # Iterate until the remaining evidence contribution is negligible.
        # Written as `not (diff > threshold)` so that -inf and nan differences
        # (which arise when logZ_live or logZ is -inf) correctly trigger termination
        # rather than hanging the loop.
        while True:
            key, subkey = jax.random.split(key)
            state, dead_info = step(subkey, state)
            dead.append(dead_info)
            if pbar is not None:
                pbar.update(self.n_delete)
            logZ_live = float(_ns_integrator(state).logZ_live)
            logZ      = float(_ns_integrator(state).logZ)
            if not (logZ_live - logZ > self.term_dlogz):
                break

        if pbar is not None:
            pbar.close()

        if self.verbose:
            print(f"\nlogZ = {float(_ns_integrator(state).logZ):.2f}")

        # Combine the per-iteration dead points with the final live points.
        # finalise() expects AdaptiveNSState (state), not the inner state.particles.
        dead_all = _bj_finalise(state, dead)

        # log_weights returns shape (n_points, n_mc): Monte-Carlo draws over
        # the stochastic prior-volume shrinkage.  Marginalise for evidence and
        # average for a single weight per point.
        key, w_key = jax.random.split(key)
        logw_mc = _bj_log_weights(w_key, dead_all)  # (n_points, n_mc)
        logZs = jax.scipy.special.logsumexp(logw_mc, axis=0)  # (n_mc,)
        logZ = float(logZs.mean())
        logw = logw_mc.mean(axis=-1)  # (n_points,)

        if self.verbose:
            print(f"log Z = {logZ:.2f} ± {float(logZs.std()):.2f}")  # noqa: E231

        # Per-parameter posterior samples.
        # dead_all is NSInfo; positions live at _dead_positions(dead_all).
        positions = _dead_positions(dead_all)   # (n_points, n_params)
        samples = {
            name: positions[:, i]
            for i, name in enumerate(self.prior.names)
        }

        # Save chains in anesthetic dead-birth format.
        if self.outdir is not None:
            os.makedirs(self.outdir, exist_ok=True)
            chains_dir = os.path.join(self.outdir, "chains")
            os.makedirs(chains_dir, exist_ok=True)
            try:
                logL       = np.asarray(_dead_logL(dead_all))
                logL_birth = np.asarray(_dead_logL_birth(dead_all))
                table = np.column_stack([np.asarray(positions), logL, logL_birth])
                np.savetxt(os.path.join(chains_dir, "chains_dead-birth.txt"), table)
                with open(os.path.join(chains_dir, "chains.paramnames"), "w") as f:
                    for name in self.prior.names:
                        f.write(f"{name}\t{name}\n")
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

    def plot_corner(
        self, result: NSResult, truth: dict = None, filename: str = None, **kwargs
    ):
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
            import matplotlib.pyplot as plt
            from anesthetic import make_2d_axes, read_chains
        except ImportError:
            raise ImportError(
                "anesthetic and matplotlib are required for corner plots.\n"
                "pip install anesthetic matplotlib"
            )

        chains_root = None
        if self.outdir is not None:
            chains_root = os.path.join(self.outdir, "chains", "chains")

        if chains_root is not None and os.path.exists(chains_root + "_dead-birth.txt"):
            samples = read_chains(chains_root, columns=self.prior.names)
        else:
            # Fall back: build NestedSamples from raw arrays
            from anesthetic import NestedSamples

            data = {n: np.array(result.samples[n]) for n in self.prior.names}
            data["logL"] = np.array(_dead_logL(result.dead))
            data["logL_birth"] = np.array(_dead_logL_birth(result.dead))
            samples = NestedSamples(
                data=data,
                logL="logL",
                logL_birth="logL_birth",
                columns=self.prior.names,
            )

        fig, axes = make_2d_axes(
            self.prior.names, figsize=(3 * self.prior.n_params,) * 2, facecolor="w"
        )
        samples.plot_2d(axes, alpha=0.9, label="posterior", **kwargs)

        if truth is not None:
            for i, name in enumerate(self.prior.names):
                if name not in truth:
                    continue
                tv = truth[name]
                axes.iloc[i, i].axvline(tv, color="red", linestyle="--", linewidth=2)
                for j in range(i):
                    axes.iloc[i, j].axhline(
                        tv, color="red", linestyle="--", linewidth=1, alpha=0.5
                    )
                    if self.prior.names[j] in truth:
                        axes.iloc[i, j].axvline(
                            truth[self.prior.names[j]],
                            color="red",
                            linestyle="--",
                            linewidth=1,
                            alpha=0.5,
                        )

        plt.suptitle("Posterior", y=1.02)
        plt.tight_layout()

        if filename is None and self.outdir is not None:
            filename = os.path.join(self.outdir, "corner.png")

        if filename is not None:
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            if self.verbose:
                print(f"Corner plot saved to {filename}")

        return fig, axes


class BatchedNestedSampler:
    """Run independent BlackJAX nested-sampling chains in one vmapped batch.

    Parameters
    ----------
    model : str or callable
        Spectra model used by the shared batched likelihood.
    dataset : BatchedDataset
        Padded batch of transient photometry.
    prior_template : Prior
        Shared prior for every transient in the batch.
    fixed_params_batch : dict
        Per-transient fixed parameters with leading shape ``(B,)``.
    fixed_params : dict, optional
        Fixed parameters shared across the whole batch.
    evaluation_mode : {"full", "compact_source", "direct_photometry"}, optional
        Model-evaluation mode passed to ``make_batched_log_likelihood``.

    Notes
    -----
    All transients share the same prior bounds and parameter order. If a batch
    needs different prior ranges, split it into groups with compatible priors.
    """

    def __init__(
        self,
        *,
        model,
        dataset,
        prior_template=None,
        prior=None,
        fixed_params_batch=None,
        fixed_params=None,
        bridges=None,
        outdir: str = 'results/',
        n_live: int = 125,
        n_delete: int = 20,
        num_mcmc_steps_multiplier: int = 5,
        termination_dlogz: float = -3.0,
        max_shrinkage: int = 25,
        max_steps: int = 10,
        max_iterations: int | None = None,
        verbose: bool = True,
        t0_key: str | None = 't0',
        evaluation_mode: str = 'full',
        compact_time_grid_size: int = 256,
        compact_grid_pad_days: float = 5.0,
        param_transforms=None,
    ):
        if not HAS_BLACKJAX:
            raise ImportError(
                "blackjax is required for batched nested sampling.\n"
                "Install with: pip install git+https://github.com/handley-lab/blackjax@proposal"
            )
        if prior_template is None:
            prior_template = prior
        if prior_template is None:
            raise ValueError("BatchedNestedSampler requires prior_template")
        if fixed_params_batch is None:
            fixed_params_batch = {}

        self.model = model
        self.dataset = dataset
        self.prior = prior_template
        self.fixed_params_batch = {
            name: jnp.asarray(value)
            for name, value in fixed_params_batch.items()
        }
        self.fixed_params = dict(fixed_params or {})
        self.outdir = outdir
        self.n_live = int(n_live)
        self.n_delete = int(n_delete)
        self.n_mcmc_steps = self.prior.n_params * int(num_mcmc_steps_multiplier)
        self.term_dlogz = float(termination_dlogz)
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.evaluation_mode = evaluation_mode

        self._log_prior_fn = self.prior.log_prob_fn()
        self._log_like_batch_fn = make_batched_log_likelihood(
            model,
            self.fixed_params_batch,
            self.prior,
            bridges,
            dataset,
            fixed_params=self.fixed_params,
            t0_key=t0_key,
            evaluation_mode=evaluation_mode,
            compact_time_grid_size=compact_time_grid_size,
            compact_grid_pad_days=compact_grid_pad_days,
            param_transforms=param_transforms,
        )
        self._log_like_fn = self._log_like_batch_fn.indexed

        self._algo = _build_nss(
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_like_fn,
            n_mcmc_steps=self.n_mcmc_steps,
            n_delete=self.n_delete,
            max_shrinkage=max_shrinkage,
            max_steps=max_steps,
        )

    def run(self, key: jax.Array) -> list[NSResult]:
        """Run the batched sampler and return one ``NSResult`` per transient."""
        key, init_key = jax.random.split(key)
        init_keys = jax.random.split(init_key, self.dataset.n_batch)
        initial_particles = jax.vmap(
            lambda k: self.prior.sample_n(k, self.n_live)
        )(init_keys)

        states = jax.vmap(self._algo.init, axis_name='batch')(initial_particles)
        converged = jnp.zeros((self.dataset.n_batch,), dtype=bool)

        if self.verbose:
            print(
                f"Batched nested sampling: B={self.dataset.n_batch}, "
                f"{self.n_live} live points/SN, {self.n_mcmc_steps} MCMC steps/iter, "
                f"mode={self.evaluation_mode}, device: {jax.devices()[0]}"
            )

        vmapped_step = jax.jit(jax.vmap(self._algo.step, axis_name='batch'))
        dead_steps = []
        active_masks = []
        iteration = 0

        if self.verbose and HAS_TQDM:
            pbar = _tqdm.tqdm(desc="Batched dead points", unit=" iter")
        else:
            pbar = None

        while True:
            active_before = ~converged
            key, step_key = jax.random.split(key)
            step_keys = jax.random.split(step_key, self.dataset.n_batch)
            new_states, dead_info = vmapped_step(step_keys, states)
            states = _freeze_converged_states(converged, states, new_states)

            dead_steps.append(dead_info)
            active_masks.append(active_before)
            iteration += 1

            logZ_live = _ns_integrator(states).logZ_live
            logZ = _ns_integrator(states).logZ
            converged = converged | ~(logZ_live - logZ > self.term_dlogz)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(active=int(np.asarray((~converged).sum())))

            if bool(jnp.all(converged)):
                break
            if self.max_iterations is not None and iteration >= self.max_iterations:
                if self.verbose:
                    print(
                        f"Warning: stopped after max_iterations={self.max_iterations}; "
                        f"{int(np.asarray((~converged).sum()))} chains not converged"
                    )
                break

        if pbar is not None:
            pbar.close()

        results = []
        for batch_idx in range(self.dataset.n_batch):
            state_i = tree_util.tree_map(lambda x: x[batch_idx], states)
            dead_i = [
                tree_util.tree_map(lambda x, idx=batch_idx: x[idx], dead_step)
                for dead_step, active in zip(dead_steps, active_masks)
                if bool(np.asarray(active)[batch_idx])
            ]
            result_i = self._finalise_one(key, state_i, dead_i, batch_idx)
            results.append(result_i)

        return results

    def _finalise_one(self, key, state, dead, batch_idx):
        dead_all = _bj_finalise(state, dead)
        key_i = jax.random.fold_in(key, batch_idx)
        logw_mc = _bj_log_weights(key_i, dead_all)
        logZs = jax.scipy.special.logsumexp(logw_mc, axis=0)
        logZ = float(logZs.mean())
        logw = logw_mc.mean(axis=-1)
        positions = _dead_positions(dead_all)
        samples = {
            name: positions[:, i]
            for i, name in enumerate(self.prior.names)
        }

        if self.verbose:
            label = self.dataset.names[batch_idx]
            print(f"{label}: log Z = {logZ:.2f} ± {float(logZs.std()):.2f}")

        if self.outdir is not None:
            label = self.dataset.names[batch_idx]
            safe_label = str(label).replace(os.sep, "_")
            chains_dir = os.path.join(self.outdir, safe_label, 'chains')
            os.makedirs(chains_dir, exist_ok=True)
            try:
                logL = np.asarray(_dead_logL(dead_all))
                logL_birth = np.asarray(_dead_logL_birth(dead_all))
                table = np.column_stack([np.asarray(positions), logL, logL_birth])
                np.savetxt(os.path.join(chains_dir, 'chains_dead-birth.txt'), table)
                with open(os.path.join(chains_dir, 'chains.paramnames'), 'w') as f:
                    for name in self.prior.names:
                        f.write(f"{name}\t{name}\n")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: could not save chains for {label}: {e}")

        return NSResult(
            logZ=logZ,
            samples=samples,
            dead=dead_all,
            log_weights=logw,
            param_names=self.prior.names,
        )


class BatchedFluxDensityNestedSampler:
    """Batched nested sampler for :class:`~redback_jax.inference.batch.BatchedFluxDensityDataset`.

    All SNe share the same prior template.  The ``t0`` parameter (if present)
    is sampled as days *before* each SN's first observation — the absolute MJD
    is recovered inside the likelihood via ``t0_ref + t0_offset``.

    Parameters
    ----------
    model : callable
        ``csm_nickel_flux_density`` or any model with the same signature.
    dataset : BatchedFluxDensityDataset
    prior : Prior
        Shared prior for all SNe. ``t0`` bounds should be in days-before-first-obs
        (e.g. ``Uniform(-150, -2, name='t0')``).
    fixed_params_batch : dict
        Per-SN fixed params as ``(B,)`` arrays — ``redshift``, ``lum_dist``.
    fixed_params : dict, optional
        Shared fixed params — ``kappa``, ``temperature_floor``, etc.
    outdir : str
        Root directory; results saved to ``outdir/<sn_name>/``.
    """

    def __init__(
        self,
        *,
        model,
        dataset,
        prior,
        fixed_params_batch,
        fixed_params=None,
        outdir='results/',
        n_live=125,
        n_delete=20,
        num_mcmc_steps_multiplier=5,
        termination_dlogz=-3.0,
        max_shrinkage=25,
        max_steps=10,
        max_iterations=None,
        verbose=True,
        t0_key='t0',
        param_transforms=None,
    ):
        if not HAS_BLACKJAX:
            raise ImportError("blackjax is required for batched nested sampling.")

        from redback_jax.inference.likelihood import make_batched_flux_density_log_likelihood

        self.model   = model
        self.dataset = dataset
        self.prior   = prior
        self.outdir  = outdir
        self.n_live  = n_live
        self.n_delete = n_delete
        self.n_mcmc_steps = prior.n_params * num_mcmc_steps_multiplier
        self.term_dlogz   = float(termination_dlogz)
        self.max_iterations = max_iterations
        self.verbose = verbose

        self._log_prior_fn = prior.log_prob_fn()
        batch_ll = make_batched_flux_density_log_likelihood(
            model, dataset, prior,
            fixed_params_batch=fixed_params_batch,
            fixed_params=fixed_params,
            t0_key=t0_key,
            param_transforms=param_transforms,
        )
        self._log_like_fn = batch_ll.indexed

        self._algo = _build_nss(
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_like_fn,
            n_mcmc_steps=self.n_mcmc_steps,
            n_delete=self.n_delete,
            max_shrinkage=max_shrinkage,
            max_steps=max_steps,
        )

    def run(self, key):
        """Run all chains and return a list of NSResult (one per SN)."""
        B = self.dataset.n_batch
        key, init_key = jax.random.split(key)
        init_keys = jax.random.split(init_key, B)
        initial_particles = jax.vmap(
            lambda k: self.prior.sample_n(k, self.n_live)
        )(init_keys)

        states    = jax.vmap(self._algo.init, axis_name='batch')(initial_particles)
        converged = jnp.zeros((B,), dtype=bool)

        if self.verbose:
            print(
                f"Batched flux-density NS: B={B}, {self.n_live} live/SN, "
                f"{self.n_mcmc_steps} MCMC steps/iter, device: {jax.devices()[0]}"
            )

        vmapped_step = jax.jit(jax.vmap(self._algo.step, axis_name='batch'))
        dead_steps, active_masks = [], []
        iteration = 0

        pbar = (_tqdm.tqdm(desc='Batched dead points', unit=' iter')
                if self.verbose and HAS_TQDM else None)

        while True:
            active_before = ~converged
            key, step_key = jax.random.split(key)
            step_keys = jax.random.split(step_key, B)
            new_states, dead_info = vmapped_step(step_keys, states)
            states = _freeze_converged_states(converged, states, new_states)

            dead_steps.append(dead_info)
            active_masks.append(active_before)
            iteration += 1

            logZ_live = _ns_integrator(states).logZ_live
            logZ      = _ns_integrator(states).logZ
            converged = converged | ~(logZ_live - logZ > self.term_dlogz)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(active=int(np.asarray((~converged).sum())))

            if bool(jnp.all(converged)):
                break
            if self.max_iterations is not None and iteration >= self.max_iterations:
                if self.verbose:
                    print(f"Warning: stopped at max_iterations={self.max_iterations}, "
                          f"{int(np.asarray((~converged).sum()))} chains not converged")
                break

        if pbar is not None:
            pbar.close()

        results = []
        for i in range(B):
            state_i = tree_util.tree_map(lambda x, idx=i: x[idx], states)
            dead_i  = [
                tree_util.tree_map(lambda x, idx=i: x[idx], ds)
                for ds, am in zip(dead_steps, active_masks)
                if bool(np.asarray(am)[i])
            ]
            results.append(self._finalise(key, state_i, dead_i, i))

        return results

    def _finalise(self, key, state, dead, batch_idx):
        dead_all = _bj_finalise(state, dead)
        key_i    = jax.random.fold_in(key, batch_idx)
        logw_mc  = _bj_log_weights(key_i, dead_all)
        logZs    = jax.scipy.special.logsumexp(logw_mc, axis=0)
        logZ     = float(logZs.mean())
        logw     = logw_mc.mean(axis=-1)
        positions = _dead_positions(dead_all)
        samples   = {name: positions[:, i] for i, name in enumerate(self.prior.names)}
        sn_name   = self.dataset.names[batch_idx]

        if self.verbose:
            print(f"  {sn_name}: log Z = {logZ:.2f} ± {float(logZs.std()):.2f}")

        if self.outdir is not None:
            import pandas as pd
            outdir_sn = os.path.join(self.outdir, str(sn_name))
            os.makedirs(outdir_sn, exist_ok=True)
            post_df = pd.DataFrame({k: np.array(v) for k, v in samples.items()})
            post_df['logZ'] = logZ
            post_df.to_csv(os.path.join(outdir_sn, 'posterior.csv'), index=False)

            chains_dir = os.path.join(outdir_sn, 'chains')
            os.makedirs(chains_dir, exist_ok=True)
            try:
                logL       = np.asarray(_dead_logL(dead_all))
                logL_birth = np.asarray(_dead_logL_birth(dead_all))
                table = np.column_stack([np.asarray(positions), logL, logL_birth])
                np.savetxt(os.path.join(chains_dir, 'chains_dead-birth.txt'), table)
                with open(os.path.join(chains_dir, 'chains.paramnames'), 'w') as f:
                    for name in self.prior.names:
                        f.write(f"{name}\t{name}\n")
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: could not save chains for {sn_name}: {e}")

        return NSResult(
            logZ=logZ, samples=samples, dead=dead_all,
            log_weights=logw, param_names=self.prior.names,
        )


def _freeze_converged_states(converged, old_state, new_state):
    def _freeze_leaf(old, new):
        cond = converged
        while cond.ndim < new.ndim:
            cond = cond[..., None]
        return jnp.where(cond, old, new)

    return tree_util.tree_map(_freeze_leaf, old_state, new_state)
