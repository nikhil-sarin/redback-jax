"""
Likelihood for redback-jax inference.

Usage::

    from redback_jax.inference import Likelihood, Prior, Uniform, NestedSampler

    prior = Prior([
        Uniform(58580, 58620, name='t0'),
        Uniform(0.05,  0.20,  name='f_nickel'),
        Uniform(0.8,   2.0,   name='mej'),
        Uniform(3000,  8000,  name='vej'),
    ])

    likelihood = Likelihood(
        model        = 'arnett_spectra',
        transient    = transient,
        fixed_params = {
            'redshift':          0.01,
            'lum_dist':          dl_cm,
            'temperature_floor': 5000.0,
            'kappa':             0.07,
            'kappa_gamma':       0.1,
        },
    )

    result = NestedSampler(likelihood, prior, outdir='results/').run(key)

The ``transient`` object must have ``.time``, ``.y``, ``.y_err``, ``.bands``.

``fixed_params`` must supply everything the model needs that is *not* in the
prior.  Free parameters automatically take precedence over fixed ones.

If ``'t0'`` is a free parameter, it is treated as an MJD explosion time and
used to shift ``transient.time`` into source-frame days automatically.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Optional

try:
    from jax_supernovae.bandpasses import register_all_bandpasses
    from jax_supernovae.timeseries import timeseries_multiband_flux
    _HAS_JAX_SUPERNOVAE = True
except ImportError:
    _HAS_JAX_SUPERNOVAE = False

from redback_jax.models import get_model


class Likelihood:
    """Gaussian photometric likelihood using a spectra model pipeline.

    Parameters
    ----------
    model : str or callable
        Model name (e.g. ``'arnett_spectra'``) or a callable with signature
        ``f(redshift, lum_dist, vej, temperature_floor, **kwargs)``
        returning a namedtuple ``(time, lambdas, spectra)``.
    transient : Transient
        Data container with ``.time``, ``.y``, ``.y_err``, ``.bands``.
    fixed_params : dict
        Parameters held fixed during inference — everything the model needs
        that is not a free parameter in the prior.
    t0_key : str or None, optional
        Name of the MJD explosion-time free parameter (default ``'t0'``).
        When present in the prior the likelihood converts ``transient.time``
        from observer-frame MJD to source-frame days automatically.
        Set to ``None`` if times are already in source-frame days.
    evaluation_mode : {"full", "compact_source", "direct_photometry"}, optional
        ``"full"`` preserves the existing model-default source grid.
        ``"compact_source"`` uses a dataset-specific source phase grid while
        still going through ``jax_supernovae.timeseries_multiband_flux``.
        ``"direct_photometry"`` bypasses full source-cube materialization for
        factory-built blackbody spectra models and integrates directly through
        the bandpasses.
    """

    def __init__(
        self,
        model,
        transient,
        fixed_params: Dict,
        t0_key: Optional[str] = 't0',
        evaluation_mode: str = 'full',
        compact_time_grid_size: int = 256,
        compact_grid_pad_days: float = 5.0,
    ):
        if not _HAS_JAX_SUPERNOVAE:
            raise ImportError(
                "Likelihood requires jax_supernovae.\n"
                "Use FluxDensityLikelihood for flux-density models instead."
            )
        register_all_bandpasses()

        if isinstance(model, str):
            self._model_fn = get_model(model)
            self.model_name = model
        else:
            self._model_fn = model
            self.model_name = getattr(model, '__name__', repr(model))

        self.transient    = transient
        self.fixed_params = dict(fixed_params)
        self.t0_key       = t0_key
        self.evaluation_mode = evaluation_mode
        self.compact_time_grid_size = int(compact_time_grid_size)
        self.compact_grid_pad_days = float(compact_grid_pad_days)

        self._obs_times    = jnp.asarray(transient.time)
        self._obs_mags     = jnp.asarray(transient.y)
        self._obs_errs     = jnp.asarray(transient.y_err)

        bands_raw          = list(transient.bands)
        self._unique_bands = list(dict.fromkeys(bands_raw))
        self._obs_band_idx = None   # built lazily in _build_bridges
        self._bridges      = None
        self._band_to_idx  = None
        self._redshift_const = float(self.fixed_params.get('redshift', 0.0))
        self._compact_time_observer_grid = None
        self._minphase = None

        self._bands_raw = bands_raw

        valid_modes = {'full', 'compact_source', 'direct_photometry'}
        if self.evaluation_mode not in valid_modes:
            raise ValueError(
                f"evaluation_mode must be one of {sorted(valid_modes)}, got {evaluation_mode!r}"
            )

    def _build_bridges(self, prior):
        """Precompute bandpass bridges using prior midpoints for free params."""
        from jax_supernovae.bandpasses import get_bandpass
        from jax_supernovae.salt3 import precompute_bandflux_bridge

        # Fill in free param midpoints so the dummy model call has all args
        dummy_kwargs = dict(self.fixed_params)
        for d in prior.distributions:
            if d.name != self.t0_key:
                dummy_kwargs.setdefault(d.name, 0.5 * (d.low + d.high))

        self._bridges   = tuple(
            precompute_bandflux_bridge(get_bandpass(b)) for b in self._unique_bands
        )
        band_to_idx = {b: i for i, b in enumerate(self._unique_bands)}
        self._obs_band_idx = jnp.array(
            [band_to_idx[b] for b in self._bands_raw]
        )
        self._band_to_idx = band_to_idx

        if self.evaluation_mode == 'full':
            self._dummy_out = self._model_fn(**dummy_kwargs)
            self._minphase = float(self._dummy_out.time[0])
        elif self.evaluation_mode == 'compact_source':
            if not getattr(self._model_fn, '_redback_jax_supports_custom_grids', False):
                raise ValueError(
                    f"Model {self.model_name!r} does not support compact_source evaluation"
                )
            self._compact_time_observer_grid = self._build_compact_time_observer_grid(prior)
            self._dummy_out = self._model_fn(
                _time_observer_frame_grid=self._compact_time_observer_grid,
                **dummy_kwargs,
            )
            self._minphase = float(self._compact_time_observer_grid[0])
        else:
            direct_fn = getattr(self._model_fn, '_redback_jax_direct_photometry', None)
            if direct_fn is None:
                raise ValueError(
                    f"Model {self.model_name!r} does not support direct_photometry evaluation"
                )

    def _build_compact_time_observer_grid(self, prior):
        """Build an observer-frame phase grid covering the whole dataset support."""
        redshift = self._redshift_const
        obs_min = float(jnp.min(self._obs_times))
        obs_max = float(jnp.max(self._obs_times))

        t0_dist = None
        if self.t0_key is not None:
            for dist in prior.distributions:
                if dist.name == self.t0_key:
                    t0_dist = dist
                    break

        if t0_dist is None:
            source_min = obs_min
            source_max = obs_max
        else:
            source_min = (obs_min - t0_dist.high) / (1.0 + redshift)
            source_max = (obs_max - t0_dist.low) / (1.0 + redshift)

        source_lo = max(0.1, source_min - self.compact_grid_pad_days)
        source_hi = max(source_lo * 1.001, source_max + self.compact_grid_pad_days)
        return jnp.geomspace(source_lo, source_hi, self.compact_time_grid_size) * (1.0 + redshift)

    def _make_log_likelihood(self, prior):
        """Return a JIT-compiled log-likelihood function ``(params,) -> scalar``."""
        self._build_bridges(prior)

        model_fn      = self._model_fn
        fixed_params  = self.fixed_params
        obs_times     = self._obs_times
        obs_mags      = self._obs_mags
        obs_errs      = self._obs_errs
        obs_band_idx  = self._obs_band_idx
        bridges       = self._bridges
        redshift      = self._redshift_const
        t0_key        = self.t0_key
        names         = prior.names
        evaluation_mode = self.evaluation_mode
        compact_time_grid = self._compact_time_observer_grid
        direct_photometry_fn = getattr(self._model_fn, '_redback_jax_direct_photometry', None)

        # Static scalars from the dummy run
        _zero_before = True
        _minphase    = self._minphase
        # zp=0 per observation: timeseries_multiband_flux normalises each flux
        # by the per-band AB zpbandflux so that -2.5*log10(result) = AB mag.
        _zps = jnp.zeros(len(self._obs_mags))

        @jax.jit
        def _log_like(params: jnp.ndarray) -> jnp.ndarray:
            param_dict = {n: params[i] for i, n in enumerate(names)}

            if t0_key is not None and t0_key in param_dict:
                t0             = param_dict.pop(t0_key)
                t_source       = (obs_times - t0) / (1.0 + redshift)  # source-frame
                t_obs_since_t0 = obs_times - t0                        # observer-frame
            else:
                t_source       = obs_times
                t_obs_since_t0 = obs_times

            model_kwargs = {**fixed_params, **param_dict}
            if evaluation_mode == 'direct_photometry':
                norm_fluxes = direct_photometry_fn(
                    obs_source_time=t_source,
                    obs_band_idx=obs_band_idx,
                    bridges=bridges,
                    **model_kwargs,
                )
            else:
                if evaluation_mode == 'compact_source':
                    out = model_fn(
                        _time_observer_frame_grid=compact_time_grid,
                        **model_kwargs,
                    )
                else:
                    out = model_fn(**model_kwargs)

                # out.time is observer-frame days since explosion; query with the
                # same convention so timeseries_multiband_flux interpolates correctly.
                norm_fluxes = timeseries_multiband_flux(
                    t_obs_since_t0, bridges, obs_band_idx,
                    out.time, out.lambdas, out.spectra,
                    1.0, _zero_before, _minphase,
                    time_degree=1, zps=_zps, zpsys='ab',
                )
            model_mags = -2.5 * jnp.log10(norm_fluxes + 1e-100)
            return -0.5 * jnp.sum(((obs_mags - model_mags) / obs_errs) ** 2)

        return _log_like

    def __repr__(self) -> str:
        return (
            f"Likelihood(model={self.model_name!r}, "
            f"n_obs={len(self._obs_mags)}, "
            f"bands={self._unique_bands}, "
            f"evaluation_mode={self.evaluation_mode!r})"
        )


class FluxDensityLikelihood:
    """Gaussian likelihood for models returning observed-frame flux density (mJy).

    Designed for use with ``general_magnetar_driven_supernova_diffrax`` and any
    other model with signature ``f(time, frequency, **params) -> jnp.ndarray``.
    No ``jax_supernovae`` dependency — operates directly on flux residuals.

    This class follows the same interface as :class:`Likelihood` and is fully
    compatible with :class:`~redback_jax.inference.NestedSampler`.

    Parameters
    ----------
    model : callable
        ``f(time, frequency, **params) -> jnp.ndarray`` (mJy), where ``time``
        is observer-frame days and ``frequency`` is observer-frame Hz.
    time : array-like
        Observer-frame times (days), shape ``(N,)``.
    frequency : array-like
        Observer-frame frequencies (Hz), shape ``(N,)``.
    flux_obs : array-like
        Observed flux density (mJy), shape ``(N,)``.
    flux_err : array-like
        Flux density uncertainties (mJy), shape ``(N,)``.
    fixed_params : dict
        Parameters held fixed during inference (e.g. ``luminosity_distance``,
        ``redshift``, ``kappa``).
    """

    def __init__(self, model, time, frequency, flux_obs, flux_err, fixed_params):
        self._model       = model
        self._t           = jnp.array(time,      dtype=jnp.float64)
        self._nu          = jnp.array(frequency,  dtype=jnp.float64)
        self._F_obs       = jnp.array(flux_obs,   dtype=jnp.float64)
        self._F_err       = jnp.array(flux_err,   dtype=jnp.float64)
        self.fixed_params = dict(fixed_params)

    def _make_log_likelihood(self, prior):
        """Return a JIT-compiled log-likelihood ``(params_array,) -> scalar``.

        Parameters
        ----------
        prior : Prior
            The composite prior; used to extract ordered parameter names.

        Returns
        -------
        callable
            JIT-compiled function with signature ``(jnp.ndarray,) -> scalar``.
        """
        model  = self._model
        t      = self._t
        nu     = self._nu
        F_obs  = self._F_obs
        F_err  = self._F_err
        fixed  = self.fixed_params
        names  = prior.names

        @jax.jit
        def _log_like(params: jnp.ndarray) -> jnp.ndarray:
            param_dict = {n: params[i] for i, n in enumerate(names)}
            F_pred = model(t, nu, **fixed, **param_dict)
            # Replace non-finite flux with 0 before computing chi2 so that
            # gradients stay finite through the jnp.where (both branches are
            # always evaluated in JAX; NaN in the unused branch still propagates
            # gradients).
            is_finite = jnp.all(jnp.isfinite(F_pred))
            F_pred_safe = jnp.where(is_finite, F_pred, jnp.zeros_like(F_pred))
            chi2 = jnp.sum(((F_pred_safe - F_obs) / F_err) ** 2)
            return jnp.where(is_finite, -0.5 * chi2, -1e30)

        # Trigger JIT compilation once with a valid sample from the prior
        dummy = prior.sample_n(jax.random.PRNGKey(0), 1)[0]
        _log_like(dummy).block_until_ready()
        return _log_like

    def __repr__(self) -> str:
        return (
            f"FluxDensityLikelihood(model={self._model.__name__!r}, "
            f"n_obs={len(self._F_obs)}, "
            f"fixed={list(self.fixed_params.keys())})"
        )
