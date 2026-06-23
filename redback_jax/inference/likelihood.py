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
import numpy as np
from typing import Dict, Optional

try:
    from jax_supernovae.bandpasses import register_all_bandpasses
    from jax_supernovae.timeseries import timeseries_multiband_flux
    _HAS_JAX_SUPERNOVAE = True
except ImportError:
    _HAS_JAX_SUPERNOVAE = False

from redback_jax.models import get_model
from redback_jax.inference.batch import BatchedDataset


def _build_bandflux_bridges(bands):
    """Build jax_supernovae bandflux bridges for a fixed band order."""
    from jax_supernovae.bandpasses import get_bandpass
    from jax_supernovae.salt3 import precompute_bandflux_bridge

    return tuple(precompute_bandflux_bridge(get_bandpass(b)) for b in bands)


def _merge_fixed_params(fixed_params, fixed_params_dynamic):
    merged = dict(fixed_params)
    if fixed_params_dynamic is not None:
        merged.update(fixed_params_dynamic)
    return merged


def _params_to_model_dict(params, names, transforms):
    param_dict = {}
    for i, name in enumerate(names):
        if name in transforms:
            model_name, fn = transforms[name]
            param_dict[model_name] = fn(params[i])
        else:
            param_dict[name] = params[i]
    return param_dict


def _make_photometric_log_likelihood_kernel(
    *,
    model_fn,
    names,
    fixed_params,
    t0_key,
    evaluation_mode,
    param_transforms,
    bridges,
    minphase,
    compact_time_grid,
):
    """Return ``(params, data..., fixed_dynamic) -> scalar`` for one transient."""
    transforms = {k: (mn, fn) for k, (mn, fn) in param_transforms.items()}
    direct_photometry_fn = getattr(model_fn, '_redback_jax_direct_photometry', None)
    zero_before = True

    def _log_like_one(
        params,
        obs_times,
        obs_mags,
        obs_errs,
        obs_band_idx,
        mask,
        fixed_params_dynamic=None,
    ):
        fixed_i = _merge_fixed_params(fixed_params, fixed_params_dynamic)
        redshift = fixed_i.get('redshift', 0.0)

        safe_times = jnp.where(mask, obs_times, obs_times[0])
        safe_mags = jnp.where(mask, obs_mags, 0.0)
        safe_errs = jnp.where(mask, obs_errs, 1.0)
        safe_band_idx = jnp.where(mask, obs_band_idx, 0)

        param_dict = _params_to_model_dict(params, names, transforms)
        if t0_key is not None and t0_key in param_dict:
            t0 = param_dict.pop(t0_key)
            t_obs_since_t0 = safe_times - t0
            t_source = t_obs_since_t0 / (1.0 + redshift)
        else:
            t_obs_since_t0 = safe_times
            t_source = safe_times

        model_kwargs = {**fixed_i, **param_dict}
        if evaluation_mode == 'direct_photometry':
            norm_fluxes = direct_photometry_fn(
                obs_source_time=t_source,
                obs_band_idx=safe_band_idx,
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

            zps = jnp.zeros_like(safe_mags)
            norm_fluxes = timeseries_multiband_flux(
                t_obs_since_t0, bridges, safe_band_idx,
                out.time, out.lambdas, out.spectra,
                1.0, zero_before, minphase,
                time_degree=1, zps=zps, zpsys='ab',
            )

        model_mags = -2.5 * jnp.log10(norm_fluxes + 1e-100)
        residual = (safe_mags - model_mags) / safe_errs
        residual = jnp.where(mask, residual, 0.0)
        chi2 = jnp.sum(residual ** 2)
        finite = jnp.all(jnp.isfinite(jnp.where(mask, model_mags, 0.0)))
        return jnp.where(finite, -0.5 * chi2, -1e30)

    return _log_like_one


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
        param_transforms: Optional[Dict] = None,
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
        # param_transforms: {sampled_name: (model_name, callable)}
        # e.g. {'log_rho': ('rho', jnp.exp)} samples log_rho but passes rho=exp(log_rho) to the model
        self.param_transforms = dict(param_transforms) if param_transforms else {}
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
        # Fill in free param midpoints so the dummy model call has all args
        dummy_kwargs = dict(self.fixed_params)
        for d in prior.distributions:
            if d.name != self.t0_key:
                val = 0.5 * (d.low + d.high)
                if d.name in self.param_transforms:
                    model_name, fn = self.param_transforms[d.name]
                    dummy_kwargs.setdefault(model_name, float(fn(jnp.array(val))))
                else:
                    dummy_kwargs.setdefault(d.name, val)

        self._bridges = _build_bandflux_bridges(self._unique_bands)
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

        obs_times = self._obs_times
        obs_mags = self._obs_mags
        obs_errs = self._obs_errs
        obs_band_idx = self._obs_band_idx
        mask = jnp.ones_like(obs_mags, dtype=bool)
        kernel = _make_photometric_log_likelihood_kernel(
            model_fn=self._model_fn,
            names=prior.names,
            fixed_params=self.fixed_params,
            t0_key=self.t0_key,
            evaluation_mode=self.evaluation_mode,
            param_transforms=self.param_transforms,
            bridges=self._bridges,
            minphase=self._minphase,
            compact_time_grid=self._compact_time_observer_grid,
        )

        @jax.jit
        def _log_like(params: jnp.ndarray) -> jnp.ndarray:
            return kernel(
                params, obs_times, obs_mags, obs_errs, obs_band_idx, mask, None
            )

        return _log_like

    def __repr__(self) -> str:
        return (
            f"Likelihood(model={self.model_name!r}, "
            f"n_obs={len(self._obs_mags)}, "
            f"bands={self._unique_bands}, "
            f"evaluation_mode={self.evaluation_mode!r})"
        )


def make_batched_log_likelihood(
    model,
    fixed_params_batch: Dict[str, jnp.ndarray],
    prior,
    bridges,
    dataset: BatchedDataset,
    *,
    fixed_params: Optional[Dict] = None,
    t0_key: Optional[str] = 't0',
    evaluation_mode: str = 'full',
    compact_time_grid_size: int = 256,
    compact_grid_pad_days: float = 5.0,
    param_transforms: Optional[Dict] = None,
):
    """Build a vmapped photometric log-likelihood for a ``BatchedDataset``.

    Parameters
    ----------
    model : str or callable
        Spectra model name or callable, as accepted by :class:`Likelihood`.
    fixed_params_batch : dict
        Per-transient fixed parameters. Each value must have shape ``(B,)``.
        Typical entries are ``redshift`` and ``lum_dist``.
    prior : Prior
        Shared prior template for all transients in the batch.
    bridges : tuple or None
        Precomputed bandflux bridges. Pass ``None`` to build them from
        ``dataset.bands``.
    dataset : BatchedDataset
        Padded photometric observations.
    fixed_params : dict, optional
        Fixed parameters shared by every transient.
    evaluation_mode : {"full", "compact_source", "direct_photometry"}, optional
        Uses the same model-evaluation modes as :class:`Likelihood`. The default
        is ``"full"`` because dense light curves can be faster through the
        spectra/interpolation path than through per-observation direct
        photometry.

    Returns
    -------
    callable
        JIT-compiled function ``params_batch -> logL_batch`` with shapes
        ``(B, n_params) -> (B,)``.
    """
    if not _HAS_JAX_SUPERNOVAE:
        raise ImportError(
            "make_batched_log_likelihood requires jax_supernovae.\n"
            "Use FluxDensityLikelihood for flux-density models instead."
        )
    register_all_bandpasses()

    if isinstance(model, str):
        model_fn = get_model(model)
        model_name = model
    else:
        model_fn = model
        model_name = getattr(model, '__name__', repr(model))

    fixed_params = dict(fixed_params or {})
    fixed_params_batch = {
        name: jnp.asarray(value)
        for name, value in fixed_params_batch.items()
    }
    for name, value in fixed_params_batch.items():
        if value.shape[0] != dataset.n_batch:
            raise ValueError(
                f"fixed_params_batch[{name!r}] has leading size {value.shape[0]}, "
                f"expected {dataset.n_batch}"
            )

    valid_modes = {'full', 'compact_source', 'direct_photometry'}
    if evaluation_mode not in valid_modes:
        raise ValueError(
            f"evaluation_mode must be one of {sorted(valid_modes)}, got {evaluation_mode!r}"
        )
    if evaluation_mode == 'compact_source' and not getattr(
        model_fn, '_redback_jax_supports_custom_grids', False
    ):
        raise ValueError(
            f"Model {model_name!r} does not support compact_source evaluation"
        )
    if evaluation_mode == 'direct_photometry' and getattr(
        model_fn, '_redback_jax_direct_photometry', None
    ) is None:
        raise ValueError(
            f"Model {model_name!r} does not support direct_photometry evaluation"
        )

    bridges = _build_bandflux_bridges(dataset.bands) if bridges is None else bridges
    dummy_kwargs = _dummy_model_kwargs(
        prior,
        fixed_params,
        fixed_params_batch,
        t0_key,
        param_transforms or {},
    )
    compact_time_grid = None
    if evaluation_mode == 'compact_source':
        compact_time_grid = _build_batched_compact_time_observer_grid(
            dataset,
            prior,
            fixed_params,
            fixed_params_batch,
            t0_key,
            compact_time_grid_size,
            compact_grid_pad_days,
        )
        model_fn(
            _time_observer_frame_grid=compact_time_grid,
            **dummy_kwargs,
        )
        minphase = float(compact_time_grid[0])
    elif evaluation_mode == 'direct_photometry':
        minphase = None
    else:
        minphase = _batched_minphase(model_fn, dummy_kwargs, fixed_params_batch)

    kernel = _make_photometric_log_likelihood_kernel(
        model_fn=model_fn,
        names=prior.names,
        fixed_params=fixed_params,
        t0_key=t0_key,
        evaluation_mode=evaluation_mode,
        param_transforms=param_transforms or {},
        bridges=bridges,
        minphase=minphase,
        compact_time_grid=compact_time_grid,
    )

    fixed_dynamic = fixed_params_batch
    obs_times, obs_mags, obs_errs, obs_band_idx, mask = dataset.data_tuple()

    def _single(params, times, mags, errs, band_idx, mask_i, fixed_i):
        return kernel(params, times, mags, errs, band_idx, mask_i, fixed_i)

    @jax.jit
    def _batched_log_like(params_batch):
        return jax.vmap(_single)(
            params_batch,
            obs_times,
            obs_mags,
            obs_errs,
            obs_band_idx,
            mask,
            fixed_dynamic,
        )

    def _indexed_log_like(params):
        idx = jax.lax.axis_index('batch')
        fixed_i = {
            name: value[idx]
            for name, value in fixed_dynamic.items()
        }
        return kernel(
            params,
            obs_times[idx],
            obs_mags[idx],
            obs_errs[idx],
            obs_band_idx[idx],
            mask[idx],
            fixed_i,
        )

    _batched_log_like.indexed = _indexed_log_like
    return _batched_log_like


def _dummy_model_kwargs(prior, fixed_params, fixed_params_batch, t0_key, transforms):
    dummy_kwargs = dict(fixed_params)
    for name, values in fixed_params_batch.items():
        dummy_kwargs[name] = float(jnp.asarray(values)[0])
    for dist in prior.distributions:
        if dist.name == t0_key:
            continue
        val = 0.5 * (dist.low + dist.high)
        if dist.name in transforms:
            model_name, fn = transforms[dist.name]
            dummy_kwargs.setdefault(model_name, float(fn(jnp.asarray(val))))
        else:
            dummy_kwargs.setdefault(dist.name, val)
    return dummy_kwargs


def _batched_minphase(model_fn, dummy_kwargs, fixed_params_batch):
    if 'redshift' not in fixed_params_batch:
        return float(model_fn(**dummy_kwargs).time[0])
    minphase = None
    for redshift in np.asarray(fixed_params_batch['redshift']):
        kwargs_i = dict(dummy_kwargs)
        kwargs_i['redshift'] = float(redshift)
        out = model_fn(**kwargs_i)
        phase_i = float(out.time[0])
        minphase = phase_i if minphase is None else min(minphase, phase_i)
    return minphase


def _build_batched_compact_time_observer_grid(
    dataset,
    prior,
    fixed_params,
    fixed_params_batch,
    t0_key,
    grid_size,
    pad_days,
):
    real_times = jnp.where(dataset.mask, dataset.obs_times, jnp.nan)
    obs_min = float(jnp.nanmin(real_times))
    obs_max = float(jnp.nanmax(real_times))

    redshifts = fixed_params_batch.get('redshift')
    if redshifts is None:
        redshift_min = redshift_max = float(fixed_params.get('redshift', 0.0))
    else:
        redshifts_np = np.asarray(redshifts, dtype=float)
        redshift_min = float(np.min(redshifts_np))
        redshift_max = float(np.max(redshifts_np))

    t0_dist = None
    if t0_key is not None:
        for dist in prior.distributions:
            if dist.name == t0_key:
                t0_dist = dist
                break

    if t0_dist is None:
        source_min = obs_min
        source_max = obs_max
    else:
        source_min = (obs_min - t0_dist.high) / (1.0 + redshift_max)
        source_max = (obs_max - t0_dist.low) / (1.0 + redshift_min)

    source_lo = max(0.1, source_min - float(pad_days))
    source_hi = max(source_lo * 1.001, source_max + float(pad_days))
    return jnp.geomspace(source_lo, source_hi, int(grid_size)) * (1.0 + redshift_max)


class FluxDensityLikelihood:
    """Gaussian likelihood for models returning observed-frame flux density (mJy).

    Designed for use with ``csm_nickel_flux_density``,
    ``general_magnetar_driven_supernova_diffrax``, and any other model with
    signature ``f(time, frequency, **params) -> jnp.ndarray``.
    No ``jax_supernovae`` dependency — operates directly on flux residuals.

    This class follows the same interface as :class:`Likelihood` and is fully
    compatible with :class:`~redback_jax.inference.NestedSampler`.

    Parameters
    ----------
    model : callable
        ``f(time, frequency, **params) -> jnp.ndarray`` (mJy), where ``time``
        is observer-frame days since explosion and ``frequency`` is Hz.
    time : array-like
        Observer-frame times in MJD (if ``t0_key`` is set) or days since
        explosion (if ``t0_key`` is ``None``), shape ``(N,)``.
    frequency : array-like
        Observer-frame frequencies (Hz), shape ``(N,)``.
    flux_obs : array-like
        Observed flux density (mJy), shape ``(N,)``.
    flux_err : array-like
        Flux density uncertainties (mJy), shape ``(N,)``.
    fixed_params : dict
        Parameters held fixed during inference.
    t0_key : str or None, optional
        Name of the explosion-time free parameter (MJD).  When present in the
        prior, the likelihood computes ``time - t0`` and passes the result as
        the ``time`` argument to the model.  Set to ``None`` if times are
        already days since explosion.
    param_transforms : dict, optional
        Same as in :class:`Likelihood` — map sampled names to model names via
        ``{sampled_name: (model_name, transform_fn)}``.
    """

    def __init__(self, model, time, frequency, flux_obs, flux_err, fixed_params,
                 t0_key: Optional[str] = None,
                 param_transforms: Optional[Dict] = None):
        self._model       = model
        self._t           = jnp.array(time,      dtype=jnp.float64)
        self._nu          = jnp.array(frequency,  dtype=jnp.float64)
        self._F_obs       = jnp.array(flux_obs,   dtype=jnp.float64)
        self._F_err       = jnp.array(flux_err,   dtype=jnp.float64)
        self.fixed_params = dict(fixed_params)
        self.t0_key       = t0_key
        self.param_transforms = dict(param_transforms) if param_transforms else {}

    def _make_log_likelihood(self, prior):
        """Return a JIT-compiled log-likelihood ``(params_array,) -> scalar``."""
        model      = self._model
        t          = self._t
        nu         = self._nu
        F_obs      = self._F_obs
        F_err      = self._F_err
        fixed      = self.fixed_params
        names      = prior.names
        t0_key     = self.t0_key
        _transforms = dict(self.param_transforms)

        @jax.jit
        def _log_like(params: jnp.ndarray) -> jnp.ndarray:
            param_dict = {}
            for i, n in enumerate(names):
                if n in _transforms:
                    model_name, fn = _transforms[n]
                    param_dict[model_name] = fn(params[i])
                else:
                    param_dict[n] = params[i]

            if t0_key is not None and t0_key in param_dict:
                t0 = param_dict.pop(t0_key)
                t_model = t - t0   # observer-frame days since explosion
            else:
                t_model = t

            F_pred = model(t_model, nu, **fixed, **param_dict)
            is_finite = jnp.all(jnp.isfinite(F_pred))
            F_pred_safe = jnp.where(is_finite, jnp.nan_to_num(F_pred), jnp.zeros_like(F_pred))
            chi2 = jnp.sum(((F_pred_safe - F_obs) / F_err) ** 2)
            return jnp.where(is_finite, -0.5 * chi2, -1e30)

        dummy = prior.sample_n(jax.random.PRNGKey(0), 1)[0]
        _log_like(dummy).block_until_ready()
        return _log_like

    def __repr__(self) -> str:
        return (
            f"FluxDensityLikelihood(model={self._model.__name__!r}, "
            f"n_obs={len(self._F_obs)}, "
            f"fixed={list(self.fixed_params.keys())})"
        )
