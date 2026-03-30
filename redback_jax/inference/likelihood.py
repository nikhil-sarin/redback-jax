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

from jax_supernovae.bandpasses import register_all_bandpasses
from jax_supernovae.timeseries import timeseries_multiband_flux

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
    """

    def __init__(
        self,
        model,
        transient,
        fixed_params: Dict,
        t0_key: Optional[str] = 't0',
    ):
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

        self._obs_times    = jnp.asarray(transient.time)
        self._obs_mags     = jnp.asarray(transient.y)
        self._obs_errs     = jnp.asarray(transient.y_err)

        bands_raw          = list(transient.bands)
        self._unique_bands = list(dict.fromkeys(bands_raw))
        self._obs_band_idx = None   # built lazily in _build_bridges
        self._bridges      = None
        self._band_to_idx  = None
        self._redshift_const = float(self.fixed_params.get('redshift', 0.0))

        self._bands_raw = bands_raw

    def _build_bridges(self, prior):
        """Precompute bandpass bridges using prior midpoints for free params."""
        from jax_supernovae.bandpasses import get_bandpass
        from jax_supernovae.salt3 import precompute_bandflux_bridge

        # Fill in free param midpoints so the dummy model call has all args
        dummy_kwargs = dict(self.fixed_params)
        for d in prior.distributions:
            if d.name != self.t0_key:
                dummy_kwargs.setdefault(d.name, 0.5 * (d.low + d.high))

        self._dummy_out = self._model_fn(**dummy_kwargs)
        self._bridges   = tuple(
            precompute_bandflux_bridge(get_bandpass(b)) for b in self._unique_bands
        )
        band_to_idx = {b: i for i, b in enumerate(self._unique_bands)}
        self._obs_band_idx = jnp.array(
            [band_to_idx[b] for b in self._bands_raw]
        )

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
        unique_bands  = self._unique_bands
        redshift      = self._redshift_const
        t0_key        = self.t0_key
        names         = prior.names

        # zero_before flag and minphase from the dummy run (static scalars)
        _zero_before = True
        _minphase    = float(self._dummy_out.time[0])

        @jax.jit
        def _log_like(params: jnp.ndarray) -> jnp.ndarray:
            param_dict = {n: params[i] for i, n in enumerate(names)}

            if t0_key is not None and t0_key in param_dict:
                t0       = param_dict.pop(t0_key)
                t_source = (obs_times - t0) / (1.0 + redshift)
            else:
                t_source = obs_times

            model_kwargs = {**fixed_params, **param_dict}
            out = model_fn(**model_kwargs)

            # Use timeseries_multiband_flux directly — avoids constructing a
            # PrecomputedSpectraSource inside JIT (which calls np.asarray on
            # traced arrays and is not JIT-safe).
            model_fluxes = timeseries_multiband_flux(
                t_source, bridges, obs_band_idx,
                out.time, out.lambdas, out.spectra,
                1.0, _zero_before, _minphase,
                time_degree=1,
            )
            # Same conversion as PrecomputedSpectraSource._compute_bandmag_single
            model_mags = -2.5 * jnp.log10(model_fluxes + 1e-100) - 48.60
            return -0.5 * jnp.sum(((obs_mags - model_mags) / obs_errs) ** 2)

        return _log_like

    def __repr__(self) -> str:
        return (
            f"Likelihood(model={self.model_name!r}, "
            f"n_obs={len(self._obs_mags)}, "
            f"bands={self._unique_bands})"
        )
