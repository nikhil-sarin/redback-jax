"""Batch helpers for multi-transient inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class BatchedDataset:
    """Padded photometric data for vmapped transient inference.

    Parameters
    ----------
    transients : sequence
        Objects with ``time``, ``y``, ``y_err``, and ``bands`` attributes.
    bands : sequence of str, optional
        Explicit global band order. If omitted, the union of bands is used in
        first-seen order across the batch.

    Notes
    -----
    Observations are padded to ``N_obs_max``. Padded rows have infinite
    uncertainty and ``mask=False`` so they contribute zero chi-squared in the
    batched likelihood.
    """

    transients: Sequence
    bands: Sequence[str] | None = None

    def __post_init__(self):
        transients = list(self.transients)
        if not transients:
            raise ValueError("BatchedDataset requires at least one transient")

        band_order = list(self.bands) if self.bands is not None else _union_bands(transients)
        if not band_order:
            raise ValueError("BatchedDataset requires at least one photometric band")

        band_to_idx = {band: i for i, band in enumerate(band_order)}
        lengths = [_validate_transient(t, band_to_idx) for t in transients]
        n_batch = len(transients)
        n_obs_max = max(lengths)

        obs_times = np.zeros((n_batch, n_obs_max), dtype=float)
        obs_mags = np.zeros((n_batch, n_obs_max), dtype=float)
        obs_errs = np.full((n_batch, n_obs_max), np.inf, dtype=float)
        obs_band_idx = np.zeros((n_batch, n_obs_max), dtype=np.int32)
        mask = np.zeros((n_batch, n_obs_max), dtype=bool)
        names = []

        for i, transient in enumerate(transients):
            n_obs = lengths[i]
            obs_times[i, :n_obs] = np.asarray(transient.time, dtype=float)
            obs_mags[i, :n_obs] = np.asarray(transient.y, dtype=float)
            obs_errs[i, :n_obs] = np.asarray(transient.y_err, dtype=float)
            obs_band_idx[i, :n_obs] = np.asarray(
                [band_to_idx[band] for band in transient.bands],
                dtype=np.int32,
            )
            mask[i, :n_obs] = True
            names.append(getattr(transient, "name", f"transient_{i}"))

        object.__setattr__(self, "transients", tuple(transients))
        object.__setattr__(self, "bands", tuple(band_order))
        object.__setattr__(self, "band_to_idx", band_to_idx)
        object.__setattr__(self, "names", tuple(names))
        object.__setattr__(self, "n_batch", n_batch)
        object.__setattr__(self, "n_obs_max", n_obs_max)
        object.__setattr__(self, "obs_counts", jnp.asarray(lengths, dtype=jnp.int32))
        object.__setattr__(self, "obs_times", jnp.asarray(obs_times))
        object.__setattr__(self, "obs_mags", jnp.asarray(obs_mags))
        object.__setattr__(self, "obs_errs", jnp.asarray(obs_errs))
        object.__setattr__(self, "obs_band_idx", jnp.asarray(obs_band_idx, dtype=jnp.int32))
        object.__setattr__(self, "mask", jnp.asarray(mask))

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(B, N_obs_max)``."""
        return self.n_batch, self.n_obs_max

    def data_tuple(self):
        """Return the padded arrays consumed by batched likelihoods."""
        return (
            self.obs_times,
            self.obs_mags,
            self.obs_errs,
            self.obs_band_idx,
            self.mask,
        )

    def __len__(self) -> int:
        return self.n_batch


def _union_bands(transients: Iterable) -> list[str]:
    bands = []
    seen = set()
    for transient in transients:
        for band in (getattr(transient, "bands", None) or []):
            if band not in seen:
                seen.add(band)
                bands.append(band)
    return bands


def _validate_transient(transient, band_to_idx: dict[str, int]) -> int:
    required = ("time", "y", "y_err", "bands")
    missing = [name for name in required if not hasattr(transient, name)]
    if missing:
        raise ValueError(f"Transient is missing required attributes: {missing}")

    if transient.time is None or transient.y is None or transient.y_err is None:
        raise ValueError("Transient time, y, and y_err must all be provided")
    if transient.bands is None:
        raise ValueError("Transient bands must be provided")

    n_obs = len(transient.time)
    if n_obs == 0:
        raise ValueError("Transient has no observations")
    if len(transient.y) != n_obs or len(transient.y_err) != n_obs:
        raise ValueError("Transient time, y, and y_err arrays must have the same length")
    if len(transient.bands) != n_obs:
        raise ValueError("Transient bands must have the same length as time")

    unknown = [band for band in transient.bands if band not in band_to_idx]
    if unknown:
        raise ValueError(f"Transient contains bands absent from global band set: {unknown}")
    return n_obs


@dataclass(frozen=True)
class BatchedFluxDensityDataset:
    """Padded flux-density observations for vmapped FluxDensityLikelihood inference.

    Parameters
    ----------
    times : (B, N_obs_max)  MJD per observation
    frequencies : (B, N_obs_max)  Hz per observation
    fluxes : (B, N_obs_max)  mJy
    flux_errs : (B, N_obs_max)  mJy (already including any error floor)
    mask : (B, N_obs_max)  bool — True for real observations
    t0_refs : (B,)  MJD reference time per SN (first observation); t0 prior is an offset from this
    names : sequence of str  SN names
    """

    obs_times:    jnp.ndarray
    obs_freq:     jnp.ndarray
    obs_flux:     jnp.ndarray
    obs_flux_err: jnp.ndarray
    mask:         jnp.ndarray
    t0_refs:      jnp.ndarray
    names:        tuple
    n_batch:      int
    n_obs_max:    int

    @classmethod
    def from_arrays(
        cls,
        times_list,
        freqs_list,
        fluxes_list,
        flux_errs_list,
        names=None,
    ):
        """Build from per-SN lists of 1-D numpy arrays."""
        B = len(times_list)
        N_max = max(len(t) for t in times_list)

        times_list = [np.asarray(t, dtype=np.float64) for t in times_list]
        freqs_list = [np.asarray(nu, dtype=np.float64) for nu in freqs_list]
        obs_times    = np.zeros((B, N_max), dtype=np.float64)
        obs_freq     = np.zeros((B, N_max), dtype=np.float64)
        obs_flux     = np.zeros((B, N_max), dtype=np.float64)
        obs_flux_err = np.full((B, N_max), np.inf, dtype=np.float64)
        mask         = np.zeros((B, N_max), dtype=bool)
        t0_refs      = np.zeros(B, dtype=np.float64)

        for i, (t, nu, f, fe) in enumerate(
            zip(times_list, freqs_list, fluxes_list, flux_errs_list)
        ):
            n = len(t)
            obs_times[i, :n]    = t
            # Pad with a valid real observation so the model is never
            # evaluated at nu=0 / t=0 in masked rows (would give NaN/inf).
            obs_times[i, n:]    = t[0]
            obs_freq[i, :n]     = nu
            obs_freq[i, n:]     = nu[0]
            obs_flux[i, :n]     = f
            obs_flux_err[i, :n] = fe
            mask[i, :n]         = True
            t0_refs[i]          = float(np.min(t))

        return cls(
            obs_times    = jnp.asarray(obs_times),
            obs_freq     = jnp.asarray(obs_freq),
            obs_flux     = jnp.asarray(obs_flux),
            obs_flux_err = jnp.asarray(obs_flux_err),
            mask         = jnp.asarray(mask),
            t0_refs      = jnp.asarray(t0_refs),
            names        = tuple(names or [f"sn_{i}" for i in range(B)]),
            n_batch      = B,
            n_obs_max    = N_max,
        )

    @property
    def shape(self):
        return self.n_batch, self.n_obs_max

    def __len__(self):
        return self.n_batch
