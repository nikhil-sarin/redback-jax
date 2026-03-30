"""
Source classes for transient modeling, wrapping jax-bandflux TimeSeriesSource.
"""

import numpy as np
import jax.numpy as jnp
from typing import Union, Optional, Dict, Tuple

from jax_supernovae.bandpasses import get_bandpass, register_all_bandpasses
from jax_supernovae.salt3 import precompute_bandflux_bridge
from jax_supernovae.source import TimeSeriesSource


class PrecomputedSpectraSource:
    """Spectral time series source backed by jax-bandflux TimeSeriesSource.

    Wraps :class:`jax_supernovae.source.TimeSeriesSource` with redback-style
    constructor keywords (``phases``, ``wavelengths``, ``flux_grid``) and
    the ``prepare_bridges`` / ``from_arnett_model`` helpers used throughout
    redback-jax.

    Magnitudes are computed via the AB system using jax-bandflux's
    ``bandmag(params, band, 'ab', phases)`` which gives correct apparent
    magnitudes (e.g. ~14–16 mag for a z=0.01 Type Ia SN).

    Parameters
    ----------
    phases : array_like
        Phase grid in days (n_phases,).
    wavelengths : array_like
        Wavelength grid in Angstroms (n_wavelengths,).
    flux_grid : array_like
        Flux density in erg/s/cm²/Å — shape (n_phases, n_wavelengths).
    zero_before : bool, optional
        Return zero flux before the first phase (default True).
    name : str, optional
        Source name.
    version : str, optional
        Version tag.
    """

    def __init__(
        self,
        phases: Union[np.ndarray, jnp.ndarray],
        wavelengths: Union[np.ndarray, jnp.ndarray],
        flux_grid: Union[np.ndarray, jnp.ndarray],
        zero_before: bool = True,
        name: Optional[str] = None,
        version: Optional[str] = None,
    ):
        self.phases      = jnp.asarray(phases)
        self.wavelengths = jnp.asarray(wavelengths)
        self.flux_grid   = jnp.asarray(flux_grid)
        self.name        = name or 'redback_source'
        self.version     = version or 'v1.0'

        self.zero_before = zero_before

        self._source = TimeSeriesSource(
            phase=np.asarray(phases),
            wave=np.asarray(wavelengths),
            flux=np.asarray(flux_grid),
            zero_before=zero_before,
            time_spline_degree=1,
            name=name,
            version=version,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bandflux(
        self,
        params: Dict,
        band: Optional[str],
        phase: Union[float, np.ndarray, jnp.ndarray],
        zp: Optional[float] = None,
        zpsys: str = 'ab',
        band_indices: Optional[jnp.ndarray] = None,
        bridges: Optional[Tuple] = None,
        unique_bands: Optional[list] = None,
    ) -> jnp.ndarray:
        """Bandflux in photons/s/cm².

        Simple mode: provide ``band`` (str).
        Optimised mode: provide ``band_indices``, ``bridges``, ``unique_bands``.
        """
        if 'amplitude' not in params:
            params = {'amplitude': 1.0, **params}
        if band_indices is not None and bridges is not None and unique_bands is not None:
            return self._source.bandflux(
                params, None, phase,
                zp=zp, zpsys=zpsys if zp is not None else None,
                band_indices=band_indices, bridges=bridges,
                unique_bands=unique_bands,
            )
        return self._source.bandflux(params, band, phase, zp=zp,
                                      zpsys=zpsys if zp is not None else None)

    def bandmag(
        self,
        params: Dict,
        band: str,
        phase: Union[float, np.ndarray, jnp.ndarray],
        magsys: str = 'ab',
    ) -> jnp.ndarray:
        """AB magnitude at given phase(s)."""
        if 'amplitude' not in params:
            params = {'amplitude': 1.0, **params}
        return self._source.bandmag(params, band, magsys, phase)

    def prepare_bridges(self, unique_bands: list) -> Tuple[Tuple, Dict[str, int]]:
        """Pre-compute bandpass bridges for the optimised multi-band path."""
        register_all_bandpasses()
        bridges = tuple(precompute_bandflux_bridge(get_bandpass(b)) for b in unique_bands)
        band_to_idx = {b: i for i, b in enumerate(unique_bands)}
        return bridges, band_to_idx

    # ------------------------------------------------------------------
    # Class method
    # ------------------------------------------------------------------

    @classmethod
    def from_arnett_model(
        cls,
        f_nickel: float,
        mej: float,
        vej: float,
        kappa: float = 0.07,
        kappa_gamma: float = 0.1,
        temperature_floor: float = 5000.0,
        redshift: float = 0.0,
        cosmo_H0: float = 67.66,
        cosmo_Om0: float = 0.3111,
        zero_before: bool = True,
        features=None,
    ) -> 'PrecomputedSpectraSource':
        """Create a source from Arnett model parameters."""
        from redback_jax.models.supernova_models import arnett_with_features_cosmology
        from redback_jax.models.sed_features import NO_SED_FEATURES

        output = arnett_with_features_cosmology(
            f_nickel=f_nickel, mej=mej, redshift=redshift,
            cosmo_H0=cosmo_H0, cosmo_Om0=cosmo_Om0, vej=vej,
            kappa=kappa, kappa_gamma=kappa_gamma,
            temperature_floor=temperature_floor,
            features=features if features is not None else NO_SED_FEATURES,
        )
        return cls(
            phases=output.time, wavelengths=output.lambdas, flux_grid=output.spectra,
            zero_before=zero_before, name='arnett_model', version='redback_jax',
        )
