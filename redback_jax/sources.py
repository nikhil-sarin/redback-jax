"""
Source classes for transient modeling, similar to sncosmo/jax-bandflux interface.

These classes provide a convenient interface for calculating bandfluxes and magnitudes
from precomputed or on-the-fly generated spectra, using JAX-bandflux's native capabilities.
"""

import jax.numpy as jnp
import numpy as np
from typing import Union, Optional, Dict, Tuple
from jax_supernovae import TimeSeriesSource
from jax_supernovae.bandpasses import get_bandpass
from jax_supernovae.salt3 import precompute_bandflux_bridge


class PrecomputedSpectraSource:
    """
    A wrapper for TimeSeriesSource that follows the new functional API pattern.

    This class provides a thin wrapper around TimeSeriesSource, following the new
    functional API where parameters are passed as dictionaries to methods rather
    than stored in the object. This enables JAX's JIT compilation and automatic
    differentiation.

    The amplitude parameter scales the entire spectral time series, allowing for
    fitting to observed photometry.

    Examples
    --------
    >>> from redback_jax.models.supernova_models import arnett_model
    >>> from redback_jax.sources import PrecomputedSpectraSource
    >>>
    >>> # Option 1: From precomputed spectra
    >>> output = arnett_model(
    ...     time=jnp.linspace(0.1, 50, 100),
    ...     f_nickel=0.1, mej=1.4, vej=5000,
    ...     kappa=0.07, kappa_gamma=0.1,
    ...     temperature_floor=5000,
    ...     redshift=0.01,
    ...     output_format='spectra'
    ... )
    >>> source = PrecomputedSpectraSource(
    ...     phases=output.time,
    ...     wavelengths=output.lambdas,
    ...     flux_grid=output.spectra
    ... )
    >>>
    >>> # Simple mode: Single band calculation
    >>> params = {'amplitude': 1.0}
    >>> flux = source.bandflux(params, 'bessellv', 15.0)
    >>> mag = source.bandmag(params, 'bessellv', 15.0)
    >>>
    >>> # Optimized mode: Pre-compute bridges for fitting
    >>> unique_bands = ['bessellb', 'bessellv', 'bessellr']
    >>> bridges, band_to_idx = source.prepare_bridges(unique_bands)
    >>>
    >>> # Now use with array of phases and band indices for fast fitting
    >>> import jax
    >>> @jax.jit
    >>> def loglikelihood(amplitude, phases, band_indices, observed_fluxes, errors):
    ...     params = {'amplitude': amplitude}
    ...     model_fluxes = source.bandflux(params, None, phases,
    ...                                     band_indices=band_indices,
    ...                                     bridges=bridges,
    ...                                     unique_bands=unique_bands)
    ...     return -0.5 * jnp.sum(((observed_fluxes - model_fluxes) / errors)**2)
    >>>
    >>> # Option 2: Direct from model
    >>> source = PrecomputedSpectraSource.from_arnett_model(
    ...     f_nickel=0.1, mej=1.4, vej=5000, redshift=0.01
    ... )
    """

    def __init__(
        self,
        phases: Union[np.ndarray, jnp.ndarray],
        wavelengths: Union[np.ndarray, jnp.ndarray],
        flux_grid: Union[np.ndarray, jnp.ndarray],
        zero_before: bool = True,
        time_spline_degree: int = 3,
        name: Optional[str] = None,
        version: Optional[str] = None
    ):
        """
        Initialize a precomputed spectra source using TimeSeriesSource.

        Parameters
        ----------
        phases : array_like
            Phase array (times) in days. Shape: (n_phases,)
        wavelengths : array_like
            Wavelength array in Angstroms. Shape: (n_wavelengths,)
        flux_grid : array_like
            Spectral flux density array in erg s^-1 cm^-2 Angstrom^-1.
            Shape: (n_phases, n_wavelengths)
        zero_before : bool, optional
            Return zero flux before first phase (default: True)
        time_spline_degree : int, optional
            Degree of spline interpolation in time (default: 3 for cubic)
        name : str, optional
            Name of the source (default: None)
        version : str, optional
            Version identifier (default: None)
        """
        self.phases = jnp.asarray(phases)
        self.wavelengths = jnp.asarray(wavelengths)
        self.flux_grid = jnp.asarray(flux_grid)

        # Validate shapes
        if self.flux_grid.shape != (len(self.phases), len(self.wavelengths)):
            raise ValueError(
                f"Flux grid shape {self.flux_grid.shape} does not match "
                f"expected shape ({len(self.phases)}, {len(self.wavelengths)})"
            )

        # Create the underlying TimeSeriesSource from jaxbandflux
        # Note: TimeSeriesSource expects flux in erg s^-1 cm^-2 Angstrom^-1
        # TimeSeriesSource takes (phase, wave, flux) not (phases, wavelengths, flux_grid)
        self._source = TimeSeriesSource(
            phase=np.asarray(self.phases),
            wave=np.asarray(self.wavelengths),
            flux=np.asarray(self.flux_grid),
            zero_before=zero_before,
            time_spline_degree=time_spline_degree,
            name=name or 'redback_source',
            version=version or 'v1.0'
        )

    def bandflux(
        self,
        params: Dict[str, float],
        band: Optional[str],
        phase: Union[float, np.ndarray, jnp.ndarray],
        zp: Optional[float] = None,
        zpsys: str = 'ab',
        band_indices: Optional[jnp.ndarray] = None,
        bridges: Optional[Tuple] = None,
        unique_bands: Optional[list] = None
    ) -> Union[float, jnp.ndarray]:
        """
        Calculate bandflux at given phase(s) using TimeSeriesSource.

        Two modes of operation:
        1. Simple mode: Provide band name for single/multiple phase calculations
        2. Optimized mode: Provide band_indices, bridges, unique_bands for fast fitting

        Parameters
        ----------
        params : dict
            Parameter dictionary with 'amplitude' key for scaling the flux
        band : str or None
            Bandpass name (e.g., 'bessellb', 'bessellv'). Required for simple mode.
        phase : float or array_like
            Phase/time in days (observer frame). Can be scalar or array.
        zp : float, optional
            Zero point (default: None)
        zpsys : str, optional
            Magnitude system (default: 'ab')
        band_indices : jnp.ndarray, optional
            Band indices for optimized mode (shape: n_observations)
        bridges : tuple, optional
            Pre-computed bandpass bridges for optimized mode
        unique_bands : list, optional
            List of unique band names corresponding to bridges

        Returns
        -------
        float or jnp.ndarray
            Bandflux in photons s^-1 cm^-2. Scalar if phase is scalar, array otherwise.

        Examples
        --------
        # Simple mode
        >>> params = {'amplitude': 1.0}
        >>> flux = source.bandflux(params, 'bessellb', 0.0)

        # Optimized mode for fitting
        >>> bridges, band_to_idx = source.prepare_bridges(['bessellb', 'bessellv'])
        >>> band_indices = jnp.array([0, 0, 1, 1])  # Two B, two V observations
        >>> phases = jnp.array([0, 5, 0, 5])
        >>> fluxes = source.bandflux(params, None, phases,
        ...                          band_indices=band_indices,
        ...                          bridges=bridges,
        ...                          unique_bands=['bessellb', 'bessellv'])
        """
        # Delegate to TimeSeriesSource's bandflux method
        return self._source.bandflux(
            params, band, phase,
            zp=zp, zpsys=zpsys,
            band_indices=band_indices,
            bridges=bridges,
            unique_bands=unique_bands
        )

    def bandmag(
        self,
        params: Dict[str, float],
        band: str,
        phase: Union[float, np.ndarray, jnp.ndarray],
        magsys: str = 'ab'
    ) -> Union[float, jnp.ndarray]:
        """
        Calculate magnitude at given phase(s) using TimeSeriesSource.

        Parameters
        ----------
        params : dict
            Parameter dictionary with 'amplitude' key for scaling the flux
        band : str
            Bandpass name (e.g., 'bessellb', 'bessellv')
        phase : float or array_like
            Phase/time in days (observer frame)
        magsys : str, optional
            Magnitude system (default: 'ab')

        Returns
        -------
        float or jnp.ndarray
            AB magnitude. Scalar if phase is scalar, array otherwise.

        Examples
        --------
        >>> params = {'amplitude': 1.0}
        >>> mag = source.bandmag(params, 'bessellv', 15.0)
        >>> mags = source.bandmag(params, 'bessellb', jnp.array([0, 5, 10]))
        """
        # Delegate to TimeSeriesSource's bandmag method
        # Note: TimeSeriesSource.bandmag signature is (params, bands, magsys, phases, ...)
        return self._source.bandmag(params, band, magsys, phase)

    def prepare_bridges(
        self,
        unique_bands: list
    ) -> Tuple[Tuple, Dict[str, int]]:
        """
        Pre-compute bandpass bridges for optimized multi-band calculations.

        This enables efficient JIT-compiled fitting by pre-computing integration
        grids for each unique bandpass.

        Parameters
        ----------
        unique_bands : list of str
            List of unique bandpass names to pre-compute

        Returns
        -------
        bridges : tuple
            Tuple of pre-computed bandpass bridges
        band_to_idx : dict
            Dictionary mapping band names to indices in bridges tuple

        Examples
        --------
        >>> unique_bands = ['bessellb', 'bessellv', 'bessellr']
        >>> bridges, band_to_idx = source.prepare_bridges(unique_bands)
        >>> # Now use these in optimized bandflux calls
        >>> band_indices = jnp.array([band_to_idx[b] for b in obs_bands])
        >>> fluxes = source.bandflux(params, None, phases,
        ...                          band_indices=band_indices,
        ...                          bridges=bridges,
        ...                          unique_bands=unique_bands)
        """
        bridges = tuple(precompute_bandflux_bridge(get_bandpass(b)) for b in unique_bands)
        band_to_idx = {band: i for i, band in enumerate(unique_bands)}
        return bridges, band_to_idx

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
        time_spline_degree: int = 3,
        features = None
    ) -> 'PrecomputedSpectraSource':
        """
        Create a source directly from Arnett model parameters.

        This is a convenience method that generates the spectra using the new
        arnett_with_features_cosmology function and creates the source in one step.

        Parameters
        ----------
        f_nickel : float
            Nickel mass fraction
        mej : float
            Ejecta mass in solar masses
        vej : float
            Ejecta velocity in km/s
        kappa : float, optional
            Optical opacity (default: 0.07)
        kappa_gamma : float, optional
            Gamma-ray opacity (default: 0.1)
        temperature_floor : float, optional
            Minimum temperature in K (default: 5000.0)
        redshift : float, optional
            Source redshift (default: 0.0)
        cosmo_H0 : float, optional
            Hubble constant in km/s/Mpc (default: 67.66, Planck18)
        cosmo_Om0 : float, optional
            Matter density parameter (default: 0.3111, Planck18)
        zero_before : bool, optional
            Return zero flux before first phase (default: True)
        time_spline_degree : int, optional
            Degree of spline interpolation in time (default: 3)
        features : SEDFeatures, optional
            Spectral features to add (default: None, uses NO_SED_FEATURES)

        Returns
        -------
        PrecomputedSpectraSource
            Source instance with precomputed Arnett spectra

        Examples
        --------
        >>> source = PrecomputedSpectraSource.from_arnett_model(
        ...     f_nickel=0.1, mej=1.4, vej=5000,
        ...     kappa=0.07, kappa_gamma=0.1,
        ...     temperature_floor=5000, redshift=0.01
        ... )
        >>> params = {'amplitude': 1.0}
        >>> mag = source.bandmag(params, 'bessellv', 15.0)
        """
        from redback_jax.models.supernova_models import arnett_with_features_cosmology
        from redback_jax.models.sed_features import NO_SED_FEATURES

        # Use the new arnett_with_features_cosmology function
        # This automatically generates the time grid and spectra
        output = arnett_with_features_cosmology(
            f_nickel=f_nickel,
            mej=mej,
            redshift=redshift,
            cosmo_H0=cosmo_H0,
            cosmo_Om0=cosmo_Om0,
            vej=vej,
            kappa=kappa,
            kappa_gamma=kappa_gamma,
            temperature_floor=temperature_floor,
            features=features if features is not None else NO_SED_FEATURES
        )

        return cls(
            phases=output.time,
            wavelengths=output.lambdas,
            flux_grid=output.spectra,
            zero_before=zero_before,
            time_spline_degree=time_spline_degree,
            name='arnett_model',
            version='redback_jax'
        )
