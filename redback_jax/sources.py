"""
Source classes for transient modeling, compatible with jax-bandflux interface.

These classes provide a convenient interface for calculating bandfluxes and magnitudes
from precomputed or on-the-fly generated spectra, using JAX-bandflux's bandpass utilities.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Union, Optional, Dict, Tuple
from functools import partial
from jax_supernovae.bandpasses import (
    get_bandpass,
    register_all_bandpasses,
    Bandpass
)
from jax_supernovae.constants import HC_ERG_AA


# Physical constants for bandflux calculation
# HC_ERG_AA is already in erg * Angstrom


def _interpolate_spectrum_1d(wavelengths: jnp.ndarray, flux: jnp.ndarray,
                              target_wave: jnp.ndarray) -> jnp.ndarray:
    """Linearly interpolate a spectrum to target wavelengths.

    Parameters
    ----------
    wavelengths : jnp.ndarray
        Source wavelengths (sorted)
    flux : jnp.ndarray
        Flux values at source wavelengths
    target_wave : jnp.ndarray
        Target wavelengths for interpolation

    Returns
    -------
    jnp.ndarray
        Interpolated flux values
    """
    return jnp.interp(target_wave, wavelengths, flux)


def _interpolate_spectrum_time(phases: jnp.ndarray, flux_grid: jnp.ndarray,
                                target_phase: float) -> jnp.ndarray:
    """Interpolate spectrum at a given phase from a time series grid.

    Parameters
    ----------
    phases : jnp.ndarray
        Phase array (n_phases,)
    flux_grid : jnp.ndarray
        Flux grid (n_phases, n_wavelengths)
    target_phase : float
        Target phase for interpolation

    Returns
    -------
    jnp.ndarray
        Interpolated spectrum at target phase (n_wavelengths,)
    """
    # Linear interpolation in time at each wavelength
    # Find surrounding indices
    idx = jnp.searchsorted(phases, target_phase)
    idx = jnp.clip(idx, 1, len(phases) - 1)

    # Get phases and fluxes for interpolation
    phase_lo = phases[idx - 1]
    phase_hi = phases[idx]
    flux_lo = flux_grid[idx - 1]
    flux_hi = flux_grid[idx]

    # Linear interpolation weight
    t = (target_phase - phase_lo) / (phase_hi - phase_lo + 1e-10)
    t = jnp.clip(t, 0.0, 1.0)

    return flux_lo + t * (flux_hi - flux_lo)


def _compute_bandflux_single(spectrum: jnp.ndarray, wavelengths: jnp.ndarray,
                               bandpass: Bandpass) -> jnp.ndarray:
    """Compute bandflux for a single spectrum and bandpass.

    The bandflux is computed as:
        flux = integral(f_lambda * T * lambda / (h*c)) dlambda

    where f_lambda is in erg s^-1 cm^-2 Angstrom^-1

    Parameters
    ----------
    spectrum : jnp.ndarray
        Spectral flux density (erg s^-1 cm^-2 Angstrom^-1)
    wavelengths : jnp.ndarray
        Wavelength array (Angstrom)
    bandpass : Bandpass
        Bandpass object with transmission curve

    Returns
    -------
    jnp.ndarray
        Bandflux in photons s^-1 cm^-2
    """
    # Get integration wavelength grid from bandpass
    wave_grid = bandpass.integration_wave
    dwave = bandpass.integration_spacing

    # Interpolate spectrum to integration grid
    interp_flux = _interpolate_spectrum_1d(wavelengths, spectrum, wave_grid)

    # Get transmission at integration wavelengths
    trans = bandpass(wave_grid)

    # Integrate: sum(f_lambda * T * lambda / (hc) * dlambda)
    # This gives photons s^-1 cm^-2
    integrand = interp_flux * trans * wave_grid / HC_ERG_AA
    bandflux = jnp.sum(integrand) * dwave

    return bandflux


def _compute_bandmag_single(bandflux: jnp.ndarray) -> jnp.ndarray:
    """Convert bandflux to AB magnitude.

    Parameters
    ----------
    bandflux : jnp.ndarray
        Bandflux in photons s^-1 cm^-2

    Returns
    -------
    jnp.ndarray
        AB magnitude
    """
    # AB magnitude system: m_AB = -2.5 * log10(flux) - 48.6
    # where flux is in erg s^-1 cm^-2 Hz^-1
    # For photon counting: m_AB = -2.5 * log10(bandflux) - 48.60
    # This is an approximation; proper AB system requires frequency integral
    # Using the standard formula for AB magnitudes from photon flux
    return -2.5 * jnp.log10(bandflux + 1e-100) - 48.60


class PrecomputedSpectraSource:
    """
    A source class for computing bandfluxes from precomputed spectral time series.

    This class provides methods for calculating bandfluxes and magnitudes from
    precomputed spectra, following a functional API where parameters are passed
    as dictionaries. This enables JAX's JIT compilation and automatic differentiation.

    The amplitude parameter scales the entire spectral time series, allowing for
    fitting to observed photometry.

    Examples
    --------
    >>> from redback_jax.models.supernova_models import arnett_with_features_cosmology
    >>> from redback_jax.sources import PrecomputedSpectraSource
    >>> from redback_jax.models.sed_features import NO_SED_FEATURES
    >>>
    >>> # Create from precomputed spectra
    >>> output = arnett_with_features_cosmology(
    ...     f_nickel=0.1, mej=1.4, vej=5000,
    ...     kappa=0.07, kappa_gamma=0.1,
    ...     temperature_floor=5000,
    ...     redshift=0.01,
    ...     features=NO_SED_FEATURES
    ... )
    >>> source = PrecomputedSpectraSource(
    ...     phases=output.time,
    ...     wavelengths=output.lambdas,
    ...     flux_grid=output.spectra
    ... )
    >>>
    >>> # Simple mode: Single band calculation
    >>> params = {'amplitude': 1.0}
    >>> flux = source.bandflux(params, 'g', 15.0)
    >>> mag = source.bandmag(params, 'g', 15.0)
    >>>
    >>> # Optimized mode: Pre-compute bridges for fitting
    >>> unique_bands = ['g', 'r', 'i']
    >>> bridges, band_to_idx = source.prepare_bridges(unique_bands)
    >>>
    >>> # Use with array of phases and band indices for fast fitting
    >>> import jax
    >>> @jax.jit
    ... def loglikelihood(amplitude, phases, band_indices, observed_fluxes, errors):
    ...     params = {'amplitude': amplitude}
    ...     model_fluxes = source.bandflux(params, None, phases,
    ...                                     band_indices=band_indices,
    ...                                     bridges=bridges,
    ...                                     unique_bands=unique_bands)
    ...     return -0.5 * jnp.sum(((observed_fluxes - model_fluxes) / errors)**2)
    """

    def __init__(
        self,
        phases: Union[np.ndarray, jnp.ndarray],
        wavelengths: Union[np.ndarray, jnp.ndarray],
        flux_grid: Union[np.ndarray, jnp.ndarray],
        zero_before: bool = True,
        name: Optional[str] = None,
        version: Optional[str] = None
    ):
        """
        Initialize a precomputed spectra source.

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
        name : str, optional
            Name of the source (default: None)
        version : str, optional
            Version identifier (default: None)
        """
        self.phases = jnp.asarray(phases)
        self.wavelengths = jnp.asarray(wavelengths)
        self.flux_grid = jnp.asarray(flux_grid)
        self.zero_before = zero_before
        self.name = name or 'redback_source'
        self.version = version or 'v1.0'

        # Validate shapes
        if self.flux_grid.shape != (len(self.phases), len(self.wavelengths)):
            raise ValueError(
                f"Flux grid shape {self.flux_grid.shape} does not match "
                f"expected shape ({len(self.phases)}, {len(self.wavelengths)})"
            )

        # Ensure bandpasses are registered
        register_all_bandpasses()

    def _get_spectrum_at_phase(self, phase: float, amplitude: float) -> jnp.ndarray:
        """Get interpolated spectrum at a given phase.

        Parameters
        ----------
        phase : float
            Observer frame time in days
        amplitude : float
            Amplitude scaling factor

        Returns
        -------
        jnp.ndarray
            Spectrum at given phase, scaled by amplitude
        """
        # Handle zero_before
        if self.zero_before:
            # Return zeros if before first phase
            spectrum = jax.lax.cond(
                phase < self.phases[0],
                lambda _: jnp.zeros_like(self.wavelengths),
                lambda _: _interpolate_spectrum_time(self.phases, self.flux_grid, phase),
                None
            )
        else:
            spectrum = _interpolate_spectrum_time(self.phases, self.flux_grid, phase)

        return amplitude * spectrum

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
        Calculate bandflux at given phase(s).

        Two modes of operation:
        1. Simple mode: Provide band name for single/multiple phase calculations
        2. Optimized mode: Provide band_indices, bridges, unique_bands for fast fitting

        Parameters
        ----------
        params : dict
            Parameter dictionary with 'amplitude' key for scaling the flux
        band : str or None
            Bandpass name (e.g., 'g', 'r', 'ztfg'). Required for simple mode.
        phase : float or array_like
            Phase/time in days (observer frame). Can be scalar or array.
        zp : float, optional
            Zero point (default: None, not used)
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
        >>> flux = source.bandflux(params, 'g', 0.0)

        # Optimized mode for fitting
        >>> bridges, band_to_idx = source.prepare_bridges(['g', 'r'])
        >>> band_indices = jnp.array([0, 0, 1, 1])  # Two g, two r observations
        >>> phases = jnp.array([0, 5, 0, 5])
        >>> fluxes = source.bandflux(params, None, phases,
        ...                          band_indices=band_indices,
        ...                          bridges=bridges,
        ...                          unique_bands=['g', 'r'])
        """
        amplitude = params.get('amplitude', 1.0)
        phase = jnp.asarray(phase)

        # Optimized mode: use precomputed bridges
        if band_indices is not None and bridges is not None and unique_bands is not None:
            return self._bandflux_optimized(
                amplitude, phase, band_indices, bridges, unique_bands
            )

        # Simple mode: single band
        if band is None:
            raise ValueError("Band must be specified in simple mode")

        bandpass = get_bandpass(band)

        # Handle scalar vs array phase
        if phase.ndim == 0:
            # Scalar phase
            spectrum = self._get_spectrum_at_phase(float(phase), amplitude)
            return _compute_bandflux_single(spectrum, self.wavelengths, bandpass)
        else:
            # Array of phases
            def compute_single(ph):
                spectrum = self._get_spectrum_at_phase(ph, amplitude)
                return _compute_bandflux_single(spectrum, self.wavelengths, bandpass)

            return jax.vmap(compute_single)(phase)

    def _bandflux_optimized(
        self,
        amplitude: float,
        phases: jnp.ndarray,
        band_indices: jnp.ndarray,
        bridges: Tuple,
        unique_bands: list
    ) -> jnp.ndarray:
        """Compute bandfluxes using precomputed bridges for multiple bands.

        Parameters
        ----------
        amplitude : float
            Amplitude scaling factor
        phases : jnp.ndarray
            Phase array (n_obs,)
        band_indices : jnp.ndarray
            Band indices (n_obs,)
        bridges : tuple
            Precomputed bandpass bridges
        unique_bands : list
            List of unique band names

        Returns
        -------
        jnp.ndarray
            Bandfluxes for each observation
        """
        # Get bandpass objects
        bandpasses = [get_bandpass(b) for b in unique_bands]

        def compute_single_obs(phase, band_idx):
            spectrum = self._get_spectrum_at_phase(phase, amplitude)
            # Select bandpass based on index
            # Use lax.switch for efficient branching
            def compute_for_band(i):
                return _compute_bandflux_single(spectrum, self.wavelengths, bandpasses[i])

            return jax.lax.switch(band_idx, [partial(compute_for_band, i) for i in range(len(bandpasses))])

        return jax.vmap(compute_single_obs)(phases, band_indices)

    def bandmag(
        self,
        params: Dict[str, float],
        band: str,
        phase: Union[float, np.ndarray, jnp.ndarray],
        magsys: str = 'ab'
    ) -> Union[float, jnp.ndarray]:
        """
        Calculate magnitude at given phase(s).

        Parameters
        ----------
        params : dict
            Parameter dictionary with 'amplitude' key for scaling the flux
        band : str
            Bandpass name (e.g., 'g', 'r', 'ztfg')
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
        >>> mag = source.bandmag(params, 'g', 15.0)
        >>> mags = source.bandmag(params, 'r', jnp.array([0, 5, 10]))
        """
        flux = self.bandflux(params, band, phase)
        return _compute_bandmag_single(flux)

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
            Tuple of pre-computed bandpass bridges (Bandpass objects)
        band_to_idx : dict
            Dictionary mapping band names to indices in bridges tuple

        Examples
        --------
        >>> unique_bands = ['g', 'r', 'i']
        >>> bridges, band_to_idx = source.prepare_bridges(unique_bands)
        >>> # Now use these in optimized bandflux calls
        >>> band_indices = jnp.array([band_to_idx[b] for b in obs_bands])
        >>> fluxes = source.bandflux(params, None, phases,
        ...                          band_indices=band_indices,
        ...                          bridges=bridges,
        ...                          unique_bands=unique_bands)
        """
        bridges = tuple(get_bandpass(b) for b in unique_bands)
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
        features = None
    ) -> 'PrecomputedSpectraSource':
        """
        Create a source directly from Arnett model parameters.

        This is a convenience method that generates the spectra using the
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
        >>> mag = source.bandmag(params, 'g', 15.0)
        """
        from redback_jax.models.supernova_models import arnett_with_features_cosmology
        from redback_jax.models.sed_features import NO_SED_FEATURES

        # Use the arnett_with_features_cosmology function
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
            name='arnett_model',
            version='redback_jax'
        )
