"""
Source classes for transient modeling, similar to sncosmo/jax-bandflux interface.

These classes provide a convenient interface for calculating bandfluxes and magnitudes
from precomputed or on-the-fly generated spectra, using JAX-bandflux's native capabilities.
"""

import jax.numpy as jnp
import numpy as np
from typing import Union, Optional, Dict
from jax_supernovae.bandpasses import get_bandpass, register_all_bandpasses
from jax_supernovae.constants import HC_ERG_AA, C_AA_PER_S


class PrecomputedSpectraSource:
    """
    A source class for precomputed spectral time series.

    This class provides an interface similar to SALT3Source from jax-bandflux,
    but for precomputed spectra (e.g., from arnett_model output). Uses JAX-bandflux's
    bandpass objects for all integration.

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
    ...     times=output.time,
    ...     wavelengths=output.lambdas,
    ...     spectra=output.spectra,
    ...     redshift=0.01
    ... )
    >>>
    >>> # Option 2: Direct from model
    >>> source = PrecomputedSpectraSource.from_arnett_model(
    ...     f_nickel=0.1, mej=1.4, vej=5000, redshift=0.01
    ... )
    >>>
    >>> # Query magnitudes (like SALT3Source)
    >>> mag = source.bandmag('bessellv', phase=15.0)
    >>> flux = source.bandflux('bessellv', phase=15.0)
    """

    def __init__(
        self,
        times: Union[np.ndarray, jnp.ndarray],
        wavelengths: Union[np.ndarray, jnp.ndarray],
        spectra: Union[np.ndarray, jnp.ndarray],
        redshift: float = 0.0,
        time_frame: str = 'observer'
    ):
        """
        Initialize a precomputed spectra source.

        Parameters
        ----------
        times : array_like
            Time array in days. Shape: (n_times,)
        wavelengths : array_like
            Wavelength array in Angstroms. Shape: (n_wavelengths,)
        spectra : array_like
            Spectral flux density array in erg s^-1 cm^-2 Angstrom^-1.
            Shape: (n_times, n_wavelengths)
        redshift : float, optional
            Source redshift (default: 0.0)
        time_frame : str, optional
            Whether times are in 'observer' or 'source' frame (default: 'observer')
        """
        self.times = jnp.asarray(times)
        self.wavelengths = jnp.asarray(wavelengths)
        self.spectra = jnp.asarray(spectra)
        self.redshift = redshift
        self.time_frame = time_frame

        # Ensure bandpasses are registered
        register_all_bandpasses()

        # Validate shapes
        if self.spectra.shape != (len(self.times), len(self.wavelengths)):
            raise ValueError(
                f"Spectra shape {self.spectra.shape} does not match "
                f"expected shape ({len(self.times)}, {len(self.wavelengths)})"
            )

    def _get_spectrum_at_phase(self, phase: float) -> jnp.ndarray:
        """
        Interpolate spectrum at given phase.

        Parameters
        ----------
        phase : float
            Time/phase to interpolate at (in same frame as self.times)

        Returns
        -------
        jnp.ndarray
            Interpolated spectrum of shape (n_wavelengths,)
        """
        # Check if phase is within bounds
        if phase < self.times[0] or phase > self.times[-1]:
            # Return zeros if outside bounds
            return jnp.zeros_like(self.wavelengths)

        # Linear interpolation in time for each wavelength
        spectrum = jnp.array([
            jnp.interp(phase, self.times, self.spectra[:, i])
            for i in range(len(self.wavelengths))
        ])

        return spectrum

    def bandflux(
        self,
        band: str,
        phase: float,
        zp: Optional[float] = None,
        zpsys: str = 'ab'
    ) -> float:
        """
        Calculate bandflux at given phase using JAX-bandflux integration.

        Parameters
        ----------
        band : str
            Bandpass name (e.g., 'bessellb', 'bessellv')
        phase : float
            Phase/time in days (observer frame)
        zp : float, optional
            Zero point (not used for flux calculation)
        zpsys : str, optional
            Magnitude system (default: 'ab')

        Returns
        -------
        float
            Bandflux in photons s^-1 cm^-2
        """
        # Get bandpass from JAX-bandflux
        bandpass = get_bandpass(band)

        # Get spectrum at this phase
        spectrum = self._get_spectrum_at_phase(phase)

        # Use bandpass's integration grid
        wave_grid = bandpass.integration_wave
        trans = bandpass(wave_grid)

        # Interpolate spectrum onto the integration grid
        flux_interp = jnp.interp(wave_grid, self.wavelengths, spectrum, left=0.0, right=0.0)

        # Integrate: int(lambda * T(lambda) * F_lambda * dlambda) / (h*c)
        # This is the standard sncosmo/JAX-bandflux integration formula
        integrand = wave_grid * trans * flux_interp
        bandflux = jnp.trapezoid(integrand, wave_grid) / HC_ERG_AA

        return float(bandflux)

    def bandmag(
        self,
        band: str,
        phase: float,
        magsys: str = 'ab',
        zp: float = 0.0
    ) -> float:
        """
        Calculate magnitude at given phase.

        Parameters
        ----------
        band : str
            Bandpass name (e.g., 'bessellb', 'bessellv')
        phase : float
            Phase/time in days (observer frame)
        magsys : str, optional
            Magnitude system (default: 'ab')
        zp : float, optional
            Zero point offset (default: 0.0)

        Returns
        -------
        float
            AB magnitude

        Examples
        --------
        >>> mag = source.bandmag('bessellv', phase=15.0)
        >>> mag = source.bandmag('bessellb', phase=10.0/(1+0.5))  # Rest frame
        """
        # Get bandpass from JAX-bandflux
        bandpass = get_bandpass(band)

        # Get spectrum at this phase
        spectrum = self._get_spectrum_at_phase(phase)

        # Use bandpass's integration grid
        wave_grid = bandpass.integration_wave
        trans = bandpass(wave_grid)
        dwave = bandpass.integration_spacing

        # Interpolate spectrum onto the integration grid
        flux_interp = jnp.interp(wave_grid, self.wavelengths, spectrum, left=0.0, right=0.0)

        # Integrate to get bandflux
        integrand = wave_grid * trans * flux_interp
        bandflux = jnp.trapezoid(integrand, wave_grid) / HC_ERG_AA

        # Calculate AB zeropoint bandflux
        # H_ERG_S = HC_ERG_AA / C_AA_PER_S
        H_ERG_S = HC_ERG_AA / C_AA_PER_S
        zpbandflux = 3631e-23 * dwave / H_ERG_S * jnp.sum(trans / wave_grid)

        # Convert to magnitude
        flux_safe = jnp.maximum(bandflux, 1e-50)
        mag = -2.5 * jnp.log10(flux_safe / zpbandflux) + zp

        return float(mag)

    def bandflux_multi(
        self,
        bands: list,
        phases: Union[float, np.ndarray, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """
        Calculate bandflux for multiple bands and/or phases.

        Parameters
        ----------
        bands : list of str
            List of bandpass names
        phases : float or array_like
            Single phase or array of phases in days

        Returns
        -------
        dict
            Dictionary mapping band names to flux arrays
        """
        phases = jnp.atleast_1d(jnp.asarray(phases))

        result = {}
        for band in bands:
            fluxes = jnp.array([self.bandflux(band, phase) for phase in phases])
            result[band] = fluxes

        return result

    def bandmag_multi(
        self,
        bands: list,
        phases: Union[float, np.ndarray, jnp.ndarray],
        magsys: str = 'ab',
        zp: float = 0.0
    ) -> Dict[str, jnp.ndarray]:
        """
        Calculate magnitudes for multiple bands and/or phases.

        Parameters
        ----------
        bands : list of str
            List of bandpass names
        phases : float or array_like
            Single phase or array of phases in days
        magsys : str, optional
            Magnitude system (default: 'ab')
        zp : float, optional
            Zero point offset (default: 0.0)

        Returns
        -------
        dict
            Dictionary mapping band names to magnitude arrays

        Examples
        --------
        >>> phases = jnp.linspace(5, 50, 20)
        >>> mags = source.bandmag_multi(['bessellb', 'bessellv'], phases)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(phases, mags['bessellb'], label='B')
        >>> plt.plot(phases, mags['bessellv'], label='V')
        """
        phases = jnp.atleast_1d(jnp.asarray(phases))

        result = {}
        for band in bands:
            mags = jnp.array([self.bandmag(band, phase, magsys, zp) for phase in phases])
            result[band] = mags

        return result

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
        time_min: float = 0.1,
        time_max: float = 100.0,
        n_times: int = 200
    ) -> 'PrecomputedSpectraSource':
        """
        Create a source directly from Arnett model parameters.

        This is a convenience method that generates the spectra and creates
        the source in one step.

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
        time_min : float, optional
            Minimum time in days (default: 0.1)
        time_max : float, optional
            Maximum time in days (default: 100.0)
        n_times : int, optional
            Number of time points (default: 200)

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
        >>> mag = source.bandmag('bessellv', phase=15.0)
        """
        from redback_jax.models.supernova_models import arnett_model

        # Generate spectra
        times = jnp.linspace(time_min, time_max, n_times)

        output = arnett_model(
            time=times,
            f_nickel=f_nickel,
            mej=mej,
            vej=vej,
            kappa=kappa,
            kappa_gamma=kappa_gamma,
            temperature_floor=temperature_floor,
            redshift=redshift,
            output_format='spectra'
        )

        return cls(
            times=output.time,
            wavelengths=output.lambdas,
            spectra=output.spectra,
            redshift=redshift,
            time_frame='observer'
        )
