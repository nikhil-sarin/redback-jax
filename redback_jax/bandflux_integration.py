"""
Integration between redback-jax spectral models and JAX-bandflux.

This module provides minimal wrapper functions to:
1. Take spectrum arrays from redback-jax models (e.g., arnett_model output)
2. Use JAX-bandflux Bandpass objects for filter handling
3. Integrate spectra over filters to calculate bandflux and magnitudes

Requires: pip install jax-bandflux
"""

import jax
import jax.numpy as jnp
from typing import Dict, List
from jax_supernovae.bandpasses import get_bandpass, Bandpass, register_all_bandpasses
from jax_supernovae.constants import HC_ERG_AA, C_AA_PER_S

# Calculate H_ERG_S from available constants
H_ERG_S = HC_ERG_AA / C_AA_PER_S

# Register all bandpasses at module import
_REGISTERED = False

def _ensure_bandpasses_registered():
    """Ensure standard bandpasses are registered."""
    global _REGISTERED
    if not _REGISTERED:
        try:
            # Register all standard bandpasses
            register_all_bandpasses()
            _REGISTERED = True
        except Exception as e:
            # If registration fails, print warning
            print(f"Warning: Could not register bandpasses: {e}")
            pass


def integrate_spectrum_bandflux(
    wavelength: jnp.ndarray,
    flux_lambda: jnp.ndarray,
    bandpass: Bandpass
) -> float:
    """
    Integrate a spectrum over a bandpass to get observed flux.

    This uses the same integration method as JAX-bandflux/sncosmo.

    Parameters
    ----------
    wavelength : jnp.ndarray
        Wavelength array in Angstroms (shape: n_wave)
    flux_lambda : jnp.ndarray
        Spectral flux density F_lambda in erg s^-1 cm^-2 Angstrom^-1 (shape: n_wave)
    bandpass : Bandpass
        JAX-bandflux Bandpass object

    Returns
    -------
    float
        Bandflux in photons s^-1 cm^-2

    Notes
    -----
    The integration follows the sncosmo/JAX-bandflux convention:
    F = integral(lambda * T(lambda) * F_lambda * dlambda) / (h*c)

    where the wavelength factor converts energy flux to photon flux.
    """
    # Get bandpass transmission on the integration grid
    wave_grid = bandpass.integration_wave
    trans = bandpass(wave_grid)

    # Interpolate spectrum onto the integration grid
    flux_interp = jnp.interp(wave_grid, wavelength, flux_lambda, left=0.0, right=0.0)

    # Integrate: wave * trans * flux
    integrand = wave_grid * trans * flux_interp
    bandflux = jnp.trapezoid(integrand, wave_grid) / HC_ERG_AA

    return bandflux


def bandflux_to_ab_mag(bandflux: float, bandpass: Bandpass, zp: float = 0.0) -> float:
    """
    Convert bandflux (in photons/s/cm^2) to AB magnitude.

    Parameters
    ----------
    bandflux : float
        Bandflux in photons s^-1 cm^-2
    bandpass : Bandpass
        JAX-bandflux Bandpass object
    zp : float, optional
        Zero point offset (default: 0.0)

    Returns
    -------
    float
        AB magnitude

    Notes
    -----
    Follows the sncosmo/JAX-bandflux convention:
    m_AB = -2.5 * log10(flux / zpbandflux) + zp

    For AB system: zpbandflux = 3631e-23 * dwave / H_ERG_S * sum(trans / wave)
    """
    wave = bandpass.integration_wave
    trans = bandpass(wave)
    dwave = bandpass.integration_spacing

    # Calculate AB zeropoint bandflux
    zpbandflux = 3631e-23 * dwave / H_ERG_S * jnp.sum(trans / wave)

    # Calculate magnitude
    # Add small epsilon to avoid log(0)
    flux_safe = jnp.maximum(bandflux, 1e-50)
    mag = -2.5 * jnp.log10(flux_safe / zpbandflux) + zp

    return mag


def spectra_to_lightcurves(
    times: jnp.ndarray,
    wavelengths: jnp.ndarray,
    spectra: jnp.ndarray,
    bands: List[str],
    output_format: str = 'magnitude',
    zp: float = 0.0
) -> Dict[str, jnp.ndarray]:
    """
    Convert spectrum array to multi-band light curves using JAX-bandflux.

    This is the main function to use with arnett_model or other redback-jax
    model outputs.

    Parameters
    ----------
    times : jnp.ndarray
        Time array (shape: n_times)
    wavelengths : jnp.ndarray
        Wavelength array in Angstroms (shape: n_wave)
    spectra : jnp.ndarray
        Spectrum array in erg s^-1 cm^-2 Angstrom^-1
        Shape: (n_times, n_wave)
    bands : List[str]
        List of bandpass names (uses JAX-bandflux naming: 'bessellb', 'bessellv', etc.)
    output_format : str, optional
        'magnitude' for AB magnitudes or 'flux' for bandflux in photons/s/cm^2
    zp : float, optional
        Zero point for magnitude calculation (default: 0.0)

    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary with filter names as keys and light curve arrays as values.
        Each light curve has shape (n_times,)

    Examples
    --------
    >>> # Get spectra from arnett model
    >>> from redback_jax.models.supernova_models import arnett_model
    >>> output = arnett_model(
    ...     time=jnp.linspace(0.1, 50, 50),
    ...     f_nickel=0.1, mej=1.4, vej=5000,
    ...     kappa=0.07, kappa_gamma=0.1,
    ...     temperature_floor=5000,
    ...     redshift=0.01,
    ...     output_format='spectra'
    ... )
    >>>
    >>> # Calculate light curves using JAX-bandflux filters
    >>> from redback_jax.bandflux_integration import spectra_to_lightcurves
    >>> lightcurves = spectra_to_lightcurves(
    ...     times=output.time,
    ...     wavelengths=output.lambdas,
    ...     spectra=output.spectra,
    ...     bands=['bessellb', 'bessellv', 'bessellr'],
    ...     output_format='magnitude'
    ... )
    >>>
    >>> # Plot results
    >>> import matplotlib.pyplot as plt
    >>> for band, mags in lightcurves.items():
    ...     plt.plot(output.time, mags, label=band)
    >>> plt.legend()
    >>> plt.xlabel('Time (days)')
    >>> plt.ylabel('AB Magnitude')
    >>> plt.gca().invert_yaxis()
    >>> plt.show()
    """
    # Ensure bandpasses are registered
    _ensure_bandpasses_registered()

    # Load bandpasses using JAX-bandflux
    bandpasses = {band: get_bandpass(band) for band in bands}

    # Initialize output dictionary
    lightcurves = {band: [] for band in bands}

    # Loop over times
    for i in range(len(times)):
        spectrum = spectra[i, :]

        # Loop over bands
        for band_name, bandpass in bandpasses.items():
            # Integrate spectrum over filter
            bandflux = integrate_spectrum_bandflux(wavelengths, spectrum, bandpass)

            # Convert to magnitude if requested
            if output_format == 'magnitude':
                value = bandflux_to_ab_mag(bandflux, bandpass, zp=zp)
            else:
                value = bandflux

            lightcurves[band_name].append(value)

    # Convert lists to JAX arrays
    lightcurves = {band: jnp.array(values) for band, values in lightcurves.items()}

    return lightcurves
