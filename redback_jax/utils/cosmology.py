"""
Cosmological constants and utilities for redback-jax.

Usage::

    from redback_jax.utils import PLANCK18_H0, PLANCK18_OM0, luminosity_distance_cm
"""

# Planck 2018 cosmology (Planck Collaboration 2018, A&A 641, A6)
PLANCK18_H0  = 67.66    # Hubble constant in km/s/Mpc
PLANCK18_OM0 = 0.3111   # Matter density parameter

MPC_TO_CM = 3.085677581e24  # cm per Mpc


def luminosity_distance_cm(redshift, H0=PLANCK18_H0, Om0=PLANCK18_OM0):
    """Luminosity distance in cm.

    Parameters
    ----------
    redshift : float
        Source redshift.
    H0 : float, optional
        Hubble constant in km/s/Mpc (default: Planck18).
    Om0 : float, optional
        Matter density parameter (default: Planck18).

    Returns
    -------
    float
        Luminosity distance in cm.

    Examples
    --------
    >>> dl_cm = luminosity_distance_cm(0.01)
    """
    from wcosmo import wcosmo as _wcosmo
    return _wcosmo.luminosity_distance(redshift, H0, Om0).value * MPC_TO_CM
