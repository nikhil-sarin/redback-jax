"""
Redback-JAX: A lightweight JAX-only version of the redback electromagnetic transient analysis package.

This package provides JAX-based implementations for electromagnetic transient modeling
and Bayesian inference, focusing on performance and automatic differentiation capabilities.
"""

__version__ = "0.1.0"
__author__ = "Nikhil Sarin"
__email__ = "nsarin.astro@gmail.com"


def __getattr__(name):
    """Lazy imports to avoid enabling JAX x64 at package load time."""
    if name == 'Transient' or name == 'Spectrum':
        from .transient import Transient, Spectrum
        globals()['Transient'] = Transient
        globals()['Spectrum'] = Spectrum
        return globals()[name]
    if name == 'PrecomputedSpectraSource':
        from .sources import PrecomputedSpectraSource
        globals()['PrecomputedSpectraSource'] = PrecomputedSpectraSource
        return PrecomputedSpectraSource
    raise AttributeError(f"module 'redback_jax' has no attribute {name!r}")
