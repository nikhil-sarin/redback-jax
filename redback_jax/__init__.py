"""
Redback-JAX: a JAX-native companion to the Redback transient-analysis stack.

The package provides selected models and composable workflows for rapid,
differentiable electromagnetic-transient analysis and Bayesian inference.
"""

__version__ = "0.4.1"
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
