"""
Redback-JAX: A lightweight JAX-only version of the redback electromagnetic transient analysis package.

This package provides JAX-based implementations for electromagnetic transient modeling
and Bayesian inference, focusing on performance and automatic differentiation capabilities.
"""

__version__ = "0.1.0"
__author__ = "Nikhil Sarin"
__email__ = "nsarin.astro@gmail.com"

import jax
jax.config.update("jax_enable_x64", True)

# Core imports
from .transient import Transient, Spectrum
from .sources import PrecomputedSpectraSource