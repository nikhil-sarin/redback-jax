"""Composable JAX primitives for native Redback afterglow models."""

from .geometry import angular_mesh, observer_angle
from .dynamics import legacy_impulsive_dynamics
from .medium import power_law_density, swept_mass_derivative
from .radiation import forward_shock_state, observer_state, synchrotron_log_flux
from .structure import jet_structure

__all__ = [
    "angular_mesh",
    "jet_structure",
    "legacy_impulsive_dynamics",
    "observer_angle",
    "observer_state",
    "power_law_density",
    "swept_mass_derivative",
    "forward_shock_state",
    "synchrotron_log_flux",
]
