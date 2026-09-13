"""Composable JAX primitives for native Redback afterglow models."""

from .core import native_afterglow_flux_density
from .dynamics import (
    arbitrary_csm_impulsive_dynamics,
    legacy_impulsive_dynamics,
    legacy_refreshed_dynamics,
    powered_thin_shell_dynamics,
)
from .engine import (
    constant_engine_cumulative_log10,
    fallback_engine_cumulative_log10,
    tabulated_engine_cumulative_log10,
)
from .geometry import angular_mesh, observer_angle
from .medium import (
    power_law_density,
    power_law_log_density,
    smoothly_broken_power_law_log_density,
    swept_mass_derivative,
    tabulated_log_density,
)
from .radiation import forward_shock_state, observer_state, synchrotron_log_flux
from .structure import jet_structure

__all__ = [
    "angular_mesh",
    "arbitrary_csm_impulsive_dynamics",
    "constant_engine_cumulative_log10",
    "fallback_engine_cumulative_log10",
    "jet_structure",
    "legacy_impulsive_dynamics",
    "legacy_refreshed_dynamics",
    "native_afterglow_flux_density",
    "observer_angle",
    "observer_state",
    "power_law_density",
    "power_law_log_density",
    "smoothly_broken_power_law_log_density",
    "powered_thin_shell_dynamics",
    "swept_mass_derivative",
    "forward_shock_state",
    "synchrotron_log_flux",
    "tabulated_engine_cumulative_log10",
    "tabulated_log_density",
]
