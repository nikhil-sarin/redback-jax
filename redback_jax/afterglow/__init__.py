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
from .geometry import (
    angular_mesh,
    angular_patch_coordinates,
    observer_angle,
    observer_angle_patches,
)
from .medium import (
    power_law_density,
    power_law_log_density,
    smoothly_broken_power_law_log_density,
    swept_mass_derivative,
    tabulated_log_density,
)
from .radiation import (
    forward_shock_state,
    legacy_radiation_prescription,
    observer_state,
    optically_thin_radiation_prescription,
    optically_thin_synchrotron_log_flux,
    smooth_synchrotron_log_flux,
    smooth_synchrotron_radiation_prescription,
    synchrotron_log_flux,
)
from .structure import jet_structure

__all__ = [
    "angular_mesh",
    "angular_patch_coordinates",
    "arbitrary_csm_impulsive_dynamics",
    "constant_engine_cumulative_log10",
    "fallback_engine_cumulative_log10",
    "jet_structure",
    "legacy_impulsive_dynamics",
    "legacy_refreshed_dynamics",
    "legacy_radiation_prescription",
    "native_afterglow_flux_density",
    "observer_angle",
    "observer_angle_patches",
    "observer_state",
    "optically_thin_radiation_prescription",
    "optically_thin_synchrotron_log_flux",
    "power_law_density",
    "power_law_log_density",
    "smoothly_broken_power_law_log_density",
    "powered_thin_shell_dynamics",
    "swept_mass_derivative",
    "forward_shock_state",
    "smooth_synchrotron_log_flux",
    "smooth_synchrotron_radiation_prescription",
    "synchrotron_log_flux",
    "tabulated_engine_cumulative_log10",
    "tabulated_log_density",
]
