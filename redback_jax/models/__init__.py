"""
JAX-based transient models for electromagnetic counterparts.
"""

# Supernova models
from .supernova_models import (
    arnett_bolometric,
    arnett_with_features_cosmology,
    blackbody_to_flux_density,
    magnetar_powered_bolometric,
    magnetar_nickel_bolometric,
    csm_interaction_bolometric,
)

# Shock-powered models
from .shock_powered_models import (
    shock_cooling_bolometric,
    shocked_cocoon_bolometric,
    shock_cooling_and_arnett_bolometric,
)

# TDE models
from .tde_models import (
    tde_analytical_bolometric,
)

# Kilonova models
from .kilonova import (
    metzger_kilonova_bolometric,
    magnetar_boosted_kilonova_bolometric,
)

# General magnetar-driven supernova (relativistic ODE, float64)
from .supernova_models import (
    general_magnetar_driven_supernova_bolometric,
    general_magnetar_driven_supernova_bolometric_and_vej,
    general_magnetar_driven_supernova_bolometric_batched,
    general_magnetar_driven_supernova_bolometric_diffrax,
    general_magnetar_driven_supernova_bolometric_and_vej_diffrax,
    general_magnetar_driven_supernova,
    general_magnetar_driven_supernova_diffrax,
)

# SED features
from .sed_features import (
    SEDFeatures,
    NO_SED_FEATURES,
    apply_sed_feature,
)

# Generic spectra factories (optional — requires jax_supernovae)
try:
    from .spectra_model import make_spectra_model, make_cutoff_spectra_model
    _SPECTRA_MODEL_AVAILABLE = True
except ImportError:
    _SPECTRA_MODEL_AVAILABLE = False
    make_spectra_model = None
    make_cutoff_spectra_model = None

# Registry and plugin loader — must come before any register_model calls
from .registry import MODEL_REGISTRY, register_model, get_model, load_plugins

# Register bolometric models
register_model("arnett_bolometric", arnett_bolometric)
register_model("arnett_with_features_cosmology", arnett_with_features_cosmology)
register_model("magnetar_powered_bolometric", magnetar_powered_bolometric)
register_model("magnetar_nickel_bolometric", magnetar_nickel_bolometric)
register_model("csm_interaction_bolometric", csm_interaction_bolometric)
register_model("shock_cooling_bolometric", shock_cooling_bolometric)
register_model("shocked_cocoon_bolometric", shocked_cocoon_bolometric)
register_model("shock_cooling_and_arnett_bolometric", shock_cooling_and_arnett_bolometric)
register_model("tde_analytical_bolometric", tde_analytical_bolometric)
register_model("metzger_kilonova_bolometric", metzger_kilonova_bolometric)
register_model("magnetar_boosted_kilonova_bolometric", magnetar_boosted_kilonova_bolometric)
register_model("general_magnetar_driven_supernova_bolometric", general_magnetar_driven_supernova_bolometric)
register_model("general_magnetar_driven_supernova_bolometric_and_vej", general_magnetar_driven_supernova_bolometric_and_vej)
register_model("general_magnetar_driven_supernova_bolometric_batched", general_magnetar_driven_supernova_bolometric_batched)
register_model("general_magnetar_driven_supernova_bolometric_diffrax", general_magnetar_driven_supernova_bolometric_diffrax)
register_model("general_magnetar_driven_supernova_bolometric_and_vej_diffrax", general_magnetar_driven_supernova_bolometric_and_vej_diffrax)
register_model("general_magnetar_driven_supernova", general_magnetar_driven_supernova)
register_model("general_magnetar_driven_supernova_diffrax", general_magnetar_driven_supernova_diffrax)

# Pre-built spectra variants (only if jax_supernovae is available)
if _SPECTRA_MODEL_AVAILABLE:
    arnett_spectra                    = make_spectra_model(arnett_bolometric)
    magnetar_powered_spectra          = make_spectra_model(magnetar_powered_bolometric)
    magnetar_nickel_spectra           = make_spectra_model(magnetar_nickel_bolometric)
    csm_interaction_spectra           = make_spectra_model(csm_interaction_bolometric)
    shock_cooling_spectra             = make_spectra_model(shock_cooling_bolometric)
    shocked_cocoon_spectra            = make_spectra_model(shocked_cocoon_bolometric)
    shock_cooling_and_arnett_spectra  = make_spectra_model(shock_cooling_and_arnett_bolometric)
    tde_analytical_spectra            = make_spectra_model(tde_analytical_bolometric)
    metzger_kilonova_spectra          = make_spectra_model(metzger_kilonova_bolometric)
    magnetar_boosted_kilonova_spectra = make_spectra_model(magnetar_boosted_kilonova_bolometric)

    # General magnetar CutoffBlackbody spectra model (vej from ODE)
    general_magnetar_supernova_spectra_diffrax = make_cutoff_spectra_model(
        general_magnetar_driven_supernova_bolometric_and_vej_diffrax
    )
    register_model("general_magnetar_supernova_spectra_diffrax", general_magnetar_supernova_spectra_diffrax)

    # Register spectra models
    register_model("arnett_spectra",                    arnett_spectra)
    register_model("magnetar_powered_spectra",          magnetar_powered_spectra)
    register_model("magnetar_nickel_spectra",           magnetar_nickel_spectra)
    register_model("csm_interaction_spectra",           csm_interaction_spectra)
    register_model("shock_cooling_spectra",             shock_cooling_spectra)
    register_model("shocked_cocoon_spectra",            shocked_cocoon_spectra)
    register_model("shock_cooling_and_arnett_spectra",  shock_cooling_and_arnett_spectra)
    register_model("tde_analytical_spectra",            tde_analytical_spectra)
    register_model("metzger_kilonova_spectra",          metzger_kilonova_spectra)
    register_model("magnetar_boosted_kilonova_spectra", magnetar_boosted_kilonova_spectra)
else:
    # When jax_supernovae is not installed these names must still exist in the
    # module namespace (they are listed in __all__) to prevent AttributeError
    # on wildcard imports or attribute access.
    arnett_spectra                    = None
    magnetar_powered_spectra          = None
    magnetar_nickel_spectra           = None
    csm_interaction_spectra           = None
    shock_cooling_spectra             = None
    shocked_cocoon_spectra            = None
    shock_cooling_and_arnett_spectra  = None
    tde_analytical_spectra            = None
    metzger_kilonova_spectra          = None
    magnetar_boosted_kilonova_spectra = None
    general_magnetar_supernova_spectra_diffrax = None

# Load any installed plugins (e.g. external model packages)
load_plugins()

__all__ = [
    # Supernova models
    'arnett_bolometric',
    'arnett_with_features_cosmology',
    'blackbody_to_flux_density',
    'magnetar_powered_bolometric',
    'magnetar_nickel_bolometric',
    'csm_interaction_bolometric',
    # Shock-powered models
    'shock_cooling_bolometric',
    'shocked_cocoon_bolometric',
    'shock_cooling_and_arnett_bolometric',
    # TDE models
    'tde_analytical_bolometric',
    # Kilonova models
    'metzger_kilonova_bolometric',
    'magnetar_boosted_kilonova_bolometric',
    # General magnetar-driven SN (ODE)
    'general_magnetar_driven_supernova_bolometric',
    'general_magnetar_driven_supernova_bolometric_and_vej',
    'general_magnetar_driven_supernova_bolometric_batched',
    'general_magnetar_driven_supernova_bolometric_diffrax',
    'general_magnetar_driven_supernova_bolometric_and_vej_diffrax',
    'general_magnetar_driven_supernova',
    'general_magnetar_driven_supernova_diffrax',
    'general_magnetar_supernova_spectra_diffrax',
    # Generic factories
    'make_spectra_model',
    'make_cutoff_spectra_model',
    # Spectra variants
    'arnett_spectra',
    'magnetar_powered_spectra',
    'magnetar_nickel_spectra',
    'csm_interaction_spectra',
    'shock_cooling_spectra',
    'shocked_cocoon_spectra',
    'shock_cooling_and_arnett_spectra',
    'tde_analytical_spectra',
    'metzger_kilonova_spectra',
    'magnetar_boosted_kilonova_spectra',
    # SED features
    'SEDFeatures',
    'NO_SED_FEATURES',
    'apply_sed_feature',
    # Registry
    'MODEL_REGISTRY',
    'register_model',
    'get_model',
    'load_plugins',
]
