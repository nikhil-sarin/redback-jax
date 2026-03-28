"""
JAX-based transient models for electromagnetic counterparts.
"""

# Supernova models
from .supernova_models import (
    arnett_bolometric,
    arnett_with_features_cosmology,
    blackbody_to_flux_density,
    magnetar_powered_bolometric,
    csm_interaction_bolometric,
)

# Shock-powered models
from .shock_powered_models import (
    shock_cooling_bolometric,
    shocked_cocoon_bolometric,
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

# SED features
from .sed_features import (
    SEDFeatures,
    NO_SED_FEATURES,
    apply_sed_feature,
)

# Generic spectra factory
from .spectra_model import make_spectra_model

# Registry and plugin loader — must come before any register_model calls
from .registry import MODEL_REGISTRY, register_model, get_model, load_plugins

# Register bolometric models
register_model("arnett_bolometric", arnett_bolometric)
register_model("arnett_with_features_cosmology", arnett_with_features_cosmology)
register_model("magnetar_powered_bolometric", magnetar_powered_bolometric)
register_model("csm_interaction_bolometric", csm_interaction_bolometric)
register_model("shock_cooling_bolometric", shock_cooling_bolometric)
register_model("shocked_cocoon_bolometric", shocked_cocoon_bolometric)
register_model("tde_analytical_bolometric", tde_analytical_bolometric)
register_model("metzger_kilonova_bolometric", metzger_kilonova_bolometric)
register_model("magnetar_boosted_kilonova_bolometric", magnetar_boosted_kilonova_bolometric)

# Pre-built spectra variants
magnetar_powered_spectra          = make_spectra_model(magnetar_powered_bolometric)
csm_interaction_spectra           = make_spectra_model(csm_interaction_bolometric)
shock_cooling_spectra             = make_spectra_model(shock_cooling_bolometric)
shocked_cocoon_spectra            = make_spectra_model(shocked_cocoon_bolometric)
tde_analytical_spectra            = make_spectra_model(tde_analytical_bolometric)
metzger_kilonova_spectra          = make_spectra_model(metzger_kilonova_bolometric)
magnetar_boosted_kilonova_spectra = make_spectra_model(magnetar_boosted_kilonova_bolometric)

# Register spectra models
register_model("magnetar_powered_spectra",          magnetar_powered_spectra)
register_model("csm_interaction_spectra",           csm_interaction_spectra)
register_model("shock_cooling_spectra",             shock_cooling_spectra)
register_model("shocked_cocoon_spectra",            shocked_cocoon_spectra)
register_model("tde_analytical_spectra",            tde_analytical_spectra)
register_model("metzger_kilonova_spectra",          metzger_kilonova_spectra)
register_model("magnetar_boosted_kilonova_spectra", magnetar_boosted_kilonova_spectra)

# Load any installed plugins (e.g. external model packages)
load_plugins()

__all__ = [
    # Supernova models
    'arnett_bolometric',
    'arnett_with_features_cosmology',
    'blackbody_to_flux_density',
    'magnetar_powered_bolometric',
    'csm_interaction_bolometric',
    # Shock-powered models
    'shock_cooling_bolometric',
    'shocked_cocoon_bolometric',
    # TDE models
    'tde_analytical_bolometric',
    # Kilonova models
    'metzger_kilonova_bolometric',
    'magnetar_boosted_kilonova_bolometric',
    # Generic factory
    'make_spectra_model',
    # Spectra variants
    'magnetar_powered_spectra',
    'csm_interaction_spectra',
    'shock_cooling_spectra',
    'shocked_cocoon_spectra',
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
