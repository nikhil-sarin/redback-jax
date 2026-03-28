"""
JAX-based transient models for electromagnetic counterparts.
"""

# Supernova models
from .supernova_models import (
    arnett_bolometric,
    arnett_with_features_cosmology,
    blackbody_to_flux_density,
)

# SED features
from .sed_features import (
    SEDFeatures,
    NO_SED_FEATURES,
    apply_sed_feature,
)

# Registry and plugin loader
from .registry import MODEL_REGISTRY, register_model, get_model, load_plugins

# Register built-in models
register_model("arnett_bolometric", arnett_bolometric)
register_model("arnett_with_features_cosmology", arnett_with_features_cosmology)

# Load any installed plugins (e.g. snmix)
load_plugins()

__all__ = [
    # Supernova models
    'arnett_bolometric',
    'arnett_with_features_cosmology',
    'blackbody_to_flux_density',
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
