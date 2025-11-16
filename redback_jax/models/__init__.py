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

__all__ = [
    # Supernova models
    'arnett_bolometric',
    'arnett_with_features_cosmology',
    'blackbody_to_flux_density',
    # SED features
    'SEDFeatures',
    'NO_SED_FEATURES',
    'apply_sed_feature',
]
