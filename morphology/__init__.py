"""Morphology module for galaxy size and shape measurements.

This module provides tools for:
- Sérsic profile fitting using PetroFit
- Concentration index calculation
- Star/galaxy classification (classical and ML-based)
- Half-light radius measurements
- Petrosian radius analysis

Example usage:
    from morphology import measure_sersic_params, concentration_index
    from morphology.star_galaxy import classify_star_galaxy

    # Measure Sérsic parameters
    params = measure_sersic_params(image, segm_map, source_id)
    print(f"Effective radius: {params['r_eff']:.2f} pixels")

    # Classify stars vs galaxies (classical method)
    is_galaxy = classify_star_galaxy(image, catalog)

    # ML-based classification (requires scikit-learn)
    from morphology import MLStarGalaxyClassifier, HAS_ML_CLASSIFIER
    if HAS_ML_CLASSIFIER:
        clf = MLStarGalaxyClassifier.load("model.joblib")
        results = clf.predict(features)
"""

from morphology.concentration import (
    calculate_concentration_c,
    concentration_index,
    petrosian_radius,
)
from morphology.sersic_fitting import (
    SersicParams,
    fit_sersic_profile,
    measure_sersic_params,
)
from morphology.star_galaxy import (
    classify_star_galaxy,
    get_stellarity_index,
)

# ML classifier with graceful fallback if sklearn not available
try:
    from morphology.ml_classifier import (
        HAS_SKLEARN as HAS_ML_CLASSIFIER,
    )
    from morphology.ml_classifier import (
        ClassificationResult,
        ClassifierMetrics,
        MLStarGalaxyClassifier,
        extract_features_from_catalog,
        extract_features_from_photometry,
    )
except ImportError:
    HAS_ML_CLASSIFIER = False
    MLStarGalaxyClassifier = None
    ClassificationResult = None
    ClassifierMetrics = None
    extract_features_from_catalog = None
    extract_features_from_photometry = None

__all__ = [
    # ML star-galaxy classification
    "HAS_ML_CLASSIFIER",
    "ClassificationResult",
    "ClassifierMetrics",
    "MLStarGalaxyClassifier",
    "SersicParams",
    "calculate_concentration_c",
    # Classical star-galaxy classification
    "classify_star_galaxy",
    # Concentration and morphology
    "concentration_index",
    "extract_features_from_catalog",
    "extract_features_from_photometry",
    "fit_sersic_profile",
    "get_stellarity_index",
    # Sérsic fitting
    "measure_sersic_params",
    "petrosian_radius",
]
