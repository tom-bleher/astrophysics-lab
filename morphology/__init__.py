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

    # ML-based classification
    from morphology import MLStarGalaxyClassifier
    clf = MLStarGalaxyClassifier.load("model.joblib")
    results = clf.predict(features)
"""

from morphology.concentration import (
    compute_morphology,
    compute_morphology_batch,
    compute_morphology_batch_parallel,
    concentration_index,
    concentration_index_batch,
    concentration_index_batch_parallel,
    half_light_radius,
    half_light_radius_batch,
    half_light_radius_batch_parallel,
)
from morphology.ml_classifier import (
    ClassificationResult,
    ClassifierMetrics,
    MLStarGalaxyClassifier,
    extract_features_from_catalog,
    extract_features_from_photometry,
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

__all__ = [
    # ML star-galaxy classification
    "ClassificationResult",
    "ClassifierMetrics",
    "MLStarGalaxyClassifier",
    "SersicParams",
    # Classical star-galaxy classification
    "classify_star_galaxy",
    # Concentration and morphology
    "compute_morphology",
    "compute_morphology_batch",
    "compute_morphology_batch_parallel",
    "concentration_index",
    "concentration_index_batch",
    "concentration_index_batch_parallel",
    "half_light_radius",
    "half_light_radius_batch",
    "half_light_radius_batch_parallel",
    "extract_features_from_catalog",
    "extract_features_from_photometry",
    "fit_sersic_profile",
    "get_stellarity_index",
    # Sérsic fitting
    "measure_sersic_params",
]
