"""Morphology module for galaxy size and shape measurements.

Provides:
- Concentration index calculation
- Half-light radius measurements
- Zoobot deep learning morphology classification

Example usage:
    from morphology import concentration_index, half_light_radius

    # Calculate concentration index
    c = concentration_index(image, x, y)

    # Calculate half-light radius
    r = half_light_radius(image, x, y)

    # Train morphology classifier from Zoobot embeddings
    from morphology import ZoobotMorphologyClassifier, train_morphology_classifier
    clf = train_morphology_classifier(catalog, label_column="visual_type")
    catalog = clf.classify_catalog(catalog)
"""

from morphology.concentration import (
    compute_morphology,
    compute_morphology_batch,
    concentration_index,
    concentration_index_batch,
    half_light_radius,
    half_light_radius_batch,
)
from morphology.zoobot_classify import (
    ZOOBOT_AVAILABLE,
    ZoobotMorphologyClassifier,
    classify_by_sed_type,
    classify_morphology,
    extract_cutouts,
    extract_embeddings,
    train_morphology_classifier,
)
from morphology.star_classifier import classify_stars

__all__ = [
    "compute_morphology",
    "compute_morphology_batch",
    "concentration_index",
    "concentration_index_batch",
    "half_light_radius",
    "half_light_radius_batch",
    "classify_morphology",
    "extract_cutouts",
    "classify_stars",
    # Zoobot embedding classification
    "ZOOBOT_AVAILABLE",
    "ZoobotMorphologyClassifier",
    "train_morphology_classifier",
    "extract_embeddings",
    "classify_by_sed_type",
]
