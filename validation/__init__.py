"""Validation module for photo-z and classification quality assessment.

Provides functions to:
- Compute photo-z quality metrics (NMAD, bias, outlier fraction)
- Validate against spectroscopic redshifts

Submodules:
- metrics: Photo-z quality metrics
- muse_specz: MUSE spectroscopic validation
- external_crossmatch: External catalog cross-matching
- zoobot_morphology: Zoobot morphology validation (optional)
"""

# Photo-z metrics and plotting
from .metrics import (
    binned_metrics,
    compare_photoz_methods,
    cross_match_catalogs,
    format_metrics_table,
    load_fernandez_soto_catalog,
    magnitude_dependent_metrics,
    odds_quality_metrics,
    photoz_metrics,
    validate_against_specz,
)

# MUSE spectroscopic validation
from .muse_specz import (
    load_muse_catalog,
    print_validation_summary,
)
from .zoobot_morphology import (
    MorphologyPrediction,
    ZoobotValidationReport,
    cross_validate_sed_with_concentration,
    extract_cutouts,
    get_morphology_from_concentration,
    interpret_zoobot_predictions,
    run_zoobot_predictions,
    validate_morphology_with_zoobot,
)

__all__ = [
    'MorphologyPrediction',
    'ZoobotValidationReport',
    'binned_metrics',
    'compare_photoz_methods',
    'cross_match_catalogs',
    'cross_validate_sed_with_concentration',
    'extract_cutouts',
    'format_metrics_table',
    'get_morphology_from_concentration',
    'interpret_zoobot_predictions',
    'load_fernandez_soto_catalog',
    'load_muse_catalog',
    'magnitude_dependent_metrics',
    'odds_quality_metrics',
    'photoz_metrics',
    'print_validation_summary',
    'run_zoobot_predictions',
    'validate_against_specz',
    'validate_morphology_with_zoobot',
]
