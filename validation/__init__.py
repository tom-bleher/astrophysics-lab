"""Validation module for photo-z and classification quality assessment.

Provides functions to:
- Compute photo-z quality metrics (NMAD, bias, outlier fraction)
- Validate against spectroscopic redshifts
- Create validation plots

Submodules:
- metrics: Photo-z quality metrics
- plots: Visualization tools
- muse_specz: MUSE spectroscopic validation
- vizier_query: VizieR query utilities
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
from .plots import (
    plot_binned_metrics,
    plot_delta_z_histogram,
    plot_photoz_vs_specz,
    plot_validation_panel,
)

# VizieR query utilities
from .vizier_query import (
    query_fernandez_soto,
    query_hdf_surface_photometry,
    query_hudf_vizier,
    query_multiple_catalogs,
    query_vizier_catalog,
)
from .wcs_utils import (
    add_sky_coordinates,
    get_wcs_from_header,
    pixel_to_sky,
)

# Zoobot morphology validation (optional - requires zoobot package)
try:
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
    HAS_ZOOBOT_MODULE = True
except ImportError:
    HAS_ZOOBOT_MODULE = False

__all__ = [
    'HAS_ZOOBOT_MODULE',
    'MorphologyPrediction',
    'ZoobotValidationReport',
    'add_sky_coordinates',
    'binned_metrics',
    'compare_photoz_methods',
    'cross_match_catalogs',
    # Zoobot morphology (optional)
    'cross_validate_sed_with_concentration',
    'extract_cutouts',
    'format_metrics_table',
    'get_morphology_from_concentration',
    # WCS utilities
    'get_wcs_from_header',
    'interpret_zoobot_predictions',
    # External catalog validation
    'load_fernandez_soto_catalog',
    # MUSE validation
    'load_muse_catalog',
    'magnitude_dependent_metrics',
    'odds_quality_metrics',
    # Photo-z metrics
    'photoz_metrics',
    'pixel_to_sky',
    'plot_binned_metrics',
    'plot_delta_z_histogram',
    # Validation plots
    'plot_photoz_vs_specz',
    'plot_validation_panel',
    'print_validation_summary',
    'query_fernandez_soto',
    'query_hdf_surface_photometry',
    'query_hudf_vizier',
    'query_multiple_catalogs',
    # VizieR queries
    'query_vizier_catalog',
    'run_zoobot_predictions',
    'validate_against_specz',
    'validate_morphology_with_zoobot',
]
