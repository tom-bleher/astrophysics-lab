"""Validation module for comparing detections against external catalogs.

Provides functions to:
- Load external HDF/HUDF catalogs from VizieR and other sources
- Cross-match catalogs by sky coordinates
- Validate photo-z against reference catalogs (including spectroscopic z)
- Validate size measurements against surface photometry catalogs
- Compute photo-z quality metrics (NMAD, bias, outlier fraction)
- Create validation plots

Submodules:
- external_catalogs: Load catalogs from various sources
- metrics: Photo-z quality metrics
- plots: Visualization tools
- muse_specz: MUSE spectroscopic validation
- vizier_query: VizieR query utilities
"""

from .external_catalogs import (
    # HUDF catalogs
    load_vizier_hudf,
    load_fernandez_soto,
    download_hlf_catalog,
    load_hlf_catalog,
    # HDF-specific VizieR catalogs
    load_vizier_hdf_photoz,
    load_vizier_hdf_surface_photometry,
    load_vizier_hdf_ugrk,
    # Hawaii H-HDF-N
    load_hawaii_hdfn,
    # Cross-matching and validation
    cross_match_catalogs,
    validate_photoz,
    validate_with_specz,
    validate_sizes,
    plot_photoz_comparison,
    # Helper functions
    get_hdf_specz_from_fernandez_soto,
    # Report classes
    ValidationReport,
    SizeValidationReport,
    # Constants
    HDF_N_CENTER_RA,
    HDF_N_CENTER_DEC,
    HDF_S_CENTER_RA,
    HDF_S_CENTER_DEC,
)

from .wcs_utils import (
    get_wcs_from_header,
    pixel_to_sky,
    add_sky_coordinates,
)

# Photo-z metrics and plotting
from .metrics import (
    photoz_metrics,
    compare_photoz_methods,
    binned_metrics,
    magnitude_dependent_metrics,
    odds_quality_metrics,
    format_metrics_table,
)

from .plots import (
    plot_photoz_vs_specz,
    plot_delta_z_histogram,
    plot_validation_panel,
    plot_binned_metrics,
)

# VizieR query utilities
from .vizier_query import (
    query_vizier_catalog,
    query_hudf_vizier,
    query_fernandez_soto,
    query_hdf_surface_photometry,
    query_multiple_catalogs,
)

# MUSE spectroscopic validation
from .muse_specz import (
    load_muse_catalog,
    print_validation_summary,
)

# Zoobot morphology validation (optional - requires zoobot package)
try:
    from .zoobot_morphology import (
        extract_cutouts,
        run_zoobot_predictions,
        interpret_zoobot_predictions,
        validate_morphology_with_zoobot,
        get_morphology_from_concentration,
        cross_validate_sed_with_concentration,
        MorphologyPrediction,
        ZoobotValidationReport,
    )
    HAS_ZOOBOT_MODULE = True
except ImportError:
    HAS_ZOOBOT_MODULE = False

__all__ = [
    # HUDF catalogs
    'load_vizier_hudf',
    'load_fernandez_soto',
    'download_hlf_catalog',
    'load_hlf_catalog',
    # HDF-specific VizieR catalogs
    'load_vizier_hdf_photoz',
    'load_vizier_hdf_surface_photometry',
    'load_vizier_hdf_ugrk',
    # Hawaii H-HDF-N
    'load_hawaii_hdfn',
    # Cross-matching and validation
    'cross_match_catalogs',
    'validate_photoz',
    'validate_with_specz',
    'validate_sizes',
    'plot_photoz_comparison',
    # Helper functions
    'get_hdf_specz_from_fernandez_soto',
    # Report classes
    'ValidationReport',
    'SizeValidationReport',
    # Constants
    'HDF_N_CENTER_RA',
    'HDF_N_CENTER_DEC',
    'HDF_S_CENTER_RA',
    'HDF_S_CENTER_DEC',
    # WCS utilities
    'get_wcs_from_header',
    'pixel_to_sky',
    'add_sky_coordinates',
    # Photo-z metrics
    'photoz_metrics',
    'compare_photoz_methods',
    'binned_metrics',
    'magnitude_dependent_metrics',
    'odds_quality_metrics',
    'format_metrics_table',
    # Validation plots
    'plot_photoz_vs_specz',
    'plot_delta_z_histogram',
    'plot_validation_panel',
    'plot_binned_metrics',
    # VizieR queries
    'query_vizier_catalog',
    'query_hudf_vizier',
    'query_fernandez_soto',
    'query_hdf_surface_photometry',
    'query_multiple_catalogs',
    # MUSE validation
    'load_muse_catalog',
    'print_validation_summary',
    # Zoobot morphology (optional)
    'extract_cutouts',
    'run_zoobot_predictions',
    'interpret_zoobot_predictions',
    'validate_morphology_with_zoobot',
    'get_morphology_from_concentration',
    'cross_validate_sed_with_concentration',
    'MorphologyPrediction',
    'ZoobotValidationReport',
    'HAS_ZOOBOT_MODULE',
]
