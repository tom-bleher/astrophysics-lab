"""Validation module for photo-z quality assessment.

This module provides functions for:
- Cross-matching with external spectroscopic catalogs
- Computing photo-z validation metrics (NMAD, outlier rate, bias)
- Applying spectroscopic redshifts to improve catalog quality
- Generating validation plots
"""

from .specz_validation import (
    FLAG_CATASTROPHIC_PHOTOZ,
    apply_spectroscopic_redshifts,
    compute_photoz_metrics,
    cross_match_with_specz,
    flag_catastrophic_outliers,
    generate_validation_plots,
    load_specz_catalogs,
    run_full_validation,
)

__all__ = [
    "FLAG_CATASTROPHIC_PHOTOZ",
    "apply_spectroscopic_redshifts",
    "compute_photoz_metrics",
    "cross_match_with_specz",
    "flag_catastrophic_outliers",
    "generate_validation_plots",
    "load_specz_catalogs",
    "run_full_validation",
]
