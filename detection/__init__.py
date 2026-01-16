"""Source detection module.

Provides source detection backends:
- SEP (Source Extractor as Python library) - fast SExtractor-compatible detection
- photutils - Astropy-affiliated photometry library (used in run_analysis.py)

Usage
-----
>>> from detection import detect_and_measure, SEP_AVAILABLE
>>> if SEP_AVAILABLE:
...     catalog, background, segmap = detect_and_measure(image_data, gain=7.0)
"""

from detection.sep_detection import (
    SEP_AVAILABLE,
    STATMORPH_AVAILABLE,
    SEPBackground,
    SEPDetectionResult,
    check_sep_available,
    circular_aperture_photometry,
    compute_flux_radii,
    compute_half_light_radius,
    compute_statmorph,
    convert_to_photutils_format,
    detect_and_measure,
    detect_sources_sep,
    estimate_background,
    kron_photometry,
)

__all__ = [
    # Availability check
    "SEP_AVAILABLE",
    "STATMORPH_AVAILABLE",
    "check_sep_available",
    # Data classes
    "SEPBackground",
    "SEPDetectionResult",
    # Core functions
    "estimate_background",
    "detect_sources_sep",
    "detect_and_measure",
    # Photometry
    "kron_photometry",
    "circular_aperture_photometry",
    "compute_flux_radii",
    "compute_half_light_radius",
    # Morphology
    "compute_statmorph",
    # Utilities
    "convert_to_photutils_format",
]
