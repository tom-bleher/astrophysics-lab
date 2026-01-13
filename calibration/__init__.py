"""Calibration module for photometric zero-point corrections.

This module provides tools for:
- EAZY-style iterative template zero-point calibration
- Flux scale corrections using spectroscopic samples

Example usage:
    from calibration import calibrate_zeropoints, apply_zeropoint_corrections

    # Calibrate using spec-z training sample
    result = calibrate_zeropoints(training_catalog, z_spec_col='z_spec')
    print(f"Final NMAD: {result.final_nmad:.4f}")

    # Apply corrections to new data
    corrected_fluxes = apply_zeropoint_corrections(fluxes, result.corrections)
"""

from calibration.zeropoint import (
    CalibrationResult,
    ZeroPointCorrection,
    apply_zeropoint_corrections,
    calibrate_zeropoints,
)

__all__ = [
    "calibrate_zeropoints",
    "apply_zeropoint_corrections",
    "ZeroPointCorrection",
    "CalibrationResult",
]
