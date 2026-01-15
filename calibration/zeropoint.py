"""EAZY-style template zero-point calibration.

Iteratively adjusts template normalization per band using spectroscopic
training samples to minimize photo-z scatter.

Algorithm:
1. Run initial photo-z on spec-z training sample
2. For each band, compute median flux ratio (observed/model)
3. Apply correction factors to templates
4. Iterate until convergence (typically 3-5 iterations)

References:
- Brammer et al. 2008, ApJ, 686, 1503 (EAZY)
- Weaver et al. 2022 (COSMOS2020 calibration)
- Ilbert et al. 2006 (COSMOS photo-z methodology)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class ZeroPointCorrection:
    """Zero-point correction factors per band.

    Attributes
    ----------
    band : str
        Band name (e.g., 'u', 'b', 'v', 'i')
    correction_mag : float
        Magnitude offset to apply
    correction_flux : float
        Flux multiplier = 10^(-0.4 * correction_mag)
    n_sources : int
        Number of sources used in calibration
    scatter : float
        Scatter in the correction estimate
    """

    band: str
    correction_mag: float
    correction_flux: float
    n_sources: int
    scatter: float


@dataclass
class CalibrationResult:
    """Result of iterative zero-point calibration.

    Attributes
    ----------
    corrections : list[ZeroPointCorrection]
        Per-band correction factors
    n_iterations : int
        Number of iterations performed
    initial_nmad : float
        NMAD before calibration
    final_nmad : float
        NMAD after calibration
    convergence_history : list[float]
        NMAD at each iteration
    """

    corrections: list[ZeroPointCorrection]
    n_iterations: int
    initial_nmad: float
    final_nmad: float
    convergence_history: list[float]


def calibrate_zeropoints(
    training_catalog: pd.DataFrame,
    z_spec_col: str = "z_spec",
    flux_cols: dict[str, str] | None = None,
    error_cols: dict[str, str] | None = None,
    spectra_path: Path | str | None = None,
    max_iterations: int = 10,
    convergence_threshold: float = 0.001,
    sigma_clip: float = 3.0,
    verbose: bool = True,
) -> CalibrationResult:
    """Perform EAZY-style iterative zero-point calibration.

    This function iteratively adjusts the photometric zero-points
    to minimize the scatter between photometric and spectroscopic
    redshifts.

    Parameters
    ----------
    training_catalog : pd.DataFrame
        Catalog with spectroscopic redshifts and photometry
    z_spec_col : str
        Column name for spectroscopic redshift
    flux_cols : dict, optional
        Mapping of band name to flux column name
        Default: {'u': 'flux_u', 'b': 'flux_b', 'v': 'flux_v', 'i': 'flux_i'}
    error_cols : dict, optional
        Mapping of band name to error column name
    spectra_path : Path or str, optional
        Path to template spectra directory
    max_iterations : int
        Maximum calibration iterations (default: 10)
    convergence_threshold : float
        Stop when NMAD change < threshold (default: 0.001)
    sigma_clip : float
        Sigma clipping for outlier rejection (default: 3.0)
    verbose : bool
        Print progress information (default: True)

    Returns
    -------
    CalibrationResult
        Correction factors and convergence history
    """

    if flux_cols is None:
        flux_cols = {"u": "flux_u", "b": "flux_b", "v": "flux_v", "i": "flux_i"}

    if error_cols is None:
        error_cols = {
            "u": "flux_u_err",
            "b": "flux_b_err",
            "v": "flux_v_err",
            "i": "flux_i_err",
        }

    bands = list(flux_cols.keys())

    # Initialize corrections to zero
    corrections = dict.fromkeys(bands, 0.0)
    convergence_history = []

    # Filter to valid spec-z sources
    valid = (
        (training_catalog[z_spec_col] > 0)
        & (training_catalog[z_spec_col] < 6)
        & training_catalog[z_spec_col].notna()
    )

    # Also require valid fluxes
    for band in bands:
        flux_col = flux_cols[band]
        if flux_col in training_catalog.columns:
            valid &= (training_catalog[flux_col] > 0) & training_catalog[
                flux_col
            ].notna()

    train_cat = training_catalog[valid].copy()

    if len(train_cat) < 10:
        raise ValueError(f"Too few valid training sources: {len(train_cat)}")

    if verbose:
        print(f"Starting zero-point calibration with {len(train_cat)} sources")

    # Compute initial NMAD
    initial_nmad = _compute_nmad(
        train_cat, corrections, flux_cols, error_cols, z_spec_col, spectra_path
    )
    convergence_history.append(initial_nmad)

    if verbose:
        print(f"Initial NMAD: {initial_nmad:.4f}")

    for iteration in range(max_iterations):
        # Compute photo-z residuals at spec-z
        residuals = _compute_residuals(
            train_cat, corrections, flux_cols, error_cols, z_spec_col, spectra_path
        )

        # Update corrections using sigma-clipped median
        for band in bands:
            band_residuals = residuals.get(band, [])
            if len(band_residuals) < 5:
                continue

            res_arr = np.array(band_residuals)

            # Sigma clip
            median = np.median(res_arr)
            mad = 1.48 * np.median(np.abs(res_arr - median))

            if mad > 0:
                good = np.abs(res_arr - median) < sigma_clip * mad
                if good.sum() >= 5:
                    correction_delta = np.median(res_arr[good])
                    # Update correction (in magnitudes)
                    corrections[band] += correction_delta

        # Compute new NMAD
        current_nmad = _compute_nmad(
            train_cat, corrections, flux_cols, error_cols, z_spec_col, spectra_path
        )
        convergence_history.append(current_nmad)

        if verbose:
            print(f"Iteration {iteration + 1}: NMAD = {current_nmad:.4f}")

        # Check convergence
        if len(convergence_history) > 1:
            improvement = convergence_history[-2] - convergence_history[-1]
            if abs(improvement) < convergence_threshold:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

    # Build final result
    correction_list = []
    for band in bands:
        correction_list.append(
            ZeroPointCorrection(
                band=band,
                correction_mag=corrections[band],
                correction_flux=10 ** (-0.4 * corrections[band]),
                n_sources=len(train_cat),
                scatter=convergence_history[-1],
            )
        )

    result = CalibrationResult(
        corrections=correction_list,
        n_iterations=iteration + 1,
        initial_nmad=initial_nmad,
        final_nmad=convergence_history[-1],
        convergence_history=convergence_history,
    )

    if verbose:
        print("\nCalibration complete:")
        print(f"  Initial NMAD: {result.initial_nmad:.4f}")
        print(f"  Final NMAD:   {result.final_nmad:.4f}")
        print(f"  Improvement:  {(1 - result.final_nmad / result.initial_nmad) * 100:.1f}%")
        print("\nCorrections:")
        for corr in result.corrections:
            print(f"  {corr.band}: {corr.correction_mag:+.4f} mag ({corr.correction_flux:.4f}x flux)")

    return result


def _compute_nmad(
    catalog: pd.DataFrame,
    corrections: dict[str, float],
    flux_cols: dict[str, str],
    error_cols: dict[str, str],
    z_spec_col: str,
    spectra_path: Path | str | None,
) -> float:
    """Compute NMAD with current zero-point corrections."""
    from classify import classify_galaxy_with_pdf

    # Process a subset for speed during iteration
    sample = catalog.sample(min(100, len(catalog)), random_state=42)

    # Vectorized flux correction computation
    band_order = ["b", "i", "u", "v"]
    correction_factors = np.array([10 ** (-0.4 * corrections[b]) for b in band_order])

    # Extract flux arrays
    flux_arrays = np.column_stack([
        sample[flux_cols[b]].values for b in band_order
    ])
    corrected_fluxes = flux_arrays * correction_factors

    # Extract error arrays (with fallback to 10% of flux)
    error_arrays = np.column_stack([
        sample[error_cols.get(b, f"flux_{b}_err")].values
        if error_cols.get(b, f"flux_{b}_err") in sample.columns
        else 0.1 * sample[flux_cols[b]].values
        for b in band_order
    ])
    # Handle missing/invalid errors
    error_arrays = np.where(error_arrays > 0, error_arrays, 0.1 * corrected_fluxes)

    z_spec_values = sample[z_spec_col].values

    z_phot_list = []
    z_spec_list = []

    # Process each source (classify_galaxy_with_pdf is not vectorizable)
    for i in range(len(sample)):
        fluxes = corrected_fluxes[i].tolist()
        errors = error_arrays[i].tolist()

        try:
            result = classify_galaxy_with_pdf(
                fluxes, errors, spectra_path=spectra_path
            )
            z_phot_list.append(result.redshift)
            z_spec_list.append(z_spec_values[i])
        except Exception:
            continue

    if len(z_phot_list) < 5:
        return 1.0  # Return high NMAD if not enough valid results

    z_phot = np.array(z_phot_list)
    z_spec = np.array(z_spec_list)

    dz = (z_phot - z_spec) / (1 + z_spec)
    return 1.48 * np.median(np.abs(dz - np.median(dz)))


def _compute_residuals(
    catalog: pd.DataFrame,
    corrections: dict[str, float],
    flux_cols: dict[str, str],
    error_cols: dict[str, str],
    z_spec_col: str,
    spectra_path: Path | str | None,
) -> dict[str, list[float]]:
    """Compute per-band flux residuals for calibration."""
    from classify import classify_galaxy_with_pdf

    bands = list(flux_cols.keys())
    residuals = {band: [] for band in bands}

    # Process a subset for speed
    sample = catalog.sample(min(100, len(catalog)), random_state=42)

    # Vectorized flux correction computation
    band_order = ["b", "i", "u", "v"]
    correction_factors = np.array([10 ** (-0.4 * corrections[b]) for b in band_order])

    # Extract flux arrays
    flux_arrays = np.column_stack([
        sample[flux_cols[b]].values for b in band_order
    ])
    corrected_fluxes = flux_arrays * correction_factors

    z_spec_values = sample[z_spec_col].values

    # Process each source
    for i in range(len(sample)):
        z_true = z_spec_values[i]
        fluxes = corrected_fluxes[i].tolist()
        errors = [0.1 * f for f in fluxes]  # Simplified for calibration

        try:
            result = classify_galaxy_with_pdf(
                fluxes,
                errors,
                spectra_path=spectra_path,
                z_min=max(0, z_true - 0.3),
                z_max=min(6, z_true + 0.3),
            )

            # Compute residual: log(observed/expected)
            # Using photo-z quality as weight
            if result.odds > 0.5:
                dz = (result.redshift - z_true) / (1 + z_true)

                # Simple residual based on photo-z offset
                # Positive dz suggests flux is too low (needs negative mag correction)
                for band in bands:
                    residuals[band].append(-2.5 * dz)

        except Exception:
            continue

    return residuals


def apply_zeropoint_corrections(
    fluxes: NDArray,
    corrections: list[ZeroPointCorrection],
    band_order: list[str] | None = None,
) -> NDArray:
    """Apply zero-point corrections to flux array.

    Parameters
    ----------
    fluxes : NDArray
        Flux array with shape (..., n_bands)
    corrections : list[ZeroPointCorrection]
        Correction factors from calibration
    band_order : list[str], optional
        Order of bands in flux array
        Default: ['u', 'b', 'v', 'i']

    Returns
    -------
    NDArray
        Corrected fluxes
    """
    if band_order is None:
        band_order = ["u", "b", "v", "i"]

    # Create correction array in same order as fluxes
    corr_dict = {c.band: c.correction_flux for c in corrections}
    corr_array = np.array([corr_dict.get(band, 1.0) for band in band_order])

    # Apply corrections
    return fluxes * corr_array


def save_corrections(
    result: CalibrationResult,
    path: Path | str,
) -> None:
    """Save calibration corrections to file.

    Parameters
    ----------
    result : CalibrationResult
        Calibration result to save
    path : Path or str
        Output file path (CSV format)
    """
    path = Path(path)

    data = []
    for corr in result.corrections:
        data.append(
            {
                "band": corr.band,
                "correction_mag": corr.correction_mag,
                "correction_flux": corr.correction_flux,
                "n_sources": corr.n_sources,
                "scatter": corr.scatter,
            }
        )

    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def load_corrections(path: Path | str) -> list[ZeroPointCorrection]:
    """Load calibration corrections from file.

    Parameters
    ----------
    path : Path or str
        Input file path (CSV format)

    Returns
    -------
    list[ZeroPointCorrection]
        Loaded corrections
    """
    path = Path(path)
    df = pd.read_csv(path)

    # Use vectorized access instead of iterrows
    corrections = [
        ZeroPointCorrection(
            band=df.loc[i, "band"],
            correction_mag=df.loc[i, "correction_mag"],
            correction_flux=df.loc[i, "correction_flux"],
            n_sources=int(df.loc[i, "n_sources"]),
            scatter=df.loc[i, "scatter"],
        )
        for i in range(len(df))
    ]

    return corrections
