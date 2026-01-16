"""Photometric zeropoint calibration using spectroscopic redshifts.

This module implements the EAZY-style zeropoint offset calibration method
to improve photometric redshift accuracy by correcting systematic offsets
in the photometry.

The method:
1. Uses sources with spectroscopic redshifts as calibrators
2. For each band, computes the median offset between observed and
   template-predicted fluxes at the known redshift
3. Derives multiplicative zeropoint corrections
4. Can be iterated until convergence

References:
- Brammer, van Dokkum & Coppi 2008, ApJ, 686, 1503 (EAZY)
- Ilbert et al. 2006, A&A, 457, 841 (zeropoint calibration method)
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class ZeropointResult(NamedTuple):
    """Result of zeropoint calibration."""
    offsets_mag: NDArray  # Magnitude offsets per band (add to observed mags)
    offsets_flux: NDArray  # Flux multipliers per band (multiply observed fluxes)
    bands: tuple[str, ...]  # Band names
    n_calibrators: int  # Number of spec-z sources used
    residual_scatter: NDArray  # Per-band scatter after correction
    converged: bool  # Whether iteration converged


# Band names and central wavelengths for HDF WFPC2
BAND_NAMES = ("f300", "f450", "f606", "f814")
BAND_WAVELENGTHS = np.array([3000, 4500, 6060, 8140], dtype=np.float64)  # Angstroms


def compute_template_flux_at_redshift(
    z: float,
    template_wl: NDArray,
    template_spec: NDArray,
    filter_centers: NDArray,
    filter_widths: NDArray,
) -> NDArray:
    """Compute synthetic photometry for a template at given redshift.

    Parameters
    ----------
    z : float
        Redshift
    template_wl : NDArray
        Template rest-frame wavelength array (Angstroms)
    template_spec : NDArray
        Template spectrum (arbitrary flux units)
    filter_centers : NDArray
        Observer-frame filter central wavelengths
    filter_widths : NDArray
        Observer-frame filter half-widths

    Returns
    -------
    NDArray
        Synthetic flux in each band (shape: n_bands)
    """
    n_bands = len(filter_centers)
    syn_flux = np.zeros(n_bands, dtype=np.float64)

    # Redshift the template
    obs_wl = template_wl * (1 + z)

    for i in range(n_bands):
        # Find wavelength range in filter
        wl_lo = filter_centers[i] - filter_widths[i]
        wl_hi = filter_centers[i] + filter_widths[i]

        mask = (obs_wl >= wl_lo) & (obs_wl <= wl_hi)
        if mask.sum() > 0:
            syn_flux[i] = np.mean(template_spec[mask])

    return syn_flux


def derive_zeropoint_offsets(
    catalog: pd.DataFrame,
    flux_columns: tuple[str, ...] = ("flux_f300", "flux_f450", "flux_f606", "flux_f814"),
    z_spec_column: str = "z_spec",
    z_phot_column: str = "redshift",
    template_path: str | Path = "./spectra",
    sigma_clip: float = 3.0,
    min_calibrators: int = 10,
) -> ZeropointResult:
    """Derive zeropoint offsets using spectroscopic redshifts.

    This implements EAZY-style zeropoint calibration: for sources with known
    spec-z, compare the observed flux COLORS to template colors at that redshift
    to derive systematic per-band offsets.

    The key insight is that we need to normalize the template to the observed
    SED first (fit for overall scaling), then measure per-band residuals.

    Parameters
    ----------
    catalog : pd.DataFrame
        Catalog with flux columns, z_spec, and optionally z_phot
    flux_columns : tuple of str
        Column names for fluxes in each band
    z_spec_column : str
        Column name for spectroscopic redshift
    z_phot_column : str
        Column name for photometric redshift (for fallback template selection)
    template_path : str or Path
        Path to template spectra directory
    sigma_clip : float
        Sigma clipping threshold for outlier rejection
    min_calibrators : int
        Minimum number of calibrators required

    Returns
    -------
    ZeropointResult
        Zeropoint offsets and diagnostics
    """
    # Filter to sources with spec-z
    has_specz = catalog[z_spec_column].notna() & (catalog[z_spec_column] > 0)
    calibrators = catalog[has_specz].copy()

    n_cal = len(calibrators)
    n_bands = len(flux_columns)

    print(f"  Zeropoint calibration: {n_cal} spec-z calibrators")

    if n_cal < min_calibrators:
        print(f"  Warning: Too few calibrators ({n_cal} < {min_calibrators}), skipping")
        return ZeropointResult(
            offsets_mag=np.zeros(n_bands),
            offsets_flux=np.ones(n_bands),
            bands=BAND_NAMES,
            n_calibrators=n_cal,
            residual_scatter=np.full(n_bands, np.nan),
            converged=False,
        )

    # Load multiple templates to find best fit for each source
    template_path = Path(template_path)
    templates = {}
    template_types = ["elliptical", "Sb", "sbt1", "sbt3", "sbt5"]

    for ttype in template_types:
        tfile = template_path / f"{ttype}.dat"
        if tfile.exists():
            wl, spec = np.loadtxt(tfile, usecols=[0, 1], unpack=True)
            templates[ttype] = (wl, spec)

    if not templates:
        print(f"  Warning: No template files found in {template_path}")
        return ZeropointResult(
            offsets_mag=np.zeros(n_bands),
            offsets_flux=np.ones(n_bands),
            bands=BAND_NAMES,
            n_calibrators=n_cal,
            residual_scatter=np.full(n_bands, np.nan),
            converged=False,
        )

    # Filter definitions
    filter_centers = BAND_WAVELENGTHS
    filter_widths = np.array([1521, 1501, 951, 766], dtype=np.float64) / 2

    # Compute per-band residuals for each calibrator
    # Residual = log10(obs_flux / (A * template_flux)) where A is best-fit normalization
    log_residuals = np.full((n_cal, n_bands), np.nan)

    for i, (idx, row) in enumerate(calibrators.iterrows()):
        z = row[z_spec_column]

        # Get observed fluxes
        obs_flux = np.array([row[col] for col in flux_columns])
        valid_bands = (obs_flux > 0) & np.isfinite(obs_flux)

        if valid_bands.sum() < 3:
            continue

        # Find best-fitting template
        best_chi2 = np.inf
        best_residuals = None

        for ttype, (wl, spec) in templates.items():
            # Compute template flux at spec-z
            template_flux = compute_template_flux_at_redshift(
                z, wl, spec, filter_centers, filter_widths
            )

            if (template_flux[valid_bands] <= 0).any():
                continue

            # Fit for overall normalization: minimize sum((obs - A*template)^2 / obs^2)
            # Optimal A = sum(obs * template) / sum(template^2) for valid bands
            A = np.sum(obs_flux[valid_bands] * template_flux[valid_bands]) / \
                np.sum(template_flux[valid_bands]**2)

            if A <= 0:
                continue

            # Compute chi-squared
            residuals = (obs_flux[valid_bands] - A * template_flux[valid_bands]) / obs_flux[valid_bands]
            chi2 = np.sum(residuals**2)

            if chi2 < best_chi2:
                best_chi2 = chi2
                # Store log residuals: positive = observed too bright, need negative offset
                best_residuals = np.full(n_bands, np.nan)
                for b in range(n_bands):
                    if valid_bands[b] and template_flux[b] > 0:
                        best_residuals[b] = np.log10(obs_flux[b] / (A * template_flux[b]))

        if best_residuals is not None:
            log_residuals[i] = best_residuals

    # Compute median offset per band with sigma clipping
    offsets_mag = np.zeros(n_bands)
    residual_scatter = np.zeros(n_bands)

    for b in range(n_bands):
        resid = log_residuals[:, b]
        valid = np.isfinite(resid)

        if valid.sum() < 5:
            print(f"  Band {BAND_NAMES[b]}: insufficient valid calibrators ({valid.sum()})")
            continue

        # Iterative sigma clipping
        resid_clean = resid[valid]
        for _ in range(3):
            med = np.median(resid_clean)
            mad = 1.4826 * np.median(np.abs(resid_clean - med))
            if mad > 0:
                keep = np.abs(resid_clean - med) < sigma_clip * mad
                if keep.sum() >= 5:
                    resid_clean = resid_clean[keep]

        # Median log residual -> magnitude offset
        # If residual > 0, observed is too bright, need to reduce (negative mag offset to add)
        median_resid = np.median(resid_clean)
        offsets_mag[b] = -2.5 * median_resid  # Convert from log10 to magnitude
        residual_scatter[b] = 2.5 * 1.4826 * np.median(np.abs(resid_clean - median_resid))

        print(f"  Band {BAND_NAMES[b]}: offset = {offsets_mag[b]:+.3f} mag, "
              f"scatter = {residual_scatter[b]:.3f} mag ({valid.sum()} calibrators)")

    # Convert to flux multipliers
    offsets_flux = 10**(-0.4 * offsets_mag)

    return ZeropointResult(
        offsets_mag=offsets_mag,
        offsets_flux=offsets_flux,
        bands=BAND_NAMES,
        n_calibrators=n_cal,
        residual_scatter=residual_scatter,
        converged=True,
    )


def apply_zeropoint_corrections(
    catalog: pd.DataFrame,
    offsets: ZeropointResult,
    flux_columns: tuple[str, ...] = ("flux_f300", "flux_f450", "flux_f606", "flux_f814"),
    error_columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Apply zeropoint corrections to catalog fluxes.

    Parameters
    ----------
    catalog : pd.DataFrame
        Input catalog
    offsets : ZeropointResult
        Zeropoint offsets from derive_zeropoint_offsets()
    flux_columns : tuple of str
        Flux column names to correct
    error_columns : tuple of str, optional
        Error column names to scale accordingly

    Returns
    -------
    pd.DataFrame
        Catalog with corrected fluxes
    """
    result = catalog.copy()

    for i, (flux_col, scale) in enumerate(zip(flux_columns, offsets.offsets_flux)):
        if flux_col in result.columns:
            result[flux_col] = result[flux_col] * scale

            # Also scale errors if provided
            if error_columns is not None and i < len(error_columns):
                err_col = error_columns[i]
                if err_col in result.columns:
                    result[err_col] = result[err_col] * scale

    return result


def iterative_zeropoint_calibration(
    catalog: pd.DataFrame,
    classify_func,
    flux_columns: tuple[str, ...] = ("flux_f300", "flux_f450", "flux_f606", "flux_f814"),
    error_columns: tuple[str, ...] | None = None,
    z_spec_column: str = "z_spec",
    max_iterations: int = 3,
    convergence_threshold: float = 0.01,
    template_path: str | Path = "./spectra",
) -> tuple[pd.DataFrame, ZeropointResult, list[dict]]:
    """Iteratively calibrate zeropoints and re-run photo-z.

    This is the full EAZY-style iterative calibration:
    1. Run photo-z with current zeropoints
    2. Derive new zeropoint offsets using spec-z
    3. Apply corrections and repeat until convergence

    Parameters
    ----------
    catalog : pd.DataFrame
        Input catalog with fluxes and spec-z
    classify_func : callable
        Function to run photo-z classification. Should accept
        (flux_array, error_array, spectra_path, **kwargs) and return dict
    flux_columns : tuple of str
        Flux column names
    error_columns : tuple of str, optional
        Error column names
    z_spec_column : str
        Spec-z column name
    max_iterations : int
        Maximum calibration iterations
    convergence_threshold : float
        Convergence threshold for offset changes (magnitudes)
    template_path : str or Path
        Path to templates

    Returns
    -------
    tuple
        (corrected_catalog, final_offsets, iteration_history)
    """
    working_cat = catalog.copy()
    cumulative_offsets = np.ones(len(flux_columns))
    history = []

    print("\n  === Iterative Zeropoint Calibration ===")

    for iteration in range(max_iterations):
        print(f"\n  Iteration {iteration + 1}/{max_iterations}")

        # Step 1: Derive zeropoint offsets
        offsets = derive_zeropoint_offsets(
            working_cat,
            flux_columns=flux_columns,
            z_spec_column=z_spec_column,
            template_path=template_path,
        )

        if not offsets.converged:
            print("  Calibration failed, stopping iteration")
            break

        # Track cumulative offsets
        cumulative_offsets *= offsets.offsets_flux

        # Record history
        history.append({
            "iteration": iteration + 1,
            "offsets_mag": offsets.offsets_mag.copy(),
            "offsets_flux": offsets.offsets_flux.copy(),
            "residual_scatter": offsets.residual_scatter.copy(),
        })

        # Check convergence
        max_offset_change = np.max(np.abs(offsets.offsets_mag))
        print(f"  Max offset change: {max_offset_change:.4f} mag")

        if max_offset_change < convergence_threshold:
            print(f"  Converged after {iteration + 1} iterations")
            break

        # Step 2: Apply corrections
        working_cat = apply_zeropoint_corrections(
            working_cat, offsets, flux_columns, error_columns
        )

        # Step 3: Re-run photo-z (if classify_func provided)
        if classify_func is not None and iteration < max_iterations - 1:
            print("  Re-running photo-z with corrected fluxes...")
            flux_array = working_cat[list(flux_columns)].values

            if error_columns:
                error_array = working_cat[list(error_columns)].values
            else:
                # Use 10% errors as fallback
                error_array = 0.1 * np.abs(flux_array)

            # Run classification
            results = classify_func(flux_array, error_array, str(template_path))

            # Update redshift in working catalog
            working_cat["redshift"] = results["redshift"]

    # Create final offsets result with cumulative values
    final_offsets = ZeropointResult(
        offsets_mag=-2.5 * np.log10(cumulative_offsets),
        offsets_flux=cumulative_offsets,
        bands=BAND_NAMES,
        n_calibrators=offsets.n_calibrators,
        residual_scatter=offsets.residual_scatter,
        converged=True,
    )

    print(f"\n  Final cumulative zeropoint offsets:")
    for i, band in enumerate(BAND_NAMES):
        print(f"    {band}: {final_offsets.offsets_mag[i]:+.3f} mag "
              f"(flux × {final_offsets.offsets_flux[i]:.3f})")

    return working_cat, final_offsets, history


def save_zeropoint_file(
    offsets: ZeropointResult,
    output_path: str | Path,
    comment: str = "",
) -> None:
    """Save zeropoint offsets to a file for future use.

    Parameters
    ----------
    offsets : ZeropointResult
        Zeropoint offsets to save
    output_path : str or Path
        Output file path
    comment : str
        Optional comment to include in header
    """
    output_path = Path(output_path)

    with open(output_path, "w") as f:
        f.write("# Photometric zeropoint offsets\n")
        f.write(f"# N_calibrators: {offsets.n_calibrators}\n")
        if comment:
            f.write(f"# {comment}\n")
        f.write("# band  offset_mag  flux_scale  scatter\n")

        for i, band in enumerate(offsets.bands):
            f.write(f"{band}  {offsets.offsets_mag[i]:+.4f}  "
                    f"{offsets.offsets_flux[i]:.4f}  "
                    f"{offsets.residual_scatter[i]:.4f}\n")

    print(f"  Saved zeropoint offsets to {output_path}")


def load_zeropoint_file(filepath: str | Path) -> ZeropointResult:
    """Load zeropoint offsets from a file.

    Parameters
    ----------
    filepath : str or Path
        Path to zeropoint file

    Returns
    -------
    ZeropointResult
        Loaded offsets
    """
    filepath = Path(filepath)

    bands = []
    offsets_mag = []
    offsets_flux = []
    scatter = []
    n_calibrators = 0

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("# N_calibrators:"):
                n_calibrators = int(line.split(":")[1])
            elif line.startswith("#") or not line:
                continue
            else:
                parts = line.split()
                bands.append(parts[0])
                offsets_mag.append(float(parts[1]))
                offsets_flux.append(float(parts[2]))
                scatter.append(float(parts[3]))

    return ZeropointResult(
        offsets_mag=np.array(offsets_mag),
        offsets_flux=np.array(offsets_flux),
        bands=tuple(bands),
        n_calibrators=n_calibrators,
        residual_scatter=np.array(scatter),
        converged=True,
    )
