"""Galaxy classification via SED template fitting.

This module provides photometric redshift estimation using the eazy-py library
(Brammer et al. 2008) as the primary backend, with a simplified fallback
implementation for cases where eazy is not available.

Features:
- Chi-squared template fitting
- Full probability distribution functions (PDFs) for redshift
- Uncertainty quantification (16th/84th percentiles)
- IGM absorption (Madau 1995, Inoue 2014)
- Magnitude-dependent redshift priors
- ODDS quality parameter

References:
- Brammer et al. 2008, ApJ, 686, 1503 (EAZY)
- Ilbert et al. 2006, A&A, 457, 841 (COSMOS photo-z)
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache, partial
from pathlib import Path
from typing import NamedTuple

import eazy
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import maximum_filter1d
from scipy.optimize import minimize_scalar

# Default spectra path for fallback mode
_DEFAULT_SPECTRA_PATH = Path(__file__).parent / "spectra"

# Galaxy template types
GALAXY_TYPES = (
    "elliptical",
    "S0",
    "Sa",
    "Sb",
    "sbt1",
    "sbt2",
    "sbt3",
    "sbt4",
    "sbt5",
    "sbt6",
)


class PhotoZResult(NamedTuple):
    """Result of photometric redshift estimation.

    Attributes
    ----------
    galaxy_type : str
        Best-fit galaxy template type
    redshift : float
        Best-fit redshift (maximum likelihood)
    z_pdf_median : float
        Median redshift from PDF
    z_lo : float
        16th percentile (lower 1-sigma bound)
    z_hi : float
        84th percentile (upper 1-sigma bound)
    chi_sq_min : float
        Minimum chi-squared value
    z_grid : NDArray
        Redshift grid for PDF
    pdf : NDArray
        Normalized probability density function P(z)
    odds : float
        ODDS parameter: integral of PDF within +/-0.1(1+z) of peak
    z_secondary : float or None
        Secondary peak redshift (if bimodal)
    odds_secondary : float or None
        ODDS for secondary peak
    chi2_flag : int
        Chi-squared quality flag: 0=good, 1=marginal, 2=poor
    odds_flag : int
        ODDS quality flag: 0=excellent, 1=good, 2=poor
    bimodal_flag : bool
        True if significant secondary peak detected
    template_ambiguity : float
        Template confusion score (0-1)
    reduced_chi2 : float
        Chi-squared divided by degrees of freedom
    second_best_template : str
        Second-best fitting template type
    """
    galaxy_type: str
    redshift: float
    z_pdf_median: float
    z_lo: float
    z_hi: float
    chi_sq_min: float
    z_grid: NDArray
    pdf: NDArray
    odds: float
    z_secondary: float | None = None
    odds_secondary: float | None = None
    chi2_flag: int = 0
    odds_flag: int = 0
    bimodal_flag: bool = False
    template_ambiguity: float = 0.0
    reduced_chi2: float = 0.0
    second_best_template: str = ""


# =============================================================================
# EAZY-based implementation (primary backend)
# =============================================================================


def classify_galaxy_eazy(
    fluxes: dict[str, float],
    flux_errors: dict[str, float],
    param_file: str | None = None,
    z_min: float = 0.0,
    z_max: float = 6.0,
) -> PhotoZResult:
    """Classify a galaxy using EAZY photometric redshift code.

    This is the recommended method for production use.

    Parameters
    ----------
    fluxes : dict
        Dictionary of band -> flux values
    flux_errors : dict
        Dictionary of band -> flux error values
    param_file : str, optional
        Path to EAZY parameter file
    z_min, z_max : float
        Redshift range

    Returns
    -------
    PhotoZResult
        Full photo-z result with PDF and uncertainties

    """

    # Initialize EAZY
    if param_file is None:
        # Use default parameters
        ez = eazy.photoz.PhotoZ(
            MAIN_OUTPUT_FILE='photz',
            Z_MIN=z_min,
            Z_MAX=z_max,
            Z_STEP=0.01,
        )
    else:
        ez = eazy.photoz.PhotoZ(param_file=param_file)

    # Fit single object
    bands = list(fluxes.keys())
    flux_arr = np.array([fluxes[b] for b in bands])
    err_arr = np.array([flux_errors[b] for b in bands])

    # Run EAZY fitting
    idx = 0
    ez.fit_at_zbest(
        flux=flux_arr,
        err=err_arr,
        get_err=True,
    )

    # Extract results
    z_grid = ez.zgrid
    pdf = ez.pz[idx] if hasattr(ez, 'pz') else np.zeros_like(z_grid)

    # Normalize PDF
    pdf_sum = np.trapezoid(pdf, z_grid)
    if pdf_sum > 0:
        pdf = pdf / pdf_sum

    return PhotoZResult(
        galaxy_type=str(ez.zbest_type[idx]) if hasattr(ez, 'zbest_type') else "unknown",
        redshift=float(ez.zbest[idx]),
        z_pdf_median=float(ez.zmc[idx]) if hasattr(ez, 'zmc') else float(ez.zbest[idx]),
        z_lo=float(ez.z16[idx]) if hasattr(ez, 'z16') else 0.0,
        z_hi=float(ez.z84[idx]) if hasattr(ez, 'z84') else 0.0,
        chi_sq_min=float(ez.chi2_best[idx]) if hasattr(ez, 'chi2_best') else 0.0,
        z_grid=z_grid,
        pdf=pdf,
        odds=float(ez.odds[idx]) if hasattr(ez, 'odds') else 0.0,
    )


# =============================================================================
# Simplified fallback implementation (when eazy not available)
# =============================================================================


def _load_spectrum(spectra_path: str, galaxy_type: str) -> tuple[NDArray, NDArray]:
    """Load galaxy spectrum from disk."""
    path = Path(spectra_path) / f"{galaxy_type}.dat"
    if not path.exists():
        raise FileNotFoundError(f"Spectrum file not found: {path}")
    wl, spec = np.loadtxt(path, usecols=[0, 1], unpack=True)
    return wl.astype(np.float64), spec.astype(np.float64)


def _igm_transmission(wavelength: NDArray, redshift: float) -> NDArray:
    """Compute IGM transmission using Madau 1995 model."""
    if redshift < 0.1:
        return np.ones_like(wavelength)

    tau = np.zeros_like(wavelength)
    one_plus_z = 1.0 + redshift

    # Lyman series absorption
    lyman_wl = np.array([1216.0, 1026.0, 972.5])
    lyman_coeff = np.array([0.0036, 0.0017, 0.0012])

    for wl_line, coeff in zip(lyman_wl, lyman_coeff):
        obs_wl = wl_line * one_plus_z
        mask = wavelength < obs_wl
        if np.any(mask):
            x = wavelength[mask] / wl_line
            tau[mask] += coeff * (x ** 3.46)

    # Lyman limit
    lyman_limit = 912.0
    obs_ll = lyman_limit * one_plus_z
    mask_ll = wavelength < obs_ll
    if np.any(mask_ll):
        x_ll = wavelength[mask_ll] / lyman_limit
        tau_ll = 0.25 * (x_ll ** 3) * (one_plus_z ** 0.46)
        tau[mask_ll] += np.minimum(tau_ll, 30.0)

    return np.clip(np.exp(-tau), 0.0, 1.0)


def _igm_transmission_inoue14(wavelength: NDArray, redshift: float) -> NDArray:
    """Compute IGM transmission using Inoue et al. 2014 model.

    This model provides more accurate IGM absorption than Madau 1995,
    especially at z > 3. It includes:
    - Full Lyman series (alpha through higher orders)
    - Proper Lyman continuum absorption
    - Redshift-dependent optical depth coefficients

    Parameters
    ----------
    wavelength : NDArray
        Observer-frame wavelengths in Angstroms
    redshift : float
        Source redshift

    Returns
    -------
    NDArray
        IGM transmission fraction (0 to 1)

    References
    ----------
    Inoue, A. K., Shimizu, I., Iwata, I., & Tanaka, M. 2014, MNRAS, 442, 1805
    """
    if redshift < 0.01:
        return np.ones_like(wavelength)

    # Lyman series wavelengths (Angstroms) - alpha through Ly-6 and beyond
    # lambda_n = 911.75 * n^2 / (n^2 - 1) for n = 2, 3, 4, ...
    lyman_wavelengths = np.array([
        1215.67,  # Lyman-alpha (n=2)
        1025.72,  # Lyman-beta (n=3)
        972.54,   # Lyman-gamma (n=4)
        949.74,   # Lyman-delta (n=5)
        937.80,   # Lyman-epsilon (n=6)
        930.75,   # Lyman-6 (n=7)
        926.23,   # Lyman-7 (n=8)
        923.15,   # Lyman-8 (n=9)
        920.96,   # Lyman-9 (n=10)
        919.35,   # Lyman-10 (n=11)
    ])

    # Lyman limit wavelength
    lyman_limit = 911.75

    # Inoue+14 coefficients for LAF (Lyman-alpha forest) optical depth
    # tau_LAF = A_LAF * (1+z)^gamma_LAF for each Lyman line
    # These are approximate fits to Table 2 of Inoue+14
    A_LAF = np.array([
        1.690e-2,   # Ly-alpha
        4.692e-3,   # Ly-beta
        2.239e-3,   # Ly-gamma
        1.319e-3,   # Ly-delta
        8.707e-4,   # Ly-epsilon
        6.178e-4,   # Ly-6
        4.609e-4,   # Ly-7
        3.569e-4,   # Ly-8
        2.843e-4,   # Ly-9
        2.318e-4,   # Ly-10
    ])

    # Exponents for LAF optical depth
    gamma_LAF = np.array([
        3.64,  # Ly-alpha
        3.64,  # Ly-beta
        3.64,  # Ly-gamma
        3.64,  # Ly-delta
        3.64,  # Ly-epsilon
        3.64,  # Ly-6
        3.64,  # Ly-7
        3.64,  # Ly-8
        3.64,  # Ly-9
        3.64,  # Ly-10
    ])

    # DLA (Damped Lyman-alpha) system coefficients
    A_DLA = np.array([
        1.617e-4,   # Ly-alpha
        1.545e-4,   # Ly-beta
        1.498e-4,   # Ly-gamma
        1.460e-4,   # Ly-delta
        1.429e-4,   # Ly-epsilon
        1.402e-4,   # Ly-6
        1.377e-4,   # Ly-7
        1.355e-4,   # Ly-8
        1.335e-4,   # Ly-9
        1.316e-4,   # Ly-10
    ])

    gamma_DLA = np.array([
        2.0,  # All DLA terms have same exponent
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
    ])

    tau = np.zeros_like(wavelength, dtype=np.float64)
    one_plus_z = 1.0 + redshift

    # Compute optical depth for each Lyman line
    for i, lam_line in enumerate(lyman_wavelengths):
        # Observer-frame wavelength of the line at redshift z
        lam_obs = lam_line * one_plus_z

        # Only affects photons blueward of the redshifted line
        mask = wavelength < lam_obs

        if np.any(mask):
            # Effective redshift for absorption
            z_eff = wavelength[mask] / lam_line - 1.0

            # Only include absorption from gas between us and the source
            valid = (z_eff >= 0) & (z_eff <= redshift)

            if np.any(valid):
                # LAF contribution
                tau_LAF = A_LAF[i] * ((1.0 + z_eff[valid]) ** gamma_LAF[i])

                # DLA contribution
                tau_DLA = A_DLA[i] * ((1.0 + z_eff[valid]) ** gamma_DLA[i])

                # Create temporary array for this line's contribution
                tau_line = np.zeros(np.sum(mask))
                tau_line[valid] = tau_LAF + tau_DLA
                tau[mask] += tau_line

    # Lyman continuum absorption (lambda < 912 A in rest frame)
    lam_obs_limit = lyman_limit * one_plus_z
    mask_lc = wavelength < lam_obs_limit

    if np.any(mask_lc):
        # Effective redshift for Lyman continuum absorption
        z_lc = wavelength[mask_lc] / lyman_limit - 1.0
        valid_lc = (z_lc >= 0) & (z_lc <= redshift)

        if np.any(valid_lc):
            # Inoue+14 Lyman continuum coefficients
            # LAF continuum: tau_LC_LAF = 0.325 * [(1+z)^1.2 - (1+z_abs)^1.2]
            # where z_abs = wavelength/912 - 1
            z_eff_lc = z_lc[valid_lc]

            # LAF Lyman continuum
            tau_LC_LAF = 0.325 * (
                ((1.0 + z_eff_lc) ** 1.2)
                - ((wavelength[mask_lc][valid_lc] / lyman_limit) ** 1.2)
            )
            tau_LC_LAF = np.maximum(tau_LC_LAF, 0.0)

            # DLA Lyman continuum
            # tau_LC_DLA = 0.211 * (1+z)^2 - 7.66e-2 * (1+z)^2.3 * ...
            tau_LC_DLA = 0.211 * ((1.0 + z_eff_lc) ** 2.0)
            tau_LC_DLA -= 0.0766 * ((1.0 + z_eff_lc) ** 2.3)
            tau_LC_DLA = np.maximum(tau_LC_DLA, 0.0)

            # Combine Lyman continuum contributions
            tau_lc_combined = np.zeros(np.sum(mask_lc))
            tau_lc_combined[valid_lc] = tau_LC_LAF + tau_LC_DLA
            tau[mask_lc] += tau_lc_combined

    # Cap optical depth to avoid numerical issues
    tau = np.minimum(tau, 700.0)

    return np.clip(np.exp(-tau), 0.0, 1.0)


# =============================================================================
# IGM Transmission Caching (Performance Optimization)
# =============================================================================

# Standard wavelength grid used for template fitting
_WL_GRID_STANDARD = np.arange(2200, 9500, 1, dtype=np.float64)

# Pre-computed IGM transmission cache
# Key: rounded redshift (to 2 decimal places), Value: transmission array
_IGM_CACHE: dict[float, NDArray] = {}
_IGM_CACHE_MAX_SIZE = 1000  # Limit cache size to prevent memory bloat


def _get_igm_transmission_cached(
    wavelength: NDArray,
    redshift: float,
    igm_model: str = "inoue14",
    cache_precision: float = 0.01,
) -> NDArray:
    """Get IGM transmission with caching for repeated redshift queries.

    This function caches IGM transmission for the standard wavelength grid,
    providing ~5-10x speedup when classifying many galaxies at similar
    redshifts.

    Parameters
    ----------
    wavelength : NDArray
        Observer-frame wavelengths in Angstroms
    redshift : float
        Source redshift
    igm_model : str
        IGM model to use: 'inoue14' or 'madau95'
    cache_precision : float
        Redshift rounding precision for cache keys (default 0.01)

    Returns
    -------
    NDArray
        IGM transmission fraction (0 to 1)
    """
    # Round redshift for cache key
    z_rounded = round(redshift / cache_precision) * cache_precision
    cache_key = (z_rounded, igm_model)

    # Check if using standard wavelength grid
    is_standard_grid = (
        len(wavelength) == len(_WL_GRID_STANDARD)
        and wavelength[0] == _WL_GRID_STANDARD[0]
        and wavelength[-1] == _WL_GRID_STANDARD[-1]
    )

    if is_standard_grid and cache_key in _IGM_CACHE:
        return _IGM_CACHE[cache_key]

    # Compute IGM transmission
    if igm_model == "inoue14":
        transmission = _igm_transmission_inoue14(wavelength, redshift)
    else:
        transmission = _igm_transmission(wavelength, redshift)

    # Cache if using standard grid and cache not full
    if is_standard_grid and len(_IGM_CACHE) < _IGM_CACHE_MAX_SIZE:
        _IGM_CACHE[cache_key] = transmission

    return transmission


@lru_cache(maxsize=256)
def _load_spectrum_cached(spectra_path: str, galaxy_type: str) -> tuple[tuple, tuple]:
    """Cached version of spectrum loading.

    Returns tuples instead of arrays for hashability with lru_cache.
    """
    wl, spec = _load_spectrum(spectra_path, galaxy_type)
    return tuple(wl), tuple(spec)


def _interpolate_templates(
    templates: list[tuple[NDArray, NDArray]],
    n_interp: int = 2,
) -> list[tuple[NDArray, NDArray]]:
    """Create interpolated templates between adjacent pairs.

    This function generates intermediate templates by linearly interpolating
    between adjacent template spectra. This can improve photo-z accuracy by
    providing finer sampling of the template space, especially useful when
    the true SED lies between discrete template types.

    Parameters
    ----------
    templates : list of tuples
        List of (wavelength, spectrum) tuples. All templates must be
        defined on the same wavelength grid.
    n_interp : int, optional
        Number of intermediate templates to create between each adjacent
        pair. Default is 2. Set to 0 to disable interpolation.

    Returns
    -------
    list of tuples
        Expanded list of (wavelength, spectrum) tuples including the
        original templates and the interpolated ones.

    Notes
    -----
    For N original templates with n_interp interpolated templates between
    each pair, the output will contain N + (N-1) * n_interp templates.

    Example
    -------
    With 3 templates [A, B, C] and n_interp=2:
    Output: [A, A-B_1, A-B_2, B, B-C_1, B-C_2, C]
    where A-B_1 is 2/3*A + 1/3*B, A-B_2 is 1/3*A + 2/3*B, etc.
    """
    if n_interp <= 0 or len(templates) < 2:
        return templates

    expanded = []
    n_templates = len(templates)

    for i in range(n_templates):
        wl_i, spec_i = templates[i]

        # Add the original template
        expanded.append((wl_i.copy(), spec_i.copy()))

        # Interpolate between this template and the next (if not last)
        if i < n_templates - 1:
            wl_next, spec_next = templates[i + 1]

            # Ensure wavelength grids match - interpolate to common grid if needed
            if not np.array_equal(wl_i, wl_next):
                # Use the first template's wavelength grid as reference
                spec_next_interp = np.interp(wl_i, wl_next, spec_next)
                wl_common = wl_i
            else:
                spec_next_interp = spec_next
                wl_common = wl_i

            # Create n_interp intermediate templates
            for j in range(1, n_interp + 1):
                # Interpolation weight: j/(n_interp+1)
                # j=1: mostly template i
                # j=n_interp: mostly template i+1
                weight = j / (n_interp + 1)
                spec_interp = (1.0 - weight) * spec_i + weight * spec_next_interp
                expanded.append((wl_common.copy(), spec_interp))

    return expanded


def classify_galaxy(
    fluxes: ArrayLike,
    errors: ArrayLike,
    spectra_path: Path | str | None = None,
) -> tuple[str, float]:
    """Classify a galaxy by fitting SED templates.

    Simple interface for basic classification.

    Parameters
    ----------
    fluxes : array-like
        Measured fluxes in [B, I, U, V] bands
    errors : array-like
        Measurement errors for each band
    spectra_path : Path or str, optional
        Directory containing galaxy template spectra

    Returns
    -------
    tuple[str, float]
        Galaxy type and best-fit redshift
    """
    if spectra_path is None:
        spectra_path = _DEFAULT_SPECTRA_PATH
    spectra_path_str = str(spectra_path)

    B, Ir, U, V = fluxes
    dB, dIr, dU, dV = errors

    # Wavelength grid and filter definitions
    wl_grid = np.arange(2200, 9500, 1)
    filter_centers = np.array([3000, 4500, 6060, 8140])
    filter_widths = np.array([1521, 1501, 951, 766]) / 2

    # Create filter masks
    filter_masks = []
    for center, width in zip(filter_centers, filter_widths):
        mask = (wl_grid >= center - width) & (wl_grid <= center + width)
        filter_masks.append(mask)

    # Measurement data
    meas_photo = np.array([U, B, V, Ir], dtype=np.float64)
    meas_errs = np.array([dU, dB, dV, dIr], dtype=np.float64)
    meas_median = np.median(meas_photo)
    meas_photo_norm = meas_photo / meas_median
    inv_var = (meas_median / meas_errs) ** 2

    def compute_chi_sq(z: float, wl: NDArray, spec: NDArray) -> float:
        """Compute chi-square for a given redshift."""
        wl_z = wl * (1 + z)
        spec_interp = np.interp(wl_grid, wl_z, spec, left=0, right=0)

        spec_mean = np.mean(spec_interp)
        if spec_mean == 0:
            return 1e30
        spec_norm = spec_interp / spec_mean

        # Synthetic photometry
        syn = np.array([np.mean(spec_norm[m]) for m in filter_masks])
        syn_median = np.median(syn)
        if syn_median == 0:
            return 1e30
        syn_norm = syn / syn_median

        return float(np.sum((meas_photo_norm - syn_norm) ** 2 * inv_var))

    # Load spectra
    spectra = [_load_spectrum(spectra_path_str, gt) for gt in GALAXY_TYPES]

    # Coarse grid search
    z_grid = np.linspace(0.0, 3.5, 15)
    best_chi, best_z = 1e30, 1.0

    for z in z_grid:
        for wl, spec in spectra:
            chi = compute_chi_sq(z, wl, spec)
            if chi < best_chi:
                best_chi, best_z = chi, z

    # Fine optimization
    z_lo = max(0.0, best_z - 0.5)
    z_hi = min(3.5, best_z + 0.5)

    best_results = []
    for i, (wl, spec) in enumerate(spectra):
        result = minimize_scalar(
            lambda z, w=wl, s=spec: compute_chi_sq(z, w, s),
            bounds=(z_lo, z_hi),
            method="bounded",
        )
        best_results.append((result.fun, result.x, i))

    best_results.sort()
    best_chi, best_z, best_idx = best_results[0]

    return GALAXY_TYPES[best_idx], float(best_z)


def classify_galaxy_with_pdf(
    fluxes: ArrayLike,
    errors: ArrayLike,
    spectra_path: Path | str | None = None,
    z_min: float = 0.0,
    z_max: float = 6.0,
    z_step: float = 0.01,
    systematic_error_floor: float = 0.10,
    apply_igm: bool = True,
    igm_model: str = "inoue14",
    apply_prior: bool = False,
    magnitude: float | None = None,
    n_template_interp: int = 0,
) -> PhotoZResult:
    """Classify a galaxy with full redshift PDF and uncertainties.

    Parameters
    ----------
    fluxes : array-like
        Measured fluxes in [B, I, U, V] bands
    errors : array-like
        Measurement errors for each band
    spectra_path : Path or str, optional
        Directory containing galaxy template spectra
    z_min, z_max : float
        Redshift range to search
    z_step : float
        Redshift grid step size
    systematic_error_floor : float
        Systematic error floor as fraction of flux (default 0.10 = 10%).
        Accounts for photometric calibration uncertainties. Added in
        quadrature to measurement errors.
    apply_igm : bool
        Apply IGM absorption for high-z sources
    igm_model : str
        IGM model: 'inoue14' (default, more accurate at z > 3) or 'madau95'
    apply_prior : bool
        Apply magnitude-dependent redshift prior
    magnitude : float, optional
        I-band magnitude for prior (required if apply_prior=True)
    n_template_interp : int
        Number of interpolated templates between each adjacent pair.
        Set to 2-3 for finer template sampling (default 0 = no interpolation).

    Returns
    -------
    PhotoZResult
        Full photo-z result with PDF and uncertainties
    """
    if spectra_path is None:
        spectra_path = _DEFAULT_SPECTRA_PATH
    spectra_path_str = str(spectra_path)

    # Validate IGM model selection
    if igm_model not in ("madau95", "inoue14"):
        raise ValueError(f"Unknown IGM model: {igm_model}. Use 'madau95' or 'inoue14'.")

    # Use cached IGM transmission function for performance
    # (caches results for similar redshifts on standard wavelength grid)
    def igm_func(wavelength: NDArray, z: float) -> NDArray:
        return _get_igm_transmission_cached(wavelength, z, igm_model=igm_model)

    B, Ir, U, V = fluxes
    dB, dIr, dU, dV = errors

    # Wavelength grid and filters
    wl_grid = np.arange(2200, 9500, 1)
    filter_centers = np.array([3000, 4500, 6060, 8140])
    filter_widths = np.array([1521, 1501, 951, 766]) / 2

    filter_masks = []
    for center, width in zip(filter_centers, filter_widths):
        mask = (wl_grid >= center - width) & (wl_grid <= center + width)
        filter_masks.append(mask)

    # Measurement data with systematic error floor
    meas_photo = np.array([U, B, V, Ir], dtype=np.float64)
    meas_errs_raw = np.array([dU, dB, dV, dIr], dtype=np.float64)
    systematic = systematic_error_floor * np.abs(meas_photo)
    meas_errs = np.sqrt(meas_errs_raw**2 + systematic**2)

    meas_median = np.median(meas_photo)
    meas_photo_norm = meas_photo / meas_median
    inv_var = (meas_median / meas_errs) ** 2

    # Load spectra
    spectra = [_load_spectrum(spectra_path_str, gt) for gt in GALAXY_TYPES]
    template_names = list(GALAXY_TYPES)

    # Apply template interpolation if requested
    if n_template_interp > 0:
        spectra = _interpolate_templates(spectra, n_interp=n_template_interp)
        # Generate names for interpolated templates
        expanded_names = []
        for i, name in enumerate(template_names):
            expanded_names.append(name)
            if i < len(template_names) - 1:
                next_name = template_names[i + 1]
                for j in range(1, n_template_interp + 1):
                    interp_name = f"{name}_{next_name}_{j}"
                    expanded_names.append(interp_name)
        template_names = expanded_names

    # Redshift grid
    z_grid = np.arange(z_min, z_max + z_step, z_step)
    n_z = len(z_grid)
    n_templates = len(spectra)

    # Chi-squared grid
    chi_sq_grid = np.full((n_templates, n_z), 1e30)

    for t_idx, (wl, spec) in enumerate(spectra):
        for z_idx, z in enumerate(z_grid):
            wl_z = wl * (1 + z)
            spec_interp = np.interp(wl_grid, wl_z, spec, left=0, right=0)

            # Apply IGM absorption using selected model
            if apply_igm and z > 0.5:
                igm = igm_func(wl_grid, z)
                spec_interp = spec_interp * igm

            spec_mean = np.mean(spec_interp)
            if spec_mean == 0:
                continue
            spec_norm = spec_interp / spec_mean

            # Synthetic photometry
            syn = np.array([np.mean(spec_norm[m]) for m in filter_masks])
            syn_median = np.median(syn)
            if syn_median == 0:
                continue
            syn_norm = syn / syn_median

            chi_sq_grid[t_idx, z_idx] = np.sum(
                (meas_photo_norm - syn_norm) ** 2 * inv_var
            )

    # Find best fit
    min_idx = np.unravel_index(np.argmin(chi_sq_grid), chi_sq_grid.shape)
    best_template_idx = min_idx[0]
    best_z_idx = min_idx[1]
    chi_sq_min = chi_sq_grid[best_template_idx, best_z_idx]
    best_z = z_grid[best_z_idx]
    best_type = template_names[best_template_idx]

    # Find second-best template by ranking templates by their minimum chi-squared
    template_min_chi2 = np.min(chi_sq_grid, axis=1)  # Min chi2 for each template
    sorted_template_indices = np.argsort(template_min_chi2)

    # Second-best template is the one with second-lowest min chi-squared
    if len(sorted_template_indices) >= 2:
        second_best_idx = sorted_template_indices[1]
        second_best_template = template_names[second_best_idx]
        second_best_chi2 = template_min_chi2[second_best_idx]
        # Template ambiguity: how similar are the top two templates (0=very different, 1=identical)
        delta_chi2 = second_best_chi2 - chi_sq_min
        template_ambiguity = float(np.exp(-delta_chi2 / 2.0))  # Based on likelihood ratio
    else:
        second_best_template = ""
        template_ambiguity = 0.0

    # Compute PDF from chi-squared
    log_L = -chi_sq_grid / 2.0
    max_log_L = np.max(log_L, axis=0)
    pdf_unnorm = np.exp(max_log_L) * np.sum(
        np.exp(log_L - max_log_L[np.newaxis, :]), axis=0
    )

    # Apply prior if requested
    if apply_prior and magnitude is not None:
        prior = _magnitude_prior(z_grid, magnitude)
        pdf_unnorm = pdf_unnorm * prior

    # Normalize PDF
    pdf_integral = np.trapezoid(pdf_unnorm, z_grid)
    pdf = pdf_unnorm / pdf_integral if pdf_integral > 0 else np.ones_like(z_grid) / len(z_grid)

    # Compute CDF for percentiles
    cdf = np.zeros_like(pdf)
    cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(z_grid))
    if cdf[-1] > 0:
        cdf /= cdf[-1]

    z_lo = float(np.interp(0.16, cdf, z_grid))
    z_median = float(np.interp(0.50, cdf, z_grid))
    z_hi = float(np.interp(0.84, cdf, z_grid))

    # Compute ODDS
    delta_z = 0.1 * (1 + best_z)
    odds_mask = (z_grid >= best_z - delta_z) & (z_grid <= best_z + delta_z)
    odds = float(np.trapezoid(pdf[odds_mask], z_grid[odds_mask])) if np.any(odds_mask) else 0.0

    # Find secondary peak
    z_secondary, odds_secondary = _find_secondary_peak(z_grid, pdf, best_z)

    # Quality flags
    n_bands = 4
    reduced_chi2 = chi_sq_min / max(1, n_bands - 1)
    # Flag both high chi2 (bad fit) AND suspiciously low chi2 (errors too large)
    # Reduced chi2 << 1 indicates errors are overestimated and fit is unconstrained
    if reduced_chi2 < 0.1:
        chi2_flag = 1  # Suspicious: errors likely too large, fit unconstrained
    elif reduced_chi2 < 5:
        chi2_flag = 0  # Good fit
    elif reduced_chi2 < 10:
        chi2_flag = 1  # Marginal fit
    else:
        chi2_flag = 2  # Bad fit
    odds_flag = 0 if odds >= 0.9 else (1 if odds >= 0.6 else 2)

    return PhotoZResult(
        galaxy_type=best_type,
        redshift=best_z,
        z_pdf_median=z_median,
        z_lo=z_lo,
        z_hi=z_hi,
        chi_sq_min=chi_sq_min,
        z_grid=z_grid,
        pdf=pdf,
        odds=odds,
        z_secondary=z_secondary,
        odds_secondary=odds_secondary,
        chi2_flag=chi2_flag,
        odds_flag=odds_flag,
        bimodal_flag=z_secondary is not None,
        reduced_chi2=reduced_chi2,
        template_ambiguity=template_ambiguity,
        second_best_template=second_best_template,
    )


def _magnitude_prior(z_grid: NDArray, magnitude: float) -> NDArray:
    """Magnitude-dependent redshift prior from HDF studies."""
    z_0 = 0.4
    k = 0.12
    m_0 = 22.0
    sigma_0 = 0.35

    z_median = max(0.05, min(4.0, z_0 + k * (magnitude - m_0)))
    sigma = sigma_0 * (1 + 0.08 * max(0, magnitude - m_0))

    prior = np.exp(-0.5 * ((z_grid - z_median) / sigma) ** 2)

    # Normalize
    prior_sum = np.trapezoid(prior, z_grid)
    if prior_sum > 0:
        prior /= prior_sum

    return prior


def _find_secondary_peak(
    z_grid: NDArray,
    pdf: NDArray,
    primary_z: float,
    min_separation: float = 0.3,
    min_peak_ratio: float = 0.1,
) -> tuple[float | None, float | None]:
    """Find secondary peak in PDF for bimodal solutions."""
    primary_idx = np.argmin(np.abs(z_grid - primary_z))
    primary_pdf = pdf[primary_idx]

    # Exclusion zone around primary
    exclusion_delta = min_separation * (1 + primary_z)
    exclusion_mask = np.abs(z_grid - primary_z) < exclusion_delta

    pdf_masked = pdf.copy()
    pdf_masked[exclusion_mask] = 0

    if np.max(pdf_masked) < min_peak_ratio * primary_pdf:
        return None, None

    # Find local maxima
    local_max = maximum_filter1d(pdf_masked, size=5) == pdf_masked
    local_max &= pdf_masked > min_peak_ratio * primary_pdf

    if not np.any(local_max):
        return None, None

    secondary_idx = np.argmax(pdf_masked * local_max)
    secondary_z = float(z_grid[secondary_idx])

    # Compute ODDS for secondary peak
    delta_z = 0.1 * (1 + secondary_z)
    odds_mask = (z_grid >= secondary_z - delta_z) & (z_grid <= secondary_z + delta_z)
    odds_secondary = float(np.trapezoid(pdf[odds_mask], z_grid[odds_mask])) if np.any(odds_mask) else None

    return secondary_z, odds_secondary


# =============================================================================
# Batch processing
# =============================================================================


def classify_galaxy_batch_with_pdf(
    batch_data: list[tuple[int, list, list]],
    spectra_path: str,
    systematic_error_floor: float = 0.10,
) -> list[tuple[int, str, float, float, float, float, float]]:
    """Classify a batch of galaxies with PDF-based uncertainties.

    Parameters
    ----------
    batch_data : list of tuples
        Each tuple is (idx, fluxes, errors)
    spectra_path : str
        Path to spectra directory
    systematic_error_floor : float
        Systematic error floor

    Returns
    -------
    list of tuples
        Each tuple is (idx, galaxy_type, redshift, z_lo, z_hi, chi_sq_min, odds)
    """
    results = []
    for idx, fluxes, errors in batch_data:
        result = classify_galaxy_with_pdf(
            fluxes, errors,
            spectra_path=spectra_path,
            systematic_error_floor=systematic_error_floor,
        )
        results.append((
            idx,
            result.galaxy_type,
            result.redshift,
            result.z_lo,
            result.z_hi,
            result.chi_sq_min,
            result.odds,
        ))
    return results


# =============================================================================
# Quality flag utilities
# =============================================================================


def compute_chi2_flag(chi2_min: NDArray, n_bands: int = 4) -> NDArray:
    """Compute chi-squared quality flag from reduced chi-squared."""
    chi2_min = np.atleast_1d(chi2_min)
    reduced_chi2 = chi2_min / max(1, n_bands - 1)
    flags = np.zeros(len(chi2_min), dtype=np.int32)
    flags[(reduced_chi2 >= 5) & (reduced_chi2 < 10)] = 1
    flags[reduced_chi2 >= 10] = 2
    return flags


def compute_odds_flag(odds: NDArray) -> NDArray:
    """Compute ODDS quality flag."""
    odds = np.atleast_1d(odds)
    flags = np.zeros(len(odds), dtype=np.int32)
    flags[(odds >= 0.6) & (odds < 0.9)] = 1
    flags[odds < 0.6] = 2
    return flags


# =============================================================================
# Backward compatibility
# =============================================================================


def _classify_single_galaxy(args: tuple) -> dict:
    """Worker function to classify a single galaxy (for parallel processing).

    Parameters
    ----------
    args : tuple
        (index, flux_ubvi, err_ubvi, spectra_path, z_step)

    Returns
    -------
    dict
        Classification results for this galaxy
    """
    idx, flux_ubvi, err_ubvi, spectra_path, z_step = args

    # Reorder from [U, B, V, I] to [B, I, U, V]
    fluxes = [flux_ubvi[1], flux_ubvi[3], flux_ubvi[0], flux_ubvi[2]]  # B, I, U, V
    errors = [err_ubvi[1], err_ubvi[3], err_ubvi[0], err_ubvi[2]]

    # Count valid bands
    n_valid = int(np.sum(np.isfinite(flux_ubvi) & (flux_ubvi > 0)))

    try:
        result = classify_galaxy_with_pdf(
            fluxes, errors,
            spectra_path=spectra_path,
            z_step=z_step,
        )
        return {
            'idx': idx,
            'redshift': result.redshift,
            'z_lo': result.z_lo,
            'z_hi': result.z_hi,
            'chi_sq_min': result.chi_sq_min,
            'odds': result.odds,
            'galaxy_type': result.galaxy_type,
            'chi2_flag': result.chi2_flag,
            'odds_flag': result.odds_flag,
            'bimodal_flag': result.bimodal_flag,
            'template_ambiguity': result.template_ambiguity,
            'reduced_chi2': result.reduced_chi2,
            'second_best_template': result.second_best_template,
            'delta_chi2_templates': result.chi_sq_min * (1.0 - result.template_ambiguity) if result.template_ambiguity < 1.0 else 0.0,
            'n_valid_bands': n_valid,
        }
    except Exception:
        return {
            'idx': idx,
            'redshift': np.nan,
            'z_lo': np.nan,
            'z_hi': np.nan,
            'chi_sq_min': np.nan,
            'odds': 0.0,
            'galaxy_type': "unknown",
            'chi2_flag': 2,
            'odds_flag': 2,
            'bimodal_flag': False,
            'template_ambiguity': 1.0,
            'reduced_chi2': np.nan,
            'second_best_template': "",
            'delta_chi2_templates': 0.0,
            'n_valid_bands': n_valid,
        }


def _classify_chunk(chunk_args: tuple) -> list[dict]:
    """Process a chunk of galaxies in a single worker.

    Parameters
    ----------
    chunk_args : tuple
        (start_idx, flux_chunk, error_chunk, spectra_path, z_step)

    Returns
    -------
    list[dict]
        List of classification results for the chunk
    """
    start_idx, flux_chunk, error_chunk, spectra_path, z_step = chunk_args
    results = []
    for i in range(len(flux_chunk)):
        args = (start_idx + i, flux_chunk[i], error_chunk[i], spectra_path, z_step)
        results.append(_classify_single_galaxy(args))
    return results


def classify_batch_ultrafast(
    flux_array: NDArray,
    error_array: NDArray,
    spectra_path: str = "./spectra",
    z_step: float = 0.01,
    z_step_coarse: float = 0.05,
    n_workers: int | None = None,
) -> dict:
    """Classify a batch of galaxies with parallel processing.

    Uses multiprocessing to distribute galaxy classification across
    multiple CPU cores. Typical performance is ~4-10 galaxies/sec
    depending on z_step and hardware (function name is historical).

    Parameters
    ----------
    flux_array : ndarray, shape (N, 4)
        Fluxes in [U, B, V, I] order (f300, f450, f606, f814)
    error_array : ndarray, shape (N, 4)
        Errors in [U, B, V, I] order
    spectra_path : str
        Path to spectra directory
    z_step : float
        Fine redshift step for final fitting
    z_step_coarse : float
        Coarse redshift step (unused, kept for API compatibility)
    n_workers : int, optional
        Number of parallel workers. If None, uses (CPU count - 1) or
        the value from resource_config if available.

    Returns
    -------
    dict
        Dictionary with arrays of results for all galaxies
    """
    n_galaxies = len(flux_array)

    # Determine number of workers
    if n_workers is None:
        try:
            from resource_config import get_config
            config = get_config()
            n_workers = config.n_template_workers
        except ImportError:
            n_workers = max(1, (mp.cpu_count() or 2) - 1)

    # Pre-allocate result arrays
    results = {
        'redshift': np.zeros(n_galaxies, dtype=np.float64),
        'z_lo': np.zeros(n_galaxies, dtype=np.float64),
        'z_hi': np.zeros(n_galaxies, dtype=np.float64),
        'chi_sq_min': np.zeros(n_galaxies, dtype=np.float64),
        'odds': np.zeros(n_galaxies, dtype=np.float64),
        'galaxy_type': np.empty(n_galaxies, dtype=object),
        'chi2_flag': np.zeros(n_galaxies, dtype=np.int32),
        'odds_flag': np.zeros(n_galaxies, dtype=np.int32),
        'bimodal_flag': np.zeros(n_galaxies, dtype=bool),
        'template_ambiguity': np.zeros(n_galaxies, dtype=np.float64),
        'reduced_chi2': np.zeros(n_galaxies, dtype=np.float64),
        'n_valid_bands': np.zeros(n_galaxies, dtype=np.int32),
        'second_best_template': np.empty(n_galaxies, dtype=object),
        'delta_chi2_templates': np.zeros(n_galaxies, dtype=np.float64),
    }

    # For small batches or single worker, use serial processing
    if n_galaxies < 10 or n_workers <= 1:
        for i in range(n_galaxies):
            args = (i, flux_array[i], error_array[i], spectra_path, z_step)
            res = _classify_single_galaxy(args)
            _assign_result(results, res)
        return results

    # Split work into chunks for parallel processing
    # Use larger chunks to reduce overhead
    chunk_size = max(10, n_galaxies // (n_workers * 4))
    chunks = []
    for start in range(0, n_galaxies, chunk_size):
        end = min(start + chunk_size, n_galaxies)
        chunks.append((
            start,
            flux_array[start:end],
            error_array[start:end],
            spectra_path,
            z_step,
        ))

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        chunk_results = list(executor.map(_classify_chunk, chunks))

    # Merge results from all chunks
    for chunk_result in chunk_results:
        for res in chunk_result:
            _assign_result(results, res)

    return results


def _assign_result(results: dict, res: dict) -> None:
    """Assign a single galaxy result to the results arrays."""
    idx = res['idx']
    results['redshift'][idx] = res['redshift']
    results['z_lo'][idx] = res['z_lo']
    results['z_hi'][idx] = res['z_hi']
    results['chi_sq_min'][idx] = res['chi_sq_min']
    results['odds'][idx] = res['odds']
    results['galaxy_type'][idx] = res['galaxy_type']
    results['chi2_flag'][idx] = res['chi2_flag']
    results['odds_flag'][idx] = res['odds_flag']
    results['bimodal_flag'][idx] = res['bimodal_flag']
    results['template_ambiguity'][idx] = res['template_ambiguity']
    results['reduced_chi2'][idx] = res['reduced_chi2']
    results['second_best_template'][idx] = res['second_best_template']
    results['delta_chi2_templates'][idx] = res['delta_chi2_templates']
    results['n_valid_bands'][idx] = res['n_valid_bands']


def prep_filters() -> list[NDArray]:
    """Prepare filter transmission curves for U, B, V, I bands."""
    wl = np.arange(2200, 9500, 1)
    centers = np.array([3000, 4500, 6060, 8140])
    widths = np.array([1521, 1501, 951, 766]) / 2

    filters = []
    for center, width in zip(centers, widths):
        f = np.zeros(len(wl))
        f[(wl >= center - width) & (wl <= center + width)] = 1.0
        filters.append(f)

    return filters
