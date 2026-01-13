"""Galaxy classification via SED template fitting.

Implements professional photometric redshift estimation with:
- Chi-squared template fitting (Bertin & Arnouts 1996 style)
- Full probability distribution functions (PDFs) for redshift
- Uncertainty quantification (16th/84th percentiles)
- Support for CWW + Kinney-style templates
- IGM absorption following Madau (1995)
- Magnitude-dependent redshift prior from HDF studies
- Template interpolation (EAZY-style linear combinations)
- Secondary peak detection for bimodal solutions

References:
- Brammer et al. 2008 (EAZY)
- Ilbert et al. 2006 (COSMOS photo-z)
- Coleman, Wu & Weedman 1980 (CWW templates)
- Madau 1995 (IGM absorption)
- Fernández-Soto et al. 1999 (HDF photo-z)
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from scipy.ndimage import maximum_filter1d

# Optional numba for JIT compilation (50-100x speedup for IGM calculations)
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: identity decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) is False else decorator(args[0]) if args else decorator
    def prange(*args):
        return range(*args)

# Default spectra path relative to this file
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

# =============================================================================
# IGM Absorption Model (Madau 1995) - JIT-compiled for performance
# =============================================================================

# JIT-compiled core for Madau95 model (50-100x faster when numba available)
@njit(cache=True, fastmath=True)
def _igm_madau95_core(wavelength: np.ndarray, redshift: float) -> np.ndarray:
    """JIT-compiled Madau 1995 IGM absorption core."""
    n = len(wavelength)
    tau_total = np.zeros(n, dtype=np.float64)

    # Lyman transitions: alpha, beta, gamma
    lyman_wl = np.array([1216.0, 1026.0, 972.5])
    lyman_coeff = np.array([0.0036, 0.0017, 0.0012])
    lyman_limit = 912.0
    one_plus_z = 1.0 + redshift

    # Forest absorption for all 3 transitions
    for j in range(3):
        obs_wl = lyman_wl[j] * one_plus_z
        for i in range(n):
            if wavelength[i] < obs_wl:
                x = wavelength[i] / lyman_wl[j]
                tau_total[i] += lyman_coeff[j] * (x ** 3.46)

    # Lyman limit continuum
    obs_ll = lyman_limit * one_plus_z
    for i in range(n):
        if wavelength[i] < obs_ll:
            x_ll = wavelength[i] / lyman_limit
            tau_ll = 0.25 * (x_ll ** 3) * (one_plus_z ** 0.46)
            tau_total[i] += min(tau_ll, 30.0)

    # Compute transmission
    transmission = np.exp(-tau_total)
    return np.clip(transmission, 0.0, 1.0)


@njit(cache=True, fastmath=True)
def _igm_inoue14_core(wavelength: np.ndarray, redshift: float) -> np.ndarray:
    """JIT-compiled Inoue et al. 2014 IGM absorption core."""
    n = len(wavelength)
    tau_total = np.zeros(n, dtype=np.float64)
    one_plus_z = 1.0 + redshift

    # Extended Lyman series (8 transitions)
    lyman_wl = np.array([1215.67, 1025.72, 972.537, 949.743, 937.803, 930.748, 926.226, 923.150])
    A_laf = np.array([1.690e-2, 4.692e-3, 2.239e-3, 1.319e-3, 8.707e-4, 6.178e-4, 4.609e-4, 3.569e-4])
    A_dla = np.array([1.617e-4, 1.545e-4, 1.498e-4, 1.460e-4, 1.429e-4, 1.402e-4, 1.377e-4, 1.355e-4])

    # LAF and DLA contributions
    for j in range(8):
        obs_lam = lyman_wl[j] * one_plus_z
        for i in range(n):
            if wavelength[i] < obs_lam:
                z_abs = wavelength[i] / lyman_wl[j] - 1.0

                # LAF optical depth
                if redshift < 1.2:
                    tau_laf = A_laf[j] * ((1.0 + z_abs) ** 1.2)
                elif redshift < 4.7:
                    tau_laf = A_laf[j] * ((1.0 + z_abs) ** 3.7)
                else:
                    tau_laf = A_laf[j] * ((1.0 + z_abs) ** 5.5)
                tau_total[i] += tau_laf

                # DLA optical depth
                if redshift < 2.0:
                    tau_dla = A_dla[j] * ((1.0 + z_abs) ** 2.0)
                else:
                    tau_dla = A_dla[j] * ((1.0 + z_abs) ** 3.0)
                tau_total[i] += tau_dla

    # LLS contribution
    lyman_limit = 911.75
    obs_ll = lyman_limit * one_plus_z
    for i in range(n):
        if wavelength[i] < obs_ll:
            z_ll = wavelength[i] / lyman_limit - 1.0
            if redshift < 1.2:
                tau_lls = 0.325 * (((1.0 + z_ll) ** 1.2) - 1.0)
            elif redshift < 4.7:
                tau_lls = 0.325 * (((1.0 + z_ll) ** 3.7) - 1.0)
            else:
                tau_lls = 0.325 * (((1.0 + z_ll) ** 5.5) - 1.0)
            tau_total[i] += min(tau_lls, 50.0)

    transmission = np.exp(-tau_total)
    return np.clip(transmission, 0.0, 1.0)


@njit(cache=True, parallel=True, fastmath=True)
def igm_absorption_grid(wavelength: np.ndarray, z_grid: np.ndarray, model_id: int = 0) -> np.ndarray:
    """
    Compute IGM absorption for entire redshift grid at once (vectorized).

    Parameters
    ----------
    wavelength : ndarray
        Wavelength array
    z_grid : ndarray
        Redshift grid
    model_id : int
        0 = madau95, 1 = inoue14

    Returns
    -------
    ndarray
        Shape (n_z, n_wavelength) transmission array
    """
    n_z = len(z_grid)
    n_wl = len(wavelength)
    result = np.ones((n_z, n_wl), dtype=np.float64)

    for z_idx in prange(n_z):
        z = z_grid[z_idx]
        if z < 0.01:
            continue
        if model_id == 0:
            result[z_idx] = _igm_madau95_core(wavelength, z)
        else:
            result[z_idx] = _igm_inoue14_core(wavelength, z)

    return result


def igm_absorption(
    wavelength: NDArray,
    redshift: float,
    model: str = "madau95",
) -> NDArray:
    """
    Calculate IGM absorption following Madau (1995) or Inoue et al. (2014).

    The intergalactic medium absorbs flux blueward of Lyman-alpha at high
    redshifts due to the Lyman-alpha forest.

    Uses JIT-compiled cores when numba is available (50-100x faster).

    Parameters
    ----------
    wavelength : NDArray
        Rest-frame wavelength in Angstroms
    redshift : float
        Source redshift
    model : str
        IGM model: 'madau95' (default) or 'inoue14' (more accurate for z>4)

    Returns
    -------
    NDArray
        Transmission fraction (0-1) at each wavelength

    References
    ----------
    - Madau 1995, ApJ, 441, 18
    - Inoue et al. 2014, MNRAS, 442, 1805
    """
    if redshift < 0.01:
        return np.ones_like(wavelength)

    # Use JIT-compiled cores for performance
    wl_arr = np.ascontiguousarray(wavelength, dtype=np.float64)
    if model == "inoue14":
        return _igm_inoue14_core(wl_arr, float(redshift))
    return _igm_madau95_core(wl_arr, float(redshift))


@njit(cache=True, fastmath=True)
def _compute_chi_sq_single_template(
    wl_template: np.ndarray,
    spec_template: np.ndarray,
    wl_grid: np.ndarray,
    z_grid: np.ndarray,
    one_plus_z: np.ndarray,
    igm_trans_grid: np.ndarray,
    apply_igm: bool,
    filter_starts: np.ndarray,
    filter_ends: np.ndarray,
    filter_lengths: np.ndarray,
    meas_photo_norm: np.ndarray,
    inv_var: np.ndarray,
) -> np.ndarray:
    """
    JIT-compiled chi-squared computation for a single template across all redshifts.

    This eliminates Python loop overhead and provides ~5-10x speedup for the inner loop.
    """
    n_z = len(z_grid)
    n_bands = len(filter_starts)
    n_wl_grid = len(wl_grid)
    chi_sq = np.full(n_z, 1e30, dtype=np.float64)
    syn_photometry = np.empty(n_bands, dtype=np.float64)

    for z_idx in range(n_z):
        # Apply redshift: observed wavelength = rest wavelength * (1+z)
        factor = one_plus_z[z_idx]

        # Interpolate template to common grid (manual linear interpolation for numba)
        spec_small = np.empty(n_wl_grid, dtype=np.float64)
        for i in range(n_wl_grid):
            target_wl = wl_grid[i]
            # Find position in redshifted template wavelength
            template_wl_target = target_wl / factor  # Convert to rest frame

            # Binary search for interpolation bounds
            if template_wl_target <= wl_template[0]:
                spec_small[i] = spec_template[0]
            elif template_wl_target >= wl_template[-1]:
                spec_small[i] = spec_template[-1]
            else:
                # Linear interpolation
                left = 0
                right = len(wl_template) - 1
                while right - left > 1:
                    mid = (left + right) // 2
                    if wl_template[mid] <= template_wl_target:
                        left = mid
                    else:
                        right = mid
                t = (template_wl_target - wl_template[left]) / (wl_template[right] - wl_template[left])
                spec_small[i] = spec_template[left] + t * (spec_template[right] - spec_template[left])

        # Apply IGM absorption
        if apply_igm and z_grid[z_idx] > 0.5:
            for i in range(n_wl_grid):
                spec_small[i] *= igm_trans_grid[z_idx, i]

        # Normalize by mean
        spec_sum = 0.0
        for i in range(n_wl_grid):
            spec_sum += spec_small[i]
        spec_mean = spec_sum / n_wl_grid
        if spec_mean == 0.0:
            continue
        for i in range(n_wl_grid):
            spec_small[i] /= spec_mean

        # Compute synthetic photometry
        for band_idx in range(n_bands):
            s = filter_starts[band_idx]
            e = filter_ends[band_idx]
            length = filter_lengths[band_idx]
            band_sum = 0.0
            for i in range(s, e):
                band_sum += spec_small[i]
            syn_photometry[band_idx] = band_sum / length

        # Compute median of 4 elements (inline sort)
        syn_sorted = np.sort(syn_photometry)
        syn_median = 0.5 * (syn_sorted[1] + syn_sorted[2])
        if syn_median == 0.0:
            continue

        # Normalize and compute chi-squared
        chi_sq_val = 0.0
        for band_idx in range(n_bands):
            syn_norm = syn_photometry[band_idx] / syn_median
            diff = meas_photo_norm[band_idx] - syn_norm
            chi_sq_val += diff * diff * inv_var[band_idx]
        chi_sq[z_idx] = chi_sq_val

    return chi_sq


def _igm_absorption_inoue14(wavelength: NDArray, redshift: float) -> NDArray:
    """Inoue et al. 2014 IGM absorption model.

    More accurate for z > 4, includes updated Lyman forest and
    Lyman limit system statistics.

    Parameters
    ----------
    wavelength : NDArray
        Rest-frame wavelength in Angstroms
    redshift : float
        Source redshift

    Returns
    -------
    NDArray
        Transmission fraction (0-1) at each wavelength
    """
    one_plus_z = 1 + redshift

    # Extended Lyman series (8 transitions)
    lyman_wavelengths = np.array([
        1215.67,  # Lyman-alpha
        1025.72,  # Lyman-beta
        972.537,  # Lyman-gamma
        949.743,  # Lyman-delta
        937.803,  # Lyman-epsilon
        930.748,  # Lyman-6
        926.226,  # Lyman-7
        923.150,  # Lyman-8
    ])

    # Inoue+14 coefficients (Table 2)
    # Effective optical depth coefficients for LAF
    A_laf = np.array([
        1.690e-2,  # alpha
        4.692e-3,  # beta
        2.239e-3,  # gamma
        1.319e-3,  # delta
        8.707e-4,  # epsilon
        6.178e-4,  # 6
        4.609e-4,  # 7
        3.569e-4,  # 8
    ])

    # Damped Lyman-alpha system coefficients
    A_dla = np.array([
        1.617e-4,
        1.545e-4,
        1.498e-4,
        1.460e-4,
        1.429e-4,
        1.402e-4,
        1.377e-4,
        1.355e-4,
    ])

    # Initialize optical depth
    tau_total = np.zeros_like(wavelength, dtype=np.float64)

    # LAF contribution (Lyman-alpha forest)
    for i, (lam_i, A_i) in enumerate(zip(lyman_wavelengths, A_laf)):
        obs_lam = lam_i * one_plus_z
        mask = wavelength < obs_lam

        if np.any(mask):
            z_abs = wavelength[mask] / lam_i - 1

            # LAF optical depth (Eq. 21 in Inoue+14)
            if redshift < 1.2:
                tau_laf = A_i * (1 + z_abs) ** 1.2
            elif redshift < 4.7:
                tau_laf = A_i * (1 + z_abs) ** 3.7
            else:
                tau_laf = A_i * (1 + z_abs) ** 5.5

            tau_total[mask] += tau_laf

    # DLA contribution
    for i, (lam_i, A_i) in enumerate(zip(lyman_wavelengths, A_dla)):
        obs_lam = lam_i * one_plus_z
        mask = wavelength < obs_lam

        if np.any(mask):
            z_abs = wavelength[mask] / lam_i - 1

            # DLA optical depth
            if redshift < 2.0:
                tau_dla = A_i * (1 + z_abs) ** 2.0
            else:
                tau_dla = A_i * (1 + z_abs) ** 3.0

            tau_total[mask] += tau_dla

    # Lyman limit systems (LLS) contribution
    lyman_limit = 911.75
    obs_ll = lyman_limit * one_plus_z
    mask_ll = wavelength < obs_ll

    if np.any(mask_ll):
        z_ll = wavelength[mask_ll] / lyman_limit - 1

        # LLS optical depth (simplified from Inoue+14 Eq. 25)
        if redshift < 1.2:
            tau_lls = 0.325 * ((1 + z_ll) ** 1.2 - 1)
        elif redshift < 4.7:
            tau_lls = 0.325 * ((1 + z_ll) ** 3.7 - 1)
        else:
            tau_lls = 0.325 * ((1 + z_ll) ** 5.5 - 1)

        tau_total[mask_ll] += np.minimum(tau_lls, 50)  # Cap for stability

    # Compute transmission
    transmission = np.exp(-tau_total)

    return np.clip(transmission, 0, 1)


# =============================================================================
# Magnitude-Dependent Redshift Prior
# =============================================================================

def magnitude_prior(z_grid: NDArray, magnitude: float, band: str = 'I') -> NDArray:
    """
    Magnitude-dependent redshift prior P(z|m) from HDF studies.

    Based on empirical N(z,m) distributions from Fernández-Soto et al. (1999)
    and COSMOS (Ilbert et al. 2006). Fainter galaxies are more likely to be
    at higher redshift.

    Parameters
    ----------
    z_grid : NDArray
        Redshift grid
    magnitude : float
        AB magnitude (typically I-band)
    band : str
        Photometric band for magnitude ('I' or 'B')

    Returns
    -------
    NDArray
        Prior probability P(z|m), normalized
    """
    # Parameters fit to HDF N(z,m) distribution
    # z_median scales with magnitude: fainter = higher z
    # Based on Fernández-Soto 1999 and Benítez 2000 (BPZ)

    if band == 'I':
        # I-band magnitude relation
        # z_m = z_0 + k * (m - m_0)
        z_0 = 0.3
        k = 0.1  # z increases ~0.1 per magnitude
        m_0 = 22.0  # Reference magnitude
        sigma_0 = 0.3  # Intrinsic scatter
    else:
        # B-band (bluer, different relation)
        z_0 = 0.2
        k = 0.08
        m_0 = 24.0
        sigma_0 = 0.35

    # Median redshift for this magnitude
    z_median = max(0.05, z_0 + k * (magnitude - m_0))

    # Width increases with magnitude (fainter = more uncertain)
    sigma = sigma_0 * (1 + 0.1 * max(0, magnitude - m_0))

    # Gaussian prior centered on z_median
    # But with asymmetric tail toward higher z
    prior = np.exp(-0.5 * ((z_grid - z_median) / sigma) ** 2)

    # Add power-law tail at high-z (galaxies at z>2 do exist)
    # Avoid divide-by-zero by using z_grid + small epsilon
    with np.errstate(divide='ignore', invalid='ignore'):
        high_z_tail = np.where(
            z_grid > 0,
            (z_grid / z_median) ** (-2) * np.exp(-z_grid / 5.0),
            0.0
        )
    high_z_mask = z_grid > z_median
    prior[high_z_mask] = np.maximum(prior[high_z_mask], 0.1 * high_z_tail[high_z_mask])

    # Low-z cutoff (very few galaxies at z<0.01)
    prior[z_grid < 0.01] *= 0.01

    # Normalize
    prior_sum = np.trapezoid(prior, z_grid)
    if prior_sum > 0:
        prior /= prior_sum

    return prior


# =============================================================================
# Template Interpolation (EAZY-style)
# =============================================================================

def interpolate_templates(
    spectra_list: list[tuple[NDArray, NDArray]],
    n_interp: int = 2
) -> list[tuple[NDArray, NDArray, str]]:
    """
    Create interpolated templates between adjacent galaxy types.

    This follows the EAZY approach of creating linear combinations
    to fill the gaps in template space, improving photo-z accuracy.

    Parameters
    ----------
    spectra_list : list
        List of (wavelength, spectrum) tuples for each template
    n_interp : int
        Number of interpolated templates between each pair

    Returns
    -------
    list
        Extended list of (wavelength, spectrum, type_name) tuples
    """
    if n_interp < 1:
        return [(wl, spec, GALAXY_TYPES[i]) for i, (wl, spec) in enumerate(spectra_list)]

    extended_templates = []

    for i, (wl1, spec1) in enumerate(spectra_list):
        # Add original template
        extended_templates.append((wl1, spec1, GALAXY_TYPES[i]))

        # Interpolate to next template (if not last)
        if i < len(spectra_list) - 1:
            wl2, spec2 = spectra_list[i + 1]

            # Interpolate both to common wavelength grid
            wl_common = np.union1d(wl1, wl2)
            spec1_interp = np.interp(wl_common, wl1, spec1, left=0, right=0)
            spec2_interp = np.interp(wl_common, wl2, spec2, left=0, right=0)

            for j in range(1, n_interp + 1):
                alpha = j / (n_interp + 1)
                spec_interp = (1 - alpha) * spec1_interp + alpha * spec2_interp
                type_name = f"{GALAXY_TYPES[i]}_{GALAXY_TYPES[i+1]}_{j}"
                extended_templates.append((wl_common, spec_interp, type_name))

    return extended_templates


def find_secondary_peak(
    z_grid: NDArray,
    pdf: NDArray,
    primary_z: float,
    min_separation: float = 0.3,
    min_peak_ratio: float = 0.1
) -> tuple[Optional[float], Optional[float]]:
    """
    Find secondary peak in PDF for bimodal redshift solutions.

    Parameters
    ----------
    z_grid : NDArray
        Redshift grid
    pdf : NDArray
        Normalized PDF
    primary_z : float
        Primary (best) redshift
    min_separation : float
        Minimum separation from primary peak in Δz/(1+z)
    min_peak_ratio : float
        Minimum ratio of secondary to primary peak height

    Returns
    -------
    tuple
        (z_secondary, pdf_secondary) or (None, None) if no secondary peak
    """
    # Find primary peak value
    primary_idx = np.argmin(np.abs(z_grid - primary_z))
    primary_pdf = pdf[primary_idx]

    # Define exclusion zone around primary peak
    exclusion_delta = min_separation * (1 + primary_z)
    exclusion_mask = np.abs(z_grid - primary_z) < exclusion_delta

    # Mask out exclusion zone
    pdf_masked = pdf.copy()
    pdf_masked[exclusion_mask] = 0

    if np.max(pdf_masked) < min_peak_ratio * primary_pdf:
        return None, None

    # Find local maxima in masked PDF
    # A point is a local max if it's greater than its neighbors
    local_max = maximum_filter1d(pdf_masked, size=5) == pdf_masked
    local_max &= pdf_masked > min_peak_ratio * primary_pdf

    if not np.any(local_max):
        return None, None

    # Get the highest secondary peak
    secondary_idx = np.argmax(pdf_masked * local_max)
    secondary_z = z_grid[secondary_idx]
    secondary_pdf = pdf[secondary_idx]

    return secondary_z, secondary_pdf


class PhotoZResult(NamedTuple):
    """Result of photometric redshift estimation with uncertainties.

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
        ODDS parameter: integral of PDF within ±0.1(1+z) of peak
    z_secondary : float or None
        Secondary peak redshift (if bimodal), None otherwise
    odds_secondary : float or None
        ODDS for secondary peak, None if no secondary peak
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
    z_secondary: Optional[float] = None
    odds_secondary: Optional[float] = None

# Common wavelength grid for interpolation
_WL_GRID = np.arange(2200, 9500, 1)


@lru_cache(maxsize=1)
def _get_filter_slices() -> tuple[tuple[int, int, int], ...]:
    """Get pre-computed filter slice indices with lengths (cached).

    Returns (start, end, length) tuples for each filter band.
    Direct slicing is faster than boolean mask indexing.
    Pre-computing length avoids repeated (end - start) calculations in hot loops.
    """
    filter_centers = np.array([3000, 4500, 6060, 8140])
    filter_widths = np.array([1521, 1501, 951, 766]) / 2
    slices = []
    for center, width in zip(filter_centers, filter_widths):
        start = int(center - width - _WL_GRID[0])
        end = int(center + width - _WL_GRID[0]) + 1
        start = max(0, start)
        end = min(len(_WL_GRID), end)
        slices.append((start, end, end - start))
    return tuple(slices)


@lru_cache(maxsize=1)
def _get_filter_arrays() -> tuple[NDArray, ...]:
    """Get pre-computed filter arrays (cached)."""
    filter_centers = np.array([3000, 4500, 6060, 8140])
    filter_widths = np.array([1521, 1501, 951, 766]) / 2
    filters = []
    for center, width in zip(filter_centers, filter_widths):
        filter_spec = np.zeros(len(_WL_GRID), dtype=np.float32)
        mask = (_WL_GRID >= (center - width)) & (_WL_GRID <= (center + width))
        filter_spec[mask] = 1.0
        filters.append(filter_spec)
    return tuple(filters)


@lru_cache(maxsize=1)
def _get_filter_masks() -> tuple[NDArray, ...]:
    """Get boolean masks for filter regions (cached)."""
    filters = _get_filter_arrays()
    return tuple(f != 0 for f in filters)


@lru_cache(maxsize=64)
def _load_spectrum(spectra_path: str, galaxy_type: str) -> tuple[NDArray, NDArray]:
    """Load and cache galaxy spectrum from disk.

    Returns wavelength and spectrum arrays. Cached to avoid repeated I/O.
    Increased cache size to 64 to handle interpolated templates without eviction.
    """
    path = Path(spectra_path) / f"{galaxy_type}.dat"
    wl, spec = np.loadtxt(path, usecols=[0, 1], unpack=True)
    return wl.astype(np.float32), spec.astype(np.float32)


def prep_filters() -> list[NDArray[np.floating]]:
    """Prepare filter transmission curves for U, B, V, I bands."""
    wl_filter_small = np.arange(2200, 9500, 1)
    filter_centers = np.array([3000, 4500, 6060, 8140])
    filter_widths = np.array([1521, 1501, 951, 766]) / 2
    filters = []
    for i in range(len(filter_centers)):
        center = filter_centers[i]
        width = filter_widths[i]
        filter_spec = np.zeros(len(wl_filter_small))
        filter_spec[
            (wl_filter_small >= (center - width)) & (wl_filter_small <= (center + width))
        ] = 1.0
        filters.append(filter_spec)
    return filters


def classify_galaxy(
    fluxes: ArrayLike,
    errors: ArrayLike,
    spectra_path: Path | str | None = None,
) -> tuple[str, float]:
    """
    Classify a galaxy by fitting SED templates to photometric observations.

    Parameters
    ----------
    fluxes : array-like
        Measured fluxes in [B, I, U, V] bands.
    errors : array-like
        Measurement errors for each band.
    spectra_path : Path or str, optional
        Directory containing galaxy template spectra. Defaults to ./spectra.

    Returns
    -------
    tuple[str, float]
        Galaxy type (e.g., 'elliptical', 'Sa') and best-fit redshift.
    """
    if spectra_path is None:
        spectra_path = _DEFAULT_SPECTRA_PATH
    spectra_path_str = str(spectra_path)

    B, Ir, U, V = fluxes
    dB, dIr, dU, dV = errors

    # Get cached filter slices (faster than boolean mask indexing)
    filter_slices = _get_filter_slices()

    # Measurement data (normalized) - use float64 for numerical stability
    meas_photo = np.array([U, B, V, Ir], dtype=np.float64)
    meas_errs = np.array([dU, dB, dV, dIr], dtype=np.float64)
    meas_median = np.median(meas_photo)
    meas_photo_norm = meas_photo / meas_median
    inv_err_sq = (meas_median / meas_errs) ** 2  # Pre-compute inverse variance

    def compute_chi_square(redshift: float, wl: NDArray, spec: NDArray) -> float:
        """Compute chi-square for a given redshift using pre-loaded spectrum."""
        # Apply redshift transformation
        wl_redshifted = wl * (1 + redshift)

        # Interpolate to common grid
        spec_small = np.interp(_WL_GRID, wl_redshifted, spec)
        # Use mean for normalization (faster than median, equivalent for smooth spectra)
        spec_mean = np.mean(spec_small)
        if spec_mean == 0:
            return 1e30
        spec_small_norm = spec_small / spec_mean

        # Compute synthetic photometry using pre-computed slices (faster than masks)
        # Using pre-computed length to avoid repeated (e-s) calculations
        syn_photometry = np.array([
            np.sum(spec_small_norm[s:e]) / length for s, e, length in filter_slices
        ], dtype=np.float64)

        # Inline median for 4 elements (faster than np.median overhead)
        syn_sorted = np.sort(syn_photometry)
        syn_median = 0.5 * (syn_sorted[1] + syn_sorted[2])
        if syn_median == 0:
            return 1e30
        syn_photometry_norm = syn_photometry / syn_median

        # Chi-square with pre-computed inverse variance
        return float(np.sum((meas_photo_norm - syn_photometry_norm) ** 2 * inv_err_sq))

    # Pre-load all spectra once (cached across calls)
    spectra_list = [_load_spectrum(spectra_path_str, gtype) for gtype in GALAXY_TYPES]

    # Coarse grid search to bracket minimum - sparser grid is sufficient
    z_grid = np.linspace(0.0, 3.5, 15)
    best_chi = 1e30
    best_z_coarse = 1.0

    for zg in z_grid:
        for wl, spec in spectra_list:
            chi = compute_chi_square(zg, wl, spec)
            if chi < best_chi:
                best_chi = chi
                best_z_coarse = zg

    # Narrower search window around coarse best
    span = 0.5
    z_lower = max(0.0, best_z_coarse - span)
    z_upper = min(3.0, best_z_coarse + span)

    if z_upper <= z_lower:
        z_upper = min(3.0, z_lower + 0.5)

    # Fine optimization for each galaxy type
    chimins = np.empty(len(GALAXY_TYPES))
    zmins = np.empty(len(GALAXY_TYPES))

    for i, (wl, spec) in enumerate(spectra_list):
        result = minimize_scalar(
            lambda z, w=wl, s=spec: compute_chi_square(z, w, s),
            bounds=(z_lower, z_upper),
            method="bounded",
            options={"xatol": 0.01},  # Looser tolerance for speed
        )
        zmins[i] = result.x
        chimins[i] = result.fun

    best_idx = int(np.argmin(chimins))
    return GALAXY_TYPES[best_idx], float(zmins[best_idx])


def classify_galaxy_with_pdf(
    fluxes: ArrayLike,
    errors: ArrayLike,
    spectra_path: Path | str | None = None,
    z_min: float = 0.0,
    z_max: float = 3.5,
    z_step: float = 0.01,
    systematic_error_floor: float = 0.02,
    apply_igm: bool = True,
    igm_model: str = "madau95",
    apply_prior: bool = False,
    magnitude: Optional[float] = None,
    n_template_interp: int = 0,
) -> PhotoZResult:
    """
    Classify a galaxy with full redshift PDF and uncertainty estimates.

    This implements professional photometric redshift estimation following
    methods used in COSMOS (Ilbert et al. 2006) and EAZY (Brammer et al. 2008).

    Parameters
    ----------
    fluxes : array-like
        Measured fluxes in [B, I, U, V] bands.
    errors : array-like
        Measurement errors for each band.
    spectra_path : Path or str, optional
        Directory containing galaxy template spectra.
    z_min : float
        Minimum redshift to search (default 0.0)
    z_max : float
        Maximum redshift to search (default 3.5, can extend to 6.0 for high-z)
    z_step : float
        Redshift grid step size (default 0.01)
    systematic_error_floor : float
        Systematic error floor as fraction of flux (default 0.02 = 2%)
        Added in quadrature to measurement errors per COSMOS methodology.
    apply_igm : bool
        Apply IGM absorption for high-z sources (default True)
    igm_model : str
        IGM absorption model: 'madau95' (default) or 'inoue14' (better for z>4)
    apply_prior : bool
        Apply magnitude-dependent redshift prior (default False)
    magnitude : float, optional
        I-band magnitude for prior calculation (required if apply_prior=True)
    n_template_interp : int
        Number of interpolated templates between each pair (default 0 = none)

    Returns
    -------
    PhotoZResult
        Named tuple containing:
        - galaxy_type: Best-fit template type
        - redshift: Best-fit redshift (chi-sq minimum)
        - z_pdf_median: Median redshift from PDF
        - z_lo, z_hi: 16th/84th percentile bounds
        - chi_sq_min: Minimum chi-squared
        - z_grid, pdf: Full redshift PDF
        - odds: Quality parameter (PDF concentration)
        - z_secondary: Secondary peak redshift (if any)
        - odds_secondary: ODDS for secondary peak
    """
    if spectra_path is None:
        spectra_path = _DEFAULT_SPECTRA_PATH
    spectra_path_str = str(spectra_path)

    B, Ir, U, V = fluxes
    dB, dIr, dU, dV = errors

    # Get cached filter slices
    filter_slices = _get_filter_slices()

    # Measurement data with systematic error floor added in quadrature
    # This follows COSMOS methodology to prevent over-fitting
    meas_photo = np.array([U, B, V, Ir], dtype=np.float64)
    meas_errs_raw = np.array([dU, dB, dV, dIr], dtype=np.float64)

    # Add systematic error floor: σ_total² = σ_meas² + (f_sys × flux)²
    systematic_errors = systematic_error_floor * np.abs(meas_photo)
    meas_errs = np.sqrt(meas_errs_raw**2 + systematic_errors**2)

    meas_median = np.median(meas_photo)
    meas_photo_norm = meas_photo / meas_median
    inv_var = (meas_median / meas_errs) ** 2  # Pre-compute inverse variance

    # Pre-load all spectra
    base_spectra = [_load_spectrum(spectra_path_str, gtype) for gtype in GALAXY_TYPES]

    # Apply template interpolation if requested
    if n_template_interp > 0:
        extended_templates = interpolate_templates(base_spectra, n_template_interp)
        spectra_list = [(wl, spec) for wl, spec, _ in extended_templates]
        template_names = [name for _, _, name in extended_templates]
    else:
        spectra_list = base_spectra
        template_names = list(GALAXY_TYPES)

    # Create fine redshift grid for PDF calculation
    z_grid = np.arange(z_min, z_max + z_step, z_step)
    n_z = len(z_grid)
    n_templates = len(spectra_list)

    # Compute chi-squared on full grid for all templates
    chi_sq_grid = np.full((n_templates, n_z), 1e30, dtype=np.float64)

    # Pre-compute (1+z) factors for all redshifts
    one_plus_z = 1.0 + z_grid

    # Pre-compute IGM transmission for entire redshift grid (vectorized, ~10x faster)
    wl_grid_arr = np.ascontiguousarray(_WL_GRID, dtype=np.float64)
    z_grid_arr = np.ascontiguousarray(z_grid, dtype=np.float64)
    if apply_igm:
        model_id = 1 if igm_model == "inoue14" else 0
        igm_trans_grid = igm_absorption_grid(wl_grid_arr, z_grid_arr, model_id)
    else:
        # Dummy grid (won't be used but needed for JIT function signature)
        igm_trans_grid = np.ones((n_z, len(_WL_GRID)), dtype=np.float64)

    # Extract filter slices as separate arrays for JIT function
    filter_starts = np.array([s for s, e, l in filter_slices], dtype=np.int64)
    filter_ends = np.array([e for s, e, l in filter_slices], dtype=np.int64)
    filter_lengths = np.array([l for s, e, l in filter_slices], dtype=np.float64)

    # Use JIT-compiled chi-squared computation for each template (~5-10x faster)
    for t_idx, (wl, spec) in enumerate(spectra_list):
        wl_arr = np.ascontiguousarray(wl, dtype=np.float64)
        spec_arr = np.ascontiguousarray(spec, dtype=np.float64)
        chi_sq_grid[t_idx] = _compute_chi_sq_single_template(
            wl_arr, spec_arr, wl_grid_arr, z_grid_arr, one_plus_z,
            igm_trans_grid, apply_igm, filter_starts, filter_ends,
            filter_lengths, meas_photo_norm, inv_var,
        )

    # Find global minimum
    min_idx = np.unravel_index(np.argmin(chi_sq_grid), chi_sq_grid.shape)
    best_template_idx = min_idx[0]
    best_z_idx = min_idx[1]
    chi_sq_min = chi_sq_grid[best_template_idx, best_z_idx]
    best_z = z_grid[best_z_idx]
    best_type = template_names[best_template_idx]

    # Map interpolated template name back to base type for reporting
    if '_' in best_type:
        # Extract first template name from interpolated name
        best_type = best_type.split('_')[0]

    # Compute likelihood: L ∝ exp(-χ²/2)
    log_likelihood = -chi_sq_grid / 2.0

    # Marginalize over templates (log-sum-exp for numerical stability)
    max_log_L = np.max(log_likelihood, axis=0)
    pdf_unnorm = np.exp(max_log_L) * np.sum(
        np.exp(log_likelihood - max_log_L[np.newaxis, :]), axis=0
    )

    # Apply magnitude-dependent prior if requested
    if apply_prior and magnitude is not None:
        prior = magnitude_prior(z_grid, magnitude, band='I')
        pdf_unnorm = pdf_unnorm * prior

    # Normalize PDF
    pdf_integral = np.trapezoid(pdf_unnorm, z_grid)
    if pdf_integral > 0:
        pdf = pdf_unnorm / pdf_integral
    else:
        pdf = np.ones_like(z_grid) / (z_max - z_min)

    # Recompute best_z from PDF after applying prior
    if apply_prior and magnitude is not None:
        best_z_idx = np.argmax(pdf)
        best_z = z_grid[best_z_idx]

    # Compute cumulative distribution for percentiles
    cdf = np.zeros_like(pdf)
    cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(z_grid))
    cdf /= cdf[-1] if cdf[-1] > 0 else 1.0

    def percentile_from_cdf(p):
        idx = np.searchsorted(cdf, p)
        if idx == 0:
            return z_grid[0]
        if idx >= len(z_grid):
            return z_grid[-1]
        f = (p - cdf[idx - 1]) / (cdf[idx] - cdf[idx - 1] + 1e-10)
        return z_grid[idx - 1] + f * (z_grid[idx] - z_grid[idx - 1])

    z_lo = percentile_from_cdf(0.16)
    z_median = percentile_from_cdf(0.50)
    z_hi = percentile_from_cdf(0.84)

    # Compute ODDS parameter
    delta_z = 0.1 * (1 + best_z)
    z_lo_odds = max(z_min, best_z - delta_z)
    z_hi_odds = min(z_max, best_z + delta_z)
    mask_odds = (z_grid >= z_lo_odds) & (z_grid <= z_hi_odds)
    if np.any(mask_odds):
        odds = np.trapezoid(pdf[mask_odds], z_grid[mask_odds])
    else:
        odds = 0.0

    # Find secondary peak (for bimodal solutions)
    z_secondary, pdf_secondary = find_secondary_peak(z_grid, pdf, best_z)
    odds_secondary = None
    if z_secondary is not None:
        delta_z_sec = 0.1 * (1 + z_secondary)
        z_lo_sec = max(z_min, z_secondary - delta_z_sec)
        z_hi_sec = min(z_max, z_secondary + delta_z_sec)
        mask_sec = (z_grid >= z_lo_sec) & (z_grid <= z_hi_sec)
        if np.any(mask_sec):
            odds_secondary = np.trapezoid(pdf[mask_sec], z_grid[mask_sec])

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
    )


def classify_galaxy_batch_with_pdf(
    batch_data: list[tuple[int, list, list]],
    spectra_path: str,
    systematic_error_floor: float = 0.02,
) -> list[tuple[int, str, float, float, float, float, float]]:
    """
    Classify a batch of galaxies with PDF-based uncertainties.

    Optimized for parallel processing with reduced IPC overhead.

    Parameters
    ----------
    batch_data : list of tuples
        Each tuple is (idx, fluxes, errors)
    spectra_path : str
        Path to spectra directory
    systematic_error_floor : float
        Systematic error floor (default 2%)

    Returns
    -------
    list of tuples
        Each tuple is (idx, galaxy_type, redshift, z_lo, z_hi, chi_sq_min, odds)
    """
    # Pre-warm caches in this process
    _get_filter_slices()
    for gt in GALAXY_TYPES:
        _load_spectrum(spectra_path, gt)

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
