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

import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import maximum_filter1d
from scipy.optimize import minimize_scalar

# Thread pool for parallel template fitting (reused across calls)
# Numba JIT releases the GIL, enabling true parallelism
_TEMPLATE_EXECUTOR: ThreadPoolExecutor | None = None
_N_TEMPLATE_WORKERS = 7  # Number of parallel workers for template fitting

# Optional numba for JIT compilation (50-100x speedup for IGM calculations)
try:
    import numba
    from numba import njit, prange
    numba.set_num_threads(_N_TEMPLATE_WORKERS)
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
# Template Grid Disk Cache (persists across runs for ~1-2s startup savings)
# =============================================================================
_CACHE_DIR = Path(__file__).parent / ".template_cache"


def _get_grid_cache_path(spectra_path: str, z_min: float, z_max: float, z_step: float) -> Path:
    """Generate a unique cache path based on grid parameters."""
    # Create hash of parameters for unique cache key
    key = f"{spectra_path}_{z_min}_{z_max}_{z_step}_{len(GALAXY_TYPES)}"
    cache_hash = hashlib.md5(key.encode()).hexdigest()[:12]
    return _CACHE_DIR / f"template_grid_{cache_hash}.pkl"


def _load_cached_grid(cache_path: Path) -> tuple | None:
    """Load template grid from disk cache if available and valid."""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except (pickle.PickleError, EOFError, OSError):
        # Corrupted cache file - remove it
        cache_path.unlink(missing_ok=True)
        return None


def _save_cached_grid(cache_path: Path, grid_data: tuple) -> None:
    """Save template grid to disk cache."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(grid_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError:
        pass  # Silently fail if we can't write cache


# =============================================================================
# Template Error Function (EAZY-style)
# =============================================================================

def template_error_function(
    wavelength: np.ndarray,
    scale: float = 0.05,
    reference_wl: float = 5500.0,
) -> np.ndarray:
    """Compute wavelength-dependent template error following EAZY methodology.

    The template error function accounts for systematic uncertainties in SED
    templates that vary with wavelength. This is critical for proper chi-squared
    weighting and prevents over-fitting to noisy UV/blue data.

    Parameters
    ----------
    wavelength : ndarray
        Wavelength array in Angstroms
    scale : float
        Base template error scale (default 0.05 = 5% uncertainty)
        EAZY uses 0.2 by default, but we use 0.05 for HST's better calibration
    reference_wl : float
        Reference wavelength (Angstroms) where error is minimal

    Returns
    -------
    ndarray
        Fractional template error at each wavelength

    Notes
    -----
    Following EAZY (Brammer et al. 2008), template errors are higher at:
    - UV wavelengths (uncertain dust attenuation, young stars)
    - Red/NIR wavelengths (TP-AGB stars, dust emission)

    The error is added in quadrature to photometric errors:
        sigma_total^2 = sigma_phot^2 + (f_template * flux)^2
    """
    # Linear error function: higher at UV and NIR
    # Minimum at optical (~5500 Å)
    wl = np.asarray(wavelength)
    delta_wl = (wl - reference_wl) / reference_wl

    # Asymmetric: larger errors in UV than NIR
    error = scale * (1.0 + 0.5 * np.abs(delta_wl))

    # Extra UV uncertainty (below 3000 Å)
    uv_mask = wl < 3000.0
    error[uv_mask] *= 1.5

    return error


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
    for _i, (lam_i, A_i) in enumerate(zip(lyman_wavelengths, A_laf, strict=False)):
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
    for _i, (lam_i, A_i) in enumerate(zip(lyman_wavelengths, A_dla, strict=False)):
        obs_lam = lam_i * one_plus_z
        mask = wavelength < obs_lam

        if np.any(mask):
            z_abs = wavelength[mask] / lam_i - 1

            # DLA optical depth
            tau_dla = A_i * (1 + z_abs) ** 2.0 if redshift < 2.0 else A_i * (1 + z_abs) ** 3.0

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

    Updated for z_max=6.0 to properly handle high-z sources in deep fields.

    Parameters
    ----------
    z_grid : NDArray
        Redshift grid (supports z up to 6.0)
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
    # Updated coefficients for extended z range based on GOODS/CANDELS

    if band == 'I':
        # I-band magnitude relation (updated for z<6)
        # z_m = z_0 + k * (m - m_0)
        z_0 = 0.4
        k = 0.12  # z increases ~0.12 per magnitude
        m_0 = 22.0  # Reference magnitude
        sigma_0 = 0.35  # Intrinsic scatter (slightly wider for high-z)
    else:
        # B-band (bluer, different relation)
        z_0 = 0.25
        k = 0.10
        m_0 = 24.0
        sigma_0 = 0.40

    # Median redshift for this magnitude
    z_median = max(0.05, z_0 + k * (magnitude - m_0))

    # Cap z_median at reasonable value (avoid extreme priors)
    z_median = min(z_median, 4.0)

    # Width increases with magnitude (fainter = more uncertain)
    sigma = sigma_0 * (1 + 0.08 * max(0, magnitude - m_0))

    # Gaussian prior centered on z_median
    prior = np.exp(-0.5 * ((z_grid - z_median) / sigma) ** 2)

    # Add extended power-law tail at high-z (galaxies at z>3 exist in deep fields)
    # Modified for z_max=6.0 to avoid cutting off real high-z sources
    with np.errstate(divide='ignore', invalid='ignore'):
        # Schechter-like high-z tail
        high_z_tail = np.where(
            z_grid > 0,
            (z_grid / z_median) ** (-1.5) * np.exp(-z_grid / 6.0),
            0.0
        )
    high_z_mask = z_grid > z_median
    prior[high_z_mask] = np.maximum(prior[high_z_mask], 0.15 * high_z_tail[high_z_mask])

    # Low-z cutoff (very few galaxies at z<0.01 in deep fields)
    prior[z_grid < 0.01] *= 0.001

    # Penalize z<0.1 for deep fields (peculiar velocities dominate, photo-z unreliable)
    # This is a soft penalty that reduces probability but doesn't exclude
    low_z_penalty = z_grid < 0.1
    prior[low_z_penalty] *= 0.3 * (1.0 + z_grid[low_z_penalty] / 0.1)  # Gradual ramp

    # Soft high-z cutoff (very few galaxies at z>5.5 in optical surveys)
    very_high_z = z_grid > 5.5
    prior[very_high_z] *= np.exp(-2 * (z_grid[very_high_z] - 5.5))

    # Normalize
    prior_sum = np.trapezoid(prior, z_grid)
    if prior_sum > 0:
        prior /= prior_sum

    return prior


def apply_magnitude_prior_batch(
    z_best: NDArray,
    z_lo: NDArray,
    z_hi: NDArray,
    chi_sq_min: NDArray,
    magnitudes: NDArray,
    z_grid: NDArray | None = None,
    band: str = 'I',
) -> dict[str, NDArray]:
    """
    Apply magnitude-based prior to batch photo-z results.

    This is a post-processing function that adjusts photo-z estimates
    using magnitude information. Sources with photo-z inconsistent with
    their magnitude get flagged.

    Parameters
    ----------
    z_best : NDArray
        Best-fit redshifts from batch processing
    z_lo, z_hi : NDArray
        Lower and upper confidence bounds
    chi_sq_min : NDArray
        Minimum chi-squared values
    magnitudes : NDArray
        I-band (or B-band) magnitudes for each source
    z_grid : NDArray, optional
        Redshift grid for prior calculation (default: 0-6 with step 0.01)
    band : str
        Photometric band ('I' or 'B')

    Returns
    -------
    dict
        Contains:
        - 'prior_weight': Weight from prior (0-1, higher = more consistent)
        - 'prior_flag': True if photo-z inconsistent with magnitude prior
        - 'z_prior_median': Expected redshift from magnitude alone
    """
    if z_grid is None:
        z_grid = np.arange(0.0, 6.01, 0.01)

    n_sources = len(z_best)
    prior_weights = np.zeros(n_sources, dtype=np.float64)
    prior_flags = np.zeros(n_sources, dtype=bool)
    z_prior_median = np.zeros(n_sources, dtype=np.float64)

    for i in range(n_sources):
        mag = magnitudes[i]
        if not np.isfinite(mag):
            prior_weights[i] = 1.0  # No prior available
            continue

        # Compute prior for this magnitude
        prior = magnitude_prior(z_grid, mag, band=band)

        # Find expected z from prior
        z_prior_median[i] = z_grid[np.argmax(prior)]

        # Evaluate prior at best-fit z
        z_idx = np.searchsorted(z_grid, z_best[i])
        z_idx = min(z_idx, len(prior) - 1)
        prior_at_z = prior[z_idx]

        # Prior weight is relative to peak
        prior_weights[i] = prior_at_z / np.max(prior) if np.max(prior) > 0 else 1.0

        # Flag if prior weight is very low (inconsistent)
        prior_flags[i] = prior_weights[i] < 0.1

    return {
        'prior_weight': prior_weights,
        'prior_flag': prior_flags,
        'z_prior_median': z_prior_median,
    }


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
) -> tuple[float | None, float | None]:
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
    """Result of photometric redshift estimation with uncertainties and quality flags.

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
    chi2_flag : int
        Chi-squared quality flag: 0=good (reduced chi2 < 5), 1=marginal (5-10), 2=poor (>10)
    odds_flag : int
        ODDS quality flag: 0=excellent (>0.9), 1=good (0.6-0.9), 2=poor (<0.6)
    bimodal_flag : bool
        True if significant secondary peak detected
    template_ambiguity : float
        Template confusion score (0-1): 1.0 = highly ambiguous (multiple templates fit equally)
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


def compute_chi2_flag(chi2_min: NDArray, n_bands: int = 4) -> NDArray:
    """Compute chi-squared quality flag from reduced chi-squared.

    Following COSMOS/EAZY conventions:
    - 0 (good): reduced chi2 < 5
    - 1 (marginal): 5 <= reduced chi2 < 10
    - 2 (poor): reduced chi2 >= 10

    Parameters
    ----------
    chi2_min : NDArray
        Minimum chi-squared values
    n_bands : int
        Number of photometric bands (degrees of freedom = n_bands - 1)

    Returns
    -------
    NDArray[int]
        Quality flags (0, 1, or 2)
    """
    chi2_min = np.atleast_1d(chi2_min)
    reduced_chi2 = chi2_min / max(1, n_bands - 1)
    flags = np.zeros(len(chi2_min), dtype=np.int32)
    flags[(reduced_chi2 >= 5) & (reduced_chi2 < 10)] = 1
    flags[reduced_chi2 >= 10] = 2
    return flags


def compute_odds_flag(odds: NDArray) -> NDArray:
    """Compute ODDS quality flag.

    Following BPZ conventions:
    - 0 (excellent): ODDS >= 0.9
    - 1 (good): 0.6 <= ODDS < 0.9
    - 2 (poor): ODDS < 0.6

    Parameters
    ----------
    odds : NDArray
        ODDS parameter values (0-1)

    Returns
    -------
    NDArray[int]
        Quality flags (0, 1, or 2)
    """
    odds = np.atleast_1d(odds)
    flags = np.zeros(len(odds), dtype=np.int32)
    flags[(odds >= 0.6) & (odds < 0.9)] = 1
    flags[odds < 0.6] = 2
    return flags


def compute_template_ambiguity(
    chi_sq_per_template: NDArray,
    template_names: list[str],
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute template confusion metrics.

    Parameters
    ----------
    chi_sq_per_template : NDArray
        Shape (n_sources, n_templates): minimum chi2 for each template
    template_names : list[str]
        Template names in order

    Returns
    -------
    template_ambiguity : NDArray[float]
        0-1 score: 1.0 = highly ambiguous (multiple templates fit equally well)
    second_best_template : NDArray[str]
        Second-best fitting template type
    delta_chi2 : NDArray[float]
        Chi2 difference between best and second-best templates
    """
    n_sources = chi_sq_per_template.shape[0]

    # Best and second-best per source
    sorted_indices = np.argsort(chi_sq_per_template, axis=1)
    best_idx = sorted_indices[:, 0]
    second_idx = sorted_indices[:, 1]

    best_chi2 = chi_sq_per_template[np.arange(n_sources), best_idx]
    second_chi2 = chi_sq_per_template[np.arange(n_sources), second_idx]

    delta_chi2 = second_chi2 - best_chi2

    # Ambiguity: 1 - normalized delta (high delta = low ambiguity)
    # Normalize: delta_chi2 < 2 is highly ambiguous, > 10 is not
    ambiguity = np.clip(1.0 - (delta_chi2 - 2.0) / 8.0, 0.0, 1.0)

    second_best = np.array([template_names[i] for i in second_idx])

    return ambiguity, second_best, delta_chi2


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
    for center, width in zip(filter_centers, filter_widths, strict=False):
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
    for center, width in zip(filter_centers, filter_widths, strict=False):
        filter_spec = np.zeros(len(_WL_GRID), dtype=np.float32)
        mask = ((center - width) <= _WL_GRID) & ((center + width) >= _WL_GRID)
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


def _get_template_executor() -> ThreadPoolExecutor:
    """Get or create thread pool for parallel template fitting.

    Numba JIT releases the GIL, enabling true parallelism across templates.
    Thread pool is reused to avoid creation overhead.
    """
    global _TEMPLATE_EXECUTOR
    if _TEMPLATE_EXECUTOR is None:
        _TEMPLATE_EXECUTOR = ThreadPoolExecutor(max_workers=_N_TEMPLATE_WORKERS)
    return _TEMPLATE_EXECUTOR


def _fit_single_template(
    t_idx: int,
    wl: NDArray,
    spec: NDArray,
    wl_grid_arr: NDArray,
    z_grid_arr: NDArray,
    one_plus_z: NDArray,
    igm_trans_grid: NDArray,
    apply_igm: bool,
    filter_starts: NDArray,
    filter_ends: NDArray,
    filter_lengths: NDArray,
    meas_photo_norm: NDArray,
    inv_var: NDArray,
) -> tuple[int, NDArray]:
    """Fit a single template across all redshifts (for parallel execution)."""
    wl_arr = np.ascontiguousarray(wl, dtype=np.float64)
    spec_arr = np.ascontiguousarray(spec, dtype=np.float64)
    chi_sq = _compute_chi_sq_single_template(
        wl_arr, spec_arr, wl_grid_arr, z_grid_arr, one_plus_z,
        igm_trans_grid, apply_igm, filter_starts, filter_ends,
        filter_lengths, meas_photo_norm, inv_var,
    )
    return t_idx, chi_sq


def classify_galaxy_with_pdf(
    fluxes: ArrayLike,
    errors: ArrayLike,
    spectra_path: Path | str | None = None,
    z_min: float = 0.0,
    z_max: float = 6.0,
    z_step: float = 0.01,
    systematic_error_floor: float = 0.02,
    apply_igm: bool = True,
    igm_model: str = "madau95",
    apply_prior: bool = False,
    magnitude: float | None = None,
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

    # Add systematic error floor: sigma_total^2 = sigma_meas^2 + (f_sys * flux)^2
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
        # Minimal dummy grid - saves ~20MB memory since it won't be accessed
        # JIT function checks apply_igm flag before accessing this array
        igm_trans_grid = np.ones((1, 1), dtype=np.float64)

    # Extract filter slices as separate arrays for JIT function
    filter_starts = np.array([s for s, e, _len in filter_slices], dtype=np.int64)
    filter_ends = np.array([e for s, e, _len in filter_slices], dtype=np.int64)
    filter_lengths = np.array([flen for _s, _e, flen in filter_slices], dtype=np.float64)

    # Parallel template fitting using thread pool (numba releases GIL)
    # Only use parallelism when there are enough templates to justify overhead
    if n_templates >= 6 and NUMBA_AVAILABLE:
        executor = _get_template_executor()
        futures = [
            executor.submit(
                _fit_single_template,
                t_idx, wl, spec,
                wl_grid_arr, z_grid_arr, one_plus_z,
                igm_trans_grid, apply_igm, filter_starts, filter_ends,
                filter_lengths, meas_photo_norm, inv_var,
            )
            for t_idx, (wl, spec) in enumerate(spectra_list)
        ]
        for future in futures:
            t_idx, chi_sq = future.result()
            chi_sq_grid[t_idx] = chi_sq
    else:
        # Sequential fallback for small template sets or no numba
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
    pdf = pdf_unnorm / pdf_integral if pdf_integral > 0 else np.ones_like(z_grid) / (z_max - z_min)

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
    odds = np.trapezoid(pdf[mask_odds], z_grid[mask_odds]) if np.any(mask_odds) else 0.0

    # Find secondary peak (for bimodal solutions)
    z_secondary, _pdf_secondary = find_secondary_peak(z_grid, pdf, best_z)
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


# =============================================================================
# CRAZY FAST: Fully Vectorized Batch Processing
# =============================================================================

@lru_cache(maxsize=1)
def _precompute_template_photometry_grid(
    spectra_path: str,
    z_min: float = 0.0,
    z_max: float = 6.0,
    z_step: float = 0.01,
) -> tuple[NDArray, NDArray, list[str]]:
    """Pre-compute synthetic photometry for all templates at all redshifts.

    This is the key to crazy fast performance - compute once, reuse for all sources.

    Returns
    -------
    template_photo_grid : NDArray
        Shape (n_templates, n_z, n_bands) - normalized synthetic photometry
    z_grid : NDArray
        Redshift grid
    template_names : list[str]
        Template names
    """
    z_grid = np.arange(z_min, z_max + z_step, z_step)
    n_z = len(z_grid)
    n_bands = 4  # U, B, V, I

    # Load templates
    spectra_list = [_load_spectrum(spectra_path, gt) for gt in GALAXY_TYPES]
    n_templates = len(spectra_list)

    # Get filter slices
    filter_slices = _get_filter_slices()

    # Pre-compute IGM absorption grid
    wl_grid_arr = np.ascontiguousarray(_WL_GRID, dtype=np.float64)
    z_grid_arr = np.ascontiguousarray(z_grid, dtype=np.float64)
    igm_trans_grid = igm_absorption_grid(wl_grid_arr, z_grid_arr, 0)  # madau95

    # Allocate output grid
    template_photo_grid = np.zeros((n_templates, n_z, n_bands), dtype=np.float64)

    # Compute synthetic photometry for each template at each redshift
    for t_idx, (wl, spec) in enumerate(spectra_list):
        for z_idx, z in enumerate(z_grid):
            # Redshift template
            wl_redshifted = wl * (1 + z)
            spec_interp = np.interp(_WL_GRID, wl_redshifted, spec, left=0, right=0)

            # Apply IGM
            if z > 0.5:
                spec_interp = spec_interp * igm_trans_grid[z_idx]

            # Normalize
            spec_mean = np.mean(spec_interp)
            if spec_mean > 0:
                spec_interp = spec_interp / spec_mean

            # Synthetic photometry
            for b_idx, (s, e, length) in enumerate(filter_slices):
                template_photo_grid[t_idx, z_idx, b_idx] = np.sum(spec_interp[s:e]) / length

        # Normalize each redshift slice by its median
        for z_idx in range(n_z):
            photo = template_photo_grid[t_idx, z_idx]
            photo_sorted = np.sort(photo)
            median = 0.5 * (photo_sorted[1] + photo_sorted[2])
            if median > 0:
                template_photo_grid[t_idx, z_idx] /= median

    return template_photo_grid, z_grid, list(GALAXY_TYPES)


@njit(parallel=True, cache=True, fastmath=True)
def _compute_chi_sq_all_sources(
    flux_array: np.ndarray,  # (n_sources, 4)
    inv_var_array: np.ndarray,  # (n_sources, 4)
    template_photo_grid: np.ndarray,  # (n_templates, n_z, 4)
) -> np.ndarray:
    """Compute chi-squared for ALL sources x ALL templates x ALL redshifts.

    This is the CRAZY FAST core - fully parallelized with numba.

    Returns
    -------
    chi_sq : NDArray
        Shape (n_sources, n_templates, n_z)
    """
    n_sources = flux_array.shape[0]
    n_templates = template_photo_grid.shape[0]
    n_z = template_photo_grid.shape[1]

    chi_sq = np.empty((n_sources, n_templates, n_z), dtype=np.float64)

    # Parallel loop over sources (main parallelization axis)
    for src_idx in prange(n_sources):
        flux = flux_array[src_idx]
        inv_var = inv_var_array[src_idx]

        # Normalize flux by median
        flux_sorted = np.sort(flux)
        flux_median = 0.5 * (flux_sorted[1] + flux_sorted[2])
        if flux_median <= 0:
            flux_median = 1.0

        flux_norm = flux / flux_median

        # Loop over templates and redshifts
        for t_idx in range(n_templates):
            for z_idx in range(n_z):
                template_photo = template_photo_grid[t_idx, z_idx]

                # Chi-squared
                chi_sq_val = 0.0
                for b in range(4):
                    diff = flux_norm[b] - template_photo[b]
                    chi_sq_val += diff * diff * inv_var[b]

                chi_sq[src_idx, t_idx, z_idx] = chi_sq_val

    return chi_sq


@njit(parallel=True, cache=True, fastmath=True)
def _extract_best_fits(
    chi_sq: np.ndarray,  # (n_sources, n_templates, n_z)
    z_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract best-fit parameters from chi-squared grid.

    Returns z_best, z_lo, z_hi, chi_sq_min, best_template_idx, odds
    """
    n_sources = chi_sq.shape[0]
    n_templates = chi_sq.shape[1]
    n_z = chi_sq.shape[2]

    z_best = np.empty(n_sources, dtype=np.float64)
    z_lo = np.empty(n_sources, dtype=np.float64)
    z_hi = np.empty(n_sources, dtype=np.float64)
    chi_sq_min = np.empty(n_sources, dtype=np.float64)
    best_template = np.empty(n_sources, dtype=np.int64)
    odds = np.empty(n_sources, dtype=np.float64)

    for src_idx in prange(n_sources):
        # Find global minimum across templates and redshifts
        min_chi = 1e30
        min_t = 0
        min_z_idx = 0

        for t_idx in range(n_templates):
            for z_idx in range(n_z):
                if chi_sq[src_idx, t_idx, z_idx] < min_chi:
                    min_chi = chi_sq[src_idx, t_idx, z_idx]
                    min_t = t_idx
                    min_z_idx = z_idx

        z_best[src_idx] = z_grid[min_z_idx]
        chi_sq_min[src_idx] = min_chi
        best_template[src_idx] = min_t

        # Compute PDF by marginalizing over templates
        # pdf[z] = sum_t exp(-chi_sq[t,z]/2)
        pdf = np.zeros(n_z, dtype=np.float64)
        max_log_L = -1e30

        for z_idx in range(n_z):
            for t_idx in range(n_templates):
                log_L = -chi_sq[src_idx, t_idx, z_idx] / 2.0
                if log_L > max_log_L:
                    max_log_L = log_L

        for z_idx in range(n_z):
            pdf_val = 0.0
            for t_idx in range(n_templates):
                log_L = -chi_sq[src_idx, t_idx, z_idx] / 2.0
                pdf_val += np.exp(log_L - max_log_L)
            pdf[z_idx] = pdf_val

        # Normalize PDF
        pdf_sum = 0.0
        for z_idx in range(n_z - 1):
            pdf_sum += 0.5 * (pdf[z_idx] + pdf[z_idx + 1]) * (z_grid[z_idx + 1] - z_grid[z_idx])
        if pdf_sum > 0:
            for z_idx in range(n_z):
                pdf[z_idx] /= pdf_sum

        # Compute CDF for percentiles
        cdf = np.zeros(n_z, dtype=np.float64)
        for z_idx in range(1, n_z):
            cdf[z_idx] = cdf[z_idx - 1] + 0.5 * (pdf[z_idx] + pdf[z_idx - 1]) * (z_grid[z_idx] - z_grid[z_idx - 1])
        if cdf[-1] > 0:
            for z_idx in range(n_z):
                cdf[z_idx] /= cdf[-1]

        # Find percentiles (16th, 84th)
        z_lo[src_idx] = z_grid[0]
        z_hi[src_idx] = z_grid[-1]
        for z_idx in range(n_z - 1):
            if cdf[z_idx] <= 0.16 <= cdf[z_idx + 1]:
                t = (0.16 - cdf[z_idx]) / (cdf[z_idx + 1] - cdf[z_idx] + 1e-10)
                z_lo[src_idx] = z_grid[z_idx] + t * (z_grid[z_idx + 1] - z_grid[z_idx])
            if cdf[z_idx] <= 0.84 <= cdf[z_idx + 1]:
                t = (0.84 - cdf[z_idx]) / (cdf[z_idx + 1] - cdf[z_idx] + 1e-10)
                z_hi[src_idx] = z_grid[z_idx] + t * (z_grid[z_idx + 1] - z_grid[z_idx])

        # Compute ODDS (integral within ±0.1(1+z) of peak)
        delta_z = 0.1 * (1 + z_best[src_idx])
        z_lo_odds = z_best[src_idx] - delta_z
        z_hi_odds = z_best[src_idx] + delta_z
        odds_val = 0.0
        for z_idx in range(n_z - 1):
            z_mid = 0.5 * (z_grid[z_idx] + z_grid[z_idx + 1])
            if z_lo_odds <= z_mid <= z_hi_odds:
                odds_val += 0.5 * (pdf[z_idx] + pdf[z_idx + 1]) * (z_grid[z_idx + 1] - z_grid[z_idx])
        odds[src_idx] = odds_val

    return z_best, z_lo, z_hi, chi_sq_min, best_template, odds


def classify_batch_vectorized(
    flux_array: NDArray,  # (n_sources, 4) in order [U, B, V, I]
    error_array: NDArray,  # (n_sources, 4)
    spectra_path: str | Path | None = None,
    systematic_error_floor: float = 0.02,
    z_min: float = 0.0,
    z_max: float = 6.0,
    z_step: float = 0.01,
) -> dict[str, NDArray]:
    """CRAZY FAST: Classify ALL sources at once with fully vectorized computation.

    This processes all sources simultaneously using pre-computed template grids
    and parallel numba loops. Expected speedup: 50-200x vs sequential processing.

    Parameters
    ----------
    flux_array : NDArray
        Shape (n_sources, 4) - fluxes in [U, B, V, I] order
    error_array : NDArray
        Shape (n_sources, 4) - errors in same order
    spectra_path : str or Path, optional
        Path to spectra directory
    systematic_error_floor : float
        Systematic error floor (default 2%)
    z_min, z_max, z_step : float
        Redshift grid parameters

    Returns
    -------
    dict with keys:
        - 'redshift': best-fit redshifts
        - 'z_lo': 16th percentile
        - 'z_hi': 84th percentile
        - 'chi_sq_min': minimum chi-squared
        - 'galaxy_type': best template names
        - 'odds': photo-z quality
        - 'chi2_flag': chi2 quality flags (0=good, 1=marginal, 2=poor)
        - 'odds_flag': ODDS quality flags (0=excellent, 1=good, 2=poor)
        - 'bimodal_flag': True if multiple templates fit equally well
        - 'template_ambiguity': template confusion score (0-1)
        - 'reduced_chi2': chi2 / degrees of freedom
        - 'second_best_template': second-best template type
        - 'delta_chi2_templates': chi2 difference to second-best
    """
    if spectra_path is None:
        spectra_path = _DEFAULT_SPECTRA_PATH
    spectra_path_str = str(spectra_path)

    # Ensure float64 contiguous arrays
    flux_array = np.ascontiguousarray(flux_array, dtype=np.float64)
    error_array = np.ascontiguousarray(error_array, dtype=np.float64)

    # Add systematic error floor
    systematic_errors = systematic_error_floor * np.abs(flux_array)
    total_errors = np.sqrt(error_array**2 + systematic_errors**2)

    # Compute inverse variance (normalized by median flux)
    # Use a tiny floor to prevent division by zero while preserving actual errors
    flux_medians = np.median(flux_array, axis=1, keepdims=True)
    flux_medians = np.where(flux_medians > 0, flux_medians, 1.0)
    inv_var_array = (flux_medians / np.maximum(total_errors, 1e-35)) ** 2

    # Get pre-computed template photometry grid (cached)
    template_photo_grid, z_grid, template_names = _precompute_template_photometry_grid(
        spectra_path_str, z_min, z_max, z_step
    )

    # CRAZY FAST: Compute chi-squared for ALL sources at once
    chi_sq = _compute_chi_sq_all_sources(flux_array, inv_var_array, template_photo_grid)

    # Extract best fits (also parallelized)
    z_best, z_lo, z_hi, chi_sq_min, best_template_idx, odds = _extract_best_fits(chi_sq, z_grid)

    # Map template indices to names
    galaxy_types = np.array([template_names[int(idx)] for idx in best_template_idx])

    # Compute quality flags
    chi2_flags = compute_chi2_flag(chi_sq_min, n_bands=4)
    odds_flags = compute_odds_flag(odds)

    # Compute reduced chi2
    n_bands = 4
    reduced_chi2 = chi_sq_min / max(1, n_bands - 1)

    # Compute template ambiguity (min chi2 per template for each source)
    chi_sq_per_template = np.min(chi_sq, axis=2)  # (n_sources, n_templates)
    ambiguity, second_best, delta_chi2 = compute_template_ambiguity(
        chi_sq_per_template, template_names
    )

    # Detect bimodal PDFs (simplified: check if there's a significant secondary template)
    # A source is "bimodal" if delta_chi2 < 2 (both templates fit well)
    bimodal_flags = delta_chi2 < 2.0

    # Flag sources that hit the redshift boundaries (fitting likely failed)
    # Tolerance of 0.01 for numerical precision
    z_at_min = z_best <= (z_min + 0.01)
    z_at_max = z_best >= (z_max - 0.01)
    z_boundary_flag = z_at_min | z_at_max

    return {
        'redshift': z_best,
        'z_lo': z_lo,
        'z_hi': z_hi,
        'chi_sq_min': chi_sq_min,
        'galaxy_type': galaxy_types,
        'odds': odds,
        'chi2_flag': chi2_flags,
        'odds_flag': odds_flags,
        'bimodal_flag': bimodal_flags,
        'template_ambiguity': ambiguity,
        'reduced_chi2': reduced_chi2,
        'second_best_template': second_best,
        'delta_chi2_templates': delta_chi2,
        'z_boundary_flag': z_boundary_flag,
    }


# =============================================================================
# ULTRA FAST: Float32 + Coarse-to-Fine + GPU Support
# =============================================================================

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


@lru_cache(maxsize=4)
def _precompute_template_grid_f32(
    spectra_path: str,
    z_min: float = 0.0,
    z_max: float = 6.0,
    z_step: float = 0.05,  # Coarser default for speed
) -> tuple[NDArray, NDArray, list[str]]:
    """Pre-compute template grid with float32 for 2x memory bandwidth.

    Uses two-level caching:
    1. LRU cache (in-memory) for repeated calls within a session
    2. Disk cache (.template_cache/) for persistence across runs (~1-2s savings)
    """
    # Check disk cache first (persists across runs)
    cache_path = _get_grid_cache_path(spectra_path, z_min, z_max, z_step)
    cached = _load_cached_grid(cache_path)
    if cached is not None:
        return cached

    # Compute the grid (expensive operation)
    z_grid = np.arange(z_min, z_max + z_step, z_step, dtype=np.float32)
    n_z = len(z_grid)

    spectra_list = [_load_spectrum(spectra_path, gt) for gt in GALAXY_TYPES]
    n_templates = len(spectra_list)
    filter_slices = _get_filter_slices()

    # Pre-compute IGM (use float64 for accuracy, convert later)
    wl_grid_arr = np.ascontiguousarray(_WL_GRID, dtype=np.float64)
    z_grid_64 = z_grid.astype(np.float64)
    igm_trans_grid = igm_absorption_grid(wl_grid_arr, z_grid_64, 0)

    template_photo_grid = np.zeros((n_templates, n_z, 4), dtype=np.float32)

    for t_idx, (wl, spec) in enumerate(spectra_list):
        for z_idx, z in enumerate(z_grid):
            wl_redshifted = wl * (1 + z)
            spec_interp = np.interp(_WL_GRID, wl_redshifted, spec, left=0, right=0)
            if z > 0.5:
                spec_interp = spec_interp * igm_trans_grid[z_idx]
            spec_mean = np.mean(spec_interp)
            if spec_mean > 0:
                spec_interp = spec_interp / spec_mean
            for b_idx, (s, e, length) in enumerate(filter_slices):
                template_photo_grid[t_idx, z_idx, b_idx] = np.sum(spec_interp[s:e]) / length

        for z_idx in range(n_z):
            photo = template_photo_grid[t_idx, z_idx]
            photo_sorted = np.sort(photo)
            median = 0.5 * (photo_sorted[1] + photo_sorted[2])
            if median > 0:
                template_photo_grid[t_idx, z_idx] /= median

    result = (template_photo_grid, z_grid, list(GALAXY_TYPES))

    # Save to disk cache for next run
    _save_cached_grid(cache_path, result)

    return result


@njit(parallel=True, cache=True, fastmath=True)
def _chi_sq_coarse_f32(
    flux_array: np.ndarray,
    inv_var_array: np.ndarray,
    template_photo_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ultra-fast coarse chi-squared with float32.

    Returns best_z_idx, best_t_idx, chi_sq_min, second_t_idx, second_chi_sq for each source.
    """
    n_sources = flux_array.shape[0]
    n_templates = template_photo_grid.shape[0]
    n_z = template_photo_grid.shape[1]

    best_z_idx = np.empty(n_sources, dtype=np.int32)
    best_t_idx = np.empty(n_sources, dtype=np.int32)
    chi_sq_min = np.empty(n_sources, dtype=np.float32)
    second_t_idx = np.empty(n_sources, dtype=np.int32)
    second_chi_sq = np.empty(n_sources, dtype=np.float32)

    for src_idx in prange(n_sources):
        flux = flux_array[src_idx]
        inv_var = inv_var_array[src_idx]

        # Normalize flux
        f0, f1, f2, f3 = flux[0], flux[1], flux[2], flux[3]
        if f0 > f1:
            f0, f1 = f1, f0
        if f2 > f3:
            f2, f3 = f3, f2
        if f0 > f2:
            f0, f2 = f2, f0
        if f1 > f3:
            f1, f3 = f3, f1
        if f1 > f2:
            f1, f2 = f2, f1
        flux_median = 0.5 * (f1 + f2)
        if flux_median <= 0:
            flux_median = 1.0

        fn0 = flux[0] / flux_median
        fn1 = flux[1] / flux_median
        fn2 = flux[2] / flux_median
        fn3 = flux[3] / flux_median

        iv0, iv1, iv2, iv3 = inv_var[0], inv_var[1], inv_var[2], inv_var[3]

        # Track best chi-squared per template (for second-best template calculation)
        min_chi_per_template = np.full(n_templates, np.float32(1e30), dtype=np.float32)

        min_chi = np.float32(1e30)
        min_t = 0
        min_z = 0

        for t_idx in range(n_templates):
            for z_idx in range(n_z):
                tp = template_photo_grid[t_idx, z_idx]
                d0 = fn0 - tp[0]
                d1 = fn1 - tp[1]
                d2 = fn2 - tp[2]
                d3 = fn3 - tp[3]
                chi = d0*d0*iv0 + d1*d1*iv1 + d2*d2*iv2 + d3*d3*iv3

                # Track best chi for this template
                if chi < min_chi_per_template[t_idx]:
                    min_chi_per_template[t_idx] = chi

                if chi < min_chi:
                    min_chi = chi
                    min_t = t_idx
                    min_z = z_idx

        # Find second-best template (different from best)
        second_min_chi = np.float32(1e30)
        second_min_t = 0
        for t_idx in range(n_templates):
            if t_idx != min_t and min_chi_per_template[t_idx] < second_min_chi:
                second_min_chi = min_chi_per_template[t_idx]
                second_min_t = t_idx

        best_z_idx[src_idx] = min_z
        best_t_idx[src_idx] = min_t
        chi_sq_min[src_idx] = min_chi
        second_t_idx[src_idx] = second_min_t
        second_chi_sq[src_idx] = second_min_chi

    return best_z_idx, best_t_idx, chi_sq_min, second_t_idx, second_chi_sq


@njit(parallel=True, cache=True, fastmath=True)
def _refine_redshift_f32(
    flux_array: np.ndarray,
    inv_var_array: np.ndarray,
    template_photo_grid_fine: np.ndarray,
    coarse_z_idx: np.ndarray,
    coarse_t_idx: np.ndarray,
    coarse_z_grid: np.ndarray,
    fine_z_grid: np.ndarray,
    refine_window: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Refine redshift around coarse best fit."""
    n_sources = flux_array.shape[0]
    n_z_fine = len(fine_z_grid)

    z_best = np.empty(n_sources, dtype=np.float32)
    z_lo = np.empty(n_sources, dtype=np.float32)
    z_hi = np.empty(n_sources, dtype=np.float32)
    chi_sq_min = np.empty(n_sources, dtype=np.float32)

    for src_idx in prange(n_sources):
        flux = flux_array[src_idx]
        inv_var = inv_var_array[src_idx]
        t_idx = coarse_t_idx[src_idx]
        z_coarse = coarse_z_grid[coarse_z_idx[src_idx]]

        # Normalize flux (unrolled for speed)
        f0, f1, f2, f3 = flux[0], flux[1], flux[2], flux[3]
        if f0 > f1:
            f0, f1 = f1, f0
        if f2 > f3:
            f2, f3 = f3, f2
        if f0 > f2:
            f0, f2 = f2, f0
        if f1 > f3:
            f1, f3 = f3, f1
        if f1 > f2:
            f1, f2 = f2, f1
        flux_median = 0.5 * (f1 + f2)
        if flux_median <= 0:
            flux_median = 1.0

        fn0 = flux[0] / flux_median
        fn1 = flux[1] / flux_median
        fn2 = flux[2] / flux_median
        fn3 = flux[3] / flux_median
        iv0, iv1, iv2, iv3 = inv_var[0], inv_var[1], inv_var[2], inv_var[3]

        # Search in refinement window
        z_lo_search = z_coarse - refine_window
        z_hi_search = z_coarse + refine_window

        min_chi = np.float32(1e30)
        best_z = z_coarse

        # Compute chi-sq and PDF simultaneously
        pdf = np.zeros(n_z_fine, dtype=np.float32)
        max_log_L = np.float32(-1e30)

        for z_idx in range(n_z_fine):
            z = fine_z_grid[z_idx]
            if z < z_lo_search or z > z_hi_search:
                pdf[z_idx] = 0.0
                continue

            tp = template_photo_grid_fine[t_idx, z_idx]
            d0 = fn0 - tp[0]
            d1 = fn1 - tp[1]
            d2 = fn2 - tp[2]
            d3 = fn3 - tp[3]
            chi = d0*d0*iv0 + d1*d1*iv1 + d2*d2*iv2 + d3*d3*iv3

            log_L = -chi / 2.0
            if log_L > max_log_L:
                max_log_L = log_L

            if chi < min_chi:
                min_chi = chi
                best_z = z

        # Compute PDF
        for z_idx in range(n_z_fine):
            z = fine_z_grid[z_idx]
            if z < z_lo_search or z > z_hi_search:
                continue
            tp = template_photo_grid_fine[t_idx, z_idx]
            d0 = fn0 - tp[0]
            d1 = fn1 - tp[1]
            d2 = fn2 - tp[2]
            d3 = fn3 - tp[3]
            chi = d0*d0*iv0 + d1*d1*iv1 + d2*d2*iv2 + d3*d3*iv3
            pdf[z_idx] = np.exp(-chi/2.0 - max_log_L)

        # Normalize and compute percentiles
        pdf_sum = np.float32(0.0)
        for z_idx in range(n_z_fine - 1):
            pdf_sum += 0.5 * (pdf[z_idx] + pdf[z_idx + 1]) * (fine_z_grid[z_idx + 1] - fine_z_grid[z_idx])

        if pdf_sum > 0:
            for z_idx in range(n_z_fine):
                pdf[z_idx] /= pdf_sum

        # CDF for percentiles
        cdf = np.float32(0.0)
        z_16 = fine_z_grid[0]
        z_84 = fine_z_grid[-1]
        for z_idx in range(n_z_fine - 1):
            cdf_prev = cdf
            cdf += 0.5 * (pdf[z_idx] + pdf[z_idx + 1]) * (fine_z_grid[z_idx + 1] - fine_z_grid[z_idx])
            if cdf_prev <= 0.16 <= cdf:
                t = (0.16 - cdf_prev) / (cdf - cdf_prev + 1e-10)
                z_16 = fine_z_grid[z_idx] + t * (fine_z_grid[z_idx + 1] - fine_z_grid[z_idx])
            if cdf_prev <= 0.84 <= cdf:
                t = (0.84 - cdf_prev) / (cdf - cdf_prev + 1e-10)
                z_84 = fine_z_grid[z_idx] + t * (fine_z_grid[z_idx + 1] - fine_z_grid[z_idx])

        z_best[src_idx] = best_z
        z_lo[src_idx] = z_16
        z_hi[src_idx] = z_84
        chi_sq_min[src_idx] = min_chi

    return z_best, z_lo, z_hi, chi_sq_min


def classify_batch_ultrafast(
    flux_array: NDArray,
    error_array: NDArray,
    spectra_path: str | Path | None = None,
    systematic_error_floor: float = 0.02,
) -> dict[str, NDArray]:
    """ULTRA FAST: Coarse-to-fine search with float32.

    2-stage approach:
    1. Coarse search (z_step=0.05) to find approximate best fit
    2. Fine refinement (z_step=0.01) around best fit

    Expected speedup: 5-10x over single-pass fine grid.

    Returns
    -------
    dict with keys:
        - 'redshift': best-fit redshifts
        - 'z_lo': 16th percentile
        - 'z_hi': 84th percentile
        - 'chi_sq_min': minimum chi-squared
        - 'galaxy_type': best template names
        - 'odds': photo-z quality
        - 'chi2_flag': chi2 quality flags (0=good, 1=marginal, 2=poor)
        - 'odds_flag': ODDS quality flags (0=excellent, 1=good, 2=poor)
        - 'bimodal_flag': True if low ODDS suggests multiple solutions
        - 'template_ambiguity': approximate template confusion (1 - odds)
        - 'reduced_chi2': chi2 / degrees of freedom
        - 'second_best_template': empty (not available in ultrafast mode)
        - 'delta_chi2_templates': zeros (not available in ultrafast mode)
        - 'n_valid_bands': number of valid (non-NaN) bands per source
    """
    if spectra_path is None:
        spectra_path = _DEFAULT_SPECTRA_PATH
    spectra_path_str = str(spectra_path)

    # Convert to float64 first to handle NaN properly, then to float32
    flux_array = np.asarray(flux_array, dtype=np.float64)
    error_array = np.asarray(error_array, dtype=np.float64)

    # Handle NaN/Inf values: replace with zero flux and very large error
    # This effectively removes that band from the chi-squared calculation
    # by giving it zero weight (inv_var → 0)
    nan_mask = ~np.isfinite(flux_array) | ~np.isfinite(error_array)
    n_valid_bands = np.sum(~nan_mask, axis=1)  # Count valid bands per source

    # Replace NaN with zero flux and large error (will get zero weight)
    flux_array = np.where(nan_mask, 0.0, flux_array)
    error_array = np.where(nan_mask, 1e30, error_array)  # Very large error → zero weight

    # Convert to float32 for 2x memory bandwidth
    flux_array = np.ascontiguousarray(flux_array, dtype=np.float32)
    error_array = np.ascontiguousarray(error_array, dtype=np.float32)

    # Add band-dependent systematic error floor (EAZY-style template error)
    # Bands are [U, B, V, I] with central wavelengths [3000, 4500, 6060, 8140] Angstroms
    # UV band gets higher error due to uncertain dust attenuation and young star contributions
    band_error_scale = np.array([0.08, 0.04, 0.03, 0.03], dtype=np.float32)  # Higher for U-band
    systematic_errors = band_error_scale * np.abs(flux_array)
    total_errors = np.sqrt(error_array**2 + systematic_errors**2)

    # Compute inverse variance (with overflow protection)
    # IMPORTANT: Use float64 for this calculation to avoid precision issues
    # with very small astronomical flux values (~1e-20 erg/s/cm²/Hz)
    flux_medians_f64 = np.median(flux_array.astype(np.float64), axis=1, keepdims=True)
    flux_medians_f64 = np.where(flux_medians_f64 > 0, flux_medians_f64, 1.0)
    total_errors_f64 = total_errors.astype(np.float64)
    # Use a tiny floor (1e-35) to prevent division by zero while preserving
    # the actual error values which are typically ~1e-20
    ratio = np.clip(flux_medians_f64 / np.maximum(total_errors_f64, 1e-35), 0, 1e6)
    inv_var_array = (ratio ** 2).astype(np.float32)

    # Stage 1: Coarse search (z_step=0.05, ~120 points for z=0-6)
    coarse_grid, coarse_z, template_names = _precompute_template_grid_f32(
        spectra_path_str, z_min=0.0, z_max=6.0, z_step=0.05
    )
    coarse_z_idx, coarse_t_idx, coarse_chi_min, second_t_idx, second_chi_sq = _chi_sq_coarse_f32(
        flux_array, inv_var_array.astype(np.float32), coarse_grid
    )

    # Stage 2: Fine refinement (z_step=0.01 around best fit)
    fine_grid, fine_z, _ = _precompute_template_grid_f32(
        spectra_path_str, z_min=0.0, z_max=6.0, z_step=0.01
    )
    z_best, z_lo, z_hi, chi_sq_min = _refine_redshift_f32(
        flux_array, inv_var_array.astype(np.float32),
        fine_grid, coarse_z_idx, coarse_t_idx,
        coarse_z, fine_z, refine_window=0.15
    )

    # Map template indices to names
    galaxy_types = np.array([template_names[int(idx)] for idx in coarse_t_idx])
    second_best_templates = np.array([template_names[int(idx)] for idx in second_t_idx])

    # Compute delta chi2 between best and second-best templates
    delta_chi2_templates = second_chi_sq - coarse_chi_min

    # Compute simple odds (approximate)
    odds = np.minimum(1.0, 0.5 / (z_hi - z_lo + 0.01))

    # Compute quality flags
    chi_sq_min_f64 = chi_sq_min.astype(np.float64)
    odds_f64 = odds.astype(np.float64)

    # Use actual valid bands for chi2 flag calculation (not fixed 4)
    chi2_flags = compute_chi2_flag(chi_sq_min_f64, n_bands=4)  # Conservative: assume 4 bands
    odds_flags = compute_odds_flag(odds_f64)

    # Compute reduced chi2 using actual number of valid bands per source
    # dof = n_valid_bands - 1 (fitting one normalization parameter)
    dof = np.maximum(n_valid_bands - 1, 1)  # At least 1 dof
    reduced_chi2 = chi_sq_min_f64 / dof

    # Flag sources with too few bands for reliable photo-z (need at least 3)
    insufficient_bands = n_valid_bands < 3

    # Simplified bimodal detection based on ODDS
    # Low odds suggests multiple solutions, approximating bimodality
    bimodal_flags = odds_f64 < 0.5

    # Template ambiguity: high delta_chi2 = low ambiguity, low delta_chi2 = high ambiguity
    # Normalize: delta_chi2 < 2 is highly ambiguous (both templates fit well), > 10 is not
    template_ambiguity = np.clip(1.0 - (delta_chi2_templates - 2.0) / 8.0, 0.0, 1.0)

    # Flag sources that hit the redshift boundaries (fitting likely failed)
    # OR have too few bands for reliable fitting
    # Ultrafast mode uses z_min=0.0, z_max=6.0
    z_at_min = z_best <= 0.01
    z_at_max = z_best >= 5.99
    z_boundary_flag = z_at_min | z_at_max | insufficient_bands

    return {
        'redshift': z_best.astype(np.float64),
        'z_lo': z_lo.astype(np.float64),
        'z_hi': z_hi.astype(np.float64),
        'chi_sq_min': chi_sq_min_f64,
        'galaxy_type': galaxy_types,
        'odds': odds_f64,
        'chi2_flag': chi2_flags,
        'odds_flag': odds_flags,
        'bimodal_flag': bimodal_flags,
        'template_ambiguity': template_ambiguity,
        'reduced_chi2': reduced_chi2,
        'second_best_template': second_best_templates,
        'delta_chi2_templates': delta_chi2_templates.astype(np.float64),
        'z_boundary_flag': z_boundary_flag,
        'n_valid_bands': n_valid_bands,
    }


# GPU-accelerated version (if CuPy available)
if HAS_CUPY:
    def classify_batch_gpu(
        flux_array: NDArray,
        error_array: NDArray,
        spectra_path: str | Path | None = None,
        systematic_error_floor: float = 0.02,
    ) -> dict[str, NDArray]:
        """GPU-accelerated classification using CuPy.

        Moves computation to GPU for massive parallelism.
        Falls back to CPU if GPU memory is insufficient.
        """
        if spectra_path is None:
            spectra_path = _DEFAULT_SPECTRA_PATH
        spectra_path_str = str(spectra_path)

        n_sources = flux_array.shape[0]

        # Get template grid (computed on CPU, transferred to GPU)
        template_grid_cpu, z_grid_cpu, template_names = _precompute_template_grid_f32(
            spectra_path_str, z_min=0.0, z_max=6.0, z_step=0.02
        )

        try:
            # Transfer to GPU
            flux_gpu = cp.asarray(flux_array, dtype=cp.float32)
            error_gpu = cp.asarray(error_array, dtype=cp.float32)
            template_grid_gpu = cp.asarray(template_grid_cpu, dtype=cp.float32)
            z_grid_gpu = cp.asarray(z_grid_cpu, dtype=cp.float32)

            # Add systematic error
            systematic = systematic_error_floor * cp.abs(flux_gpu)
            total_error = cp.sqrt(error_gpu**2 + systematic**2)

            # Normalize flux
            flux_median = cp.median(flux_gpu, axis=1, keepdims=True)
            flux_median = cp.where(flux_median > 0, flux_median, 1.0)
            flux_norm = flux_gpu / flux_median
            inv_var = (flux_median / total_error) ** 2

            # Compute chi-squared: (n_sources, n_templates, n_z)
            # Using broadcasting for GPU parallelism
            _n_templates, n_z, _n_bands = template_grid_gpu.shape

            # Reshape for broadcasting
            flux_exp = flux_norm[:, cp.newaxis, cp.newaxis, :]  # (n_src, 1, 1, 4)
            inv_var_exp = inv_var[:, cp.newaxis, cp.newaxis, :]  # (n_src, 1, 1, 4)
            template_exp = template_grid_gpu[cp.newaxis, :, :, :]  # (1, n_t, n_z, 4)

            # Chi-squared computation (fully vectorized on GPU)
            diff = flux_exp - template_exp
            chi_sq = cp.sum(diff * diff * inv_var_exp, axis=3)  # (n_src, n_t, n_z)

            # Find minimum
            chi_sq_flat = chi_sq.reshape(n_sources, -1)
            min_idx = cp.argmin(chi_sq_flat, axis=1)
            best_t_idx = min_idx // n_z
            best_z_idx = min_idx % n_z

            z_best = z_grid_gpu[best_z_idx]
            chi_sq_min = cp.min(chi_sq_flat, axis=1)

            # Simple uncertainty estimate
            z_lo = cp.maximum(0.0, z_best - 0.1)
            z_hi = cp.minimum(3.5, z_best + 0.1)

            # Transfer back to CPU
            return {
                'redshift': cp.asnumpy(z_best).astype(np.float64),
                'z_lo': cp.asnumpy(z_lo).astype(np.float64),
                'z_hi': cp.asnumpy(z_hi).astype(np.float64),
                'chi_sq_min': cp.asnumpy(chi_sq_min).astype(np.float64),
                'galaxy_type': np.array([template_names[int(i)] for i in cp.asnumpy(best_t_idx)]),
                'odds': np.ones(n_sources),  # Simplified
            }

        except cp.cuda.memory.OutOfMemoryError:
            # Fall back to CPU
            return classify_batch_ultrafast(flux_array, error_array, spectra_path, systematic_error_floor)


def classify_batch_fastest(
    flux_array: NDArray,
    error_array: NDArray,
    spectra_path: str | Path | None = None,
    systematic_error_floor: float = 0.02,
) -> dict[str, NDArray]:
    """Automatically select fastest available method.

    Priority: GPU > UltraFast (coarse-to-fine) > Vectorized
    """
    if HAS_CUPY:
        try:
            return classify_batch_gpu(flux_array, error_array, spectra_path, systematic_error_floor)
        except Exception:
            pass

    return classify_batch_ultrafast(flux_array, error_array, spectra_path, systematic_error_floor)
