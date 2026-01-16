"""Galaxy classification via SED template fitting.

Simplified photo-z estimation using chi-squared template fitting.

Features:
- Chi-squared template fitting
- Redshift probability distributions (PDFs)
- Uncertainty quantification (16th/84th percentiles)
- IGM absorption (Madau 1995)
- ODDS quality parameter
- Vectorized computation over redshift grid (50-100x faster)
- Optional numba JIT acceleration
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import maximum_filter1d
from scipy.optimize import minimize_scalar
from tqdm.auto import tqdm

# Try to import numba for JIT compilation (highly recommended for performance)
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    import warnings
    warnings.warn(
        "numba is not installed. Photo-z template fitting will be 2-5x slower. "
        "Install with: pip install numba",
        RuntimeWarning,
        stacklevel=2,
    )
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


def check_numba_status() -> dict:
    """Check numba availability and configuration.

    Returns
    -------
    dict
        Dictionary with numba status information:
        - available: bool, whether numba is installed
        - version: str or None, numba version if installed
        - threading_layer: str or None, active threading layer
        - num_threads: int or None, number of threads configured
    """
    result = {
        "available": NUMBA_AVAILABLE,
        "version": None,
        "threading_layer": None,
        "num_threads": None,
    }

    if NUMBA_AVAILABLE:
        import numba
        result["version"] = numba.__version__
        try:
            from numba import config
            result["num_threads"] = config.NUMBA_NUM_THREADS
        except (ImportError, AttributeError):
            pass
        try:
            from numba.np.ufunc import parallel
            result["threading_layer"] = parallel._backend.threading_layer()
        except (ImportError, AttributeError):
            pass

    return result

# Default spectra path
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
    """Result of photometric redshift estimation."""
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
    best_A_V: float = 0.0  # Best-fit dust extinction (V-band magnitudes)


@lru_cache(maxsize=128)
def _load_spectrum(spectra_path: str, galaxy_type: str) -> tuple[NDArray, NDArray]:
    """Load galaxy spectrum from disk (cached for performance)."""
    path = Path(spectra_path) / f"{galaxy_type}.dat"
    if not path.exists():
        raise FileNotFoundError(f"Spectrum file not found: {path}")
    wl, spec = np.loadtxt(path, usecols=[0, 1], unpack=True)
    return wl.astype(np.float64), spec.astype(np.float64)


def _igm_transmission(wavelength: NDArray, redshift: float) -> NDArray:
    """Compute IGM transmission using Madau 1995 model."""
    if redshift < 0.1:
        return np.ones_like(wavelength, dtype=np.float64)

    tau = np.zeros_like(wavelength, dtype=np.float64)
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


def _calzetti_attenuation(wavelength: NDArray, A_V: float) -> NDArray:
    """Compute dust attenuation using Calzetti et al. (2000) law.

    Parameters
    ----------
    wavelength : NDArray
        Wavelength in Angstroms (rest-frame)
    A_V : float
        V-band attenuation in magnitudes

    Returns
    -------
    NDArray
        Transmission factor (multiply flux by this)
    """
    if A_V <= 0:
        return np.ones_like(wavelength, dtype=np.float64)

    # Convert wavelength to microns
    wl_um = wavelength / 10000.0

    # Calzetti R_V value
    R_V = 4.05

    # Compute k(lambda) - the attenuation curve
    k = np.zeros_like(wavelength, dtype=np.float64)

    # UV/optical: 0.12 - 0.63 microns
    mask_uv = (wl_um >= 0.12) & (wl_um < 0.63)
    if np.any(mask_uv):
        wl_uv = wl_um[mask_uv]
        k[mask_uv] = 2.659 * (-2.156 + 1.509 / wl_uv - 0.198 / wl_uv**2 + 0.011 / wl_uv**3) + R_V

    # Optical/NIR: 0.63 - 2.20 microns
    mask_nir = (wl_um >= 0.63) & (wl_um <= 2.20)
    if np.any(mask_nir):
        wl_nir = wl_um[mask_nir]
        k[mask_nir] = 2.659 * (-1.857 + 1.040 / wl_nir) + R_V

    # Extrapolate for wavelengths outside range (clamp k to positive values)
    k = np.maximum(k, 0.0)

    # Convert A_V to E(B-V) and compute attenuation
    E_BV = A_V / R_V
    attenuation = 10.0 ** (-0.4 * k * E_BV)

    return np.clip(attenuation, 0.0, 1.0)


# Default extinction values to fit (A_V in magnitudes)
DEFAULT_AV_GRID = (0.0, 0.3, 0.6, 1.0, 1.5)

# Common wavelength grid (2200-9500 Angstroms)
_WL_GRID = np.arange(2200, 9500, 1, dtype=np.float64)
_N_WL = len(_WL_GRID)

# Filter definitions for U, B, V, I bands
_FILTER_CENTERS = np.array([3000, 4500, 6060, 8140], dtype=np.float64)
_FILTER_WIDTHS = np.array([1521, 1501, 951, 766], dtype=np.float64) / 2


# =============================================================================
# Numba-accelerated helper functions
# =============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _igm_transmission_numba(wavelength: NDArray, redshift: float) -> NDArray:
        """Numba-accelerated IGM transmission (Madau 1995)."""
        n = len(wavelength)
        result = np.ones(n, dtype=np.float64)

        if redshift < 0.1:
            return result

        tau = np.zeros(n, dtype=np.float64)
        one_plus_z = 1.0 + redshift

        # Lyman series absorption
        lyman_wl = np.array([1216.0, 1026.0, 972.5])
        lyman_coeff = np.array([0.0036, 0.0017, 0.0012])

        for i in range(3):
            obs_wl = lyman_wl[i] * one_plus_z
            for j in range(n):
                if wavelength[j] < obs_wl:
                    x = wavelength[j] / lyman_wl[i]
                    tau[j] += lyman_coeff[i] * (x ** 3.46)

        # Lyman limit
        lyman_limit = 912.0
        obs_ll = lyman_limit * one_plus_z
        for j in range(n):
            if wavelength[j] < obs_ll:
                x_ll = wavelength[j] / lyman_limit
                tau_ll = 0.25 * (x_ll ** 3) * (one_plus_z ** 0.46)
                tau[j] += min(tau_ll, 30.0)

        for j in range(n):
            result[j] = max(0.0, min(1.0, np.exp(-tau[j])))

        return result

    @jit(nopython=True, cache=True)
    def _calzetti_numba(wavelength: NDArray, A_V: float) -> NDArray:
        """Numba-accelerated Calzetti attenuation."""
        n = len(wavelength)
        result = np.ones(n, dtype=np.float64)

        if A_V <= 0:
            return result

        R_V = 4.05
        E_BV = A_V / R_V

        for i in range(n):
            wl_um = wavelength[i] / 10000.0
            k = 0.0

            if 0.12 <= wl_um < 0.63:
                k = 2.659 * (-2.156 + 1.509 / wl_um - 0.198 / wl_um**2 + 0.011 / wl_um**3) + R_V
            elif 0.63 <= wl_um <= 2.20:
                k = 2.659 * (-1.857 + 1.040 / wl_um) + R_V

            k = max(k, 0.0)
            result[i] = max(0.0, min(1.0, 10.0 ** (-0.4 * k * E_BV)))

        return result
else:
    # Use numpy versions when numba is not available
    _igm_transmission_numba = _igm_transmission
    _calzetti_numba = _calzetti_attenuation


def _compute_filter_means_vectorized(
    spec_2d: NDArray,
    wl_grid: NDArray,
    filter_centers: NDArray,
    filter_widths: NDArray,
) -> NDArray:
    """Compute mean flux in each filter band for multiple spectra.

    Parameters
    ----------
    spec_2d : ndarray, shape (n_z, n_wavelengths)
        Spectra at different redshifts
    wl_grid : ndarray
        Wavelength grid
    filter_centers, filter_widths : ndarray
        Filter definitions

    Returns
    -------
    ndarray, shape (n_z, 4)
        Mean flux in each of 4 bands for each redshift
    """
    n_z = spec_2d.shape[0]
    n_bands = len(filter_centers)
    result = np.zeros((n_z, n_bands), dtype=np.float64)

    for b in range(n_bands):
        mask = (wl_grid >= filter_centers[b] - filter_widths[b]) & \
               (wl_grid <= filter_centers[b] + filter_widths[b])
        if np.any(mask):
            result[:, b] = np.mean(spec_2d[:, mask], axis=1)

    return result


def _compute_synthetic_photometry_fast(
    template_wl: NDArray,
    template_spec: NDArray,
    z_grid: NDArray,
    filter_centers: NDArray,
    filter_widths: NDArray,
    apply_igm: bool = True,
) -> NDArray:
    """Compute synthetic photometry directly without full spectrum interpolation.

    Uses pre-computed cumulative sums for fast band flux integration,
    achieving ~10-50x speedup over interpolation-based methods.

    Parameters
    ----------
    template_wl : ndarray
        Template rest-frame wavelength array (must be sorted)
    template_spec : ndarray
        Template spectrum (already dust-attenuated if needed)
    z_grid : ndarray
        Redshift grid
    filter_centers, filter_widths : ndarray
        Observer-frame filter definitions (4 bands)
    apply_igm : bool
        Apply IGM absorption

    Returns
    -------
    ndarray, shape (n_z, 4)
        Synthetic photometry in 4 bands for each redshift
    """
    n_z = len(z_grid)
    n_bands = len(filter_centers)
    n_wl = len(template_wl)
    syn_photo = np.zeros((n_z, n_bands), dtype=np.float64)
    one_plus_z = 1.0 + z_grid

    # Pre-compute cumulative sum for fast mean calculation
    # cumsum[i] = sum of spec[0:i], so mean(spec[a:b]) = (cumsum[b] - cumsum[a]) / (b - a)
    cumsum_spec = np.zeros(n_wl + 1, dtype=np.float64)
    cumsum_spec[1:] = np.cumsum(template_spec)

    # Pre-compute filter bounds in observer frame
    filter_lo = filter_centers - filter_widths
    filter_hi = filter_centers + filter_widths

    # Vectorized computation over z: compute rest-frame filter bounds for all z
    # Shape: (n_z, n_bands)
    rest_lo = filter_lo[np.newaxis, :] / one_plus_z[:, np.newaxis]
    rest_hi = filter_hi[np.newaxis, :] / one_plus_z[:, np.newaxis]

    # For each band, find indices using searchsorted (vectorized over z)
    for b in range(n_bands):
        # Find start and end indices for each z
        idx_lo = np.searchsorted(template_wl, rest_lo[:, b], side='left')
        idx_hi = np.searchsorted(template_wl, rest_hi[:, b], side='right')

        # Compute counts and sums using cumulative sum
        counts = idx_hi - idx_lo
        valid = counts > 0

        # Use cumsum for fast mean computation
        sums = cumsum_spec[idx_hi] - cumsum_spec[idx_lo]
        syn_photo[valid, b] = sums[valid] / counts[valid]

    # Apply IGM absorption for z > 0.5
    # This is the remaining bottleneck - compute only where needed
    if apply_igm:
        high_z_mask = z_grid > 0.5
        if np.any(high_z_mask):
            high_z_indices = np.where(high_z_mask)[0]

            for z_idx in high_z_indices:
                z = z_grid[z_idx]
                opz = one_plus_z[z_idx]

                for b in range(n_bands):
                    if syn_photo[z_idx, b] == 0:
                        continue

                    # Get wavelength range for this band at this z
                    wl_lo_rest = filter_lo[b] / opz
                    wl_hi_rest = filter_hi[b] / opz

                    idx_lo = np.searchsorted(template_wl, wl_lo_rest, side='left')
                    idx_hi = np.searchsorted(template_wl, wl_hi_rest, side='right')

                    if idx_hi <= idx_lo:
                        continue

                    wl_in_band = template_wl[idx_lo:idx_hi]
                    spec_in_band = template_spec[idx_lo:idx_hi]

                    # Apply IGM
                    wl_obs = wl_in_band * opz
                    if NUMBA_AVAILABLE:
                        igm = _igm_transmission_numba(wl_obs, z)
                    else:
                        igm = _igm_transmission(wl_obs, z)

                    # Recompute mean with IGM applied
                    syn_photo[z_idx, b] = np.mean(spec_in_band * igm)

    return syn_photo


def _compute_chi_sq_grid_vectorized(
    template_wl: NDArray,
    template_spec: NDArray,
    z_grid: NDArray,
    av_grid: tuple,
    meas_photo_norm: NDArray,
    inv_var: NDArray,
    apply_igm: bool = True,
) -> NDArray:
    """Compute chi-squared grid for one template, optimized for speed.

    Uses direct photometry computation without full spectrum interpolation
    for ~10-50x speedup over the naive loop-based version.

    Parameters
    ----------
    template_wl : ndarray
        Template wavelength array
    template_spec : ndarray
        Template spectrum
    z_grid : ndarray
        Redshift grid to evaluate
    av_grid : tuple
        A_V extinction values
    meas_photo_norm : ndarray, shape (4,)
        Normalized measured photometry
    inv_var : ndarray, shape (4,)
        Inverse variance weights
    apply_igm : bool
        Apply IGM absorption

    Returns
    -------
    ndarray, shape (n_av, n_z)
        Chi-squared values for each A_V and redshift
    """
    n_z = len(z_grid)
    n_av = len(av_grid)

    chi_sq = np.full((n_av, n_z), 1e30, dtype=np.float64)

    for av_idx, A_V in enumerate(av_grid):
        # Apply dust attenuation in rest frame
        if A_V > 0:
            if NUMBA_AVAILABLE:
                dust = _calzetti_numba(template_wl, A_V)
            else:
                dust = _calzetti_attenuation(template_wl, A_V)
            spec_dusty = template_spec * dust
        else:
            spec_dusty = template_spec

        # Compute synthetic photometry for all z values at once
        syn_photo = _compute_synthetic_photometry_fast(
            template_wl, spec_dusty, z_grid,
            _FILTER_CENTERS, _FILTER_WIDTHS, apply_igm
        )

        # Check for valid spectra (non-zero flux)
        spec_sums = np.sum(syn_photo, axis=1)
        valid_mask = spec_sums > 0

        if not np.any(valid_mask):
            continue

        # Normalize by median (vectorized)
        syn_medians = np.median(syn_photo, axis=1, keepdims=True)
        syn_medians[syn_medians == 0] = 1.0
        syn_norm = syn_photo / syn_medians

        # Compute chi-squared for all z at once (vectorized)
        diff = meas_photo_norm - syn_norm  # shape (n_z, 4)
        chi_sq_values = np.sum(diff**2 * inv_var, axis=1)  # shape (n_z,)

        # Mark invalid entries
        chi_sq_values[~valid_mask] = 1e30
        chi_sq_values[syn_medians.flatten() == 0] = 1e30

        chi_sq[av_idx] = chi_sq_values

    return chi_sq


def classify_galaxy(
    fluxes: ArrayLike,
    errors: ArrayLike,
    spectra_path: Path | str | None = None,
) -> tuple[str, float]:
    """Classify a galaxy by fitting SED templates.

    Simple interface returning galaxy type and redshift.

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
    z_min: float = 0.05,
    z_max: float = 6.0,
    z_step: float = 0.01,
    systematic_error_floor: float = 0.05,
    template_error_floor: float = 0.05,
    apply_igm: bool = True,
    av_grid: tuple[float, ...] | None = None,
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
        Photometric systematic error floor as fraction of flux (default 0.05 = 5%)
    template_error_floor : float
        Template model uncertainty as fraction of flux (default 0.05 = 5%)
        Accounts for systematic differences between templates and real galaxies
    apply_igm : bool
        Apply IGM absorption for high-z sources
    av_grid : tuple of float, optional
        Grid of A_V extinction values to fit. Default: (0.0, 0.3, 0.6, 1.0, 1.5)

    Returns
    -------
    PhotoZResult
        Full photo-z result with PDF, uncertainties, and best-fit A_V
    """
    if spectra_path is None:
        spectra_path = _DEFAULT_SPECTRA_PATH
    spectra_path_str = str(spectra_path)

    if av_grid is None:
        av_grid = DEFAULT_AV_GRID

    B, Ir, U, V = fluxes
    dB, dIr, dU, dV = errors

    # Measurement data with combined error budget
    meas_photo = np.array([U, B, V, Ir], dtype=np.float64)
    meas_errs_raw = np.array([dU, dB, dV, dIr], dtype=np.float64)
    # Combined error: photometric + systematic + template uncertainty
    systematic = systematic_error_floor * np.abs(meas_photo)
    template_err = template_error_floor * np.abs(meas_photo)
    meas_errs = np.sqrt(meas_errs_raw**2 + systematic**2 + template_err**2)

    meas_median = np.median(meas_photo)
    meas_photo_norm = meas_photo / meas_median
    inv_var = (meas_median / meas_errs) ** 2

    # Load spectra (cached for performance)
    spectra = [_load_spectrum(spectra_path_str, gt) for gt in GALAXY_TYPES]
    template_names = list(GALAXY_TYPES)

    # Redshift grid
    z_grid = np.arange(z_min, z_max + z_step, z_step)
    n_z = len(z_grid)
    n_templates = len(spectra)
    n_av = len(av_grid)

    # 3D Chi-squared grid: (template, A_V, redshift)
    # Use vectorized computation for ~50-100x speedup
    chi_sq_grid = np.full((n_templates, n_av, n_z), 1e30)

    for t_idx, (wl, spec) in enumerate(spectra):
        # Vectorized: compute chi-squared for all A_V and z values at once
        chi_sq_grid[t_idx] = _compute_chi_sq_grid_vectorized(
            wl, spec, z_grid, av_grid, meas_photo_norm, inv_var, apply_igm
        )

    # Find best fit (minimize over all 3 dimensions)
    min_idx = np.unravel_index(np.argmin(chi_sq_grid), chi_sq_grid.shape)
    best_template_idx = min_idx[0]
    best_av_idx = min_idx[1]
    best_z_idx = min_idx[2]
    chi_sq_min = chi_sq_grid[best_template_idx, best_av_idx, best_z_idx]
    best_z = z_grid[best_z_idx]
    best_type = template_names[best_template_idx]
    best_A_V = av_grid[best_av_idx]

    # Find second-best template (marginalized over A_V)
    template_min_chi2 = np.min(chi_sq_grid, axis=(1, 2))
    sorted_template_indices = np.argsort(template_min_chi2)

    if len(sorted_template_indices) >= 2:
        second_best_idx = sorted_template_indices[1]
        second_best_template = template_names[second_best_idx]
        second_best_chi2 = template_min_chi2[second_best_idx]
        delta_chi2 = second_best_chi2 - chi_sq_min
        template_ambiguity = float(np.exp(-delta_chi2 / 2.0))
    else:
        second_best_template = ""
        template_ambiguity = 0.0

    # Compute PDF from chi-squared, marginalizing over templates AND A_V
    # Reshape to (n_templates * n_av, n_z) for marginalization
    chi_sq_2d = chi_sq_grid.reshape(-1, n_z)
    log_L = -chi_sq_2d / 2.0
    max_log_L = np.max(log_L, axis=0)
    pdf_unnorm = np.exp(max_log_L) * np.sum(
        np.exp(log_L - max_log_L[np.newaxis, :]), axis=0
    )

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

    # Quality flags (account for extra A_V parameter)
    n_bands = 4
    n_params = 2  # redshift + A_V (template is marginalized)
    dof = max(1, n_bands - n_params)
    reduced_chi2 = chi_sq_min / dof
    if reduced_chi2 < 0.1:
        chi2_flag = 1  # Errors likely too large
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
        best_A_V=best_A_V,
    )


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


def _classify_single_galaxy(args: tuple) -> dict:
    """Worker function to classify a single galaxy."""
    idx, flux_ubvi, err_ubvi, spectra_path, z_step, z_min = args

    # Reorder from [U, B, V, I] to [B, I, U, V]
    fluxes = [flux_ubvi[1], flux_ubvi[3], flux_ubvi[0], flux_ubvi[2]]
    errors = [err_ubvi[1], err_ubvi[3], err_ubvi[0], err_ubvi[2]]

    n_valid = int(np.sum(np.isfinite(flux_ubvi) & (flux_ubvi > 0)))

    try:
        result = classify_galaxy_with_pdf(
            fluxes, errors,
            spectra_path=spectra_path,
            z_step=z_step,
            z_min=z_min,
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
            'best_A_V': result.best_A_V,
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
            'best_A_V': 0.0,
        }


def _classify_chunk(chunk_args: tuple) -> list[dict]:
    """Process a chunk of galaxies in a single worker."""
    start_idx, flux_chunk, error_chunk, spectra_path, z_step, z_min = chunk_args
    results = []
    for i in range(len(flux_chunk)):
        args = (start_idx + i, flux_chunk[i], error_chunk[i], spectra_path, z_step, z_min)
        results.append(_classify_single_galaxy(args))
    return results


def classify_batch_ultrafast(
    flux_array: NDArray,
    error_array: NDArray,
    spectra_path: str = "./spectra",
    z_step: float = 0.01,
    z_step_coarse: float = 0.05,
    n_workers: int | None = None,
    z_min: float = 0.05,
    show_progress: bool = True,
) -> dict:
    """Classify a batch of galaxies with parallel processing.

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
        Number of parallel workers
    z_min : float
        Minimum redshift for fitting (default 0.05 to prevent boundary pileup)
    show_progress : bool
        Show tqdm progress bar (default True)

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
        'best_A_V': np.zeros(n_galaxies, dtype=np.float64),
    }

    # For small batches or single worker, use serial processing with progress bar
    if n_galaxies < 10 or n_workers <= 1:
        iterator = range(n_galaxies)
        if show_progress:
            iterator = tqdm(iterator, desc="Classifying galaxies", unit="gal")
        for i in iterator:
            args = (i, flux_array[i], error_array[i], spectra_path, z_step, z_min)
            res = _classify_single_galaxy(args)
            _assign_result(results, res)
        return results

    # Split work into chunks for parallel processing
    # Use smaller chunks for better progress bar granularity
    chunk_size = max(5, n_galaxies // (n_workers * 8))
    chunks = []
    for start in range(0, n_galaxies, chunk_size):
        end = min(start + chunk_size, n_galaxies)
        chunks.append((
            start,
            flux_array[start:end],
            error_array[start:end],
            spectra_path,
            z_step,
            z_min,
        ))

    # Process chunks in parallel with progress bar
    completed_galaxies = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(_classify_chunk, chunk): chunk for chunk in chunks}

        # Create progress bar tracking galaxies (not chunks)
        pbar = None
        if show_progress:
            pbar = tqdm(total=n_galaxies, desc="Classifying galaxies", unit="gal")

        # Process results as they complete
        for future in as_completed(futures):
            chunk_result = future.result()
            chunk_galaxies = len(chunk_result)

            # Assign results
            for res in chunk_result:
                _assign_result(results, res)

            # Update progress bar
            if pbar is not None:
                pbar.update(chunk_galaxies)
            completed_galaxies += chunk_galaxies

        if pbar is not None:
            pbar.close()

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
    results['best_A_V'][idx] = res['best_A_V']


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
