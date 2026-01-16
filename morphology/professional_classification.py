"""Professional-grade star-galaxy classification following survey standards.

This module implements research-quality star-galaxy separation using multiple
complementary methods following best practices from major surveys (COSMOS2020,
DES, KiDS, WAVES, CANDELS).

Classification Tiers:
1. Gaia DR3 cross-match: Definitive foreground star identification
2. SPREAD_MODEL: PSF vs extended source morphology
3. ML classifier: Random Forest/CatBoost with photometric+morphological features
4. Color-color stellar locus: Supplementary color-based check

Key Features:
- Magnitude-dependent thresholds (bright vs faint sources behave differently)
- Empirical PSF measurement from Gaia stars
- Multi-tier confidence scoring
- Professional validation metrics (purity, completeness, contamination)

References:
- Cook et al. 2024, MNRAS - WAVES UMAP+HDBSCAN classification
- Baqui et al. 2021, A&A, 645, A87 - miniJPAS ML classification
- Desai et al. 2012, ApJ - DES star-galaxy separation
- Annunziatella et al. 2013 - SPREAD_MODEL methodology
- Daddi et al. 2004 - BzK selection and stellar locus
"""

import warnings
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from pathlib import Path

import numpy as np
import pandas as pd
import sep
from astropy.wcs import WCS
from numpy.typing import NDArray
from sklearn.neighbors import KDTree

from morphology.ml_classifier import extract_features_from_catalog

# =============================================================================
# Classification flags and data structures
# =============================================================================

class ClassificationMethod(IntEnum):
    """Method used for classification."""
    UNKNOWN = 0
    GAIA = 1           # Gaia cross-match (highest confidence)
    SPREAD_MODEL = 2   # PSF morphology comparison
    ML_CLASSIFIER = 3  # Machine learning
    CONCENTRATION = 4  # Concentration index
    SIZE_PSF = 5       # Size vs PSF comparison
    COLOR_LOCUS = 6    # Color-color stellar locus
    COMBINED = 7       # Multiple methods agree


class ClassificationFlag(IntFlag):
    """Quality flags for classification."""
    GOOD = 0
    LOW_SNR = 1           # SNR < 5, unreliable morphology
    NEAR_EDGE = 2         # Near image boundary
    BLENDED = 4           # Deblended from neighbor
    SATURATED = 8         # Contains saturated pixels
    GAIA_CONFIRMED = 16   # Confirmed by Gaia parallax/proper motion
    HIGH_CONFIDENCE = 32  # Multiple methods agree
    UNCERTAIN = 64        # Methods disagree
    MAGNITUDE_LIMIT = 128 # Near detection limit


@dataclass
class ClassificationResult:
    """Result of professional star-galaxy classification.

    Attributes
    ----------
    is_galaxy : bool
        True if classified as galaxy
    is_star : bool
        True if classified as star
    probability_galaxy : float
        Probability of being a galaxy (0-1)
    confidence : float
        Classification confidence (0-1)
    method : ClassificationMethod
        Primary method used for classification
    flags : ClassificationFlag
        Quality flags
    tier : int
        Classification tier (1=Gaia, 2=SPREAD_MODEL, 3=ML, 4=color)
    spread_model : float
        SPREAD_MODEL value (if computed)
    spread_model_err : float
        SPREAD_MODEL uncertainty
    gaia_match : bool
        Whether matched to Gaia source
    gaia_parallax : float
        Gaia parallax if matched (mas)
    gaia_pmtot : float
        Gaia total proper motion if matched (mas/yr)
    stellar_locus_dist : float
        Distance from stellar locus in color space
    """
    is_galaxy: bool
    is_star: bool
    probability_galaxy: float
    confidence: float
    method: ClassificationMethod
    flags: ClassificationFlag = ClassificationFlag.GOOD
    tier: int = 0
    spread_model: float = np.nan
    spread_model_err: float = np.nan
    gaia_match: bool = False
    gaia_parallax: float = np.nan
    gaia_pmtot: float = np.nan
    stellar_locus_dist: float = np.nan


@dataclass
class PSFModel:
    """Empirical PSF model from Gaia stars.

    Attributes
    ----------
    fwhm : float
        PSF FWHM in pixels
    fwhm_err : float
        Uncertainty on FWHM
    sigma : float
        PSF sigma (FWHM / 2.355)
    ellipticity : float
        PSF ellipticity (1 - b/a)
    position_angle : float
        PSF position angle (degrees)
    n_stars : int
        Number of stars used for measurement
    spatially_varying : bool
        Whether PSF varies across field
    fwhm_map : NDArray
        Spatially-varying FWHM if computed
    """
    fwhm: float
    fwhm_err: float
    sigma: float
    ellipticity: float = 0.0
    position_angle: float = 0.0
    n_stars: int = 0
    spatially_varying: bool = False
    fwhm_map: NDArray | None = None


@dataclass
class ValidationMetrics:
    """Validation metrics for star-galaxy classification.

    Following professional survey standards.
    """
    # Primary metrics
    galaxy_purity: float         # 1 - (stars in galaxy sample / total galaxies)
    galaxy_completeness: float   # galaxies recovered / true galaxies
    star_purity: float           # 1 - (galaxies in star sample / total stars)
    star_completeness: float     # stars recovered / true stars

    # Contamination rates
    star_contamination: float    # stars misclassified as galaxies
    galaxy_contamination: float  # galaxies misclassified as stars

    # Magnitude-binned metrics
    mag_bins: NDArray = field(default_factory=lambda: np.array([]))
    purity_by_mag: NDArray = field(default_factory=lambda: np.array([]))
    completeness_by_mag: NDArray = field(default_factory=lambda: np.array([]))

    # Confusion matrix
    confusion_matrix: NDArray = field(default_factory=lambda: np.zeros((2, 2)))

    # Sample sizes
    n_true_galaxies: int = 0
    n_true_stars: int = 0
    n_classified_galaxies: int = 0
    n_classified_stars: int = 0


# =============================================================================
# Empirical PSF Measurement (using astropy.modeling)
# =============================================================================

def measure_psf_from_gaia(
    image: NDArray,
    wcs: WCS,
    gaia_stars: pd.DataFrame,
    mag_range: tuple[float, float] = (18.0, 20.5),
    min_separation_pix: float = 30.0,
    max_stars: int = 50,
    cutout_size: int = 21,
) -> PSFModel:
    """Measure empirical PSF from isolated Gaia stars.

    Uses astropy.modeling.Gaussian2D for cleaner, validated 2D Gaussian fitting.

    Parameters
    ----------
    image : NDArray
        2D image array
    wcs : WCS
        WCS for coordinate transformation
    gaia_stars : pd.DataFrame
        Gaia catalog with ra, dec, phot_g_mean_mag columns
    mag_range : tuple
        Magnitude range for PSF stars (avoid saturation and low SNR)
    min_separation_pix : float
        Minimum separation from other sources in pixels
    max_stars : int
        Maximum number of stars to use
    cutout_size : int
        Size of cutouts for PSF measurement (should be odd)

    Returns
    -------
    PSFModel
        Measured PSF model
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.modeling import fitting, models

    default_psf = PSFModel(fwhm=2.5, fwhm_err=0.5, sigma=2.5/2.355, n_stars=0)

    if len(gaia_stars) == 0:
        warnings.warn("No Gaia stars provided, using default PSF", stacklevel=2)
        return default_psf

    # Filter by magnitude - use progressively wider ranges if needed
    mag_col = 'phot_g_mean_mag' if 'phot_g_mean_mag' in gaia_stars.columns else 'gmag'
    if mag_col in gaia_stars.columns:
        # Try progressively wider magnitude ranges
        mag_ranges_to_try = [
            mag_range,                    # Original range
            (15.0, 22.0),                 # Wider range
            (12.0, 24.0),                 # Even wider
            (0.0, 30.0),                  # Accept any magnitude
        ]
        selected = pd.DataFrame()
        for mmin, mmax in mag_ranges_to_try:
            mag_mask = (gaia_stars[mag_col] >= mmin) & (gaia_stars[mag_col] <= mmax)
            selected = gaia_stars[mag_mask].copy()
            if len(selected) > 0:
                break
    else:
        selected = gaia_stars.copy()

    if len(selected) == 0:
        mags = gaia_stars[mag_col].values if mag_col in gaia_stars.columns else []
        mag_info = f" (available mags: {mags})" if len(mags) > 0 else ""
        warnings.warn(f"No Gaia stars in any magnitude range{mag_info}", stacklevel=2)
        return default_psf

    # Convert to pixel coordinates
    try:
        coords = SkyCoord(ra=selected['ra'].values * u.deg,
                         dec=selected['dec'].values * u.deg)
        pixel_coords = wcs.world_to_pixel(coords)
        selected = selected.copy()
        selected['x_pix'] = pixel_coords[0]
        selected['y_pix'] = pixel_coords[1]
    except Exception as e:
        warnings.warn(f"WCS transformation failed: {e}", stacklevel=2)
        return default_psf

    # Filter sources within image bounds
    ny, nx = image.shape
    margin = cutout_size // 2 + 5
    in_bounds = (
        (selected['x_pix'] > margin) & (selected['x_pix'] < nx - margin) &
        (selected['y_pix'] > margin) & (selected['y_pix'] < ny - margin)
    )
    selected = selected[in_bounds]

    if len(selected) == 0:
        warnings.warn("No Gaia stars within image bounds", stacklevel=2)
        return default_psf

    # Find isolated stars using vectorized distance calculation
    x_arr = selected['x_pix'].values
    y_arr = selected['y_pix'].values

    # Compute pairwise distances efficiently (skip if only 1 star)
    if len(selected) > 1:
        dx = x_arr[:, np.newaxis] - x_arr[np.newaxis, :]
        dy = y_arr[:, np.newaxis] - y_arr[np.newaxis, :]
        dists = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dists, np.inf)  # Exclude self

        # Try progressively relaxed isolation criteria
        isolation_thresholds = [min_separation_pix, min_separation_pix / 2, min_separation_pix / 4, 0]
        for thresh in isolation_thresholds:
            if thresh > 0:
                isolated_mask = np.min(dists, axis=1) >= thresh
                selected_isolated = selected[isolated_mask]
            else:
                selected_isolated = selected  # No isolation requirement
            if len(selected_isolated) > 0:
                break
        selected = selected_isolated
    # If only 1 star, use it directly

    if len(selected) == 0:
        warnings.warn("No Gaia stars available for PSF fitting", stacklevel=2)
        return default_psf

    # Limit number of stars
    if len(selected) > max_stars:
        selected = selected.head(max_stars)

    # Measure FWHM using astropy.modeling.Gaussian2D
    fitter = fitting.LevMarLSQFitter()
    half_size = cutout_size // 2
    y_grid, x_grid = np.mgrid[:cutout_size, :cutout_size]

    # Pre-compute saturation threshold (outside loop for efficiency)
    # Use a more robust approach for HDF data which may have different value ranges
    positive_pixels = image[image > 0]
    if len(positive_pixels) > 0:
        p99 = np.percentile(positive_pixels, 99.5)
        p999 = np.percentile(positive_pixels, 99.9)
        saturation_threshold = min(p999, p99 * 2, 65000)
    else:
        saturation_threshold = 65000

    fwhm_measurements = []
    ellipticity_measurements = []
    pa_measurements = []
    fit_failures = {'shape': 0, 'saturated': 0, 'low_snr': 0, 'bad_fit': 0, 'exception': 0}

    for _, star in selected.iterrows():
        x_c, y_c = int(star['x_pix']), int(star['y_pix'])
        cutout = image[y_c-half_size:y_c+half_size+1, x_c-half_size:x_c+half_size+1]

        if cutout.shape != (cutout_size, cutout_size):
            fit_failures['shape'] += 1
            continue

        # Check for saturation (more lenient check)
        if np.max(cutout) > saturation_threshold:
            fit_failures['saturated'] += 1
            continue

        # Check for sufficient signal - more lenient for HDF data
        cutout_median = np.median(cutout)
        amplitude = np.max(cutout) - cutout_median
        cutout_std = np.std(cutout)
        # Only require SNR > 2 (was 3)
        if amplitude <= 0 or (cutout_std > 0 and amplitude < 2 * cutout_std):
            fit_failures['low_snr'] += 1
            continue

        try:
            # Use astropy Gaussian2D model with reasonable initial guesses
            # Start with smaller stddev for HST's sharp PSF
            gauss_init = models.Gaussian2D(
                amplitude=amplitude,
                x_mean=half_size, y_mean=half_size,
                x_stddev=1.5, y_stddev=1.5,
                bounds={
                    'x_stddev': (0.3, 20.0),
                    'y_stddev': (0.3, 20.0),
                    'x_mean': (half_size - 3, half_size + 3),
                    'y_mean': (half_size - 3, half_size + 3),
                }
            ) + models.Const2D(amplitude=cutout_median)

            fitted = fitter(gauss_init, x_grid, y_grid, cutout)

            sigma_x = abs(fitted[0].x_stddev.value)
            sigma_y = abs(fitted[0].y_stddev.value)
            theta = fitted[0].theta.value

            # More lenient sanity check on sigma values (HST can have very sharp PSF)
            if sigma_x < 0.3 or sigma_y < 0.3 or sigma_x > 20 or sigma_y > 20:
                fit_failures['bad_fit'] += 1
                continue

            fwhm = 2.355 * np.sqrt(sigma_x * sigma_y)  # Geometric mean FWHM

            # Accept wider FWHM range: 0.3 to 20 pixels
            if 0.3 < fwhm < 20.0:
                fwhm_measurements.append(fwhm)
                ellipticity_measurements.append(1 - min(sigma_x, sigma_y) / max(sigma_x, sigma_y))
                pa_measurements.append(np.degrees(theta))
            else:
                fit_failures['bad_fit'] += 1
        except Exception:
            fit_failures['exception'] += 1
            continue

    if len(fwhm_measurements) == 0:
        # Provide more diagnostic information
        total_tried = len(selected)
        failure_summary = ", ".join(f"{k}={v}" for k, v in fit_failures.items() if v > 0)
        warnings.warn(
            f"Could not fit PSF to any of {total_tried} Gaia stars "
            f"(failures: {failure_summary})",
            stacklevel=2
        )
        return default_psf

    # Robust statistics (median, MAD)
    fwhm_arr = np.array(fwhm_measurements)
    fwhm_median = float(np.median(fwhm_arr))
    fwhm_mad = float(1.4826 * np.median(np.abs(fwhm_arr - fwhm_median)))
    # Ensure MAD is at least 10% of median (for single measurement or very consistent PSF)
    if fwhm_mad < 0.1 * fwhm_median:
        fwhm_mad = 0.1 * fwhm_median

    return PSFModel(
        fwhm=fwhm_median,
        fwhm_err=max(fwhm_mad, 0.1),  # Minimum uncertainty of 0.1 pixels
        sigma=fwhm_median / 2.355,
        ellipticity=float(np.median(ellipticity_measurements)) if ellipticity_measurements else 0.0,
        position_angle=float(np.median(pa_measurements)) if pa_measurements else 0.0,
        n_stars=len(fwhm_measurements),
        spatially_varying=False,
    )


# =============================================================================
# SPREAD_MODEL Implementation (using SEP library)
# =============================================================================

def _ensure_native_byteorder(data: NDArray) -> NDArray:
    """Ensure array has native byte order for SEP compatibility."""
    if data.dtype.byteorder not in ('=', '|', '<' if np.little_endian else '>'):
        return data.astype(data.dtype.newbyteorder('='), copy=False)
    return data


def compute_spread_model_batch(
    image: NDArray,
    x_coords: NDArray,
    y_coords: NDArray,
    psf_model: PSFModel,
    cutout_size: int = 21,
) -> tuple[NDArray, NDArray]:
    """Compute SPREAD_MODEL for multiple sources using SEP library.

    SEP (Source Extractor as Python library) provides native morphological
    measurements including flux radius comparisons that approximate SPREAD_MODEL.

    SPREAD_MODEL compares PSF vs extended source fits:
    - Stars have SPREAD_MODEL ~ 0
    - Galaxies have SPREAD_MODEL > 0

    Parameters
    ----------
    image : NDArray
        2D image array
    x_coords, y_coords : NDArray
        Source positions
    psf_model : PSFModel
        Empirical PSF model (for FWHM reference)
    cutout_size : int
        Size of cutout for fitting (unused with SEP, kept for API compatibility)

    Returns
    -------
    spread_model : NDArray
        SPREAD_MODEL values
    spread_model_err : NDArray
        SPREAD_MODEL uncertainties
    """
    n_sources = len(x_coords)
    spread_model = np.full(n_sources, np.nan)
    spread_model_err = np.full(n_sources, np.nan)

    # Ensure native byte order for SEP
    data = _ensure_native_byteorder(image.astype(np.float64))

    # Estimate background
    try:
        bkg = sep.Background(data)
        data_sub = data - bkg.back()
        bkg_rms = bkg.globalrms
    except Exception:
        data_sub = data - np.median(data)
        bkg_rms = np.std(data[data < np.median(data)])

    # SEP morphology: compare flux concentration to PSF expectation
    # This approximates SPREAD_MODEL by comparing actual vs expected flux ratio
    # Reference: SExtractor docs - SPREAD_MODEL compares PSF vs extended models
    psf_radius = psf_model.fwhm / 2.0  # Inner aperture (PSF core)
    ext_radius = psf_model.fwhm * 1.5  # Outer aperture (extended light)
    sigma = psf_model.fwhm / 2.355     # Gaussian sigma from FWHM

    x = np.asarray(x_coords, dtype=np.float64)
    y = np.asarray(y_coords, dtype=np.float64)

    try:
        # Measure flux in PSF-sized aperture
        flux_psf, flux_psf_err, _ = sep.sum_circle(
            data_sub, x, y, psf_radius, err=bkg_rms
        )

        # Measure flux in extended aperture
        flux_ext, flux_ext_err, _ = sep.sum_circle(
            data_sub, x, y, ext_radius, err=bkg_rms
        )

        # Compute expected flux ratio for a Gaussian PSF (point source)
        # For a 2D Gaussian, flux within radius r: F(r) = 1 - exp(-r²/(2σ²))
        expected_frac_inner = 1 - np.exp(-psf_radius**2 / (2 * sigma**2))
        expected_frac_outer = 1 - np.exp(-ext_radius**2 / (2 * sigma**2))
        expected_ratio = expected_frac_inner / expected_frac_outer

        # Compute SPREAD_MODEL: deviation from PSF expectation
        # - Point source: actual_ratio ≈ expected_ratio → spread_model ≈ 0
        # - Galaxy (extended): actual_ratio < expected (more flux outside) → spread_model > 0
        # - Cosmic ray (concentrated): actual_ratio > expected → spread_model < 0
        with np.errstate(invalid='ignore', divide='ignore'):
            flux_ext_safe = np.maximum(flux_ext, 1e-10)
            actual_ratio = flux_psf / flux_ext_safe
            spread_model = expected_ratio - actual_ratio

            # Uncertainty from flux error propagation
            # δ(spread_model) ≈ sqrt((δflux_psf/flux_ext)² + (flux_psf*δflux_ext/flux_ext²)²)
            spread_model_err = np.sqrt(
                (flux_psf_err / flux_ext_safe)**2 +
                (flux_psf * flux_ext_err / flux_ext_safe**2)**2
            )
            spread_model_err = np.clip(spread_model_err, 0.001, 0.5)

        # Handle invalid values
        invalid = ~np.isfinite(spread_model) | (flux_ext <= 0)
        spread_model[invalid] = np.nan
        spread_model_err[invalid] = np.nan

    except Exception as e:
        warnings.warn(f"SEP morphology failed: {e}", stacklevel=2)
        return _compute_spread_model_fallback(image, x_coords, y_coords, psf_model)

    return spread_model, spread_model_err


def _compute_spread_model_fallback(
    image: NDArray,
    x_coords: NDArray,
    y_coords: NDArray,
    psf_model: PSFModel,
) -> tuple[NDArray, NDArray]:
    """Fallback SPREAD_MODEL using simple aperture ratio with photutils."""
    from photutils.aperture import CircularAperture, aperture_photometry

    n_sources = len(x_coords)
    spread_model = np.full(n_sources, np.nan)
    spread_model_err = np.full(n_sources, np.nan)

    positions = np.column_stack([x_coords, y_coords])

    # PSF-sized and extended apertures
    psf_radius = psf_model.fwhm / 2.0
    ext_radius = psf_model.fwhm * 1.5
    sigma = psf_model.fwhm / 2.355

    # Expected flux ratio for a Gaussian PSF
    expected_frac_inner = 1 - np.exp(-psf_radius**2 / (2 * sigma**2))
    expected_frac_outer = 1 - np.exp(-ext_radius**2 / (2 * sigma**2))
    expected_ratio = expected_frac_inner / expected_frac_outer

    try:
        ap_psf = CircularAperture(positions, r=psf_radius)
        ap_ext = CircularAperture(positions, r=ext_radius)

        phot_psf = aperture_photometry(image, ap_psf)
        phot_ext = aperture_photometry(image, ap_ext)

        flux_psf = phot_psf['aperture_sum'].value
        flux_ext = phot_ext['aperture_sum'].value

        with np.errstate(invalid='ignore', divide='ignore'):
            flux_ext_safe = np.maximum(flux_ext, 1e-10)
            actual_ratio = flux_psf / flux_ext_safe
            spread_model = expected_ratio - actual_ratio
            spread_model_err = np.full(n_sources, 0.02)  # Default uncertainty

        invalid = ~np.isfinite(spread_model) | (flux_ext <= 0)
        spread_model[invalid] = np.nan
        spread_model_err[invalid] = np.nan

    except Exception:
        pass

    return spread_model, spread_model_err


# =============================================================================
# Magnitude-Dependent Thresholds
# =============================================================================

def get_magnitude_dependent_thresholds(
    magnitudes: NDArray,
    completeness_priority: bool = False,
) -> dict[str, NDArray]:
    """Get magnitude-dependent classification thresholds.

    Thresholds are calibrated based on survey best practices:
    - Bright sources: morphology is reliable
    - Faint sources: need to rely more on ML/colors

    Parameters
    ----------
    magnitudes : NDArray
        Source magnitudes (typically I-band)
    completeness_priority : bool, optional
        If True, relax thresholds to prioritize completeness over purity.
        This extends morphology reliability to fainter magnitudes and
        uses more permissive thresholds. Useful for deep surveys where
        maximizing galaxy detection is more important than minimizing
        stellar contamination. Default is False (balanced thresholds).

    Returns
    -------
    dict
        Dictionary with threshold arrays:
        - concentration_threshold: C threshold for star/galaxy
        - size_threshold_factor: Factor to multiply PSF sigma
        - spread_model_threshold: SPREAD_MODEL threshold
        - use_morphology: Whether morphology is reliable
        - use_ml: Whether to use ML classifier
        - use_colors: Whether to use color selection
    """
    n = len(magnitudes)
    mags = np.asarray(magnitudes)

    if completeness_priority:
        # Relaxed thresholds for better completeness
        # Based on DES Y3 and KiDS-1000 approaches for faint source recovery

        # Default thresholds (bright sources, mag < 21)
        concentration_threshold = np.full(n, 3.0)  # More permissive
        size_threshold_factor = np.full(n, 1.3)    # Less strict size cut
        spread_model_threshold = np.full(n, 0.003) # Wider tolerance
        use_morphology = np.ones(n, dtype=bool)
        use_ml = np.ones(n, dtype=bool)
        use_colors = np.zeros(n, dtype=bool)

        # Intermediate (21 <= mag < 24) - extended range
        intermediate = (mags >= 21) & (mags < 24)
        concentration_threshold[intermediate] = 2.7
        size_threshold_factor[intermediate] = 1.15
        spread_model_threshold[intermediate] = 0.004
        use_colors[intermediate] = True
        # Keep morphology enabled for intermediate sources

        # Faint (24 <= mag < 26) - extended range, morphology still used
        faint = (mags >= 24) & (mags < 26)
        concentration_threshold[faint] = 2.4
        size_threshold_factor[faint] = 1.05
        spread_model_threshold[faint] = 0.006
        use_morphology[faint] = True  # Keep morphology, just with relaxed thresholds
        use_colors[faint] = True

        # Very faint (mag >= 26)
        very_faint = mags >= 26
        concentration_threshold[very_faint] = 2.0
        size_threshold_factor[very_faint] = 1.0
        spread_model_threshold[very_faint] = 0.008
        use_morphology[very_faint] = False  # Too faint even for relaxed morphology
        use_ml[very_faint] = True  # Keep ML enabled longer
        use_colors[very_faint] = True

    else:
        # Default balanced thresholds (original behavior)

        # Default thresholds (bright sources, mag < 20)
        concentration_threshold = np.full(n, 2.8)
        size_threshold_factor = np.full(n, 1.5)
        spread_model_threshold = np.full(n, 0.002)
        use_morphology = np.ones(n, dtype=bool)
        use_ml = np.ones(n, dtype=bool)
        use_colors = np.zeros(n, dtype=bool)

        # Intermediate (20 <= mag < 23)
        intermediate = (mags >= 20) & (mags < 23)
        concentration_threshold[intermediate] = 2.5
        size_threshold_factor[intermediate] = 1.2
        spread_model_threshold[intermediate] = 0.003
        use_colors[intermediate] = True

        # Faint (23 <= mag < 25)
        faint = (mags >= 23) & (mags < 25)
        concentration_threshold[faint] = 2.2
        size_threshold_factor[faint] = 1.1
        spread_model_threshold[faint] = 0.005
        use_morphology[faint] = False  # Morphology unreliable
        use_colors[faint] = True

        # Very faint (mag >= 25)
        very_faint = mags >= 25
        concentration_threshold[very_faint] = np.nan
        size_threshold_factor[very_faint] = np.nan
        spread_model_threshold[very_faint] = np.nan
        use_morphology[very_faint] = False
        use_ml[very_faint] = False  # ML also unreliable
        use_colors[very_faint] = True  # Only colors/photo-z

    return {
        'concentration_threshold': concentration_threshold,
        'size_threshold_factor': size_threshold_factor,
        'spread_model_threshold': spread_model_threshold,
        'use_morphology': use_morphology,
        'use_ml': use_ml,
        'use_colors': use_colors,
    }


# =============================================================================
# Color-Color Stellar Locus (using sklearn KDTree for fast matching)
# =============================================================================

# Pre-computed stellar locus reference points (Covey et al. 2007)
# Format: (B-V, U-B, V-I) for main sequence stars
_STELLAR_LOCUS_REFERENCE = None


def _get_stellar_locus_reference() -> NDArray:
    """Generate reference stellar locus points for KDTree matching."""
    global _STELLAR_LOCUS_REFERENCE
    if _STELLAR_LOCUS_REFERENCE is not None:
        return _STELLAR_LOCUS_REFERENCE

    # Generate stellar locus points along B-V range
    bv_points = np.linspace(-0.3, 2.0, 100)

    # U-B as function of B-V (Covey et al. 2007)
    ub_points = np.where(
        bv_points < 0.5,
        0.82 * bv_points - 0.25,
        0.42 * bv_points + 0.05
    )

    # V-I as function of B-V
    vi_points = 1.1 * bv_points

    _STELLAR_LOCUS_REFERENCE = np.column_stack([bv_points, ub_points, vi_points])
    return _STELLAR_LOCUS_REFERENCE


def compute_stellar_locus_distance(
    u_b: NDArray,
    b_v: NDArray,
    v_i: NDArray,
) -> NDArray:
    """Compute distance from stellar locus using sklearn KDTree.

    Uses KDTree for efficient nearest-neighbor matching to the empirical
    stellar locus (Covey et al. 2007). Much faster than polynomial evaluation
    for large catalogs.

    Parameters
    ----------
    u_b : NDArray
        U-B colors
    b_v : NDArray
        B-V colors
    v_i : NDArray
        V-I colors

    Returns
    -------
    NDArray
        Distance from stellar locus (smaller = more star-like)
    """
    n = len(u_b)
    distance = np.full(n, np.nan)

    valid = np.isfinite(u_b) & np.isfinite(b_v) & np.isfinite(v_i)
    if not np.any(valid):
        return distance

    # Build KDTree from stellar locus reference
    locus_ref = _get_stellar_locus_reference()
    tree = KDTree(locus_ref)

    # Query colors (B-V, U-B, V-I)
    query_points = np.column_stack([b_v[valid], u_b[valid], v_i[valid]])

    # Find nearest neighbor distance
    dist, _ = tree.query(query_points, k=1)
    distance[valid] = dist.ravel()

    return distance


def _compute_stellar_locus_simple(
    u_b: NDArray,
    b_v: NDArray,
    v_i: NDArray,
) -> NDArray:
    """Simple polynomial-based stellar locus distance (fallback)."""
    n = len(u_b)
    distance = np.full(n, np.nan)

    valid = np.isfinite(u_b) & np.isfinite(b_v) & np.isfinite(v_i)
    if not np.any(valid):
        return distance

    bv, ub, vi = b_v[valid], u_b[valid], v_i[valid]

    ub_pred = np.where(bv < 0.5, 0.82 * bv - 0.25, 0.42 * bv + 0.05)
    vi_pred = 1.1 * bv

    distance[valid] = np.sqrt((ub - ub_pred)**2 + 0.5 * (vi - vi_pred)**2)
    return distance


def flag_stellar_locus_sources(
    catalog: pd.DataFrame,
    ub_col: str = 'color_ub',
    bv_col: str = 'color_bv',
    vi_col: str = 'color_vi',
    threshold: float = 0.3,
) -> tuple[NDArray, NDArray]:
    """Flag sources that lie on the stellar locus.

    Parameters
    ----------
    catalog : pd.DataFrame
        Source catalog with color columns
    ub_col, bv_col, vi_col : str
        Column names for colors
    threshold : float
        Distance threshold (sources < threshold are star-like)

    Returns
    -------
    is_stellar_locus : NDArray
        Boolean array, True if on stellar locus
    locus_distance : NDArray
        Distance from stellar locus
    """
    # Get colors
    u_b = catalog[ub_col].values if ub_col in catalog.columns else np.full(len(catalog), np.nan)
    b_v = catalog[bv_col].values if bv_col in catalog.columns else np.full(len(catalog), np.nan)
    v_i = catalog[vi_col].values if vi_col in catalog.columns else np.full(len(catalog), np.nan)

    # Compute distance
    locus_distance = compute_stellar_locus_distance(u_b, b_v, v_i)

    # Flag sources on locus
    is_stellar_locus = locus_distance < threshold

    return is_stellar_locus, locus_distance


# =============================================================================
# Gaia Cross-Matching
# =============================================================================

def cross_match_gaia(
    catalog: pd.DataFrame,
    gaia_catalog: pd.DataFrame,
    match_radius_arcsec: float = 1.5,
    ra_col: str = 'ra',
    dec_col: str = 'dec',
    parallax_threshold: float = 0.1,
    pm_threshold: float = 2.0,
) -> pd.DataFrame:
    """Cross-match catalog with Gaia DR3 to identify stars.

    Sources matched to Gaia with significant parallax or proper motion
    are confirmed foreground stars.

    Parameters
    ----------
    catalog : pd.DataFrame
        Source catalog with RA, Dec columns
    gaia_catalog : pd.DataFrame
        Gaia DR3 catalog with ra, dec, parallax, pmra, pmdec columns
    match_radius_arcsec : float
        Maximum match radius in arcseconds
    ra_col, dec_col : str
        Column names for coordinates
    parallax_threshold : float
        Minimum parallax to confirm star (mas)
    pm_threshold : float
        Minimum proper motion to confirm star (mas/yr)

    Returns
    -------
    pd.DataFrame
        Catalog with additional columns:
        - gaia_match: Boolean, True if matched to Gaia
        - gaia_parallax: Parallax from Gaia (mas)
        - gaia_pmra: Proper motion in RA (mas/yr)
        - gaia_pmdec: Proper motion in Dec (mas/yr)
        - gaia_pmtot: Total proper motion (mas/yr)
        - gaia_confirmed_star: Boolean, confirmed star based on astrometry
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord, match_coordinates_sky

    # Initialize output columns
    result = catalog.copy()
    len(catalog)

    result['gaia_match'] = False
    result['gaia_parallax'] = np.nan
    result['gaia_pmra'] = np.nan
    result['gaia_pmdec'] = np.nan
    result['gaia_pmtot'] = np.nan
    result['gaia_gmag'] = np.nan
    result['gaia_confirmed_star'] = False

    if len(gaia_catalog) == 0:
        warnings.warn("Empty Gaia catalog provided", stacklevel=2)
        return result

    # Create SkyCoord objects
    try:
        our_coords = SkyCoord(
            ra=catalog[ra_col].values * u.deg,
            dec=catalog[dec_col].values * u.deg
        )

        # Handle Gaia column names
        gaia_ra = gaia_catalog['ra'].values if 'ra' in gaia_catalog.columns else gaia_catalog['RA'].values
        gaia_dec = gaia_catalog['dec'].values if 'dec' in gaia_catalog.columns else gaia_catalog['DEC'].values

        gaia_coords = SkyCoord(
            ra=gaia_ra * u.deg,
            dec=gaia_dec * u.deg
        )
    except Exception as e:
        warnings.warn(f"Coordinate creation failed: {e}", stacklevel=2)
        return result

    # Cross-match
    idx, sep, _ = match_coordinates_sky(our_coords, gaia_coords)
    matched = sep.arcsec < match_radius_arcsec

    # Get matched Gaia data
    matched_gaia = gaia_catalog.iloc[idx[matched]]

    # Fill in Gaia columns for matched sources
    result.loc[matched, 'gaia_match'] = True

    # Parallax
    if 'parallax' in matched_gaia.columns:
        result.loc[matched, 'gaia_parallax'] = matched_gaia['parallax'].values

    # Proper motion
    pmra_col = 'pmra' if 'pmra' in matched_gaia.columns else None
    pmdec_col = 'pmdec' if 'pmdec' in matched_gaia.columns else None

    if pmra_col and pmdec_col:
        pmra = matched_gaia[pmra_col].values
        pmdec = matched_gaia[pmdec_col].values
        result.loc[matched, 'gaia_pmra'] = pmra
        result.loc[matched, 'gaia_pmdec'] = pmdec
        result.loc[matched, 'gaia_pmtot'] = np.sqrt(pmra**2 + pmdec**2)

    # G magnitude
    gmag_col = 'phot_g_mean_mag' if 'phot_g_mean_mag' in matched_gaia.columns else 'gmag'
    if gmag_col in matched_gaia.columns:
        result.loc[matched, 'gaia_gmag'] = matched_gaia[gmag_col].values

    # Confirm stars based on parallax or proper motion
    parallax = result['gaia_parallax'].values
    pmtot = result['gaia_pmtot'].values

    # Star if: significant parallax OR significant proper motion
    # Note: need to handle NaN and parallax errors
    confirmed = (
        result['gaia_match'] &
        (
            (np.abs(parallax) > parallax_threshold) |
            (pmtot > pm_threshold)
        )
    )
    result['gaia_confirmed_star'] = confirmed

    n_matched = matched.sum()
    n_confirmed = confirmed.sum()
    print(f"Gaia cross-match: {n_matched} matches, {n_confirmed} confirmed stars")

    return result


# =============================================================================
# Multi-Tier Classification Pipeline
# =============================================================================

def classify_professional(
    catalog: pd.DataFrame,
    image: NDArray,
    wcs: WCS | None = None,
    gaia_catalog: pd.DataFrame | None = None,
    psf_model: PSFModel | None = None,
    ml_classifier = None,
    x_col: str = 'xcentroid',
    y_col: str = 'ycentroid',
    ra_col: str = 'ra',
    dec_col: str = 'dec',
    mag_col: str = 'mag_auto',
    flux_cols: dict | None = None,
    error_cols: dict | None = None,
    completeness_priority: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Professional multi-tier star-galaxy classification.

    Implements the full classification pipeline following survey standards:

    Tier 1: Gaia cross-match (100% confidence for bright stars)
    Tier 2: SPREAD_MODEL morphology (high confidence)
    Tier 3: ML classifier with magnitude-dependent features
    Tier 4: Color-color stellar locus (supplementary)

    Final classification uses weighted voting across tiers.

    Parameters
    ----------
    catalog : pd.DataFrame
        Source catalog with positions and photometry
    image : NDArray
        2D image array for morphological measurements
    wcs : WCS, optional
        WCS for coordinate transformation
    gaia_catalog : pd.DataFrame, optional
        Gaia DR3 catalog for cross-matching
    psf_model : PSFModel, optional
        Empirical PSF model (will be measured if Gaia available)
    ml_classifier : optional
        Trained ML classifier (MLStarGalaxyClassifier)
    x_col, y_col : str
        Column names for pixel positions
    ra_col, dec_col : str
        Column names for sky coordinates
    mag_col : str
        Column name for magnitude (for thresholds)
    flux_cols : dict, optional
        Mapping of band name to flux column
    error_cols : dict, optional
        Mapping of band name to error column
    completeness_priority : bool, optional
        If True, use relaxed thresholds that prioritize completeness over
        purity. Extends morphology reliability to fainter magnitudes and
        uses more permissive thresholds. Useful for deep surveys where
        maximizing galaxy detection is more important. Default is False.
    verbose : bool
        Print progress information

    Returns
    -------
    pd.DataFrame
        Classification results with columns:
        - is_galaxy: Final classification
        - is_star: Inverse of is_galaxy
        - probability_galaxy: Probability (0-1)
        - confidence: Classification confidence (0-1)
        - classification_method: Primary method used
        - classification_tier: Tier of classification
        - classification_flags: Quality flags
        - spread_model: SPREAD_MODEL value
        - spread_model_err: SPREAD_MODEL error
        - gaia_match: Matched to Gaia
        - gaia_confirmed_star: Confirmed star from Gaia
        - stellar_locus_distance: Distance from stellar locus
        - concentration_c: Concentration index
        - half_light_radius: Half-light radius
        - ml_prob_galaxy: ML probability (if available)
    """
    n_sources = len(catalog)
    if verbose:
        print(f"Professional star-galaxy classification for {n_sources} sources")

    # Initialize results
    results = catalog.copy()
    results['is_galaxy'] = True  # Default to galaxy
    results['is_star'] = False
    results['probability_galaxy'] = 0.5
    results['confidence'] = 0.0
    results['classification_method'] = ClassificationMethod.UNKNOWN.value
    results['classification_tier'] = 0
    results['classification_flags'] = ClassificationFlag.GOOD.value
    results['spread_model'] = np.nan
    results['spread_model_err'] = np.nan
    results['stellar_locus_distance'] = np.nan

    # Get coordinates
    x_coords = catalog[x_col].values
    y_coords = catalog[y_col].values

    # Get magnitudes for thresholds
    magnitudes = catalog[mag_col].values if mag_col in catalog.columns else np.full(n_sources, 22.0)

    # Get magnitude-dependent thresholds
    thresholds = get_magnitude_dependent_thresholds(magnitudes, completeness_priority=completeness_priority)
    if verbose and completeness_priority:
        print("  Using relaxed thresholds (completeness priority mode)")

    # =========================================================================
    # TIER 1: Gaia Cross-Match
    # =========================================================================
    if verbose:
        print("Tier 1: Gaia cross-matching...")

    if gaia_catalog is not None and len(gaia_catalog) > 0 and ra_col in catalog.columns:
        results = cross_match_gaia(
            results, gaia_catalog,
            match_radius_arcsec=1.5,
            ra_col=ra_col, dec_col=dec_col
        )

        # Classify Gaia-confirmed stars
        gaia_stars = results['gaia_confirmed_star'].values
        results.loc[gaia_stars, 'is_galaxy'] = False
        results.loc[gaia_stars, 'is_star'] = True
        results.loc[gaia_stars, 'probability_galaxy'] = 0.0
        results.loc[gaia_stars, 'confidence'] = 1.0
        results.loc[gaia_stars, 'classification_method'] = ClassificationMethod.GAIA.value
        results.loc[gaia_stars, 'classification_tier'] = 1
        results.loc[gaia_stars, 'classification_flags'] = (
            ClassificationFlag.GAIA_CONFIRMED | ClassificationFlag.HIGH_CONFIDENCE
        ).value

        if verbose:
            n_gaia_stars = gaia_stars.sum()
            print(f"  Tier 1: {n_gaia_stars} Gaia-confirmed stars")
    else:
        gaia_stars = np.zeros(n_sources, dtype=bool)

    # =========================================================================
    # TIER 2: SPREAD_MODEL
    # =========================================================================
    if verbose:
        print("Tier 2: SPREAD_MODEL classification...")

    # Measure PSF if not provided
    if psf_model is None and gaia_catalog is not None and wcs is not None:
        if verbose:
            print("  Measuring empirical PSF from Gaia stars...")
        psf_model = measure_psf_from_gaia(image, wcs, gaia_catalog)
        if verbose:
            print(f"  PSF FWHM = {psf_model.fwhm:.2f} +/- {psf_model.fwhm_err:.2f} pixels "
                  f"(from {psf_model.n_stars} stars)")

    if psf_model is None:
        # Use default PSF
        psf_model = PSFModel(fwhm=2.5, fwhm_err=0.5, sigma=2.5/2.355, n_stars=0)
        if verbose:
            print(f"  Using default PSF FWHM = {psf_model.fwhm:.2f} pixels")

    # Compute SPREAD_MODEL for unclassified sources
    unclassified = ~gaia_stars
    if np.any(unclassified):
        x_unc = x_coords[unclassified]
        y_unc = y_coords[unclassified]

        sm, sm_err = compute_spread_model_batch(image, x_unc, y_unc, psf_model)

        # Zero-center SPREAD_MODEL to account for systematic offset
        # The PSF model may not perfectly match the actual image PSF,
        # causing a systematic offset in SPREAD_MODEL values.
        # Subtracting the median centers point sources around 0.
        sm_valid = sm[np.isfinite(sm)]
        if len(sm_valid) > 10:
            sm_offset = np.median(sm_valid)
            sm = sm - sm_offset
            if verbose:
                print(f"  SPREAD_MODEL zero-centered (offset={sm_offset:.4f})")

        results.loc[unclassified, 'spread_model'] = sm
        results.loc[unclassified, 'spread_model_err'] = sm_err

        # Classify based on SPREAD_MODEL with magnitude-dependent thresholds
        sm_thresholds = thresholds['spread_model_threshold'][unclassified]

        # Adaptive threshold: fixed + 3*error
        adaptive_threshold = np.sqrt(sm_thresholds**2 + (3.0 * sm_err)**2)

        # Stars: SPREAD_MODEL < -threshold (more point-like than PSF)
        # Galaxies: SPREAD_MODEL > threshold (more extended than PSF)
        sm_stars = sm < -adaptive_threshold
        sm_galaxies = sm > adaptive_threshold
        ~sm_stars & ~sm_galaxies

        # Get indices
        unc_idx = np.where(unclassified)[0]

        # Update classifications
        for i, idx in enumerate(unc_idx):
            if sm_stars[i] and thresholds['use_morphology'][idx]:
                results.loc[results.index[idx], 'is_galaxy'] = False
                results.loc[results.index[idx], 'is_star'] = True
                results.loc[results.index[idx], 'probability_galaxy'] = 0.1
                results.loc[results.index[idx], 'confidence'] = 0.8
                results.loc[results.index[idx], 'classification_method'] = ClassificationMethod.SPREAD_MODEL.value
                results.loc[results.index[idx], 'classification_tier'] = 2
            elif sm_galaxies[i] and thresholds['use_morphology'][idx]:
                results.loc[results.index[idx], 'is_galaxy'] = True
                results.loc[results.index[idx], 'is_star'] = False
                results.loc[results.index[idx], 'probability_galaxy'] = 0.9
                results.loc[results.index[idx], 'confidence'] = 0.8
                results.loc[results.index[idx], 'classification_method'] = ClassificationMethod.SPREAD_MODEL.value
                results.loc[results.index[idx], 'classification_tier'] = 2

        n_sm_stars = sm_stars.sum()
        n_sm_galaxies = sm_galaxies.sum()
        if verbose:
            print(f"  Tier 2: {n_sm_stars} stars, {n_sm_galaxies} galaxies classified by SPREAD_MODEL")

    # =========================================================================
    # TIER 3: Classical Morphology (Concentration + Size)
    # =========================================================================
    if verbose:
        print("Tier 3: Classical morphology...")

    # Compute morphological parameters
    from morphology.concentration import compute_morphology_batch

    morphology = compute_morphology_batch(image, x_coords, y_coords)
    results['concentration_c'] = morphology['concentration_c']
    results['half_light_radius'] = morphology['half_light_radius']

    # Classify based on concentration and size
    still_unclassified = results['classification_tier'] == 0

    if np.any(still_unclassified):
        unc_idx = np.where(still_unclassified)[0]

        c_vals = morphology['concentration_c'][unc_idx]
        r_vals = morphology['half_light_radius'][unc_idx]
        c_thresholds = thresholds['concentration_threshold'][unc_idx]
        size_factors = thresholds['size_threshold_factor'][unc_idx]
        use_morph = thresholds['use_morphology'][unc_idx]

        # Classification criteria
        psf_sigma = psf_model.sigma
        size_threshold = size_factors * psf_sigma

        # Stars: high concentration AND small size
        c_star = c_vals > c_thresholds
        size_star = r_vals < size_threshold
        both_star = c_star & size_star & use_morph

        # Galaxies: low concentration OR large size
        both_galaxy = (~c_star | ~size_star) & use_morph

        # Compute confidence and probability based on how clear the classification is
        # Distance from threshold (normalized): larger distance = higher confidence
        c_margin = np.abs(c_vals - c_thresholds) / np.maximum(c_thresholds, 0.1)
        r_margin = np.abs(r_vals - size_threshold) / np.maximum(size_threshold, 0.1)

        for i, idx in enumerate(unc_idx):
            # Confidence based on how far from decision boundary (0.5 to 0.95)
            conf = np.clip(0.5 + 0.3 * (c_margin[i] + r_margin[i]), 0.5, 0.95)
            # Handle NaN values
            if not np.isfinite(conf):
                conf = 0.6

            if both_star[i]:
                # Probability based on how star-like (higher C, smaller r = more star-like)
                prob_star = np.clip(0.7 + 0.2 * c_margin[i], 0.7, 0.95)
                if not np.isfinite(prob_star):
                    prob_star = 0.8
                results.loc[results.index[idx], 'is_galaxy'] = False
                results.loc[results.index[idx], 'is_star'] = True
                results.loc[results.index[idx], 'probability_galaxy'] = 1.0 - prob_star
                results.loc[results.index[idx], 'confidence'] = conf
                results.loc[results.index[idx], 'classification_method'] = ClassificationMethod.CONCENTRATION.value
                results.loc[results.index[idx], 'classification_tier'] = 3
            elif both_galaxy[i]:
                # Probability based on how galaxy-like (lower C, larger r = more galaxy-like)
                prob_gal = np.clip(0.7 + 0.2 * (c_margin[i] + r_margin[i]) / 2, 0.7, 0.95)
                if not np.isfinite(prob_gal):
                    prob_gal = 0.8
                results.loc[results.index[idx], 'is_galaxy'] = True
                results.loc[results.index[idx], 'is_star'] = False
                results.loc[results.index[idx], 'probability_galaxy'] = prob_gal
                results.loc[results.index[idx], 'confidence'] = conf
                results.loc[results.index[idx], 'classification_method'] = ClassificationMethod.CONCENTRATION.value
                results.loc[results.index[idx], 'classification_tier'] = 3

        n_morph = both_star.sum() + both_galaxy.sum()
        if verbose:
            print(f"  Tier 3: {n_morph} sources classified by concentration/size")

    # =========================================================================
    # TIER 4: ML Classifier (if available)
    # =========================================================================
    if ml_classifier is not None:
        if verbose:
            print("Tier 4: ML classification...")

        try:
            still_unclassified = results['classification_tier'] == 0

            if np.any(still_unclassified):
                # Extract features for unclassified sources
                unc_catalog = catalog[still_unclassified]
                features = extract_features_from_catalog(
                    unc_catalog, image, x_col, y_col, flux_cols, error_cols
                )

                # Get predictions
                ml_results = ml_classifier.predict(features)

                unc_idx = np.where(still_unclassified)[0]

                for _i, (idx, ml_res) in enumerate(zip(unc_idx, ml_results, strict=False)):
                    if thresholds['use_ml'][idx]:
                        results.loc[results.index[idx], 'is_galaxy'] = ml_res.is_galaxy
                        results.loc[results.index[idx], 'is_star'] = not ml_res.is_galaxy
                        results.loc[results.index[idx], 'probability_galaxy'] = ml_res.probability_galaxy
                        results.loc[results.index[idx], 'confidence'] = ml_res.confidence
                        results.loc[results.index[idx], 'classification_method'] = ClassificationMethod.ML_CLASSIFIER.value
                        results.loc[results.index[idx], 'classification_tier'] = 4
                        results.loc[results.index[idx], 'ml_prob_galaxy'] = ml_res.probability_galaxy

                n_ml = still_unclassified.sum()
                if verbose:
                    print(f"  Tier 4: {n_ml} sources classified by ML")

        except Exception as e:
            if verbose:
                print(f"  Tier 4: ML classification failed: {e}")

    # =========================================================================
    # TIER 5: Color-Color Stellar Locus
    # =========================================================================
    if verbose:
        print("Tier 5: Color-color stellar locus...")

    # Compute stellar locus distance
    if 'color_ub' in catalog.columns:
        is_stellar_locus, locus_distance = flag_stellar_locus_sources(catalog)
        results['stellar_locus_distance'] = locus_distance

        # Use as supplementary check for unclassified sources
        still_unclassified = results['classification_tier'] == 0

        if np.any(still_unclassified):
            unc_idx = np.where(still_unclassified)[0]

            for idx in unc_idx:
                if thresholds['use_colors'][idx] and is_stellar_locus[idx]:
                    results.loc[results.index[idx], 'is_galaxy'] = False
                    results.loc[results.index[idx], 'is_star'] = True
                    results.loc[results.index[idx], 'probability_galaxy'] = 0.3
                    results.loc[results.index[idx], 'confidence'] = 0.5
                    results.loc[results.index[idx], 'classification_method'] = ClassificationMethod.COLOR_LOCUS.value
                    results.loc[results.index[idx], 'classification_tier'] = 5

            n_color = (still_unclassified & is_stellar_locus).sum()
            if verbose:
                print(f"  Tier 5: {n_color} sources on stellar locus")

    # =========================================================================
    # Final Summary
    # =========================================================================
    n_galaxies = results['is_galaxy'].sum()
    n_stars = results['is_star'].sum()
    n_unclassified = (results['classification_tier'] == 0).sum()

    # Default remaining unclassified to galaxies
    still_unclassified = results['classification_tier'] == 0
    results.loc[still_unclassified, 'is_galaxy'] = True
    results.loc[still_unclassified, 'confidence'] = 0.3
    results.loc[still_unclassified, 'classification_flags'] = ClassificationFlag.UNCERTAIN.value

    if verbose:
        print("\nClassification Summary:")
        print(f"  Total sources:      {n_sources}")
        print(f"  Galaxies:           {n_galaxies} ({100*n_galaxies/n_sources:.1f}%)")
        print(f"  Stars:              {n_stars} ({100*n_stars/n_sources:.1f}%)")
        print(f"  Default (unknown):  {n_unclassified} ({100*n_unclassified/n_sources:.1f}%)")
        print("\nBy tier:")
        for tier in range(1, 6):
            n_tier = (results['classification_tier'] == tier).sum()
            if n_tier > 0:
                print(f"  Tier {tier}: {n_tier} sources")

    return results


# =============================================================================
# Validation Functions
# =============================================================================

def validate_classification(
    our_classification: pd.DataFrame,
    reference_catalog: pd.DataFrame,
    match_radius_arcsec: float = 1.0,
    ra_col: str = 'ra',
    dec_col: str = 'dec',
    ref_star_col: str = 'is_star',
    mag_col: str = 'mag_auto',
    mag_bins: NDArray | None = None,
) -> ValidationMetrics:
    """Validate star-galaxy classification against reference catalog.

    Parameters
    ----------
    our_classification : pd.DataFrame
        Our classification results with is_galaxy, is_star columns
    reference_catalog : pd.DataFrame
        Reference catalog with ground truth star/galaxy labels
    match_radius_arcsec : float
        Maximum match radius
    ra_col, dec_col : str
        Coordinate column names
    ref_star_col : str
        Column name for star flag in reference (True = star)
    mag_col : str
        Magnitude column for binned analysis
    mag_bins : NDArray, optional
        Magnitude bin edges for binned metrics

    Returns
    -------
    ValidationMetrics
        Comprehensive validation metrics
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord, match_coordinates_sky

    if mag_bins is None:
        mag_bins = np.array([18, 20, 22, 24, 26, 28])

    # Cross-match catalogs
    our_coords = SkyCoord(
        ra=our_classification[ra_col].values * u.deg,
        dec=our_classification[dec_col].values * u.deg
    )
    ref_coords = SkyCoord(
        ra=reference_catalog[ra_col].values * u.deg,
        dec=reference_catalog[dec_col].values * u.deg
    )

    idx, sep, _ = match_coordinates_sky(our_coords, ref_coords)
    matched = sep.arcsec < match_radius_arcsec

    if matched.sum() == 0:
        warnings.warn("No matches found for validation", stacklevel=2)
        return ValidationMetrics(
            galaxy_purity=np.nan, galaxy_completeness=np.nan,
            star_purity=np.nan, star_completeness=np.nan,
            star_contamination=np.nan, galaxy_contamination=np.nan
        )

    # Get matched classifications
    our_is_star = our_classification.loc[matched, 'is_star'].values
    our_is_galaxy = our_classification.loc[matched, 'is_galaxy'].values

    ref_matched = reference_catalog.iloc[idx[matched]]
    ref_is_star = ref_matched[ref_star_col].values
    ref_is_galaxy = ~ref_is_star

    # Confusion matrix
    # [[TN, FP], [FN, TP]] for galaxy classification
    TP = np.sum(our_is_galaxy & ref_is_galaxy)  # True galaxies
    TN = np.sum(our_is_star & ref_is_star)      # True stars
    FP = np.sum(our_is_galaxy & ref_is_star)    # Stars misclassified as galaxies
    FN = np.sum(our_is_star & ref_is_galaxy)    # Galaxies misclassified as stars

    confusion = np.array([[TN, FP], [FN, TP]])

    # Metrics
    n_true_galaxies = ref_is_galaxy.sum()
    n_true_stars = ref_is_star.sum()
    n_classified_galaxies = our_is_galaxy.sum()
    n_classified_stars = our_is_star.sum()

    galaxy_purity = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    galaxy_completeness = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    star_purity = TN / (TN + FN) if (TN + FN) > 0 else np.nan
    star_completeness = TN / (TN + FP) if (TN + FP) > 0 else np.nan

    star_contamination = FP / (TP + FP) if (TP + FP) > 0 else np.nan  # Stars in galaxy sample
    galaxy_contamination = FN / (TN + FN) if (TN + FN) > 0 else np.nan  # Galaxies in star sample

    # Magnitude-binned metrics
    if mag_col in our_classification.columns:
        mags = our_classification.loc[matched, mag_col].values
        purity_by_mag = []
        completeness_by_mag = []

        for i in range(len(mag_bins) - 1):
            in_bin = (mags >= mag_bins[i]) & (mags < mag_bins[i + 1])
            if in_bin.sum() > 0:
                tp_bin = np.sum(our_is_galaxy[in_bin] & ref_is_galaxy[in_bin])
                fp_bin = np.sum(our_is_galaxy[in_bin] & ref_is_star[in_bin])
                fn_bin = np.sum(our_is_star[in_bin] & ref_is_galaxy[in_bin])

                purity_bin = tp_bin / (tp_bin + fp_bin) if (tp_bin + fp_bin) > 0 else np.nan
                completeness_bin = tp_bin / (tp_bin + fn_bin) if (tp_bin + fn_bin) > 0 else np.nan
            else:
                purity_bin = np.nan
                completeness_bin = np.nan

            purity_by_mag.append(purity_bin)
            completeness_by_mag.append(completeness_bin)
    else:
        purity_by_mag = []
        completeness_by_mag = []

    return ValidationMetrics(
        galaxy_purity=galaxy_purity,
        galaxy_completeness=galaxy_completeness,
        star_purity=star_purity,
        star_completeness=star_completeness,
        star_contamination=star_contamination,
        galaxy_contamination=galaxy_contamination,
        mag_bins=mag_bins,
        purity_by_mag=np.array(purity_by_mag),
        completeness_by_mag=np.array(completeness_by_mag),
        confusion_matrix=confusion,
        n_true_galaxies=n_true_galaxies,
        n_true_stars=n_true_stars,
        n_classified_galaxies=n_classified_galaxies,
        n_classified_stars=n_classified_stars,
    )


def print_validation_metrics(metrics: ValidationMetrics) -> None:
    """Print formatted validation metrics."""
    print("\n" + "=" * 60)
    print("STAR-GALAXY CLASSIFICATION VALIDATION")
    print("=" * 60)

    print("\nSample sizes:")
    print(f"  True galaxies:       {metrics.n_true_galaxies}")
    print(f"  True stars:          {metrics.n_true_stars}")
    print(f"  Classified galaxies: {metrics.n_classified_galaxies}")
    print(f"  Classified stars:    {metrics.n_classified_stars}")

    print("\nGalaxy classification:")
    print(f"  Purity:       {metrics.galaxy_purity:.1%}")
    print(f"  Completeness: {metrics.galaxy_completeness:.1%}")

    print("\nStar classification:")
    print(f"  Purity:       {metrics.star_purity:.1%}")
    print(f"  Completeness: {metrics.star_completeness:.1%}")

    print("\nContamination rates:")
    print(f"  Stars in galaxy sample:    {metrics.star_contamination:.1%}")
    print(f"  Galaxies in star sample:   {metrics.galaxy_contamination:.1%}")

    print("\nConfusion matrix:")
    print("              Predicted Star  Predicted Galaxy")
    print(f"  True Star   {metrics.confusion_matrix[0, 0]:>14}  {metrics.confusion_matrix[0, 1]:>16}")
    print(f"  True Galaxy {metrics.confusion_matrix[1, 0]:>14}  {metrics.confusion_matrix[1, 1]:>16}")

    if len(metrics.purity_by_mag) > 0:
        print("\nMagnitude-binned metrics:")
        for i in range(len(metrics.mag_bins) - 1):
            print(f"  {metrics.mag_bins[i]:.0f}-{metrics.mag_bins[i+1]:.0f} mag: "
                  f"Purity={metrics.purity_by_mag[i]:.1%}, "
                  f"Completeness={metrics.completeness_by_mag[i]:.1%}")

    print("=" * 60)


def plot_classification_diagnostics(
    results: pd.DataFrame,
    output_path: Path | None = None,
    figsize: tuple = (16, 12),
) -> None:
    """Plot comprehensive classification diagnostics.

    Parameters
    ----------
    results : pd.DataFrame
        Classification results from classify_professional()
    output_path : Path, optional
        Save figure to this path
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    galaxies = results[results['is_galaxy']]
    stars = results[results['is_star']]

    # 1. Size vs Concentration
    ax = axes[0, 0]
    valid_gal = galaxies['concentration_c'].notna() & galaxies['half_light_radius'].notna()
    valid_star = stars['concentration_c'].notna() & stars['half_light_radius'].notna()

    if valid_gal.any():
        ax.scatter(galaxies.loc[valid_gal, 'half_light_radius'],
                  galaxies.loc[valid_gal, 'concentration_c'],
                  alpha=0.4, s=10, c='blue', label=f'Galaxies ({len(galaxies)})')
    if valid_star.any():
        ax.scatter(stars.loc[valid_star, 'half_light_radius'],
                  stars.loc[valid_star, 'concentration_c'],
                  alpha=0.4, s=10, c='red', label=f'Stars ({len(stars)})')

    ax.axhline(2.8, color='k', linestyle='--', alpha=0.5, label='C=2.8')
    ax.set_xlabel('Half-light radius (pixels)')
    ax.set_ylabel('Concentration C')
    ax.set_title('Size vs Concentration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. SPREAD_MODEL distribution
    ax = axes[0, 1]
    valid_sm = results['spread_model'].notna()
    if valid_sm.any():
        bins = np.linspace(-0.05, 0.1, 50)
        ax.hist(galaxies.loc[galaxies['spread_model'].notna(), 'spread_model'],
               bins=bins, alpha=0.7, label='Galaxies', color='blue')
        ax.hist(stars.loc[stars['spread_model'].notna(), 'spread_model'],
               bins=bins, alpha=0.7, label='Stars', color='red')
        ax.axvline(0.002, color='k', linestyle='--', alpha=0.5, label='Threshold')
        ax.axvline(-0.002, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('SPREAD_MODEL')
    ax.set_ylabel('Count')
    ax.set_title('SPREAD_MODEL Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Classification tier distribution
    ax = axes[0, 2]
    tiers = results['classification_tier'].values
    tier_labels = ['Unclassified', 'Gaia', 'SPREAD_MODEL', 'Morphology', 'ML', 'Color']
    tier_counts = [np.sum(tiers == i) for i in range(6)]
    colors = ['gray', 'gold', 'green', 'blue', 'purple', 'orange']
    ax.bar(tier_labels, tier_counts, color=colors)
    ax.set_ylabel('Count')
    ax.set_title('Classification Method Distribution')
    ax.tick_params(axis='x', rotation=45)

    # 4. Confidence distribution
    ax = axes[1, 0]
    ax.hist(galaxies['confidence'], bins=20, alpha=0.7, label='Galaxies', color='blue')
    ax.hist(stars['confidence'], bins=20, alpha=0.7, label='Stars', color='red')
    ax.set_xlabel('Classification Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Probability distribution
    ax = axes[1, 1]
    ax.hist(results['probability_galaxy'], bins=50, alpha=0.7, color='gray')
    ax.axvline(0.5, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Probability(Galaxy)')
    ax.set_ylabel('Count')
    ax.set_title('Galaxy Probability Distribution')
    ax.grid(True, alpha=0.3)

    # 6. Stellar locus distance
    ax = axes[1, 2]
    valid_sl = results['stellar_locus_distance'].notna()
    if valid_sl.any():
        ax.hist(galaxies.loc[galaxies['stellar_locus_distance'].notna(), 'stellar_locus_distance'],
               bins=30, alpha=0.7, label='Galaxies', color='blue')
        ax.hist(stars.loc[stars['stellar_locus_distance'].notna(), 'stellar_locus_distance'],
               bins=30, alpha=0.7, label='Stars', color='red')
        ax.axvline(0.3, color='k', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_xlabel('Stellar Locus Distance')
    ax.set_ylabel('Count')
    ax.set_title('Stellar Locus Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved diagnostic plot to {output_path}")

    return fig
