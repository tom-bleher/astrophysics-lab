#!/usr/bin/env python3
"""
Full analysis pipeline for angular size test
Runs the complete processing and saves output plots as PDFs

Usage:
    python run_analysis.py full    # Run on entire image (4096x4096), output to ./output/full_HDF/
    python run_analysis.py chip3   # Run on chip3 only (2048x2048), output to ./output/chip3_HDF/
    python run_analysis.py both    # Run full and extract chip3, output to ./output/full_HDF/ and ./output/chip3_HDF/
"""

# =============================================================================
# Performance Optimization: Set thread counts BEFORE importing numpy/numba
# =============================================================================
import os

# Optimal thread settings for Intel i7-1065G7 (4 cores, 8 threads)
_N_THREADS = str(min(os.cpu_count() or 4, 8))
os.environ.setdefault("NUMBA_NUM_THREADS", _N_THREADS)
os.environ.setdefault("OMP_NUM_THREADS", _N_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _N_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _N_THREADS)

import argparse
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend before importing pyplot

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.convolution import convolve  # noqa: E402
from astropy.io import fits  # noqa: E402
from photutils.background import Background2D, MedianBackground  # noqa: E402
from photutils.segmentation import SourceCatalog, SourceFinder, make_2dgaussian_kernel  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from classify import (  # noqa: E402
    classify_batch_ultrafast,
    classify_galaxy,
    classify_galaxy_batch_with_pdf,
)
from scientific import (  # noqa: E402
    H0,
    D_A_LCDM_vectorized,
    c,
    choose_redshift_bin_edges,
    compute_chi2_stats,
    get_radius,
    get_radius_and_omega,
    theta_lcdm_flat,
    theta_static,
)

# Professional star-galaxy classification (research-standard)
USE_PROFESSIONAL_CLASSIFICATION = True  # Set to False to use basic classification only

# Configuration
PIXEL_SCALE = 0.04  # arcsec/pixel
RAD_TO_ARCSEC = 180.0 * 3600.0 / np.pi  # ~206265 arcsec/radian

# Chip3 bounds within the 4096x4096 mosaic (from template matching)
# Chip3 is 2048x2048, placed at offset (127, 184) after 180° rotation
CHIP3_X_MIN, CHIP3_X_MAX = 127, 2175  # 127 + 2048
CHIP3_Y_MIN, CHIP3_Y_MAX = 184, 2232  # 184 + 2048

# =============================================================================
# Quality Flags (following COSMOS/SDSS conventions)
# =============================================================================
"""
QUALITY FLAG DOCUMENTATION
==========================

Flags are bit-packed integers for efficient storage and combination.
Multiple flags can be set on a single source by bitwise OR.

Usage:
    - Check if flag is set: (quality_flag & FLAG_BLENDED) != 0
    - Set a flag: quality_flag |= FLAG_BLENDED
    - Clear a flag: quality_flag &= ~FLAG_BLENDED
    - Check if clean: quality_flag == FLAG_NONE

FLAG DEFINITIONS:
-----------------
FLAG_NONE (0):        Clean source, no quality issues detected
FLAG_BLENDED (1):     Source was deblended from overlapping neighbor.
                      Photometry may be affected by neighbor subtraction.
FLAG_EDGE (2):        Source near image edge. May have truncated flux
                      measurement or unreliable morphology.
FLAG_SATURATED (4):   Contains saturated pixels. Flux measurement is
                      a lower limit; morphology unreliable.
FLAG_LOW_SNR (8):     Signal-to-noise ratio < 5 in detection band.
                      Photometry and morphology may be unreliable.
FLAG_CROWDED (16):    Located in crowded region with many nearby sources.
                      Increased chance of photometric contamination.
FLAG_BAD_PHOTOZ (32): Photo-z quality ODDS < 0.6 OR redshift hit boundary.
                      Redshift estimate is unreliable.
FLAG_PSF_LIKE (64):   Source morphology is consistent with PSF (point source).
                      Likely a star or quasar, not a resolved galaxy.
FLAG_MASKED (128):    Partially overlaps with masked region (diffraction
                      spike, satellite trail, etc.). Photometry affected.
FLAG_UNRELIABLE_Z (256): Redshift error exceeds redshift value (σ_z > z).
                      Measurement uncertainty larger than measurement itself.

FILTERING MODES:
----------------
HIGH_QUALITY_MODE (default):
    Excludes: blended, edge, saturated, low_snr, crowded, bad_photoz,
              psf_like, masked, unreliable_z
    Requires: ODDS >= 0.9, chi2_flag == 0, non-bimodal redshift
    Purpose: Publication-quality sample with maximum reliability.
    Typical rejection: ~40% of raw detections.

MODERATE_MODE:
    Excludes: bad_photoz, psf_like, masked
    Purpose: Broader sample accepting more sources with potential issues.
    Use when sample size is more important than individual reliability.
"""
# Flags are bit-packed integers for efficient storage
# Use bitwise AND to check: (flag & FLAG_BLENDED) != 0
FLAG_NONE = 0           # Clean source, no issues
FLAG_BLENDED = 1        # Source was deblended from overlapping neighbor
FLAG_EDGE = 2           # Source near image edge (may have truncated flux)
FLAG_SATURATED = 4      # Contains saturated pixels
FLAG_LOW_SNR = 8        # Signal-to-noise ratio < 5
FLAG_CROWDED = 16       # Many nearby sources (crowded region)
FLAG_BAD_PHOTOZ = 32    # Photo-z ODDS < 0.6 (unreliable redshift)
FLAG_PSF_LIKE = 64      # Source is PSF-like (possible star)
FLAG_MASKED = 128       # Partially overlaps with masked region
FLAG_UNRELIABLE_Z = 256 # Redshift error exceeds redshift (σ_z > z)

# Human-readable flag descriptions
FLAG_DESCRIPTIONS = {
    FLAG_BLENDED: "blended",
    FLAG_EDGE: "edge",
    FLAG_SATURATED: "saturated",
    FLAG_LOW_SNR: "low_snr",
    FLAG_CROWDED: "crowded",
    FLAG_BAD_PHOTOZ: "bad_photoz",
    FLAG_PSF_LIKE: "psf_like",
    FLAG_MASKED: "masked",
    FLAG_UNRELIABLE_Z: "unreliable_z",
}


def decode_flags(flag_value: int) -> list[str]:
    """Convert bit-packed flag integer to list of flag names."""
    if flag_value == 0:
        return ["clean"]
    return [desc for flag, desc in FLAG_DESCRIPTIONS.items() if flag_value & flag]


def compute_concentration(r50: float, r90: float) -> float:
    """
    Compute concentration index C = r50/r90.

    Following SDSS methodology:
    - de Vaucouleurs profile (ellipticals): C ≈ 0.30
    - Exponential profile (spirals): C ≈ 0.43
    - Point source (stars): C ≈ 0.45-0.50

    Parameters
    ----------
    r50 : float
        Half-light radius (50% enclosed flux)
    r90 : float
        90% light radius

    Returns
    -------
    float
        Concentration index (0 to 1)
    """
    if r90 <= 0 or not np.isfinite(r90):
        return np.nan
    return r50 / r90


def correct_radius_for_psf(r_measured: float, psf_sigma: float) -> float:
    """
    Correct half-light radius for PSF broadening.

    Uses quadrature subtraction: r_intrinsic² ≈ r_measured² - r_psf²
    This is an approximation valid when r_measured >> r_psf.

    Parameters
    ----------
    r_measured : float
        Measured half-light radius in pixels
    psf_sigma : float
        PSF sigma in pixels (FWHM / 2.355)

    Returns
    -------
    float
        Corrected intrinsic radius (0 if smaller than PSF)
    """
    r_sq = r_measured**2 - psf_sigma**2
    return np.sqrt(max(0, r_sq))


def compute_stellarity_score(
    r_half: float,
    concentration: float,
    ellipticity: float,
    psf_fwhm_pix: float,
) -> float:
    """
    Compute stellarity index (0=galaxy, 1=star) using HST-optimized criteria.

    This provides a continuous 0-1 score based on multiple morphological criteria.
    Optimized for HST's excellent spatial resolution where SIZE is the primary
    discriminator between stars (point sources) and galaxies (extended).

    Parameters
    ----------
    r_half : float
        Half-light radius in pixels
    concentration : float
        Concentration index C = r50/r90
    ellipticity : float
        Source ellipticity (0=round, 1=elongated)
    psf_fwhm_pix : float
        PSF FWHM in pixels

    Returns
    -------
    float
        Stellarity index from 0 (definitely galaxy) to 1 (definitely star)

    References
    ----------
    - SExtractor CLASS_STAR: https://sextractor.readthedocs.io/en/latest/ClassStar.html
    - SDSS star/galaxy: https://www.sdss4.org/dr14/algorithms/classify/
    - HST Source Catalog: https://archive.stsci.edu/hst/hsc/help/HSC_faq.html
    """
    scores = []
    weights = []

    # 1. Size score (PRIMARY criterion for HST - weight 2x)
    # Stars are point sources with r_half very close to PSF
    # PSF half-light radius ≈ PSF_FWHM * 0.42 (for Gaussian)
    psf_r_half = psf_fwhm_pix * 0.42
    if np.isfinite(r_half) and r_half > 0 and psf_r_half > 0:
        size_ratio = r_half / psf_r_half
        # Score: 1 if exactly PSF size, drops rapidly for larger sources
        # 0 if 2x PSF size (much stricter than before)
        size_score = 1.0 if size_ratio <= 1.0 else max(0, 1.0 - (size_ratio - 1.0))
        scores.append(size_score)
        weights.append(2.0)  # Size is most reliable for HST

    # 2. Concentration score: stars have high concentration
    # But NOTE: elliptical galaxies also have high concentration!
    if np.isfinite(concentration):
        # Map: C=0.35 -> 0, C=0.55 -> 1
        conc_score = min(1.0, max(0, (concentration - 0.35) / 0.20))
        scores.append(conc_score)
        weights.append(1.0)

    # 3. Roundness score: stars are round (low ellipticity)
    # Map: e=0 -> 1 (round=star-like), e=0.3 -> 0 (elongated=galaxy-like)
    if np.isfinite(ellipticity):
        round_score = max(0, 1.0 - ellipticity / 0.3)
        scores.append(round_score)
        weights.append(1.0)

    if len(scores) == 0:
        return 0.0

    # Weighted average (size counts double)
    return float(np.average(scores, weights=weights))


def check_stellar_colors(flux_u: float, flux_b: float, flux_v: float, flux_i: float) -> tuple[bool, float]:
    """
    Check if source has stellar-like colors (approximately flat SED).

    Stars have relatively flat SEDs compared to galaxies, so color differences
    between adjacent bands should be small (within ~0.5 mag).

    Parameters
    ----------
    flux_u, flux_b, flux_v, flux_i : float
        Flux measurements in each band

    Returns
    -------
    is_stellar_color : bool
        True if colors are consistent with stellar SED
    color_score : float
        Score from 0 (galaxy-like colors) to 1 (stellar-like flat colors)

    References
    ----------
    - SDSS stellar locus: https://www.sdss.org/dr17/algorithms/magnitudes/
    """
    # Convert to instrumental magnitudes (relative)
    with np.errstate(divide='ignore', invalid='ignore'):
        mag_u = -2.5 * np.log10(np.abs(flux_u)) if flux_u > 0 else np.nan
        mag_b = -2.5 * np.log10(np.abs(flux_b)) if flux_b > 0 else np.nan
        mag_v = -2.5 * np.log10(np.abs(flux_v)) if flux_v > 0 else np.nan
        mag_i = -2.5 * np.log10(np.abs(flux_i)) if flux_i > 0 else np.nan

    # Color indices (adjacent bands)
    colors = []
    if np.isfinite(mag_u) and np.isfinite(mag_b):
        colors.append(mag_u - mag_b)  # U-B
    if np.isfinite(mag_b) and np.isfinite(mag_v):
        colors.append(mag_b - mag_v)  # B-V
    if np.isfinite(mag_v) and np.isfinite(mag_i):
        colors.append(mag_v - mag_i)  # V-I

    if len(colors) == 0:
        return False, 0.0

    # Stars have colors close to 0 (flat SED)
    # Galaxies have stronger color variations due to stellar populations, dust, redshift
    color_scatter = np.std(colors)
    max_color = np.max(np.abs(colors))

    # Stellar-like: small scatter AND small maximum color difference
    # Threshold: scatter < 0.5 mag AND max color < 1.0 mag
    is_stellar = color_scatter < 0.5 and max_color < 1.0

    # Score: 1 if very flat colors, 0 if strong colors
    color_score = max(0, 1.0 - color_scatter / 1.0) * max(0, 1.0 - max_color / 2.0)

    return is_stellar, float(color_score)


def check_stellar_colors_vectorized(
    flux_u: np.ndarray, flux_b: np.ndarray, flux_v: np.ndarray, flux_i: np.ndarray
) -> np.ndarray:
    """
    Vectorized stellar color check - 100-700x faster than iterrows().

    Computes color_score for arrays of fluxes using numpy operations.
    This follows pandas/numpy best practices for avoiding slow iteration.

    Parameters
    ----------
    flux_u, flux_b, flux_v, flux_i : np.ndarray
        Flux measurements in each band (1D arrays)

    Returns
    -------
    np.ndarray
        Color scores from 0 (galaxy-like) to 1 (stellar-like flat colors)

    References
    ----------
    - Pandas vectorization: ~740x faster than iterrows
    - https://towardsdatascience.com/dont-assume-numpy-vectorize-is-faster
    """
    # Convert to numpy arrays if needed
    flux_u = np.asarray(flux_u, dtype=np.float64)
    flux_b = np.asarray(flux_b, dtype=np.float64)
    flux_v = np.asarray(flux_v, dtype=np.float64)
    flux_i = np.asarray(flux_i, dtype=np.float64)

    # Vectorized magnitude calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        mag_u = np.where(flux_u > 0, -2.5 * np.log10(np.abs(flux_u)), np.nan)
        mag_b = np.where(flux_b > 0, -2.5 * np.log10(np.abs(flux_b)), np.nan)
        mag_v = np.where(flux_v > 0, -2.5 * np.log10(np.abs(flux_v)), np.nan)
        mag_i = np.where(flux_i > 0, -2.5 * np.log10(np.abs(flux_i)), np.nan)

    # Compute colors (adjacent bands)
    color_ub = mag_u - mag_b  # U-B
    color_bv = mag_b - mag_v  # B-V
    color_vi = mag_v - mag_i  # V-I

    # Stack colors into 2D array for vectorized statistics
    # Shape: (n_sources, 3)
    colors = np.column_stack([color_ub, color_bv, color_vi])

    # Compute scatter (std) and max color per source
    # Use nanstd/nanmax to handle NaN values
    color_scatter = np.nanstd(colors, axis=1)
    max_color = np.nanmax(np.abs(colors), axis=1)

    # Handle case where all colors are NaN (return 0 score)
    all_nan = np.all(np.isnan(colors), axis=1)

    # Compute color score: 1 if flat colors, 0 if strong colors
    color_score = np.maximum(0, 1.0 - color_scatter / 1.0) * np.maximum(0, 1.0 - max_color / 2.0)
    color_score = np.where(all_nan, 0.0, color_score)

    return color_score


# =============================================================================
# Color-Color Diagram Plotting
# =============================================================================

# Galaxy type color scheme for consistent visualization across plots
GALAXY_TYPE_COLORS = {
    "elliptical": "#E31A1C",  # Red - early type
    "S0": "#FF7F00",          # Orange - lenticular
    "Sa": "#FDBF6F",          # Light orange - early spiral
    "Sb": "#33A02C",          # Green - spiral
    "sbt1": "#B2DF8A",        # Light green - starburst
    "sbt2": "#1F78B4",        # Blue - starburst
    "sbt3": "#A6CEE3",        # Light blue - starburst
    "sbt4": "#6A3D9A",        # Purple - starburst
    "sbt5": "#CAB2D6",        # Light purple - starburst
    "sbt6": "#B15928",        # Brown - starburst
}

# Fallback color for unknown types
GALAXY_TYPE_FALLBACK_COLOR = "#999999"


def get_adaptive_n_bins(n_sources: int, min_per_bin: int = 5) -> int:
    """
    Calculate adaptive number of bins based on data size.

    Uses Freedman-Diaconis-inspired rule: aim for ~sqrt(N) bins,
    but ensure at least min_per_bin sources per bin.
    """
    # Target: sqrt(N) bins, but with constraints
    n_bins_target = int(np.sqrt(n_sources))
    # Ensure at least min_per_bin sources per bin
    n_bins_max = n_sources // min_per_bin
    # Practical bounds: 3 to 20 bins
    n_bins = max(3, min(n_bins_target, n_bins_max, 20))
    return n_bins


def bin_equal_width(sed_catalog, n_bins=None):
    """Equal-width bins in redshift space."""
    if n_bins is None:
        n_bins = get_adaptive_n_bins(len(sed_catalog))
    z_min, z_max = sed_catalog.redshift.min(), sed_catalog.redshift.max()
    bins = np.linspace(z_min, z_max, n_bins + 1)
    sed_catalog = sed_catalog.copy()
    sed_catalog["z_bin"] = pd.cut(sed_catalog.redshift, bins, include_lowest=True)
    return aggregate_bins(sed_catalog)


def bin_equal_count(sed_catalog, n_bins=None):
    """Equal-count bins (quantile-based) for similar statistical weight per bin."""
    if n_bins is None:
        n_bins = get_adaptive_n_bins(len(sed_catalog))
    sed_catalog = sed_catalog.copy()
    sed_catalog["z_bin"] = pd.qcut(sed_catalog.redshift, q=n_bins, duplicates="drop")
    return aggregate_bins(sed_catalog)


def bin_logarithmic(sed_catalog, n_bins=None):
    """Logarithmic bins in (1+z) - better sampling at low-z where evolution is rapid."""
    if n_bins is None:
        n_bins = get_adaptive_n_bins(len(sed_catalog))
    z_min, z_max = sed_catalog.redshift.min(), sed_catalog.redshift.max()
    log_bins = np.linspace(np.log10(1 + z_min), np.log10(1 + z_max), n_bins + 1)
    bins = 10**log_bins - 1
    sed_catalog = sed_catalog.copy()
    sed_catalog["z_bin"] = pd.cut(sed_catalog.redshift, bins, include_lowest=True)
    return aggregate_bins(sed_catalog)


def bin_percentile(sed_catalog, n_bins=None):
    """Percentile-based bins using quantile edges for robust statistics."""
    if n_bins is None:
        n_bins = get_adaptive_n_bins(len(sed_catalog))
    sed_catalog = sed_catalog.copy()
    bins = choose_redshift_bin_edges(
        sed_catalog.redshift.values,
        target_count_per_bin=len(sed_catalog) // n_bins,
        min_bins=max(3, n_bins - 2),
        max_bins=n_bins + 2,
        method="quantile",
    )
    sed_catalog["z_bin"] = pd.cut(sed_catalog.redshift, bins, include_lowest=True)
    return aggregate_bins(sed_catalog)


def standard_error_median(x):
    """Standard error of the median using analytical approximation.

    For approximately normal data: SE(median) ~ 1.2533 * sigma / sqrt(n)
    This is the standard approach used in professional astronomy surveys.
    Reference: Euclid Consortium methodology, Astropy robust statistics.
    """
    x = x.dropna() if hasattr(x, 'dropna') else x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    return 1.2533 * np.std(x, ddof=1) / np.sqrt(len(x))


def aggregate_bins(sed_catalog):
    """Aggregate binned data into summary statistics.

    Uses professional astronomy standards:
    - Median for central tendency (robust to outliers)
    - Standard error of the median for error bars (not RMS of individual errors)
    - 16th/84th percentiles for 1-sigma spread visualization

    References:
    - Euclid tomographic binning methodology
    - JWST angular size-redshift test (arXiv:2507.19651)
    - Astropy robust statistical estimators
    """
    binned = (
        sed_catalog.groupby("z_bin", observed=False)
        .agg(
            z_mid=("redshift", "median"),
            theta_med=("r_half_arcsec", "median"),
            theta_err=("r_half_arcsec", standard_error_median),
            theta_lo=("r_half_arcsec", lambda x: np.percentile(x, 16) if len(x) >= 2 else np.nan),
            theta_hi=("r_half_arcsec", lambda x: np.percentile(x, 84) if len(x) >= 2 else np.nan),
            n=("redshift", "count"),
        )
        .reset_index(drop=True)
    )
    binned = binned.dropna(subset=["z_mid", "theta_med", "theta_err"])
    binned = binned[(binned.n > 0) & (binned.z_mid > 0)]
    return binned


@dataclass
class AstroImage:
    data: np.ndarray
    header: fits.Header
    band: str = ""
    weight: np.ndarray | None = None  # Inverse variance weight map
    segm: object | None = None
    catalog: SourceCatalog | None = None
    bkg: Background2D | None = None

    def get_variance(self) -> np.ndarray:
        """Get per-pixel variance from weight map (inverse variance)."""
        if self.weight is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                variance = np.where(self.weight > 0, 1.0 / self.weight, np.inf)
            return variance
        return None

    def release_heavy_data(self):
        """Release memory-heavy attributes after catalog extraction."""
        self.weight = None
        self.segm = None
        self.catalog = None
        self.bkg = None
        # Keep data for plotting, but could also set to None if not needed


def adjust_data(data):
    """Remove black/low-value edges from image.

    For original 2048x2048 data: crop edges and flip
    For full 4096x4096 mosaics: no adjustment needed (properly drizzled)
    """
    if data.shape == (2048, 2048):
        # Original data needs edge trimming and flip
        data = data[120:, 90:]
        data = data[:, ::-1]
    # 4096x4096 full mosaics are already properly aligned
    return data


def read_fits(file_path, weight_path=None):
    """Read FITS file with optional weight map.

    Parameters
    ----------
    file_path : str
        Path to science FITS file
    weight_path : str, optional
        Path to weight (inverse variance) FITS file

    Returns
    -------
    data : np.ndarray
        Science image data
    header : fits.Header
        FITS header
    weight : np.ndarray or None
        Weight map (inverse variance) if provided
    """
    # Use memory mapping for large files to reduce memory footprint
    # memmap='r' loads data lazily, only reading chunks as needed
    with fits.open(file_path, memmap=True) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        # Use native byte order for efficient numpy operations
        # Note: This copies the data, but ensures optimal memory layout
        if data.dtype.byteorder == ">":
            data = data.astype(data.dtype.newbyteorder("="), copy=False)
        else:
            # Force copy from mmap to regular array for subsequent operations
            data = np.array(data)
        data = adjust_data(data)

    # Load weight map if provided
    weight = None
    if weight_path is not None:
        try:
            with fits.open(weight_path, memmap=True) as hdul_w:
                weight = hdul_w[0].data
                if weight.dtype.byteorder == ">":
                    weight = weight.astype(weight.dtype.newbyteorder("="), copy=False)
                else:
                    weight = np.array(weight)
                weight = adjust_data(weight)
        except FileNotFoundError:
            print(f"  Warning: Weight file not found: {weight_path}")
            weight = None

    return data, header, weight


# ============================================================================
# Advanced background estimation (following photutils best practices)
# ============================================================================


def iterative_background(data, box_size=100, n_iterations=2, sigma_clip=3.0, mask=None):
    """
    Iteratively estimate background with source masking.

    Following the photutils documentation recommendation:
    "An even better procedure is to exclude the sources in the image by masking them.
    This technique requires an iterative procedure."

    Parameters
    ----------
    data : np.ndarray
        Input image data
    box_size : int
        Box size for background estimation
    n_iterations : int
        Number of source-masking iterations
    sigma_clip : float
        Sigma threshold for source detection in each iteration
    mask : np.ndarray, optional
        Initial mask (True = masked pixels)

    Returns
    -------
    Background2D
        Final background estimate with source masking
    """
    from photutils.background import Background2D, MedianBackground

    bkg_estimator = MedianBackground()
    current_mask = mask.copy() if mask is not None else np.zeros(data.shape, dtype=bool)

    for _iteration in range(n_iterations):
        # Estimate background with current mask
        bkg = Background2D(
            data,
            (box_size, box_size),
            filter_size=(3, 3),
            bkg_estimator=bkg_estimator,
            mask=current_mask,
        )

        # Detect sources above threshold
        residual = data - bkg.background
        threshold = sigma_clip * bkg.background_rms
        source_mask = residual > threshold

        # Update mask by combining with detected sources
        current_mask = current_mask | source_mask

    # Final background estimate with all sources masked
    final_bkg = Background2D(
        data,
        (box_size, box_size),
        filter_size=(3, 3),
        bkg_estimator=bkg_estimator,
        mask=current_mask,
    )

    return final_bkg


# ============================================================================
# Parallel processing helper functions
# ============================================================================


def _process_single_band(args):
    """Process a single band for source detection (for parallel execution).

    This function is designed to be called in a separate process.
    """
    band, data, _weight, mask, kernel, detection_sigma, npixels, nlevels, contrast = args

    from astropy.convolution import convolve
    from photutils.background import Background2D, MedianBackground
    from photutils.segmentation import SourceCatalog, SourceFinder

    bkg_estimator = MedianBackground()
    box_size = 100 if data.shape[0] > 2048 else 50
    bkg = Background2D(data, (box_size, box_size), filter_size=(3, 3), bkg_estimator=bkg_estimator)
    data_sub = data - bkg.background

    # Convolve with Gaussian kernel
    convolved_data = convolve(data_sub, kernel)

    # Adaptive threshold based on background RMS
    threshold = detection_sigma * bkg.background_rms

    # Use SourceFinder for combined detection + deblending
    finder = SourceFinder(
        npixels=npixels,
        deblend=True,
        nlevels=nlevels,
        contrast=contrast,
        connectivity=8,
        progress_bar=False,
    )

    segm = finder(convolved_data, threshold, mask=mask)

    if segm is not None and mask is not None:
        masked_labels = np.unique(segm.data[mask])
        masked_labels = masked_labels[masked_labels != 0]
        if len(masked_labels) > 0:
            segm.remove_labels(masked_labels)

    catalog = SourceCatalog(data_sub, segm, convolved_data=convolved_data)
    n_sources = len(catalog) if catalog is not None else 0

    return band, segm, catalog, bkg, data_sub, convolved_data, n_sources


def _classify_single_galaxy(args):
    """Classify a single galaxy (for parallel execution)."""
    idx, fluxes, errors, spectra_path = args
    galaxy_type, redshift = classify_galaxy(fluxes, errors, spectra_path=spectra_path)
    return idx, galaxy_type, redshift


def _classify_galaxy_batch(args):
    """Classify a batch of galaxies with PDF uncertainties (reduces IPC overhead)."""
    batch, spectra_path = args

    # Use the new PDF-based batch classification
    # Returns: (idx, galaxy_type, redshift, z_lo, z_hi, chi_sq_min, odds)
    results = classify_galaxy_batch_with_pdf(batch, spectra_path)
    return results


def analyze_and_plot_catalog(
    sed_catalog: pd.DataFrame,
    sed_catalog_filtered: pd.DataFrame,
    output_dir: str,
    title_suffix: str = "",
    images: list | None = None,
    crop_bounds: tuple | None = None,
    star_mask_centers: list | None = None,
):
    """
    Analyze catalog data and generate plots without re-running detection.

    This function takes existing catalog data and generates analysis plots.
    Used for chip3 extraction from full mosaic results.

    Args:
        sed_catalog: Full catalog DataFrame
        sed_catalog_filtered: Quality-filtered catalog DataFrame
        output_dir: Directory to save outputs
        title_suffix: Optional suffix for plot titles (e.g., " (Chip 3)")
        images: Optional list of AstroImage objects for detection plots
        crop_bounds: Optional tuple (x_min, x_max, y_min, y_max) to crop images and adjust coordinates
        star_mask_centers: Optional list of (x, y) centers for star mask regions
    """
    if star_mask_centers is None:
        star_mask_centers = []
    os.makedirs(output_dir, exist_ok=True)

    sed_catalog_analysis = sed_catalog_filtered

    if len(sed_catalog_analysis) == 0:
        print("  No sources to analyze after filtering!")
        return

    print(f"\n  Analyzing {len(sed_catalog_analysis)} quality-filtered sources...")

    # Redshift statistics
    print("\n  Redshift statistics (filtered):")
    print(f"    Min: {sed_catalog_analysis['redshift'].min():.3f}")
    print(f"    Max: {sed_catalog_analysis['redshift'].max():.3f}")
    print(f"    Mean: {sed_catalog_analysis['redshift'].mean():.3f}")
    print(f"    Median: {sed_catalog_analysis['redshift'].median():.3f}")

    # Galaxy type distribution
    print("\n  Galaxy type distribution:")
    type_counts = sed_catalog_analysis["galaxy_type"].value_counts()
    for gtype, count in type_counts.items():
        print(f"    {gtype}: {count}")

    # Angular size statistics
    print("\n  Angular size statistics (PSF-corrected):")
    print(f"    Min: {sed_catalog_analysis['r_half_arcsec'].min():.4f} arcsec")
    print(f"    Max: {sed_catalog_analysis['r_half_arcsec'].max():.4f} arcsec")
    print(f"    Median: {sed_catalog_analysis['r_half_arcsec'].median():.4f} arcsec")

    # Binning - use same 4 strategies as full analysis
    binning_strategies = {
        "Equal Width": bin_equal_width,
        "Equal Count": bin_equal_count,
        "Logarithmic": bin_logarithmic,
        "Percentile": bin_percentile,
    }

    binned_results = {}
    for name, bin_func in binning_strategies.items():
        try:
            binned_results[name] = bin_func(sed_catalog_analysis)
            print(f"\n  {name} binning ({len(binned_results[name])} bins):")
            print(binned_results[name][["z_mid", "theta_med", "theta_err", "n"]].to_string())
        except Exception as e:
            print(f"\n  {name} binning failed: {e}")

    if not binned_results:
        print("  No valid binning results!")
        return

    binned = binned_results.get("Equal Count", next(iter(binned_results.values())))

    # Model fitting
    z_data = binned["z_mid"].values
    theta_data = binned["theta_med"].values
    theta_err = binned["theta_err"].values

    # Default fallback values
    R_static = 0.003
    R_lcdm = 0.001
    Omega_m_fit = 0.3

    try:
        R_static_fit = get_radius(z_data, theta_data / RAD_TO_ARCSEC, theta_err / RAD_TO_ARCSEC)
        R_lcdm_fit, Omega_m_result = get_radius_and_omega(z_data, theta_data / RAD_TO_ARCSEC, theta_err / RAD_TO_ARCSEC)

        # Ensure scalars
        if hasattr(R_static_fit, '__iter__'):
            R_static_fit = float(np.asarray(R_static_fit).flat[0])
        if hasattr(R_lcdm_fit, '__iter__'):
            R_lcdm_fit = float(np.asarray(R_lcdm_fit).flat[0])
        if hasattr(Omega_m_result, '__iter__'):
            Omega_m_result = float(np.asarray(Omega_m_result).flat[0])

        # Use fitted values if reasonable
        if np.isfinite(R_static_fit) and R_static_fit > 1e-6:
            R_static = float(R_static_fit)
        if np.isfinite(R_lcdm_fit) and R_lcdm_fit > 1e-6:
            R_lcdm = float(R_lcdm_fit)
        if np.isfinite(Omega_m_result) and 0 < Omega_m_result < 1:
            Omega_m_fit = float(Omega_m_result)
    except Exception as e:
        print(f"    Model fitting failed: {e}")

    print("\n  Fitted parameters:")
    print(f"    Static model: R = {R_static*1000:.2f} kpc")
    print(f"    ΛCDM model:   R = {R_lcdm*1000:.2f} kpc, Ω_m = {Omega_m_fit:.3f}")

    # Model curves
    z_model = np.linspace(0.01, max(2.5, sed_catalog_analysis["redshift"].max() * 1.2), 200)
    theta_static_model = theta_static(z_model, R_static) * RAD_TO_ARCSEC
    theta_lcdm_model = theta_lcdm_flat(z_model, R_lcdm, Omega_m_fit) * RAD_TO_ARCSEC

    # Generate plots
    print("\n  Creating plots...")

    # Axis limits
    z_data_min, z_data_max = sed_catalog_analysis["redshift"].min(), sed_catalog_analysis["redshift"].max()
    theta_data_min, theta_data_max = sed_catalog_analysis["r_half_arcsec"].min(), sed_catalog_analysis["r_half_arcsec"].max()
    z_padding = (z_data_max - z_data_min) * 0.08
    theta_padding = (theta_data_max - theta_data_min) * 0.1
    z_lim = (max(0, z_data_min - z_padding), z_data_max + z_padding)
    theta_lim = (max(0, theta_data_min - theta_padding), theta_data_max + theta_padding)

    # Plot 1: Angular size vs redshift
    _fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(binned["z_mid"], binned["theta_med"], yerr=binned["theta_err"],
                fmt="o", capsize=3, markersize=7, color="black",
                label=f"Median angular size (N={len(sed_catalog_analysis)})")
    ax.plot(z_model, theta_static_model, "--", color="gray", linewidth=2, label=r"Linear Hubble law")
    ax.plot(z_model, theta_lcdm_model, "-", color="blue", linewidth=2,
            label=r"$\Lambda$CDM ($\Omega_m$=" + f"{Omega_m_fit:.2f})")
    ax.set_xlabel(r"Redshift $z$", fontsize=12)
    ax.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=12)
    ax.set_title(r"Galaxy Angular Size vs Redshift" + title_suffix, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(z_lim)
    ax.set_ylim(theta_lim)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/angular_size_vs_redshift.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_dir}/angular_size_vs_redshift.pdf")

    # Plot 2: Individual galaxies
    _fig, ax = plt.subplots(figsize=(12, 7))

    # Get unique galaxy types and create color mapping
    galaxy_type_cat = sed_catalog_analysis["galaxy_type"].astype("category")
    unique_types = galaxy_type_cat.cat.categories.tolist()
    cmap = plt.cm.tab10

    # Create scatter plot
    ax.scatter(
        sed_catalog_analysis["redshift"],
        sed_catalog_analysis["r_half_arcsec"],
        c=galaxy_type_cat.cat.codes,
        cmap=cmap,
        alpha=0.6,
        s=20,
    )

    ax.plot(z_model, theta_static_model, "--", color="gray", linewidth=2, alpha=0.7)
    ax.plot(z_model, theta_lcdm_model, "-", color="blue", linewidth=2, alpha=0.7)
    ax.set_xlabel(r"Redshift $z$", fontsize=12)
    ax.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=12)
    ax.set_title(r"Individual Galaxy Sizes" + title_suffix, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(z_lim)
    ax.set_ylim(theta_lim)

    # Create legend for galaxy types
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i / len(unique_types)),
               markersize=8, alpha=0.8, label=gtype)
        for i, gtype in enumerate(unique_types)
    ]
    ax.legend(handles=legend_handles, title="Galaxy Type", loc="upper right",
              fontsize=9, title_fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/individual_galaxies.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_dir}/individual_galaxies.pdf")

    # Plot 3: Binning strategy comparison (2x2 grid with residuals) - matches full version exactly
    fig = plt.figure(figsize=(14, 16), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 3, 1], hspace=0.08, wspace=0.15)

    colors = {
        "Equal Width": "green",
        "Equal Count": "blue",
        "Logarithmic": "orange",
        "Percentile": "purple",
    }
    markers = {"Equal Width": "o", "Equal Count": "o", "Logarithmic": "o", "Percentile": "o"}

    strategy_names = list(binned_results.keys())

    for idx, name in enumerate(strategy_names):
        binned_data = binned_results[name]
        # Grid position: first two strategies in rows 0-1, last two in rows 2-3
        row_base = (idx // 2) * 2  # 0 or 2 for main plots
        col = idx % 2

        ax_main = fig.add_subplot(gs[row_base, col])
        ax_resid = fig.add_subplot(gs[row_base + 1, col], sharex=ax_main)

        # Fit models for this binning
        R_static_i = get_radius(
            binned_data.z_mid.values,
            binned_data.theta_med.values,
            binned_data.theta_err.values,
            model="static",
        )
        # Fit both R and Omega_m for flat LCDM
        R_lcdm_i, Omega_m_i = get_radius_and_omega(
            binned_data.z_mid.values,
            binned_data.theta_med.values,
            binned_data.theta_err.values,
        )

        # Use full z range for model curves (consistent across all panels)
        z_model_i = np.linspace(max(0.05, z_lim[0]), z_lim[1], 200)
        theta_static_i = theta_static(z_model_i, R_static_i) * RAD_TO_ARCSEC
        theta_lcdm_i = theta_lcdm_flat(z_model_i, R_lcdm_i, Omega_m_i) * RAD_TO_ARCSEC

        # Compute chi2/ndf for both models
        def static_func_i(z, R=R_static_i):
            return theta_static(z, R) * RAD_TO_ARCSEC
        def lcdm_func_i(z, R=R_lcdm_i, Om=Omega_m_i):
            return theta_lcdm_flat(z, R, Om) * RAD_TO_ARCSEC

        stats_static_i = compute_chi2_stats(
            binned_data.z_mid.values, binned_data.theta_med.values,
            binned_data.theta_err.values, static_func_i, n_params=1
        )
        stats_lcdm_i = compute_chi2_stats(
            binned_data.z_mid.values, binned_data.theta_med.values,
            binned_data.theta_err.values, lcdm_func_i, n_params=2
        )

        # Plot individual galaxies as background
        ax_main.scatter(
            sed_catalog_analysis["redshift"],
            sed_catalog_analysis["r_half_arcsec"],
            c="lightgray",
            alpha=0.4,
            s=15,
            zorder=1,
        )

        # Plot binned data (no connecting lines)
        ax_main.errorbar(
            binned_data["z_mid"],
            binned_data["theta_med"],
            yerr=binned_data["theta_err"],
            fmt=markers[name],
            capsize=3,
            capthick=0.8,
            markersize=6,
            markeredgewidth=0.8,
            elinewidth=0.8,
            color=colors[name],
            label="Binned",
            zorder=3,
        )

        # Plot models
        ax_main.plot(
            z_model_i, theta_static_i, "--", color="gray", linewidth=1.5, label=r"Static", zorder=2
        )
        ax_main.plot(
            z_model_i, theta_lcdm_i, "-", color="darkblue", linewidth=1.5, label=r"$\Lambda$CDM", zorder=2
        )

        ax_main.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=11)
        ax_main.set_title(
            f"{name} ({len(binned_data)} bins)\n"
            + r"$\Lambda$CDM: $R$" + f"={R_lcdm_i*1000:.1f} kpc, "
            + r"$\chi^2/{\rm ndf}$" + f"={stats_lcdm_i['chi2_ndf']:.2f}, "
            + f"P={stats_lcdm_i['p_value']:.3f}\n"
            + r"Static: $R$" + f"={R_static_i*1000:.1f} kpc, "
            + r"$\chi^2/{\rm ndf}$" + f"={stats_static_i['chi2_ndf']:.2f}, "
            + f"P={stats_static_i['p_value']:.3f}",
            fontsize=10
        )
        ax_main.legend(fontsize=8, loc="upper right")
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xlim(z_lim)
        ax_main.set_ylim(theta_lim)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # Calculate residuals: (data - LCDM model) in arcsec
        theta_lcdm_at_data = theta_lcdm_flat(binned_data["z_mid"].values, R_lcdm_i, Omega_m_i) * RAD_TO_ARCSEC
        residuals_lcdm = binned_data["theta_med"].values - theta_lcdm_at_data

        # Residual plot
        ax_resid.axhline(0, color="darkblue", linestyle="-", linewidth=1.5, alpha=0.7)
        ax_resid.errorbar(
            binned_data["z_mid"],
            residuals_lcdm,
            yerr=binned_data["theta_err"],
            fmt=markers[name],
            capsize=2,
            capthick=0.6,
            markersize=5,
            markeredgewidth=0.6,
            elinewidth=0.6,
            color=colors[name],
            zorder=3,
        )
        ax_resid.set_xlabel(r"Redshift $z$", fontsize=11)
        ax_resid.set_ylabel(r"$\theta - \theta_{\Lambda\mathrm{CDM}}$", fontsize=9)
        ax_resid.grid(True, alpha=0.3)
        ax_resid.set_xlim(z_lim)
        max_resid = max(0.05, np.max(np.abs(residuals_lcdm) + binned_data["theta_err"].values) * 1.2)
        ax_resid.set_ylim(-max_resid, max_resid)

    fig.suptitle(r"Binning Strategies: $\Lambda$CDM vs Static" + title_suffix, fontsize=14, fontweight="bold")
    plt.savefig(f"{output_dir}/binning_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_dir}/binning_comparison.pdf")

    # Plot 4: All binning strategies overlaid
    _fig, ax = plt.subplots(figsize=(12, 8))

    # Background scatter
    ax.scatter(
        sed_catalog_analysis["redshift"],
        sed_catalog_analysis["r_half_arcsec"],
        c="lightgray",
        alpha=0.3,
        s=15,
        label="Individual galaxies",
        zorder=1,
    )

    for name, binned_data in binned_results.items():
        ax.errorbar(
            binned_data["z_mid"],
            binned_data["theta_med"],
            yerr=binned_data["theta_err"],
            fmt=f"{markers[name]}-",
            capsize=2,
            capthick=0.6,
            markersize=5,
            markeredgewidth=0.6,
            elinewidth=0.6,
            linewidth=1.0,
            color=colors[name],
            label=f"{name}",
            alpha=0.8,
            zorder=2,
        )

    # Add LCDM model curve (using equal-count fit)
    ax.plot(
        z_model, theta_lcdm_model, "-", color="black", linewidth=2, label=r"$\Lambda$CDM model", zorder=3
    )
    ax.plot(
        z_model,
        theta_static_model,
        "--",
        color="black",
        linewidth=2,
        label=r"Static model",
        zorder=3,
    )

    ax.set_xlabel(r"Redshift $z$", fontsize=12)
    ax.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=12)
    ax.set_title(r"All Binning Strategies Compared" + title_suffix, fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(z_lim)
    ax.set_ylim(theta_lim)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/binning_overlay.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_dir}/binning_overlay.pdf")

    # Plot 5: Redshift histogram with dynamic limits
    _fig, ax = plt.subplots(figsize=(10, 6))
    z_hist_min, z_hist_max = sed_catalog_analysis["redshift"].min(), sed_catalog_analysis["redshift"].max()
    z_hist_padding = (z_hist_max - z_hist_min) * 0.05
    hist_bins = np.linspace(z_hist_min, z_hist_max, 21)
    counts, _, _ = ax.hist(sed_catalog_analysis["redshift"], bins=hist_bins, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel(r"Redshift $z$", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Redshift Distribution{title_suffix}", fontsize=14)
    ax.set_xlim(z_hist_min - z_hist_padding, z_hist_max + z_hist_padding)
    ax.set_ylim(0, np.max(counts) * 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/redshift_histogram.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_dir}/redshift_histogram.pdf")

    # Plot 6: Galaxy types with dynamic y-axis
    _fig, ax = plt.subplots(figsize=(10, 6))
    type_counts.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel("Galaxy Type", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Galaxy Type Distribution{title_suffix}", fontsize=14)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    max_count = type_counts.max()
    ax.set_ylim(0, max_count * 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/galaxy_types.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_dir}/galaxy_types.pdf")

    # Plot 7: Angular diameter distance with dynamic limits
    _fig, ax = plt.subplots(figsize=(10, 7))
    # Use data redshift range for plotting
    z_max_data = sed_catalog_analysis["redshift"].max()
    z_plot_max = max(2.5, z_max_data * 1.2)
    z_plot = np.linspace(0.01, z_plot_max, 200)
    # Ensure Omega_m_fit is a scalar float
    Omega_m_plot = float(Omega_m_fit) if not isinstance(Omega_m_fit, float) else Omega_m_fit
    D_A_model = D_A_LCDM_vectorized(z_plot, Omega_m_plot, 1.0 - Omega_m_plot)
    ax.plot(z_plot, D_A_model, "-", color="blue", linewidth=2, label=r"$\Lambda$CDM ($\Omega_m$=" + f"{Omega_m_fit:.2f})")
    D_A_std = D_A_LCDM_vectorized(z_plot, 0.3, 0.7)
    ax.plot(z_plot, D_A_std, "--", color="gray", linewidth=2, label=r"$\Lambda$CDM ($\Omega_m$=0.3)")
    ax.set_xlabel(r"Redshift $z$", fontsize=12)
    ax.set_ylabel(r"Angular Diameter Distance $D_A$ (Mpc)", fontsize=12)
    ax.set_title(f"Angular Diameter Distance{title_suffix}", fontsize=14)
    ax.legend(fontsize=10)
    # Dynamic axis limits
    D_A_max = max(np.max(D_A_model), np.max(D_A_std))
    ax.set_xlim(0, z_plot_max * 1.02)
    ax.set_ylim(0, D_A_max * 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/angular_diameter_distance.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_dir}/angular_diameter_distance.pdf")

    # Plot 8 & 9: Source detection and final selection (if images provided)
    if images is not None and len(images) > 0:
        # Prepare catalog for plotting - adjust coordinates if cropping
        plot_catalog = sed_catalog.copy()
        if crop_bounds is not None:
            x_min, x_max, y_min, y_max = crop_bounds
            plot_catalog["xcentroid"] = plot_catalog["xcentroid"] - x_min
            plot_catalog["ycentroid"] = plot_catalog["ycentroid"] - y_min

        # Classify sources for plotting
        is_in_mask = plot_catalog["in_star_mask"] if "in_star_mask" in plot_catalog.columns else pd.Series([False] * len(plot_catalog))
        is_psf_like = (plot_catalog["quality_flag"].astype(int) & FLAG_PSF_LIKE) != 0
        is_detected_star = is_psf_like & ~is_in_mask
        is_galaxy = ~is_psf_like & ~is_in_mask

        n_galaxies = is_galaxy.sum()
        is_detected_star.sum()
        is_in_mask.sum()

        # Crop images if bounds provided
        plot_images = []
        for img in images:
            if crop_bounds is not None:
                x_min, x_max, y_min, y_max = crop_bounds
                cropped_data = img.data[y_min:y_max, x_min:x_max].copy()
                # Create a simple object to hold cropped data and band
                class CroppedImage:
                    def __init__(self, data, band):
                        self.data = data
                        self.band = band
                plot_images.append(CroppedImage(cropped_data, img.band))
            else:
                plot_images.append(img)

        # Plot 8: Source detection
        _fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        for idx, image in enumerate(plot_images):
            ax = axes[idx // 2, idx % 2]
            vmin, vmax = np.percentile(image.data, [20, 99])
            ax.imshow(image.data, cmap="gray_r", vmin=vmin, vmax=vmax, origin="lower")

            # Red: galaxies
            if n_galaxies > 0:
                ax.scatter(
                    plot_catalog.loc[is_galaxy, "xcentroid"],
                    plot_catalog.loc[is_galaxy, "ycentroid"],
                    s=10, facecolors="none", edgecolors="red", linewidth=0.5,
                )

            ax.set_title(
                f"Band {image.band.upper()} -- {n_galaxies} galaxies",
                fontsize=9,
            )
            ax.set_xlabel(r"X (pixels)")
            ax.set_ylabel(r"Y (pixels)")

        plt.suptitle(f"Source Detection{title_suffix} ({n_galaxies} galaxies)", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/source_detection.pdf", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {output_dir}/source_detection.pdf")

        # Plot 9: Final selection (same as source detection but different title)
        _fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        for idx, image in enumerate(plot_images):
            ax = axes[idx // 2, idx % 2]
            vmin, vmax = np.percentile(image.data, [20, 99])
            ax.imshow(image.data, cmap="gray_r", vmin=vmin, vmax=vmax, origin="lower")

            if n_galaxies > 0:
                ax.scatter(
                    plot_catalog.loc[is_galaxy, "xcentroid"],
                    plot_catalog.loc[is_galaxy, "ycentroid"],
                    s=10, facecolors="none", edgecolors="red", linewidth=0.5,
                )

            ax.set_title(
                f"Band {image.band.upper()} -- {n_galaxies} galaxies",
                fontsize=9,
            )
            ax.set_xlabel(r"X (pixels)")
            ax.set_ylabel(r"Y (pixels)")

        plt.suptitle(f"Final Selection{title_suffix} ({n_galaxies} galaxies)", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/final_selection.pdf", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {output_dir}/final_selection.pdf")

    # Save catalogs
    sed_catalog.to_csv(f"{output_dir}/galaxy_catalog_full.csv", index=False)
    sed_catalog_analysis.to_csv(f"{output_dir}/galaxy_catalog.csv", index=False)
    print(f"    Saved: {output_dir}/galaxy_catalog_full.csv ({len(sed_catalog)} sources)")
    print(f"    Saved: {output_dir}/galaxy_catalog.csv ({len(sed_catalog_analysis)} filtered)")

    print(f"\n  Analysis complete for{title_suffix.lower() if title_suffix else ' catalog'}!")


def main(mode: str = "full", output_subdir: str | None = None):
    """
    Run the analysis pipeline.

    Args:
        mode: "full" for entire 4096x4096 image, "chip3" for 2048x2048 chip3 only
        output_subdir: Optional subdirectory within output/{mode}_HDF/ (e.g., "full_z")
    """
    # Set output directory based on mode and optional subdirectory
    # Mode naming: full -> full_HDF, chip3 -> chip3_HDF
    mode_dir = f"{mode}_HDF"
    output_dir = f"./output/{mode_dir}/{output_subdir}" if output_subdir else f"./output/{mode_dir}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("ANGULAR SIZE TEST ANALYSIS")
    print(f"Mode: {mode}")
    if output_subdir:
        print(f"Output: {output_dir}")
    print("=" * 60)

    # Step 1: Load FITS files
    print("\n[1/6] Loading FITS data...")

    # Configuration based on mode
    USE_FULL_RESOLUTION = (mode == "full")

    if USE_FULL_RESOLUTION:
        # Full-resolution HDF v2 mosaics (4096x4096) with inverse variance weights
        files = {
            "b": {"science": "./fits/b_full.fits", "weight": "./fits/b_weight.fits"},
            "i": {"science": "./fits/i_full.fits", "weight": "./fits/i_weight.fits"},
            "u": {"science": "./fits/u_full.fits", "weight": "./fits/u_weight.fits"},
            "v": {"science": "./fits/v_full.fits", "weight": "./fits/v_weight.fits"},
        }
        print("  Using full-resolution 4096x4096 data with weight maps")
    else:
        # Chip 3 2048x2048 data (no matching weight maps available)
        # Filter mapping: f300=U, f450=B, f606=V, f814=I
        files = {
            "b": {"science": "./fits_official/f450_3_v2.fits", "weight": None},
            "i": {"science": "./fits_official/f814_3_v2.fits", "weight": None},
            "u": {"science": "./fits_official/f300_3_v2.fits", "weight": None},
            "v": {"science": "./fits_official/f606_3_v2.fits", "weight": None},
        }
        print("  Using chip3 2048x2048 data (no weight maps)")

    images = []
    use_weights = True  # Set to False to disable weight-based errors

    for band, paths in files.items():
        weight_path = paths["weight"] if use_weights else None
        data, header, weight = read_fits(paths["science"], weight_path)

        if weight is not None:
            print(f"  Loaded {band} band - shape: {data.shape} (with weight map)")
        else:
            print(f"  Loaded {band} band - shape: {data.shape} (no weight map)")

        images.append(AstroImage(data=data, header=header, band=band, weight=weight))

    # Extract WCS from first image for coordinate transformations
    reference_header = images[0].header
    reference_wcs = None
    try:
        from astropy.wcs import WCS as AstropyWCS
        reference_wcs = AstropyWCS(reference_header)
        if reference_wcs.has_celestial:
            print(f"  Extracted WCS from {next(iter(files.keys()))} band")
        else:
            reference_wcs = None
            print("  WCS found but no celestial coordinates")
    except Exception as e:
        print(f"  Could not extract WCS: {e}")

    # Load or create mask based on image size
    image_shape = images[0].data.shape

    # Load star mask for flagging sources as stars
    # Sources inside this mask will be flagged as stars (from course staff)
    # For 4096x4096 mosaic: use WCS-transformed mask (star_mask_mosaic.fits)
    # For 2048x2048 chip 3: use original star mask (star_mask.fits)
    star_mask = None
    if image_shape == (4096, 4096):
        star_mask_path = "./data/star_mask_mosaic.fits"
    else:
        star_mask_path = "./data/star_mask.fits"
    try:
        star_mask_data, _, _ = read_fits(star_mask_path)
        if star_mask_data.shape == image_shape:
            star_mask = star_mask_data.astype(bool)
            print(f"  Loaded star mask: {star_mask_path} (shape: {star_mask.shape})")
        else:
            # Embed smaller mask into larger array (no scaling, just padding with zeros)
            # Place mask in bottom-left corner (FITS convention: origin at bottom-left)
            star_mask = np.zeros(image_shape, dtype=bool)
            mask_h, mask_w = star_mask_data.shape
            star_mask[0:mask_h, 0:mask_w] = star_mask_data.astype(bool)
            print(f"  Loaded star mask: {star_mask_path} (embedded {star_mask_data.shape} into {star_mask.shape}, rest is empty)")
    except FileNotFoundError:
        print(f"  Star mask not found: {star_mask_path}")

    # Extract star mask region centers for plotting (show all mask regions regardless of detection)
    star_mask_centers = []
    if star_mask is not None:
        labeled_mask, n_regions = ndimage.label(star_mask)
        for i in range(1, n_regions + 1):
            coords = np.where(labeled_mask == i)
            cy, cx = np.mean(coords[0]), np.mean(coords[1])
            star_mask_centers.append((cx, cy))
        print(f"  Star mask has {n_regions} regions (course staff identified stars)")

    if USE_FULL_RESOLUTION:
        # For 4096x4096 data, use the combined mask if available, or no mask
        combined_mask_path = "./data/combined_mask.fits"
        try:
            mask_data, _, _ = read_fits(combined_mask_path)
            if mask_data.shape == image_shape:
                mask = mask_data.astype(bool)
                print(f"  Loaded detection mask: {combined_mask_path} (shape: {mask.shape})")
            else:
                # Mask doesn't match - create empty mask
                mask = np.zeros(image_shape, dtype=bool)
                print(f"  No matching mask for {image_shape} - using no mask")
        except FileNotFoundError:
            mask = np.zeros(image_shape, dtype=bool)
            print("  No mask file found - using no mask")
    else:
        # For 2048x2048 data, don't mask during detection - we'll flag stars instead
        # This allows sources in star regions to be detected and flagged
        mask = np.zeros(image_shape, dtype=bool)
        print("  Detection mask: none (star flagging enabled via star_mask)")

    # Step 2: Source detection using adaptive PSF-based parameters
    print("\n[2/6] Detecting sources (adaptive PSF-based method)...")

    # Hubble ACS/WFC PSF parameters
    # PSF FWHM ~0.08-0.10 arcsec for optical bands
    PSF_FWHM_ARCSEC = 0.09  # Typical HST ACS PSF FWHM
    PSF_FWHM_PIX = PSF_FWHM_ARCSEC / PIXEL_SCALE  # ~2.25 pixels at 0.04"/pix

    # Adaptive kernel FWHM: Use kernel matched to PSF for optimal detection
    # Using 1.5x PSF FWHM to balance point source rejection with faint galaxy detection
    KERNEL_FWHM_PIX = max(3.0, PSF_FWHM_PIX * 1.5)

    # Adaptive npixels based on PSF area: pi * (FWHM/2)^2
    # Use 3x PSF area - sensitive enough to detect faint galaxies
    PSF_AREA = np.pi * (PSF_FWHM_PIX / 2) ** 2
    BASE_NPIXELS = int(np.ceil(PSF_AREA * 3.0))  # ~12 pixels - more sensitive

    # Scale npixels with image resolution (larger images have more pixels per source)
    def get_adaptive_npixels(image_shape, base_npixels):
        """Calculate adaptive npixels based on image resolution."""
        # Scale factor: 1.0 for 2048, 2.0 for 4096
        scale = image_shape[0] / 2048.0
        return int(base_npixels * scale)

    # Detection threshold: sigma above background RMS
    # Lower threshold for sensitive detection of faint galaxies
    DETECTION_SIGMA = 1.5  # Lower threshold to detect faint sources

    # Deblending parameters for separating overlapping galaxies
    NLEVELS = 32  # Multi-thresholding levels
    CONTRAST = 0.01  # Higher contrast = more conservative deblending (fewer splits)

    print(f"  PSF FWHM: {PSF_FWHM_ARCSEC:.3f} arcsec ({PSF_FWHM_PIX:.2f} pixels)")
    print(f"  Kernel FWHM: {KERNEL_FWHM_PIX:.2f} pixels")
    print(f"  Base npixels: {BASE_NPIXELS} (PSF area x 3.0 - sensitive)")
    print(f"  Detection threshold: {DETECTION_SIGMA}-sigma above background")
    print(f"  Deblending: nlevels={NLEVELS}, contrast={CONTRAST}")

    # Create 2D Gaussian kernel for source detection
    # make_2dgaussian_kernel returns a normalized kernel by default
    kernel = make_2dgaussian_kernel(KERNEL_FWHM_PIX, size=5)

    # Process bands SEQUENTIALLY to minimize memory usage
    # (Parallel processing requires holding multiple convolved arrays in memory)
    def process_band(image):
        """Process a single band for source detection."""
        import gc as gc_local

        bkg_estimator = MedianBackground()
        box_size = 100 if image.data.shape[0] > 2048 else 50
        bkg = Background2D(image.data, (box_size, box_size), filter_size=(3, 3), bkg_estimator=bkg_estimator)

        # Compute background-subtracted data
        data_sub = image.data - bkg.background
        image.bkg = bkg

        # Convolve - this creates a large temporary array
        convolved_data = convolve(data_sub, kernel)
        threshold = DETECTION_SIGMA * bkg.background_rms
        npixels = get_adaptive_npixels(image.data.shape, BASE_NPIXELS)

        finder = SourceFinder(
            npixels=npixels,
            deblend=True,
            nlevels=NLEVELS,
            contrast=CONTRAST,
            connectivity=8,
            progress_bar=False,
        )

        image.segm = finder(convolved_data, threshold, mask=mask)

        if image.segm is not None and mask is not None:
            masked_labels = np.unique(image.segm.data[mask])
            masked_labels = masked_labels[masked_labels != 0]
            if len(masked_labels) > 0:
                image.segm.remove_labels(masked_labels)

        # Create catalog - needs data_sub and convolved_data
        image.catalog = SourceCatalog(data_sub, image.segm, convolved_data=convolved_data)

        # Release intermediate arrays we no longer need
        del data_sub, convolved_data
        gc_local.collect()

        return image.band, len(image.catalog), npixels

    # Process bands sequentially to minimize peak memory usage
    print(f"  Processing {len(images)} bands sequentially (memory-optimized)...")
    for img in images:
        band, n_sources, npixels = process_band(img)
        print(f"  Band {band}: {n_sources} sources (npixels={npixels})")

    # Step 3: Cross-match sources
    print("\n[3/6] Cross-matching sources across bands...")

    # PSF parameters for corrections
    PSF_SIGMA_PIX = PSF_FWHM_PIX / 2.355  # Convert FWHM to sigma

    # Import gc early for memory management
    import gc

    from scipy.ndimage import sum as ndsum

    band_catalogs = {}

    # Process each image and IMMEDIATELY release its heavy data to save memory
    # This is critical for low-memory systems
    for image in images:
        print(f"    Processing {image.band} band catalog...")

        # Extract catalog data to pandas
        cat = image.catalog.to_table().to_pandas()

        # Use float32 for coordinates and measurements to save memory
        cat["r_half_pix"] = image.catalog.fluxfrac_radius(0.5).astype(np.float32)
        cat["r_90_pix"] = image.catalog.fluxfrac_radius(0.9).astype(np.float32)
        cat["source_sky"] = cat["segment_flux"].astype(np.float32)

        # Calculate concentration index (SDSS methodology)
        r90_vals = cat["r_90_pix"].to_numpy()
        r90_safe = np.where(r90_vals > 0, r90_vals, np.nan)
        cat["concentration"] = (cat["r_half_pix"].to_numpy() / r90_safe).astype(np.float32)

        # Calculate photometric errors using weight map if available
        if image.weight is not None:
            # Get variance map (1/weight)
            with np.errstate(divide="ignore", invalid="ignore"):
                variance_map = np.where(image.weight > 0, 1.0 / image.weight, np.inf)

            labels = image.catalog.labels
            segm_data = image.segm.data

            # Vectorized: sum variance over all segments at once
            segment_variances = ndsum(variance_map, labels=segm_data, index=labels)

            # Free variance_map immediately - this is ~128MB
            del variance_map
            gc.collect()

            # Count pixels per segment using bincount
            label_counts = np.bincount(segm_data.ravel())
            segment_areas = label_counts[labels]

            # Add background variance contribution
            bkg_rms = image.bkg.background_rms_median
            total_variances = segment_variances + segment_areas * bkg_rms**2
            cat["source_error"] = np.sqrt(total_variances).astype(np.float32)
        else:
            # Fallback: Poisson noise + background RMS
            cat["source_error"] = np.sqrt(
                np.abs(cat["segment_flux"]) + cat["area"] * image.bkg.background_rms_median**2
            ).astype(np.float32)

        # Calculate SNR for quality flagging
        cat["snr"] = (np.abs(cat["source_sky"]) / (cat["source_error"] + 1e-10)).astype(np.float32)

        cat["r_half_pix_error"] = (
            cat["r_half_pix"] * cat["source_error"] / (2 * np.abs(cat["segment_flux"]) + 1e-10)
        ).astype(np.float32)

        # Get ellipticity from catalog (for stellarity computation)
        try:
            cat["ellipticity"] = np.array(image.catalog.ellipticity, dtype=np.float32)
        except (AttributeError, KeyError, TypeError):
            cat["ellipticity"] = np.float32(np.nan)

        # Store only the columns we need (float32 to save memory)
        band_catalogs[image.band] = cat[
            [
                "xcentroid",
                "ycentroid",
                "source_sky",
                "source_error",
                "r_half_pix",
                "r_half_pix_error",
                "r_90_pix",
                "concentration",
                "snr",
                "ellipticity",
            ]
        ].copy()

        # CRITICAL: Release heavy memory from this image immediately
        # This frees ~500MB per image (weight map, segm, catalog, bkg)
        image.weight = None
        image.segm = None
        image.catalog = None
        image.bkg = None
        gc.collect()

    print("    Released image processing memory")

    # Convert image data to float32 for plotting (saves 50% memory)
    for image in images:
        if image.data is not None and image.data.dtype == np.float64:
            image.data = image.data.astype(np.float32)
    gc.collect()

    reference_band = "b"
    matched_sources = []
    match_radius = 3.0

    # Build KD-trees for fast spatial matching (O(N log N) instead of O(N²))
    other_bands = ["i", "u", "v"]
    kdtrees = {}
    coords_arrays = {}
    for band in other_bands:
        cat = band_catalogs[band]
        coords = np.column_stack([cat["xcentroid"].values, cat["ycentroid"].values])
        coords_arrays[band] = coords
        kdtrees[band] = cKDTree(coords)

    ref_cat = band_catalogs[reference_band]
    ref_coords = np.column_stack([ref_cat["xcentroid"].values, ref_cat["ycentroid"].values])

    # Image dimensions for edge detection
    image_shape = images[0].data.shape

    # Minimum number of bands required for a valid detection
    # For reliable photo-z, we need all 4 bands (U, B, V, I)
    # 3-band detections have much higher photo-z failure rates
    MIN_BANDS_REQUIRED = 4  # Require all 4 bands for reliable photo-z

    # ==========================================================================
    # OPTIMIZATION: Vectorized batch KD-tree queries (2-3x faster than loop)
    # Query all reference sources against all bands at once
    # ==========================================================================
    batch_distances = {}
    batch_indices = {}
    for band in other_bands:
        # Batch query: get nearest neighbor for ALL ref sources at once
        distances, indices = kdtrees[band].query(ref_coords, k=1)
        batch_distances[band] = distances
        batch_indices[band] = indices

    # Pre-extract reference band arrays for fast indexing
    ref_source_sky = ref_cat["source_sky"].values
    ref_source_error = ref_cat["source_error"].values
    ref_r_half = ref_cat["r_half_pix"].values
    ref_r_half_error = ref_cat["r_half_pix_error"].values
    ref_concentration = ref_cat["concentration"].values
    ref_snr = ref_cat["snr"].values
    ref_ellipticity = ref_cat["ellipticity"].values

    # Pre-extract other band arrays
    band_arrays = {}
    for band in other_bands:
        cat = band_catalogs[band]
        band_arrays[band] = {
            "source_sky": cat["source_sky"].values,
            "source_error": cat["source_error"].values,
            "r_half_pix": cat["r_half_pix"].values,
            "r_half_pix_error": cat["r_half_pix_error"].values,
            "concentration": cat["concentration"].values,
            "snr": cat["snr"].values,
            "ellipticity": cat["ellipticity"].values,
        }

    for idx in range(len(ref_cat)):
        ref_x, ref_y = ref_coords[idx]
        matches = {"xcentroid": ref_x, "ycentroid": ref_y}
        r_half_values = [ref_r_half[idx]]
        r_half_pix_errors = [ref_r_half_error[idx]]
        concentration_values = [ref_concentration[idx]]
        snr_values = [ref_snr[idx]]
        ellipticity_values = [ref_ellipticity[idx]]

        matches[f"source_sky_{reference_band}"] = ref_source_sky[idx]
        matches[f"source_error_{reference_band}"] = ref_source_error[idx]

        bands_matched = [reference_band]  # Reference band is always matched
        for band in other_bands:
            # Use pre-computed batch query results (already computed above)
            dist = batch_distances[band][idx]
            nearest_idx = batch_indices[band][idx]

            if dist < match_radius:
                arr = band_arrays[band]
                matches[f"source_sky_{band}"] = arr["source_sky"][nearest_idx]
                matches[f"source_error_{band}"] = arr["source_error"][nearest_idx]
                r_half_values.append(arr["r_half_pix"][nearest_idx])
                r_half_pix_errors.append(arr["r_half_pix_error"][nearest_idx])
                concentration_values.append(arr["concentration"][nearest_idx])
                snr_values.append(arr["snr"][nearest_idx])
                ellipticity_values.append(arr["ellipticity"][nearest_idx])
                bands_matched.append(band)
            else:
                # No match in this band - use NaN for flux
                matches[f"source_sky_{band}"] = np.nan
                matches[f"source_error_{band}"] = np.nan

        # Accept sources detected in at least MIN_BANDS_REQUIRED bands
        if len(bands_matched) >= MIN_BANDS_REQUIRED:
            matches["n_bands"] = len(bands_matched)
            matches["bands_matched"] = ",".join(bands_matched)
            r_half_arr = np.array(r_half_values)
            r_half_err_arr = np.array(r_half_pix_errors)
            matches["r_half_pix"] = np.median(r_half_arr)
            measurement_error = np.sqrt(np.sum(r_half_err_arr**2)) / len(r_half_err_arr)
            scatter_error = np.std(r_half_arr)
            matches["r_half_pix_error"] = np.sqrt(measurement_error**2 + scatter_error**2)

            # Median concentration, ellipticity, and SNR across bands
            conc_arr = np.array([c for c in concentration_values if np.isfinite(c)])
            matches["concentration"] = np.median(conc_arr) if len(conc_arr) > 0 else np.nan

            ell_arr = np.array([e for e in ellipticity_values if np.isfinite(e)])
            matches["ellipticity"] = np.median(ell_arr) if len(ell_arr) > 0 else np.nan

            matches["snr_median"] = np.median(snr_values)
            matches["snr_min"] = np.min(snr_values)  # Weakest detection

            # Apply PSF correction to half-light radius
            # r_intrinsic² ≈ r_measured² - r_psf² (quadrature subtraction)
            r_half_corrected = correct_radius_for_psf(matches["r_half_pix"], PSF_SIGMA_PIX)
            matches["r_half_pix_corrected"] = r_half_corrected

            # Compute stellarity score (multi-criteria, 0=galaxy, 1=star)
            # Following SExtractor CLASS_STAR methodology
            matches["stellarity"] = compute_stellarity_score(
                matches["r_half_pix"],
                matches["concentration"],
                matches["ellipticity"],
                PSF_FWHM_PIX,
            )

            # Compute initial quality flags
            flag = FLAG_NONE

            # Check for low SNR (< 5 in any band)
            if matches["snr_min"] < 5.0:
                flag |= FLAG_LOW_SNR

            # Check for edge proximity (within 50 pixels of edge)
            edge_margin = 50
            if (ref_x < edge_margin or ref_x > image_shape[1] - edge_margin or
                ref_y < edge_margin or ref_y > image_shape[0] - edge_margin):
                flag |= FLAG_EDGE

            # Check for PSF-like source using multi-criteria (professional approach)
            # Based on: SExtractor CLASS_STAR, SDSS DR14, HST Source Catalog (HSC)
            #
            # Key insight from research:
            # - SExtractor uses CLASS_STAR > 0.95 for confident stars
            # - HST ACS PSF FWHM is 0.10-0.13 arcsec, so stars have r_half < ~0.06 arcsec
            # - Elliptical galaxies have high concentration BUT are extended (r_half >> PSF)
            # - Stars are point sources: compact, round, high concentration
            #
            # References:
            # - https://sextractor.readthedocs.io/en/latest/ClassStar.html
            # - https://www.sdss4.org/dr14/algorithms/classify/
            # - https://archive.stsci.edu/hst/hsc/help/HSC_faq.html

            is_star = False

            # PSF half-light radius (Gaussian approximation: r_half ≈ 0.42 * FWHM)
            psf_r_half_pix = PSF_FWHM_PIX * 0.42  # ~0.95 pixels for HST ACS

            # Size ratio: how many times larger than PSF
            size_ratio = matches["r_half_pix"] / psf_r_half_pix if psf_r_half_pix > 0 else 999

            # Criterion 1: Very high stellarity (confident star classification)
            # SExtractor recommends > 0.95, we use 0.8 as compromise
            if matches["stellarity"] > 0.8:
                is_star = True

            # Criterion 2: Compact + round + concentrated (all three required)
            # This catches stars that have moderate stellarity but clear point-source morphology
            elif size_ratio < 1.8:  # Very compact (within 1.8x PSF size)
                is_round = matches["ellipticity"] < 0.2  # Stars are very round
                is_concentrated = matches["concentration"] > 0.45  # High light concentration
                if is_round and is_concentrated:
                    is_star = True

            # Criterion 3: Stellar colors + compact morphology
            # (handled separately in color_stellarity check later)

            if is_star:
                flag |= FLAG_PSF_LIKE

            # Check if source is inside star mask OR near a mask region center
            # Sources in/near these regions are forced to be classified as stars
            # This handles cases where cross-matching offsets cause misalignment
            STAR_MASK_RADIUS = 30  # pixels - flag sources within this radius of mask centers
            matches["in_star_mask"] = False

            if star_mask is not None:
                # Method 1: Check if pixel is directly in mask
                px, py = round(ref_x), round(ref_y)
                if 0 <= py < star_mask.shape[0] and 0 <= px < star_mask.shape[1] and star_mask[py, px]:
                    flag |= FLAG_PSF_LIKE
                    matches["in_star_mask"] = True

            # Method 2: Check proximity to mask region centers (catches offset sources)
            if star_mask_centers and not matches["in_star_mask"]:
                for cx, cy in star_mask_centers:
                    dist = np.sqrt((ref_x - cx)**2 + (ref_y - cy)**2)
                    if dist < STAR_MASK_RADIUS:
                        flag |= FLAG_PSF_LIKE
                        matches["in_star_mask"] = True
                        break

            matches["quality_flag"] = flag
            matched_sources.append(matches)

    cross_matched_catalog = pd.DataFrame(matched_sources)
    n_4band = (cross_matched_catalog["n_bands"] == 4).sum()
    n_3band = (cross_matched_catalog["n_bands"] == 3).sum()
    print(f"  Cross-matched: {len(cross_matched_catalog)} sources (≥{MIN_BANDS_REQUIRED} bands)")
    print(f"    4-band detections: {n_4band}, 3-band detections: {n_3band}")

    # Report flag statistics
    n_clean = (cross_matched_catalog["quality_flag"] == FLAG_NONE).sum()
    n_low_snr = ((cross_matched_catalog["quality_flag"] & FLAG_LOW_SNR) != 0).sum()
    n_edge = ((cross_matched_catalog["quality_flag"] & FLAG_EDGE) != 0).sum()
    n_psf_like = ((cross_matched_catalog["quality_flag"] & FLAG_PSF_LIKE) != 0).sum()
    n_in_star_mask = cross_matched_catalog["in_star_mask"].sum() if "in_star_mask" in cross_matched_catalog.columns else 0
    print(f"  Quality flags: {n_clean} clean, {n_low_snr} low_snr, {n_edge} edge, {n_psf_like} psf_like")
    if n_in_star_mask > 0:
        print(f"  Star mask: {n_in_star_mask} sources flagged as stars (from course staff mask)")

    # Add FLAG_MASKED to sources inside the star mask (known stars from class)
    if "in_star_mask" in cross_matched_catalog.columns:
        star_mask_sources = cross_matched_catalog["in_star_mask"]
        cross_matched_catalog.loc[star_mask_sources, "quality_flag"] = (
            cross_matched_catalog.loc[star_mask_sources, "quality_flag"].astype(int) | FLAG_MASKED
        )
        print(f"  Added FLAG_MASKED to {star_mask_sources.sum()} sources in star mask")

    # Step 4: Convert to physical units and classify
    print("\n[4/6] Converting fluxes and classifying galaxies...")
    conversion_factors = {
        "b": 8.8e-18,
        "i": 2.45e-18,
        "u": 5.99e-17,
        "v": 1.89e-18,
    }

    # Build SED catalog using vectorized operations (much faster than iterrows)
    sed_catalog = cross_matched_catalog[[
        "r_half_pix", "r_half_pix_error", "r_half_pix_corrected",
        "xcentroid", "ycentroid", "concentration", "ellipticity", "stellarity",
        "snr_median", "snr_min", "quality_flag", "in_star_mask"
    ]].copy()

    # Vectorized flux conversion for all bands
    for band in ["b", "i", "u", "v"]:
        sed_catalog[f"flux_{band}"] = cross_matched_catalog[f"source_sky_{band}"] * conversion_factors[band]
        sed_catalog[f"error_{band}"] = cross_matched_catalog[f"source_error_{band}"] * conversion_factors[band]

    # Add sky coordinates (RA, Dec) for Gaia cross-matching
    if reference_wcs is not None:
        try:
            pixel_coords = np.column_stack([
                sed_catalog["xcentroid"].values,
                sed_catalog["ycentroid"].values
            ])
            sky_coords = reference_wcs.pixel_to_world(pixel_coords[:, 0], pixel_coords[:, 1])
            sed_catalog["ra"] = sky_coords.ra.deg
            sed_catalog["dec"] = sky_coords.dec.deg
            print(f"  Added sky coordinates (RA, Dec) for {len(sed_catalog)} sources")
        except Exception as e:
            print(f"  Could not compute sky coordinates: {e}")
            sed_catalog["ra"] = np.nan
            sed_catalog["dec"] = np.nan
    else:
        # Use approximate coordinates based on HDF-N center
        HDF_CENTER_RA = 189.228621
        HDF_CENTER_DEC = 62.212572
        # Approximate: 0.04 arcsec/pixel, centered on HDF
        center_x, center_y = image_shape[1] / 2, image_shape[0] / 2
        sed_catalog["ra"] = HDF_CENTER_RA + (sed_catalog["xcentroid"] - center_x) * PIXEL_SCALE / 3600.0
        sed_catalog["dec"] = HDF_CENTER_DEC + (sed_catalog["ycentroid"] - center_y) * PIXEL_SCALE / 3600.0
        print("  Added approximate sky coordinates (no WCS available)")

    # Force garbage collection before heavy computation
    gc.collect()

    # ULTRA FAST: Classify ALL galaxies at once with coarse-to-fine + float32 + parallel numba
    # This eliminates ProcessPoolExecutor overhead and processes everything in parallel
    # Speed: ~600,000 galaxies/second!
    n_galaxies = len(sed_catalog)
    print(f"  Classifying {n_galaxies} galaxies with ULTRA FAST classifier (~600k/sec)...")

    # Build flux and error arrays in [U, B, V, I] order for vectorized classifier
    flux_array = np.column_stack([
        sed_catalog["flux_u"].to_numpy(),
        sed_catalog["flux_b"].to_numpy(),
        sed_catalog["flux_v"].to_numpy(),
        sed_catalog["flux_i"].to_numpy(),
    ])
    error_array = np.column_stack([
        sed_catalog["error_u"].to_numpy(),
        sed_catalog["error_b"].to_numpy(),
        sed_catalog["error_v"].to_numpy(),
        sed_catalog["error_i"].to_numpy(),
    ])

    # ULTRA FAST: Coarse-to-fine + float32 + parallel numba
    import time
    t0 = time.perf_counter()
    results = classify_batch_ultrafast(flux_array, error_array, spectra_path="./spectra")
    t1 = time.perf_counter()
    print(f"    Classified {n_galaxies} galaxies in {t1-t0:.4f}s ({n_galaxies/(t1-t0):.0f} galaxies/sec)")

    # Directly assign results (already aligned arrays)
    sed_catalog["redshift"] = results['redshift']
    sed_catalog["redshift_lo"] = results['z_lo']  # 16th percentile
    sed_catalog["redshift_hi"] = results['z_hi']  # 84th percentile
    sed_catalog["redshift_err"] = (results['z_hi'] - results['z_lo']) / 2.0  # Symmetric error estimate
    sed_catalog["chi_sq_min"] = results['chi_sq_min']
    sed_catalog["photo_z_odds"] = results['odds']  # Photo-z quality (BPZ ODDS parameter)
    sed_catalog["galaxy_type"] = results['galaxy_type']

    # Add quality control flags from classification
    sed_catalog["chi2_flag"] = results['chi2_flag']
    sed_catalog["odds_flag"] = results['odds_flag']
    sed_catalog["bimodal_flag"] = results['bimodal_flag']
    sed_catalog["template_ambiguity"] = results['template_ambiguity']
    sed_catalog["reduced_chi2"] = results['reduced_chi2']
    sed_catalog["second_best_template"] = results['second_best_template']
    sed_catalog["delta_chi2_templates"] = results['delta_chi2_templates']

    # Add boundary flag (sources at z_min or z_max - fitting likely failed)
    if 'z_boundary_flag' in results:
        sed_catalog["z_boundary_flag"] = results['z_boundary_flag']
    else:
        # Fallback: compute from redshift values
        sed_catalog["z_boundary_flag"] = (results['redshift'] <= 0.01) | (results['redshift'] >= 5.99)

    # Track number of valid bands used in photo-z fitting
    if 'n_valid_bands' in results:
        sed_catalog["n_valid_bands"] = results['n_valid_bands']
    else:
        # Fallback: count non-NaN flux values
        flux_cols = [f"flux_{b}" for b in ["u", "b", "v", "i"]]
        sed_catalog["n_valid_bands"] = sed_catalog[flux_cols].notna().sum(axis=1)

    # Use PSF-corrected radius for angular size (vectorized)
    sed_catalog["r_half_arcsec"] = sed_catalog["r_half_pix_corrected"] * PIXEL_SCALE
    sed_catalog["r_half_arcsec_error"] = sed_catalog["r_half_pix_error"] * PIXEL_SCALE

    # Update quality flag for bad photo-z (ODDS < 0.6 OR hit redshift boundary) - vectorized
    bad_photoz_mask = (results['odds'] < 0.6) | sed_catalog["z_boundary_flag"]
    sed_catalog.loc[bad_photoz_mask, "quality_flag"] = (
        sed_catalog.loc[bad_photoz_mask, "quality_flag"].astype(int) | FLAG_BAD_PHOTOZ
    )

    # Flag unreliable redshifts where error exceeds redshift value (σ_z > z)
    # These measurements have uncertainty larger than the measurement itself
    unreliable_z_mask = (
        (sed_catalog["redshift_err"] > sed_catalog["redshift"]) &
        (sed_catalog["redshift"] > 0)  # Only for positive redshifts
    )
    n_unreliable_z = unreliable_z_mask.sum()
    if n_unreliable_z > 0:
        sed_catalog.loc[unreliable_z_mask, "quality_flag"] = (
            sed_catalog.loc[unreliable_z_mask, "quality_flag"].astype(int) | FLAG_UNRELIABLE_Z
        )
        print(f"  Flagged {n_unreliable_z} sources with unreliable redshifts (σ_z > z)")

    # Add color-based stellar contamination check
    # Stars have flat SEDs (small color scatter), galaxies have strong colors
    # Using vectorized computation (100-700x faster than iterrows)
    print("  Checking for stellar contamination via colors (vectorized)...")
    sed_catalog["color_stellarity"] = check_stellar_colors_vectorized(
        sed_catalog["flux_u"].to_numpy(),
        sed_catalog["flux_b"].to_numpy(),
        sed_catalog["flux_v"].to_numpy(),
        sed_catalog["flux_i"].to_numpy(),
    )

    # Update FLAG_PSF_LIKE for sources with stellar colors AND high morphological stellarity
    # This is a more stringent test: must look like star AND have stellar-like colors
    stellar_color_mask = (sed_catalog["color_stellarity"] > 0.7) & (sed_catalog["stellarity"] > 0.5)
    n_stellar_colors = stellar_color_mask.sum()
    if n_stellar_colors > 0:
        sed_catalog.loc[stellar_color_mask, "quality_flag"] = (
            sed_catalog.loc[stellar_color_mask, "quality_flag"].astype(int) | FLAG_PSF_LIKE
        )
        print(f"    Flagged {n_stellar_colors} additional sources with stellar colors + morphology")

    print(f"  Classified {len(sed_catalog)} galaxies")

    # =========================================================================
    # PROFESSIONAL STAR-GALAXY CLASSIFICATION (Research Standard)
    # Uses multi-tier approach: Gaia → SPREAD_MODEL → ML → Color-color
    # =========================================================================
    if USE_PROFESSIONAL_CLASSIFICATION:
        print("\n  Running professional star-galaxy classification...")
        try:
            from morphology.professional_classification import classify_professional
            from morphology.star_galaxy import query_gaia_for_classification

            # Query Gaia DR3 for foreground stars
            HDF_CENTER_RA = 189.228621
            HDF_CENTER_DEC = 62.212572
            print("    Querying Gaia DR3 for foreground stars...")
            gaia_catalog = query_gaia_for_classification(
                ra_center=HDF_CENTER_RA,
                dec_center=HDF_CENTER_DEC,
                radius_arcmin=3.0,
                magnitude_limit=21.0,
            )
            if len(gaia_catalog) > 0:
                print(f"    Retrieved {len(gaia_catalog)} Gaia sources")
            else:
                print("    Warning: No Gaia sources retrieved (proceeding without Gaia)")
                gaia_catalog = None

            # Get reference image for morphological analysis (use I-band)
            ref_image = None
            for img in images:
                if img.band == 'i':
                    ref_image = img.data
                    break
            if ref_image is None:
                ref_image = images[0].data

            # Run professional classification
            classification_results = classify_professional(
                catalog=sed_catalog,
                image=ref_image,
                wcs=reference_wcs,
                gaia_catalog=gaia_catalog,
                x_col="xcentroid",
                y_col="ycentroid",
                ra_col="ra",
                dec_col="dec",
                pixel_scale=PIXEL_SCALE,
                verbose=True,
            )

            # Merge professional classification results
            pro_cols = ['is_galaxy', 'is_star', 'probability_galaxy', 'confidence',
                       'classification_method', 'classification_tier', 'spread_model',
                       'gaia_confirmed_star', 'concentration_c', 'half_light_radius']
            for col in pro_cols:
                if col in classification_results.columns:
                    sed_catalog[f"pro_{col}"] = classification_results[col].values

            # Update FLAG_PSF_LIKE based on professional classification
            # Only flag sources that professional classification confirms as stars
            if 'is_star' in classification_results.columns:
                pro_stars = classification_results['is_star'].values
                n_pro_stars = pro_stars.sum()

                # Clear basic PSF_LIKE flags and use professional classification
                # This replaces the basic heuristic with the professional result
                sed_catalog.loc[pro_stars, "quality_flag"] = (
                    sed_catalog.loc[pro_stars, "quality_flag"].astype(int) | FLAG_PSF_LIKE
                )

                # Sources confirmed as galaxies by professional classification
                # can have their FLAG_PSF_LIKE cleared if it was set by basic method
                pro_galaxies = classification_results['is_galaxy'].values
                high_conf_galaxies = pro_galaxies & (classification_results['confidence'].values > 0.7)
                sed_catalog.loc[high_conf_galaxies, "quality_flag"] = (
                    sed_catalog.loc[high_conf_galaxies, "quality_flag"].astype(int) & ~FLAG_PSF_LIKE
                )

                print(f"    Professional classification: {n_pro_stars} stars, "
                      f"{pro_galaxies.sum()} galaxies")

                # Report classification by tier
                if 'classification_tier' in classification_results.columns:
                    tier_names = {1: "Gaia", 2: "SPREAD_MODEL", 3: "Morphology",
                                 4: "ML", 5: "Color-color"}
                    for tier, name in tier_names.items():
                        n_tier = (classification_results['classification_tier'] == tier).sum()
                        if n_tier > 0:
                            print(f"      Tier {tier} ({name}): {n_tier} sources")

                # Skip diagnostic plot (star_galaxy_classification.pdf) - not needed
                # try:
                #     from morphology.professional_classification import plot_classification_diagnostics
                #     plot_classification_diagnostics(
                #         classification_results,
                #         output_path=f"{output_dir}/star_galaxy_classification.pdf"
                #     )
                #     print(f"    Saved: {output_dir}/star_galaxy_classification.pdf")
                # except Exception as plot_error:
                #     print(f"    Could not save diagnostic plot: {plot_error}")

        except ImportError as e:
            print(f"    Warning: Professional classification not available: {e}")
            print("    Using basic classification only")
        except Exception as e:
            print(f"    Warning: Professional classification failed: {e}")
            print("    Using basic classification only")

    # Cross-match with 3D-HST catalog for additional star identification
    # This catches stars our morphological methods miss
    print("\n  Cross-matching with 3D-HST catalog (Skelton et al. 2014)...")
    try:
        from validation.external_crossmatch import crossmatch_with_3dhst

        sed_catalog = crossmatch_with_3dhst(sed_catalog, match_radius_arcsec=1.0)

        # Flag sources identified as stars by 3D-HST
        if 'star_3dhst' in sed_catalog.columns:
            n_3dhst_stars = sed_catalog['star_3dhst'].sum()
            if n_3dhst_stars > 0:
                # Update FLAG_PSF_LIKE for 3D-HST identified stars
                sed_catalog.loc[sed_catalog['star_3dhst'], "quality_flag"] = (
                    sed_catalog.loc[sed_catalog['star_3dhst'], "quality_flag"].astype(int) | FLAG_PSF_LIKE
                )
                print(f"    Flagged {n_3dhst_stars} additional stars from 3D-HST")

                # Update pro_is_star column if it exists
                if 'pro_is_star' in sed_catalog.columns:
                    sed_catalog.loc[sed_catalog['star_3dhst'], 'pro_is_star'] = True

        # Apply spectroscopic redshifts from 3D-HST where available
        # This fixes catastrophic photo-z failures
        from validation.external_crossmatch import apply_spectroscopic_redshifts
        sed_catalog = apply_spectroscopic_redshifts(
            sed_catalog,
            z_col='redshift',
            match_radius_arcsec=1.0,
            flag_catastrophic=True,
            catastrophic_threshold=0.15
        )

        # Report how many redshifts were updated
        if 'has_specz' in sed_catalog.columns:
            n_specz = sed_catalog['has_specz'].sum()
            if n_specz > 0:
                print(f"    Applied {n_specz} spectroscopic redshifts from 3D-HST")
                if 'catastrophic_photoz' in sed_catalog.columns:
                    n_catastrophic = sed_catalog['catastrophic_photoz'].sum()
                    if n_catastrophic > 0:
                        print(f"    Fixed {n_catastrophic} catastrophic photo-z failures")

    except ImportError:
        print("    Warning: 3D-HST cross-match not available (install astroquery)")
    except Exception as e:
        print(f"    Warning: 3D-HST cross-match failed: {e}")

    # Report photo-z quality statistics
    odds_good = (sed_catalog["photo_z_odds"] >= 0.9).sum()
    odds_medium = ((sed_catalog["photo_z_odds"] >= 0.6) & (sed_catalog["photo_z_odds"] < 0.9)).sum()
    odds_poor = (sed_catalog["photo_z_odds"] < 0.6).sum()
    print(f"  Photo-z quality: {odds_good} excellent (ODDS>=0.9), {odds_medium} good (0.6<=ODDS<0.9), {odds_poor} poor (ODDS<0.6)")

    # Step 5: Analysis and plotting
    print("\n[5/6] Analyzing results...")
    sed_catalog = sed_catalog.dropna(subset=["redshift", "r_half_arcsec", "r_half_arcsec_error"])
    # Filter out sources with inf errors (from zero-weight pixels)
    sed_catalog = sed_catalog[np.isfinite(sed_catalog["r_half_arcsec_error"])]
    # Filter out sources with zero angular size (smaller than PSF)
    sed_catalog = sed_catalog[sed_catalog["r_half_arcsec"] > 0]
    print(f"  Valid sources after basic filtering: {len(sed_catalog)}")

    # Report final quality flag distribution
    print("\n  Final quality flag distribution:")
    n_clean = (sed_catalog["quality_flag"] == FLAG_NONE).sum()
    n_any_flag = (sed_catalog["quality_flag"] != FLAG_NONE).sum()
    print(f"    Clean (no flags): {n_clean}")
    print(f"    Flagged (any): {n_any_flag}")
    for flag, name in FLAG_DESCRIPTIONS.items():
        count = ((sed_catalog["quality_flag"].astype(int) & flag) != 0).sum()
        if count > 0:
            print(f"      {name}: {count}")

    # ==========================================================================
    # HIGH-QUALITY FILTERING FOR PRECISE RESULTS
    # ==========================================================================
    # For publication-quality plots, we apply strict quality cuts following
    # best practices from COSMOS, DES, and other major surveys:
    #
    # 1. FLAG EXCLUSIONS:
    #    - blended: Photometry may be contaminated by neighbor subtraction
    #    - edge: Truncated aperture may bias flux/size measurements
    #    - saturated: Flux is lower limit, morphology unreliable
    #    - low_snr: SNR < 5, measurements dominated by noise
    #    - crowded: High chance of photometric contamination
    #    - bad_photoz: ODDS < 0.6 or hit redshift grid boundary
    #    - psf_like: Likely star or quasar, not a resolved galaxy
    #    - masked: Overlaps masked regions (diffraction spikes, etc.)
    #    - unreliable_z: σ_z > z (error exceeds measurement)
    #
    # 2. PHOTO-Z QUALITY (ODDS >= 0.9):
    #    The ODDS parameter measures the integrated probability within
    #    ±0.1(1+z) of the peak. ODDS >= 0.9 indicates a well-constrained,
    #    unimodal redshift solution.
    #
    # 3. SED FIT QUALITY (chi2_flag == 0, reduced chi2 < 5):
    #    Good template fit ensures the photometry is consistent with
    #    galaxy SED models. High chi2 may indicate calibration issues,
    #    AGN contamination, or photometric errors.
    #
    # 4. NON-BIMODAL SOLUTIONS:
    #    Excludes sources where multiple redshift solutions fit equally
    #    well (e.g., z=0.5 vs z=3 degeneracy from Lyman/4000Å break).

    HIGH_QUALITY_MODE = True  # Set to False for broader (but noisier) sample

    if HIGH_QUALITY_MODE:
        # Strict flag exclusions - exclude any source with potential issues
        STRICT_EXCLUDE_FLAGS = (
            FLAG_BLENDED | FLAG_EDGE | FLAG_SATURATED | FLAG_LOW_SNR |
            FLAG_CROWDED | FLAG_BAD_PHOTOZ | FLAG_PSF_LIKE | FLAG_MASKED |
            FLAG_UNRELIABLE_Z  # Exclude sources where σ_z > z
        )

        # Apply all quality cuts
        quality_mask = (
            # No problematic flags
            ((sed_catalog["quality_flag"].astype(int) & STRICT_EXCLUDE_FLAGS) == 0) &
            # Excellent photo-z confidence (ODDS >= 0.9)
            (sed_catalog["odds_flag"] == 0) &
            # Good chi-squared fit (reduced chi2 < 5)
            (sed_catalog["chi2_flag"] == 0) &
            # Not bimodal (single clear redshift solution)
            (~sed_catalog["bimodal_flag"])
        )

        sed_catalog_filtered = sed_catalog[quality_mask]

        # Report filtering breakdown
        n_total = len(sed_catalog)
        n_kept = len(sed_catalog_filtered)
        n_rejected = n_total - n_kept

        print(f"\n  HIGH-QUALITY MODE: Applied strict quality filtering")
        print(f"    Input: {n_total} sources")
        print(f"    Rejection breakdown:")

        # Count sources rejected by each criterion
        n_flag_rejected = ((sed_catalog["quality_flag"].astype(int) & STRICT_EXCLUDE_FLAGS) != 0).sum()
        n_odds_rejected = (sed_catalog["odds_flag"] != 0).sum()
        n_chi2_rejected = (sed_catalog["chi2_flag"] != 0).sum()
        n_bimodal_rejected = sed_catalog["bimodal_flag"].sum()

        print(f"      - Quality flags:     {n_flag_rejected} ({100*n_flag_rejected/n_total:.1f}%)")
        print(f"      - ODDS < 0.9:        {n_odds_rejected} ({100*n_odds_rejected/n_total:.1f}%)")
        print(f"      - Chi2 > 5:          {n_chi2_rejected} ({100*n_chi2_rejected/n_total:.1f}%)")
        print(f"      - Bimodal redshift:  {n_bimodal_rejected} ({100*n_bimodal_rejected/n_total:.1f}%)")
        print(f"    Result: {n_kept} high-confidence galaxies ({100*n_kept/n_total:.1f}% retained)")
    else:
        # Moderate filtering - excludes only clearly bad sources
        EXCLUDE_FLAGS = FLAG_BAD_PHOTOZ | FLAG_PSF_LIKE | FLAG_MASKED
        sed_catalog_filtered = sed_catalog[
            (sed_catalog["quality_flag"].astype(int) & EXCLUDE_FLAGS) == 0
        ]
        print(f"\n  MODERATE MODE: Basic quality filtering")
        print(f"    - After filtering (excluding bad_photoz, psf_like, masked): {len(sed_catalog_filtered)}")

    # Use filtered catalog for analysis
    sed_catalog_analysis = sed_catalog_filtered

    # Print redshift distribution
    print("\n  Redshift statistics (filtered):")
    print(f"    Min: {sed_catalog_analysis.redshift.min():.3f}")
    print(f"    Max: {sed_catalog_analysis.redshift.max():.3f}")
    print(f"    Mean: {sed_catalog_analysis.redshift.mean():.3f}")
    print(f"    Median: {sed_catalog_analysis.redshift.median():.3f}")
    print(f"    Median uncertainty: {sed_catalog_analysis.redshift_err.median():.3f}")

    # Galaxy type distribution
    print("\n  Galaxy type distribution:")
    for gtype in sed_catalog_analysis.galaxy_type.unique():
        count = (sed_catalog_analysis.galaxy_type == gtype).sum()
        print(f"    {gtype}: {count}")

    # Angular size statistics
    print("\n  Angular size statistics (PSF-corrected):")
    print(f"    Min: {sed_catalog_analysis.r_half_arcsec.min():.4f} arcsec")
    print(f"    Max: {sed_catalog_analysis.r_half_arcsec.max():.4f} arcsec")
    print(f"    Median: {sed_catalog_analysis.r_half_arcsec.median():.4f} arcsec")

    # Concentration index statistics
    print("\n  Concentration index statistics (C = r50/r90):")
    print(f"    Min: {sed_catalog_analysis.concentration.min():.3f}")
    print(f"    Max: {sed_catalog_analysis.concentration.max():.3f}")
    print(f"    Median: {sed_catalog_analysis.concentration.median():.3f}")

    # Stellarity statistics
    print("\n  Stellarity statistics (0=galaxy, 1=star):")
    print(f"    Min: {sed_catalog_analysis.stellarity.min():.3f}")
    print(f"    Max: {sed_catalog_analysis.stellarity.max():.3f}")
    print(f"    Median: {sed_catalog_analysis.stellarity.median():.3f}")

    # Ellipticity statistics
    if "ellipticity" in sed_catalog_analysis.columns:
        ell_valid = sed_catalog_analysis.ellipticity.dropna()
        if len(ell_valid) > 0:
            print("\n  Ellipticity statistics (0=round, 1=elongated):")
            print(f"    Min: {ell_valid.min():.3f}")
            print(f"    Max: {ell_valid.max():.3f}")
            print(f"    Median: {ell_valid.median():.3f}")

    # Dynamic binning with multiple strategies (use filtered catalog)
    z_min_raw, z_max_raw = sed_catalog_analysis.redshift.min(), sed_catalog_analysis.redshift.max()
    if not np.isfinite(z_min_raw) or not np.isfinite(z_max_raw) or z_min_raw == z_max_raw:
        raise ValueError("Cannot bin redshifts: invalid or zero range")

    # Calculate adaptive number of bins based on data size
    n_bins_adaptive = get_adaptive_n_bins(len(sed_catalog_analysis))
    print(f"\n  Using adaptive binning: {n_bins_adaptive} bins for {len(sed_catalog_analysis)} quality-filtered galaxies")

    # Define binning strategies
    binning_strategies = {
        "Equal Width": bin_equal_width,
        "Equal Count": bin_equal_count,
        "Logarithmic": bin_logarithmic,
        "Percentile": bin_percentile,
    }

    # Apply all binning strategies with adaptive bin count
    binned_results = {}
    for name, bin_func in binning_strategies.items():
        try:
            binned_results[name] = bin_func(sed_catalog_analysis)  # Use filtered catalog
            print(f"\n  {name} binning ({len(binned_results[name])} bins):")
            print(binned_results[name][["z_mid", "theta_med", "theta_err", "n"]].to_string())
        except Exception as e:
            print(f"\n  {name} binning failed: {e}")

    # Use equal-count as the primary binning (most physically justified)
    binned = binned_results.get("Equal Count", bin_equal_width(sed_catalog_analysis))

    # Fit models
    z_min = max(1e-4, binned.z_mid.min())
    z_model = np.linspace(z_min, binned.z_mid.max(), 300)

    # Static model: fit only R
    R_static = get_radius(
        binned.z_mid.values, binned.theta_med.values, binned.theta_err.values, model="static"
    )

    # Flat ΛCDM model: fit both R and Omega_m (with Omega_L = 1 - Omega_m)
    R_lcdm, Omega_m_fit = get_radius_and_omega(
        binned.z_mid.values, binned.theta_med.values, binned.theta_err.values
    )
    Omega_L_fit = 1.0 - Omega_m_fit

    print("\n  Fitted parameters:")
    print(f"    Static model: R = {R_static * 1000:.2f} kpc ({R_static:.6f} Mpc)")
    print(f"    ΛCDM model:   R = {R_lcdm * 1000:.2f} kpc ({R_lcdm:.6f} Mpc)")
    print(f"                  Ω_m = {Omega_m_fit:.3f}, Ω_Λ = {Omega_L_fit:.3f}")

    # Convert model output from radians to arcseconds
    theta_static_model = theta_static(z_model, R_static) * RAD_TO_ARCSEC
    theta_lcdm_model = theta_lcdm_flat(z_model, R_lcdm, Omega_m_fit) * RAD_TO_ARCSEC

    # Step 6: Create and save plots
    print("\n[6/6] Creating plots and saving as PDFs...")

    # Ensure output directory exists before saving
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isdir(output_dir):
        raise RuntimeError(f"Failed to create output directory: {output_dir}")

    # Calculate dynamic axis limits with padding (use filtered catalog)
    z_data_min, z_data_max = sed_catalog_analysis.redshift.min(), sed_catalog_analysis.redshift.max()
    theta_data_min = sed_catalog_analysis.r_half_arcsec.min()
    theta_data_max = sed_catalog_analysis.r_half_arcsec.max()

    z_padding = (z_data_max - z_data_min) * 0.08
    theta_padding = (theta_data_max - theta_data_min) * 0.1

    z_lim = (max(0, z_data_min - z_padding), z_data_max + z_padding)
    theta_lim = (max(0, theta_data_min - theta_padding), theta_data_max + theta_padding)

    # Plot 1: Main angular size vs redshift
    _fig, ax = plt.subplots(figsize=(10, 7))

    ax.errorbar(
        binned["z_mid"],
        binned["theta_med"],
        yerr=binned["theta_err"],
        fmt="o",
        capsize=3,
        capthick=0.8,
        markersize=7,
        markeredgewidth=0.8,
        elinewidth=0.8,
        color="black",
        label=f"Median angular size (N={len(sed_catalog_analysis)})",
    )

    ax.plot(
        z_model,
        theta_static_model,
        "--",
        color="gray",
        linewidth=2,
        label=r"Linear Hubble law (Euclidean)",
    )
    ax.plot(
        z_model,
        theta_lcdm_model,
        "-",
        color="blue",
        linewidth=2,
        label=r"$\Lambda$CDM ($\Omega_m$=" + f"{Omega_m_fit:.2f})",
    )

    ax.set_xlabel(r"Redshift $z$", fontsize=12)
    ax.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=12)
    ax.set_title(r"Galaxy Angular Size vs Redshift" + "\n" + r"(Hubble Deep Field)", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(z_lim)
    ax.set_ylim(theta_lim)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/angular_size_vs_redshift.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/angular_size_vs_redshift.pdf")

    # Plot 2: Individual galaxies scatter
    _fig, ax = plt.subplots(figsize=(12, 7))

    # Get unique galaxy types and create color mapping
    galaxy_type_cat = sed_catalog_analysis["galaxy_type"].astype("category")
    unique_types = galaxy_type_cat.cat.categories.tolist()
    cmap = plt.cm.tab10

    ax.scatter(
        sed_catalog_analysis["redshift"],
        sed_catalog_analysis["r_half_arcsec"],
        c=galaxy_type_cat.cat.codes,
        cmap=cmap,
        alpha=0.6,
        s=20,
    )

    ax.plot(
        z_model,
        theta_static_model,
        "--",
        color="gray",
        linewidth=2,
        alpha=0.8,
    )
    ax.plot(
        z_model, theta_lcdm_model, "-", color="blue", linewidth=2, alpha=0.8
    )

    ax.set_xlabel(r"Redshift $z$", fontsize=12)
    ax.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=12)
    ax.set_title(r"Individual Galaxy Angular Sizes" + "\n" + r"(colored by galaxy type)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(z_lim)
    ax.set_ylim(theta_lim)

    # Create legend for galaxy types
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i / len(unique_types)),
               markersize=8, alpha=0.8, label=gtype)
        for i, gtype in enumerate(unique_types)
    ]
    ax.legend(handles=legend_handles, title="Galaxy Type", loc="upper right",
              fontsize=9, title_fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/individual_galaxies.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/individual_galaxies.pdf")

    # Plot 3: Binning strategy comparison (2x2 grid with residuals)
    fig = plt.figure(figsize=(14, 16), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 3, 1], hspace=0.08, wspace=0.15)

    colors = {
        "Equal Width": "green",
        "Equal Count": "blue",
        "Logarithmic": "orange",
        "Percentile": "purple",
    }
    markers = {"Equal Width": "o", "Equal Count": "o", "Logarithmic": "o", "Percentile": "o"}

    strategy_names = list(binned_results.keys())

    for idx, name in enumerate(strategy_names):
        binned_data = binned_results[name]
        # Grid position: first two strategies in rows 0-1, last two in rows 2-3
        row_base = (idx // 2) * 2  # 0 or 2 for main plots
        col = idx % 2

        ax_main = fig.add_subplot(gs[row_base, col])
        ax_resid = fig.add_subplot(gs[row_base + 1, col], sharex=ax_main)

        # Fit models for this binning
        R_static_i = get_radius(
            binned_data.z_mid.values,
            binned_data.theta_med.values,
            binned_data.theta_err.values,
            model="static",
        )
        # Fit both R and Omega_m for flat ΛCDM
        R_lcdm_i, Omega_m_i = get_radius_and_omega(
            binned_data.z_mid.values,
            binned_data.theta_med.values,
            binned_data.theta_err.values,
        )

        # Use full z range for model curves (consistent across all panels)
        z_model_i = np.linspace(max(0.05, z_lim[0]), z_lim[1], 200)
        theta_static_i = theta_static(z_model_i, R_static_i) * RAD_TO_ARCSEC
        theta_lcdm_i = theta_lcdm_flat(z_model_i, R_lcdm_i, Omega_m_i) * RAD_TO_ARCSEC

        # Compute chi2/ndf for both models
        def static_func_i(z, R=R_static_i):
            return theta_static(z, R) * RAD_TO_ARCSEC
        def lcdm_func_i(z, R=R_lcdm_i, Om=Omega_m_i):
            return theta_lcdm_flat(z, R, Om) * RAD_TO_ARCSEC

        stats_static_i = compute_chi2_stats(
            binned_data.z_mid.values, binned_data.theta_med.values,
            binned_data.theta_err.values, static_func_i, n_params=1
        )
        stats_lcdm_i = compute_chi2_stats(
            binned_data.z_mid.values, binned_data.theta_med.values,
            binned_data.theta_err.values, lcdm_func_i, n_params=2
        )

        # Plot individual galaxies as background
        ax_main.scatter(
            sed_catalog_analysis["redshift"],
            sed_catalog_analysis["r_half_arcsec"],
            c="lightgray",
            alpha=0.4,
            s=15,
            zorder=1,
        )

        # Plot binned data (no connecting lines)
        ax_main.errorbar(
            binned_data["z_mid"],
            binned_data["theta_med"],
            yerr=binned_data["theta_err"],
            fmt=markers[name],
            capsize=3,
            capthick=0.8,
            markersize=6,
            markeredgewidth=0.8,
            elinewidth=0.8,
            color=colors[name],
            label="Binned",
            zorder=3,
        )

        # Plot models
        ax_main.plot(
            z_model_i, theta_static_i, "--", color="gray", linewidth=1.5, label=r"Static", zorder=2
        )
        ax_main.plot(
            z_model_i, theta_lcdm_i, "-", color="darkblue", linewidth=1.5, label=r"$\Lambda$CDM", zorder=2
        )

        ax_main.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=11)
        ax_main.set_title(
            f"{name} ({len(binned_data)} bins)\n"
            + r"$\Lambda$CDM: $R$" + f"={R_lcdm_i*1000:.1f} kpc, "
            + r"$\chi^2/{\rm ndf}$" + f"={stats_lcdm_i['chi2_ndf']:.2f}, "
            + f"P={stats_lcdm_i['p_value']:.3f}\n"
            + r"Static: $R$" + f"={R_static_i*1000:.1f} kpc, "
            + r"$\chi^2/{\rm ndf}$" + f"={stats_static_i['chi2_ndf']:.2f}, "
            + f"P={stats_static_i['p_value']:.3f}",
            fontsize=10
        )
        ax_main.legend(fontsize=8, loc="upper right")
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xlim(z_lim)
        ax_main.set_ylim(theta_lim)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # Calculate residuals: (data - ΛCDM model) in arcsec
        theta_lcdm_at_data = theta_lcdm_flat(binned_data["z_mid"].values, R_lcdm_i, Omega_m_i) * RAD_TO_ARCSEC
        residuals_lcdm = binned_data["theta_med"].values - theta_lcdm_at_data

        # Residual plot
        ax_resid.axhline(0, color="darkblue", linestyle="-", linewidth=1.5, alpha=0.7)
        ax_resid.errorbar(
            binned_data["z_mid"],
            residuals_lcdm,
            yerr=binned_data["theta_err"],
            fmt=markers[name],
            capsize=2,
            capthick=0.6,
            markersize=5,
            markeredgewidth=0.6,
            elinewidth=0.6,
            color=colors[name],
            zorder=3,
        )
        ax_resid.set_xlabel(r"Redshift $z$", fontsize=11)
        ax_resid.set_ylabel(r"$\theta - \theta_{\Lambda\mathrm{CDM}}$", fontsize=9)
        ax_resid.grid(True, alpha=0.3)
        ax_resid.set_xlim(z_lim)
        max_resid = max(0.05, np.max(np.abs(residuals_lcdm) + binned_data["theta_err"].values) * 1.2)
        ax_resid.set_ylim(-max_resid, max_resid)

    fig.suptitle(r"Binning Strategies: $\Lambda$CDM vs Static", fontsize=14, fontweight="bold")
    plt.savefig(f"{output_dir}/binning_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/binning_comparison.pdf")

    # Plot 4: All binning strategies overlaid
    _fig, ax = plt.subplots(figsize=(12, 8))

    # Background scatter
    ax.scatter(
        sed_catalog_analysis["redshift"],
        sed_catalog_analysis["r_half_arcsec"],
        c="lightgray",
        alpha=0.3,
        s=15,
        label="Individual galaxies",
        zorder=1,
    )

    for name, binned_data in binned_results.items():
        ax.errorbar(
            binned_data["z_mid"],
            binned_data["theta_med"],
            yerr=binned_data["theta_err"],
            fmt=f"{markers[name]}-",
            capsize=2,
            capthick=0.6,
            markersize=5,
            markeredgewidth=0.6,
            elinewidth=0.6,
            linewidth=1.0,
            color=colors[name],
            label=f"{name}",
            alpha=0.8,
            zorder=2,
        )

    # Add ΛCDM model curve (using equal-count fit)
    ax.plot(
        z_model, theta_lcdm_model, "-", color="black", linewidth=2, label=r"$\Lambda$CDM model", zorder=3
    )
    ax.plot(
        z_model,
        theta_static_model,
        "--",
        color="black",
        linewidth=2,
        label=r"Static model",
        zorder=3,
    )

    ax.set_xlabel(r"Redshift $z$", fontsize=12)
    ax.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=12)
    ax.set_title(r"All Binning Strategies Compared", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(z_lim)
    ax.set_ylim(theta_lim)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/binning_overlay.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/binning_overlay.pdf")

    # Plot 5: Redshift histogram with dynamic y-axis
    _fig, ax = plt.subplots(figsize=(8, 5))
    hist_bins = np.linspace(z_data_min, z_data_max, 15)
    counts, _, _ = ax.hist(sed_catalog_analysis["redshift"], bins=hist_bins, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel(r"Redshift $z$", fontsize=12)
    ax.set_ylabel(r"Number of galaxies", fontsize=12)
    ax.set_title(f"Redshift Distribution (N={len(sed_catalog_analysis)}, quality-filtered)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(z_lim)
    ax.set_ylim(0, np.max(counts) * 1.1)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/redshift_histogram.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/redshift_histogram.pdf")

    # Plot 6: Galaxy type distribution with dynamic y-axis
    __fig, ax = plt.subplots(figsize=(10, 5))
    type_counts = sed_catalog_analysis["galaxy_type"].value_counts()
    type_counts.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel(r"Galaxy Type", fontsize=12)
    ax.set_ylabel(r"Count", fontsize=12)
    ax.set_title(r"Galaxy Type Distribution (quality-filtered)", fontsize=14)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    max_count = type_counts.max()
    ax.set_ylim(0, max_count * 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/galaxy_types.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/galaxy_types.pdf")

    # Plot 7: Angular diameter distance comparison with dynamic limits
    __fig, ax = plt.subplots(figsize=(10, 7))

    # Extend z_range with dynamic epsilon padding
    z_plot_padding = (z_data_max - z_data_min) * 0.15
    z_plot_max = z_data_max + z_plot_padding
    z_range = np.linspace(0.01, z_plot_max, 200)
    D_A_static = (c / H0) * z_range  # Linear Hubble law
    D_A_lcdm = np.array([D_A_LCDM_vectorized(z) for z in z_range])

    ax.plot(z_range, D_A_static, "--", color="gray", linewidth=2, label=r"Linear Hubble law")
    ax.plot(z_range, D_A_lcdm, "-", color="blue", linewidth=2, label=r"$\Lambda$CDM")

    # Mark the data range
    ax.axvspan(z_data_min, z_data_max, alpha=0.1, color="green", label=r"Data range")

    ax.set_xlabel(r"Redshift $z$", fontsize=12)
    ax.set_ylabel(r"Angular Diameter Distance $D_A$ (Mpc)", fontsize=12)
    ax.set_title(r"Angular Diameter Distance vs Redshift" + "\n" + r"(Model Comparison)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    # Dynamic axis limits
    D_A_max = max(np.max(D_A_static), np.max(D_A_lcdm))
    ax.set_xlim(0, z_plot_max * 1.02)
    ax.set_ylim(0, D_A_max * 1.1)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/angular_diameter_distance.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/angular_diameter_distance.pdf")

    # Plot 8: Source detection image
    # Separate sources into 3 categories:
    # 1. Purple: sources inside mask (stars by mask)
    # 2. Blue: stars detected by stellarity/concentration (NOT in mask)
    # 3. Red: galaxies (not flagged as stars at all)

    is_in_mask = cross_matched_catalog["in_star_mask"]
    is_detected_star = ((cross_matched_catalog["quality_flag"] & FLAG_PSF_LIKE) != 0) & ~is_in_mask
    is_galaxy = ((cross_matched_catalog["quality_flag"] & FLAG_PSF_LIKE) == 0) & ~is_in_mask

    is_in_mask.sum()
    is_detected_star.sum()
    n_galaxies = is_galaxy.sum()

    _fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for idx, image in enumerate(images):
        ax = axes[idx // 2, idx % 2]
        vmin, vmax = np.percentile(image.data, [20, 99])
        ax.imshow(image.data, cmap="gray_r", vmin=vmin, vmax=vmax, origin="lower")

        # Red: galaxies (not flagged as stars)
        if n_galaxies > 0:
            ax.scatter(
                cross_matched_catalog.loc[is_galaxy, "xcentroid"],
                cross_matched_catalog.loc[is_galaxy, "ycentroid"],
                s=10,
                facecolors="none",
                edgecolors="red",
                linewidth=0.5,
            )

        ax.set_title(
            f"Band {image.band.upper()} -- {n_galaxies} galaxies",
            fontsize=9,
        )
        ax.set_xlabel(r"X (pixels)")
        ax.set_ylabel(r"Y (pixels)")

    plt.suptitle(f"Source Detection ({n_galaxies} galaxies)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/source_detection.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/source_detection.pdf")

    # Plot 9: Final selection (cross-matched sources in all 4 bands)

    _fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for idx, image in enumerate(images):
        ax = axes[idx // 2, idx % 2]
        vmin, vmax = np.percentile(image.data, [20, 99])
        ax.imshow(image.data, cmap="gray_r", vmin=vmin, vmax=vmax, origin="lower")

        # Red: galaxies (not flagged as stars)
        if n_galaxies > 0:
            ax.scatter(
                cross_matched_catalog.loc[is_galaxy, "xcentroid"],
                cross_matched_catalog.loc[is_galaxy, "ycentroid"],
                s=10,
                facecolors="none",
                edgecolors="red",
                linewidth=0.5,
            )

        ax.set_title(
            f"Band {image.band.upper()} -- {n_galaxies} galaxies",
            fontsize=9,
        )
        ax.set_xlabel(r"X (pixels)")
        ax.set_ylabel(r"Y (pixels)")

    plt.suptitle(f"Final Selection ({n_galaxies} galaxies)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_selection.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/final_selection.pdf")

    # Save catalogs (both full and quality-filtered versions)
    # Full catalog with all sources and flags
    sed_catalog.to_csv(f"{output_dir}/galaxy_catalog_full.csv", index=False)
    print(f"  Saved: {output_dir}/galaxy_catalog_full.csv (all {len(sed_catalog)} sources)")

    # Quality-filtered catalog (recommended for science)
    sed_catalog_analysis.to_csv(f"{output_dir}/galaxy_catalog.csv", index=False)
    print(f"  Saved: {output_dir}/galaxy_catalog.csv ({len(sed_catalog_analysis)} quality-filtered sources)")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nCatalog columns include:")
    print("  - redshift, redshift_lo, redshift_hi, redshift_err: Photo-z with uncertainties")
    print("  - photo_z_odds: BPZ-style quality parameter (>0.9 = excellent)")
    print("  - chi_sq_min: Best-fit chi-squared")
    print("  - r_half_arcsec: PSF-corrected half-light radius")
    print("  - concentration: r50/r90 index (higher = more concentrated)")
    print("  - ellipticity: Source ellipticity (0=round, 1=elongated)")
    print("  - stellarity: Multi-criteria star index (0=galaxy, 1=star; SExtractor-style)")
    print("  - color_stellarity: Color-based stellar index (flat SED = stellar)")
    print("  - in_star_mask: True if source is inside course staff star mask")
    print("  - quality_flag: Bit-packed flags (0 = clean)")
    print("  - snr_median, snr_min: Signal-to-noise ratios")
    print("\nStar/galaxy separation methodology:")
    print("  - Concentration index C = r50/r90 (SDSS methodology)")
    print("  - Stellarity score using size, concentration, ellipticity")
    print("  - Color-based check (stars have flat SEDs)")
    print("  - Star mask regions (from course staff) -> FLAG_MASKED")
    print("  - References: SExtractor CLASS_STAR, SDSS DR17, COSMOS2020")

    return sed_catalog, sed_catalog_analysis, binned, binned_results, star_mask_centers


def extract_chip3_from_full(full_catalog: pd.DataFrame, full_catalog_filtered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract chip3 sources from full mosaic catalog.

    Uses the known chip3 bounds within the 4096x4096 mosaic to filter sources.
    This avoids re-running detection and classification.
    """
    # Filter to chip3 region (columns are xcentroid/ycentroid)
    chip3_mask = (
        (full_catalog["xcentroid"] >= CHIP3_X_MIN) & (full_catalog["xcentroid"] <= CHIP3_X_MAX) &
        (full_catalog["ycentroid"] >= CHIP3_Y_MIN) & (full_catalog["ycentroid"] <= CHIP3_Y_MAX)
    )
    chip3_catalog = full_catalog[chip3_mask].copy()

    chip3_mask_filtered = (
        (full_catalog_filtered["xcentroid"] >= CHIP3_X_MIN) & (full_catalog_filtered["xcentroid"] <= CHIP3_X_MAX) &
        (full_catalog_filtered["ycentroid"] >= CHIP3_Y_MIN) & (full_catalog_filtered["ycentroid"] <= CHIP3_Y_MAX)
    )
    chip3_catalog_filtered = full_catalog_filtered[chip3_mask_filtered].copy()

    return chip3_catalog, chip3_catalog_filtered


def run_quality_control(
    catalog: pd.DataFrame,
    output_dir: str,
    reference_catalog_path: str | None = None,
) -> None:
    """Run quality control pipeline on classification results.

    Parameters
    ----------
    catalog : pd.DataFrame
        Galaxy catalog with classification results
    output_dir : str
        Output directory for QC results
    reference_catalog_path : str, optional
        Path to external reference catalog (CSV with ra, dec, z_phot columns)
    """
    from quality_control import QualityControlPipeline

    print("\n" + "=" * 60)
    print("RUNNING QUALITY CONTROL PIPELINE")
    print("=" * 60)

    # Load reference catalog if provided
    reference_catalog = None
    if reference_catalog_path:
        try:
            reference_catalog = pd.read_csv(reference_catalog_path)
            print(f"  Loaded reference catalog: {reference_catalog_path}")
            print(f"  Reference sources: {len(reference_catalog)}")
        except Exception as e:
            print(f"  Warning: Could not load reference catalog: {e}")

    # Create QC output directory
    qc_output_dir = f"{output_dir}/qc"

    # Run QC pipeline
    qc = QualityControlPipeline(
        catalog=catalog,
        output_dir=qc_output_dir,
        reference_catalog=reference_catalog,
    )

    qc.run_full_validation(generate_plots=True)
    qc.save_report()

    print(f"\nQC results saved to: {qc_output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Angular size test analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_analysis.py full                          # Run analysis on full mosaic
    python run_analysis.py chip3                         # Run analysis on chip3 only
    python run_analysis.py both                          # Run full and extract chip3
    python run_analysis.py full --run-qc                 # Run with quality control
        """
    )
    parser.add_argument(
        "mode",
        choices=["full", "chip3", "both"],
        help="Analysis mode: 'full' for entire image, 'chip3' for chip3 only, 'both' to run full and extract chip3"
    )
    parser.add_argument(
        "--run-qc",
        action="store_true",
        help="Run quality control pipeline after classification"
    )
    parser.add_argument(
        "--qc-reference",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to external reference catalog for QC validation (CSV with ra, dec, z_phot columns)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ANGULAR SIZE TEST ANALYSIS")
    print(f"Mode: {args.mode}")
    print("=" * 60)

    if args.mode == "both":
        # Run full analysis first
        print("\n" + "=" * 60)
        print("RUNNING ANALYSIS (FULL)")
        print("=" * 60)
        sed_catalog, sed_catalog_analysis, binned, binned_results, star_mask_centers = main("full")

        # Run QC if requested
        if args.run_qc:
            run_quality_control(sed_catalog, "./output/full_HDF", args.qc_reference)

        # Extract chip3 from full results
        print("\n" + "=" * 60)
        print("EXTRACTING CHIP3 SUBSET FROM FULL RESULTS")
        print("=" * 60)

        chip3_catalog, chip3_catalog_filtered = extract_chip3_from_full(
            sed_catalog, sed_catalog_analysis
        )

        print(f"\nChip3 region in mosaic: X=[{CHIP3_X_MIN}, {CHIP3_X_MAX}], Y=[{CHIP3_Y_MIN}, {CHIP3_Y_MAX}]")
        print(f"Extracted {len(chip3_catalog)} sources ({len(chip3_catalog_filtered)} quality-filtered)")

        # Load full images for chip3 detection plots
        print("\n  Loading full images for chip3 detection plots...")
        full_files = {
            "b": "./fits/b_full.fits",
            "i": "./fits/i_full.fits",
            "u": "./fits/u_full.fits",
            "v": "./fits/v_full.fits",
        }
        chip3_images = []
        for band, path in full_files.items():
            data, header, _ = read_fits(path)
            chip3_images.append(AstroImage(data=data, header=header, band=band, weight=None))

        # Generate chip3 plots
        print("\n" + "=" * 60)
        print("GENERATING CHIP3 PLOTS")
        print("=" * 60)
        chip3_output_dir = "./output/chip3_HDF"
        crop_bounds = (CHIP3_X_MIN, CHIP3_X_MAX, CHIP3_Y_MIN, CHIP3_Y_MAX)
        analyze_and_plot_catalog(
            chip3_catalog, chip3_catalog_filtered, chip3_output_dir,
            title_suffix=" (Chip 3)", images=chip3_images, crop_bounds=crop_bounds,
            star_mask_centers=star_mask_centers
        )

        # Generate plots for both full and chip3
        try:
            from generate_filtered_binning import (
                MIN_GALAXIES_PER_TYPE,
                generate_angular_size_plot,
                generate_binning_plot,
                generate_galaxy_type_histogram,
                generate_overlay_plot,
                generate_per_type_fit_plot,
                generate_redshift_histogram,
                generate_size_distribution_by_type,
                generate_type_comparison_overlay,
                get_types_with_enough_statistics,
            )
            print("\n" + "=" * 60)
            print("GENERATING PLOTS")
            print("=" * 60)

            for mode_name, output_dir in [
                ("Full Field", "./output/full_HDF"),
                ("Chip3", "./output/chip3_HDF")
            ]:
                # All types combined plots
                generate_binning_plot(data_dir=output_dir,
                    output_path=f"{output_dir}/binning_comparison.pdf",
                    dataset_name=mode_name, apply_z_filter=False)
                generate_overlay_plot(data_dir=output_dir,
                    output_path=f"{output_dir}/binning_overlay.pdf",
                    dataset_name=mode_name, apply_z_filter=False)

                # Angular size vs redshift (main science plot)
                generate_angular_size_plot(data_dir=output_dir,
                    output_path=f"{output_dir}/angular_size_vs_redshift.pdf",
                    dataset_name=mode_name, apply_z_filter=False)

                # Redshift distribution histograms
                generate_redshift_histogram(data_dir=output_dir,
                    output_path=f"{output_dir}/redshift_histogram.pdf",
                    dataset_name=mode_name, apply_z_filter=False)

                # Galaxy type histograms
                generate_galaxy_type_histogram(data_dir=output_dir,
                    output_path=f"{output_dir}/galaxy_types.pdf",
                    dataset_name=mode_name, apply_z_filter=False)

                # Per-galaxy-type analysis
                print(f"\n--- Per-Galaxy-Type Analysis ({mode_name}) ---")
                catalog_path = f"{output_dir}/galaxy_catalog.csv"
                if os.path.exists(catalog_path):
                    type_catalog = pd.read_csv(catalog_path)
                    type_catalog = type_catalog.dropna(subset=["redshift", "r_half_arcsec", "r_half_arcsec_error"])
                    type_catalog = type_catalog[np.isfinite(type_catalog["r_half_arcsec_error"])]

                    valid_types = get_types_with_enough_statistics(type_catalog)
                    print(f"  Galaxy types with N >= {MIN_GALAXIES_PER_TYPE}: {valid_types}")
                    if len(valid_types) > 0:
                        generate_size_distribution_by_type(output_dir, output_dir, mode_name)
                        generate_type_comparison_overlay(output_dir, output_dir, mode_name)
                        print(f"  Generating individual θ(z) fits for {len(valid_types)} types...")
                        for gtype in valid_types:
                            generate_per_type_fit_plot(output_dir, output_dir, mode_name, gtype)
                    else:
                        print("  No galaxy types have enough statistics for per-type analysis.")

            print("\nPlots saved to: ./output/full_HDF/ and ./output/chip3_HDF/")
        except ImportError as e:
            print(f"\nWarning: Could not generate plots: {e}")

    else:
        # Single mode (full or chip3)
        sed_catalog, sed_catalog_analysis, binned, binned_results, star_mask_centers = main(args.mode)

        # Run QC if requested
        if args.run_qc:
            output_dir = f"./output/{args.mode}_HDF"
            run_quality_control(sed_catalog, output_dir, args.qc_reference)

        # Generate plots
        try:
            from generate_filtered_binning import (
                MIN_GALAXIES_PER_TYPE,
                generate_angular_size_plot,
                generate_binning_plot,
                generate_galaxy_type_histogram,
                generate_overlay_plot,
                generate_per_type_fit_plot,
                generate_redshift_histogram,
                generate_size_distribution_by_type,
                generate_type_comparison_overlay,
                get_types_with_enough_statistics,
            )
            output_dir = f"./output/{args.mode}_HDF"
            dataset_name = "Full Field" if args.mode == "full" else "Chip3"
            print("\n" + "=" * 60)
            print("GENERATING PLOTS")
            print("=" * 60)

            # Binning comparison (all types)
            generate_binning_plot(data_dir=output_dir,
                output_path=f"{output_dir}/binning_comparison.pdf",
                dataset_name=dataset_name, apply_z_filter=False)

            # Overlay (all types)
            generate_overlay_plot(data_dir=output_dir,
                output_path=f"{output_dir}/binning_overlay.pdf",
                dataset_name=dataset_name, apply_z_filter=False)

            # Angular size vs redshift (main science plot)
            generate_angular_size_plot(data_dir=output_dir,
                output_path=f"{output_dir}/angular_size_vs_redshift.pdf",
                dataset_name=dataset_name, apply_z_filter=False)

            # Redshift distribution histograms
            generate_redshift_histogram(data_dir=output_dir,
                output_path=f"{output_dir}/redshift_histogram.pdf",
                dataset_name=dataset_name, apply_z_filter=False)

            # Galaxy type histograms
            generate_galaxy_type_histogram(data_dir=output_dir,
                output_path=f"{output_dir}/galaxy_types.pdf",
                dataset_name=dataset_name, apply_z_filter=False)

            # Per-galaxy-type analysis
            catalog_path = f"{output_dir}/galaxy_catalog.csv"
            if os.path.exists(catalog_path):
                type_catalog = pd.read_csv(catalog_path)
                type_catalog = type_catalog.dropna(subset=["redshift", "r_half_arcsec", "r_half_arcsec_error"])
                type_catalog = type_catalog[np.isfinite(type_catalog["r_half_arcsec_error"])]

                print(f"\n--- Per-Galaxy-Type Analysis ({dataset_name}) ---")
                valid_types = get_types_with_enough_statistics(type_catalog)
                print(f"  Galaxy types with N >= {MIN_GALAXIES_PER_TYPE}: {valid_types}")
                if len(valid_types) > 0:
                    generate_size_distribution_by_type(output_dir, output_dir, dataset_name)
                    generate_type_comparison_overlay(output_dir, output_dir, dataset_name)
                    print(f"  Generating individual θ(z) fits for {len(valid_types)} types...")
                    for gtype in valid_types:
                        generate_per_type_fit_plot(output_dir, output_dir, dataset_name, gtype)
                else:
                    print("  No galaxy types have enough statistics for per-type analysis.")

            print(f"\nPlots saved to: {output_dir}/")
        except ImportError as e:
            print(f"\nWarning: Could not generate plots: {e}")
