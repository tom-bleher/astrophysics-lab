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

# Import resource configuration FIRST (before numpy/numba)
from resource_config import get_config

# Get resource configuration (auto-detects or uses ASTRO_RESOURCE_PROFILE env var)
_RESOURCE_CONFIG = get_config()
_RESOURCE_CONFIG.apply_environment()

# Legacy compatibility
_N_THREADS = str(_RESOURCE_CONFIG.n_threads)

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend before importing pyplot

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Configure matplotlib (LaTeX disabled for compatibility)
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "axes.labelsize": 12,
    "font.size": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
import numpy as np
import pandas as pd
from astropy.convolution import convolve
from astropy.io import fits
from astropy.stats import bayesian_blocks
from astropy.wcs import WCS as AstropyWCS
from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture, aperture_photometry
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import (
    SourceCatalog,
    SourceFinder,
    deblend_sources,
    detect_sources,
    make_2dgaussian_kernel,
)
from photutils.utils import calc_total_error
from scipy import ndimage
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from classify import (
    classify_batch_ultrafast,
)
from generate_filtered_binning import (
    MIN_GALAXIES_PER_TYPE,
    MIN_GALAXIES_PER_TYPE_BINNING,
    generate_angular_size_plot,
    generate_binning_plot,
    generate_galaxy_type_histogram,
    generate_per_type_binning_comparison,
    generate_per_type_fit_plot,
    generate_redshift_histogram,
    generate_size_distribution_by_type,
    generate_type_comparison_overlay,
    get_types_with_enough_statistics,
)
from scientific import (
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
from morphology.zoobot_classify import extract_cutouts, classify_morphology
from morphology.star_classifier import classify_stars

# SEP detection backend (optional, faster alternative to photutils)
try:
    from detection import SEP_AVAILABLE, detect_and_measure as sep_detect_and_measure
except ImportError:
    SEP_AVAILABLE = False

# =============================================================================
# Configuration
# =============================================================================

# Configuration
PIXEL_SCALE = 0.04  # arcsec/pixel
RAD_TO_ARCSEC = 180.0 * 3600.0 / np.pi  # ~206265 arcsec/radian

# Photo-z redshift grid boundaries
# z_min=0.05 prevents boundary pileup at z=0 from template fitting failures
Z_MIN = 0.05  # Minimum redshift for photo-z fitting
Z_MAX = 6.0   # Maximum redshift for photo-z fitting

# Chip3 bounds within the 4096x4096 mosaic (from template matching)
# Chip3 is 2048x2048, placed at offset (127, 184) after 180° rotation
CHIP3_X_MIN, CHIP3_X_MAX = 127, 2175  # 127 + 2048
CHIP3_Y_MIN, CHIP3_Y_MAX = 184, 2232  # 184 + 2048

# =============================================================================
# Photometry Constants (HST WFPC2 HDF-N specific)
# =============================================================================
# References:
# - HDF Data Reduction: Williams et al. 1996, AJ, 112, 1335
# - DrizzlePac Handbook: https://hst-docs.stsci.edu/drizzpac
# - WFPC2 Instrument Handbook

# WFPC2 gain for HDF observations
# Note: The HDF v2 drizzled mosaics are in units of electrons/second, not DN
# For drizzled data, gain should be None (background noise dominates)
# Setting gain=None tells SEP to use only the background RMS for noise estimation
WFPC2_GAIN = None  # None for drizzled e-/s data (not raw DN)

# Drizzle correlated noise correction factor
# HDF v2 used pixfrac=0.6, scale=0.5 (output pixel = 0.5x input)
# This creates noise correlation between adjacent output pixels
# Correction factor depends on drizzle parameters:
#   R ≈ 1 / sqrt(pixfrac) for pixfrac < 1
# For pixfrac=0.6: R ≈ 1.29
# Adding margin for aperture summation effects: ~1.4
# Reference: Fruchter & Hook 2002, PASP, 114, 144
DRIZZLE_NOISE_CORR = 1.4

# Background estimation uncertainty factor
# When estimating background from annulus, there's uncertainty in that estimate
# This adds variance: area² × σ_sky² / nsky
# For segment-based photometry, we approximate nsky ~ 10 × source_area
SKY_ANNULUS_FACTOR = 10.0

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
FLAG_VERY_LOW_SNR = 512 # Very low SNR (median < 0.5) - unreliable photometry
FLAG_HIGH_Z_UNRELIABLE = 1024  # High redshift (z > 2) with large uncertainty
FLAG_CATASTROPHIC_PHOTOZ = 2048  # Photo-z catastrophic outlier (|Δz/(1+z)| > 0.15 vs spec-z)

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
    FLAG_VERY_LOW_SNR: "very_low_snr",
    FLAG_HIGH_Z_UNRELIABLE: "high_z_unreliable",
    FLAG_CATASTROPHIC_PHOTOZ: "catastrophic_photoz",
}


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

    # Handle case where all colors are NaN (check before computing stats)
    all_nan = np.all(np.isnan(colors), axis=1)

    # Compute scatter (std) and max color per source
    # Use nanstd/nanmax to handle NaN values, suppress expected warnings for all-NaN rows
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice encountered')
            warnings.filterwarnings('ignore', 'Degrees of freedom <= 0 for slice')
            color_scatter = np.nanstd(colors, axis=1)
            max_color = np.nanmax(np.abs(colors), axis=1)

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


def bin_percentile(sed_catalog, n_bins=None):
    """Percentile bins (quantile-based) for similar statistical weight per bin."""
    if n_bins is None:
        n_bins = get_adaptive_n_bins(len(sed_catalog))
    sed_catalog = sed_catalog.copy()
    sed_catalog["z_bin"] = pd.qcut(sed_catalog.redshift, q=n_bins, duplicates="drop")
    return aggregate_bins(sed_catalog)


def bin_bayesian_blocks(sed_catalog, p0=0.05):
    """Bayesian Blocks adaptive binning (Scargle et al. 2013).

    Uses astropy's bayesian_blocks algorithm to find optimal bin edges
    based on the data structure. This is data-driven binning that creates
    more bins where data is dense/variable and fewer where sparse/uniform.

    Args:
        sed_catalog: DataFrame with 'redshift' and 'r_half_arcsec' columns
        p0: False alarm probability (default 0.05). Lower values = fewer bins.

    Returns:
        DataFrame with binned statistics (z_mid, theta_med, theta_err, n)
    """
    sed_catalog = sed_catalog.copy()
    z = sed_catalog.redshift.values
    theta = sed_catalog.r_half_arcsec.values

    # Use point measures fitness with angular sizes as values
    # sigma estimated as MAD-based robust standard deviation
    sigma = np.median(np.abs(theta - np.median(theta))) * 1.4826
    sigma = max(sigma, 0.01 * np.median(theta))  # Floor at 1% of median

    try:
        # Get optimal bin edges from Bayesian Blocks
        edges = bayesian_blocks(z, theta, sigma=sigma, fitness='measures', p0=p0)

        # Ensure we have at least 3 bins for meaningful analysis
        if len(edges) < 4:
            # Fall back to equal-width with 3 bins if BB gives too few
            edges = np.linspace(z.min(), z.max(), 4)

        # Limit to reasonable number of bins (max ~15)
        if len(edges) > 16:
            # Too many bins - increase p0 to reduce sensitivity
            edges = bayesian_blocks(z, theta, sigma=sigma, fitness='measures', p0=0.2)
            if len(edges) > 16:
                edges = np.linspace(z.min(), z.max(), 12)
    except Exception:
        # Fallback to equal-width binning if Bayesian Blocks fails
        n_bins = get_adaptive_n_bins(len(sed_catalog))
        edges = np.linspace(z.min(), z.max(), n_bins + 1)

    sed_catalog["z_bin"] = pd.cut(sed_catalog.redshift, edges, include_lowest=True)
    return aggregate_bins(sed_catalog)


def standard_error_median(x):
    """Standard error of the median using analytical approximation.

    For approximately normal data: SE(median) ~ 1.2533 * sigma / sqrt(n)
    This is the standard approach used in professional astronomy surveys.
    Reference: Euclid Consortium methodology, Astropy robust statistics.
    """
    x = x.dropna() if hasattr(x, 'dropna') else x[np.isfinite(x)]
    if len(x) < 2:
        # Single source: use a default fractional error (10% of median value)
        return 0.1 * np.median(x) if len(x) == 1 else np.nan
    if len(x) < 3:
        # Two sources: use half the range as error estimate
        return (np.max(x) - np.min(x)) / 2.0
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


class SEPCatalogWrapper:
    """Wrapper to make SEP catalog output compatible with photutils SourceCatalog interface.

    Provides the same interface expected by the cross-matching code:
    - to_table().to_pandas() -> returns the catalog DataFrame
    - fluxfrac_radius(frac) -> returns radius containing frac of flux
    - labels -> segment labels
    """

    def __init__(
        self,
        catalog_df: pd.DataFrame,
        segmentation: np.ndarray | None = None,
        r_half: np.ndarray | None = None,
        r_90: np.ndarray | None = None,
    ):
        self._catalog = catalog_df
        self._segm = segmentation
        if r_half is not None:
            self._r_half = r_half
        elif 'r_half' in catalog_df.columns:
            self._r_half = catalog_df['r_half'].values
        else:
            self._r_half = np.full(len(catalog_df), np.nan)
        if r_90 is not None:
            self._r_90 = r_90
        elif 'r_90' in catalog_df.columns:
            self._r_90 = catalog_df['r_90'].values
        else:
            self._r_90 = np.full(len(catalog_df), np.nan)
        # Labels are 1-indexed sequential for SEP
        self._labels = np.arange(1, len(catalog_df) + 1)

    def to_table(self):
        """Return self to allow chaining .to_pandas()."""
        return self

    def to_pandas(self):
        """Return the catalog as a pandas DataFrame."""
        df = self._catalog.copy()
        # Rename SEP columns to match photutils convention
        if 'x' in df.columns and 'xcentroid' not in df.columns:
            df = df.rename(columns={'x': 'xcentroid', 'y': 'ycentroid'})
        return df

    def fluxfrac_radius(self, fraction: float) -> np.ndarray:
        """Return radius containing specified fraction of flux."""
        if fraction <= 0.5:
            return self._r_half
        else:
            return self._r_90

    @property
    def labels(self) -> np.ndarray:
        """Return segment labels."""
        return self._labels

    def __len__(self):
        return len(self._catalog)


class SEPBackgroundWrapper:
    """Wrapper to make SEP background compatible with photutils Background2D interface."""

    def __init__(self, sep_background):
        self.background = sep_background.background
        self.background_rms = sep_background.background_rms
        self.background_rms_median = float(np.median(sep_background.background_rms))


@dataclass
class AstroImage:
    data: np.ndarray
    header: fits.Header
    band: str = ""
    weight: np.ndarray | None = None  # Inverse variance weight map
    segm: object | None = None
    catalog: SourceCatalog | SEPCatalogWrapper | None = None
    bkg: Background2D | SEPBackgroundWrapper | None = None

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


def read_fits(file_path, weight_path=None, clean_cosmic_rays_flag=None, instrument='WFPC2'):
    """Read FITS file with optional weight map and cosmic ray removal.

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
    with fits.open(file_path, memmap=True) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        # Use native byte order for efficient numpy operations
        if data.dtype.byteorder == ">":
            data = data.astype(data.dtype.newbyteorder("="), copy=False)
        else:
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
# Analysis and Plotting
# ============================================================================


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

    # Exclude z_boundary_flag sources from analysis (photo-z boundary failures)
    if "z_boundary_flag" in sed_catalog_analysis.columns:
        n_before = len(sed_catalog_analysis)
        sed_catalog_analysis = sed_catalog_analysis[sed_catalog_analysis["z_boundary_flag"] == False].copy()
        n_excluded = n_before - len(sed_catalog_analysis)
        if n_excluded > 0:
            print(f"  Excluded {n_excluded} sources with z_boundary_flag=True (photo-z boundary failures)")

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
        "Percentile": bin_percentile,
        "Bayesian Blocks": bin_bayesian_blocks,
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

    binned = binned_results.get("Percentile", next(iter(binned_results.values())))

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

    # Plot 3: Binning strategy comparison (1x3 grid with residuals)
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[3, 1, 1], hspace=0.08, wspace=0.15)

    colors = {
        "Equal Width": "green",
        "Percentile": "blue",
        "Bayesian Blocks": "purple",
    }
    markers = {"Equal Width": "o", "Percentile": "o", "Bayesian Blocks": "o"}

    strategy_names = list(binned_results.keys())

    for idx, name in enumerate(strategy_names):
        binned_data = binned_results[name]
        col = idx  # 1x3 layout: each strategy in its own column

        ax_main = fig.add_subplot(gs[0, col])
        ax_resid_lcdm = fig.add_subplot(gs[1, col], sharex=ax_main)
        ax_resid_static = fig.add_subplot(gs[2, col], sharex=ax_main)

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

        # Calculate residuals
        theta_lcdm_at_data = theta_lcdm_flat(binned_data["z_mid"].values, R_lcdm_i, Omega_m_i) * RAD_TO_ARCSEC
        theta_static_at_data = theta_static(binned_data["z_mid"].values, R_static_i) * RAD_TO_ARCSEC
        residuals_lcdm = binned_data["theta_med"].values - theta_lcdm_at_data
        residuals_static = binned_data["theta_med"].values - theta_static_at_data

        # ΛCDM residual plot
        ax_resid_lcdm.axhline(0, color="darkblue", linestyle="-", linewidth=1.5, alpha=0.7)
        ax_resid_lcdm.errorbar(
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
        ax_resid_lcdm.set_ylabel(r"$\theta - \theta_{\Lambda\mathrm{CDM}}$", fontsize=9)
        ax_resid_lcdm.grid(True, alpha=0.3)
        ax_resid_lcdm.set_xlim(z_lim)
        max_resid_lcdm = max(0.05, np.max(np.abs(residuals_lcdm) + binned_data["theta_err"].values) * 1.2)
        ax_resid_lcdm.set_ylim(-max_resid_lcdm, max_resid_lcdm)
        plt.setp(ax_resid_lcdm.get_xticklabels(), visible=False)

        # Static residual plot
        ax_resid_static.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax_resid_static.errorbar(
            binned_data["z_mid"],
            residuals_static,
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
        ax_resid_static.set_xlabel(r"Redshift $z$", fontsize=11)
        ax_resid_static.set_ylabel(r"$\theta - \theta_{\mathrm{Static}}$", fontsize=9)
        ax_resid_static.grid(True, alpha=0.3)
        ax_resid_static.set_xlim(z_lim)
        max_resid_static = max(0.05, np.max(np.abs(residuals_static) + binned_data["theta_err"].values) * 1.2)
        ax_resid_static.set_ylim(-max_resid_static, max_resid_static)

    fig.suptitle(r"Binning Strategies: $\Lambda$CDM vs Static" + title_suffix, fontsize=14, fontweight="bold")
    plt.savefig(f"{output_dir}/binning_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_dir}/binning_comparison.pdf")

    # Plot 3b: Per-galaxy-type binning comparison plots
    # Only for galaxy types with enough statistics (>= 20 sources for meaningful binning)
    MIN_SOURCES_FOR_TYPE_PLOT = 20
    for gtype, gtype_count in type_counts.items():
        if gtype_count < MIN_SOURCES_FOR_TYPE_PLOT:
            continue

        # Filter catalog for this galaxy type
        type_catalog = sed_catalog_analysis[sed_catalog_analysis["galaxy_type"] == gtype].copy()

        # Compute binning for this type
        type_binned_results = {}
        for name, bin_func in binning_strategies.items():
            try:
                type_binned_results[name] = bin_func(type_catalog)
            except Exception:
                pass  # Skip if binning fails for this type

        if len(type_binned_results) < 2:
            continue  # Need at least 2 strategies to make a comparison plot

        # Create 1x3 grid with residuals plot
        fig_type = plt.figure(figsize=(18, 10), constrained_layout=True)
        gs_type = fig_type.add_gridspec(3, 3, height_ratios=[3, 1, 1], hspace=0.08, wspace=0.15)

        # Dynamic axis limits for this type
        z_min_type = type_catalog["redshift"].min()
        z_max_type = type_catalog["redshift"].max()
        z_padding_type = (z_max_type - z_min_type) * 0.1
        z_lim_type = (max(0, z_min_type - z_padding_type), z_max_type + z_padding_type)

        theta_min_type = type_catalog["r_half_arcsec"].min()
        theta_max_type = type_catalog["r_half_arcsec"].max()
        theta_padding_type = (theta_max_type - theta_min_type) * 0.1
        theta_lim_type = (max(0, theta_min_type - theta_padding_type), theta_max_type + theta_padding_type)

        strategy_names_type = list(type_binned_results.keys())

        for idx, name in enumerate(strategy_names_type[:3]):  # Max 3 strategies for 1x3
            binned_data = type_binned_results[name]
            col = idx  # 1x3 layout: each strategy in its own column

            ax_main = fig_type.add_subplot(gs_type[0, col])
            ax_resid_lcdm = fig_type.add_subplot(gs_type[1, col], sharex=ax_main)
            ax_resid_static = fig_type.add_subplot(gs_type[2, col], sharex=ax_main)

            # Fit models for this binning
            R_static_type = get_radius(
                binned_data.z_mid.values,
                binned_data.theta_med.values,
                binned_data.theta_err.values,
                model="static",
            )
            R_lcdm_type, Omega_m_type = get_radius_and_omega(
                binned_data.z_mid.values,
                binned_data.theta_med.values,
                binned_data.theta_err.values,
            )

            z_model_type = np.linspace(max(0.05, z_lim_type[0]), z_lim_type[1], 200)
            theta_static_type = theta_static(z_model_type, R_static_type) * RAD_TO_ARCSEC
            theta_lcdm_type = theta_lcdm_flat(z_model_type, R_lcdm_type, Omega_m_type) * RAD_TO_ARCSEC

            # Compute chi2/ndf for both models
            def static_func_type(z, R=R_static_type):
                return theta_static(z, R) * RAD_TO_ARCSEC
            def lcdm_func_type(z, R=R_lcdm_type, Om=Omega_m_type):
                return theta_lcdm_flat(z, R, Om) * RAD_TO_ARCSEC

            stats_static_type = compute_chi2_stats(
                binned_data.z_mid.values, binned_data.theta_med.values,
                binned_data.theta_err.values, static_func_type, n_params=1
            )
            stats_lcdm_type = compute_chi2_stats(
                binned_data.z_mid.values, binned_data.theta_med.values,
                binned_data.theta_err.values, lcdm_func_type, n_params=2
            )

            # Plot individual galaxies as background
            ax_main.scatter(
                type_catalog["redshift"],
                type_catalog["r_half_arcsec"],
                c="lightgray",
                alpha=0.4,
                s=15,
                zorder=1,
            )

            # Plot binned data
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
                z_model_type, theta_static_type, "--", color="gray", linewidth=1.5, label=r"Static", zorder=2
            )
            ax_main.plot(
                z_model_type, theta_lcdm_type, "-", color="darkblue", linewidth=1.5, label=r"$\Lambda$CDM", zorder=2
            )

            ax_main.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=11)
            ax_main.set_title(
                f"{name} ({len(binned_data)} bins)\n"
                + r"$\Lambda$CDM: $R$" + f"={R_lcdm_type*1000:.1f} kpc, "
                + r"$\chi^2/{\rm ndf}$" + f"={stats_lcdm_type['chi2_ndf']:.2f}, "
                + f"P={stats_lcdm_type['p_value']:.3f}\n"
                + r"Static: $R$" + f"={R_static_type*1000:.1f} kpc, "
                + r"$\chi^2/{\rm ndf}$" + f"={stats_static_type['chi2_ndf']:.2f}, "
                + f"P={stats_static_type['p_value']:.3f}",
                fontsize=10
            )
            ax_main.legend(fontsize=8, loc="upper right")
            ax_main.grid(True, alpha=0.3)
            ax_main.set_xlim(z_lim_type)
            ax_main.set_ylim(theta_lim_type)
            plt.setp(ax_main.get_xticklabels(), visible=False)

            # Calculate residuals
            theta_lcdm_at_data_type = theta_lcdm_flat(binned_data["z_mid"].values, R_lcdm_type, Omega_m_type) * RAD_TO_ARCSEC
            theta_static_at_data_type = theta_static(binned_data["z_mid"].values, R_static_type) * RAD_TO_ARCSEC
            residuals_lcdm_type = binned_data["theta_med"].values - theta_lcdm_at_data_type
            residuals_static_type = binned_data["theta_med"].values - theta_static_at_data_type

            # ΛCDM residual plot
            ax_resid_lcdm.axhline(0, color="darkblue", linestyle="-", linewidth=1.5, alpha=0.7)
            ax_resid_lcdm.errorbar(
                binned_data["z_mid"],
                residuals_lcdm_type,
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
            ax_resid_lcdm.set_ylabel(r"$\theta - \theta_{\Lambda\mathrm{CDM}}$", fontsize=9)
            ax_resid_lcdm.grid(True, alpha=0.3)
            ax_resid_lcdm.set_xlim(z_lim_type)
            max_resid_lcdm_type = max(0.05, np.max(np.abs(residuals_lcdm_type) + binned_data["theta_err"].values) * 1.2)
            ax_resid_lcdm.set_ylim(-max_resid_lcdm_type, max_resid_lcdm_type)
            plt.setp(ax_resid_lcdm.get_xticklabels(), visible=False)

            # Static residual plot
            ax_resid_static.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
            ax_resid_static.errorbar(
                binned_data["z_mid"],
                residuals_static_type,
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
            ax_resid_static.set_xlabel(r"Redshift $z$", fontsize=11)
            ax_resid_static.set_ylabel(r"$\theta - \theta_{\mathrm{Static}}$", fontsize=9)
            ax_resid_static.grid(True, alpha=0.3)
            ax_resid_static.set_xlim(z_lim_type)
            max_resid_static_type = max(0.05, np.max(np.abs(residuals_static_type) + binned_data["theta_err"].values) * 1.2)
            ax_resid_static.set_ylim(-max_resid_static_type, max_resid_static_type)

        # Create safe filename from galaxy type
        safe_gtype = gtype.replace(" ", "_").replace("/", "-").lower()
        fig_type.suptitle(f"Binning Strategies: {gtype}" + title_suffix + f" (n={gtype_count})", fontsize=14, fontweight="bold")
        plt.savefig(f"{output_dir}/binning_comparison_{safe_gtype}.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {output_dir}/binning_comparison_{safe_gtype}.pdf")

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
        # Prepare catalog for plotting - use FILTERED catalog to match saved CSV
        plot_catalog = sed_catalog_analysis.copy()
        if crop_bounds is not None:
            x_min, x_max, y_min, y_max = crop_bounds
            plot_catalog["xcentroid"] = plot_catalog["xcentroid"] - x_min
            plot_catalog["ycentroid"] = plot_catalog["ycentroid"] - y_min

        # Count galaxies from the filtered catalog (all sources in filtered catalog are galaxies)
        # since FLAG_PSF_LIKE sources are already excluded by quality filtering
        n_galaxies = len(plot_catalog)

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

        # Plot 8: Final selection showing galaxies, detected stars, and mask stars
        # Prepare full catalog for star plotting (need to identify stars from full catalog)
        plot_full_catalog = sed_catalog.copy()
        if crop_bounds is not None:
            x_min, x_max, y_min, y_max = crop_bounds
            plot_full_catalog["xcentroid"] = plot_full_catalog["xcentroid"] - x_min
            plot_full_catalog["ycentroid"] = plot_full_catalog["ycentroid"] - y_min

        # Identify different source types from the FULL catalog
        in_mask = plot_full_catalog["in_star_mask"].fillna(False).astype(bool) if "in_star_mask" in plot_full_catalog.columns else pd.Series([False] * len(plot_full_catalog))
        has_psf_flag = (plot_full_catalog["quality_flag"].astype(int) & FLAG_PSF_LIKE) != 0

        # Mask stars: sources in star mask
        mask_stars = plot_full_catalog[in_mask]
        n_mask_stars = len(mask_stars)

        # Detected stars: sources with PSF-like flag but NOT in mask
        detected_stars = plot_full_catalog[has_psf_flag & ~in_mask]
        n_detected_stars = len(detected_stars)

        _fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        for idx, image in enumerate(plot_images):
            ax = axes[idx // 2, idx % 2]
            vmin, vmax = np.percentile(image.data, [20, 99])
            ax.imshow(image.data, cmap="gray_r", vmin=vmin, vmax=vmax, origin="lower")

            # Plot galaxies (green)
            if n_galaxies > 0:
                ax.scatter(
                    plot_catalog["xcentroid"],
                    plot_catalog["ycentroid"],
                    s=10, facecolors="none", edgecolors="green", linewidth=0.5,
                    label=f"Galaxies ({n_galaxies})"
                )

            # Plot detected stars (blue)
            if n_detected_stars > 0:
                ax.scatter(
                    detected_stars["xcentroid"],
                    detected_stars["ycentroid"],
                    s=15, facecolors="none", edgecolors="blue", linewidth=0.7,
                    label=f"Detected stars ({n_detected_stars})"
                )

            # Plot mask stars (red)
            if n_mask_stars > 0:
                ax.scatter(
                    mask_stars["xcentroid"],
                    mask_stars["ycentroid"],
                    s=20, facecolors="none", edgecolors="red", linewidth=0.9,
                    label=f"Mask stars ({n_mask_stars})"
                )

            ax.set_title(f"Band {image.band.upper()}", fontsize=9)
            ax.set_xlabel(r"X (pixels)")
            ax.set_ylabel(r"Y (pixels)")
            if idx == 0:
                ax.legend(loc="upper right", fontsize=7)

        plt.suptitle(f"Final Selection{title_suffix}\n({n_galaxies} galaxies, {n_detected_stars} detected stars, {n_mask_stars} mask stars)",
                     fontsize=12, fontweight="bold", y=1.02)
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


# =============================================================================
# Professional Source Detection and Photometry Functions
# =============================================================================


def detect_sources_professional(
    data: np.ndarray,
    weight: np.ndarray | None = None,
    gain: float = WFPC2_GAIN,
    box_size: int = 64,
    filter_size: int = 5,
    kernel_fwhm: float = 3.0,
    nsigma: float = 2.0,
    npixels: int = 10,
    nlevels: int = 32,
    contrast: float = 0.001,
    connectivity: int = 8,
    mask: np.ndarray | None = None,
) -> tuple[SourceCatalog | None, object | None, Background2D]:
    """
    Professional source detection using photutils best practices.

    This function implements a complete source detection pipeline following
    photutils documentation and professional astronomy standards:

    1. 2D background estimation using Background2D with MedianBackground
    2. Matched-filter detection using a 2D Gaussian kernel
    3. Source detection using detect_sources with sigma-clipped threshold
    4. Deblending overlapping sources with deblend_sources
    5. Proper error estimation using calc_total_error (includes Poisson noise)
    6. Source catalog extraction with SourceCatalog

    Parameters
    ----------
    data : np.ndarray
        2D image data array (science image in DN or electrons)
    weight : np.ndarray or None, optional
        Inverse variance weight map. If provided, used to compute error map.
        If None, error is estimated from background RMS and Poisson noise.
    gain : float, optional
        Detector gain in electrons/DN (default: WFPC2_GAIN = 7.0).
        Used for Poisson noise calculation in calc_total_error.
    box_size : int, optional
        Size of the boxes for background estimation (default: 50).
        Larger values give smoother background but may miss local variations.
    filter_size : int, optional
        Size of the median filter for smoothing the background (default: 3).
    kernel_fwhm : float, optional
        FWHM of the 2D Gaussian detection kernel in pixels (default: 3.0).
        Should match or slightly exceed the PSF FWHM for optimal detection.
    nsigma : float, optional
        Detection threshold in units of background RMS (default: 2.0).
        Lower values detect more sources but increase false positives.
    npixels : int, optional
        Minimum number of connected pixels above threshold (default: 10).
        Larger values reject cosmic rays and noise spikes.
    nlevels : int, optional
        Number of multi-thresholding levels for deblending (default: 32).
    contrast : float, optional
        Contrast parameter for deblending (default: 0.001).
        Lower values split sources more aggressively.
    connectivity : int, optional
        Pixel connectivity (4 or 8) for source identification (default: 8).
    mask : np.ndarray or None, optional
        Boolean mask where True values are excluded from detection.

    Returns
    -------
    catalog : SourceCatalog or None
        Source catalog with photometry and morphology. None if no sources found.
    segm_deblend : SegmentationImage or None
        Deblended segmentation map. None if no sources found.
    bkg : Background2D
        2D background model.

    Examples
    --------
    >>> data, header, weight = read_fits('image.fits', 'weight.fits')
    >>> catalog, segm, bkg = detect_sources_professional(data, weight=weight)
    >>> print(f"Detected {len(catalog)} sources")
    >>> # Access source properties
    >>> fluxes = catalog.segment_flux
    >>> positions = catalog.centroid

    Notes
    -----
    The error map is computed using photutils.utils.calc_total_error which
    properly combines background error and Poisson noise from source counts:

        total_error = sqrt(bkg_error**2 + data/gain)

    This is the standard approach used in professional photometry pipelines
    (DAOPHOT, SExtractor, etc.).

    References
    ----------
    - photutils documentation: https://photutils.readthedocs.io/
    - DAOPHOT II: Stetson 1987, PASP, 99, 191
    - SExtractor: Bertin & Arnouts 1996, A&AS, 117, 393
    """
    # Step 1: Estimate 2D background with MedianBackground
    # MedianBackground is robust to source contamination
    bkg_estimator = MedianBackground()
    bkg = Background2D(
        data,
        (box_size, box_size),
        filter_size=(filter_size, filter_size),
        bkg_estimator=bkg_estimator,
        exclude_percentile=10.0,  # Exclude bright pixels from background
    )

    # Step 2: Subtract background
    data_sub = data - bkg.background

    # Step 3: Create matched-filter detection kernel
    # The kernel should approximate the PSF for optimal SNR
    kernel = make_2dgaussian_kernel(kernel_fwhm, size=int(4 * kernel_fwhm) | 1)  # Ensure odd size

    # Step 4: Convolve with detection kernel (matched filter)
    convolved_data = convolve(data_sub, kernel)

    # Step 5: Calculate proper error map using calc_total_error
    # This combines background RMS and Poisson noise from source counts
    if weight is not None:
        # Use weight map to get background error
        # Weight = inverse variance, so error = 1/sqrt(weight)
        with np.errstate(divide='ignore', invalid='ignore'):
            bkg_error = np.where(weight > 0, 1.0 / np.sqrt(weight), bkg.background_rms)
    else:
        # Use background RMS as error estimate
        bkg_error = bkg.background_rms

    # calc_total_error adds Poisson noise from source counts
    # This is the professional standard for photometric errors
    # Note: data must be in electrons or DN with appropriate gain
    effective_gain = gain if gain > 0 else 1.0
    total_error = calc_total_error(
        data_sub,
        bkg_error,
        effective_gain=effective_gain
    )

    # Step 6: Calculate detection threshold
    threshold = nsigma * bkg.background_rms

    # Step 7: Detect sources using segmentation
    segm = detect_sources(
        convolved_data,
        threshold,
        npixels=npixels,
        connectivity=connectivity,
        mask=mask,
    )

    if segm is None:
        print("  No sources detected!")
        return None, None, bkg

    # Step 8: Deblend overlapping sources
    segm_deblend = deblend_sources(
        convolved_data,
        segm,
        npixels=npixels,
        nlevels=nlevels,
        contrast=contrast,
        progress_bar=False,
    )

    # Step 9: Create source catalog with error map
    catalog = SourceCatalog(
        data_sub,
        segm_deblend,
        convolved_data=convolved_data,
        error=total_error,
    )

    return catalog, segm_deblend, bkg


def aperture_photometry_with_local_background(
    data: np.ndarray,
    positions: np.ndarray | list,
    aperture_radius: float = 5.0,
    annulus_r_in: float = 10.0,
    annulus_r_out: float = 15.0,
    error: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    method: str = 'exact',
) -> dict:
    """
    Perform aperture photometry with local background estimation.

    This function implements professional aperture photometry following
    photutils best practices:

    1. CircularAperture for source flux measurement
    2. CircularAnnulus for local background estimation
    3. ApertureStats for robust background statistics (median, sigma-clipping)
    4. Proper error propagation including background subtraction uncertainty

    Parameters
    ----------
    data : np.ndarray
        2D image data array (background-subtracted or raw).
    positions : np.ndarray or list
        Source positions as Nx2 array of (x, y) coordinates or list of tuples.
    aperture_radius : float, optional
        Radius of the circular source aperture in pixels (default: 5.0).
        Should be 2-3x the PSF FWHM for point sources.
    annulus_r_in : float, optional
        Inner radius of the background annulus in pixels (default: 10.0).
        Should be far enough from source to avoid contamination.
    annulus_r_out : float, optional
        Outer radius of the background annulus in pixels (default: 15.0).
        Larger annuli give more robust background but may include neighbors.
    error : np.ndarray or None, optional
        Error map (1-sigma uncertainties per pixel). If provided, used
        for error propagation in aperture sums.
    mask : np.ndarray or None, optional
        Boolean mask where True values are excluded from photometry.
    method : str, optional
        Aperture photometry method: 'exact', 'center', or 'subpixel'.
        'exact' (default) gives accurate fractional pixel contributions.

    Returns
    -------
    dict
        Dictionary containing:
        - 'flux': Background-subtracted aperture flux (array)
        - 'flux_error': Total flux uncertainty (array)
        - 'bkg_median': Local background median per pixel (array)
        - 'bkg_std': Local background standard deviation (array)
        - 'bkg_total': Total background subtracted from aperture (array)
        - 'aperture_area': Area of source aperture in pixels (float)
        - 'annulus_area': Area of background annulus in pixels (float)
        - 'raw_flux': Raw aperture flux before background subtraction (array)

    Examples
    --------
    >>> # Detect sources first
    >>> catalog, segm, bkg = detect_sources_professional(data)
    >>> positions = np.column_stack([catalog.xcentroid, catalog.ycentroid])
    >>>
    >>> # Perform aperture photometry
    >>> phot = aperture_photometry_with_local_background(
    ...     data - bkg.background,
    ...     positions,
    ...     aperture_radius=3.0,  # ~1.5x PSF FWHM
    ...     annulus_r_in=6.0,
    ...     annulus_r_out=10.0,
    ...     error=total_error,
    ... )
    >>> print(f"Flux: {phot['flux'][0]:.2f} +/- {phot['flux_error'][0]:.2f}")

    Notes
    -----
    The local background is estimated using the median of pixels in the
    annulus, which is robust to outliers (cosmic rays, neighboring sources).
    ApertureStats provides sigma-clipped statistics for even better robustness.

    Error propagation follows the standard formula:
        flux_error = sqrt(aperture_flux_err^2 + (area * bkg_std)^2 / n_bkg)

    where n_bkg is the number of pixels in the background annulus.

    References
    ----------
    - photutils aperture photometry: https://photutils.readthedocs.io/en/stable/aperture.html
    - DAOPHOT aperture photometry: Stetson 1987, PASP, 99, 191
    """
    # Convert positions to numpy array
    positions = np.atleast_2d(positions)
    if positions.shape[1] != 2:
        raise ValueError("Positions must be Nx2 array of (x, y) coordinates")

    n_sources = len(positions)

    # Create apertures
    source_apertures = CircularAperture(positions, r=aperture_radius)
    background_annuli = CircularAnnulus(
        positions,
        r_in=annulus_r_in,
        r_out=annulus_r_out,
    )

    # Calculate aperture areas
    aperture_area = source_apertures.area  # pi * r^2
    annulus_area = background_annuli.area  # pi * (r_out^2 - r_in^2)

    # Perform raw aperture photometry
    phot_table = aperture_photometry(
        data,
        source_apertures,
        error=error,
        mask=mask,
        method=method,
    )

    raw_flux = np.array(phot_table['aperture_sum'])

    # Get raw flux errors if error map provided
    if error is not None and 'aperture_sum_err' in phot_table.colnames:
        raw_flux_err = np.array(phot_table['aperture_sum_err'])
    else:
        raw_flux_err = np.zeros(n_sources)

    # Estimate local background using ApertureStats
    # ApertureStats provides robust statistics with sigma-clipping
    bkg_stats = ApertureStats(data, background_annuli, sigma_clip=None)

    # Use median for robust background estimate
    bkg_median = np.array(bkg_stats.median)
    bkg_std = np.array(bkg_stats.std)

    # Handle NaN values (sources near edge or in masked regions)
    bkg_median = np.nan_to_num(bkg_median, nan=0.0)
    bkg_std = np.nan_to_num(bkg_std, nan=0.0)

    # Calculate total background to subtract
    bkg_total = bkg_median * aperture_area

    # Background-subtracted flux
    flux = raw_flux - bkg_total

    # Error propagation
    # Total error includes:
    # 1. Aperture flux error (from error map, includes Poisson + read noise)
    # 2. Background estimation uncertainty: area * bkg_std / sqrt(n_bkg_pixels)
    n_bkg_pixels = annulus_area  # Approximate, actual may differ due to masking
    bkg_error = aperture_area * bkg_std / np.sqrt(np.maximum(n_bkg_pixels, 1.0))

    flux_error = np.sqrt(raw_flux_err**2 + bkg_error**2)

    return {
        'flux': flux,
        'flux_error': flux_error,
        'bkg_median': bkg_median,
        'bkg_std': bkg_std,
        'bkg_total': bkg_total,
        'aperture_area': aperture_area,
        'annulus_area': annulus_area,
        'raw_flux': raw_flux,
    }


def main(
    mode: str = "full",
    output_subdir: str | None = None,
    use_additional_star_detection: bool = True,
    detection_backend: str = "photutils",
):
    """
    Run the analysis pipeline.

    Args:
        mode: "full" for entire 4096x4096 image, "chip3" for 2048x2048 chip3 only
        output_subdir: Optional subdirectory within output/{mode}_HDF/ (e.g., "full_z")
        use_additional_star_detection: If True, use stellarity/morphology criteria in addition
            to the star mask. If False, only use the star mask for star identification.
        detection_backend: "photutils" (default) or "sep" for SExtractor-based detection.
    """
    # Set output directory based on mode and optional subdirectory
    # Mode naming: full -> full_HDF, chip3 -> chip3_HDF
    mode_dir = f"{mode}_HDF"
    output_dir = f"./output/{mode_dir}/{output_subdir}" if output_subdir else f"./output/{mode_dir}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("ANGULAR SIZE TEST ANALYSIS")
    print(f"Mode: {mode}")
    if use_additional_star_detection:
        star_detection_mode = "mask + detection everywhere"
    else:
        star_detection_mode = "mask only in chip3, detection outside"
    print(f"Star detection: {star_detection_mode}")
    if output_subdir:
        print(f"Output: {output_dir}")
    print("=" * 60)

    # Step 1: Load FITS files
    print("\n[1/6] Loading FITS data...")

    # Configuration based on mode
    USE_FULL_RESOLUTION = (mode == "full")

    if USE_FULL_RESOLUTION:
        # Official STScI HDF v2 mosaics (4096x4096) with context/weight maps
        # Filter mapping: F300W=UV, F450W=B, F606W=V, F814W=I
        files = {
            "f300": {"science": "./data/hdf_north/mosaics/f300_mosaic_v2.fits", "weight": "./data/hdf_north/mosaics/f300c_mosaic_v2.fits"},
            "f450": {"science": "./data/hdf_north/mosaics/f450_mosaic_v2.fits", "weight": "./data/hdf_north/mosaics/f450c_mosaic_v2.fits"},
            "f606": {"science": "./data/hdf_north/mosaics/f606_mosaic_v2.fits", "weight": "./data/hdf_north/mosaics/f606c_mosaic_v2.fits"},
            "f814": {"science": "./data/hdf_north/mosaics/f814_mosaic_v2.fits", "weight": "./data/hdf_north/mosaics/f814c_mosaic_v2.fits"},
        }
        print("  Using official STScI HDF v2 4096x4096 mosaics with weight maps")
    else:
        # Official STScI HDF v2 chip3 drizzled images (2048x2048) with context maps
        files = {
            "f300": {"science": "./data/hdf_north/chips/f300_3_v2.fits", "weight": "./data/hdf_north/chips/f300_3c_v2.fits"},
            "f450": {"science": "./data/hdf_north/chips/f450_3_v2.fits", "weight": "./data/hdf_north/chips/f450_3c_v2.fits"},
            "f606": {"science": "./data/hdf_north/chips/f606_3_v2.fits", "weight": "./data/hdf_north/chips/f606_3c_v2.fits"},
            "f814": {"science": "./data/hdf_north/chips/f814_3_v2.fits", "weight": "./data/hdf_north/chips/f814_3c_v2.fits"},
        }
        print("  Using official STScI HDF v2 chip3 2048x2048 drizzled data")

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
    # Star masks for flagging bright stars (6 stars, same region in both)
    star_mask = None
    if image_shape == (4096, 4096):
        star_mask_path = "./data/hdf_north/star_mask_mosaic.fits"
    else:
        star_mask_path = "./data/hdf_north/star_mask_chip3.fits"
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

    # Create mask for zero-coverage regions in drizzled mosaic
    # These are regions where no input images contributed (exactly 0 in all bands)
    # Mask pixels that are zero in ALL bands (no coverage)
    zero_coverage_mask = np.ones(image_shape, dtype=bool)
    for img in images:
        zero_coverage_mask &= (img.data == 0)
    n_masked = zero_coverage_mask.sum()
    print(f"  Zero-coverage mask: {n_masked} pixels ({100*n_masked/zero_coverage_mask.size:.1f}% of image)")
    mask = zero_coverage_mask

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
        # Use sqrt scaling - linear scaling (2x for 4096) was too aggressive
        # and caused zero detections in low-SNR bands like UV (f300)
        # sqrt scaling: 1.0 for 2048, 1.41 for 4096
        scale = np.sqrt(image_shape[0] / 2048.0)
        return int(base_npixels * scale)

    # Detection threshold: sigma above background RMS
    # Higher threshold on low-resource machines detects fewer sources (faster)
    DETECTION_SIGMA = _RESOURCE_CONFIG.detection_sigma

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
    def process_band_photutils(image):
        """Process a single band for source detection using photutils."""
        import gc as gc_local

        bkg_estimator = MedianBackground()
        # Larger box/filter sizes for faster background estimation (2-3x speedup)
        box_size = 128 if image.data.shape[0] > 2048 else 64
        bkg = Background2D(image.data, (box_size, box_size), filter_size=(5, 5), bkg_estimator=bkg_estimator)

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
            masked_labels = np.unique(np.asarray(image.segm)[mask])
            masked_labels = masked_labels[masked_labels != 0]
            if len(masked_labels) > 0:
                image.segm.remove_labels(masked_labels)

        # Create catalog - needs data_sub and convolved_data
        image.catalog = SourceCatalog(data_sub, image.segm, convolved_data=convolved_data)

        # Release intermediate arrays we no longer need
        del data_sub, convolved_data
        gc_local.collect()

        return image.band, len(image.catalog), npixels

    # Reference band for statmorph computation (best S/N band)
    STATMORPH_BAND = "f450"  # B band - deepest for morphology

    def process_band_sep(image):
        """Process a single band for source detection using SEP (SExtractor)."""
        import gc as gc_local
        from detection import detect_sources_sep, kron_photometry, compute_half_light_radius

        box_size = 128 if image.data.shape[0] > 2048 else 64
        npixels = get_adaptive_npixels(image.data.shape, BASE_NPIXELS)

        # Run SEP detection
        result = detect_sources_sep(
            image.data,
            threshold=DETECTION_SIGMA,
            min_area=npixels,
            mask=mask.astype(np.uint8) if mask is not None else None,
            gain=WFPC2_GAIN,
            box_size=box_size,
            filter_size=5,
            deblend_nthresh=NLEVELS,
            deblend_cont=CONTRAST,
            segmentation_map=True,
        )

        if result.n_sources == 0:
            # Add expected columns to empty catalog for compatibility
            empty_catalog = result.catalog.copy()
            for col in ['flux_auto', 'flux_auto_err', 'segment_flux', 'segment_fluxerr', 'r_half']:
                if col not in empty_catalog.columns:
                    empty_catalog[col] = pd.Series(dtype=np.float64)
            image.bkg = SEPBackgroundWrapper(result.background)
            image.segm = result.segmentation
            image.catalog = SEPCatalogWrapper(empty_catalog)
            return image.band, 0, npixels

        # Get background-subtracted data for photometry
        data_sub = image.data - result.background.background

        # Compute Kron photometry (AUTO magnitudes)
        catalog = kron_photometry(
            data_sub,
            result.catalog,
            result.background.background_rms,
            gain=WFPC2_GAIN,
            mask=mask.astype(np.uint8) if mask is not None else None,
        )

        # Compute r_half and r_90 together using vectorized function
        # This computes both radii efficiently by reusing the flux grid
        from detection.sep_detection import compute_flux_radii
        flux_radii = compute_flux_radii(
            data_sub,
            catalog,
            result.background.background_rms,
            fractions=(0.5, 0.9),
            mask=mask.astype(np.uint8) if mask is not None else None,
        )
        catalog['r_half'] = flux_radii[0.5]
        r_90 = flux_radii[0.9]

        # Compute statmorph morphology for reference band only
        # This provides gini, m20, sersic_n for star/galaxy classification
        if image.band == STATMORPH_BAND and result.segmentation is not None:
            from detection import compute_statmorph, STATMORPH_AVAILABLE
            if STATMORPH_AVAILABLE:
                print(f"  Computing statmorph morphology for {image.band} band...")
                # Use gain=1.0 for drizzled e-/s data (WFPC2_GAIN is None for drizzled)
                # Statmorph needs gain for noise estimation; 1.0 is reasonable for e-/s units
                statmorph_gain = WFPC2_GAIN if WFPC2_GAIN is not None else 1.0
                catalog = compute_statmorph(
                    data_sub,
                    result.segmentation,
                    catalog,
                    gain=statmorph_gain,
                    psf_fwhm=PSF_FWHM_PIX,
                )

        # Add columns expected by rest of pipeline
        catalog['segment_flux'] = catalog['flux_auto']
        catalog['segment_fluxerr'] = catalog['flux_auto_err']

        # Wrap in compatible interface
        image.bkg = SEPBackgroundWrapper(result.background)
        image.segm = result.segmentation
        image.catalog = SEPCatalogWrapper(
            catalog,
            segmentation=result.segmentation,
            r_half=catalog['r_half'].values,
            r_90=r_90,
        )

        del data_sub
        gc_local.collect()

        return image.band, len(image.catalog), npixels

    # Choose detection backend
    if detection_backend == "sep" and SEP_AVAILABLE:
        process_band = process_band_sep
        print(f"  Using SEP (SExtractor) detection backend")
    else:
        process_band = process_band_photutils
        print(f"  Using photutils detection backend")

    # DEBUG: Print data statistics to diagnose detection issues
    print("  DEBUG: Data statistics before detection:")
    for img in images:
        data = img.data
        finite_data = data[np.isfinite(data)]
        n_finite = len(finite_data)
        n_positive = (finite_data > 0).sum()
        n_zero = (data == 0).sum()
        print(f"    {img.band}: shape={data.shape}, finite={n_finite}/{data.size} ({100*n_finite/data.size:.1f}%), "
              f">0={n_positive}, zero={n_zero}, min={np.nanmin(data):.2f}, max={np.nanmax(data):.2f}, "
              f"median={np.nanmedian(data):.4f}, std={np.nanstd(data):.4f}")

    # Check if parallel detection is enabled (based on resource profile)
    if _RESOURCE_CONFIG.parallel_detection and len(images) > 1:
        # Parallel band detection using ThreadPoolExecutor
        # ThreadPool is preferred over ProcessPool here because:
        # 1. numpy/sep release the GIL during computation
        # 2. Avoids serialization overhead for large image arrays
        # 3. Shared memory access to image data
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n_workers = min(len(images), _RESOURCE_CONFIG.n_threads)
        print(f"  Processing {len(images)} bands in parallel ({n_workers} workers)...")

        results = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_img = {executor.submit(process_band, img): img for img in images}
            for future in tqdm(as_completed(future_to_img), total=len(images),
                               desc="  Detecting sources", unit="band"):
                img = future_to_img[future]
                try:
                    band, n_sources, npixels = future.result()
                    results[band] = (n_sources, npixels)
                    tqdm.write(f"  Band {band}: {n_sources} sources (npixels={npixels})")
                except Exception as e:
                    tqdm.write(f"  Band {img.band}: FAILED - {e}")
                    raise
    else:
        # Sequential processing to minimize peak memory usage
        print(f"  Processing {len(images)} bands sequentially (memory-optimized)...")
        for img in tqdm(images, desc="  Detecting sources", unit="band"):
            band, n_sources, npixels = process_band(img)
            tqdm.write(f"  Band {band}: {n_sources} sources (npixels={npixels})")

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
    for image in tqdm(images, desc="  Processing bands", unit="band"):
        tqdm.write(f"    Processing {image.band} band catalog...")

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

        # Calculate photometric errors using professional DAOPHOT-style formula
        # Reference: DAOPHOT II (Stetson 1987), DrizzlePac photometry_tools
        # Formula: err = sqrt(poisson_var + sky_var + sky_uncertainty_var) × corr_factor
        #
        # Components:
        # 1. Poisson noise: source_flux / gain (shot noise from source photons)
        # 2. Sky variance: area × σ_sky² (background noise in aperture)
        # 3. Sky uncertainty: area² × σ_sky² / nsky (uncertainty in background estimate)
        # 4. Drizzle correlation: multiply by DRIZZLE_NOISE_CORR (~1.4 for HDF)
        #
        # The weight map provides per-pixel inverse variance from drizzle combination

        labels = image.catalog.labels
        segm_data = np.asarray(image.segm)
        bkg_rms = image.bkg.background_rms_median

        # Count pixels per segment using bincount
        label_counts = np.bincount(segm_data.ravel())
        segment_areas = label_counts[labels]

        if image.weight is not None:
            # Get variance map from weight (weight = inverse variance)
            with np.errstate(divide="ignore", invalid="ignore"):
                variance_map = np.where(image.weight > 0, 1.0 / image.weight, np.inf)

            # Sum variance over all segments (this is the drizzle-combined variance)
            # This already includes readnoise and dark current from pipeline
            drizzle_variances = ndsum(variance_map, labels=segm_data, index=labels)

            # Free variance_map immediately - this is ~128MB
            del variance_map
            gc.collect()

            # Component 1: Poisson noise from source (if not fully captured in weight map)
            # HDF weight maps are inverse variance, but we add explicit Poisson term
            # to ensure shot noise is properly accounted for
            source_flux = np.abs(cat["segment_flux"].to_numpy())
            # Skip Poisson term for drizzled data (WFPC2_GAIN=None) - already in e-/s units
            poisson_var = source_flux / WFPC2_GAIN if WFPC2_GAIN is not None else 0.0

            # Component 2: Sky variance in aperture
            sky_var = segment_areas * bkg_rms**2

            # Component 3: Uncertainty in background estimation
            # nsky ≈ SKY_ANNULUS_FACTOR × source_area (typical annulus is ~10× source)
            nsky = segment_areas * SKY_ANNULUS_FACTOR
            sky_uncertainty_var = (segment_areas**2 * bkg_rms**2) / np.maximum(nsky, 1.0)

            # Total variance: drizzle + poisson + sky + sky_uncertainty
            # Note: drizzle_variances may already include some of these, but
            # weight maps often underestimate due to correlated noise
            total_variances = drizzle_variances + poisson_var + sky_var + sky_uncertainty_var

            # Apply drizzle correlated noise correction factor
            # Drizzle creates pixel-to-pixel correlations that cause summed variance
            # to underestimate true aperture noise by factor of ~1.2-1.5
            cat["source_error"] = (np.sqrt(total_variances) * DRIZZLE_NOISE_CORR).astype(np.float32)
        else:
            # Fallback without weight map: full DAOPHOT formula
            source_flux = np.abs(cat["segment_flux"].to_numpy())
            segment_areas = cat["area"].to_numpy()

            # Skip Poisson term for drizzled data (WFPC2_GAIN=None) - already in e-/s units
            poisson_var = source_flux / WFPC2_GAIN if WFPC2_GAIN is not None else 0.0
            sky_var = segment_areas * bkg_rms**2
            nsky = segment_areas * SKY_ANNULUS_FACTOR
            sky_uncertainty_var = (segment_areas**2 * bkg_rms**2) / np.maximum(nsky, 1.0)

            total_variances = poisson_var + sky_var + sky_uncertainty_var
            cat["source_error"] = (np.sqrt(total_variances) * DRIZZLE_NOISE_CORR).astype(np.float32)

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
        # Include statmorph columns if they exist
        base_cols = [
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
        # Add statmorph columns if present (only for reference band)
        statmorph_cols = ['gini', 'm20', 'sersic_n', 'asymmetry', 'smoothness', 'statmorph_flag']
        cols_to_store = base_cols + [c for c in statmorph_cols if c in cat.columns]
        band_catalogs[image.band] = cat[cols_to_store].copy()

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

    reference_band = "f450"  # B band (blue)
    matched_sources = []
    match_radius = 3.0

    # Build KD-trees for fast spatial matching (O(N log N) instead of O(N²))
    other_bands = ["f814", "f300", "f606"]  # I, U, V bands
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
    # 3 bands (B, V, I) sufficient for basic photo-z; UV (f300) often too shallow
    # Sources with all 4 bands will have more reliable photo-z estimates
    MIN_BANDS_REQUIRED = 3  # Accept 3+ bands; flag 4-band sources as higher quality

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

    # Statmorph columns from reference band (may not exist if statmorph unavailable)
    statmorph_cols = ['gini', 'm20', 'sersic_n', 'asymmetry', 'smoothness', 'statmorph_flag']
    ref_statmorph = {}
    for col in statmorph_cols:
        if col in ref_cat.columns:
            ref_statmorph[col] = ref_cat[col].values
        else:
            ref_statmorph[col] = None

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

            # Add statmorph morphology from reference band (if available)
            for col in statmorph_cols:
                if ref_statmorph[col] is not None:
                    matches[col] = ref_statmorph[col][idx]
                else:
                    matches[col] = np.nan

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

            # Check for very low SNR (median < 0.5) - unreliable photometry
            # Sources with SNR < 0.5 have 50% catastrophic photo-z rate
            if matches["snr_median"] < 0.5:
                flag |= FLAG_VERY_LOW_SNR

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

            # Additional star detection via morphology/stellarity
            if use_additional_star_detection:
                # PSF half-light radius (Gaussian approximation: r_half ≈ 0.42 * FWHM)
                psf_r_half_pix = PSF_FWHM_PIX * 0.42  # ~0.95 pixels for HST ACS

                # Size ratio: how many times larger than PSF
                size_ratio = matches["r_half_pix"] / psf_r_half_pix if psf_r_half_pix > 0 else 999

                # Criterion 1: High stellarity (star classification)
                # Our stellarity metric ranges 0-0.5 for this data, so use 0.35 threshold
                # (SExtractor CLASS_STAR uses 0.8-0.95, but our metric is different)
                if matches["stellarity"] > 0.35:
                    is_star = True

                # Criterion 2: Compact + round + concentrated (all three required)
                # This catches stars that have moderate stellarity but clear point-source morphology
                elif size_ratio < 2.5:  # Compact (within 2.5x PSF size, relaxed from 1.8)
                    is_round = matches["ellipticity"] < 0.25  # Stars are fairly round
                    is_concentrated = matches["concentration"] > 0.40  # Moderate light concentration
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

    # Guard against empty cross-match results
    if len(cross_matched_catalog) == 0:
        print(f"  WARNING: No sources matched across ≥{MIN_BANDS_REQUIRED} bands!")
        print("  Detection counts per band:")
        for band, cat in band_catalogs.items():
            print(f"    {band}: {len(cat)} sources")
        raise ValueError(
            f"Cross-matching produced 0 sources. This typically means:\n"
            f"  1. Too few detections in one or more bands (especially UV/f300)\n"
            f"  2. MIN_BANDS_REQUIRED={MIN_BANDS_REQUIRED} is too strict\n"
            f"  3. Detection parameters (npixels, threshold) are too conservative\n"
            f"Try reducing MIN_BANDS_REQUIRED to 3, or check detection parameters."
        )

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
    # Debug: show detected stars breakdown
    n_detected_stars = n_psf_like - n_in_star_mask
    print(f"  Detected stars (morphology, not in mask): {n_detected_stars}")

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
        "f450": 8.8e-18,   # B band (blue)
        "f814": 2.45e-18,  # I band (infrared)
        "f300": 5.99e-17,  # U band (ultraviolet)
        "f606": 1.89e-18,  # V band (visual)
    }

    # Build SED catalog using vectorized operations (much faster than iterrows)
    # Include statmorph columns if available
    base_cols = [
        "r_half_pix", "r_half_pix_error", "r_half_pix_corrected",
        "xcentroid", "ycentroid", "concentration", "ellipticity", "stellarity",
        "snr_median", "snr_min", "quality_flag", "in_star_mask"
    ]
    # Add statmorph columns if they exist in cross_matched_catalog
    statmorph_cols_to_add = ['gini', 'm20', 'sersic_n', 'asymmetry', 'smoothness', 'statmorph_flag']
    for col in statmorph_cols_to_add:
        if col in cross_matched_catalog.columns:
            base_cols.append(col)

    sed_catalog = cross_matched_catalog[base_cols].copy()

    # Add half_light_radius column for star classifier (alias for r_half_pix)
    sed_catalog["half_light_radius"] = sed_catalog["r_half_pix"]

    # Vectorized flux conversion for all bands
    for band in ["f450", "f814", "f300", "f606"]:
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

    # ==========================================================================
    # ZEROPOINT CALIBRATION (using spectroscopic redshifts)
    # This corrects systematic offsets in the photometry before photo-z fitting
    # NOTE: Testing showed offsets are <0.05 mag (well-calibrated), so this is
    # disabled by default. The poor photo-z quality is due to template/degeneracy
    # issues with only 4 bands, not zeropoint calibration.
    # ==========================================================================
    USE_ZEROPOINT_CALIBRATION = True  # Enable zeropoint calibration

    if USE_ZEROPOINT_CALIBRATION:
        try:
            from validation.specz_validation import load_specz_catalogs, cross_match_with_specz
            from validation.zeropoint_calibration import (
                derive_zeropoint_offsets,
                apply_zeropoint_corrections,
                save_zeropoint_file,
            )

            print("\n  === Photometric Zeropoint Calibration ===")

            # Load spectroscopic catalogs for calibration
            specz_catalogs_cal = load_specz_catalogs()

            if specz_catalogs_cal:
                # Cross-match to get spec-z for calibration
                sed_catalog = cross_match_with_specz(sed_catalog, specz_catalogs_cal, match_radius=1.0)

                # Check if we have enough calibrators
                n_specz = sed_catalog["z_spec"].notna().sum()
                print(f"  Found {n_specz} spec-z matches for zeropoint calibration")

                if n_specz >= 20:  # Need minimum calibrators
                    # Derive zeropoint offsets
                    flux_cols = ("flux_f300", "flux_f450", "flux_f606", "flux_f814")
                    error_cols = ("error_f300", "error_f450", "error_f606", "error_f814")

                    offsets = derive_zeropoint_offsets(
                        sed_catalog,
                        flux_columns=flux_cols,
                        z_spec_column="z_spec",
                        template_path="./spectra",
                        min_calibrators=10,
                    )

                    if offsets.converged:
                        # Apply corrections to ALL sources
                        print("  Applying zeropoint corrections to photometry...")
                        sed_catalog = apply_zeropoint_corrections(
                            sed_catalog, offsets, flux_cols, error_cols
                        )

                        # Save offsets for reference
                        save_zeropoint_file(offsets, f"{output_dir}/zeropoint_offsets.txt",
                                          comment=f"Derived from {n_specz} spec-z calibrators")
                    else:
                        print("  Zeropoint calibration did not converge, using uncorrected photometry")
                else:
                    print(f"  Insufficient spec-z calibrators ({n_specz} < 20), skipping zeropoint calibration")
            else:
                print("  No spectroscopic catalogs available for calibration")

        except ImportError as e:
            print(f"  Zeropoint calibration skipped (missing module): {e}")
        except Exception as e:
            print(f"  Zeropoint calibration failed: {e}")
            import traceback
            traceback.print_exc()

    # Force garbage collection before heavy computation
    gc.collect()

    # Classify galaxies using template fitting with parallel processing
    # Uses ProcessPoolExecutor for multi-core parallelism
    # Note: Speed is ~4-10 galaxies/sec depending on z_step and hardware
    n_galaxies = len(sed_catalog)
    print(f"  Classifying {n_galaxies} galaxies with template fitting...")

    # Build flux and error arrays in [U, B, V, I] order for vectorized classifier
    flux_array = np.column_stack([
        sed_catalog["flux_f300"].to_numpy(),  # U band
        sed_catalog["flux_f450"].to_numpy(),  # B band
        sed_catalog["flux_f606"].to_numpy(),  # V band
        sed_catalog["flux_f814"].to_numpy(),  # I band
    ])
    error_array = np.column_stack([
        sed_catalog["error_f300"].to_numpy(),  # U band
        sed_catalog["error_f450"].to_numpy(),  # B band
        sed_catalog["error_f606"].to_numpy(),  # V band
        sed_catalog["error_f814"].to_numpy(),  # I band
    ])

    # Template fitting with configurable redshift step
    # z_step from resource config: larger steps for faster processing on constrained machines
    # z_min=0.05 prevents boundary pileup at z=0 from template fitting failures
    import time
    t0 = time.perf_counter()
    results = classify_batch_ultrafast(
        flux_array, error_array,
        spectra_path="./spectra",
        z_step=_RESOURCE_CONFIG.z_step,
        z_step_coarse=_RESOURCE_CONFIG.z_step_coarse,
        z_min=Z_MIN,
    )
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
    sed_catalog["best_A_V"] = results['best_A_V']  # Dust extinction (Calzetti law)

    # Add boundary flag (sources at z_min or z_max - fitting likely failed)
    if 'z_boundary_flag' in results:
        sed_catalog["z_boundary_flag"] = results['z_boundary_flag']
    else:
        # Fallback: compute from redshift values (use Z_MIN + tolerance)
        sed_catalog["z_boundary_flag"] = (results['redshift'] <= Z_MIN + 0.01) | (results['redshift'] >= Z_MAX - 0.01)

    # Track number of valid bands used in photo-z fitting
    if 'n_valid_bands' in results:
        sed_catalog["n_valid_bands"] = results['n_valid_bands']
    else:
        # Fallback: count non-NaN flux values
        flux_cols = [f"flux_{b}" for b in ["f300", "f450", "f606", "f814"]]
        sed_catalog["n_valid_bands"] = sed_catalog[flux_cols].notna().sum(axis=1)

    # ==========================================================================
    # PHOTO-Z RELIABILITY FLAG BASED ON BAND COUNT
    # Sources with <4 bands (especially missing U-band/F300W) suffer from severe
    # color-redshift degeneracy. Only 4-band sources have reliable photo-z.
    # See: Fernandez-Soto et al. 1999, Benitez 2000
    # ==========================================================================
    n_4band = (sed_catalog["n_valid_bands"] >= 4).sum()
    n_3band = (sed_catalog["n_valid_bands"] == 3).sum()
    n_fewer = (sed_catalog["n_valid_bands"] < 3).sum()

    # Flag for photo-z reliability: True = reliable (4+ bands)
    sed_catalog["photoz_reliable"] = sed_catalog["n_valid_bands"] >= 4

    # Check U-band specifically (critical for breaking degeneracy)
    has_uband = sed_catalog["flux_f300"].notna() & (sed_catalog["flux_f300"] > 0)
    sed_catalog["has_uband"] = has_uband

    print(f"  Photo-z band coverage:")
    print(f"    4 bands (reliable): {n_4band} ({100*n_4band/len(sed_catalog):.1f}%)")
    print(f"    3 bands (degenerate): {n_3band} ({100*n_3band/len(sed_catalog):.1f}%)")
    if n_fewer > 0:
        print(f"    <3 bands (unreliable): {n_fewer} ({100*n_fewer/len(sed_catalog):.1f}%)")
    print(f"    U-band (F300W) detections: {has_uband.sum()} ({100*has_uband.sum()/len(sed_catalog):.1f}%)")

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
        sed_catalog["flux_f300"].to_numpy(),  # U band
        sed_catalog["flux_f450"].to_numpy(),  # B band
        sed_catalog["flux_f606"].to_numpy(),  # V band
        sed_catalog["flux_f814"].to_numpy(),  # I band
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

    # Report photo-z quality statistics
    odds_good = (sed_catalog["photo_z_odds"] >= 0.9).sum()
    odds_medium = ((sed_catalog["photo_z_odds"] >= 0.6) & (sed_catalog["photo_z_odds"] < 0.9)).sum()
    odds_poor = (sed_catalog["photo_z_odds"] < 0.6).sum()
    print(f"  Photo-z quality: {odds_good} excellent (ODDS>=0.9), {odds_medium} good (0.6<=ODDS<0.9), {odds_poor} poor (ODDS<0.6)")

    # ML-based star classification using morphological features
    # Required features: concentration, gini, half_light_radius, sersic_n (psf_ratio is computed)
    required_morph_cols = ["concentration", "gini", "half_light_radius", "sersic_n"]
    has_training_data = "in_star_mask" in sed_catalog.columns
    has_morph_features = all(col in sed_catalog.columns for col in required_morph_cols)
    missing_cols = [col for col in required_morph_cols if col not in sed_catalog.columns]

    if has_training_data and has_morph_features:
        try:
            sed_catalog["star_prob_ml"] = classify_stars(sed_catalog, "in_star_mask")
        except Exception as e:
            print(f"  Star classifier skipped: {e}")
    elif has_training_data and not has_morph_features:
        print(f"  Star classifier skipped: missing morphology columns {missing_cols}")

    # Optional: Zoobot deep learning morphology classification
    try:
        print("\n  Running Zoobot morphology classification...")
        cutout_dir = Path(output_dir) / "cutouts"
        # Use I-band (F814W) for cutouts - best depth and resolution
        i_band_img = next((img for img in images if img.band == "f814"), images[-1])
        cutout_paths = extract_cutouts(i_band_img.data, sed_catalog, cutout_dir)
        if cutout_paths:
            sed_catalog = classify_morphology(sed_catalog, cutout_paths)
    except Exception as e:
        print(f"  Zoobot skipped: {e}")

    # Add SED-based morphology classification (baseline from template types)
    from morphology.zoobot_classify import classify_by_sed_type
    sed_catalog = classify_by_sed_type(sed_catalog)
    print(f"  SED morphology distribution: {sed_catalog['sed_morphology'].value_counts().to_dict()}")

    # === SPECTROSCOPIC VALIDATION AND IMPROVEMENT ===
    # Cross-match with external spec-z catalogs (3D-HST, Hawaii Yang 2014, Fernandez-Soto 1999)
    # to validate photo-z quality and optionally replace unreliable photo-z with spec-z
    USE_SPECZ_VALIDATION = True  # Set to False to skip spec-z validation
    APPLY_SPECZ_CORRECTIONS = True  # Set to False to only validate without replacing

    if USE_SPECZ_VALIDATION:
        try:
            from validation.specz_validation import (
                load_specz_catalogs,
                cross_match_with_specz,
                apply_spectroscopic_redshifts,
                flag_catastrophic_outliers,
                compute_photoz_metrics,
                generate_validation_plots,
            )

            print("\n  === Spectroscopic Validation ===")

            # Load external spectroscopic catalogs
            specz_catalogs = load_specz_catalogs()

            if specz_catalogs:
                # Cross-match with 1 arcsec radius
                sed_catalog = cross_match_with_specz(sed_catalog, specz_catalogs, match_radius=1.0)

                # Compute and report validation metrics (before any corrections)
                matched = sed_catalog[sed_catalog['z_spec'].notna()]
                if len(matched) >= 5:
                    metrics = compute_photoz_metrics(matched['redshift'], matched['z_spec'])
                    print(f"\n  Photo-z validation metrics (N={metrics['n_matched']} matched):")
                    print(f"    NMAD = {metrics['nmad']:.4f}")
                    print(f"    Bias = {metrics['bias']:.4f}")
                    print(f"    Outlier rate (>0.15) = {100*metrics['outlier_015']:.1f}%")
                    print(f"    Catastrophic rate (>0.3) = {100*metrics['outlier_030']:.1f}%")

                # Apply spec-z corrections (replace photo-z with spec-z when available)
                if APPLY_SPECZ_CORRECTIONS:
                    sed_catalog = apply_spectroscopic_redshifts(sed_catalog)
                    sed_catalog = flag_catastrophic_outliers(sed_catalog)

                # Generate validation plots
                generate_validation_plots(sed_catalog, output_dir)

                # Report final spec-z statistics
                n_specz = (sed_catalog['redshift_source'] == 'spec').sum() if 'redshift_source' in sed_catalog.columns else 0
                print(f"\n  Final redshift sources: {n_specz} spec-z, {len(sed_catalog) - n_specz} photo-z")
            else:
                print("  No spectroscopic catalogs available, skipping validation")

        except Exception as e:
            print(f"  Spec-z validation skipped: {e}")

    # Step 5: Analysis and plotting
    print("\n[5/6] Analyzing results...")

    # Debug: Check data before filtering
    print(f"  DEBUG: sed_catalog shape before filtering: {sed_catalog.shape}")
    for col in ["redshift", "r_half_arcsec", "r_half_arcsec_error", "r_half_pix_corrected"]:
        if col in sed_catalog.columns:
            vals = sed_catalog[col]
            finite_vals = vals[np.isfinite(vals)]
            n_finite = len(finite_vals)
            n_positive = (finite_vals > 0).sum() if n_finite > 0 else 0
            min_val = finite_vals.min() if n_finite > 0 else float('nan')
            max_val = finite_vals.max() if n_finite > 0 else float('nan')
            print(f"  DEBUG: {col}: total={len(vals)}, finite={n_finite}, >0={n_positive}, min={min_val:.4f}, max={max_val:.4f}")
        else:
            print(f"  DEBUG: {col}: COLUMN MISSING")

    sed_catalog = sed_catalog.dropna(subset=["redshift", "r_half_arcsec", "r_half_arcsec_error"])
    # Filter out sources with inf errors (from zero-weight pixels)
    sed_catalog = sed_catalog[np.isfinite(sed_catalog["r_half_arcsec_error"])]

    # Flag unresolved sources (r_half_arcsec <= 0 after PSF correction) as PSF-like
    # but keep them in catalog with a minimum floor value for analysis
    n_unresolved = (sed_catalog["r_half_arcsec"] <= 0).sum()
    if n_unresolved > 0:
        print(f"  Found {n_unresolved} unresolved sources (r_half <= PSF), flagging as PSF-like")
        unresolved_mask = sed_catalog["r_half_arcsec"] <= 0
        sed_catalog.loc[unresolved_mask, "quality_flag"] = (
            sed_catalog.loc[unresolved_mask, "quality_flag"].astype(int) | FLAG_PSF_LIKE
        )
        # Set minimum floor for unresolved sources (half the pixel scale)
        MIN_ARCSEC = PIXEL_SCALE * 0.5
        sed_catalog.loc[unresolved_mask, "r_half_arcsec"] = MIN_ARCSEC

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
            FLAG_UNRELIABLE_Z |  # Exclude sources where σ_z > z
            FLAG_VERY_LOW_SNR |  # Exclude sources with SNR < 0.5 (50% catastrophic rate)
            FLAG_HIGH_Z_UNRELIABLE |  # Exclude uncertain high-z sources
            FLAG_CATASTROPHIC_PHOTOZ  # Exclude catastrophic photo-z outliers identified by spec-z
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

        print("\n  HIGH-QUALITY MODE: Applied strict quality filtering")
        print(f"    Input: {n_total} sources")
        print("    Rejection breakdown:")

        # Count sources rejected by each criterion
        n_flag_rejected = ((sed_catalog["quality_flag"].astype(int) & STRICT_EXCLUDE_FLAGS) != 0).sum()
        n_odds_rejected = (sed_catalog["odds_flag"] != 0).sum()
        n_chi2_rejected = (sed_catalog["chi2_flag"] != 0).sum()
        n_bimodal_rejected = sed_catalog["bimodal_flag"].sum()

        if n_total > 0:
            print(f"      - Quality flags:     {n_flag_rejected} ({100*n_flag_rejected/n_total:.1f}%)")
            print(f"      - ODDS < 0.9:        {n_odds_rejected} ({100*n_odds_rejected/n_total:.1f}%)")
            print(f"      - Chi2 > 5:          {n_chi2_rejected} ({100*n_chi2_rejected/n_total:.1f}%)")
            print(f"      - Bimodal redshift:  {n_bimodal_rejected} ({100*n_bimodal_rejected/n_total:.1f}%)")
            print(f"    Result: {n_kept} high-confidence galaxies ({100*n_kept/n_total:.1f}% retained)")
        else:
            print("      No sources to analyze")
    else:
        # Moderate filtering - excludes only clearly bad sources
        EXCLUDE_FLAGS = FLAG_BAD_PHOTOZ | FLAG_PSF_LIKE | FLAG_MASKED
        sed_catalog_filtered = sed_catalog[
            (sed_catalog["quality_flag"].astype(int) & EXCLUDE_FLAGS) == 0
        ]
        print("\n  MODERATE MODE: Basic quality filtering")
        print(f"    - After filtering (excluding bad_photoz, psf_like, masked): {len(sed_catalog_filtered)}")

    # Use filtered catalog for analysis
    sed_catalog_analysis = sed_catalog_filtered

    # Fallback to moderate filtering if high-quality mode yields too few sources
    MIN_SOURCES_FOR_ANALYSIS = 10
    if len(sed_catalog_analysis) < MIN_SOURCES_FOR_ANALYSIS and HIGH_QUALITY_MODE:
        print(f"\n  WARNING: High-quality filtering yielded only {len(sed_catalog_analysis)} sources")
        print("  Falling back to moderate filtering to enable analysis...")

        # Moderate filtering - excludes only clearly bad sources
        FALLBACK_FLAGS = FLAG_BAD_PHOTOZ | FLAG_PSF_LIKE | FLAG_MASKED
        sed_catalog_analysis = sed_catalog[
            (sed_catalog["quality_flag"].astype(int) & FALLBACK_FLAGS) == 0
        ]
        print(f"  After moderate filtering: {len(sed_catalog_analysis)} sources")

        # If still too few, use minimal filtering (only masked sources)
        if len(sed_catalog_analysis) < MIN_SOURCES_FOR_ANALYSIS:
            print("  WARNING: Moderate filtering still too restrictive")
            print("  Falling back to minimal filtering (excluding only masked sources)...")
            MINIMAL_FLAGS = FLAG_MASKED | FLAG_PSF_LIKE
            sed_catalog_analysis = sed_catalog[
                (sed_catalog["quality_flag"].astype(int) & MINIMAL_FLAGS) == 0
            ]
            print(f"  After minimal filtering: {len(sed_catalog_analysis)} sources")

    # ALWAYS exclude z_boundary_flag sources from analysis
    # These are photo-z fitting failures where the best-fit hit the z_min or z_max boundary
    # Their confidence intervals often don't include the boundary value, indicating unreliable redshifts
    if "z_boundary_flag" in sed_catalog_analysis.columns:
        n_before = len(sed_catalog_analysis)
        sed_catalog_analysis = sed_catalog_analysis[sed_catalog_analysis["z_boundary_flag"] == False].copy()
        n_excluded = n_before - len(sed_catalog_analysis)
        if n_excluded > 0:
            print(f"\n  Excluded {n_excluded} sources with z_boundary_flag=True (photo-z boundary failures)")
            print(f"  Remaining sources for analysis: {len(sed_catalog_analysis)}")

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
    if len(sed_catalog_analysis) < 3 or not np.isfinite(z_min_raw) or not np.isfinite(z_max_raw) or z_min_raw == z_max_raw:
        print("\n  WARNING: Insufficient data for binning analysis")
        print(f"    Sources: {len(sed_catalog_analysis)}, z range: [{z_min_raw:.3f}, {z_max_raw:.3f}]")
        print("    Skipping binning and model fitting. Results will be incomplete.")
        # Save catalogs before returning
        sed_catalog.to_csv(f"{output_dir}/galaxy_catalog_full.csv", index=False)
        sed_catalog_analysis.to_csv(f"{output_dir}/galaxy_catalog.csv", index=False)
        print(f"  Saved: {output_dir}/galaxy_catalog.csv ({len(sed_catalog_analysis)} sources)")
        # Return minimal results
        binned = pd.DataFrame(columns=["z_mid", "theta_med", "theta_err", "n"])
        binned_results = {}
        return sed_catalog, sed_catalog_analysis, binned, binned_results, star_mask_centers

    # Calculate adaptive number of bins based on data size
    n_bins_adaptive = get_adaptive_n_bins(len(sed_catalog_analysis))
    print(f"\n  Using adaptive binning: {n_bins_adaptive} bins for {len(sed_catalog_analysis)} quality-filtered galaxies")

    # Define binning strategies
    binning_strategies = {
        "Equal Width": bin_equal_width,
        "Percentile": bin_percentile,
        "Bayesian Blocks": bin_bayesian_blocks,
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

    # Use percentile as the primary binning (most physically justified)
    binned = binned_results.get("Percentile", bin_equal_width(sed_catalog_analysis))

    # Fit models
    z_min = max(1e-4, binned.z_mid.min())
    z_data_max_for_curve = sed_catalog_analysis.redshift.max()
    z_model = np.linspace(z_min, z_data_max_for_curve, 300)

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

    # Plot 3: Binning strategy comparison (1x3 grid with residuals)
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[3, 1, 1], hspace=0.08, wspace=0.15)

    colors = {
        "Equal Width": "green",
        "Percentile": "blue",
        "Bayesian Blocks": "purple",
    }
    markers = {"Equal Width": "o", "Percentile": "o", "Bayesian Blocks": "o"}

    strategy_names = list(binned_results.keys())

    for idx, name in enumerate(strategy_names):
        binned_data = binned_results[name]
        col = idx  # 1x3 layout: each strategy in its own column

        ax_main = fig.add_subplot(gs[0, col])
        ax_resid_lcdm = fig.add_subplot(gs[1, col], sharex=ax_main)
        ax_resid_static = fig.add_subplot(gs[2, col], sharex=ax_main)

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

        # Calculate residuals
        theta_lcdm_at_data = theta_lcdm_flat(binned_data["z_mid"].values, R_lcdm_i, Omega_m_i) * RAD_TO_ARCSEC
        theta_static_at_data = theta_static(binned_data["z_mid"].values, R_static_i) * RAD_TO_ARCSEC
        residuals_lcdm = binned_data["theta_med"].values - theta_lcdm_at_data
        residuals_static = binned_data["theta_med"].values - theta_static_at_data

        # ΛCDM residual plot
        ax_resid_lcdm.axhline(0, color="darkblue", linestyle="-", linewidth=1.5, alpha=0.7)
        ax_resid_lcdm.errorbar(
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
        ax_resid_lcdm.set_ylabel(r"$\theta - \theta_{\Lambda\mathrm{CDM}}$", fontsize=9)
        ax_resid_lcdm.grid(True, alpha=0.3)
        ax_resid_lcdm.set_xlim(z_lim)
        max_resid_lcdm = max(0.05, np.max(np.abs(residuals_lcdm) + binned_data["theta_err"].values) * 1.2)
        ax_resid_lcdm.set_ylim(-max_resid_lcdm, max_resid_lcdm)
        plt.setp(ax_resid_lcdm.get_xticklabels(), visible=False)

        # Static residual plot
        ax_resid_static.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax_resid_static.errorbar(
            binned_data["z_mid"],
            residuals_static,
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
        ax_resid_static.set_xlabel(r"Redshift $z$", fontsize=11)
        ax_resid_static.set_ylabel(r"$\theta - \theta_{\mathrm{Static}}$", fontsize=9)
        ax_resid_static.grid(True, alpha=0.3)
        ax_resid_static.set_xlim(z_lim)
        max_resid_static = max(0.05, np.max(np.abs(residuals_static) + binned_data["theta_err"].values) * 1.2)
        ax_resid_static.set_ylim(-max_resid_static, max_resid_static)

    fig.suptitle(r"Binning Strategies: $\Lambda$CDM vs Static", fontsize=14, fontweight="bold")
    plt.savefig(f"{output_dir}/binning_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/binning_comparison.pdf")

    # Plot 3b: Per-galaxy-type binning comparison plots
    # Only for galaxy types with enough statistics (>= 20 sources for meaningful binning)
    MIN_SOURCES_FOR_TYPE_PLOT = 20
    type_counts_for_binning = sed_catalog_analysis["galaxy_type"].value_counts()
    for gtype, gtype_count in type_counts_for_binning.items():
        if gtype_count < MIN_SOURCES_FOR_TYPE_PLOT:
            continue

        # Filter catalog for this galaxy type
        type_catalog = sed_catalog_analysis[sed_catalog_analysis["galaxy_type"] == gtype].copy()

        # Compute binning for this type
        type_binned_results = {}
        for name, bin_func in binning_strategies.items():
            try:
                type_binned_results[name] = bin_func(type_catalog)
            except Exception:
                pass  # Skip if binning fails for this type

        if len(type_binned_results) < 2:
            continue  # Need at least 2 strategies to make a comparison plot

        # Create 1x3 grid with residuals plot
        fig_type = plt.figure(figsize=(18, 10), constrained_layout=True)
        gs_type = fig_type.add_gridspec(3, 3, height_ratios=[3, 1, 1], hspace=0.08, wspace=0.15)

        # Dynamic axis limits for this type
        z_min_type = type_catalog["redshift"].min()
        z_max_type = type_catalog["redshift"].max()
        z_padding_type = (z_max_type - z_min_type) * 0.1
        z_lim_type = (max(0, z_min_type - z_padding_type), z_max_type + z_padding_type)

        theta_min_type = type_catalog["r_half_arcsec"].min()
        theta_max_type = type_catalog["r_half_arcsec"].max()
        theta_padding_type = (theta_max_type - theta_min_type) * 0.1
        theta_lim_type = (max(0, theta_min_type - theta_padding_type), theta_max_type + theta_padding_type)

        strategy_names_type = list(type_binned_results.keys())

        for idx, name in enumerate(strategy_names_type[:3]):  # Max 3 strategies for 1x3
            binned_data = type_binned_results[name]
            col = idx  # 1x3 layout: each strategy in its own column

            ax_main = fig_type.add_subplot(gs_type[0, col])
            ax_resid_lcdm = fig_type.add_subplot(gs_type[1, col], sharex=ax_main)
            ax_resid_static = fig_type.add_subplot(gs_type[2, col], sharex=ax_main)

            # Fit models for this binning
            R_static_type = get_radius(
                binned_data.z_mid.values,
                binned_data.theta_med.values,
                binned_data.theta_err.values,
                model="static",
            )
            R_lcdm_type, Omega_m_type = get_radius_and_omega(
                binned_data.z_mid.values,
                binned_data.theta_med.values,
                binned_data.theta_err.values,
            )

            z_model_type = np.linspace(max(0.05, z_lim_type[0]), z_lim_type[1], 200)
            theta_static_type = theta_static(z_model_type, R_static_type) * RAD_TO_ARCSEC
            theta_lcdm_type = theta_lcdm_flat(z_model_type, R_lcdm_type, Omega_m_type) * RAD_TO_ARCSEC

            # Compute chi2/ndf for both models
            def static_func_type(z, R=R_static_type):
                return theta_static(z, R) * RAD_TO_ARCSEC
            def lcdm_func_type(z, R=R_lcdm_type, Om=Omega_m_type):
                return theta_lcdm_flat(z, R, Om) * RAD_TO_ARCSEC

            stats_static_type = compute_chi2_stats(
                binned_data.z_mid.values, binned_data.theta_med.values,
                binned_data.theta_err.values, static_func_type, n_params=1
            )
            stats_lcdm_type = compute_chi2_stats(
                binned_data.z_mid.values, binned_data.theta_med.values,
                binned_data.theta_err.values, lcdm_func_type, n_params=2
            )

            # Plot individual galaxies as background
            ax_main.scatter(
                type_catalog["redshift"],
                type_catalog["r_half_arcsec"],
                c="lightgray",
                alpha=0.4,
                s=15,
                zorder=1,
            )

            # Plot binned data
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
                z_model_type, theta_static_type, "--", color="gray", linewidth=1.5, label=r"Static", zorder=2
            )
            ax_main.plot(
                z_model_type, theta_lcdm_type, "-", color="darkblue", linewidth=1.5, label=r"$\Lambda$CDM", zorder=2
            )

            ax_main.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=11)
            ax_main.set_title(
                f"{name} ({len(binned_data)} bins)\n"
                + r"$\Lambda$CDM: $R$" + f"={R_lcdm_type*1000:.1f} kpc, "
                + r"$\chi^2/{\rm ndf}$" + f"={stats_lcdm_type['chi2_ndf']:.2f}, "
                + f"P={stats_lcdm_type['p_value']:.3f}\n"
                + r"Static: $R$" + f"={R_static_type*1000:.1f} kpc, "
                + r"$\chi^2/{\rm ndf}$" + f"={stats_static_type['chi2_ndf']:.2f}, "
                + f"P={stats_static_type['p_value']:.3f}",
                fontsize=10
            )
            ax_main.legend(fontsize=8, loc="upper right")
            ax_main.grid(True, alpha=0.3)
            ax_main.set_xlim(z_lim_type)
            ax_main.set_ylim(theta_lim_type)
            plt.setp(ax_main.get_xticklabels(), visible=False)

            # Calculate residuals
            theta_lcdm_at_data_type = theta_lcdm_flat(binned_data["z_mid"].values, R_lcdm_type, Omega_m_type) * RAD_TO_ARCSEC
            theta_static_at_data_type = theta_static(binned_data["z_mid"].values, R_static_type) * RAD_TO_ARCSEC
            residuals_lcdm_type = binned_data["theta_med"].values - theta_lcdm_at_data_type
            residuals_static_type = binned_data["theta_med"].values - theta_static_at_data_type

            # ΛCDM residual plot
            ax_resid_lcdm.axhline(0, color="darkblue", linestyle="-", linewidth=1.5, alpha=0.7)
            ax_resid_lcdm.errorbar(
                binned_data["z_mid"],
                residuals_lcdm_type,
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
            ax_resid_lcdm.set_ylabel(r"$\theta - \theta_{\Lambda\mathrm{CDM}}$", fontsize=9)
            ax_resid_lcdm.grid(True, alpha=0.3)
            ax_resid_lcdm.set_xlim(z_lim_type)
            max_resid_lcdm_type = max(0.05, np.max(np.abs(residuals_lcdm_type) + binned_data["theta_err"].values) * 1.2)
            ax_resid_lcdm.set_ylim(-max_resid_lcdm_type, max_resid_lcdm_type)
            plt.setp(ax_resid_lcdm.get_xticklabels(), visible=False)

            # Static residual plot
            ax_resid_static.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
            ax_resid_static.errorbar(
                binned_data["z_mid"],
                residuals_static_type,
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
            ax_resid_static.set_xlabel(r"Redshift $z$", fontsize=11)
            ax_resid_static.set_ylabel(r"$\theta - \theta_{\mathrm{Static}}$", fontsize=9)
            ax_resid_static.grid(True, alpha=0.3)
            ax_resid_static.set_xlim(z_lim_type)
            max_resid_static_type = max(0.05, np.max(np.abs(residuals_static_type) + binned_data["theta_err"].values) * 1.2)
            ax_resid_static.set_ylim(-max_resid_static_type, max_resid_static_type)

        # Create safe filename from galaxy type
        safe_gtype = gtype.replace(" ", "_").replace("/", "-").lower()
        fig_type.suptitle(f"Binning Strategies: {gtype} (n={gtype_count})", fontsize=14, fontweight="bold")
        plt.savefig(f"{output_dir}/binning_comparison_{safe_gtype}.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {output_dir}/binning_comparison_{safe_gtype}.pdf")

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

    # Plot 8: Final selection showing galaxies, detected stars, and mask stars
    n_galaxies = len(sed_catalog_analysis)

    # Identify different source types from the FULL catalog
    in_mask = sed_catalog["in_star_mask"].fillna(False).astype(bool) if "in_star_mask" in sed_catalog.columns else pd.Series([False] * len(sed_catalog))
    has_psf_flag = (sed_catalog["quality_flag"].astype(int) & FLAG_PSF_LIKE) != 0

    # Mask stars: sources in star mask
    mask_stars = sed_catalog[in_mask]
    n_mask_stars = len(mask_stars)

    # Detected stars: sources with PSF-like flag but NOT in mask
    detected_stars = sed_catalog[has_psf_flag & ~in_mask]
    n_detected_stars = len(detected_stars)

    _fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for idx, image in enumerate(images):
        ax = axes[idx // 2, idx % 2]
        vmin, vmax = np.percentile(image.data, [20, 99])
        ax.imshow(image.data, cmap="gray_r", vmin=vmin, vmax=vmax, origin="lower")

        # Plot galaxies (green)
        if n_galaxies > 0:
            ax.scatter(
                sed_catalog_analysis["xcentroid"],
                sed_catalog_analysis["ycentroid"],
                s=10, facecolors="none", edgecolors="green", linewidth=0.5,
                label=f"Galaxies ({n_galaxies})"
            )

        # Plot detected stars (blue)
        if n_detected_stars > 0:
            ax.scatter(
                detected_stars["xcentroid"],
                detected_stars["ycentroid"],
                s=15, facecolors="none", edgecolors="blue", linewidth=0.7,
                label=f"Detected stars ({n_detected_stars})"
            )

        # Plot mask stars (red)
        if n_mask_stars > 0:
            ax.scatter(
                mask_stars["xcentroid"],
                mask_stars["ycentroid"],
                s=20, facecolors="none", edgecolors="red", linewidth=0.9,
                label=f"Mask stars ({n_mask_stars})"
            )

        ax.set_title(f"Band {image.band.upper()}", fontsize=9)
        ax.set_xlabel(r"X (pixels)")
        ax.set_ylabel(r"Y (pixels)")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=7)

    plt.suptitle(f"Final Selection\n({n_galaxies} galaxies, {n_detected_stars} detected stars, {n_mask_stars} mask stars)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_selection.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/final_selection.pdf")

    # Save catalogs (both full and quality-filtered versions)
    # Add is_star and is_galaxy columns for consistency with chip3 output
    if "in_star_mask" in sed_catalog.columns:
        in_mask = sed_catalog["in_star_mask"].fillna(False).astype(bool)
        sed_catalog["is_star"] = in_mask
        sed_catalog["is_galaxy"] = ~in_mask
    if "in_star_mask" in sed_catalog_analysis.columns:
        # Use .copy() to avoid SettingWithCopyWarning
        sed_catalog_analysis = sed_catalog_analysis.copy()
        in_mask_filtered = sed_catalog_analysis["in_star_mask"].fillna(False).astype(bool)
        sed_catalog_analysis["is_star"] = in_mask_filtered
        sed_catalog_analysis["is_galaxy"] = ~in_mask_filtered

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

    For chip3, star classification is simplified: only sources inside the
    course staff star mask are classified as stars. This provides a cleaner
    galaxy sample for the angular size analysis.
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

    # Override star classification for chip3: only star mask sources are stars
    # This simplifies the classification to use the course staff's star mask
    if "in_star_mask" in chip3_catalog.columns:
        in_mask = chip3_catalog["in_star_mask"].fillna(False).astype(bool)
        n_original_stars = chip3_catalog["is_star"].sum() if "is_star" in chip3_catalog.columns else 0
        n_mask_stars = in_mask.sum()

        # Set is_star = True only for sources in the star mask
        chip3_catalog["is_star"] = in_mask
        chip3_catalog["is_galaxy"] = ~in_mask

        print(f"  Chip3 star classification override: {n_original_stars} -> {n_mask_stars} stars (mask only)")

    if "in_star_mask" in chip3_catalog_filtered.columns:
        in_mask_filtered = chip3_catalog_filtered["in_star_mask"].fillna(False).astype(bool)
        chip3_catalog_filtered["is_star"] = in_mask_filtered
        chip3_catalog_filtered["is_galaxy"] = ~in_mask_filtered

    return chip3_catalog, chip3_catalog_filtered


def generate_all_plots(output_dir: str, dataset_name: str):
    """Generate all plots for a given output directory."""
    # Binning comparison (all types)
    generate_binning_plot(data_dir=output_dir,
        output_path=f"{output_dir}/binning_comparison.pdf",
        dataset_name=dataset_name)

    # Angular size vs redshift (main science plot)
    generate_angular_size_plot(data_dir=output_dir,
        output_path=f"{output_dir}/angular_size_vs_redshift.pdf",
        dataset_name=dataset_name)

    # Redshift distribution histograms
    generate_redshift_histogram(data_dir=output_dir,
        output_path=f"{output_dir}/redshift_histogram.pdf",
        dataset_name=dataset_name)

    # Galaxy type histograms
    generate_galaxy_type_histogram(data_dir=output_dir,
        output_path=f"{output_dir}/galaxy_types.pdf",
        dataset_name=dataset_name)

    # Per-galaxy-type analysis
    catalog_path = f"{output_dir}/galaxy_catalog.csv"
    if os.path.exists(catalog_path):
        type_catalog = pd.read_csv(catalog_path)

        # Guard against empty catalog
        if len(type_catalog) == 0:
            print(f"\n--- Per-Galaxy-Type Analysis ({dataset_name}) ---")
            print("  Catalog is empty. Skipping per-type analysis.")
            return

        type_catalog = type_catalog.dropna(subset=["redshift", "r_half_arcsec", "r_half_arcsec_error"])

        if len(type_catalog) == 0:
            print(f"\n--- Per-Galaxy-Type Analysis ({dataset_name}) ---")
            print("  No valid sources after filtering. Skipping per-type analysis.")
            return

        type_catalog = type_catalog[np.isfinite(type_catalog["r_half_arcsec_error"])]
        # Exclude z_boundary_flag sources (photo-z boundary failures)
        if "z_boundary_flag" in type_catalog.columns:
            type_catalog = type_catalog[type_catalog["z_boundary_flag"] == False]

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

        # Per-type binning comparison plots (lower threshold for more types)
        types_for_binning = get_types_with_enough_statistics(type_catalog, min_count=MIN_GALAXIES_PER_TYPE_BINNING)
        print(f"\n--- Per-Type Binning Comparisons ({dataset_name}) ---")
        print(f"  Galaxy types with N >= {MIN_GALAXIES_PER_TYPE_BINNING}: {types_for_binning}")
        if len(types_for_binning) > 0:
            print(f"  Generating binning comparison plots for {len(types_for_binning)} types...")
            for gtype in types_for_binning:
                generate_per_type_binning_comparison(output_dir, output_dir, dataset_name, gtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Angular size test analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_analysis.py full                          # Run analysis on full mosaic
    python run_analysis.py chip3                         # Run analysis on chip3 only
    python run_analysis.py both                          # Run full and extract chip3
    python run_analysis.py full --resource-profile low   # Run with low resource usage (constrained machines)
    python run_analysis.py full --resource-profile high  # Run with high resource usage (workstations)
    python run_analysis.py full --detection-backend sep  # Use SEP (SExtractor) for faster detection

Resource Profiles:
    low:    2-4GB RAM, 2 cores - uses coarser redshift grid (z_step=0.02), fewer threads
    medium: 4-8GB RAM, 4 cores - standard precision (z_step=0.01)
    high:   8GB+ RAM, 8+ cores - full precision with maximum parallelism
    auto:   Automatically detect based on available system resources (default)

Detection Backends:
    photutils: Astropy-affiliated photometry library (default, well-documented)
    sep:       SEP (Source Extractor as Python library) - faster for large images,
               produces SExtractor-compatible catalogs with Kron aperture photometry
    auto:      Use SEP if available, otherwise photutils

Output Structure:
    Results are saved to output/{mode}_HDF/ directory.
    Star identification uses both the star mask and morphology-based detection.
        """
    )
    parser.add_argument(
        "mode",
        choices=["full", "chip3", "both"],
        help="Analysis mode: 'full' for entire image, 'chip3' for chip3 only, 'both' to run full and extract chip3"
    )
    parser.add_argument(
        "--resource-profile",
        type=str,
        choices=["low", "medium", "high", "auto"],
        default="auto",
        help="Resource usage profile: 'low' for constrained machines (2-4GB RAM), "
             "'medium' for typical laptops (4-8GB), 'high' for workstations (8GB+), "
             "'auto' to detect automatically (default)"
    )
    parser.add_argument(
        "--detection-backend",
        type=str,
        choices=["photutils", "sep", "auto"],
        default="auto",
        help="Source detection backend: 'photutils' (default, Astropy-affiliated), "
             "'sep' (SExtractor algorithm, faster for large images), "
             "'auto' to use sep if available, else photutils"
    )
    args = parser.parse_args()

    # Apply resource profile if specified via command line
    # Note: For full effect, set ASTRO_RESOURCE_PROFILE env var before running
    if args.resource_profile != "auto":
        import os
        os.environ["ASTRO_RESOURCE_PROFILE"] = args.resource_profile

    # Get the active config (may have been set by env var or command line)
    active_config = get_config()

    # Determine detection backend
    if args.detection_backend == "auto":
        detection_backend = "sep" if SEP_AVAILABLE else "photutils"
    elif args.detection_backend == "sep" and not SEP_AVAILABLE:
        print("Warning: SEP not available, falling back to photutils")
        detection_backend = "photutils"
    else:
        detection_backend = args.detection_backend

    print("=" * 60)
    print("ANGULAR SIZE TEST ANALYSIS")
    print(f"Mode: {args.mode}")
    print(f"Resource profile: {active_config.profile.value} "
          f"(threads={active_config.n_threads}, z_step={active_config.z_step})")
    print(f"Detection backend: {detection_backend}")
    print("=" * 60)

    # Determine which dataset modes to run
    if args.mode == "both":
        dataset_modes = ["full", "chip3"]
    else:
        dataset_modes = [args.mode]

    # Run analysis for each dataset mode
    # Star detection always uses mask + morphology-based detection
    all_output_dirs = []
    for dataset_mode in dataset_modes:
        print("\n" + "=" * 60)
        mode_desc = "Full Field" if dataset_mode == "full" else "Chip3"
        print(f"RUNNING ANALYSIS: {mode_desc}")
        print("=" * 60)

        # Run analysis with this configuration
        sed_catalog, sed_catalog_analysis, binned, binned_results, star_mask_centers = main(
            dataset_mode,
            use_additional_star_detection=True,
            detection_backend=detection_backend,
        )

        # Track output directory for plot generation
        output_dir = f"./output/{dataset_mode}_HDF"
        all_output_dirs.append((output_dir, mode_desc))

    # Generate plots for all output directories
    print("\n" + "=" * 60)
    print("GENERATING PLOTS FOR ALL CONFIGURATIONS")
    print("=" * 60)

    for output_dir, dataset_name in all_output_dirs:
        print(f"\n--- Generating plots for {dataset_name} ---")
        generate_all_plots(output_dir, dataset_name)

    # Print summary of output locations
    print("\n" + "=" * 60)
    print("OUTPUT SUMMARY")
    print("=" * 60)
    for output_dir, dataset_name in all_output_dirs:
        print(f"  {dataset_name}: {output_dir}/")
    print("\nEach directory contains:")
    print("  - galaxy_catalog.csv: Full catalog with all measurements")
    print("  - galaxy_catalog_filtered.csv: Quality-filtered catalog for analysis")
    print("  - Various PDF plots (binning, redshift histogram, angular size, etc.)")
