#!/usr/bin/env python3
"""
Full analysis pipeline for angular size test
Runs the complete processing and saves output plots as PDFs

Usage:
    python run_analysis.py full    # Run on entire image (4096x4096), output to ./output/full/
    python run_analysis.py chip3   # Run on chip3 only (2048x2048), output to ./output/chip3/
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial

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
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import SourceCatalog, SourceFinder, make_2dgaussian_kernel
from scipy.spatial import cKDTree
from scipy import ndimage

from classify import classify_galaxy, classify_galaxy_batch_with_pdf, PhotoZResult
from scientific import (
    D_A_LCDM_vectorized,
    H0,
    c,
    choose_redshift_bin_edges,
    get_radius,
    get_radius_and_omega,
    theta_lcdm,
    theta_lcdm_flat,
    theta_static,
)

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
    Compute stellarity index (0=galaxy, 1=star) following SExtractor methodology.

    This provides a continuous 0-1 score based on multiple morphological criteria,
    similar to SExtractor's CLASS_STAR but using interpretable parameters.

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
    - SDSS star/galaxy: https://www.sdss.org/dr16/algorithms/classify/
    """
    scores = []

    # 1. Size score: stars have r_half close to PSF
    # PSF half-light radius ≈ PSF_FWHM * 0.42 (for Gaussian)
    psf_r_half = psf_fwhm_pix * 0.42
    if np.isfinite(r_half) and r_half > 0:
        size_ratio = r_half / psf_r_half
        # Score: 1 if exactly PSF size, 0 if 3x larger
        size_score = max(0, 1.0 - abs(size_ratio - 1.0) / 2.0)
        scores.append(size_score)

    # 2. Concentration score: stars have C > 0.45 (SDSS-style r50/r90)
    # Higher concentration = more star-like
    if np.isfinite(concentration):
        # Map: C=0.30 -> 0, C=0.50 -> 1
        conc_score = min(1.0, max(0, (concentration - 0.30) / 0.20))
        scores.append(conc_score)

    # 3. Roundness score: stars are round (low ellipticity)
    # Map: e=0 -> 1 (round=star-like), e=0.5 -> 0 (elongated=galaxy-like)
    if np.isfinite(ellipticity):
        round_score = max(0, 1.0 - ellipticity / 0.5)
        scores.append(round_score)

    if len(scores) == 0:
        return 0.0

    return float(np.mean(scores))


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


def plot_color_color_diagram(
    sed_catalog: pd.DataFrame,
    output_path: str,
    title_suffix: str = "",
    show_templates: bool = True,
    spectra_path: str = "./spectra",
) -> None:
    """
    Create a U-B vs B-V color-color diagram for galaxy classification analysis.

    This diagnostic plot shows how galaxies distribute in color-color space,
    which is sensitive to stellar populations, star formation history, and redshift.
    Early-type (elliptical) galaxies tend toward red colors (positive U-B, B-V),
    while late-type starbursts are bluer.

    Parameters
    ----------
    sed_catalog : pd.DataFrame
        Galaxy catalog with flux columns: flux_u, flux_b, flux_v, flux_i
        and galaxy_type column for classification
    output_path : str
        Output file path for the PDF
    title_suffix : str, optional
        Suffix to add to plot title (e.g., " (Chip 3)")
    show_templates : bool, optional
        If True, overlay expected template colors at different redshifts
    spectra_path : str, optional
        Path to spectra directory for template calculations

    Notes
    -----
    Magnitudes are computed as instrumental magnitudes: m = -2.5 * log10(flux)
    Color indices are differences: U-B = m_U - m_B, B-V = m_B - m_V

    References
    ----------
    - SDSS color-color diagrams: https://www.sdss.org/dr17/algorithms/magnitudes/
    - Galaxy color evolution: Strateva et al. (2001), AJ 122, 1861
    """
    # Convert fluxes to instrumental magnitudes
    with np.errstate(divide="ignore", invalid="ignore"):
        mag_u = np.where(
            sed_catalog["flux_u"] > 0,
            -2.5 * np.log10(sed_catalog["flux_u"]),
            np.nan,
        )
        mag_b = np.where(
            sed_catalog["flux_b"] > 0,
            -2.5 * np.log10(sed_catalog["flux_b"]),
            np.nan,
        )
        mag_v = np.where(
            sed_catalog["flux_v"] > 0,
            -2.5 * np.log10(sed_catalog["flux_v"]),
            np.nan,
        )

    # Compute color indices
    color_ub = mag_u - mag_b  # U-B
    color_bv = mag_b - mag_v  # B-V

    # Filter out invalid colors
    valid_mask = np.isfinite(color_ub) & np.isfinite(color_bv)
    color_ub_valid = color_ub[valid_mask]
    color_bv_valid = color_bv[valid_mask]
    galaxy_types_valid = sed_catalog.loc[valid_mask, "galaxy_type"].values

    if len(color_ub_valid) == 0:
        print(f"    Warning: No valid colors for color-color diagram, skipping")
        return

    # Create figure
    _fig, ax = plt.subplots(figsize=(10, 8))

    # Plot galaxies by type with consistent colors
    unique_types = np.unique(galaxy_types_valid)

    for gtype in unique_types:
        mask = galaxy_types_valid == gtype
        color = GALAXY_TYPE_COLORS.get(gtype, GALAXY_TYPE_FALLBACK_COLOR)

        ax.scatter(
            color_bv_valid[mask],
            color_ub_valid[mask],
            c=color,
            s=25,
            alpha=0.6,
            label=gtype,
            edgecolors="none",
        )

    # Overlay template tracks at different redshifts (optional)
    if show_templates:
        try:
            _overlay_template_colors(ax, spectra_path)
        except Exception as e:
            print(f"    Warning: Could not overlay template colors: {e}")

    # Axis labels and styling
    ax.set_xlabel(r"$B - V$ (mag)", fontsize=12)
    ax.set_ylabel(r"$U - B$ (mag)", fontsize=12)
    ax.set_title(
        r"Color-Color Diagram" + title_suffix + f"\n(N={len(color_ub_valid)} galaxies)",
        fontsize=14,
    )

    # Add legend with galaxy types
    ax.legend(
        fontsize=9,
        loc="upper left",
        framealpha=0.9,
        title="Galaxy Type",
        title_fontsize=10,
    )

    ax.grid(True, alpha=0.3)

    # Let matplotlib auto-scale to include both data and template tracks
    # Add small margin for aesthetics
    ax.margins(0.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _overlay_template_colors(ax, spectra_path: str) -> None:
    """
    Overlay expected galaxy template colors at different redshifts.

    Draws tracks showing how each galaxy type moves in color-color space
    as redshift increases (k-correction effects).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on
    spectra_path : str
        Path to spectra directory
    """
    from classify import GALAXY_TYPES, _load_spectrum

    # Redshift grid for template tracks
    z_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

    # Filter wavelength ranges (center, half-width) in Angstroms
    # HST/WFPC2 filters: F300W (U), F450W (B), F606W (V), F814W (I)
    filter_ranges = {
        "u": (3000.0, 760.0),   # F300W: ~2240-3760 A
        "b": (4500.0, 750.0),   # F450W: ~3750-5250 A
        "v": (6060.0, 475.0),   # F606W: ~5585-6535 A
        "i": (8140.0, 383.0),   # F814W: ~7757-8523 A
    }

    # Only plot tracks for a subset of types to avoid clutter
    template_types = ["elliptical", "Sa", "Sb", "sbt3", "sbt6"]

    for gtype in template_types:
        if gtype not in GALAXY_TYPES:
            continue

        try:
            wl, spec = _load_spectrum(spectra_path, gtype)
        except Exception:
            continue

        colors_bv = []
        colors_ub = []

        for z in z_grid:
            # Compute synthetic photometry at this redshift
            # Redshift the spectrum: observed wavelength = rest wavelength * (1+z)
            wl_obs = wl * (1 + z)

            # Compute flux in each filter (simplified - integrate spectrum over filter)
            fluxes = {}
            for band, (center, half_width) in filter_ranges.items():
                wl_lo = center - half_width
                wl_hi = center + half_width
                # Find spectrum indices within filter range
                in_filter = (wl_obs >= wl_lo) & (wl_obs <= wl_hi)
                if np.sum(in_filter) > 2:
                    # Simple trapezoidal integration
                    fluxes[band] = np.trapezoid(spec[in_filter], wl_obs[in_filter])
                else:
                    fluxes[band] = np.nan

            # Convert to magnitudes and colors
            if all(fluxes.get(b, 0) > 0 for b in ["u", "b", "v"]):
                mag_u = -2.5 * np.log10(fluxes["u"])
                mag_b = -2.5 * np.log10(fluxes["b"])
                mag_v = -2.5 * np.log10(fluxes["v"])
                colors_bv.append(mag_b - mag_v)
                colors_ub.append(mag_u - mag_b)
            else:
                colors_bv.append(np.nan)
                colors_ub.append(np.nan)

        # Plot template track
        colors_bv = np.array(colors_bv)
        colors_ub = np.array(colors_ub)
        valid = np.isfinite(colors_bv) & np.isfinite(colors_ub)

        if np.sum(valid) > 1:
            color = GALAXY_TYPE_COLORS.get(gtype, GALAXY_TYPE_FALLBACK_COLOR)
            # Plot template track as dashed line with markers
            ax.plot(
                colors_bv[valid],
                colors_ub[valid],
                "--",
                color=color,
                linewidth=2.5,
                alpha=0.9,
                zorder=5,
            )
            # Mark redshift points with small circles
            ax.scatter(
                colors_bv[valid],
                colors_ub[valid],
                c=color,
                s=30,
                marker="o",
                edgecolors="white",
                linewidths=0.5,
                zorder=6,
            )
            # Annotate redshift values (only at key points to avoid clutter)
            for i, z in enumerate(z_grid):
                if valid[i] and z in [0.0, 0.5, 1.0, 2.0]:
                    ax.annotate(
                        f"z={z:.1f}",
                        (colors_bv[i], colors_ub[i]),
                        fontsize=7,
                        fontweight="bold",
                        alpha=0.8,
                        xytext=(5, 5),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
                    )


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

    For approximately normal data: SE(median) ≈ 1.2533 × σ / √n
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
    - 16th/84th percentiles for 1σ spread visualization

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
    with fits.open(file_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        # Use native byte order for efficient numpy operations
        if data.dtype.byteorder == ">":
            data = data.astype(data.dtype.newbyteorder("="), copy=False)
        data = adjust_data(data)

    # Load weight map if provided
    weight = None
    if weight_path is not None:
        try:
            with fits.open(weight_path) as hdul_w:
                weight = hdul_w[0].data
                if weight.dtype.byteorder == ">":
                    weight = weight.astype(weight.dtype.newbyteorder("="), copy=False)
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
    from photutils.background import Background2D, MedianBackground, SExtractorBackground

    bkg_estimator = MedianBackground()
    current_mask = mask.copy() if mask is not None else np.zeros(data.shape, dtype=bool)

    for iteration in range(n_iterations):
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
    band, data, weight, mask, kernel, detection_sigma, npixels, nlevels, contrast = args

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
    from classify import classify_galaxy
    galaxy_type, redshift = classify_galaxy(fluxes, errors, spectra_path=spectra_path)
    return idx, galaxy_type, redshift


def _classify_galaxy_batch(args):
    """Classify a batch of galaxies with PDF uncertainties (reduces IPC overhead)."""
    batch, spectra_path = args
    from classify import classify_galaxy_batch_with_pdf

    # Use the new PDF-based batch classification
    # Returns: (idx, galaxy_type, redshift, z_lo, z_hi, chi_sq_min, odds)
    results = classify_galaxy_batch_with_pdf(batch, spectra_path)
    return results


def plot_photoz_quality(
    catalog: pd.DataFrame,
    output_dir: str,
    title_suffix: str = "",
) -> None:
    """
    Plot photo-z quality metrics (ODDS and chi-squared distributions).

    Creates a 2-panel figure showing:
    - Left: ODDS distribution histogram with quality thresholds
    - Right: Chi-squared distribution histogram

    Both panels include:
    - Quality threshold lines
    - Fraction of galaxies passing cuts
    - Median/mean annotations

    Parameters
    ----------
    catalog : pd.DataFrame
        Galaxy catalog containing 'photo_z_odds' and 'chi_sq_min' columns
    output_dir : str
        Directory to save the output PDF
    title_suffix : str, optional
        Suffix to add to plot title (e.g., " (Chip 3)")

    Notes
    -----
    ODDS thresholds follow BPZ conventions:
    - ODDS > 0.95: Excellent (highly confident photo-z)
    - ODDS > 0.90: Good (reliable photo-z)
    - ODDS > 0.60: Acceptable (usable with caution)
    - ODDS < 0.60: Poor (unreliable, flagged as FLAG_BAD_PHOTOZ)

    Chi-squared interpretation:
    - chi_sq < 1: Very good fit (may indicate overestimated errors)
    - chi_sq ~ 1-3: Good fit
    - chi_sq > 10: Poor fit (may indicate bad template match or outlier)
    """
    # Check if required columns exist
    has_odds = "photo_z_odds" in catalog.columns
    has_chi2 = "chi_sq_min" in catalog.columns

    if not has_odds and not has_chi2:
        print(f"  Warning: No photo-z quality columns found, skipping photoz_quality plot")
        return

    # Determine figure layout based on available data
    if has_odds and has_chi2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    elif has_odds:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = None
    else:
        fig, ax2 = plt.subplots(figsize=(8, 5))
        ax1 = None

    # Panel 1: ODDS distribution
    if has_odds and ax1 is not None:
        odds = catalog["photo_z_odds"].dropna().values
        n_total = len(odds)

        if n_total > 0:
            # Compute statistics
            odds_median = np.median(odds)
            odds_mean = np.mean(odds)

            # Quality thresholds
            n_excellent = np.sum(odds >= 0.95)
            n_good = np.sum((odds >= 0.90) & (odds < 0.95))
            n_acceptable = np.sum((odds >= 0.60) & (odds < 0.90))
            n_poor = np.sum(odds < 0.60)

            frac_excellent = n_excellent / n_total * 100
            frac_good = n_good / n_total * 100
            frac_acceptable = n_acceptable / n_total * 100
            frac_poor = n_poor / n_total * 100

            # Histogram
            bins = np.linspace(0, 1, 25)
            ax1.hist(odds, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")

            # Threshold lines
            ax1.axvline(0.95, color="green", linestyle="--", linewidth=2, label=r"Excellent ($\geq$0.95)")
            ax1.axvline(0.90, color="orange", linestyle="--", linewidth=2, label=r"Good ($\geq$0.90)")
            ax1.axvline(0.60, color="red", linestyle="--", linewidth=2, label=r"Acceptable ($\geq$0.60)")

            # Median/mean lines
            ax1.axvline(odds_median, color="darkblue", linestyle="-", linewidth=2, alpha=0.8)
            ax1.axvline(odds_mean, color="purple", linestyle=":", linewidth=2, alpha=0.8)

            # Annotations
            ax1.text(
                0.02, 0.95,
                f"N = {n_total}\n"
                f"Median = {odds_median:.3f}\n"
                f"Mean = {odds_mean:.3f}",
                transform=ax1.transAxes,
                verticalalignment="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Quality breakdown annotation (right side)
            ax1.text(
                0.98, 0.95,
                f"Excellent: {frac_excellent:.1f}\\%\n"
                f"Good: {frac_good:.1f}\\%\n"
                f"Acceptable: {frac_acceptable:.1f}\\%\n"
                f"Poor: {frac_poor:.1f}\\%",
                transform=ax1.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            ax1.set_xlabel(r"Photo-z ODDS", fontsize=12)
            ax1.set_ylabel(r"Number of galaxies", fontsize=12)
            ax1.set_title(r"Photo-z Quality: ODDS Distribution", fontsize=14)
            ax1.set_xlim(0, 1.05)
            ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
            ax1.grid(True, alpha=0.3)

    # Panel 2: Chi-squared distribution
    if has_chi2 and ax2 is not None:
        chi2 = catalog["chi_sq_min"].dropna().values
        # Filter out extreme values for better visualization
        chi2_valid = chi2[(chi2 > 0) & (chi2 < 100)]
        n_total_chi2 = len(chi2)
        n_valid = len(chi2_valid)

        if n_valid > 0:
            # Compute statistics
            chi2_median = np.median(chi2_valid)
            chi2_mean = np.mean(chi2_valid)

            # Quality thresholds
            n_very_good = np.sum(chi2_valid <= 1.0)
            n_good = np.sum((chi2_valid > 1.0) & (chi2_valid <= 3.0))
            n_fair = np.sum((chi2_valid > 3.0) & (chi2_valid <= 10.0))
            n_poor = np.sum(chi2_valid > 10.0)

            frac_very_good = n_very_good / n_valid * 100
            frac_good = n_good / n_valid * 100
            frac_fair = n_fair / n_valid * 100
            frac_poor = n_poor / n_valid * 100

            # Use logarithmic bins for chi-squared (spans wide range)
            chi2_max = min(np.percentile(chi2_valid, 99), 50)
            bins = np.logspace(np.log10(max(0.01, chi2_valid.min())), np.log10(chi2_max), 30)
            ax2.hist(chi2_valid, bins=bins, edgecolor="black", alpha=0.7, color="coral")
            ax2.set_xscale("log")

            # Threshold lines
            ax2.axvline(1.0, color="green", linestyle="--", linewidth=2, label=r"$\chi^2 = 1$")
            ax2.axvline(3.0, color="orange", linestyle="--", linewidth=2, label=r"$\chi^2 = 3$")
            ax2.axvline(10.0, color="red", linestyle="--", linewidth=2, label=r"$\chi^2 = 10$")

            # Median line
            ax2.axvline(chi2_median, color="darkblue", linestyle="-", linewidth=2, alpha=0.8)

            # Annotations
            ax2.text(
                0.02, 0.95,
                f"N = {n_valid}"
                + (f" (of {n_total_chi2})" if n_valid < n_total_chi2 else "")
                + f"\nMedian = {chi2_median:.2f}\n"
                f"Mean = {chi2_mean:.2f}",
                transform=ax2.transAxes,
                verticalalignment="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Quality breakdown annotation (right side)
            ax2.text(
                0.98, 0.95,
                f"Very good ($\\leq$1): {frac_very_good:.1f}\\%\n"
                f"Good (1-3): {frac_good:.1f}\\%\n"
                f"Fair (3-10): {frac_fair:.1f}\\%\n"
                f"Poor ($>$10): {frac_poor:.1f}\\%",
                transform=ax2.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            ax2.set_xlabel(r"$\chi^2_{\mathrm{min}}$ (SED fit)", fontsize=12)
            ax2.set_ylabel(r"Number of galaxies", fontsize=12)
            ax2.set_title(r"Photo-z Quality: $\chi^2$ Distribution", fontsize=14)
            ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
            ax2.grid(True, alpha=0.3)

    # Overall title if two panels
    if has_odds and has_chi2:
        fig.suptitle(f"Photo-z Quality Metrics{title_suffix}", fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/photoz_quality.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/photoz_quality.pdf")


def plot_physical_size_distribution(
    catalog: pd.DataFrame,
    output_path: str,
    Omega_m: float = 0.3,
    Omega_L: float = 0.7,
    title_suffix: str = "",
    by_type: bool = False,
    by_redshift_bin: bool = False,
    n_redshift_bins: int = 3,
) -> dict:
    """
    Plot histogram of physical galaxy sizes in kpc.

    Converts angular sizes (arcsec) to physical sizes (kpc) using the LCDM
    angular diameter distance D_A(z).

    Physical size formula: R = theta * D_A(z)
    where theta is in radians and D_A is in Mpc, giving R in Mpc (converted to kpc).

    Parameters
    ----------
    catalog : pd.DataFrame
        Galaxy catalog with columns: 'redshift', 'r_half_arcsec', and optionally 'galaxy_type'
    output_path : str
        Path to save the output PDF
    Omega_m : float
        Matter density parameter for LCDM cosmology (default: 0.3)
    Omega_L : float
        Dark energy density parameter (default: 0.7)
    title_suffix : str
        Optional suffix for plot title (e.g., " (Chip 3)")
    by_type : bool
        If True, show separate histograms for different galaxy types
    by_redshift_bin : bool
        If True, show separate histograms for different redshift bins
    n_redshift_bins : int
        Number of redshift bins if by_redshift_bin is True

    Returns
    -------
    dict
        Statistics: mean, median, std, min, max of physical sizes in kpc
    """
    # Filter valid data
    valid_mask = (
        np.isfinite(catalog["redshift"]) &
        np.isfinite(catalog["r_half_arcsec"]) &
        (catalog["redshift"] > 0) &
        (catalog["r_half_arcsec"] > 0)
    )
    df = catalog[valid_mask].copy()

    if len(df) == 0:
        print(f"  Warning: No valid data for physical size distribution")
        return {}

    # Convert angular size (arcsec) to radians
    ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
    theta_rad = df["r_half_arcsec"].values * ARCSEC_TO_RAD

    # Calculate angular diameter distance D_A(z) in Mpc
    D_A_Mpc = D_A_LCDM_vectorized(df["redshift"].values, Omega_m, Omega_L)

    # Physical size R = theta * D_A in Mpc, convert to kpc
    R_Mpc = theta_rad * D_A_Mpc
    R_kpc = R_Mpc * 1000.0  # Convert Mpc to kpc
    df["R_kpc"] = R_kpc

    # Calculate statistics
    stats = {
        "mean_kpc": float(np.mean(R_kpc)),
        "median_kpc": float(np.median(R_kpc)),
        "std_kpc": float(np.std(R_kpc)),
        "min_kpc": float(np.min(R_kpc)),
        "max_kpc": float(np.max(R_kpc)),
        "n_galaxies": len(R_kpc),
    }

    # Determine number of histogram bins adaptively
    n_hist_bins = max(10, min(30, int(np.sqrt(len(R_kpc)))))

    # Create figure based on options
    if by_type and "galaxy_type" in df.columns:
        # Separate histograms by galaxy type
        galaxy_types = df["galaxy_type"].dropna().unique()
        n_types = len(galaxy_types)
        if n_types > 1:
            fig, axes = plt.subplots(1, n_types + 1, figsize=(4 * (n_types + 1), 5))
            axes = np.atleast_1d(axes)

            # Overall histogram
            ax = axes[0]
            ax.hist(R_kpc, bins=n_hist_bins, edgecolor="black", alpha=0.7, color="steelblue")
            ax.axvline(stats["median_kpc"], color="red", linestyle="--", linewidth=2,
                       label=f"Median = {stats['median_kpc']:.1f} kpc")
            ax.axvline(stats["mean_kpc"], color="orange", linestyle="-", linewidth=2,
                       label=f"Mean = {stats['mean_kpc']:.1f} kpc")
            ax.set_xlabel(r"Physical Size $R$ (kpc)", fontsize=12)
            ax.set_ylabel(r"Number of galaxies", fontsize=12)
            ax.set_title(f"All Types (N={len(R_kpc)})", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            # Per-type histograms
            colors = plt.cm.tab10(np.linspace(0, 1, n_types))
            for i, gtype in enumerate(sorted(galaxy_types)):
                ax = axes[i + 1]
                mask = df["galaxy_type"] == gtype
                R_type = df.loc[mask, "R_kpc"].values
                if len(R_type) > 0:
                    type_median = np.median(R_type)
                    type_mean = np.mean(R_type)
                    ax.hist(R_type, bins=n_hist_bins, edgecolor="black", alpha=0.7, color=colors[i])
                    ax.axvline(type_median, color="red", linestyle="--", linewidth=2,
                               label=f"Median = {type_median:.1f} kpc")
                    ax.axvline(type_mean, color="orange", linestyle="-", linewidth=2,
                               label=f"Mean = {type_mean:.1f} kpc")
                ax.set_xlabel(r"Physical Size $R$ (kpc)", fontsize=12)
                ax.set_ylabel(r"Number of galaxies", fontsize=12)
                ax.set_title(f"{gtype} (N={len(R_type)})", fontsize=11)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

            plt.suptitle(r"Physical Size Distribution by Galaxy Type" + title_suffix,
                         fontsize=14, fontweight="bold")
            plt.tight_layout()
        else:
            # Only one type, fall through to simple histogram
            by_type = False

    if by_redshift_bin and not by_type:
        # Separate histograms by redshift bin
        z_edges = np.quantile(df["redshift"].values, np.linspace(0, 1, n_redshift_bins + 1))
        z_edges = np.unique(z_edges)  # Remove duplicates
        n_bins_actual = len(z_edges) - 1

        if n_bins_actual > 1:
            fig, axes = plt.subplots(1, n_bins_actual + 1, figsize=(4 * (n_bins_actual + 1), 5))
            axes = np.atleast_1d(axes)

            # Overall histogram
            ax = axes[0]
            ax.hist(R_kpc, bins=n_hist_bins, edgecolor="black", alpha=0.7, color="steelblue")
            ax.axvline(stats["median_kpc"], color="red", linestyle="--", linewidth=2,
                       label=f"Median = {stats['median_kpc']:.1f} kpc")
            ax.axvline(stats["mean_kpc"], color="orange", linestyle="-", linewidth=2,
                       label=f"Mean = {stats['mean_kpc']:.1f} kpc")
            ax.set_xlabel(r"Physical Size $R$ (kpc)", fontsize=12)
            ax.set_ylabel(r"Number of galaxies", fontsize=12)
            ax.set_title(f"All Redshifts (N={len(R_kpc)})", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            # Per-redshift-bin histograms
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_bins_actual))
            for i in range(n_bins_actual):
                ax = axes[i + 1]
                z_lo, z_hi = z_edges[i], z_edges[i + 1]
                mask = (df["redshift"] >= z_lo) & (df["redshift"] < z_hi)
                if i == n_bins_actual - 1:
                    mask = (df["redshift"] >= z_lo) & (df["redshift"] <= z_hi)
                R_bin = df.loc[mask, "R_kpc"].values
                if len(R_bin) > 0:
                    bin_median = np.median(R_bin)
                    bin_mean = np.mean(R_bin)
                    ax.hist(R_bin, bins=n_hist_bins, edgecolor="black", alpha=0.7, color=colors[i])
                    ax.axvline(bin_median, color="red", linestyle="--", linewidth=2,
                               label=f"Median = {bin_median:.1f} kpc")
                    ax.axvline(bin_mean, color="orange", linestyle="-", linewidth=2,
                               label=f"Mean = {bin_mean:.1f} kpc")
                ax.set_xlabel(r"Physical Size $R$ (kpc)", fontsize=12)
                ax.set_ylabel(r"Number of galaxies", fontsize=12)
                ax.set_title(f"$z \\in [{z_lo:.2f}, {z_hi:.2f}]$ (N={len(R_bin)})", fontsize=11)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

            plt.suptitle(r"Physical Size Distribution by Redshift" + title_suffix,
                         fontsize=14, fontweight="bold")
            plt.tight_layout()
        else:
            # Not enough bins, fall through to simple histogram
            by_redshift_bin = False

    if not by_type and not by_redshift_bin:
        # Simple single histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(R_kpc, bins=n_hist_bins, edgecolor="black", alpha=0.7, color="steelblue")
        ax.axvline(stats["median_kpc"], color="red", linestyle="--", linewidth=2,
                   label=f"Median = {stats['median_kpc']:.1f} kpc")
        ax.axvline(stats["mean_kpc"], color="orange", linestyle="-", linewidth=2,
                   label=f"Mean = {stats['mean_kpc']:.1f} kpc")
        ax.set_xlabel(r"Physical Size $R$ (kpc)", fontsize=12)
        ax.set_ylabel(r"Number of galaxies", fontsize=12)
        ax.set_title(
            r"Physical Galaxy Size Distribution" + title_suffix + f"\n(N={len(R_kpc)}, "
            + r"$\Omega_m$=" + f"{Omega_m:.2f})",
            fontsize=14
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

    return stats


def analyze_and_plot_catalog(
    sed_catalog: pd.DataFrame,
    sed_catalog_filtered: pd.DataFrame,
    output_dir: str,
    title_suffix: str = "",
    images: list = None,
    crop_bounds: tuple = None,
    star_mask_centers: list = None,
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
    print(f"\n  Redshift statistics (filtered):")
    print(f"    Min: {sed_catalog_analysis['redshift'].min():.3f}")
    print(f"    Max: {sed_catalog_analysis['redshift'].max():.3f}")
    print(f"    Mean: {sed_catalog_analysis['redshift'].mean():.3f}")
    print(f"    Median: {sed_catalog_analysis['redshift'].median():.3f}")

    # Galaxy type distribution
    print(f"\n  Galaxy type distribution:")
    type_counts = sed_catalog_analysis["galaxy_type"].value_counts()
    for gtype, count in type_counts.items():
        print(f"    {gtype}: {count}")

    # Angular size statistics
    print(f"\n  Angular size statistics (PSF-corrected):")
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

    binned = binned_results.get("Equal Count", list(binned_results.values())[0])

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

    print(f"\n  Fitted parameters:")
    print(f"    Static model: R = {R_static*1000:.2f} kpc")
    print(f"    ΛCDM model:   R = {R_lcdm*1000:.2f} kpc, Ω_m = {Omega_m_fit:.3f}")

    # Model curves
    z_model = np.linspace(0.01, max(2.5, sed_catalog_analysis["redshift"].max() * 1.2), 200)
    theta_static_model = theta_static(z_model, R_static) * RAD_TO_ARCSEC
    theta_lcdm_model = theta_lcdm_flat(z_model, R_lcdm, Omega_m_fit) * RAD_TO_ARCSEC

    # Generate plots
    print(f"\n  Creating plots...")

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
            label=f"Binned (n={[int(x) for x in binned_data['n'].values]})",
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
            f"{name} Binning\n" + r"($\Omega_m$" + f"={Omega_m_i:.2f}, " + r"$R_{\Lambda \rm CDM}$" + f"={R_lcdm_i:.4f} Mpc)", fontsize=12
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

    fig.suptitle(r"Comparison of Binning Strategies" + title_suffix, fontsize=14, fontweight="bold")
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

    # Plot 5: Redshift histogram
    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sed_catalog_analysis["redshift"], bins=20, edgecolor="black", alpha=0.7)
    ax.set_xlabel(r"Redshift $z$", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Redshift Distribution{title_suffix}", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/redshift_histogram.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_dir}/redshift_histogram.pdf")

    # Plot 5b: Photo-z quality metrics (ODDS and chi-squared distributions)
    plot_photoz_quality(sed_catalog_analysis, output_dir, title_suffix)

    # Plot 6: Galaxy types
    _fig, ax = plt.subplots(figsize=(10, 6))
    type_counts.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Galaxy Type", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Galaxy Type Distribution{title_suffix}", fontsize=14)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/galaxy_types.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_dir}/galaxy_types.pdf")

    # Plot 6b: Color-color diagram
    plot_color_color_diagram(
        sed_catalog_analysis,
        f"{output_dir}/color_color_diagram.pdf",
        title_suffix=title_suffix,
        show_templates=True,
        spectra_path="./spectra",
    )
    print(f"    Saved: {output_dir}/color_color_diagram.pdf")

    # Plot 7: Angular diameter distance
    _fig, ax = plt.subplots(figsize=(10, 7))
    z_plot = np.linspace(0.01, 3.0, 200)
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
        n_detected_stars = is_detected_star.sum()
        n_mask_stars = is_in_mask.sum()

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
            # Blue: detected stars
            if n_detected_stars > 0:
                ax.scatter(
                    plot_catalog.loc[is_detected_star, "xcentroid"],
                    plot_catalog.loc[is_detected_star, "ycentroid"],
                    s=10, facecolors="none", edgecolors="blue", linewidth=0.5,
                )
            # Yellow: mask stars
            if n_mask_stars > 0:
                ax.scatter(
                    plot_catalog.loc[is_in_mask, "xcentroid"],
                    plot_catalog.loc[is_in_mask, "ycentroid"],
                    s=80, facecolors="none", edgecolors="yellow", linewidth=2.0,
                )

            ax.set_title(
                f"Band {image.band.upper()} -- {n_galaxies} galaxies (red), "
                f"{n_detected_stars} detected stars (blue), {n_mask_stars} mask stars (yellow)",
                fontsize=9,
            )
            ax.set_xlabel(r"X (pixels)")
            ax.set_ylabel(r"Y (pixels)")

        plt.suptitle(f"Source Detection{title_suffix} (red=galaxies, blue=detected stars, yellow=mask stars)", fontsize=14, fontweight="bold", y=1.01)
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
            if n_detected_stars > 0:
                ax.scatter(
                    plot_catalog.loc[is_detected_star, "xcentroid"],
                    plot_catalog.loc[is_detected_star, "ycentroid"],
                    s=10, facecolors="none", edgecolors="blue", linewidth=0.5,
                )
            if n_mask_stars > 0:
                ax.scatter(
                    plot_catalog.loc[is_in_mask, "xcentroid"],
                    plot_catalog.loc[is_in_mask, "ycentroid"],
                    s=80, facecolors="none", edgecolors="yellow", linewidth=2.0,
                )

            # Draw markers at ALL mask region centers (regardless of detection)
            if star_mask_centers:
                # Adjust for crop bounds if needed
                adjusted_centers = []
                for cx, cy in star_mask_centers:
                    if crop_bounds is not None:
                        x_min, x_max, y_min, y_max = crop_bounds
                        # Only include centers within crop bounds
                        if x_min <= cx <= x_max and y_min <= cy <= y_max:
                            adjusted_centers.append((cx - x_min, cy - y_min))
                    else:
                        adjusted_centers.append((cx, cy))
                if adjusted_centers:
                    mask_x = [c[0] for c in adjusted_centers]
                    mask_y = [c[1] for c in adjusted_centers]
                    ax.scatter(mask_x, mask_y, s=200, marker='s', facecolors="none",
                              edgecolors="lime", linewidth=2.5, label="Mask regions")

            ax.set_title(
                f"Band {image.band.upper()} -- {n_galaxies} galaxies (red), "
                f"{n_detected_stars} detected stars (blue), {len(star_mask_centers)} mask regions (green sq)",
                fontsize=9,
            )
            ax.set_xlabel(r"X (pixels)")
            ax.set_ylabel(r"Y (pixels)")

        plt.suptitle(f"Final Selection{title_suffix} (red=galaxies, blue=stars, green sq=mask regions)", fontsize=14, fontweight="bold", y=1.01)
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


def main(mode: str = "full"):
    """
    Run the analysis pipeline.

    Args:
        mode: "full" for entire 4096x4096 image, "chip3" for 2048x2048 chip3 only
    """
    # Set output directory based on mode
    output_dir = f"./output/{mode}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("ANGULAR SIZE TEST ANALYSIS")
    print(f"Mode: {mode}")
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
            print(f"  No mask file found - using no mask")
    else:
        # For 2048x2048 data, don't mask during detection - we'll flag stars instead
        # This allows sources in star regions to be detected and flagged
        mask = np.zeros(image_shape, dtype=bool)
        print(f"  Detection mask: none (star flagging enabled via star_mask)")

    # Step 2: Source detection using adaptive PSF-based parameters
    print("\n[2/6] Detecting sources (adaptive PSF-based method)...")

    # Hubble ACS/WFC PSF parameters
    # PSF FWHM ~0.08-0.10 arcsec for optical bands
    PSF_FWHM_ARCSEC = 0.09  # Typical HST ACS PSF FWHM
    PSF_FWHM_PIX = PSF_FWHM_ARCSEC / PIXEL_SCALE  # ~2.25 pixels at 0.04"/pix

    # Adaptive kernel FWHM: Use larger kernel to enhance extended sources (galaxies)
    # Using 2x PSF FWHM for stricter galaxy detection (filters point sources)
    KERNEL_FWHM_PIX = max(4.0, PSF_FWHM_PIX * 2.0)

    # Adaptive npixels based on PSF area: π × (FWHM/2)²
    # STRICT: Use 5x PSF area to require substantial extended emission
    PSF_AREA = np.pi * (PSF_FWHM_PIX / 2) ** 2
    BASE_NPIXELS = int(np.ceil(PSF_AREA * 5.0))  # ~20 pixels - stricter filtering

    # Scale npixels with image resolution (larger images have more pixels per source)
    def get_adaptive_npixels(image_shape, base_npixels):
        """Calculate adaptive npixels based on image resolution."""
        # Scale factor: 1.0 for 2048, 2.0 for 4096
        scale = image_shape[0] / 2048.0
        return int(base_npixels * scale)

    # Detection threshold: sigma above background RMS
    # STRICT: Higher threshold requires stronger signal
    DETECTION_SIGMA = 2.5  # Higher threshold filters faint/uncertain sources

    # Deblending parameters for separating overlapping galaxies
    NLEVELS = 32  # Multi-thresholding levels
    CONTRAST = 0.01  # Higher contrast = more conservative deblending (fewer splits)

    print(f"  PSF FWHM: {PSF_FWHM_ARCSEC:.3f} arcsec ({PSF_FWHM_PIX:.2f} pixels)")
    print(f"  Kernel FWHM: {KERNEL_FWHM_PIX:.2f} pixels")
    print(f"  Base npixels: {BASE_NPIXELS} (PSF area × 5.0 - strict)")
    print(f"  Detection threshold: {DETECTION_SIGMA}σ above background")
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
        except Exception:
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

    for idx in range(len(ref_cat)):
        ref_x, ref_y = ref_coords[idx]
        ref_row = ref_cat.iloc[idx]
        matches = {"xcentroid": ref_x, "ycentroid": ref_y}
        r_half_values = [ref_row["r_half_pix"]]
        r_half_pix_errors = [ref_row["r_half_pix_error"]]
        concentration_values = [ref_row["concentration"]]
        snr_values = [ref_row["snr"]]
        ellipticity_values = [ref_row["ellipticity"]]

        matches[f"source_sky_{reference_band}"] = ref_row["source_sky"]
        matches[f"source_error_{reference_band}"] = ref_row["source_error"]

        found_in_all_bands = True
        for band in other_bands:
            # Query KD-tree for nearest neighbor
            dist, nearest_idx = kdtrees[band].query([ref_x, ref_y], k=1)

            if dist < match_radius:
                cat = band_catalogs[band]
                matches[f"source_sky_{band}"] = cat.iloc[nearest_idx]["source_sky"]
                matches[f"source_error_{band}"] = cat.iloc[nearest_idx]["source_error"]
                r_half_values.append(cat.iloc[nearest_idx]["r_half_pix"])
                r_half_pix_errors.append(cat.iloc[nearest_idx]["r_half_pix_error"])
                concentration_values.append(cat.iloc[nearest_idx]["concentration"])
                snr_values.append(cat.iloc[nearest_idx]["snr"])
                ellipticity_values.append(cat.iloc[nearest_idx]["ellipticity"])
            else:
                found_in_all_bands = False
                break

        if found_in_all_bands:
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
            # Flag if stellarity > 0.6 (likely star) OR concentration > 0.45 (high concentration)
            # References: SExtractor CLASS_STAR, SDSS star/galaxy separation
            if matches["stellarity"] > 0.6 or matches["concentration"] > 0.45:
                flag |= FLAG_PSF_LIKE

            # Check if source is inside star mask OR near a mask region center
            # Sources in/near these regions are forced to be classified as stars
            # This handles cases where cross-matching offsets cause misalignment
            STAR_MASK_RADIUS = 30  # pixels - flag sources within this radius of mask centers
            matches["in_star_mask"] = False

            if star_mask is not None:
                # Method 1: Check if pixel is directly in mask
                px, py = int(round(ref_x)), int(round(ref_y))
                if 0 <= py < star_mask.shape[0] and 0 <= px < star_mask.shape[1]:
                    if star_mask[py, px]:
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
    print(f"  Cross-matched: {len(cross_matched_catalog)} sources in all 4 bands")

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
        star_mask_sources = cross_matched_catalog["in_star_mask"] == True
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

    # Force garbage collection before heavy computation
    gc.collect()

    # Classify galaxies in parallel using batched processing with PDF uncertainties
    # Use fewer workers to reduce memory pressure (each worker needs ~50-100MB)
    n_workers = min(os.cpu_count() or 6, 6)  # Limit to 6 workers
    print(f"  Classifying galaxies in parallel ({n_workers} workers, with PDF uncertainties)...")

    # Prepare data for batch processing (vectorized approach - avoids slow iterrows)
    # Extract flux and error arrays directly from DataFrame
    flux_arrays = {band: sed_catalog[f"flux_{band}"].to_numpy() for band in ["b", "i", "u", "v"]}
    error_arrays = {band: sed_catalog[f"error_{band}"].to_numpy() for band in ["b", "i", "u", "v"]}
    indices = sed_catalog.index.to_numpy()

    # Build galaxy_data list using numpy indexing (faster than iterrows)
    galaxy_data = [
        (
            indices[i],
            [flux_arrays["b"][i], flux_arrays["i"][i], flux_arrays["u"][i], flux_arrays["v"][i]],
            [error_arrays["b"][i], error_arrays["i"][i], error_arrays["u"][i], error_arrays["v"][i]],
        )
        for i in range(len(indices))
    ]

    # Split into smaller batches to reduce peak memory usage
    # Each galaxy uses ~30KB for chi-sq grid, so 100 galaxies = 3MB per batch
    n_galaxies = len(galaxy_data)
    MAX_BATCH_SIZE = 100  # Limit batch size for memory efficiency
    batch_size = min(MAX_BATCH_SIZE, max(1, n_galaxies // n_workers))
    batches = []
    for i in range(0, n_galaxies, batch_size):
        batch = galaxy_data[i:i + batch_size]
        batches.append((batch, "./spectra"))

    # Run batched classification in parallel
    # New format: (idx, galaxy_type, redshift, z_lo, z_hi, chi_sq_min, odds)
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_classify_galaxy_batch, batch): i for i, batch in enumerate(batches)}
        completed_batches = 0
        for future in as_completed(futures):
            batch_results = future.result()
            for idx, galaxy_type, redshift, z_lo, z_hi, chi_sq_min, odds in batch_results:
                results[idx] = (galaxy_type, redshift, z_lo, z_hi, chi_sq_min, odds)
            completed_batches += 1
            print(f"    Completed batch {completed_batches}/{len(batches)} ({len(results)}/{n_galaxies} galaxies)")

    # Update catalog with results using vectorized operations (much faster than iterrows)
    # Convert results dict to arrays aligned with sed_catalog index
    n_rows = len(sed_catalog)
    galaxy_types = [None] * n_rows
    redshifts = np.empty(n_rows)
    z_los = np.empty(n_rows)
    z_his = np.empty(n_rows)
    chi_sq_mins = np.empty(n_rows)
    odds_arr = np.empty(n_rows)

    for i, idx in enumerate(sed_catalog.index):
        galaxy_type, redshift, z_lo, z_hi, chi_sq_min, odds = results[idx]
        galaxy_types[i] = galaxy_type
        redshifts[i] = redshift
        z_los[i] = z_lo
        z_his[i] = z_hi
        chi_sq_mins[i] = chi_sq_min
        odds_arr[i] = odds

    # Assign all columns at once (vectorized)
    sed_catalog["redshift"] = redshifts
    sed_catalog["redshift_lo"] = z_los  # 16th percentile
    sed_catalog["redshift_hi"] = z_his  # 84th percentile
    sed_catalog["redshift_err"] = (z_his - z_los) / 2.0  # Symmetric error estimate
    sed_catalog["chi_sq_min"] = chi_sq_mins
    sed_catalog["photo_z_odds"] = odds_arr  # Photo-z quality (BPZ ODDS parameter)
    sed_catalog["galaxy_type"] = galaxy_types

    # Use PSF-corrected radius for angular size (vectorized)
    sed_catalog["r_half_arcsec"] = sed_catalog["r_half_pix_corrected"] * PIXEL_SCALE
    sed_catalog["r_half_arcsec_error"] = sed_catalog["r_half_pix_error"] * PIXEL_SCALE

    # Update quality flag for bad photo-z (ODDS < 0.6) - vectorized
    bad_photoz_mask = odds_arr < 0.6
    sed_catalog.loc[bad_photoz_mask, "quality_flag"] = (
        sed_catalog.loc[bad_photoz_mask, "quality_flag"].astype(int) | FLAG_BAD_PHOTOZ
    )

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

    # Option to filter by quality (comment/uncomment as needed)
    # For strict analysis, use only clean sources:
    # sed_catalog = sed_catalog[sed_catalog["quality_flag"] == FLAG_NONE]
    # For moderate filtering, exclude bad photo-z, PSF-like, and masked (known stars):
    EXCLUDE_FLAGS = FLAG_BAD_PHOTOZ | FLAG_PSF_LIKE | FLAG_MASKED
    sed_catalog_filtered = sed_catalog[
        (sed_catalog["quality_flag"].astype(int) & EXCLUDE_FLAGS) == 0
    ]
    print(f"\n  After quality filtering (excluding bad_photoz, psf_like, masked): {len(sed_catalog_filtered)}")

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
            label=f"Binned (n={[int(x) for x in binned_data['n'].values]})",
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
            f"{name} Binning\n" + r"($\Omega_m$" + f"={Omega_m_i:.2f}, " + r"$R_{\Lambda \rm CDM}$" + f"={R_lcdm_i:.4f} Mpc)", fontsize=12
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

    fig.suptitle(r"Comparison of Binning Strategies", fontsize=14, fontweight="bold")
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

    # Plot 5: Redshift histogram
    _fig, ax = plt.subplots(figsize=(8, 5))
    hist_bins = np.linspace(z_data_min, z_data_max, 15)
    ax.hist(sed_catalog_analysis["redshift"], bins=hist_bins, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel(r"Redshift $z$", fontsize=12)
    ax.set_ylabel(r"Number of galaxies", fontsize=12)
    ax.set_title(f"Redshift Distribution (N={len(sed_catalog_analysis)}, quality-filtered)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(z_lim)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/redshift_histogram.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/redshift_histogram.pdf")

    # Plot 5b: Photo-z quality metrics (ODDS and chi-squared distributions)
    plot_photoz_quality(sed_catalog_analysis, output_dir)

    # Plot 6: Galaxy type distribution
    __fig, ax = plt.subplots(figsize=(10, 5))
    type_counts = sed_catalog_analysis["galaxy_type"].value_counts()
    type_counts.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.7)
    ax.set_xlabel(r"Galaxy Type", fontsize=12)
    ax.set_ylabel(r"Count", fontsize=12)
    ax.set_title(r"Galaxy Type Distribution (quality-filtered)", fontsize=14)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/galaxy_types.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/galaxy_types.pdf")

    # Plot 6b: Color-color diagram
    plot_color_color_diagram(
        sed_catalog_analysis,
        f"{output_dir}/color_color_diagram.pdf",
        title_suffix="",
        show_templates=True,
        spectra_path="./spectra",
    )
    print(f"  Saved: {output_dir}/color_color_diagram.pdf")

    # Plot 6c: Physical size distribution
    # Use the fitted Omega_m from the LCDM model
    physical_size_stats = plot_physical_size_distribution(
        sed_catalog_analysis,
        f"{output_dir}/physical_size_distribution.pdf",
        Omega_m=Omega_m_fit,
        Omega_L=1.0 - Omega_m_fit,
        title_suffix="",
        by_type=False,
        by_redshift_bin=False,
    )
    print(f"  Physical size statistics:")
    print(f"    Mean: {physical_size_stats.get('mean_kpc', 0):.2f} kpc")
    print(f"    Median: {physical_size_stats.get('median_kpc', 0):.2f} kpc")
    print(f"    Std: {physical_size_stats.get('std_kpc', 0):.2f} kpc")

    # Plot 7: Angular diameter distance comparison
    __fig, ax = plt.subplots(figsize=(10, 7))

    # Extend z_range slightly beyond data for context
    z_range = np.linspace(0.01, z_data_max + 0.3, 200)
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
    ax.set_xlim(0, z_data_max + 0.3)

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

    n_mask_stars = is_in_mask.sum()
    n_detected_stars = is_detected_star.sum()
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

        # Blue: stars detected by stellarity/concentration (not in mask)
        if n_detected_stars > 0:
            ax.scatter(
                cross_matched_catalog.loc[is_detected_star, "xcentroid"],
                cross_matched_catalog.loc[is_detected_star, "ycentroid"],
                s=10,
                facecolors="none",
                edgecolors="blue",
                linewidth=0.5,
            )

        # Yellow: sources inside mask (stars by mask from course staff)
        if n_mask_stars > 0:
            ax.scatter(
                cross_matched_catalog.loc[is_in_mask, "xcentroid"],
                cross_matched_catalog.loc[is_in_mask, "ycentroid"],
                s=80,
                facecolors="none",
                edgecolors="yellow",
                linewidth=2.0,
            )

        # Green squares: ALL mask region centers (regardless of detection)
        if star_mask_centers:
            mask_x = [c[0] for c in star_mask_centers]
            mask_y = [c[1] for c in star_mask_centers]
            ax.scatter(mask_x, mask_y, s=200, marker='s', facecolors="none",
                      edgecolors="lime", linewidth=2.5)

        ax.set_title(
            f"Band {image.band.upper()} -- {n_galaxies} galaxies (red), "
            f"{n_detected_stars} detected stars (blue), {len(star_mask_centers)} mask regions (green sq)",
            fontsize=9,
        )
        ax.set_xlabel(r"X (pixels)")
        ax.set_ylabel(r"Y (pixels)")

    plt.suptitle("Source Detection (red=galaxies, blue=stars, green sq=mask regions)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/source_detection.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/source_detection.pdf")

    # Plot 9: Final selection (cross-matched sources in all 4 bands)
    # Uses same categories as Plot 8: yellow=mask stars, blue=detected stars, red=galaxies

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

        # Blue: stars detected by stellarity/concentration (not in mask)
        if n_detected_stars > 0:
            ax.scatter(
                cross_matched_catalog.loc[is_detected_star, "xcentroid"],
                cross_matched_catalog.loc[is_detected_star, "ycentroid"],
                s=10,
                facecolors="none",
                edgecolors="blue",
                linewidth=0.5,
            )

        # Yellow: sources inside mask (stars by mask from course staff)
        if n_mask_stars > 0:
            ax.scatter(
                cross_matched_catalog.loc[is_in_mask, "xcentroid"],
                cross_matched_catalog.loc[is_in_mask, "ycentroid"],
                s=80,
                facecolors="none",
                edgecolors="yellow",
                linewidth=2.0,
            )

        # Green squares: ALL mask region centers (regardless of detection)
        if star_mask_centers:
            mask_x = [c[0] for c in star_mask_centers]
            mask_y = [c[1] for c in star_mask_centers]
            ax.scatter(mask_x, mask_y, s=200, marker='s', facecolors="none",
                      edgecolors="lime", linewidth=2.5)

        ax.set_title(
            f"Band {image.band.upper()} -- {n_galaxies} galaxies (red), "
            f"{n_detected_stars} detected stars (blue), {len(star_mask_centers)} mask regions (green sq)",
            fontsize=9,
        )
        ax.set_xlabel(r"X (pixels)")
        ax.set_ylabel(r"Y (pixels)")

    plt.suptitle("Final Selection (red=galaxies, blue=stars, green sq=mask regions)", fontsize=14, fontweight="bold", y=1.01)
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


def plot_example_seds(
    catalog: pd.DataFrame,
    output_dir: str = "./output/full",
    spectra_path: str = "./spectra",
) -> None:
    """
    Plot example SED fits for representative galaxies.

    Creates a 2x2 subplot showing observed photometry and best-fit template SEDs
    for 3-4 representative galaxies of different types (elliptical/S0, spiral, starburst).

    Parameters
    ----------
    catalog : pd.DataFrame
        Galaxy catalog with columns: galaxy_type, redshift, photo_z_odds,
        flux_u, flux_b, flux_v, flux_i, error_u, error_b, error_v, error_i
    output_dir : str
        Output directory for saving the plot
    spectra_path : str
        Path to directory containing template spectra files
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from classify import (
        classify_galaxy_with_pdf,
        _load_spectrum,
        _WL_GRID,
        igm_absorption,
    )

    os.makedirs(output_dir, exist_ok=True)

    # Filter bandpass definitions (from classify.py)
    # U, B, V, I bands with their effective wavelengths and widths
    filter_info = {
        'U': {'center': 3000, 'width': 1521, 'color': 'purple'},
        'B': {'center': 4500, 'width': 1501, 'color': 'blue'},
        'V': {'center': 6060, 'width': 951, 'color': 'green'},
        'I': {'center': 8140, 'width': 766, 'color': 'red'},
    }

    # Define representative galaxy types to select
    # Priority: elliptical/S0 (early-type), Sa/Sb (spiral), sbt1-6 (starburst)
    target_types = [
        ('elliptical', 'S0'),           # Early-type (elliptical or S0)
        ('Sa', 'Sb'),                   # Spiral
        ('sbt1', 'sbt2', 'sbt3'),       # Starburst (intense)
        ('sbt4', 'sbt5', 'sbt6'),       # Starburst (moderate)
    ]

    # Select representative galaxies with good ODDS values
    selected_galaxies = []
    min_odds = 0.8  # Require good photo-z quality

    for type_group in target_types:
        # Find galaxies of this type with good ODDS
        mask = catalog['galaxy_type'].isin(type_group) & (catalog['photo_z_odds'] >= min_odds)
        candidates = catalog[mask].copy()

        if len(candidates) == 0:
            # Try with lower ODDS threshold
            mask = catalog['galaxy_type'].isin(type_group) & (catalog['photo_z_odds'] >= 0.6)
            candidates = catalog[mask].copy()

        if len(candidates) > 0:
            # Sort by ODDS (best first) and select the best one
            candidates = candidates.sort_values('photo_z_odds', ascending=False)
            selected_galaxies.append(candidates.iloc[0])

        if len(selected_galaxies) >= 4:
            break

    # If we don't have 4 galaxies, fill with any good galaxies
    if len(selected_galaxies) < 4:
        remaining = catalog[catalog['photo_z_odds'] >= min_odds].copy()
        remaining = remaining.sort_values('photo_z_odds', ascending=False)
        for _, row in remaining.iterrows():
            if len(selected_galaxies) >= 4:
                break
            # Check if this galaxy type is already represented
            already_have = [g['galaxy_type'] for g in selected_galaxies]
            if row['galaxy_type'] not in already_have:
                selected_galaxies.append(row)

    if len(selected_galaxies) == 0:
        print("  Warning: No suitable galaxies found for SED plot")
        return

    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, galaxy in enumerate(selected_galaxies[:4]):
        ax = axes[idx]

        # Extract galaxy properties
        gtype = galaxy['galaxy_type']
        z_best = galaxy['redshift']
        odds = galaxy['photo_z_odds']

        # Get observed photometry (in flux units)
        obs_flux = np.array([
            galaxy['flux_u'],
            galaxy['flux_b'],
            galaxy['flux_v'],
            galaxy['flux_i'],
        ])
        obs_err = np.array([
            galaxy['error_u'],
            galaxy['error_b'],
            galaxy['error_v'],
            galaxy['error_i'],
        ])

        # Filter effective wavelengths (observer frame)
        filter_wavelengths = np.array([3000, 4500, 6060, 8140])  # U, B, V, I

        # Load the best-fit template spectrum
        try:
            wl_template, spec_template = _load_spectrum(spectra_path, gtype)
        except FileNotFoundError:
            # Try base type if interpolated name
            base_type = gtype.split('_')[0] if '_' in gtype else gtype
            wl_template, spec_template = _load_spectrum(spectra_path, base_type)

        # Redshift the template to observer frame
        wl_redshifted = wl_template * (1 + z_best)

        # Apply IGM absorption for high-z sources
        if z_best > 0.5:
            igm_trans = igm_absorption(wl_template, z_best, model="madau95")
            spec_with_igm = spec_template * igm_trans
        else:
            spec_with_igm = spec_template

        # Interpolate template to common grid for synthetic photometry
        spec_interp = np.interp(_WL_GRID, wl_redshifted, spec_with_igm, left=0, right=0)

        # Compute synthetic photometry to normalize template to data
        syn_flux = []
        for band_name in ['U', 'B', 'V', 'I']:
            info = filter_info[band_name]
            center, width = info['center'], info['width'] / 2
            fmask = (_WL_GRID >= center - width) & (_WL_GRID <= center + width)
            syn_flux.append(np.mean(spec_interp[fmask]) if np.any(fmask) else 0)
        syn_flux = np.array(syn_flux)

        # Normalize template to match observed data (median normalization)
        obs_median = np.median(obs_flux[obs_flux > 0])
        syn_median = np.median(syn_flux[syn_flux > 0])
        if syn_median > 0 and obs_median > 0:
            norm_factor = obs_median / syn_median
        else:
            norm_factor = 1.0

        spec_normalized = spec_with_igm * norm_factor

        # Plot the normalized template SED
        valid_mask = (wl_redshifted >= 2000) & (wl_redshifted <= 10000) & (spec_normalized > 0)
        if np.any(valid_mask):
            ax.plot(
                wl_redshifted[valid_mask],
                spec_normalized[valid_mask],
                '-', color='gray', linewidth=1.5, alpha=0.8,
                label=f'Best-fit template ({gtype})'
            )

        # Plot observed photometry with error bars
        band_names = ['U', 'B', 'V', 'I']
        for i, (wl, flux, err, band_name) in enumerate(zip(filter_wavelengths, obs_flux, obs_err, band_names)):
            color = filter_info[band_name]['color']
            ax.errorbar(
                wl, flux, yerr=err,
                fmt='o', markersize=10, capsize=4, capthick=1.5,
                color=color, markeredgecolor='black', markeredgewidth=0.8,
                label=f'{band_name} band' if idx == 0 else None,
                zorder=10
            )

        # Add filter transmission curves (scaled and offset for visibility)
        flux_range = np.max(obs_flux) - np.min(obs_flux[obs_flux > 0])
        baseline = np.min(obs_flux[obs_flux > 0]) * 0.3
        for band_name in ['U', 'B', 'V', 'I']:
            info = filter_info[band_name]
            center, width = info['center'], info['width'] / 2
            # Simple box filter representation
            filter_wl = np.array([center - width, center - width, center + width, center + width])
            filter_trans = np.array([0, 1, 1, 0]) * flux_range * 0.15 + baseline
            ax.fill(filter_wl, filter_trans, color=info['color'], alpha=0.15)

        # Create inset for P(z) PDF
        ax_inset = inset_axes(ax, width="35%", height="30%", loc='upper right')

        # Recompute PDF for this galaxy to get z_grid and pdf
        fluxes = [galaxy['flux_b'], galaxy['flux_i'], galaxy['flux_u'], galaxy['flux_v']]
        errors = [galaxy['error_b'], galaxy['error_i'], galaxy['error_u'], galaxy['error_v']]
        try:
            result = classify_galaxy_with_pdf(fluxes, errors, spectra_path=spectra_path)
            z_grid = result.z_grid
            pdf = result.pdf

            # Plot PDF
            ax_inset.fill_between(z_grid, pdf, alpha=0.5, color='steelblue')
            ax_inset.plot(z_grid, pdf, '-', color='steelblue', linewidth=1)
            ax_inset.axvline(z_best, color='red', linestyle='--', linewidth=1.5, label=f'$z$={z_best:.2f}')

            # Mark 68% confidence interval
            z_lo, z_hi = result.z_lo, result.z_hi
            ax_inset.axvspan(z_lo, z_hi, alpha=0.2, color='red')

            ax_inset.set_xlabel(r'$z$', fontsize=8)
            ax_inset.set_ylabel(r'$P(z)$', fontsize=8)
            ax_inset.set_xlim(0, min(3.5, z_best + 1.5))
            ax_inset.tick_params(axis='both', which='major', labelsize=7)
            ax_inset.set_title(r'$P(z)$ PDF', fontsize=8)
        except Exception:
            ax_inset.text(0.5, 0.5, 'PDF unavailable', transform=ax_inset.transAxes,
                         ha='center', va='center', fontsize=8)

        # Labels and title
        ax.set_xlabel(r'Wavelength (\AA)', fontsize=11)
        ax.set_ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)', fontsize=11)

        # Format galaxy type for display
        type_display = {
            'elliptical': 'Elliptical',
            'S0': 'S0 (Lenticular)',
            'Sa': 'Sa Spiral',
            'Sb': 'Sb Spiral',
            'sbt1': 'Starburst 1',
            'sbt2': 'Starburst 2',
            'sbt3': 'Starburst 3',
            'sbt4': 'Starburst 4',
            'sbt5': 'Starburst 5',
            'sbt6': 'Starburst 6',
        }.get(gtype, gtype)

        ax.set_title(
            f'{type_display}\n' + r'$z_{\rm phot}$' + f' = {z_best:.3f}, ODDS = {odds:.2f}',
            fontsize=11
        )

        ax.set_xlim(2000, 10000)
        ax.grid(True, alpha=0.3)

        # Set y-axis to show positive fluxes only
        ymin = min(0, np.min(obs_flux - obs_err) * 0.9)
        ymax = np.max(obs_flux + obs_err) * 1.3
        ax.set_ylim(ymin, ymax)

        # Use scientific notation for y-axis
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))

    # Add legend to first subplot only
    if len(selected_galaxies) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        # Filter unique labels
        unique = {}
        for h, l in zip(handles, labels):
            if l not in unique:
                unique[l] = h
        axes[0].legend(unique.values(), unique.keys(), loc='upper left', fontsize=8)

    # Hide unused subplots
    for idx in range(len(selected_galaxies), 4):
        axes[idx].set_visible(False)

    fig.suptitle(r'Example SED Fits for Representative HDF Galaxies', fontsize=14, fontweight='bold')
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.25)
    plt.savefig(f"{output_dir}/example_seds.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/example_seds.pdf")


def plot_galaxy_type_vs_redshift(
    catalog: pd.DataFrame,
    output_dir: str,
    title_suffix: str = "",
    n_bins: int = None,
) -> None:
    """
    Create a visualization showing the distribution of galaxy types as a function of redshift.

    This plot reveals:
    - Which galaxy types dominate at different redshifts
    - Selection effects (e.g., are starbursts preferentially detected at high-z?)
    - The evolution of the galaxy population with cosmic time

    Uses a stacked bar chart with both absolute counts and fractional representation,
    plus annotations showing selection effects and population statistics.

    Parameters
    ----------
    catalog : pd.DataFrame
        Galaxy catalog with 'galaxy_type', 'z' or 'redshift', and optionally 'r_half_arcsec' columns
    output_dir : str
        Directory to save the output plot
    title_suffix : str, optional
        Suffix to add to plot title (e.g., " (Chip 3)")
    n_bins : int, optional
        Number of redshift bins. If None, uses adaptive binning based on sample size.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Handle both 'z' and 'redshift' column names
    z_col = 'z' if 'z' in catalog.columns else 'redshift'
    if z_col not in catalog.columns:
        print(f"  Warning: No redshift column found in catalog. Skipping galaxy type vs redshift plot.")
        return

    # Filter to valid data
    valid_mask = catalog[z_col].notna() & catalog['galaxy_type'].notna()
    data = catalog[valid_mask].copy()

    if len(data) < 10:
        print(f"  Warning: Only {len(data)} galaxies with valid type and redshift. Skipping plot.")
        return

    # Determine number of bins adaptively
    if n_bins is None:
        n_bins = get_adaptive_n_bins(len(data), min_per_bin=3)
        n_bins = min(n_bins, 10)  # Cap at 10 bins for readability

    # Create redshift bins
    z_min, z_max = data[z_col].min(), data[z_col].max()
    z_edges = np.linspace(z_min, z_max, n_bins + 1)
    data['z_bin'] = pd.cut(data[z_col], bins=z_edges, include_lowest=True)

    # Get unique galaxy types and sort them logically
    # Group starburst types together, then spirals, then ellipticals
    all_types = data['galaxy_type'].unique()

    def type_sort_key(t):
        """Sort galaxy types: elliptical/S0 first, then spirals, then starbursts."""
        t_lower = t.lower()
        if t_lower == 'elliptical':
            return (0, t)
        elif t_lower == 's0':
            return (1, t)
        elif t_lower.startswith('s') and t_lower[1:].isalpha():  # Sa, Sb, Sc, etc.
            return (2, t)
        elif t_lower.startswith('sbt'):  # Starburst types
            return (3, t)
        else:
            return (4, t)

    sorted_types = sorted(all_types, key=type_sort_key)

    # Create pivot table for counts
    pivot_counts = data.groupby(['z_bin', 'galaxy_type'], observed=False).size().unstack(fill_value=0)

    # Reorder columns by sorted types (only include types present in data)
    pivot_counts = pivot_counts[[t for t in sorted_types if t in pivot_counts.columns]]

    # Calculate fractions for the second panel
    pivot_fractions = pivot_counts.div(pivot_counts.sum(axis=1), axis=0).fillna(0)

    # Get bin centers for x-axis
    bin_centers = [(interval.left + interval.right) / 2 for interval in pivot_counts.index]

    # Define color palette - using tab10 extended for more types
    n_types = len(pivot_counts.columns)
    if n_types <= 10:
        cmap = plt.cm.tab10
        colors = [cmap(i) for i in range(n_types)]
    else:
        cmap = plt.cm.tab20
        colors = [cmap(i) for i in range(n_types)]

    type_colors = dict(zip(pivot_counts.columns, colors))

    # Create figure with two panels: counts and fractions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.08})

    # Panel 1: Stacked bar chart (counts)
    bar_width = (z_max - z_min) / n_bins * 0.85
    bottoms = np.zeros(len(bin_centers))

    for gtype in pivot_counts.columns:
        counts = pivot_counts[gtype].values
        ax1.bar(bin_centers, counts, bar_width, bottom=bottoms,
                label=gtype, color=type_colors[gtype], edgecolor='white', linewidth=0.5)
        bottoms += counts

    ax1.set_ylabel(r'Number of galaxies', fontsize=12)
    ax1.set_title(r'Galaxy Type Distribution vs Redshift' + title_suffix, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, ncol=min(3, (n_types + 2) // 3),
               title='Galaxy Type', title_fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add total count annotations on top of each bar
    for i, (x, total) in enumerate(zip(bin_centers, pivot_counts.sum(axis=1).values)):
        if total > 0:
            ax1.annotate(f'{int(total)}', xy=(x, total), ha='center', va='bottom',
                        fontsize=8, color='black')

    # Panel 2: Stacked bar chart (fractions)
    bottoms_frac = np.zeros(len(bin_centers))

    for gtype in pivot_fractions.columns:
        fracs = pivot_fractions[gtype].values
        ax2.bar(bin_centers, fracs, bar_width, bottom=bottoms_frac,
                color=type_colors[gtype], edgecolor='white', linewidth=0.5)
        bottoms_frac += fracs

    ax2.set_xlabel(r'Redshift $z$', fontsize=12)
    ax2.set_ylabel(r'Fraction', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add horizontal lines at key fractions
    for frac in [0.25, 0.5, 0.75]:
        ax2.axhline(frac, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    # Set x-axis limits with padding
    z_padding = (z_max - z_min) * 0.05
    ax2.set_xlim(z_min - z_padding, z_max + z_padding)

    # Add redshift bin edge labels
    ax2.set_xticks(bin_centers)
    bin_labels = [f'{bc:.2f}' for bc in bin_centers]
    ax2.set_xticklabels(bin_labels, rotation=45, ha='right')

    # Calculate and display selection effect statistics
    # Check if starburst types dominate at high-z
    starburst_types = [t for t in sorted_types if t.lower().startswith('sbt')]
    early_types = [t for t in sorted_types if t.lower() in ['elliptical', 's0']]

    if len(bin_centers) >= 2:
        z_median = np.median(data[z_col])

        # Low-z vs high-z fractions
        low_z_mask = data[z_col] < z_median
        high_z_mask = data[z_col] >= z_median

        low_z_count = low_z_mask.sum()
        high_z_count = high_z_mask.sum()

        if low_z_count > 0 and high_z_count > 0:
            # Starburst fraction at low vs high z
            if starburst_types:
                sb_low = data.loc[low_z_mask, 'galaxy_type'].isin(starburst_types).sum() / low_z_count
                sb_high = data.loc[high_z_mask, 'galaxy_type'].isin(starburst_types).sum() / high_z_count

                # Add text annotation about selection effects
                selection_text = (
                    f"Selection effects:\n"
                    f"Starburst fraction:\n"
                    f"  $z < {z_median:.2f}$: {sb_low:.1%}\n"
                    f"  $z \\geq {z_median:.2f}$: {sb_high:.1%}"
                )

                # Add annotation box
                ax2.text(0.02, 0.02, selection_text, transform=ax2.transAxes,
                        fontsize=9, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Use subplots_adjust instead of tight_layout for better compatibility
    fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1)
    plt.savefig(f"{output_dir}/galaxy_type_vs_redshift.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/galaxy_type_vs_redshift.pdf")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Angular size test analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_analysis.py full    # Run on entire 4096x4096 image
    python run_analysis.py chip3   # Run on chip3 only (2048x2048)
    python run_analysis.py both    # Run full, then extract chip3 (most efficient)
        """
    )
    parser.add_argument(
        "mode",
        choices=["full", "chip3", "both"],
        help="Analysis mode: 'full' for entire image, 'chip3' for chip3 only, 'both' to run full and extract chip3"
    )
    args = parser.parse_args()

    if args.mode == "both":
        # Run full analysis first
        print("Running full analysis...")
        sed_catalog, sed_catalog_analysis, binned, binned_results, star_mask_centers = main("full")

        # Extract chip3 from full results (reuses classification - no re-computation!)
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

        # Generate chip3 plots from extracted catalog (reuses classification!)
        print("\n" + "=" * 60)
        print("GENERATING CHIP3 PLOTS FROM EXTRACTED DATA")
        print("=" * 60)
        chip3_output_dir = "./output/chip3"
        crop_bounds = (CHIP3_X_MIN, CHIP3_X_MAX, CHIP3_Y_MIN, CHIP3_Y_MAX)
        analyze_and_plot_catalog(
            chip3_catalog, chip3_catalog_filtered, chip3_output_dir,
            title_suffix=" (Chip 3)", images=chip3_images, crop_bounds=crop_bounds,
            star_mask_centers=star_mask_centers
        )
    else:
        sed_catalog, sed_catalog_analysis, binned, binned_results, star_mask_centers = main(args.mode)
