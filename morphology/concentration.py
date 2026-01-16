"""Concentration index and morphology measurements.

This module provides galaxy morphology measurements using the statmorph library
(Rodriguez-Gomez et al. 2019).

Measurements include:
- Concentration (C), Asymmetry (A), Smoothness (S) - CAS system
- Gini coefficient and M20 statistic
- Petrosian radius and half-light radius
- Sersic profile fitting
- PSF-aware morphology measurements for accurate deconvolution

Primary functions:
- compute_morphology(): Standard morphology measurements using statmorph
- compute_morphology_with_psf(): PSF-aware measurements with extended cutouts
- compute_morphology_single(): Single source measurements
- compute_morphology_parallel(): Parallel morphology for multiple sources

Utility functions:
- gini_coefficient(): Gini coefficient calculation
- m20_coefficient(): M20 statistic (second-order moment of brightest 20%)
- asymmetry_index(): Rotational asymmetry with background correction

References:
- Rodriguez-Gomez et al. 2019, MNRAS, 483, 4140 (statmorph)
- Abraham et al. 1994, ApJ, 432, 75
- Conselice et al. 2000, ApJ, 529, 886 (asymmetry)
- Lotz et al. 2004, AJ, 128, 163 (Gini-M20)
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import statmorph
from numpy.typing import NDArray
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.morphology import gini as photutils_gini
from scipy.ndimage import rotate


def compute_morphology(
    image: NDArray,
    segmap: NDArray,
    gain: float = 1.0,
    psf: NDArray | None = None,
) -> dict[str, NDArray]:
    """Compute morphological parameters for all sources using statmorph.

    This is the recommended method for computing morphology measurements.
    It provides validated, professional-grade measurements.

    Parameters
    ----------
    image : NDArray
        2D image array (background-subtracted)
    segmap : NDArray
        Segmentation map with sources labeled (1, 2, 3, ...)
    gain : float
        Detector gain (e-/ADU) for noise estimation
    psf : NDArray, optional
        PSF image for accurate measurements

    Returns
    -------
    dict[str, NDArray]
        Dictionary containing arrays for each morphological parameter:
        - label: Source labels from segmap
        - concentration: Concentration index C
        - asymmetry: Asymmetry index A
        - smoothness: Smoothness/clumpiness S
        - gini: Gini coefficient
        - m20: M20 statistic
        - half_light_radius: Half-light radius (elliptical)
        - petrosian_radius: Petrosian radius (circular)
        - r20, r80: Radii containing 20% and 80% of light
        - sersic_n: Sersic index (NaN if fit failed)
        - flag: Quality flag (0 = good)
        - flag_sersic: Sersic fit flag (0 = good)

    Examples
    --------
    >>> from photutils.segmentation import detect_sources
    >>> segmap = detect_sources(image, threshold, npixels=5)
    >>> morphs = compute_morphology(image, segmap.data, gain=2.5)
    >>> print(f"Mean concentration: {np.nanmean(morphs['concentration']):.2f}")
    """
    # Run statmorph on all sources
    source_morphs = statmorph.source_morphology(
        image, segmap, gain=gain, psf=psf
    )

    n = len(source_morphs)
    if n == 0:
        return _empty_morphology_result()

    # Extract all parameters into arrays
    return {
        "label": np.array([m.label for m in source_morphs], dtype=int),
        "concentration": np.array([m.concentration for m in source_morphs]),
        "asymmetry": np.array([m.asymmetry for m in source_morphs]),
        "smoothness": np.array([m.smoothness for m in source_morphs]),
        "gini": np.array([m.gini for m in source_morphs]),
        "m20": np.array([m.m20 for m in source_morphs]),
        "half_light_radius": np.array([m.rhalf_ellip for m in source_morphs]),
        "petrosian_radius": np.array([m.rpetro_circ for m in source_morphs]),
        "r20": np.array([m.r20 for m in source_morphs]),
        "r80": np.array([m.r80 for m in source_morphs]),
        "sersic_n": np.array([
            m.sersic_n if m.flag_sersic == 0 else np.nan
            for m in source_morphs
        ]),
        "sersic_rhalf": np.array([
            m.sersic_rhalf if m.flag_sersic == 0 else np.nan
            for m in source_morphs
        ]),
        "ellipticity": np.array([m.ellipticity_asymmetry for m in source_morphs]),
        "flag": np.array([m.flag for m in source_morphs], dtype=int),
        "flag_sersic": np.array([m.flag_sersic for m in source_morphs], dtype=int),
    }


def compute_morphology_single(
    image: NDArray,
    segmap: NDArray,
    source_id: int = 1,
    gain: float = 1.0,
    psf: NDArray | None = None,
) -> dict:
    """Compute morphological parameters for a single source.

    Parameters
    ----------
    image : NDArray
        2D image array (background-subtracted)
    segmap : NDArray
        Segmentation map with source labeled
    source_id : int
        Label of source in segmentation map
    gain : float
        Detector gain (e-/ADU)
    psf : NDArray, optional
        PSF image

    Returns
    -------
    dict
        Morphological parameters for the source
    """
    source_morphs = statmorph.source_morphology(
        image, segmap, gain=gain, psf=psf
    )

    # Find the requested source
    for m in source_morphs:
        if m.label == source_id:
            return {
                "concentration": m.concentration,
                "asymmetry": m.asymmetry,
                "smoothness": m.smoothness,
                "gini": m.gini,
                "m20": m.m20,
                "half_light_radius": m.rhalf_ellip,
                "petrosian_radius": m.rpetro_circ,
                "r20": m.r20,
                "r80": m.r80,
                "sersic_n": m.sersic_n if m.flag_sersic == 0 else np.nan,
                "flag": m.flag,
            }

    return {"error": f"Source {source_id} not found"}


def compute_morphology_with_psf(
    image: NDArray,
    segmap: NDArray,
    psf: NDArray,
    gain: float = 1.0,
) -> dict[str, NDArray]:
    """Compute PSF-aware morphological parameters for all sources.

    This function provides more accurate morphology measurements by properly
    accounting for the point spread function. It uses a larger cutout extent
    to ensure accurate measurements for extended sources.

    Parameters
    ----------
    image : NDArray
        2D image array (background-subtracted)
    segmap : NDArray
        Segmentation map with sources labeled (1, 2, 3, ...)
    psf : NDArray
        Point spread function image. Will be normalized before use.
    gain : float
        Detector gain (e-/ADU) for noise estimation

    Returns
    -------
    dict[str, NDArray]
        Dictionary containing arrays for each morphological parameter:
        - label: Source labels from segmap
        - concentration: Concentration index C
        - asymmetry: Asymmetry index A
        - smoothness: Smoothness/clumpiness S
        - gini: Gini coefficient
        - m20: M20 statistic
        - half_light_radius: Half-light radius (elliptical)
        - petrosian_radius: Petrosian radius (circular)
        - r20, r80: Radii containing 20% and 80% of light
        - sersic_n: Sersic index (NaN if fit failed)
        - sersic_rhalf: Sersic half-light radius (NaN if fit failed)
        - sersic_amplitude: Sersic amplitude (NaN if fit failed)
        - sersic_ellip: Sersic ellipticity (NaN if fit failed)
        - ellipticity: Ellipticity from asymmetry measurement
        - flag: Quality flag (0 = good)
        - flag_sersic: Sersic fit flag (0 = good)

    Notes
    -----
    The PSF is normalized to sum to 1.0 before being passed to statmorph.
    A larger cutout_extent (2.5) is used compared to the default (1.5) to
    ensure accurate measurements for extended sources and proper PSF handling.

    Examples
    --------
    >>> from photutils.segmentation import detect_sources
    >>> segmap = detect_sources(image, threshold, npixels=5)
    >>> # Load or create PSF
    >>> psf = fits.getdata('psf.fits')
    >>> morphs = compute_morphology_with_psf(image, segmap.data, psf, gain=2.5)
    >>> print(f"Mean Gini: {np.nanmean(morphs['gini']):.3f}")
    """
    # Validate PSF
    if psf is None or psf.size == 0:
        raise ValueError("PSF array cannot be None or empty")

    # Normalize PSF to sum to 1.0
    psf_sum = np.sum(psf)
    if psf_sum <= 0 or not np.isfinite(psf_sum):
        raise ValueError("PSF must have positive, finite total flux")
    psf_normalized = psf / psf_sum

    # Run statmorph with PSF and larger cutout extent for accurate measurements
    source_morphs = statmorph.source_morphology(
        image,
        segmap,
        gain=gain,
        psf=psf_normalized,
        cutout_extent=2.5,  # Larger than default (1.5) for PSF-convolved sources
    )

    n = len(source_morphs)
    if n == 0:
        return _empty_morphology_result()

    # Extract all parameters into arrays with proper error handling for Sersic fits
    labels = []
    concentrations = []
    asymmetries = []
    smoothnesses = []
    ginis = []
    m20s = []
    half_light_radii = []
    petrosian_radii = []
    r20s = []
    r80s = []
    sersic_ns = []
    sersic_rhalfs = []
    sersic_amplitudes = []
    sersic_ellips = []
    ellipticities = []
    flags = []
    flags_sersic = []

    for m in source_morphs:
        labels.append(m.label)
        concentrations.append(m.concentration)
        asymmetries.append(m.asymmetry)
        smoothnesses.append(m.smoothness)
        ginis.append(m.gini)
        m20s.append(m.m20)
        half_light_radii.append(m.rhalf_ellip)
        petrosian_radii.append(m.rpetro_circ)
        r20s.append(m.r20)
        r80s.append(m.r80)
        ellipticities.append(m.ellipticity_asymmetry)
        flags.append(m.flag)
        flags_sersic.append(m.flag_sersic)

        # Handle Sersic fit results with proper error checking
        if m.flag_sersic == 0:
            # Sersic fit succeeded
            try:
                sersic_ns.append(m.sersic_n if np.isfinite(m.sersic_n) else np.nan)
                sersic_rhalfs.append(
                    m.sersic_rhalf if np.isfinite(m.sersic_rhalf) else np.nan
                )
                sersic_amplitudes.append(
                    m.sersic_amplitude if np.isfinite(m.sersic_amplitude) else np.nan
                )
                sersic_ellips.append(
                    m.sersic_ellip if np.isfinite(m.sersic_ellip) else np.nan
                )
            except (AttributeError, TypeError):
                # Attribute access failed despite flag being 0
                sersic_ns.append(np.nan)
                sersic_rhalfs.append(np.nan)
                sersic_amplitudes.append(np.nan)
                sersic_ellips.append(np.nan)
        else:
            # Sersic fit failed
            sersic_ns.append(np.nan)
            sersic_rhalfs.append(np.nan)
            sersic_amplitudes.append(np.nan)
            sersic_ellips.append(np.nan)

    return {
        "label": np.array(labels, dtype=int),
        "concentration": np.array(concentrations),
        "asymmetry": np.array(asymmetries),
        "smoothness": np.array(smoothnesses),
        "gini": np.array(ginis),
        "m20": np.array(m20s),
        "half_light_radius": np.array(half_light_radii),
        "petrosian_radius": np.array(petrosian_radii),
        "r20": np.array(r20s),
        "r80": np.array(r80s),
        "sersic_n": np.array(sersic_ns),
        "sersic_rhalf": np.array(sersic_rhalfs),
        "sersic_amplitude": np.array(sersic_amplitudes),
        "sersic_ellip": np.array(sersic_ellips),
        "ellipticity": np.array(ellipticities),
        "flag": np.array(flags, dtype=int),
        "flag_sersic": np.array(flags_sersic, dtype=int),
    }


def _empty_morphology_result() -> dict[str, NDArray]:
    """Return empty arrays for morphology result."""
    return {
        "label": np.array([], dtype=int),
        "concentration": np.array([]),
        "asymmetry": np.array([]),
        "smoothness": np.array([]),
        "gini": np.array([]),
        "m20": np.array([]),
        "half_light_radius": np.array([]),
        "petrosian_radius": np.array([]),
        "r20": np.array([]),
        "r80": np.array([]),
        "sersic_n": np.array([]),
        "sersic_rhalf": np.array([]),
        "ellipticity": np.array([]),
        "flag": np.array([], dtype=int),
        "flag_sersic": np.array([], dtype=int),
    }


# =============================================================================
# Standalone utility functions (minimal implementations)
# =============================================================================


def gini_coefficient(data: NDArray, mask: NDArray | None = None) -> float:
    """Calculate Gini coefficient of pixel values.

    Parameters
    ----------
    data : NDArray
        2D image array
    mask : NDArray, optional
        Boolean mask for pixels to include

    Returns
    -------
    float
        Gini coefficient (0 = uniform, 1 = all flux in one pixel)
    """
    return float(photutils_gini(data, mask=mask))


def m20_coefficient(data: NDArray, mask: NDArray | None = None) -> float:
    """Calculate M20 statistic (second-order moment of brightest 20% of pixels).

    M20 is defined as the normalized second-order moment of the brightest
    20% of a galaxy's flux, following Lotz et al. (2004). It traces the
    spatial distribution of bright clumps and is sensitive to mergers.

    Parameters
    ----------
    data : NDArray
        2D image array (background-subtracted)
    mask : NDArray, optional
        Boolean mask where True indicates pixels belonging to the source.
        If None, all pixels with positive flux are used.

    Returns
    -------
    float
        M20 statistic, typically in range [-3, 0].
        More negative values indicate more concentrated light distributions.
        Values closer to 0 indicate multiple bright regions (e.g., mergers).

    Notes
    -----
    This is a fallback implementation for when statmorph is unavailable.
    The statmorph implementation is more robust and should be preferred.

    The M20 statistic is computed as:
        M20 = log10(sum_i M_i / M_tot)

    where M_i is the second-order moment of pixel i (among the brightest 20%),
    and M_tot is the total second-order moment of all source pixels.

    References
    ----------
    Lotz, J. M., Primack, J., & Madau, P. 2004, AJ, 128, 163

    Examples
    --------
    >>> from photutils.segmentation import detect_sources
    >>> segmap = detect_sources(image, threshold, npixels=5)
    >>> source_mask = segmap.data == 1
    >>> m20 = m20_coefficient(image, mask=source_mask)
    >>> print(f"M20 = {m20:.2f}")
    """
    # Extract pixel values within the mask
    if mask is not None:
        values = data[mask].ravel()
        # Get pixel coordinates within the mask
        y_coords, x_coords = np.where(mask)
    else:
        values = data.ravel()
        y_coords, x_coords = np.indices(data.shape)
        y_coords = y_coords.ravel()
        x_coords = x_coords.ravel()

    # Filter for positive, finite values
    valid = np.isfinite(values) & (values > 0)
    values = values[valid]
    x_coords = x_coords[valid]
    y_coords = y_coords[valid]

    if len(values) < 2:
        return np.nan

    # Calculate flux-weighted centroid
    total_flux = np.sum(values)
    if total_flux <= 0:
        return np.nan

    x_center = np.sum(x_coords * values) / total_flux
    y_center = np.sum(y_coords * values) / total_flux

    # Calculate second-order moment for each pixel: M_i = f_i * r_i^2
    # where r_i is distance from centroid
    r_squared = (x_coords - x_center)**2 + (y_coords - y_center)**2
    moments = values * r_squared

    # Total second-order moment
    m_total = np.sum(moments)
    if m_total <= 0:
        return np.nan

    # Sort pixels by flux (descending) to find brightest 20%
    sort_indices = np.argsort(values)[::-1]
    sorted_flux = values[sort_indices]
    sorted_moments = moments[sort_indices]

    # Find cumulative flux and determine brightest 20%
    cumulative_flux = np.cumsum(sorted_flux)
    threshold_flux = 0.2 * total_flux

    # Find index where cumulative flux reaches 20% of total
    bright_mask = cumulative_flux <= threshold_flux

    # Include at least one pixel
    if not np.any(bright_mask):
        bright_mask[0] = True

    # Sum of second moments for brightest 20%
    m_20 = np.sum(sorted_moments[bright_mask])

    if m_20 <= 0:
        return np.nan

    # M20 = log10(M_20 / M_total)
    return float(np.log10(m_20 / m_total))


def asymmetry_index(
    data: NDArray,
    mask: NDArray | None = None,
    background: NDArray | None = None,
) -> float:
    """Calculate rotational asymmetry index with background correction.

    The asymmetry parameter A measures the degree to which a galaxy's light
    distribution is rotationally symmetric about its center. It is computed
    by rotating the image 180 degrees and comparing with the original.

    Parameters
    ----------
    data : NDArray
        2D image array (background-subtracted)
    mask : NDArray, optional
        Boolean mask where True indicates pixels belonging to the source.
        If None, uses the central region of the image.
    background : NDArray, optional
        Background region for asymmetry correction. If provided, the
        asymmetry of this region is subtracted from the source asymmetry.

    Returns
    -------
    float
        Asymmetry index A, typically in range [0, 1].
        A = 0 indicates perfect symmetry.
        Higher values indicate more asymmetric/disturbed morphology.

    Notes
    -----
    This is a fallback implementation for when statmorph is unavailable.
    The statmorph implementation is more robust and should be preferred.

    The asymmetry is computed as:
        A = sum(|I - I_180|) / (2 * sum(|I|)) - A_background

    where I_180 is the image rotated 180 degrees about the center.

    References
    ----------
    Conselice, C. J., Bershady, M. A., & Jangren, A. 2000, ApJ, 529, 886
    Lotz, J. M., Primack, J., & Madau, P. 2004, AJ, 128, 163

    Examples
    --------
    >>> from photutils.segmentation import detect_sources
    >>> segmap = detect_sources(image, threshold, npixels=5)
    >>> source_mask = segmap.data == 1
    >>> asym = asymmetry_index(image, mask=source_mask)
    >>> print(f"Asymmetry = {asym:.3f}")
    """
    # Work with a copy to avoid modifying input
    image = np.array(data, dtype=float)

    if mask is not None:
        # Extract bounding box around masked region
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0:
            return np.nan

        y_min, y_max = y_indices.min(), y_indices.max() + 1
        x_min, x_max = x_indices.min(), x_indices.max() + 1

        # Extract cutout
        cutout = image[y_min:y_max, x_min:x_max].copy()
        cutout_mask = mask[y_min:y_max, x_min:x_max]

        # Zero out pixels outside the mask
        cutout[~cutout_mask] = 0
    else:
        cutout = image.copy()
        cutout_mask = np.ones(cutout.shape, dtype=bool)

    # Ensure cutout is finite
    cutout = np.nan_to_num(cutout, nan=0.0, posinf=0.0, neginf=0.0)

    if cutout.size == 0 or np.sum(np.abs(cutout)) == 0:
        return np.nan

    # Calculate flux-weighted centroid for rotation center
    total_flux = np.sum(np.abs(cutout[cutout_mask]))
    if total_flux <= 0:
        return np.nan

    y_grid, x_grid = np.indices(cutout.shape)
    x_center = np.sum(x_grid[cutout_mask] * np.abs(cutout[cutout_mask])) / total_flux
    y_center = np.sum(y_grid[cutout_mask] * np.abs(cutout[cutout_mask])) / total_flux

    # Rotate image 180 degrees around the center
    # Use scipy rotate with reshape=False to maintain array size
    rotated = rotate(cutout, 180, reshape=False, order=1, mode='constant', cval=0)

    # Also rotate the mask to ensure we compare same regions
    rotated_mask = rotate(
        cutout_mask.astype(float), 180, reshape=False, order=0, mode='constant', cval=0
    ) > 0.5

    # Combined mask: only compare pixels valid in both original and rotated
    combined_mask = cutout_mask & rotated_mask

    if not np.any(combined_mask):
        return np.nan

    # Calculate asymmetry: A = sum(|I - I_180|) / (2 * sum(|I|))
    residual = np.abs(cutout[combined_mask] - rotated[combined_mask])
    total_abs = np.sum(np.abs(cutout[combined_mask]))

    if total_abs <= 0:
        return np.nan

    asymmetry = np.sum(residual) / (2 * total_abs)

    # Background asymmetry correction
    if background is not None:
        bg_asym = _compute_background_asymmetry(background)
        if np.isfinite(bg_asym):
            asymmetry = max(0, asymmetry - bg_asym)

    return float(asymmetry)


def _compute_background_asymmetry(background: NDArray) -> float:
    """Compute asymmetry of a background region for correction.

    Parameters
    ----------
    background : NDArray
        2D array of background pixels

    Returns
    -------
    float
        Background asymmetry value
    """
    bg = np.array(background, dtype=float)
    bg = np.nan_to_num(bg, nan=0.0, posinf=0.0, neginf=0.0)

    if bg.size == 0 or np.sum(np.abs(bg)) == 0:
        return 0.0

    # Rotate 180 degrees
    rotated = rotate(bg, 180, reshape=False, order=1, mode='constant', cval=0)

    # Calculate asymmetry
    residual = np.abs(bg - rotated)
    total_abs = np.sum(np.abs(bg))

    if total_abs <= 0:
        return 0.0

    return float(np.sum(residual) / (2 * total_abs))


def concentration_index(
    image: NDArray,
    x: float,
    y: float,
    r_inner: float = 3.0,
    r_outer: float = 10.0,
) -> float:
    """Calculate simple concentration index using aperture photometry.

    For full morphology measurements, use compute_morphology() instead.

    The concentration index measures how centrally concentrated the light is.
    Higher values indicate more concentrated (star-like) profiles.
    Lower values indicate more extended (galaxy-like) profiles.

    The formula approximates the CAS concentration C = 5*log10(r80/r20)
    using aperture photometry, giving values in a similar range:
    - Stars (point sources): C > 3
    - Galaxies (extended): C < 2.5

    Parameters
    ----------
    image : NDArray
        2D image array
    x, y : float
        Source center position
    r_inner, r_outer : float
        Inner and outer aperture radii in pixels

    Returns
    -------
    float
        Concentration index (higher = more concentrated)
    """
    aper_inner = CircularAperture((x, y), r=r_inner)
    aper_outer = CircularAperture((x, y), r=r_outer)

    phot_inner = aperture_photometry(image, aper_inner)
    phot_outer = aperture_photometry(image, aper_outer)

    flux_inner = float(phot_inner["aperture_sum"][0])
    flux_outer = float(phot_outer["aperture_sum"][0])

    if flux_outer > 0 and flux_inner > 0:
        # Compute flux ratio (fraction of light in inner aperture)
        flux_ratio = flux_inner / flux_outer

        # Transform to concentration scale where higher = more concentrated
        # For stars: flux_ratio ~ 0.8-0.95, gives C ~ 3-5
        # For galaxies: flux_ratio ~ 0.2-0.5, gives C ~ 0.5-1.5
        # Add small epsilon to avoid log(0) when flux_ratio = 1
        fraction_outside = 1.0 - flux_ratio + 0.01
        return -5.0 * np.log10(fraction_outside)

    return np.nan


def half_light_radius(
    image: NDArray,
    x: float,
    y: float,
    max_radius: float = 50.0,
) -> float:
    """Calculate half-light radius using cumulative flux profile.

    For full morphology measurements, use compute_morphology() instead.

    Parameters
    ----------
    image : NDArray
        2D image array
    x, y : float
        Source center position
    max_radius : float
        Maximum radius to search

    Returns
    -------
    float
        Half-light radius in pixels
    """
    ny, nx = image.shape

    # Validate coordinates
    if not (np.isfinite(x) and np.isfinite(y)):
        return np.nan
    if not (5 <= x < nx - 5 and 5 <= y < ny - 5):
        return np.nan

    # Compute radial profile
    r_int = min(int(max_radius), int(min(x, y, nx - x, ny - y)) - 1)
    if r_int < 5:
        return np.nan

    y_min, y_max = int(y) - r_int, int(y) + r_int + 1
    x_min, x_max = int(x) - r_int, int(x) + r_int + 1

    cutout = image[y_min:y_max, x_min:x_max]
    cy, cx = y - y_min, x - x_min

    Y, X = np.ogrid[:cutout.shape[0], :cutout.shape[1]]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)

    # Cumulative flux at each radius
    radii = np.arange(1, r_int + 1, 0.5)
    cumulative = np.array([np.sum(cutout[r >= R]) for r in radii])

    total = cumulative[-1]
    if total <= 0:
        return np.nan

    return float(np.interp(0.5 * total, cumulative, radii))


# =============================================================================
# Backward compatibility aliases
# =============================================================================

# These functions are deprecated - use compute_morphology() instead
compute_morphology_statmorph = compute_morphology
compute_morphology_batch_statmorph = compute_morphology


# Legacy batch functions - redirect to statmorph
def compute_morphology_batch(
    image: NDArray,
    x_coords: NDArray,
    y_coords: NDArray,
    max_radius: float = 50.0,
) -> dict[str, NDArray]:
    """Compute morphology for sources at given coordinates.

    DEPRECATED: Use compute_morphology() with a segmentation map instead.
    This function provides limited measurements compared to statmorph.
    """
    n = len(x_coords)
    # Compute concentration index for all sources
    concentration_values = np.array([
        concentration_index(image, x, y) for x, y in zip(x_coords, y_coords)
    ])
    return {
        "concentration": concentration_values,
        # Use concentration_index as concentration_c (aperture-based concentration)
        # This is a simplified version of the CAS C parameter
        "concentration_c": concentration_values,
        "half_light_radius": np.array([
            half_light_radius(image, x, y, max_radius)
            for x, y in zip(x_coords, y_coords)
        ]),
    }


# Alias for backward compatibility
compute_morphology_batch_fast = compute_morphology_batch


# =============================================================================
# Vectorized batch functions
# =============================================================================


def concentration_index_batch(
    image: NDArray,
    x_coords: NDArray,
    y_coords: NDArray,
    r_inner: float = 3.0,
    r_outer: float = 10.0,
) -> NDArray:
    """Calculate concentration index for multiple sources.

    Parameters
    ----------
    image : NDArray
        2D image array
    x_coords, y_coords : NDArray
        Source center positions
    r_inner, r_outer : float
        Inner and outer aperture radii in pixels

    Returns
    -------
    NDArray
        Concentration index for each source
    """
    return np.array([
        concentration_index(image, x, y, r_inner, r_outer)
        for x, y in zip(x_coords, y_coords)
    ])


def half_light_radius_batch(
    image: NDArray,
    x_coords: NDArray,
    y_coords: NDArray,
    max_radius: float = 50.0,
) -> NDArray:
    """Calculate half-light radius for multiple sources.

    Parameters
    ----------
    image : NDArray
        2D image array
    x_coords, y_coords : NDArray
        Source center positions
    max_radius : float
        Maximum radius to search

    Returns
    -------
    NDArray
        Half-light radius for each source
    """
    return np.array([
        half_light_radius(image, x, y, max_radius)
        for x, y in zip(x_coords, y_coords)
    ])


# =============================================================================
# Parallel batch functions
# =============================================================================


def _compute_concentration_worker(args: tuple) -> float:
    """Worker function for parallel concentration calculation."""
    image, x, y, r_inner, r_outer = args
    return concentration_index(image, x, y, r_inner, r_outer)


def _compute_hlr_worker(args: tuple) -> float:
    """Worker function for parallel half-light radius calculation."""
    image, x, y, max_radius = args
    return half_light_radius(image, x, y, max_radius)


def concentration_index_batch_parallel(
    image: NDArray,
    x_coords: NDArray,
    y_coords: NDArray,
    r_inner: float = 3.0,
    r_outer: float = 10.0,
    n_workers: int | None = None,
) -> NDArray:
    """Calculate concentration index for multiple sources in parallel.

    Uses ThreadPoolExecutor for parallelization since the computation
    is CPU-bound but shares image data.

    Parameters
    ----------
    image : NDArray
        2D image array
    x_coords, y_coords : NDArray
        Source center positions
    r_inner, r_outer : float
        Inner and outer aperture radii in pixels
    n_workers : int, optional
        Number of parallel workers. If None, auto-detects.

    Returns
    -------
    NDArray
        Concentration index for each source
    """
    if n_workers is None:
        try:
            from resource_config import get_config
            config = get_config()
            n_workers = config.n_threads
        except ImportError:
            n_workers = max(1, (mp.cpu_count() or 2) - 1)

    n_sources = len(x_coords)

    # For small numbers of sources, use serial processing
    if n_sources < 10 or n_workers <= 1:
        return concentration_index_batch(image, x_coords, y_coords, r_inner, r_outer)

    # Prepare arguments for workers
    args_list = [
        (image, x, y, r_inner, r_outer)
        for x, y in zip(x_coords, y_coords)
    ]

    # Use ThreadPoolExecutor since we're sharing the image array
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_compute_concentration_worker, args_list))

    return np.array(results)


def half_light_radius_batch_parallel(
    image: NDArray,
    x_coords: NDArray,
    y_coords: NDArray,
    max_radius: float = 50.0,
    n_workers: int | None = None,
) -> NDArray:
    """Calculate half-light radius for multiple sources in parallel.

    Uses ThreadPoolExecutor for parallelization since the computation
    is CPU-bound but shares image data.

    Parameters
    ----------
    image : NDArray
        2D image array
    x_coords, y_coords : NDArray
        Source center positions
    max_radius : float
        Maximum radius to search
    n_workers : int, optional
        Number of parallel workers. If None, auto-detects.

    Returns
    -------
    NDArray
        Half-light radius for each source
    """
    if n_workers is None:
        try:
            from resource_config import get_config
            config = get_config()
            n_workers = config.n_threads
        except ImportError:
            n_workers = max(1, (mp.cpu_count() or 2) - 1)

    n_sources = len(x_coords)

    # For small numbers of sources, use serial processing
    if n_sources < 10 or n_workers <= 1:
        return half_light_radius_batch(image, x_coords, y_coords, max_radius)

    # Prepare arguments for workers
    args_list = [
        (image, x, y, max_radius)
        for x, y in zip(x_coords, y_coords)
    ]

    # Use ThreadPoolExecutor since we're sharing the image array
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_compute_hlr_worker, args_list))

    return np.array(results)


def compute_morphology_batch_parallel(
    image: NDArray,
    x_coords: NDArray,
    y_coords: NDArray,
    max_radius: float = 50.0,
    n_workers: int | None = None,
) -> dict[str, NDArray]:
    """Compute morphology for sources at given coordinates in parallel.

    This function computes concentration index and half-light radius
    for multiple sources using parallel processing.

    Parameters
    ----------
    image : NDArray
        2D image array
    x_coords, y_coords : NDArray
        Source center positions
    max_radius : float
        Maximum radius to search for half-light radius
    n_workers : int, optional
        Number of parallel workers. If None, auto-detects.

    Returns
    -------
    dict[str, NDArray]
        Dictionary with 'concentration', 'concentration_c', 'half_light_radius'
    """
    n = len(x_coords)

    # Compute both measurements in parallel
    concentration = concentration_index_batch_parallel(
        image, x_coords, y_coords, n_workers=n_workers
    )
    hlr = half_light_radius_batch_parallel(
        image, x_coords, y_coords, max_radius, n_workers=n_workers
    )

    return {
        "concentration": concentration,
        "concentration_c": np.full(n, np.nan),  # Not computed in simple mode
        "half_light_radius": hlr,
    }
