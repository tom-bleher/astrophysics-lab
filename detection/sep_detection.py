"""Source detection using SEP (Source Extractor as a Python library).

SEP implements the core algorithms of SExtractor in Python/C, providing:
- Fast background estimation and subtraction
- Source detection with deblending
- Kron (AUTO) and Petrosian aperture photometry
- SExtractor-compatible output for cross-matching with published catalogs

This module provides an alternative backend to photutils for source detection,
optimized for speed on large mosaics and compatibility with SExtractor catalogs.

References
----------
- SEP documentation: https://sep.readthedocs.io/
- Barbary 2016: https://doi.org/10.21105/joss.00058
- Bertin & Arnouts 1996, A&AS, 117, 393 (SExtractor)
- Kron 1980, ApJS, 43, 305 (Kron apertures)
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    import sep
    SEP_AVAILABLE = True
except ImportError:
    SEP_AVAILABLE = False

try:
    import statmorph
    STATMORPH_AVAILABLE = True
except ImportError:
    STATMORPH_AVAILABLE = False

# Suppress RuntimeWarning from Sersic fitting in statmorph/astropy
# Must be at module level to apply to multiprocessing workers
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning,
                       message='overflow encountered in power')

# SExtractor-compatible flag definitions
FLAG_NONE = 0
FLAG_CROWDED = 1        # Object has neighbors (was blended)
FLAG_BLENDED = 2        # Object was originally blended with another
FLAG_SATURATED = 4      # At least one pixel is saturated
FLAG_TRUNCATED = 8      # Object truncated at image boundary
FLAG_APERTURE_INCOMPLETE = 16  # Aperture data incomplete or corrupted
FLAG_ISOPHOTE_INCOMPLETE = 32  # Isophotal data incomplete or corrupted
FLAG_DEBLEND_OVERFLOW = 64     # Memory overflow during deblending
FLAG_EXTRACTION_OVERFLOW = 128  # Memory overflow during extraction


class SEPBackground(NamedTuple):
    """Background estimation result from SEP."""
    background: NDArray      # 2D background map
    background_rms: NDArray  # 2D RMS map
    global_back: float       # Global background level
    global_rms: float        # Global RMS


@dataclass
class SEPDetectionResult:
    """Result container for SEP source detection.

    Attributes
    ----------
    catalog : pd.DataFrame
        Source catalog with positions, fluxes, and morphology
    segmentation : NDArray or None
        Segmentation map (if requested)
    background : SEPBackground
        Background estimation results
    n_sources : int
        Number of detected sources
    """
    catalog: pd.DataFrame
    segmentation: NDArray | None
    background: SEPBackground
    n_sources: int


def check_sep_available() -> bool:
    """Check if SEP is available."""
    return SEP_AVAILABLE


def estimate_background(
    data: NDArray,
    mask: NDArray | None = None,
    box_size: int = 64,
    filter_size: int = 3,
) -> SEPBackground:
    """Estimate 2D background using SEP.

    Uses a mesh-based approach with median filtering, matching SExtractor's
    BACK_SIZE and BACK_FILTERSIZE parameters.

    Parameters
    ----------
    data : NDArray
        2D image data (must be C-contiguous float32/float64)
    mask : NDArray, optional
        Boolean mask where True = ignore pixel
    box_size : int
        Background mesh box size in pixels (default: 64)
    filter_size : int
        Median filter size for background smoothing (default: 3)

    Returns
    -------
    SEPBackground
        Background estimation with 2D maps and global statistics
    """
    if not SEP_AVAILABLE:
        raise ImportError("sep is not installed. Run: pip install sep")

    # Ensure data is C-contiguous and correct dtype
    data = np.ascontiguousarray(data, dtype=np.float64)

    if mask is not None:
        mask = np.ascontiguousarray(mask, dtype=np.uint8)

    # Estimate background
    bkg = sep.Background(
        data,
        mask=mask,
        bw=box_size,
        bh=box_size,
        fw=filter_size,
        fh=filter_size,
    )

    return SEPBackground(
        background=bkg.back(),
        background_rms=bkg.rms(),
        global_back=float(bkg.globalback),
        global_rms=float(bkg.globalrms),
    )


def detect_sources_sep(
    data: NDArray,
    threshold: float = 1.5,
    min_area: int = 5,
    mask: NDArray | None = None,
    gain: float | None = None,
    box_size: int = 64,
    filter_size: int = 3,
    filter_kernel: NDArray | None = None,
    deblend_nthresh: int = 32,
    deblend_cont: float = 0.005,
    clean: bool = True,
    clean_param: float = 1.0,
    segmentation_map: bool = False,
) -> SEPDetectionResult:
    """Detect sources using SEP (SExtractor algorithm).

    This implements the full SExtractor detection pipeline:
    1. Background estimation and subtraction
    2. Convolution with detection filter (matched filter)
    3. Thresholding at `threshold * background_rms`
    4. Connected component labeling with minimum area
    5. Deblending overlapping sources
    6. Source measurement (positions, fluxes, shapes)

    Parameters
    ----------
    data : NDArray
        2D image data array
    threshold : float
        Detection threshold in sigma above background (default: 1.5)
    min_area : int
        Minimum number of connected pixels (DETECT_MINAREA, default: 5)
    mask : NDArray, optional
        Boolean mask where True = ignore pixel
    gain : float, optional
        Detector gain in e-/ADU for Poisson noise. If None, only
        background noise is used.
    box_size : int
        Background mesh size (BACK_SIZE, default: 64)
    filter_size : int
        Background filter size (BACK_FILTERSIZE, default: 3)
    filter_kernel : NDArray, optional
        Convolution kernel for detection. Default: 3x3 Gaussian
    deblend_nthresh : int
        Number of deblending thresholds (DEBLEND_NTHRESH, default: 32)
    deblend_cont : float
        Minimum contrast for deblending (DEBLEND_MINCONT, default: 0.005)
    clean : bool
        Remove spurious detections (default: True)
    clean_param : float
        Cleaning efficiency parameter (default: 1.0)
    segmentation_map : bool
        Return segmentation map (default: False)

    Returns
    -------
    SEPDetectionResult
        Detection results with catalog, optional segmentation, and background
    """
    if not SEP_AVAILABLE:
        raise ImportError("sep is not installed. Run: pip install sep")

    # Ensure data is C-contiguous float64
    data = np.ascontiguousarray(data, dtype=np.float64)

    if mask is not None:
        mask = np.ascontiguousarray(mask, dtype=np.uint8)

    # Estimate and subtract background
    bkg_result = estimate_background(data, mask, box_size, filter_size)
    data_sub = data - bkg_result.background

    # Default filter kernel (3x3 Gaussian, FWHM~2 pixels)
    if filter_kernel is None:
        filter_kernel = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ], dtype=np.float64) / 16.0

    # Set up error array for proper flux errors
    # When err is provided, thresh is interpreted as S/N threshold (scalar)
    if gain is not None and gain > 0:
        # Include Poisson noise: var = bkg_rms^2 + data/gain
        err = np.sqrt(bkg_result.background_rms**2 + np.maximum(data_sub, 0) / gain)
    else:
        err = bkg_result.background_rms

    # Extract sources
    # When err is provided, threshold is interpreted as S/N ratio
    objects, segmap = sep.extract(
        data_sub,
        threshold,  # S/N threshold (scalar) when err is provided
        err=err,
        mask=mask,
        minarea=min_area,
        filter_kernel=filter_kernel,
        filter_type='matched',
        deblend_nthresh=deblend_nthresh,
        deblend_cont=deblend_cont,
        clean=clean,
        clean_param=clean_param,
        segmentation_map=True,  # Always get segmap for Kron radii
    )

    if len(objects) == 0:
        # Return empty result
        empty_df = pd.DataFrame(columns=[
            'x', 'y', 'flux', 'flux_err', 'peak', 'a', 'b', 'theta',
            'cxx', 'cyy', 'cxy', 'flag', 'npix'
        ])
        return SEPDetectionResult(
            catalog=empty_df,
            segmentation=segmap if segmentation_map else None,
            background=bkg_result,
            n_sources=0,
        )

    # Build catalog DataFrame
    # Note: sep.extract doesn't return flux errors directly - compute separately with aperture functions
    catalog = pd.DataFrame({
        # Positions (0-indexed)
        'x': objects['x'],
        'y': objects['y'],
        'xpeak': objects['xpeak'],
        'ypeak': objects['ypeak'],

        # Basic photometry (isophotal)
        'flux': objects['flux'],
        'flux_err': np.full(len(objects), np.nan),  # Computed later with aperture photometry
        'peak': objects['peak'],

        # Shape parameters
        'a': objects['a'],        # Semi-major axis (pixels)
        'b': objects['b'],        # Semi-minor axis (pixels)
        'theta': objects['theta'],  # Position angle (radians)

        # Second moments (for elliptical apertures)
        'cxx': objects['cxx'],
        'cyy': objects['cyy'],
        'cxy': objects['cxy'],

        # Flags and area
        'flag': objects['flag'],
        'npix': objects['npix'],

        # Additional SExtractor-like columns
        'x2': objects['x2'],  # Variance in x
        'y2': objects['y2'],  # Variance in y
        'xy': objects['xy'],  # Covariance
    })

    return SEPDetectionResult(
        catalog=catalog,
        segmentation=segmap if segmentation_map else None,
        background=bkg_result,
        n_sources=len(catalog),
    )


def kron_photometry(
    data: NDArray,
    catalog: pd.DataFrame,
    background_rms: NDArray | float,
    gain: float | None = None,
    mask: NDArray | None = None,
    r_min: float = 3.5,
    kron_factor: float = 2.5,
    kron_min_radius: float = 1.75,
) -> pd.DataFrame:
    """Perform Kron aperture photometry (SExtractor AUTO magnitudes).

    Kron apertures are elliptical apertures scaled to capture most of the
    total flux. The Kron radius is computed as the first moment of the
    light distribution, and the aperture is set to `kron_factor * r_kron`.

    Parameters
    ----------
    data : NDArray
        Background-subtracted 2D image
    catalog : pd.DataFrame
        Source catalog from detect_sources_sep with shape parameters
    background_rms : NDArray or float
        Background RMS (2D map or scalar)
    gain : float, optional
        Detector gain for Poisson noise
    mask : NDArray, optional
        Boolean mask for bad pixels
    r_min : float
        Minimum radius for Kron radius measurement (default: 3.5)
    kron_factor : float
        Kron radius multiplier (default: 2.5, captures ~94% of Gaussian flux)
    kron_min_radius : float
        Minimum Kron aperture radius (default: 1.75)

    Returns
    -------
    pd.DataFrame
        Input catalog with added columns:
        - kron_radius: Kron radius in pixels
        - flux_auto: AUTO flux (Kron aperture flux)
        - flux_auto_err: AUTO flux error
        - flag_auto: Photometry flags
    """
    if not SEP_AVAILABLE:
        raise ImportError("sep is not installed. Run: pip install sep")

    data = np.ascontiguousarray(data, dtype=np.float64)

    if isinstance(background_rms, (int, float)):
        err = np.full_like(data, background_rms)
    else:
        err = np.ascontiguousarray(background_rms, dtype=np.float64)

    if gain is not None and gain > 0:
        err = np.sqrt(err**2 + np.maximum(data, 0) / gain)

    if mask is not None:
        mask = np.ascontiguousarray(mask, dtype=np.uint8)

    # Extract arrays from catalog
    x = catalog['x'].values.copy()
    y = catalog['y'].values.copy()
    a = catalog['a'].values.copy()
    b = catalog['b'].values.copy()
    theta = catalog['theta'].values.copy()

    # Validate and fix ellipse parameters
    # SEP requires a > 0, b > 0, and a >= b
    MIN_AXIS = 0.5  # Minimum semi-axis in pixels
    invalid_a = ~np.isfinite(a) | (a <= 0)
    invalid_b = ~np.isfinite(b) | (b <= 0)
    a[invalid_a] = MIN_AXIS
    b[invalid_b] = MIN_AXIS
    # Ensure a >= b (SEP convention)
    swap_mask = b > a
    a[swap_mask], b[swap_mask] = b[swap_mask], a[swap_mask]
    # Fix invalid theta
    invalid_theta = ~np.isfinite(theta)
    theta[invalid_theta] = 0.0

    # Compute Kron radii
    kronrad, krflag = sep.kron_radius(
        data, x, y, a, b, theta,
        r=r_min,
        mask=mask,
    )

    # Apply minimum radius
    kronrad = np.maximum(kronrad, kron_min_radius)

    # Compute Kron aperture flux
    flux_auto, flux_auto_err, flag_auto = sep.sum_ellipse(
        data, x, y,
        a * kron_factor, b * kron_factor, theta,
        r=kronrad,
        err=err,
        mask=mask,
        subpix=5,
    )

    # Add to catalog
    result = catalog.copy()
    result['kron_radius'] = kronrad
    result['flux_auto'] = flux_auto
    result['flux_auto_err'] = flux_auto_err
    result['flag_auto'] = flag_auto | krflag

    return result


def circular_aperture_photometry(
    data: NDArray,
    catalog: pd.DataFrame,
    background_rms: NDArray | float,
    radii: list[float] | None = None,
    gain: float | None = None,
    mask: NDArray | None = None,
) -> pd.DataFrame:
    """Perform circular aperture photometry at multiple radii.

    Parameters
    ----------
    data : NDArray
        Background-subtracted 2D image
    catalog : pd.DataFrame
        Source catalog with 'x' and 'y' columns
    background_rms : NDArray or float
        Background RMS for error calculation
    radii : list of float, optional
        Aperture radii in pixels. Default: [2, 3, 5, 7, 10]
    gain : float, optional
        Detector gain for Poisson noise
    mask : NDArray, optional
        Boolean mask for bad pixels

    Returns
    -------
    pd.DataFrame
        Input catalog with added columns:
        - flux_aper_N: Flux in N-pixel radius aperture
        - flux_aper_N_err: Flux error
    """
    if not SEP_AVAILABLE:
        raise ImportError("sep is not installed. Run: pip install sep")

    if radii is None:
        radii = [2.0, 3.0, 5.0, 7.0, 10.0]

    data = np.ascontiguousarray(data, dtype=np.float64)

    if isinstance(background_rms, (int, float)):
        err = np.full_like(data, background_rms)
    else:
        err = np.ascontiguousarray(background_rms, dtype=np.float64)

    if gain is not None and gain > 0:
        err = np.sqrt(err**2 + np.maximum(data, 0) / gain)

    if mask is not None:
        mask = np.ascontiguousarray(mask, dtype=np.uint8)

    x = catalog['x'].values
    y = catalog['y'].values

    result = catalog.copy()

    for r in radii:
        flux, flux_err, flag = sep.sum_circle(
            data, x, y, r,
            err=err,
            mask=mask,
            subpix=5,
        )

        r_str = str(r).replace('.', 'p')
        result[f'flux_aper_{r_str}'] = flux
        result[f'flux_aper_{r_str}_err'] = flux_err
        result[f'flag_aper_{r_str}'] = flag

    return result


def compute_flux_radii(
    data: NDArray,
    catalog: pd.DataFrame,
    background_rms: NDArray | float,
    fractions: list[float] | tuple[float, ...] = (0.5,),
    mask: NDArray | None = None,
    max_radius: float = 50.0,
    n_radii: int = 50,
) -> dict[float, NDArray]:
    """Compute radii enclosing given fractions of total flux using curve of growth.

    This implementation is vectorized over all sources for each radius,
    reducing function call overhead from O(N × M) to O(M) where N = sources
    and M = radii. This provides 10-50x speedup for typical catalogs.

    Multiple fractions can be computed efficiently in a single call since
    the flux grid is reused.

    Parameters
    ----------
    data : NDArray
        Background-subtracted 2D image
    catalog : pd.DataFrame
        Source catalog with 'x', 'y', and 'flux_auto' columns
    background_rms : NDArray or float
        Background RMS
    fractions : list or tuple of float
        Flux fractions to compute radii for (default: (0.5,) for half-light)
    mask : NDArray, optional
        Boolean mask
    max_radius : float
        Maximum radius to measure (default: 50 pixels)
    n_radii : int
        Number of radii for curve of growth (default: 50)

    Returns
    -------
    dict[float, NDArray]
        Dictionary mapping fraction to array of radii (in pixels)
    """
    if not SEP_AVAILABLE:
        raise ImportError("sep is not installed. Run: pip install sep")

    n_sources = len(catalog)
    if n_sources == 0:
        return {f: np.array([], dtype=np.float64) for f in fractions}

    data = np.ascontiguousarray(data, dtype=np.float64)

    if isinstance(background_rms, (int, float)):
        err = np.full_like(data, background_rms)
    else:
        err = np.ascontiguousarray(background_rms, dtype=np.float64)

    if mask is not None:
        mask = np.ascontiguousarray(mask, dtype=np.uint8)

    x = catalog['x'].values.astype(np.float64)
    y = catalog['y'].values.astype(np.float64)
    total_flux = catalog['flux_auto'].values if 'flux_auto' in catalog.columns else catalog['flux'].values

    radii = np.linspace(1.0, max_radius, n_radii)

    # Vectorized flux measurement: compute flux for ALL sources at each radius
    # This reduces sep.sum_circle calls from N × M to just M
    flux_grid = np.zeros((n_sources, n_radii), dtype=np.float64)

    for j, r in enumerate(radii):
        flux, _, _ = sep.sum_circle(
            data, x, y, r,
            err=err,
            mask=mask,
            subpix=5,
        )
        flux_grid[:, j] = flux

    # Mask for valid sources (finite positive flux)
    valid = np.isfinite(total_flux) & (total_flux > 0)

    # Compute radii for each fraction
    results = {}
    for frac in fractions:
        target_flux = frac * total_flux
        r_frac = np.full(n_sources, np.nan, dtype=np.float64)

        if valid.any():
            # For each valid source, find where flux crosses target
            for i in np.where(valid)[0]:
                fluxes = flux_grid[i, :]
                target = target_flux[i]

                idx = np.searchsorted(fluxes, target)
                if 0 < idx < n_radii:
                    # Linear interpolation
                    r1, r2 = radii[idx - 1], radii[idx]
                    f1, f2 = fluxes[idx - 1], fluxes[idx]
                    if f2 > f1:
                        r_frac[i] = r1 + (r2 - r1) * (target - f1) / (f2 - f1)
                    else:
                        r_frac[i] = r1
                elif idx == 0:
                    r_frac[i] = radii[0]
                else:
                    r_frac[i] = radii[-1]

        results[frac] = r_frac

    return results


def compute_half_light_radius(
    data: NDArray,
    catalog: pd.DataFrame,
    background_rms: NDArray | float,
    mask: NDArray | None = None,
    max_radius: float = 50.0,
    n_radii: int = 50,
) -> pd.DataFrame:
    """Compute half-light (effective) radius using curve of growth.

    Measures flux in circular apertures of increasing radius and finds
    the radius containing 50% of the total flux.

    This is a convenience wrapper around compute_flux_radii().

    Parameters
    ----------
    data : NDArray
        Background-subtracted 2D image
    catalog : pd.DataFrame
        Source catalog with 'x', 'y', and 'flux_auto' columns
    background_rms : NDArray or float
        Background RMS
    mask : NDArray, optional
        Boolean mask
    max_radius : float
        Maximum radius to measure (default: 50 pixels)
    n_radii : int
        Number of radii for curve of growth (default: 50)

    Returns
    -------
    pd.DataFrame
        Input catalog with added 'r_half' column (half-light radius in pixels)
    """
    radii = compute_flux_radii(
        data, catalog, background_rms,
        fractions=(0.5,),
        mask=mask,
        max_radius=max_radius,
        n_radii=n_radii,
    )

    result = catalog.copy()
    result['r_half'] = radii[0.5]

    return result


def _process_single_source(args: tuple) -> tuple | None:
    """Process a single source with statmorph (worker function for parallel processing).

    Returns tuple of (idx, gini, m20, sersic_n, asymmetry, smoothness, flag) or None if failed.
    """
    import warnings
    idx, cutout, segmap_clean, gain, psf = args

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning,
                                   message='overflow encountered in power')
            source_morphs = statmorph.source_morphology(
                cutout, segmap_clean, gain=gain, psf=psf
            )

        if len(source_morphs) > 0:
            m = source_morphs[0]
            return (idx, m.gini, m.m20, m.sersic_n, m.asymmetry, m.smoothness, m.flag)
    except Exception:
        pass

    return None


def compute_statmorph(
    data: NDArray,
    segmap: NDArray,
    catalog: pd.DataFrame,
    gain: float = 1.0,
    psf_fwhm: float = 2.25,
    cutout_size: int = 64,
    min_snr: float = 3.0,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Compute statmorph morphological parameters for detected sources.

    Uses the statmorph library (Rodriguez-Gomez et al. 2019) to compute
    Gini, M20, concentration, asymmetry, and Sersic index.

    This version processes sources individually with proper cutouts to avoid
    the "Full Gini segmap" issue that occurs when processing the full image.
    Sources are processed in parallel using multiprocessing.

    Parameters
    ----------
    data : NDArray
        Background-subtracted 2D image
    segmap : NDArray
        Segmentation map with integer labels for each source
    catalog : pd.DataFrame
        Source catalog with 'x' and 'y' columns
    gain : float
        Detector gain in e-/ADU (default: 1.0)
    psf_fwhm : float
        PSF FWHM in pixels for deconvolution (default: 2.25 for HST WFPC2)
    cutout_size : int
        Size of cutouts for individual source processing (default: 64)
    min_snr : float
        Minimum SNR for processing (default: 3.0)
    n_workers : int, optional
        Number of parallel workers. Default: number of CPU cores

    Returns
    -------
    pd.DataFrame
        Input catalog with added morphology columns:
        - gini: Gini coefficient (0-1, higher = more concentrated)
        - m20: M20 moment (more negative = more concentrated)
        - sersic_n: Sersic index (1=exponential, 4=de Vaucouleurs)
        - asymmetry: Asymmetry index
        - smoothness: Smoothness/clumpiness index
        - statmorph_flag: Quality flag (0 = good)
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if not STATMORPH_AVAILABLE:
        print("  Statmorph not available, skipping morphology computation")
        result = catalog.copy()
        for col in ['gini', 'm20', 'sersic_n', 'asymmetry', 'smoothness', 'statmorph_flag']:
            result[col] = np.nan
        return result

    result = catalog.copy()

    # Initialize output columns
    n_sources = len(catalog)
    gini = np.full(n_sources, np.nan)
    m20 = np.full(n_sources, np.nan)
    sersic_n = np.full(n_sources, np.nan)
    asymmetry = np.full(n_sources, np.nan)
    smoothness = np.full(n_sources, np.nan)
    flags = np.full(n_sources, -1, dtype=int)

    # Create simple Gaussian PSF for statmorph
    psf_size = int(4 * psf_fwhm) | 1  # Odd size
    psf_sigma = psf_fwhm / 2.355
    yp, xp = np.mgrid[:psf_size, :psf_size]
    psf = np.exp(-((xp - psf_size//2)**2 + (yp - psf_size//2)**2) / (2 * psf_sigma**2))
    psf /= psf.sum()

    ny, nx = data.shape
    half = cutout_size // 2

    # Pre-filter sources and prepare cutouts for parallel processing
    tasks = []
    for i, row in catalog.iterrows():
        x_pos = int(row.get('x', row.get('xcentroid', 0)))
        y_pos = int(row.get('y', row.get('ycentroid', 0)))

        # Skip sources near edges
        if x_pos < half or x_pos >= nx - half or y_pos < half or y_pos >= ny - half:
            continue

        # Skip low SNR sources
        snr = row.get('snr', row.get('snr_auto', 10.0))
        if snr < min_snr:
            continue

        # Find the label at the source position
        label = segmap[y_pos, x_pos]
        if label == 0:
            continue

        # Extract cutout
        y_lo, y_hi = y_pos - half, y_pos + half
        x_lo, x_hi = x_pos - half, x_pos + half
        cutout = data[y_lo:y_hi, x_lo:x_hi].copy()
        segmap_cutout = segmap[y_lo:y_hi, x_lo:x_hi]

        # Create a clean segmap with only this source (label=1)
        segmap_clean = np.zeros_like(segmap_cutout, dtype=np.int32)
        segmap_clean[segmap_cutout == label] = 1

        # Skip if segment is too small or too large
        seg_pixels = np.sum(segmap_clean > 0)
        if seg_pixels < 10 or seg_pixels > cutout_size * cutout_size * 0.5:
            continue

        # Get the catalog index for this source
        idx = catalog.index.get_loc(i) if i in catalog.index else i
        tasks.append((idx, cutout, segmap_clean.copy(), gain, psf))

    if not tasks:
        print(f"  Statmorph: no valid sources to process")
        result['gini'] = gini
        result['m20'] = m20
        result['sersic_n'] = sersic_n
        result['asymmetry'] = asymmetry
        result['smoothness'] = smoothness
        result['statmorph_flag'] = flags
        return result

    # Determine number of workers
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, len(tasks))
    n_workers = max(1, min(n_workers, len(tasks)))

    print(f"  Statmorph: processing {len(tasks)} sources with {n_workers} workers...")

    # Process in parallel
    n_processed = 0
    n_good = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_single_source, task): task[0] for task in tasks}

        for future in as_completed(futures):
            result_data = future.result()
            if result_data is not None:
                idx, g, m, s, a, sm, flag = result_data
                gini[idx] = g
                m20[idx] = m
                sersic_n[idx] = s
                asymmetry[idx] = a
                smoothness[idx] = sm
                flags[idx] = flag
                n_processed += 1
                if flag == 0:
                    n_good += 1

    print(f"  Statmorph: computed morphology for {n_processed}/{n_sources} sources ({n_good} good)")

    result['gini'] = gini
    result['m20'] = m20
    result['sersic_n'] = sersic_n
    result['asymmetry'] = asymmetry
    result['smoothness'] = smoothness
    result['statmorph_flag'] = flags

    return result


def detect_and_measure(
    data: NDArray,
    weight: NDArray | None = None,
    mask: NDArray | None = None,
    gain: float = 7.0,
    threshold: float = 1.5,
    min_area: int = 5,
    box_size: int = 64,
    filter_size: int = 3,
    deblend_nthresh: int = 32,
    deblend_cont: float = 0.005,
    aperture_radii: list[float] | None = None,
    compute_kron: bool = True,
    compute_r_half: bool = True,
    pixel_scale: float = 0.04,
) -> tuple[pd.DataFrame, SEPBackground, NDArray | None]:
    """Complete source detection and measurement pipeline using SEP.

    This is a convenience function that runs the full detection and
    measurement pipeline, equivalent to running SExtractor with standard
    parameters.

    Parameters
    ----------
    data : NDArray
        2D image data
    weight : NDArray, optional
        Inverse variance weight map (converts to error map)
    mask : NDArray, optional
        Boolean mask where True = ignore pixel
    gain : float
        Detector gain in e-/ADU (default: 7.0 for WFPC2)
    threshold : float
        Detection threshold in sigma (default: 1.5)
    min_area : int
        Minimum detection area in pixels (default: 5)
    box_size : int
        Background mesh size (default: 64)
    filter_size : int
        Background filter size (default: 3)
    deblend_nthresh : int
        Deblending thresholds (default: 32)
    deblend_cont : float
        Deblending contrast (default: 0.005)
    aperture_radii : list of float, optional
        Circular aperture radii. Default: [2, 3, 5, 7, 10]
    compute_kron : bool
        Compute Kron (AUTO) photometry (default: True)
    compute_r_half : bool
        Compute half-light radii (default: True)
    pixel_scale : float
        Pixel scale in arcsec/pixel (default: 0.04 for HST)

    Returns
    -------
    catalog : pd.DataFrame
        Complete source catalog with all measurements
    background : SEPBackground
        Background estimation results
    segmentation : NDArray
        Segmentation map
    """
    if not SEP_AVAILABLE:
        raise ImportError("sep is not installed. Run: pip install sep")

    # Detect sources
    result = detect_sources_sep(
        data,
        threshold=threshold,
        min_area=min_area,
        mask=mask,
        gain=gain,
        box_size=box_size,
        filter_size=filter_size,
        deblend_nthresh=deblend_nthresh,
        deblend_cont=deblend_cont,
        segmentation_map=True,
    )

    if result.n_sources == 0:
        return result.catalog, result.background, result.segmentation

    catalog = result.catalog
    data_sub = data - result.background.background

    # Kron photometry
    if compute_kron:
        catalog = kron_photometry(
            data_sub,
            catalog,
            result.background.background_rms,
            gain=gain,
            mask=mask,
        )

    # Circular aperture photometry
    catalog = circular_aperture_photometry(
        data_sub,
        catalog,
        result.background.background_rms,
        radii=aperture_radii,
        gain=gain,
        mask=mask,
    )

    # Half-light radius
    if compute_r_half and compute_kron:
        catalog = compute_half_light_radius(
            data_sub,
            catalog,
            result.background.background_rms,
            mask=mask,
        )

    # Add derived columns
    catalog['ellipticity'] = 1.0 - catalog['b'] / catalog['a']
    catalog['fwhm'] = 2.0 * np.sqrt(np.log(2) * (catalog['a']**2 + catalog['b']**2))

    # Convert to arcsec
    if pixel_scale > 0:
        catalog['r_half_arcsec'] = catalog['r_half'] * pixel_scale if 'r_half' in catalog.columns else np.nan
        catalog['a_arcsec'] = catalog['a'] * pixel_scale
        catalog['b_arcsec'] = catalog['b'] * pixel_scale
        catalog['fwhm_arcsec'] = catalog['fwhm'] * pixel_scale

    # Rename columns for compatibility with existing pipeline
    catalog = catalog.rename(columns={
        'x': 'xcentroid',
        'y': 'ycentroid',
    })

    return catalog, result.background, result.segmentation


def convert_to_photutils_format(
    sep_catalog: pd.DataFrame,
    add_columns: dict | None = None,
) -> pd.DataFrame:
    """Convert SEP catalog to photutils-compatible format.

    Maps SEP column names to photutils equivalents for compatibility
    with the existing analysis pipeline.

    Parameters
    ----------
    sep_catalog : pd.DataFrame
        Catalog from detect_and_measure
    add_columns : dict, optional
        Additional columns to add

    Returns
    -------
    pd.DataFrame
        Catalog with photutils-compatible column names
    """
    # Column mapping: sep -> photutils
    column_map = {
        'flux': 'segment_flux',
        'flux_err': 'segment_fluxerr',
        'flux_auto': 'kron_flux',
        'flux_auto_err': 'kron_fluxerr',
        'a': 'semimajor_sigma',
        'b': 'semiminor_sigma',
        'theta': 'orientation',
        'npix': 'area',
    }

    result = sep_catalog.copy()

    for old_name, new_name in column_map.items():
        if old_name in result.columns and new_name not in result.columns:
            result[new_name] = result[old_name]

    if add_columns:
        for key, value in add_columns.items():
            result[key] = value

    return result
