"""Concentration index and Petrosian radius measurements.

The concentration index C = 5 * log10(r_80 / r_20) is a non-parametric
measure of light profile shape, useful for:
- Star/galaxy classification (stars have higher C)
- Morphological classification (bulge-dominated vs disk-dominated)

Petrosian radius is the radius where the local surface brightness
equals a fixed fraction of the mean interior surface brightness.

References:
- Abraham et al. 1994, ApJ, 432, 75
- Petrosian 1976, ApJ, 209, L1
- Bershady et al. 2000, AJ, 119, 2645
"""

import numpy as np
from numpy.typing import NDArray

# Optional Numba for JIT compilation (10-50x speedup)
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback: define no-op decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(*args):
        return range(*args)


def _compute_cumulative_radial_profile(
    image: NDArray,
    x: float,
    y: float,
    max_radius: float = 50.0,
    step: float = 1.0,
) -> tuple[NDArray, NDArray]:
    """Compute cumulative flux profile using vectorized radial binning.

    This is much faster than calling aperture_photometry in a loop.

    Parameters
    ----------
    image : NDArray
        2D image array
    x, y : float
        Source center position
    max_radius : float
        Maximum radius to compute
    step : float
        Radial step size

    Returns
    -------
    radii : NDArray
        Array of radii
    cumulative_flux : NDArray
        Cumulative flux within each radius
    """
    # Get image dimensions and compute cutout bounds
    ny, nx = image.shape

    # Check if coordinates are valid
    if not (np.isfinite(x) and np.isfinite(y)):
        radii = np.arange(step, max_radius + step, step)
        return radii, np.zeros(len(radii))

    # Limit max_radius based on distance to edge to avoid edge effects
    dist_to_edge = min(x, y, nx - 1 - x, ny - 1 - y)
    effective_max_radius = min(max_radius, max(dist_to_edge - 2, 5.0))  # At least 5 pixels

    r_int = int(effective_max_radius) + 1

    # Define cutout region (with bounds checking)
    y_min = max(0, int(y) - r_int)
    y_max = min(ny, int(y) + r_int + 1)
    x_min = max(0, int(x) - r_int)
    x_max = min(nx, int(x) + r_int + 1)

    # Check cutout size is valid
    if y_max - y_min < 3 or x_max - x_min < 3:
        radii = np.arange(step, max_radius + step, step)
        return radii, np.zeros(len(radii))

    # Extract cutout
    cutout = image[y_min:y_max, x_min:x_max].copy()

    # Handle NaN values - replace with local background estimate
    nan_mask = ~np.isfinite(cutout)
    if np.any(nan_mask):
        valid_values = cutout[~nan_mask]
        if len(valid_values) > 0:
            background = np.median(valid_values)
            cutout[nan_mask] = background
        else:
            radii = np.arange(step, max_radius + step, step)
            return radii, np.zeros(len(radii))

    # Estimate and subtract local background (outer annulus)
    cy = y - y_min
    cx = x - x_min
    Y, X = np.ogrid[: cutout.shape[0], : cutout.shape[1]]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # Use outer annulus for background estimation (r > 0.8 * effective_max_radius)
    outer_mask = R > 0.8 * effective_max_radius
    if np.sum(outer_mask) > 10:  # Need enough pixels for background
        background = np.median(cutout[outer_mask])
        cutout = cutout - background

    # Compute cumulative flux at each radius using vectorized operations
    radii = np.arange(step, max_radius + step, step)
    cumulative_flux = np.zeros(len(radii))

    for i, r in enumerate(radii):
        if r <= effective_max_radius:
            mask = r >= R
            cumulative_flux[i] = np.sum(cutout[mask])
        elif i > 0:
            # Extrapolate for radii beyond effective max
            cumulative_flux[i] = cumulative_flux[i-1]

    return radii, cumulative_flux


def _compute_radial_profile(
    image: NDArray,
    x: float,
    y: float,
    max_radius: float = 50.0,
    annulus_width: float = 1.0,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute radial surface brightness profile using vectorized binning.

    Parameters
    ----------
    image : NDArray
        2D image array
    x, y : float
        Source center position
    max_radius : float
        Maximum radius to compute
    annulus_width : float
        Width of radial bins

    Returns
    -------
    radii : NDArray
        Array of radii (bin centers)
    sb_local : NDArray
        Local surface brightness at each radius
    sb_mean : NDArray
        Mean surface brightness within each radius
    """
    # Get image dimensions and compute cutout bounds
    ny, nx = image.shape
    r_int = int(max_radius) + 1

    # Define cutout region (with bounds checking)
    y_min = max(0, int(y) - r_int)
    y_max = min(ny, int(y) + r_int + 1)
    x_min = max(0, int(x) - r_int)
    x_max = min(nx, int(x) + r_int + 1)

    # Extract cutout
    cutout = image[y_min:y_max, x_min:x_max]

    # Create coordinate grids relative to source center
    cy = y - y_min
    cx = x - x_min
    Y, X = np.ogrid[: cutout.shape[0], : cutout.shape[1]]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # Compute profiles using vectorized operations
    radii = np.arange(annulus_width, max_radius + annulus_width, annulus_width)
    sb_local = np.zeros(len(radii))
    sb_mean = np.zeros(len(radii))


    for i, r in enumerate(radii):
        # Annulus mask
        r_in = r - annulus_width / 2
        r_out = r + annulus_width / 2
        annulus_mask = (r_in <= R) & (r_out > R)
        aperture_mask = r >= R

        # Local surface brightness (in annulus)
        ann_area = np.sum(annulus_mask)
        if ann_area > 0:
            sb_local[i] = np.sum(cutout[annulus_mask]) / ann_area
        else:
            sb_local[i] = np.nan

        # Mean surface brightness (within aperture)
        aper_area = np.sum(aperture_mask)
        if aper_area > 0:
            sb_mean[i] = np.sum(cutout[aperture_mask]) / aper_area
        else:
            sb_mean[i] = np.nan

    return radii, sb_local, sb_mean


def concentration_index(
    image: NDArray,
    x: float,
    y: float,
    r_inner: float = 3.0,
    r_outer: float = 10.0,
) -> float:
    """Calculate concentration index using fixed apertures.

    This is a simplified concentration measure using the ratio
    of flux in two circular apertures.

    Parameters
    ----------
    image : NDArray
        2D image array
    x, y : float
        Source center position
    r_inner : float
        Inner aperture radius in pixels
    r_outer : float
        Outer aperture radius in pixels

    Returns
    -------
    float
        Concentration index (-2.5 * log10(flux_inner / flux_outer))
    """
    from photutils.aperture import CircularAperture, aperture_photometry

    aper_inner = CircularAperture((x, y), r=r_inner)
    aper_outer = CircularAperture((x, y), r=r_outer)

    phot_inner = aperture_photometry(image, aper_inner)
    phot_outer = aperture_photometry(image, aper_outer)

    flux_inner = phot_inner["aperture_sum"][0]
    flux_outer = phot_outer["aperture_sum"][0]

    if flux_outer > 0 and flux_inner > 0:
        return -2.5 * np.log10(flux_inner / flux_outer)

    return np.nan


def calculate_concentration_c(
    image: NDArray,
    x: float,
    y: float,
    eta_20: float = 0.2,
    eta_80: float = 0.8,
    max_radius: float = 50.0,
) -> float:
    """Calculate standard concentration index C = 5 * log10(r_80 / r_20).

    This measures the radii enclosing 20% and 80% of the total light.

    Parameters
    ----------
    image : NDArray
        2D image array
    x, y : float
        Source center position
    eta_20 : float
        Fraction for inner radius (default: 0.2)
    eta_80 : float
        Fraction for outer radius (default: 0.8)
    max_radius : float
        Maximum radius to search in pixels

    Returns
    -------
    float
        Concentration index C
    """
    # Use vectorized radial profile computation (much faster than aperture_photometry loop)
    radii, cumulative_flux = _compute_cumulative_radial_profile(
        image, x, y, max_radius=max_radius, step=1.0
    )

    # Normalize to total flux
    if cumulative_flux[-1] <= 0:
        return np.nan

    cumulative_frac = cumulative_flux / cumulative_flux[-1]

    # Find radii enclosing eta_20 and eta_80 of the light
    r_20 = np.interp(eta_20, cumulative_frac, radii)
    r_80 = np.interp(eta_80, cumulative_frac, radii)

    if r_20 <= 0 or r_80 <= 0:
        return np.nan

    return 5.0 * np.log10(r_80 / r_20)


def petrosian_radius(
    image: NDArray,
    x: float,
    y: float,
    eta: float = 0.2,
    max_radius: float = 50.0,
) -> float:
    """Calculate Petrosian radius.

    The Petrosian radius r_P is defined where:
        I(r_P) / <I(r<r_P)> = eta

    where I(r) is the surface brightness at radius r and <I(r<r_P)>
    is the mean surface brightness within r_P.

    Parameters
    ----------
    image : NDArray
        2D image array
    x, y : float
        Source center position
    eta : float
        Petrosian ratio (default: 0.2)
    max_radius : float
        Maximum radius to search

    Returns
    -------
    float
        Petrosian radius in pixels
    """
    # Use vectorized radial profile computation (much faster than aperture_photometry loop)
    radii, sb_local, sb_mean = _compute_radial_profile(
        image, x, y, max_radius=max_radius, annulus_width=1.0
    )

    # Skip first radius (too small for reliable ratio)
    radii = radii[1:]
    sb_local = sb_local[1:]
    sb_mean = sb_mean[1:]

    # Compute Petrosian ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        petrosian_ratio = sb_local / sb_mean

    # Find radius where ratio equals eta
    valid = np.isfinite(petrosian_ratio)
    if not np.any(valid):
        return np.nan

    # Interpolate to find crossing point
    r_valid = radii[valid]
    pr_valid = petrosian_ratio[valid]

    # Find where ratio crosses eta (from above)
    crossings = np.where(
        (pr_valid[:-1] > eta) & (pr_valid[1:] < eta)
    )[0]

    if len(crossings) == 0:
        # Ratio never crosses eta - return max radius where ratio > eta
        above_eta = pr_valid > eta
        if np.any(above_eta):
            return r_valid[np.where(above_eta)[0][-1]]
        return np.nan

    # Interpolate to get exact crossing
    i = crossings[0]
    r_p = np.interp(eta, [pr_valid[i + 1], pr_valid[i]], [r_valid[i + 1], r_valid[i]])

    return r_p


def half_light_radius(
    image: NDArray,
    x: float,
    y: float,
    total_flux: float | None = None,
    max_radius: float = 50.0,
) -> float:
    """Calculate half-light radius.

    Parameters
    ----------
    image : NDArray
        2D image array
    x, y : float
        Source center position
    total_flux : float, optional
        Total flux. If None, uses flux within max_radius.
    max_radius : float
        Maximum radius to search

    Returns
    -------
    float
        Half-light radius in pixels
    """
    # Use vectorized radial profile computation (much faster than aperture_photometry loop)
    radii, cumulative_flux = _compute_cumulative_radial_profile(
        image, x, y, max_radius=max_radius, step=0.5
    )

    if total_flux is None:
        total_flux = cumulative_flux[-1]

    if total_flux <= 0:
        return np.nan

    # Find radius enclosing 50% of flux
    return np.interp(0.5 * total_flux, cumulative_flux, radii)


def asymmetry_index(
    image: NDArray,
    x: float,
    y: float,
    radius: float = 15.0,
) -> float:
    """Calculate asymmetry index A.

    A = sum(|I - I_180|) / (2 * sum(|I|))

    where I_180 is the image rotated 180 degrees about the center.

    Parameters
    ----------
    image : NDArray
        2D image array
    x, y : float
        Source center position
    radius : float
        Radius for asymmetry calculation

    Returns
    -------
    float
        Asymmetry index (0 = symmetric, 1 = completely asymmetric)
    """
    from scipy.ndimage import rotate

    # Extract cutout
    r_int = int(radius)
    x_int, y_int = int(x), int(y)

    y_min = max(0, y_int - r_int)
    y_max = min(image.shape[0], y_int + r_int)
    x_min = max(0, x_int - r_int)
    x_max = min(image.shape[1], x_int + r_int)

    cutout = image[y_min:y_max, x_min:x_max].copy()

    # Create circular mask
    Y, X = np.ogrid[: cutout.shape[0], : cutout.shape[1]]
    cx = x - x_min
    cy = y - y_min
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    mask = r2 <= radius**2

    # Rotate 180 degrees
    rotated = rotate(cutout, 180, reshape=False, order=1)

    # Calculate asymmetry
    diff = np.abs(cutout - rotated)
    total = np.abs(cutout) + np.abs(rotated)

    if np.sum(total[mask]) > 0:
        return np.sum(diff[mask]) / np.sum(total[mask])

    return np.nan


def gini_coefficient(
    image: NDArray,
    mask: NDArray | None = None,
) -> float:
    """Calculate Gini coefficient of pixel values.

    The Gini coefficient measures inequality in pixel flux distribution.
    G = 0 means all pixels have equal flux, G = 1 means all flux
    is in one pixel.

    Parameters
    ----------
    image : NDArray
        2D image array
    mask : NDArray, optional
        Boolean mask for pixels to include

    Returns
    -------
    float
        Gini coefficient (0 to 1)
    """
    values = image[mask].flatten() if mask is not None else image.flatten()

    # Remove negative values
    values = values[values > 0]

    if len(values) < 2:
        return np.nan

    # Sort values
    values = np.sort(values)
    n = len(values)

    # Gini formula
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n


def m20_statistic(
    image: NDArray,
    x: float,
    y: float,
    total_flux: float | None = None,
) -> float:
    """Calculate M20 statistic.

    M20 is the normalized second-order moment of the brightest
    20% of the galaxy's flux.

    Parameters
    ----------
    image : NDArray
        2D image array
    x, y : float
        Source center
    total_flux : float, optional
        Total flux

    Returns
    -------
    float
        M20 statistic
    """
    if total_flux is None:
        total_flux = np.sum(image[image > 0])

    if total_flux <= 0:
        return np.nan

    # Sort pixels by flux
    Y, X = np.ogrid[: image.shape[0], : image.shape[1]]
    flat_image = image.flatten()
    flat_x = X.flatten()
    flat_y = Y.flatten()

    # Get indices sorted by flux (descending)
    sort_idx = np.argsort(flat_image)[::-1]

    # Find pixels containing brightest 20% of flux
    cumsum = np.cumsum(flat_image[sort_idx])
    n_bright = np.searchsorted(cumsum, 0.2 * total_flux) + 1

    bright_idx = sort_idx[:n_bright]

    # Calculate second moment
    x_bright = flat_x[bright_idx]
    y_bright = flat_y[bright_idx]
    f_bright = flat_image[bright_idx]

    m_tot = np.sum(f_bright * ((x_bright - x) ** 2 + (y_bright - y) ** 2))

    # Normalize by total second moment
    m_total = np.sum(
        flat_image[flat_image > 0]
        * ((flat_x[flat_image.flatten() > 0] - x) ** 2 + (flat_y[flat_image.flatten() > 0] - y) ** 2)
    )

    if m_total > 0:
        return np.log10(m_tot / m_total)

    return np.nan


# =============================================================================
# Batch processing functions for vectorized operations on multiple sources
# =============================================================================


def concentration_index_batch(
    image: NDArray,
    x_coords: NDArray,
    y_coords: NDArray,
    r_inner: float = 3.0,
    r_outer: float = 10.0,
) -> NDArray:
    """Calculate concentration index for multiple sources.

    Vectorized version that processes all sources efficiently.

    Parameters
    ----------
    image : NDArray
        2D image array
    x_coords, y_coords : NDArray
        Source center positions (1D arrays of same length)
    r_inner : float
        Inner aperture radius in pixels
    r_outer : float
        Outer aperture radius in pixels

    Returns
    -------
    NDArray
        Concentration indices for each source
    """
    n_sources = len(x_coords)
    concentrations = np.full(n_sources, np.nan)

    for i in range(n_sources):
        x, y = x_coords[i], y_coords[i]
        concentrations[i] = concentration_index(image, x, y, r_inner, r_outer)

    return concentrations


def calculate_concentration_c_batch(
    image: NDArray,
    x_coords: NDArray,
    y_coords: NDArray,
    eta_20: float = 0.2,
    eta_80: float = 0.8,
    max_radius: float = 50.0,
) -> NDArray:
    """Calculate standard concentration index C for multiple sources.

    Vectorized version that processes all sources efficiently.

    Parameters
    ----------
    image : NDArray
        2D image array
    x_coords, y_coords : NDArray
        Source center positions (1D arrays of same length)
    eta_20, eta_80 : float
        Fractions for inner and outer radii
    max_radius : float
        Maximum radius to search in pixels

    Returns
    -------
    NDArray
        Concentration indices C for each source
    """
    n_sources = len(x_coords)
    concentrations = np.full(n_sources, np.nan)

    for i in range(n_sources):
        x, y = x_coords[i], y_coords[i]
        concentrations[i] = calculate_concentration_c(
            image, x, y, eta_20, eta_80, max_radius
        )

    return concentrations


def half_light_radius_batch(
    image: NDArray,
    x_coords: NDArray,
    y_coords: NDArray,
    total_fluxes: NDArray | None = None,
    max_radius: float = 50.0,
) -> NDArray:
    """Calculate half-light radius for multiple sources.

    Vectorized version that processes all sources efficiently.

    Parameters
    ----------
    image : NDArray
        2D image array
    x_coords, y_coords : NDArray
        Source center positions (1D arrays of same length)
    total_fluxes : NDArray, optional
        Total flux for each source. If None, computed from image.
    max_radius : float
        Maximum radius to search

    Returns
    -------
    NDArray
        Half-light radii for each source
    """
    n_sources = len(x_coords)
    radii = np.full(n_sources, np.nan)

    for i in range(n_sources):
        x, y = x_coords[i], y_coords[i]
        total_flux = total_fluxes[i] if total_fluxes is not None else None
        radii[i] = half_light_radius(image, x, y, total_flux, max_radius)

    return radii


def compute_morphology_batch(
    image: NDArray,
    x_coords: NDArray,
    y_coords: NDArray,
    max_radius: float = 50.0,
) -> dict[str, NDArray]:
    """Compute all morphological parameters for multiple sources at once.

    This is the most efficient way to compute multiple morphological
    parameters for a catalog of sources.

    Parameters
    ----------
    image : NDArray
        2D image array
    x_coords, y_coords : NDArray
        Source center positions (1D arrays of same length)
    max_radius : float
        Maximum radius to search

    Returns
    -------
    dict[str, NDArray]
        Dictionary with keys:
        - 'concentration': Simple concentration index
        - 'concentration_c': Standard C = 5*log10(r80/r20)
        - 'half_light_radius': Half-light radius in pixels
    """
    n_sources = len(x_coords)

    result = {
        'concentration': np.full(n_sources, np.nan),
        'concentration_c': np.full(n_sources, np.nan),
        'half_light_radius': np.full(n_sources, np.nan),
    }

    ny, nx = image.shape

    for i in range(n_sources):
        x, y = x_coords[i], y_coords[i]

        # Skip sources with invalid coordinates
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        # Skip sources too close to edge (need at least 5 pixels)
        if x < 5 or y < 5 or x > nx - 6 or y > ny - 6:
            continue

        try:
            # Use single profile computation for all metrics
            radii, cumulative_flux = _compute_cumulative_radial_profile(
                image, x, y, max_radius=max_radius, step=0.5
            )

            # Skip if no positive flux (after background subtraction)
            total_flux = cumulative_flux[-1]
            if total_flux <= 0:
                # Try with smaller radius for faint/compact sources
                radii_small, flux_small = _compute_cumulative_radial_profile(
                    image, x, y, max_radius=15.0, step=0.5
                )
                if flux_small[-1] > 0:
                    radii, cumulative_flux = radii_small, flux_small
                    total_flux = cumulative_flux[-1]
                else:
                    continue

            # Half-light radius
            half_flux = 0.5 * total_flux
            # Ensure cumulative_flux is monotonically increasing for interpolation
            if np.all(np.diff(cumulative_flux) >= 0):
                result['half_light_radius'][i] = np.interp(half_flux, cumulative_flux, radii)
            else:
                # Find first radius where we exceed half flux
                idx = np.searchsorted(cumulative_flux, half_flux)
                if idx < len(radii):
                    result['half_light_radius'][i] = radii[min(idx, len(radii)-1)]

            # Concentration C (need cumulative fractions)
            cumulative_frac = cumulative_flux / total_flux
            # Handle edge case where cumulative_frac may not reach 0.2 or 0.8
            if cumulative_frac[-1] >= 0.8 and cumulative_frac[0] <= 0.2:
                r_20 = np.interp(0.2, cumulative_frac, radii)
                r_80 = np.interp(0.8, cumulative_frac, radii)
                if r_20 > 0.1 and r_80 > r_20:  # Sanity checks
                    result['concentration_c'][i] = 5.0 * np.log10(r_80 / r_20)

            # Simple concentration (ratio of inner/outer flux)
            r_inner_idx = np.searchsorted(radii, 3.0)
            r_outer_idx = np.searchsorted(radii, 10.0)
            if r_inner_idx < len(cumulative_flux) and r_outer_idx < len(cumulative_flux):
                flux_inner = cumulative_flux[r_inner_idx]
                flux_outer = cumulative_flux[r_outer_idx]
                if flux_inner > 0 and flux_outer > flux_inner:  # Outer must be larger
                    result['concentration'][i] = -2.5 * np.log10(flux_inner / flux_outer)

        except Exception:
            # Skip sources that cause any computation errors
            continue

    return result


# =============================================================================
# JIT-COMPILED FAST PATH (10-50x speedup with Numba)
# =============================================================================

if HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _compute_source_morphology_jit(
        image: np.ndarray,
        x: float,
        y: float,
        max_radius: float,
    ) -> tuple[float, float, float]:
        """JIT-compiled morphology computation for a single source.

        Returns (concentration, concentration_c, half_light_radius).
        """
        ny, nx = image.shape
        r_int = int(max_radius) + 1

        # Bounds check
        y_min = max(0, int(y) - r_int)
        y_max = min(ny, int(y) + r_int + 1)
        x_min = max(0, int(x) - r_int)
        x_max = min(nx, int(x) + r_int + 1)

        # Extract cutout
        cutout = image[y_min:y_max, x_min:x_max]
        if cutout.size == 0:
            return np.nan, np.nan, np.nan

        cy = y - y_min
        cx = x - x_min
        h, w = cutout.shape

        # Compute cumulative flux at key radii
        n_radii = int(max_radius * 2)  # 0.5 step
        radii = np.empty(n_radii, dtype=np.float64)
        cumulative_flux = np.empty(n_radii, dtype=np.float64)

        for r_idx in range(n_radii):
            r = (r_idx + 1) * 0.5
            radii[r_idx] = r
            r_sq = r * r
            flux_sum = 0.0

            for yi in range(h):
                for xi in range(w):
                    dist_sq = (xi - cx) ** 2 + (yi - cy) ** 2
                    if dist_sq <= r_sq:
                        flux_sum += cutout[yi, xi]

            cumulative_flux[r_idx] = flux_sum

        total_flux = cumulative_flux[-1]
        if total_flux <= 0:
            return np.nan, np.nan, np.nan

        # Half-light radius (linear interpolation)
        half_flux = 0.5 * total_flux
        hlr = np.nan
        for i in range(n_radii - 1):
            if cumulative_flux[i] <= half_flux <= cumulative_flux[i + 1]:
                t = (half_flux - cumulative_flux[i]) / (cumulative_flux[i + 1] - cumulative_flux[i] + 1e-10)
                hlr = radii[i] + t * (radii[i + 1] - radii[i])
                break

        # Concentration C = 5 * log10(r_80 / r_20)
        cumulative_frac = cumulative_flux / total_flux
        r_20 = np.nan
        r_80 = np.nan
        for i in range(n_radii - 1):
            if cumulative_frac[i] <= 0.2 <= cumulative_frac[i + 1]:
                t = (0.2 - cumulative_frac[i]) / (cumulative_frac[i + 1] - cumulative_frac[i] + 1e-10)
                r_20 = radii[i] + t * (radii[i + 1] - radii[i])
            if cumulative_frac[i] <= 0.8 <= cumulative_frac[i + 1]:
                t = (0.8 - cumulative_frac[i]) / (cumulative_frac[i + 1] - cumulative_frac[i] + 1e-10)
                r_80 = radii[i] + t * (radii[i + 1] - radii[i])

        conc_c = np.nan
        if r_20 > 0 and r_80 > 0:
            conc_c = 5.0 * np.log10(r_80 / r_20)

        # Simple concentration (r=3 / r=10)
        # Find flux at r=3 and r=10
        flux_3 = np.nan
        flux_10 = np.nan
        for i in range(n_radii):
            if radii[i] >= 3.0 and np.isnan(flux_3):
                flux_3 = cumulative_flux[i]
            if radii[i] >= 10.0 and np.isnan(flux_10):
                flux_10 = cumulative_flux[i]
                break

        conc = np.nan
        if flux_3 > 0 and flux_10 > 0:
            conc = -2.5 * np.log10(flux_3 / flux_10)

        return conc, conc_c, hlr


    @njit(parallel=True, cache=True, fastmath=True)
    def _compute_morphology_batch_jit(
        image: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        max_radius: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """JIT-compiled parallel morphology batch computation.

        Uses numba.prange for true parallelism across CPU cores.
        Expected speedup: 10-50x vs Python loop version.
        """
        n_sources = len(x_coords)

        concentrations = np.full(n_sources, np.nan, dtype=np.float64)
        conc_c = np.full(n_sources, np.nan, dtype=np.float64)
        hlr = np.full(n_sources, np.nan, dtype=np.float64)

        # Parallel loop over sources
        for i in prange(n_sources):
            conc, cc, r = _compute_source_morphology_jit(
                image, x_coords[i], y_coords[i], max_radius
            )
            concentrations[i] = conc
            conc_c[i] = cc
            hlr[i] = r

        return concentrations, conc_c, hlr


def compute_morphology_batch_fast(
    image: NDArray,
    x_coords: NDArray,
    y_coords: NDArray,
    max_radius: float = 50.0,
) -> dict[str, NDArray]:
    """Fast morphology computation using JIT if available.

    Automatically uses Numba JIT-compiled parallel version if available,
    falling back to pure Python version otherwise.

    Parameters
    ----------
    image : NDArray
        2D image array
    x_coords, y_coords : NDArray
        Source center positions (1D arrays of same length)
    max_radius : float
        Maximum radius to search

    Returns
    -------
    dict[str, NDArray]
        Dictionary with keys:
        - 'concentration': Simple concentration index
        - 'concentration_c': Standard C = 5*log10(r80/r20)
        - 'half_light_radius': Half-light radius in pixels
    """
    x_coords = np.asarray(x_coords, dtype=np.float64)
    y_coords = np.asarray(y_coords, dtype=np.float64)
    image = np.ascontiguousarray(image, dtype=np.float64)

    if HAS_NUMBA:
        conc, conc_c, hlr = _compute_morphology_batch_jit(
            image, x_coords, y_coords, max_radius
        )
        return {
            'concentration': conc,
            'concentration_c': conc_c,
            'half_light_radius': hlr,
        }
    else:
        # Fallback to pure Python version
        return compute_morphology_batch(image, x_coords, y_coords, max_radius)
