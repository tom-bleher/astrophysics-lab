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
    from photutils.aperture import CircularAperture, aperture_photometry

    # Measure cumulative flux profile
    radii = np.arange(1, max_radius, 1)
    cumulative_flux = np.zeros(len(radii))

    for i, r in enumerate(radii):
        aper = CircularAperture((x, y), r=r)
        phot = aperture_photometry(image, aper)
        cumulative_flux[i] = phot["aperture_sum"][0]

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
    from photutils.aperture import (
        CircularAperture,
        CircularAnnulus,
        aperture_photometry,
    )

    radii = np.arange(2, max_radius, 1)
    petrosian_ratio = np.zeros(len(radii))

    for i, r in enumerate(radii):
        # Surface brightness at radius r (using thin annulus)
        dr = 1.0
        annulus = CircularAnnulus((x, y), r_in=r - dr / 2, r_out=r + dr / 2)
        aper = CircularAperture((x, y), r=r)

        phot_ann = aperture_photometry(image, annulus)
        phot_aper = aperture_photometry(image, aper)

        flux_ann = phot_ann["aperture_sum"][0]
        flux_aper = phot_aper["aperture_sum"][0]

        area_ann = annulus.area
        area_aper = aper.area

        if area_ann > 0 and flux_aper > 0:
            sb_local = flux_ann / area_ann
            sb_mean = flux_aper / area_aper
            petrosian_ratio[i] = sb_local / sb_mean
        else:
            petrosian_ratio[i] = np.nan

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
    from photutils.aperture import CircularAperture, aperture_photometry

    radii = np.arange(1, max_radius, 0.5)
    cumulative_flux = np.zeros(len(radii))

    for i, r in enumerate(radii):
        aper = CircularAperture((x, y), r=r)
        phot = aperture_photometry(image, aper)
        cumulative_flux[i] = phot["aperture_sum"][0]

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
    if mask is not None:
        values = image[mask].flatten()
    else:
        values = image.flatten()

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
