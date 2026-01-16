"""Concentration index and morphology measurements.

This module provides galaxy morphology measurements using the statmorph library
(Rodriguez-Gomez et al. 2019).

Primary functions:
- compute_morphology(): Full morphology measurements using statmorph
- concentration_index(): Simple aperture-based concentration
- half_light_radius(): Half-light radius from radial profile

References:
- Rodriguez-Gomez et al. 2019, MNRAS, 483, 4140 (statmorph)
- Abraham et al. 1994, ApJ, 432, 75
"""

import numpy as np
import statmorph
from numpy.typing import NDArray
from photutils.aperture import CircularAperture, aperture_photometry


def compute_morphology(
    image: NDArray,
    segmap: NDArray,
    gain: float = 1.0,
    psf: NDArray | None = None,
) -> dict[str, NDArray]:
    """Compute morphological parameters for all sources using statmorph.

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
        - sersic_n: Sersic index (NaN if fit failed)
        - flag: Quality flag (0 = good)
    """
    source_morphs = statmorph.source_morphology(
        image, segmap, gain=gain, psf=psf
    )

    n = len(source_morphs)
    if n == 0:
        return _empty_morphology_result()

    return {
        "label": np.array([m.label for m in source_morphs], dtype=int),
        "concentration": np.array([m.concentration for m in source_morphs]),
        "asymmetry": np.array([m.asymmetry for m in source_morphs]),
        "smoothness": np.array([m.smoothness for m in source_morphs]),
        "gini": np.array([m.gini for m in source_morphs]),
        "m20": np.array([m.m20 for m in source_morphs]),
        "half_light_radius": np.array([m.rhalf_ellip for m in source_morphs]),
        "sersic_n": np.array([m.sersic_n for m in source_morphs]),
        "flag": np.array([m.flag for m in source_morphs], dtype=int),
    }


def _empty_morphology_result() -> dict[str, NDArray]:
    """Return empty result dict when no sources found."""
    return {
        "label": np.array([], dtype=int),
        "concentration": np.array([]),
        "asymmetry": np.array([]),
        "smoothness": np.array([]),
        "gini": np.array([]),
        "m20": np.array([]),
        "half_light_radius": np.array([]),
        "sersic_n": np.array([]),
        "flag": np.array([], dtype=int),
    }


def concentration_index(
    image: NDArray,
    x: float,
    y: float,
    r_inner: float = 3.0,
    r_outer: float = 10.0,
) -> float:
    """Calculate simple concentration index using aperture photometry.

    Higher values indicate more concentrated (star-like) profiles.
    Lower values indicate more extended (galaxy-like) profiles.

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
        flux_ratio = flux_inner / flux_outer
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

    if not (np.isfinite(x) and np.isfinite(y)):
        return np.nan
    if not (5 <= x < nx - 5 and 5 <= y < ny - 5):
        return np.nan

    r_int = min(int(max_radius), int(min(x, y, nx - x, ny - y)) - 1)
    if r_int < 5:
        return np.nan

    y_min, y_max = int(y) - r_int, int(y) + r_int + 1
    x_min, x_max = int(x) - r_int, int(x) + r_int + 1

    cutout = image[y_min:y_max, x_min:x_max]
    cy, cx = y - y_min, x - x_min

    Y, X = np.ogrid[:cutout.shape[0], :cutout.shape[1]]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)

    radii = np.arange(1, r_int + 1, 0.5)
    cumulative = np.array([np.sum(cutout[R <= r]) for r in radii])

    total = cumulative[-1]
    if total <= 0:
        return np.nan

    return float(np.interp(0.5 * total, cumulative, radii))


def concentration_index_batch(
    image: NDArray,
    x_coords: NDArray,
    y_coords: NDArray,
    r_inner: float = 3.0,
    r_outer: float = 10.0,
) -> NDArray:
    """Calculate concentration index for multiple sources."""
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
    """Calculate half-light radius for multiple sources."""
    return np.array([
        half_light_radius(image, x, y, max_radius)
        for x, y in zip(x_coords, y_coords)
    ])


def compute_morphology_batch(
    image: NDArray,
    x_coords: NDArray,
    y_coords: NDArray,
    max_radius: float = 50.0,
) -> dict[str, NDArray]:
    """Compute basic morphology for sources at given coordinates.

    For full morphology measurements, use compute_morphology() with a
    segmentation map instead.
    """
    concentration_values = concentration_index_batch(image, x_coords, y_coords)
    return {
        "concentration": concentration_values,
        "concentration_c": concentration_values,
        "half_light_radius": half_light_radius_batch(image, x_coords, y_coords, max_radius),
    }


# Backward compatibility aliases
compute_morphology_batch_parallel = compute_morphology_batch
concentration_index_batch_parallel = concentration_index_batch
half_light_radius_batch_parallel = half_light_radius_batch
