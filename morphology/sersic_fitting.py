"""Sérsic profile fitting for robust size measurements.

The Sérsic profile is the standard model for galaxy surface brightness:
    I(r) = I_e * exp(-b_n * [(r/r_e)^(1/n) - 1])

where:
- r_e: effective (half-light) radius
- n: Sérsic index (n=1 exponential, n=4 de Vaucouleurs)
- I_e: intensity at r_e

This module provides tools for fitting Sérsic profiles to galaxy images.
For best results, use with PSF convolution via PetroFit.

References:
- Sérsic 1963, BAAA, 6, 41
- Graham & Driver 2005, PASA, 22, 118
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SersicParams:
    """Sérsic profile parameters.

    Attributes
    ----------
    r_eff : float
        Effective (half-light) radius in pixels
    r_eff_arcsec : float
        Effective radius in arcseconds (if pixel_scale provided)
    n : float
        Sérsic index
    ellip : float
        Ellipticity (1 - b/a)
    pa : float
        Position angle in degrees
    amplitude : float
        Central amplitude
    x_0, y_0 : float
        Center position
    chi2 : float
        Fit quality (reduced chi-squared)
    success : bool
        Whether fit converged
    """

    r_eff: float
    r_eff_arcsec: float
    n: float
    ellip: float
    pa: float
    amplitude: float
    x_0: float
    y_0: float
    chi2: float
    success: bool


def get_bn(n: float) -> float:
    """Calculate b_n coefficient for Sérsic profile.

    The coefficient b_n ensures that r_e encloses half the total light.

    Parameters
    ----------
    n : float
        Sérsic index

    Returns
    -------
    float
        b_n coefficient
    """
    # Approximation valid for n > 0.36 (Ciotti & Bertin 1999)
    return 1.9992 * n - 0.3271


def sersic_1d(r: NDArray, amplitude: float, r_eff: float, n: float) -> NDArray:
    """1D Sérsic profile.

    Parameters
    ----------
    r : NDArray
        Radial distance array
    amplitude : float
        Central amplitude
    r_eff : float
        Effective radius
    n : float
        Sérsic index

    Returns
    -------
    NDArray
        Surface brightness at each radius
    """
    bn = get_bn(n)
    return amplitude * np.exp(-bn * ((r / r_eff) ** (1 / n) - 1))


def fit_sersic_1d(
    r: NDArray,
    intensity: NDArray,
    weights: NDArray | None = None,
    n_fixed: float | None = None,
) -> dict:
    """Fit 1D Sérsic profile to radial profile.

    Parameters
    ----------
    r : NDArray
        Radial distances
    intensity : NDArray
        Surface brightness values
    weights : NDArray, optional
        Fit weights (1/error^2)
    n_fixed : float, optional
        If provided, fix Sérsic index to this value

    Returns
    -------
    dict
        Fit parameters: amplitude, r_eff, n, chi2
    """
    from scipy.optimize import curve_fit

    # Filter valid data
    valid = (r > 0) & (intensity > 0) & np.isfinite(r) & np.isfinite(intensity)
    r = r[valid]
    intensity = intensity[valid]

    if weights is not None:
        weights = weights[valid]

    if len(r) < 3:
        return {"success": False, "error": "Too few valid points"}

    # Initial guesses
    amp_0 = np.max(intensity)
    r_eff_0 = np.median(r)
    n_0 = 2.0

    try:
        if n_fixed is not None:
            # Fit with fixed n
            def model(r, amp, r_eff):
                return sersic_1d(r, amp, r_eff, n_fixed)

            popt, _pcov = curve_fit(
                model,
                r,
                intensity,
                p0=[amp_0, r_eff_0],
                sigma=1 / np.sqrt(weights) if weights is not None else None,
                maxfev=1000,
            )

            result = {
                "amplitude": popt[0],
                "r_eff": popt[1],
                "n": n_fixed,
                "success": True,
            }
        else:
            # Fit all parameters
            def model(r, amp, r_eff, n):
                return sersic_1d(r, amp, r_eff, n)

            popt, _pcov = curve_fit(
                model,
                r,
                intensity,
                p0=[amp_0, r_eff_0, n_0],
                bounds=([0, 0.1, 0.3], [np.inf, np.max(r) * 2, 10]),
                sigma=1 / np.sqrt(weights) if weights is not None else None,
                maxfev=1000,
            )

            result = {
                "amplitude": popt[0],
                "r_eff": popt[1],
                "n": popt[2],
                "success": True,
            }

        # Calculate chi-squared
        model_values = sersic_1d(r, result["amplitude"], result["r_eff"], result["n"])
        residuals = intensity - model_values

        if weights is not None:
            chi2 = np.sum(residuals**2 * weights) / (len(r) - len(popt))
        else:
            chi2 = np.sum(residuals**2) / (len(r) - len(popt))

        result["chi2"] = chi2

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


def measure_sersic_params(
    image: NDArray,
    segm_map: NDArray,
    source_id: int,
    psf: NDArray | None = None,
    pixel_scale: float = 0.04,
) -> SersicParams:
    """Fit Sérsic profile to get robust effective radius.

    Parameters
    ----------
    image : NDArray
        2D image array
    segm_map : NDArray
        Segmentation map
    source_id : int
        Source ID in segmentation map
    psf : NDArray, optional
        Point spread function for PSF convolution
    pixel_scale : float
        Pixel scale in arcsec/pixel (default: 0.04 for HST/WFC)

    Returns
    -------
    SersicParams
        Fitted parameters including effective radius
    """
    # Try to use PetroFit if available
    try:
        return _fit_with_petrofit(image, segm_map, source_id, psf, pixel_scale)
    except ImportError:
        pass

    # Fallback to simple radial profile fitting
    return _fit_radial_profile(image, segm_map, source_id, pixel_scale)


def _fit_with_petrofit(
    image: NDArray,
    segm_map: NDArray,
    source_id: int,
    psf: NDArray | None,
    pixel_scale: float,
) -> SersicParams:
    """Fit Sérsic profile using PetroFit."""
    from astropy.modeling.models import Sersic2D
    from petrofit.modeling import PSFConvolvedModel2D, fit_model

    # Get source position from segmentation
    source_mask = segm_map == source_id
    if not np.any(source_mask):
        return SersicParams(
            r_eff=np.nan,
            r_eff_arcsec=np.nan,
            n=np.nan,
            ellip=np.nan,
            pa=np.nan,
            amplitude=np.nan,
            x_0=np.nan,
            y_0=np.nan,
            chi2=np.nan,
            success=False,
        )

    y_idx, x_idx = np.where(source_mask)
    x_cen = np.mean(x_idx)
    y_cen = np.mean(y_idx)

    # Extract cutout around source
    margin = 30
    y_min = max(0, int(y_cen - margin))
    y_max = min(image.shape[0], int(y_cen + margin))
    x_min = max(0, int(x_cen - margin))
    x_max = min(image.shape[1], int(x_cen + margin))

    cutout = image[y_min:y_max, x_min:x_max]
    x_local = x_cen - x_min
    y_local = y_cen - y_min

    # Initial Sérsic model
    sersic_model = Sersic2D(
        amplitude=np.max(cutout),
        r_eff=5,  # Initial guess
        n=2.5,
        x_0=x_local,
        y_0=y_local,
        ellip=0.3,
        theta=0,
    )

    # Fit with PSF convolution if available
    model = PSFConvolvedModel2D(sersic_model, psf=psf) if psf is not None else sersic_model

    try:
        fitted_model, fit_info = fit_model(cutout, model, maxiter=500, epsilon=1e-6)

        return SersicParams(
            r_eff=fitted_model.r_eff.value,
            r_eff_arcsec=fitted_model.r_eff.value * pixel_scale,
            n=fitted_model.n.value,
            ellip=fitted_model.ellip.value,
            pa=np.degrees(fitted_model.theta.value),
            amplitude=fitted_model.amplitude.value,
            x_0=fitted_model.x_0.value + x_min,
            y_0=fitted_model.y_0.value + y_min,
            chi2=fit_info.get("chi2", np.nan) if isinstance(fit_info, dict) else np.nan,
            success=True,
        )
    except Exception:
        return SersicParams(
            r_eff=np.nan,
            r_eff_arcsec=np.nan,
            n=np.nan,
            ellip=np.nan,
            pa=np.nan,
            amplitude=np.nan,
            x_0=np.nan,
            y_0=np.nan,
            chi2=np.nan,
            success=False,
        )


def _fit_radial_profile(
    image: NDArray,
    segm_map: NDArray,
    source_id: int,
    pixel_scale: float,
) -> SersicParams:
    """Fit Sérsic profile from azimuthally-averaged radial profile."""
    from photutils.aperture import CircularAperture, aperture_photometry

    # Get source centroid
    source_mask = segm_map == source_id
    if not np.any(source_mask):
        return SersicParams(
            r_eff=np.nan,
            r_eff_arcsec=np.nan,
            n=np.nan,
            ellip=np.nan,
            pa=np.nan,
            amplitude=np.nan,
            x_0=np.nan,
            y_0=np.nan,
            chi2=np.nan,
            success=False,
        )

    y_idx, x_idx = np.where(source_mask)
    x_cen = np.mean(x_idx)
    y_cen = np.mean(y_idx)

    # Measure radial profile
    radii = np.arange(1, 30, 1)
    profile = []

    for r in radii:
        aper = CircularAperture((x_cen, y_cen), r=r)
        phot = aperture_photometry(image, aper)
        flux = phot["aperture_sum"][0]
        area = np.pi * r**2
        profile.append(flux / area)

    profile = np.array(profile)

    # Convert to surface brightness (differential)
    sb = np.diff(profile * radii**2) / np.diff(radii**2 * np.pi)
    r_mid = (radii[:-1] + radii[1:]) / 2

    # Fit Sérsic profile
    fit_result = fit_sersic_1d(r_mid, sb)

    if not fit_result.get("success", False):
        return SersicParams(
            r_eff=np.nan,
            r_eff_arcsec=np.nan,
            n=np.nan,
            ellip=np.nan,
            pa=np.nan,
            amplitude=np.nan,
            x_0=x_cen,
            y_0=y_cen,
            chi2=np.nan,
            success=False,
        )

    return SersicParams(
        r_eff=fit_result["r_eff"],
        r_eff_arcsec=fit_result["r_eff"] * pixel_scale,
        n=fit_result["n"],
        ellip=0.0,  # Not measured in 1D fit
        pa=0.0,
        amplitude=fit_result["amplitude"],
        x_0=x_cen,
        y_0=y_cen,
        chi2=fit_result.get("chi2", np.nan),
        success=True,
    )


def fit_sersic_profile(
    image: NDArray,
    x: float,
    y: float,
    _initial_r_eff: float = 5.0,
    _initial_n: float = 2.5,
    pixel_scale: float = 0.04,
) -> SersicParams:
    """Fit Sérsic profile at a given position.

    Simplified interface for fitting a single source.

    Parameters
    ----------
    image : NDArray
        2D image array
    x, y : float
        Source position
    initial_r_eff : float
        Initial guess for effective radius
    initial_n : float
        Initial guess for Sérsic index
    pixel_scale : float
        Pixel scale in arcsec/pixel

    Returns
    -------
    SersicParams
        Fitted parameters
    """
    # Create a simple segmentation mask
    Y, X = np.ogrid[: image.shape[0], : image.shape[1]]
    r2 = (X - x) ** 2 + (Y - y) ** 2
    segm_map = (r2 < 50**2).astype(int)

    return _fit_radial_profile(image, segm_map, 1, pixel_scale)
