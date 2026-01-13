#!/usr/bin/env python3
"""
Utilities for working with Hubble Deep Field data.

This module provides functions for:
- Loading HDF FITS images with proper weight maps
- Creating and applying star masks
- Proper error propagation using inverse variance maps

Usage:
    from hdf_utils import load_hdf_image, load_star_mask, get_flux_error
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


@dataclass
class HDFImage:
    """Container for HDF image data with associated weight map."""

    data: np.ndarray
    header: fits.Header
    weight: np.ndarray | None = None
    variance: np.ndarray | None = None
    wcs: WCS | None = None
    band: str = ""
    pixel_scale: float = 0.04  # arcsec/pixel

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def get_error(self, include_background: bool = True, background_rms: float = 0.0) -> np.ndarray:
        """
        Calculate per-pixel error from inverse variance map.

        Parameters
        ----------
        include_background : bool
            If True, add background RMS contribution
        background_rms : float
            Background RMS to add in quadrature

        Returns
        -------
        error : np.ndarray
            Per-pixel 1-sigma error
        """
        if self.variance is not None:
            error = np.sqrt(self.variance)
        elif self.weight is not None:
            # Weight map is inverse variance
            with np.errstate(divide="ignore", invalid="ignore"):
                error = np.where(self.weight > 0, 1.0 / np.sqrt(self.weight), np.inf)
        else:
            # Fallback: Poisson noise from data
            error = np.sqrt(np.maximum(self.data, 0))

        if include_background and background_rms > 0:
            error = np.sqrt(error**2 + background_rms**2)

        return error


def load_hdf_image(
    science_path: str | Path,
    weight_path: str | Path | None = None,
    band: str = "",
    adjust_edges: bool = True,
) -> HDFImage:
    """
    Load an HDF FITS image with optional weight map.

    Parameters
    ----------
    science_path : str or Path
        Path to science FITS file
    weight_path : str or Path, optional
        Path to weight (inverse variance) FITS file
    band : str
        Band name (e.g., 'b', 'v', 'i', 'u')
    adjust_edges : bool
        If True, crop low-value edges (for 2048x2048 data)

    Returns
    -------
    image : HDFImage
        Container with data, header, weight, and metadata
    """
    science_path = Path(science_path)

    with fits.open(science_path) as hdul:
        data = hdul[0].data.copy()
        header = hdul[0].header.copy()

    # Handle byte order
    if data.dtype.byteorder == ">":
        data = data.view(data.dtype.newbyteorder()).byteswap()

    # Try to get WCS
    try:
        wcs = WCS(header)
        if not wcs.has_celestial:
            wcs = None
    except Exception:
        wcs = None

    # Get pixel scale from header
    cd1_1 = header.get("CD1_1", header.get("CDELT1", 0.04 / 3600))
    pixel_scale = abs(cd1_1) * 3600  # arcsec/pixel

    # Load weight map if provided
    weight = None
    variance = None
    if weight_path is not None:
        weight_path = Path(weight_path)
        if weight_path.exists():
            with fits.open(weight_path) as hdul:
                weight = hdul[0].data.copy()
            if weight.dtype.byteorder == ">":
                weight = weight.view(weight.dtype.newbyteorder()).byteswap()

            # Convert inverse variance to variance
            with np.errstate(divide="ignore", invalid="ignore"):
                variance = np.where(weight > 0, 1.0 / weight, np.inf)

    # Adjust edges if needed (for 2048x2048 data)
    if adjust_edges and data.shape == (2048, 2048):
        data = data[120:, 90:]
        data = data[:, ::-1]
        if weight is not None:
            weight = weight[120:, 90:]
            weight = weight[:, ::-1]
        if variance is not None:
            variance = variance[120:, 90:]
            variance = variance[:, ::-1]

    return HDFImage(
        data=data,
        header=header,
        weight=weight,
        variance=variance,
        wcs=wcs,
        band=band,
        pixel_scale=pixel_scale,
    )


def load_star_mask(
    mask_path: str | Path,
    adjust_edges: bool = True,
    expected_shape: tuple | None = None,
) -> np.ndarray:
    """
    Load a star mask from FITS file.

    Parameters
    ----------
    mask_path : str or Path
        Path to mask FITS file
    adjust_edges : bool
        If True, apply same edge adjustment as for 2048x2048 data
    expected_shape : tuple, optional
        Expected shape to validate against

    Returns
    -------
    mask : np.ndarray
        Boolean mask (True = masked/bad pixels)
    """
    mask_path = Path(mask_path)

    with fits.open(mask_path) as hdul:
        mask = hdul[0].data.copy()

    # Ensure boolean
    mask = mask.astype(bool)

    # Adjust edges if needed
    if adjust_edges and mask.shape == (2048, 2048):
        mask = mask[120:, 90:]
        mask = mask[:, ::-1]

    # Validate shape if provided
    if expected_shape is not None and mask.shape != expected_shape:
        print(f"Warning: Mask shape {mask.shape} != expected {expected_shape}")
        # Try to resize if sizes are compatible
        if mask.size == np.prod(expected_shape):
            mask = mask.reshape(expected_shape)

    return mask


def combine_masks(*masks: np.ndarray) -> np.ndarray:
    """
    Combine multiple masks with OR operation.

    Parameters
    ----------
    *masks : np.ndarray
        Boolean masks to combine

    Returns
    -------
    combined : np.ndarray
        Combined boolean mask
    """
    if not masks:
        raise ValueError("At least one mask required")

    combined = masks[0].copy()
    for mask in masks[1:]:
        if mask.shape == combined.shape:
            combined |= mask
        else:
            print(f"Warning: Mask shape mismatch {mask.shape} vs {combined.shape}")

    return combined


def get_aperture_flux_error(
    data: np.ndarray,
    weight: np.ndarray | None,
    aperture_mask: np.ndarray,
    background_rms: float = 0.0,
) -> tuple[float, float]:
    """
    Calculate flux and error for an aperture using proper error propagation.

    Parameters
    ----------
    data : np.ndarray
        Science image data
    weight : np.ndarray or None
        Inverse variance weight map
    aperture_mask : np.ndarray
        Boolean mask defining aperture (True = in aperture)
    background_rms : float
        Background RMS per pixel

    Returns
    -------
    flux : float
        Total flux in aperture
    error : float
        1-sigma error on flux
    """
    # Sum flux in aperture
    flux = np.sum(data[aperture_mask])

    # Calculate error
    if weight is not None:
        # Use inverse variance weights
        # Variance = sum(1/weight) for each pixel
        weights_in_aperture = weight[aperture_mask]
        with np.errstate(divide="ignore", invalid="ignore"):
            pixel_variances = np.where(weights_in_aperture > 0, 1.0 / weights_in_aperture, 0)
        variance = np.sum(pixel_variances)
    else:
        # Poisson noise from source + background
        n_pixels = np.sum(aperture_mask)
        variance = np.abs(flux) + n_pixels * background_rms**2

    error = np.sqrt(variance)

    return flux, error


def find_hdf_files(
    fits_dir: str | Path,
    include_weights: bool = True,
) -> dict[str, dict]:
    """
    Find HDF FITS files in a directory.

    Parameters
    ----------
    fits_dir : str or Path
        Directory containing FITS files
    include_weights : bool
        If True, also look for weight maps

    Returns
    -------
    files : dict
        Dictionary mapping band -> {'science': path, 'weight': path}
    """
    fits_dir = Path(fits_dir)

    files = {}

    # Check for full-resolution files first
    for band in ["u", "b", "v", "i"]:
        band_files = {"science": None, "weight": None}

        # Try full-resolution files
        full_path = fits_dir / f"{band}_full.fits"
        if full_path.exists():
            band_files["science"] = full_path
            weight_path = fits_dir / f"{band}_weight.fits"
            if weight_path.exists() and include_weights:
                band_files["weight"] = weight_path
        else:
            # Fall back to original files
            orig_path = fits_dir / f"{band}.fits"
            if orig_path.exists():
                band_files["science"] = orig_path

        if band_files["science"] is not None:
            files[band] = band_files

    return files


def print_hdf_info(fits_dir: str | Path) -> None:
    """Print information about HDF files in a directory."""
    fits_dir = Path(fits_dir)
    files = find_hdf_files(fits_dir)

    print("=" * 60)
    print("HDF DATA SUMMARY")
    print("=" * 60)

    for band, paths in files.items():
        sci_path = paths["science"]
        wht_path = paths["weight"]

        print(f"\n[{band.upper()}] Band:")

        with fits.open(sci_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header

        print(f"  Science: {sci_path.name}")
        print(f"    Shape: {data.shape}")
        print(f"    Min/Max: {np.nanmin(data):.2f} / {np.nanmax(data):.2f}")

        # Get pixel scale
        cd1_1 = header.get("CD1_1", header.get("CDELT1", 0.04 / 3600))
        pixel_scale = abs(cd1_1) * 3600
        print(f"    Pixel scale: {pixel_scale:.4f} arcsec/pixel")

        if wht_path is not None:
            with fits.open(wht_path) as hdul:
                weight = hdul[0].data
            print(f"  Weight: {wht_path.name}")
            print(f"    Shape: {weight.shape}")


if __name__ == "__main__":
    # If run directly, print info about local HDF files
    script_dir = Path(__file__).parent
    fits_dir = script_dir / "fits"

    if fits_dir.exists():
        print_hdf_info(fits_dir)
    else:
        print(f"No fits directory found at {fits_dir}")
