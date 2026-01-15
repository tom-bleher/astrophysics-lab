"""WCS utilities for coordinate transformations.

Handles pixel-to-sky coordinate conversion accounting for
the cropping/flipping applied to HDF images.
"""

import numpy as np
from astropy.wcs import WCS


def get_wcs_from_header(header) -> WCS:
    """Extract WCS from FITS header."""
    try:
        return WCS(header)
    except Exception as e:
        print(f"Could not create WCS from header: {e}")
        return None


def pixel_to_sky(
    x: np.ndarray,
    y: np.ndarray,
    wcs: WCS,
    crop_offset: tuple = (90, 120),  # (x_offset, y_offset) from adjust_data
    flip_x: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel coordinates to RA/Dec, accounting for cropping/flipping.

    Parameters
    ----------
    x, y : array-like
        Pixel coordinates in the processed (cropped/flipped) image
    wcs : WCS
        WCS object from original FITS header
    crop_offset : tuple
        (x_offset, y_offset) applied during cropping
    flip_x : bool
        Whether the image was flipped in x

    Returns
    -------
    ra, dec : np.ndarray
        Sky coordinates in degrees
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    # Get original image dimensions from WCS
    if hasattr(wcs, 'pixel_shape') and wcs.pixel_shape is not None:
        orig_nx = wcs.pixel_shape[0]
    else:
        # Fallback: estimate from common HDF dimensions
        orig_nx = 4096

    # Reverse the transformations applied in adjust_data()
    # Original code: data = data[120:, 90:]  -> y offset 120, x offset 90
    # Then: data = data[:, ::-1]  -> flip in x

    # First, unflip x if needed
    if flip_x:
        # Get current image width (after cropping)
        # We don't have this directly, but can estimate
        cropped_nx = orig_nx - crop_offset[0]
        x_orig = cropped_nx - 1 - x
    else:
        x_orig = x

    # Then add back crop offsets
    x_full = x_orig + crop_offset[0]
    y_full = y + crop_offset[1]

    # Convert to sky coordinates using WCS
    try:
        ra, dec = wcs.all_pix2world(x_full, y_full, 0)
        return ra, dec
    except Exception as e:
        print(f"WCS conversion failed: {e}")
        return np.full_like(x, np.nan), np.full_like(y, np.nan)


def sky_to_pixel(
    ra: np.ndarray,
    dec: np.ndarray,
    wcs: WCS,
    crop_offset: tuple = (90, 120),
    flip_x: bool = True,
    image_width: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert RA/Dec to pixel coordinates, accounting for cropping/flipping.

    This is the inverse of pixel_to_sky().

    Parameters
    ----------
    ra, dec : array-like
        Sky coordinates in degrees
    wcs : WCS
        WCS object from original FITS header
    crop_offset : tuple
        (x_offset, y_offset) applied during image cropping
    flip_x : bool
        Whether the image was flipped in x
    image_width : int, optional
        Width of the processed image (after cropping). If None, estimated from WCS.

    Returns
    -------
    x, y : np.ndarray
        Pixel coordinates in the processed (cropped/flipped) image
    """
    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)

    # Convert sky to pixel coordinates in original image frame
    try:
        x_full, y_full = wcs.all_world2pix(ra, dec, 0)
    except Exception as e:
        print(f"WCS conversion failed: {e}")
        return np.full_like(ra, np.nan, dtype=float), np.full_like(dec, np.nan, dtype=float)

    # Subtract crop offsets
    x_cropped = x_full - crop_offset[0]
    y_cropped = y_full - crop_offset[1]

    # Apply x-flip if needed
    if flip_x:
        if image_width is not None:
            cropped_nx = image_width
        elif hasattr(wcs, 'pixel_shape') and wcs.pixel_shape is not None:
            cropped_nx = wcs.pixel_shape[0] - crop_offset[0]
        else:
            cropped_nx = 4096 - crop_offset[0]
        x_cropped = cropped_nx - 1 - x_cropped

    return x_cropped, y_cropped


def add_sky_coordinates(
    catalog,
    header,
    x_col: str = 'xcentroid',
    y_col: str = 'ycentroid',
    crop_offset: tuple = (90, 120),
    flip_x: bool = True
):
    """
    Add RA/Dec columns to a catalog with pixel coordinates.

    Parameters
    ----------
    catalog : pd.DataFrame
        Catalog with pixel coordinate columns
    header : fits.Header
        FITS header with WCS information
    x_col, y_col : str
        Column names for pixel coordinates
    crop_offset : tuple
        Offset applied during image cropping
    flip_x : bool
        Whether image was flipped

    Returns
    -------
    pd.DataFrame
        Catalog with added 'ra' and 'dec' columns
    """
    wcs = get_wcs_from_header(header)

    if wcs is None:
        print("Cannot add sky coordinates: no valid WCS")
        return catalog

    ra, dec = pixel_to_sky(
        catalog[x_col].values,
        catalog[y_col].values,
        wcs,
        crop_offset=crop_offset,
        flip_x=flip_x
    )

    catalog = catalog.copy()
    catalog['ra'] = ra
    catalog['dec'] = dec

    print("Added RA/Dec coordinates to catalog")
    print(f"  RA range: {ra.min():.4f} to {ra.max():.4f} deg")
    print(f"  Dec range: {dec.min():.4f} to {dec.max():.4f} deg")

    return catalog
