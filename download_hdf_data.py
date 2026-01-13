#!/usr/bin/env python3
"""
Download Hubble Deep Field data and create star masks.

This script downloads:
1. Full HDF-N v2 mosaics (4096x4096) from STScI archive
2. Inverse variance (weight) maps for proper error propagation
3. Creates star masks using Gaia DR3 catalog

Requirements:
    pip install astropy astroquery photutils requests tqdm

Usage:
    python download_hdf_data.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import requests
from astropy import units as u
from astropy.convolution import convolve
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import CircularAperture
from tqdm import tqdm

# Configuration
HDF_CENTER_RA = 189.228621  # degrees (12h 36m 54.87s)
HDF_CENTER_DEC = 62.212572  # degrees (+62° 12' 45.3")
HDF_RADIUS_ARCMIN = 2.5  # arcmin - covers the full HDF field

# STScI archive URLs for HDF v2 mosaics (4096x4096, 0.04"/pixel)
STSCI_BASE_URL = "https://archive.stsci.edu/pub/hdf/v2/mosaics/x4096"

# Band mapping: your files -> STScI filter names
BAND_MAPPING = {
    "u": "f300",  # F300W (UV)
    "b": "f450",  # F450W (Blue)
    "v": "f606",  # F606W (Visual)
    "i": "f814",  # F814W (Infrared)
}

# AB magnitude zeropoints for each band (from STScI documentation)
AB_ZEROPOINTS = {
    "u": 20.78,  # F300W
    "b": 21.92,  # F450W
    "v": 23.02,  # F606W
    "i": 22.08,  # F814W
}


def download_file(url: str, dest_path: Path, description: str = "") -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=description or dest_path.name,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True
    except requests.RequestException as e:
        print(f"  Error downloading {url}: {e}")
        return False


def download_hdf_mosaics(output_dir: Path, bands: list[str] | None = None) -> dict:
    """
    Download HDF v2 mosaics from STScI archive.

    Returns dict mapping band -> (science_path, weight_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if bands is None:
        bands = list(BAND_MAPPING.keys())

    downloaded = {}

    print("\n" + "=" * 60)
    print("DOWNLOADING HDF-N v2 MOSAICS FROM STScI")
    print("=" * 60)
    print(f"Resolution: 4096x4096 pixels, 0.04\"/pixel")
    print(f"Output directory: {output_dir}")
    print()

    for band in bands:
        filter_name = BAND_MAPPING[band]

        # Science image
        sci_filename = f"{filter_name}_mosaic_v2.fits"
        sci_url = f"{STSCI_BASE_URL}/{sci_filename}"
        sci_path = output_dir / f"{band}_full.fits"

        # Inverse variance (weight) map
        wht_filename = f"{filter_name}c_mosaic_v2.fits"
        wht_url = f"{STSCI_BASE_URL}/{wht_filename}"
        wht_path = output_dir / f"{band}_weight.fits"

        print(f"\n[{band.upper()}] Downloading {filter_name.upper()} band...")

        # Download science image
        if sci_path.exists():
            print(f"  Science image already exists: {sci_path}")
        else:
            print(f"  Downloading science image...")
            if not download_file(sci_url, sci_path, f"{band}_sci"):
                continue

        # Download weight map
        if wht_path.exists():
            print(f"  Weight map already exists: {wht_path}")
        else:
            print(f"  Downloading weight map...")
            if not download_file(wht_url, wht_path, f"{band}_wht"):
                continue

        downloaded[band] = (sci_path, wht_path)
        print(f"  Done: {band.upper()} band complete")

    return downloaded


def query_gaia_stars(
    ra: float,
    dec: float,
    radius_arcmin: float = 3.0,
    magnitude_limit: float = 25.0,
) -> list[dict]:
    """
    Query Gaia DR3 for foreground stars in the HDF region.

    Stars are identified by:
    1. Having parallax > 0.1 mas (within ~10 kpc, definitely foreground)
    2. Being point sources (low astrometric excess noise)

    Returns list of dicts with ra, dec, gmag, parallax
    """
    try:
        from astroquery.gaia import Gaia
    except ImportError:
        print("  Warning: astroquery not installed. Install with: pip install astroquery")
        print("  Returning empty star list.")
        return []

    print(f"\n  Querying Gaia DR3 at RA={ra:.4f}, Dec={dec:.4f}, radius={radius_arcmin}'...")

    # ADQL query for Gaia DR3
    query = f"""
    SELECT
        source_id,
        ra,
        dec,
        parallax,
        parallax_error,
        pmra,
        pmdec,
        phot_g_mean_mag,
        astrometric_excess_noise,
        ruwe
    FROM gaiadr3.gaia_source
    WHERE
        CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius_arcmin / 60.0})
        ) = 1
        AND phot_g_mean_mag < {magnitude_limit}
        AND parallax IS NOT NULL
    ORDER BY phot_g_mean_mag ASC
    """

    try:
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        Gaia.ROW_LIMIT = -1  # No limit

        job = Gaia.launch_job(query)
        results = job.get_results()

        stars = []
        for row in results:
            # Filter for likely foreground stars
            # Parallax > 0.1 mas means distance < 10 kpc (definitely foreground)
            parallax = row["parallax"]
            parallax_error = row["parallax_error"]

            # Accept if parallax is significant and positive
            if parallax is not None and parallax > 0.1:
                # Check if parallax is statistically significant
                if parallax_error is not None and parallax / parallax_error > 2:
                    stars.append({
                        "source_id": row["source_id"],
                        "ra": float(row["ra"]),
                        "dec": float(row["dec"]),
                        "gmag": float(row["phot_g_mean_mag"]),
                        "parallax": float(parallax),
                        "parallax_error": float(parallax_error) if parallax_error else 0,
                    })

        print(f"  Found {len(stars)} foreground stars with significant parallax")
        return stars

    except Exception as e:
        print(f"  Error querying Gaia: {e}")
        return []


def create_star_mask(
    fits_path: Path,
    stars: list[dict],
    mask_radius_arcsec: float = 2.0,
    output_path: Path | None = None,
) -> np.ndarray:
    """
    Create a boolean mask for foreground stars.

    Parameters
    ----------
    fits_path : Path
        Path to FITS image (used for WCS and dimensions)
    stars : list of dict
        List of stars with 'ra' and 'dec' keys
    mask_radius_arcsec : float
        Radius around each star to mask (in arcseconds)
    output_path : Path, optional
        If provided, save mask as FITS file

    Returns
    -------
    mask : np.ndarray
        Boolean mask (True = star region to mask)
    """
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    shape = data.shape

    # Try to get WCS from header
    try:
        wcs = WCS(header)
        has_wcs = wcs.has_celestial
    except Exception:
        has_wcs = False

    mask = np.zeros(shape, dtype=bool)

    if not stars:
        print("  No stars to mask")
        if output_path:
            save_mask(mask, header, output_path)
        return mask

    if has_wcs:
        # Use WCS to convert RA/Dec to pixel coordinates
        print(f"  Using WCS to place {len(stars)} star masks...")

        coords = SkyCoord(
            ra=[s["ra"] for s in stars] * u.deg,
            dec=[s["dec"] for s in stars] * u.deg,
        )

        try:
            pixel_coords = wcs.world_to_pixel(coords)
            x_pixels = pixel_coords[0]
            y_pixels = pixel_coords[1]
        except Exception as e:
            print(f"  WCS conversion failed: {e}")
            print("  Falling back to approximate placement...")
            has_wcs = False

    if not has_wcs:
        # Approximate conversion using header info
        print("  Using approximate coordinate conversion...")

        # Get reference pixel and value
        crpix1 = header.get("CRPIX1", shape[1] / 2)
        crpix2 = header.get("CRPIX2", shape[0] / 2)
        crval1 = header.get("CRVAL1", HDF_CENTER_RA)
        crval2 = header.get("CRVAL2", HDF_CENTER_DEC)

        # Get pixel scale (degrees/pixel)
        cd1_1 = header.get("CD1_1", header.get("CDELT1", 0.04 / 3600))
        cd2_2 = header.get("CD2_2", header.get("CDELT2", -0.04 / 3600))

        pixel_scale = abs(cd1_1) * 3600  # arcsec/pixel

        x_pixels = []
        y_pixels = []
        for star in stars:
            dx_deg = (star["ra"] - crval1) * np.cos(np.radians(crval2))
            dy_deg = star["dec"] - crval2

            x_pix = crpix1 + dx_deg / cd1_1
            y_pix = crpix2 + dy_deg / cd2_2

            x_pixels.append(x_pix)
            y_pixels.append(y_pix)

        x_pixels = np.array(x_pixels)
        y_pixels = np.array(y_pixels)

    # Get pixel scale for radius conversion
    cd1_1 = header.get("CD1_1", header.get("CDELT1", 0.04 / 3600))
    pixel_scale = abs(cd1_1) * 3600  # arcsec/pixel
    mask_radius_pix = mask_radius_arcsec / pixel_scale

    print(f"  Pixel scale: {pixel_scale:.4f} arcsec/pixel")
    print(f"  Mask radius: {mask_radius_arcsec:.1f} arcsec = {mask_radius_pix:.1f} pixels")

    # Create circular masks
    stars_in_image = 0
    for i, (x, y) in enumerate(zip(x_pixels, y_pixels)):
        # Check if star is within image bounds
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            # Create circular mask using photutils aperture
            aperture = CircularAperture((x, y), r=mask_radius_pix)
            aperture_mask = aperture.to_mask(method="center")

            # Apply to full image mask
            if aperture_mask is not None:
                # Get the slice and mask data
                slices = aperture_mask.get_overlap_slices(shape)
                if slices[0] is not None:
                    mask[slices[0]] |= aperture_mask.data[slices[1]] > 0
                    stars_in_image += 1

    print(f"  Masked {stars_in_image} stars within image bounds")

    if output_path:
        save_mask(mask, header, output_path)

    return mask


def save_mask(mask: np.ndarray, header: fits.Header, output_path: Path) -> None:
    """Save mask as FITS file."""
    mask_header = header.copy()
    mask_header["BUNIT"] = "MASK"
    mask_header["COMMENT"] = "Star mask created from Gaia DR3 catalog"
    mask_header["COMMENT"] = "1 = masked (star region), 0 = unmasked (science region)"

    hdu = fits.PrimaryHDU(mask.astype(np.uint8), header=mask_header)
    hdu.writeto(output_path, overwrite=True)
    print(f"  Saved mask to: {output_path}")


def upscale_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Upscale a boolean mask to a larger size using nearest-neighbor interpolation.

    NOTE: This is a simple zoom - use transform_mask_wcs() for proper alignment
    between images with different WCS.
    """
    from scipy.ndimage import zoom

    zoom_y = target_shape[0] / mask.shape[0]
    zoom_x = target_shape[1] / mask.shape[1]

    print(f"  Upscaling mask from {mask.shape} to {target_shape} (zoom: {zoom_y:.1f}x)")
    upscaled = zoom(mask.astype(np.uint8), (zoom_y, zoom_x), order=0)

    return upscaled.astype(bool)


def transform_mask_wcs(
    mask: np.ndarray,
    source_fits_path: Path,
    target_fits_path: Path,
    mask_radius_pix: float = 3.0,
) -> np.ndarray:
    """
    Transform a mask from one image's coordinate system to another using WCS.

    This properly handles different image orientations, pixel scales, and offsets.

    Parameters
    ----------
    mask : np.ndarray
        Source mask (True = masked pixels)
    source_fits_path : Path
        FITS file with WCS for the source mask coordinates
    target_fits_path : Path
        FITS file with WCS for the target image
    mask_radius_pix : float
        Radius in target pixels to mask around each transformed position

    Returns
    -------
    transformed_mask : np.ndarray
        Mask in target image coordinates
    """
    from astropy.wcs import WCS
    from scipy.ndimage import binary_dilation

    # Load WCS from both images
    with fits.open(source_fits_path) as hdul:
        source_header = hdul[0].header
    with fits.open(target_fits_path) as hdul:
        target_data = hdul[0].data
        target_header = hdul[0].header

    target_shape = target_data.shape

    # Try to create WCS objects
    try:
        source_wcs = WCS(source_header)
        target_wcs = WCS(target_header)
    except Exception as e:
        print(f"  WCS creation failed: {e}")
        print(f"  Falling back to simple upscaling")
        return upscale_mask(mask, target_shape)

    # Get masked pixel positions
    y_src, x_src = np.where(mask)
    if len(y_src) == 0:
        return np.zeros(target_shape, dtype=bool)

    print(f"  Transforming {len(y_src)} masked pixels via WCS...")

    # Convert source pixel coords to world coords (RA, Dec)
    try:
        # Try using the WCS to get sky coordinates
        world_coords = source_wcs.pixel_to_world(x_src, y_src)
        ra = world_coords.ra.deg if hasattr(world_coords, 'ra') else None
        dec = world_coords.dec.deg if hasattr(world_coords, 'dec') else None

        if ra is None or dec is None:
            raise ValueError("Could not get RA/Dec from WCS")

    except Exception as e:
        print(f"  WCS transformation failed: {e}")
        # Fall back to full CD matrix transformation
        crpix1_src = source_header.get("CRPIX1", mask.shape[1] / 2)
        crpix2_src = source_header.get("CRPIX2", mask.shape[0] / 2)
        crval1_src = source_header.get("CRVAL1", HDF_CENTER_RA)
        crval2_src = source_header.get("CRVAL2", HDF_CENTER_DEC)
        cd1_1 = source_header.get("CD1_1", 0.04 / 3600)
        cd1_2 = source_header.get("CD1_2", 0.0)
        cd2_1 = source_header.get("CD2_1", 0.0)
        cd2_2 = source_header.get("CD2_2", -0.04 / 3600)

        # Full CD matrix transformation: (ra, dec) = CRVAL + CD @ (x - CRPIX, y - CRPIX)
        dx = x_src - crpix1_src
        dy = y_src - crpix2_src
        ra = crval1_src + cd1_1 * dx + cd1_2 * dy
        dec = crval2_src + cd2_1 * dx + cd2_2 * dy

    # Convert world coords to target pixel coords
    try:
        from astropy.coordinates import SkyCoord
        from astropy import units as u

        coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        x_tgt, y_tgt = target_wcs.world_to_pixel(coords)
        x_tgt = np.array(x_tgt)
        y_tgt = np.array(y_tgt)
        print(f"  Target WCS transformation successful")
        print(f"  Target coords range: x=[{x_tgt.min():.1f}, {x_tgt.max():.1f}], y=[{y_tgt.min():.1f}, {y_tgt.max():.1f}]")

    except Exception as e:
        print(f"  Target WCS transformation failed: {e}")
        # Fall back to approximate transformation
        crpix1_tgt = target_header.get("CRPIX1", target_shape[1] / 2)
        crpix2_tgt = target_header.get("CRPIX2", target_shape[0] / 2)
        crval1_tgt = target_header.get("CRVAL1", HDF_CENTER_RA)
        crval2_tgt = target_header.get("CRVAL2", HDF_CENTER_DEC)
        cd1_1_tgt = target_header.get("CD1_1", -0.04 / 3600)
        cd2_2_tgt = target_header.get("CD2_2", 0.04 / 3600)

        x_tgt = crpix1_tgt + (ra - crval1_tgt) / cd1_1_tgt
        y_tgt = crpix2_tgt + (dec - crval2_tgt) / cd2_2_tgt

    # Create mask at target positions
    transformed_mask = np.zeros(target_shape, dtype=bool)

    # Round to integer pixel positions and filter valid ones
    x_int = np.round(x_tgt).astype(int)
    y_int = np.round(y_tgt).astype(int)

    valid = (x_int >= 0) & (x_int < target_shape[1]) & \
            (y_int >= 0) & (y_int < target_shape[0])

    n_valid = np.sum(valid)
    print(f"  {n_valid}/{len(x_src)} pixels mapped to target image")

    if n_valid > 0:
        transformed_mask[y_int[valid], x_int[valid]] = True

        # Expand slightly to cover any gaps from coordinate transformation
        if mask_radius_pix > 1:
            structure = np.ones((int(mask_radius_pix * 2 + 1), int(mask_radius_pix * 2 + 1)))
            transformed_mask = binary_dilation(transformed_mask, structure=structure)

    print(f"  Transformed mask: {np.sum(transformed_mask)} pixels")

    return transformed_mask


def create_combined_mask(
    fits_path: Path,
    stars: list[dict],
    existing_mask_path: Path | None = None,
    star_mask_radius_arcsec: float = 2.0,
    output_path: Path | None = None,
) -> np.ndarray:
    """
    Create a combined mask including stars and any existing mask.

    Parameters
    ----------
    fits_path : Path
        Path to reference FITS image
    stars : list of dict
        List of Gaia stars
    existing_mask_path : Path, optional
        Path to existing mask (e.g., planet_mask.fits) - will be upscaled if needed
    star_mask_radius_arcsec : float
        Radius to mask around each star
    output_path : Path, optional
        Where to save combined mask

    Returns
    -------
    combined_mask : np.ndarray
        Boolean mask (True = masked region)
    """
    # Create star mask
    star_mask = create_star_mask(
        fits_path, stars, mask_radius_arcsec=star_mask_radius_arcsec
    )

    target_shape = star_mask.shape

    # Load existing mask if provided
    if existing_mask_path and existing_mask_path.exists():
        print(f"  Loading existing mask: {existing_mask_path}")
        with fits.open(existing_mask_path) as hdul:
            existing_mask = hdul[0].data.astype(bool)

        # Upscale if sizes don't match
        if existing_mask.shape != target_shape:
            print(f"  Existing mask shape {existing_mask.shape} != target {target_shape}")
            existing_mask = upscale_mask(existing_mask, target_shape)

        # Combine masks
        combined_mask = star_mask | existing_mask
        print(f"  Star mask: {np.sum(star_mask)} pixels")
        print(f"  Existing mask (upscaled): {np.sum(existing_mask)} pixels")
        print(f"  Combined mask: {np.sum(combined_mask)} pixels masked")
    else:
        combined_mask = star_mask

    if output_path:
        with fits.open(fits_path) as hdul:
            header = hdul[0].header
        save_mask(combined_mask, header, output_path)

    return combined_mask


def create_morphological_star_mask(
    fits_path: Path,
    psf_fwhm_arcsec: float = 0.09,
    pixel_scale_arcsec: float = 0.04,
    star_size_threshold: float = 1.5,
    detection_sigma: float = 3.0,
    output_path: Path | None = None,
) -> np.ndarray:
    """
    Create a star mask based on morphological classification (point-source detection).

    Stars are identified as sources whose half-light radius is close to the PSF size.
    Galaxies are extended and have larger half-light radii.

    Parameters
    ----------
    fits_path : Path
        Path to FITS science image
    psf_fwhm_arcsec : float
        PSF FWHM in arcseconds (HST ACS ~0.09")
    pixel_scale_arcsec : float
        Pixel scale in arcsec/pixel
    star_size_threshold : float
        Sources with r_half < star_size_threshold * PSF_FWHM/2 are classified as stars
    detection_sigma : float
        Detection threshold in sigma above background
    output_path : Path, optional
        If provided, save mask as FITS file

    Returns
    -------
    mask : np.ndarray
        Boolean mask (True = star/point-source region to mask)
    star_catalog : list of dict
        List of detected stars with positions and sizes
    """
    from photutils.background import Background2D, MedianBackground
    from photutils.segmentation import SourceCatalog, SourceFinder, make_2dgaussian_kernel

    print(f"\n  Creating morphological star mask from: {fits_path}")

    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(float)
        header = hdul[0].header

    shape = data.shape
    psf_fwhm_pix = psf_fwhm_arcsec / pixel_scale_arcsec

    # Expected half-light radius for point source (PSF)
    # For Gaussian: r_half ≈ FWHM/2 * sqrt(ln(2)) ≈ 0.42 * FWHM
    psf_half_light = psf_fwhm_pix * 0.42

    # Star threshold: sources smaller than this are likely stars
    star_r_half_max = star_size_threshold * psf_half_light

    print(f"  PSF FWHM: {psf_fwhm_arcsec:.3f}\" ({psf_fwhm_pix:.2f} pix)")
    print(f"  Expected PSF half-light radius: {psf_half_light:.2f} pix")
    print(f"  Star threshold: r_half < {star_r_half_max:.2f} pix")

    # Background estimation
    box_size = 100 if shape[0] > 2048 else 50
    bkg = Background2D(data, box_size, filter_size=3, bkg_estimator=MedianBackground())
    data_sub = data - bkg.background

    # Source detection
    kernel_fwhm = max(4.0, psf_fwhm_pix * 2.0)
    kernel = make_2dgaussian_kernel(kernel_fwhm, size=int(kernel_fwhm * 4) | 1)
    convolved_data = convolve(data_sub, kernel, normalize_kernel=True)

    # Minimum pixels for detection (stricter to avoid noise)
    npixels = int(np.pi * (psf_fwhm_pix / 2) ** 2 * 3)

    finder = SourceFinder(npixels=npixels, progress_bar=False)
    segm = finder(convolved_data, detection_sigma * bkg.background_rms)

    if segm is None:
        print("  No sources detected")
        return np.zeros(shape, dtype=bool), []

    # Create source catalog
    catalog = SourceCatalog(data_sub, segm, convolved_data=convolved_data)

    # Get half-light radii
    try:
        r_half_raw = catalog.fluxfrac_radius(0.5)
        # Convert from Quantity to plain array if needed
        if hasattr(r_half_raw, 'value'):
            r_half = np.array(r_half_raw.value)
        else:
            r_half = np.array(r_half_raw)
    except Exception:
        r_half = np.full(len(catalog), np.nan)

    # Get source properties
    table = catalog.to_table()

    # Classify stars: small, round sources
    star_mask = np.zeros(shape, dtype=bool)
    star_catalog = []

    n_stars = 0
    for i, (rh, label) in enumerate(zip(r_half, catalog.labels)):
        if np.isfinite(rh) and float(rh) < star_r_half_max and float(rh) > 0.5:
            # This is likely a star - add to mask
            star_mask |= (segm.data == label)
            n_stars += 1

            # Record star properties
            star_catalog.append({
                "x": float(table["xcentroid"][i]),
                "y": float(table["ycentroid"][i]),
                "r_half_pix": float(rh),
                "r_half_arcsec": float(rh * pixel_scale_arcsec),
                "flux": float(table["segment_flux"][i]),
            })

    print(f"  Detected {len(catalog)} sources, classified {n_stars} as stars")

    # Expand star regions slightly (to mask halos/wings)
    if n_stars > 0:
        from scipy.ndimage import binary_dilation
        expand_radius = int(psf_fwhm_pix * 2)
        structure = np.ones((expand_radius * 2 + 1, expand_radius * 2 + 1))
        star_mask = binary_dilation(star_mask, structure=structure)
        print(f"  Expanded mask by {expand_radius} pixels to cover PSF wings")

    print(f"  Star mask: {np.sum(star_mask)} pixels ({100*np.mean(star_mask):.2f}%)")

    if output_path:
        save_mask(star_mask, header, output_path)

    return star_mask, star_catalog


def create_full_combined_mask(
    fits_path: Path,
    existing_mask_path: Path | None = None,
    use_gaia: bool = True,
    use_morphological: bool = True,
    gaia_mask_radius_arcsec: float = 2.5,
    output_path: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Create a comprehensive mask combining:
    1. Existing manual mask (upscaled if needed)
    2. Gaia DR3 star positions
    3. Morphological point-source detection

    Parameters
    ----------
    fits_path : Path
        Reference FITS image for coordinates and dimensions
    existing_mask_path : Path, optional
        Path to existing mask (e.g., planet_mask2.fits)
    use_gaia : bool
        Whether to include Gaia DR3 star positions
    use_morphological : bool
        Whether to detect stars by morphology
    gaia_mask_radius_arcsec : float
        Radius around Gaia stars to mask
    output_path : Path, optional
        Where to save combined mask

    Returns
    -------
    combined_mask : np.ndarray
        Boolean mask (True = masked region)
    stats : dict
        Statistics about each mask component
    """
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    target_shape = data.shape
    combined_mask = np.zeros(target_shape, dtype=bool)
    stats = {"total_shape": target_shape}

    print("\n" + "=" * 60)
    print("CREATING COMPREHENSIVE MASK")
    print("=" * 60)
    print(f"Target shape: {target_shape}")

    # 1. Load and transform existing manual mask using WCS
    if existing_mask_path and existing_mask_path.exists():
        print(f"\n[1/3] Loading existing mask: {existing_mask_path}")
        with fits.open(existing_mask_path) as hdul:
            existing_mask = hdul[0].data.astype(bool)

        if existing_mask.shape != target_shape:
            print(f"  Original shape: {existing_mask.shape}")
            # Find the source FITS file that matches the mask's coordinate system
            # The 2048 mask was made for the 2048 images (b.fits, etc.)
            script_dir = Path(__file__).parent
            source_fits_candidates = [
                script_dir / "fits" / "b.fits",
                script_dir / "fits" / "i.fits",
                script_dir / "fits" / "v.fits",
            ]
            source_fits = None
            for candidate in source_fits_candidates:
                if candidate.exists():
                    with fits.open(candidate) as h:
                        if h[0].data.shape == existing_mask.shape:
                            source_fits = candidate
                            break

            if source_fits:
                print(f"  Using WCS from {source_fits.name} for coordinate transformation")
                existing_mask = transform_mask_wcs(
                    existing_mask, source_fits, fits_path, mask_radius_pix=3.0
                )
            else:
                print(f"  No matching source FITS found, using simple upscaling")
                existing_mask = upscale_mask(existing_mask, target_shape)

        combined_mask |= existing_mask
        stats["existing_pixels"] = int(np.sum(existing_mask))
        print(f"  Existing mask: {stats['existing_pixels']} pixels")
    else:
        print("\n[1/3] No existing mask found")
        stats["existing_pixels"] = 0

    # 2. Gaia DR3 star mask
    if use_gaia:
        print("\n[2/3] Creating Gaia DR3 star mask...")
        stars = query_gaia_stars(
            HDF_CENTER_RA,
            HDF_CENTER_DEC,
            radius_arcmin=HDF_RADIUS_ARCMIN,
            magnitude_limit=25.0,
        )
        if stars:
            gaia_mask = create_star_mask(
                fits_path, stars, mask_radius_arcsec=gaia_mask_radius_arcsec
            )
            combined_mask |= gaia_mask
            stats["gaia_stars"] = len(stars)
            stats["gaia_pixels"] = int(np.sum(gaia_mask))
        else:
            stats["gaia_stars"] = 0
            stats["gaia_pixels"] = 0
    else:
        print("\n[2/3] Skipping Gaia mask")
        stats["gaia_stars"] = 0
        stats["gaia_pixels"] = 0

    # 3. Morphological star detection
    if use_morphological:
        print("\n[3/3] Creating morphological star mask...")
        morph_mask, morph_stars = create_morphological_star_mask(
            fits_path,
            psf_fwhm_arcsec=0.09,
            pixel_scale_arcsec=0.04,
            star_size_threshold=1.5,
        )
        combined_mask |= morph_mask
        stats["morph_stars"] = len(morph_stars)
        stats["morph_pixels"] = int(np.sum(morph_mask))
    else:
        print("\n[3/3] Skipping morphological mask")
        stats["morph_stars"] = 0
        stats["morph_pixels"] = 0

    stats["combined_pixels"] = int(np.sum(combined_mask))
    stats["combined_fraction"] = float(np.mean(combined_mask))

    print("\n" + "-" * 40)
    print("MASK SUMMARY:")
    print(f"  Existing mask: {stats['existing_pixels']:,} pixels")
    print(f"  Gaia stars: {stats['gaia_stars']} stars, {stats['gaia_pixels']:,} pixels")
    print(f"  Morphological: {stats['morph_stars']} stars, {stats['morph_pixels']:,} pixels")
    print(f"  Combined: {stats['combined_pixels']:,} pixels ({100*stats['combined_fraction']:.2f}%)")

    if output_path:
        save_mask(combined_mask, header, output_path)

    return combined_mask, stats


def verify_downloads(downloaded: dict) -> None:
    """Verify downloaded files and print summary."""
    print("\n" + "=" * 60)
    print("DOWNLOAD VERIFICATION")
    print("=" * 60)

    for band, (sci_path, wht_path) in downloaded.items():
        print(f"\n[{band.upper()}] {BAND_MAPPING[band].upper()} band:")

        for path, name in [(sci_path, "Science"), (wht_path, "Weight")]:
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)

                with fits.open(path) as hdul:
                    shape = hdul[0].data.shape
                    dtype = hdul[0].data.dtype

                print(f"  {name}: {path.name}")
                print(f"    Size: {size_mb:.1f} MB")
                print(f"    Shape: {shape}")
                print(f"    Dtype: {dtype}")
            else:
                print(f"  {name}: MISSING!")


def main():
    """Main function to download HDF data and create star masks."""
    # Setup paths
    script_dir = Path(__file__).parent
    fits_dir = script_dir / "fits"
    data_dir = script_dir / "data"

    fits_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("HUBBLE DEEP FIELD DATA DOWNLOAD AND STAR MASK CREATION")
    print("=" * 60)
    print(f"\nHDF-N Center: RA={HDF_CENTER_RA:.4f}°, Dec={HDF_CENTER_DEC:.4f}°")
    print(f"Field radius: {HDF_RADIUS_ARCMIN} arcmin")

    # Check what we already have
    existing_files = list(fits_dir.glob("*.fits"))
    print(f"\nExisting FITS files: {len(existing_files)}")
    for f in existing_files:
        print(f"  - {f.name}")

    # Ask user what to do
    print("\n" + "-" * 60)
    print("OPTIONS:")
    print("  1. Download full 4096x4096 HDF mosaics + weight maps from STScI")
    print("  2. Create star masks for existing images only (Gaia only)")
    print("  3. Both (download + create masks)")
    print("  4. Create comprehensive mask (Gaia + morphological + existing)")
    print("-" * 60)

    # Default to option 4 for automation
    choice = "4"
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        try:
            choice = input("\nEnter choice [1/2/3/4, default=4]: ").strip() or "4"
        except EOFError:
            choice = "4"

    downloaded = {}

    if choice in ["1", "3"]:
        # Download HDF mosaics
        downloaded = download_hdf_mosaics(fits_dir)
        verify_downloads(downloaded)

    if choice in ["2", "3"]:
        # Query Gaia for foreground stars
        print("\n" + "=" * 60)
        print("QUERYING GAIA DR3 FOR FOREGROUND STARS")
        print("=" * 60)

        stars = query_gaia_stars(
            HDF_CENTER_RA,
            HDF_CENTER_DEC,
            radius_arcmin=HDF_RADIUS_ARCMIN,
            magnitude_limit=25.0,
        )

        if stars:
            print(f"\nForeground stars found: {len(stars)}")
            print("\nBrightest stars:")
            for i, star in enumerate(sorted(stars, key=lambda x: x["gmag"])[:10]):
                dist_pc = 1000 / star["parallax"] if star["parallax"] > 0 else float("inf")
                print(
                    f"  {i+1}. G={star['gmag']:.2f} mag, "
                    f"parallax={star['parallax']:.2f} mas, "
                    f"dist~{dist_pc:.0f} pc"
                )

        # Create star masks
        print("\n" + "=" * 60)
        print("CREATING STAR MASKS")
        print("=" * 60)

        # Determine which FITS file to use as reference
        if downloaded:
            # Use newly downloaded files
            ref_band = list(downloaded.keys())[0]
            ref_fits = downloaded[ref_band][0]
        else:
            # Use existing files
            existing = list(fits_dir.glob("*.fits"))
            if existing:
                ref_fits = existing[0]
            else:
                print("  No FITS files found!")
                return

        print(f"\nUsing reference image: {ref_fits}")

        # Check for existing planet mask
        existing_mask = data_dir / "planet_mask2.fits"
        if not existing_mask.exists():
            existing_mask = data_dir / "planet_mask.fits"
        if not existing_mask.exists():
            existing_mask = None

        # Create combined star + artifact mask
        star_mask_path = data_dir / "star_mask.fits"
        combined_mask_path = data_dir / "combined_mask.fits"

        print(f"\nCreating star mask...")
        star_mask = create_star_mask(
            ref_fits,
            stars,
            mask_radius_arcsec=2.5,  # Slightly larger to be safe
            output_path=star_mask_path,
        )

        print(f"\nCreating combined mask...")
        combined_mask = create_combined_mask(
            ref_fits,
            stars,
            existing_mask_path=existing_mask,
            star_mask_radius_arcsec=2.5,
            output_path=combined_mask_path,
        )

        print(f"\nMask statistics:")
        print(f"  Star mask: {np.sum(star_mask)} pixels ({100*np.mean(star_mask):.2f}%)")
        print(f"  Combined mask: {np.sum(combined_mask)} pixels ({100*np.mean(combined_mask):.2f}%)")

    if choice == "4":
        # Comprehensive mask: Gaia + morphological + existing (upscaled)
        # Find reference FITS file
        full_fits = list(fits_dir.glob("*_full.fits"))
        if full_fits:
            ref_fits = full_fits[0]
        else:
            existing = list(fits_dir.glob("*.fits"))
            if existing:
                ref_fits = existing[0]
            else:
                print("  No FITS files found!")
                return

        print(f"\nUsing reference image: {ref_fits}")

        # Check for existing manual mask
        existing_mask = data_dir / "planet_mask2.fits"
        if not existing_mask.exists():
            existing_mask = data_dir / "planet_mask.fits"
        if not existing_mask.exists():
            existing_mask = None

        combined_mask_path = data_dir / "combined_mask.fits"

        # Create comprehensive mask with all methods
        combined_mask, stats = create_full_combined_mask(
            ref_fits,
            existing_mask_path=existing_mask,
            use_gaia=True,
            use_morphological=True,
            gaia_mask_radius_arcsec=2.5,
            output_path=combined_mask_path,
        )

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print("\nCreated files:")

    for f in sorted(fits_dir.glob("*_full.fits")) + sorted(fits_dir.glob("*_weight.fits")):
        print(f"  {f}")

    for f in sorted(data_dir.glob("*_mask.fits")):
        print(f"  {f}")

    print("\nTo use in your analysis:")
    print("  1. Update run_analysis.py to use the new *_full.fits files for higher resolution")
    print("  2. Use *_weight.fits for proper error propagation")
    print("  3. Use combined_mask.fits instead of planet_mask2.fits")


if __name__ == "__main__":
    main()
