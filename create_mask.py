#!/usr/bin/env python3
"""
Create a combined mask for the full 4096x4096 HDF images.

This script:
1. Transforms the original 2048x2048 planet mask via WCS to the 4096x4096 coordinate system
2. Detects stars (point sources) using professional multi-criteria morphological classification
3. Combines both masks and saves the result

Star/Galaxy Separation Methodology:
- Concentration index C = r50/r90 (SDSS standard)
- Peak surface brightness (MU_MAX) vs magnitude relation
- Multi-criteria scoring with stellarity index output
- References: SExtractor CLASS_STAR, SDSS DR17, COSMOS2020

References:
- SExtractor: https://sextractor.readthedocs.io/en/latest/ClassStar.html
- SDSS: https://www.sdss.org/dr16/algorithms/classify/
- HST ACS PSF: https://hst-docs.stsci.edu/acsihb/chapter-5-imaging/5-6-acs-point-spread-functions
"""

from pathlib import Path

import numpy as np
from astropy.convolution import convolve
from astropy.io import fits
from astropy.wcs import WCS
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import SourceCatalog, SourceFinder, make_2dgaussian_kernel
from scipy.ndimage import binary_dilation, maximum

# Configuration
PIXEL_SCALE = 0.04  # arcsec/pixel
PSF_FWHM_ARCSEC = 0.09  # HST ACS PSF (note: varies 20-25% across field)
PSF_FWHM_PIX = PSF_FWHM_ARCSEC / PIXEL_SCALE  # ~2.25 pixels

# Concentration index thresholds (SDSS-style: C = r50/r90)
# - Stars/PSF-like: C > 0.45
# - Exponential disk (spirals): C ≈ 0.43
# - de Vaucouleurs (ellipticals): C ≈ 0.30
CONCENTRATION_STAR_THRESHOLD = 0.45


def transform_mask_wcs(
    old_mask: np.ndarray,
    old_fits_path: Path,
    new_fits_path: Path,
    dilation_radius: int = 3,
) -> np.ndarray:
    """
    Transform a mask from old image coordinates to new image coordinates using WCS.

    Parameters
    ----------
    old_mask : np.ndarray
        Boolean mask in old image coordinates
    old_fits_path : Path
        FITS file with WCS for the old image
    new_fits_path : Path
        FITS file with WCS for the new image
    dilation_radius : int
        Radius in pixels to dilate the mask to fill gaps from transformation

    Returns
    -------
    new_mask : np.ndarray
        Mask transformed to new image coordinates
    """
    print("  Transforming mask via WCS...")

    # Load WCS from both images
    with fits.open(old_fits_path) as hdul:
        old_wcs = WCS(hdul[0].header)

    with fits.open(new_fits_path) as hdul:
        new_data = hdul[0].data
        new_wcs = WCS(hdul[0].header)

    new_shape = new_data.shape

    # Get all masked pixel positions
    y_old, x_old = np.where(old_mask)
    print(f"    Old mask has {len(y_old)} masked pixels")

    if len(y_old) == 0:
        return np.zeros(new_shape, dtype=bool)

    # Transform via WCS: old pixels -> sky coords -> new pixels
    ra, dec = old_wcs.pixel_to_world_values(x_old, y_old)
    x_new, y_new = new_wcs.world_to_pixel_values(ra, dec)

    # Round to integers
    x_new = np.round(x_new).astype(int)
    y_new = np.round(y_new).astype(int)

    # Filter valid positions
    valid = (x_new >= 0) & (x_new < new_shape[1]) & (y_new >= 0) & (y_new < new_shape[0])
    x_new = x_new[valid]
    y_new = y_new[valid]

    print(f"    {len(x_new)} pixels mapped to new image")

    # Create mask
    new_mask = np.zeros(new_shape, dtype=bool)
    new_mask[y_new, x_new] = True

    # Dilate to fill gaps from coordinate transformation
    if dilation_radius > 0:
        structure = np.ones((dilation_radius * 2 + 1, dilation_radius * 2 + 1))
        new_mask = binary_dilation(new_mask, structure=structure)
        print(f"    Dilated by {dilation_radius} pixels -> {np.sum(new_mask)} pixels")

    return new_mask


def compute_stellarity(
    r_half: float,
    concentration: float,
    ellipticity: float,
    fwhm: float,
    mu_max_deviation: float,
    psf_fwhm: float = PSF_FWHM_PIX,
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
    fwhm : float
        Source FWHM in pixels
    mu_max_deviation : float
        Deviation from stellar MU_MAX-magnitude relation (0=on stellar locus)
    psf_fwhm : float
        PSF FWHM in pixels

    Returns
    -------
    float
        Stellarity index from 0 (definitely galaxy) to 1 (definitely star)
    """
    scores = []

    # 1. Size score: stars have r_half close to PSF
    # PSF half-light radius ≈ PSF_FWHM * 0.42
    psf_r_half = psf_fwhm * 0.42
    if np.isfinite(r_half) and r_half > 0:
        size_ratio = r_half / psf_r_half
        size_score = max(0, 1.0 - abs(size_ratio - 1.0) / 2.0)
        scores.append(size_score)

    # 2. Concentration score: stars have C > 0.45 (SDSS-style r50/r90)
    if np.isfinite(concentration):
        # Higher concentration = more star-like
        conc_score = min(1.0, max(0, (concentration - 0.30) / 0.20))
        scores.append(conc_score)

    # 3. Roundness score: stars are round (low ellipticity)
    if np.isfinite(ellipticity):
        round_score = max(0, 1.0 - ellipticity / 0.5)
        scores.append(round_score)

    # 4. FWHM score: stars have FWHM close to PSF
    if np.isfinite(fwhm) and fwhm > 0:
        fwhm_ratio = abs(fwhm - psf_fwhm) / psf_fwhm
        fwhm_score = max(0, 1.0 - fwhm_ratio)
        scores.append(fwhm_score)

    # 5. MU_MAX score: stars follow tight MU_MAX-magnitude relation
    if np.isfinite(mu_max_deviation):
        # Deviation from stellar locus; 0 = on locus
        mu_score = max(0, 1.0 - abs(mu_max_deviation) / 1.0)
        scores.append(mu_score)

    if len(scores) == 0:
        return 0.0

    return float(np.mean(scores))


def detect_stars_morphology(
    data: np.ndarray,
    detection_sigma: float = 2.0,
    star_r_half_max: float = 3.5,
    star_concentration_min: float = 0.42,
    star_ellipticity_max: float = 0.3,
    min_flux_percentile: float = 70.0,
    mask_dilation_factor: float = 5.0,
    stellarity_threshold: float = 0.6,
) -> tuple[np.ndarray, list]:
    """
    Detect stars using professional multi-criteria morphological classification.

    Stars are identified using multiple criteria based on professional surveys:
    1. Half-light radius close to PSF (compact)
    2. High concentration index C = r50/r90 > 0.42 (SDSS-style, standardized)
    3. Low ellipticity (stars are round, e < 0.3)
    4. High brightness (foreground stars are typically bright)
    5. Peak surface brightness (MU_MAX) follows stellar locus
    6. Stellarity score > threshold (continuous 0-1 score)

    References:
    - SExtractor CLASS_STAR: https://sextractor.readthedocs.io/en/latest/ClassStar.html
    - SDSS star/galaxy: https://www.sdss.org/dr16/algorithms/classify/
    - MU_MAX method: stars follow tight peak surface brightness vs magnitude relation

    Parameters
    ----------
    data : np.ndarray
        Science image
    detection_sigma : float
        Detection threshold in sigma above background
    star_r_half_max : float
        Maximum half-light radius in pixels (stars have r_half ~ PSF_FWHM * 0.42)
    star_concentration_min : float
        Minimum concentration index C = r50/r90 for stars (SDSS: stars > 0.45)
    star_ellipticity_max : float
        Maximum ellipticity for stars (stars are round, e < 0.3)
    min_flux_percentile : float
        Only consider sources above this flux percentile
    mask_dilation_factor : float
        Dilate star masks by this factor * PSF FWHM
    stellarity_threshold : float
        Minimum stellarity score (0-1) to classify as star

    Returns
    -------
    star_mask : np.ndarray
        Boolean mask (True = star region)
    star_catalog : list of dict
        Detected star properties including stellarity score
    """
    print("  Detecting stars via professional multi-criteria morphology...")
    print(f"    Criteria: r_half < {star_r_half_max:.1f} pix, C > {star_concentration_min:.2f} (r50/r90), e < {star_ellipticity_max:.1f}")
    print(f"    Stellarity threshold: {stellarity_threshold:.2f}")

    # Background estimation
    box_size = 100
    bkg = Background2D(data, box_size, filter_size=3, bkg_estimator=MedianBackground())
    data_sub = data - bkg.background

    # Source detection with Gaussian kernel
    kernel_fwhm = max(3.0, PSF_FWHM_PIX * 1.5)
    kernel = make_2dgaussian_kernel(kernel_fwhm, size=int(kernel_fwhm * 4) | 1)
    convolved = convolve(data_sub, kernel, normalize_kernel=True)

    # Detection threshold
    threshold = detection_sigma * bkg.background_rms

    # Minimum pixels for detection
    npixels = max(5, int(np.pi * (PSF_FWHM_PIX / 2) ** 2 * 2))

    print(f"    Detection threshold: {detection_sigma}-sigma, npixels >= {npixels}")

    finder = SourceFinder(npixels=npixels, progress_bar=False)
    segm = finder(convolved, threshold)

    if segm is None:
        print("    No sources detected")
        return np.zeros(data.shape, dtype=bool), []

    # Create source catalog
    catalog = SourceCatalog(data_sub, segm, convolved_data=convolved)
    print(f"    Detected {len(catalog)} total sources")

    # Get morphological properties for multi-criteria classification
    table = catalog.to_table()
    fluxes = np.array(table["segment_flux"])

    # Half-light radius (r50)
    try:
        r_half_raw = catalog.fluxfrac_radius(0.5)
        r_half = np.array(r_half_raw.value if hasattr(r_half_raw, 'value') else r_half_raw)
    except Exception:
        r_half = np.full(len(catalog), np.nan)

    # Get r90 for SDSS-style concentration index C = r50/r90
    try:
        r90_raw = catalog.fluxfrac_radius(0.9)
        r90 = np.array(r90_raw.value if hasattr(r90_raw, 'value') else r90_raw)
    except Exception:
        r90 = np.full(len(catalog), np.nan)

    # SDSS-style concentration index: C = r50/r90
    # Stars/PSF-like: C > 0.45 (high concentration = compact)
    # Exponential disk: C ≈ 0.43
    # de Vaucouleurs: C ≈ 0.30
    with np.errstate(divide='ignore', invalid='ignore'):
        concentration = r_half / r90

    # Ellipticity from catalog (stars should be round, e < 0.3)
    try:
        ellipticity = np.array(catalog.ellipticity)
    except Exception:
        ellipticity = np.full(len(catalog), np.nan)

    # FWHM from catalog (for comparison with PSF)
    try:
        fwhm = np.array(catalog.fwhm.value if hasattr(catalog.fwhm, 'value') else catalog.fwhm)
    except Exception:
        fwhm = np.full(len(catalog), np.nan)

    # Compute MU_MAX (peak surface brightness) for each source
    # Stars follow tight MU_MAX vs magnitude relation
    # Vectorized: use scipy.ndimage.maximum instead of per-segment loop
    peaks = maximum(data_sub, labels=segm.data, index=catalog.labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        mu_max = np.where(peaks > 0, -2.5 * np.log10(peaks), np.nan)

    # Compute instrumental magnitudes for MU_MAX relation
    with np.errstate(divide='ignore', invalid='ignore'):
        mag_auto = -2.5 * np.log10(np.abs(fluxes))

    # Fit stellar locus: for bright point sources, MU_MAX ≈ MAG_AUTO + const
    # Stars follow tight relation; galaxies scatter below (fainter peak)
    valid_for_fit = np.isfinite(mu_max) & np.isfinite(mag_auto) & (fluxes > 0)
    if np.sum(valid_for_fit) > 10:
        # Use bright, compact sources to define stellar locus
        bright_compact = valid_for_fit & (r_half < star_r_half_max) & (fluxes > np.percentile(fluxes[valid_for_fit], 80))
        if np.sum(bright_compact) >= 5:
            stellar_offset = np.median(mu_max[bright_compact] - mag_auto[bright_compact])
        else:
            stellar_offset = np.median(mu_max[valid_for_fit] - mag_auto[valid_for_fit])
        mu_max_expected = mag_auto + stellar_offset
        mu_max_deviation = mu_max - mu_max_expected
    else:
        mu_max_deviation = np.full(len(catalog), np.nan)

    # Filter by flux - focus on bright sources (foreground stars are bright)
    valid_flux = np.isfinite(fluxes) & (fluxes > 0)
    flux_threshold = np.percentile(fluxes[valid_flux], min_flux_percentile)

    # Classify stars using stellarity score (professional approach)
    star_mask = np.zeros(data.shape, dtype=bool)
    star_catalog = []

    # Collect statistics
    stats = {'r_half': [], 'concentration': [], 'ellipticity': [], 'fwhm': [], 'stellarity': []}

    for i, lab in enumerate(catalog.labels):
        rh = r_half[i]
        flux = fluxes[i]
        c = concentration[i]
        e = ellipticity[i]
        fw = fwhm[i]
        mu_dev = mu_max_deviation[i] if i < len(mu_max_deviation) else np.nan

        # Skip invalid measurements
        if not np.isfinite(rh) or rh <= 0.3:
            continue

        stats['r_half'].append(rh)
        if np.isfinite(c):
            stats['concentration'].append(c)
        if np.isfinite(e):
            stats['ellipticity'].append(e)
        if np.isfinite(fw):
            stats['fwhm'].append(fw)

        # Compute stellarity score (0=galaxy, 1=star)
        stellarity = compute_stellarity(rh, c, e, fw, mu_dev, PSF_FWHM_PIX)
        stats['stellarity'].append(stellarity)

        # Classification criteria:
        # 1. Must be bright (foreground stars are typically bright)
        is_bright = flux >= flux_threshold

        # 2. Must be compact (small half-light radius)
        is_compact = rh < star_r_half_max

        # 3. High concentration (SDSS-style: stars have C > threshold)
        is_concentrated = np.isfinite(c) and (c > star_concentration_min)

        # 4. Round (low ellipticity)
        is_round = (not np.isfinite(e)) or (e < star_ellipticity_max)

        # 5. High stellarity score
        is_stellar = stellarity >= stellarity_threshold

        # Classification: bright AND compact AND (stellarity OR (concentrated AND round))
        if is_bright and is_compact and (is_stellar or (is_concentrated and is_round)):
            star_mask |= (segm.data == lab)
            star_catalog.append({
                "x": float(table["xcentroid"][i]),
                "y": float(table["ycentroid"][i]),
                "r_half_pix": float(rh),
                "concentration": float(c) if np.isfinite(c) else None,
                "ellipticity": float(e) if np.isfinite(e) else None,
                "fwhm": float(fw) if np.isfinite(fw) else None,
                "flux": float(flux),
                "mu_max": float(mu_max[i]) if np.isfinite(mu_max[i]) else None,
                "mu_max_deviation": float(mu_dev) if np.isfinite(mu_dev) else None,
                "stellarity": float(stellarity),
            })

    # Print statistics for diagnostics
    if stats['r_half']:
        arr = np.array(stats['r_half'])
        print(f"    r_half: min={arr.min():.2f}, median={np.median(arr):.2f}, max={arr.max():.2f}")
    if stats['concentration']:
        arr = np.array(stats['concentration'])
        print(f"    concentration (r50/r90): min={arr.min():.3f}, median={np.median(arr):.3f}, max={arr.max():.3f}")
    if stats['ellipticity']:
        arr = np.array(stats['ellipticity'])
        print(f"    ellipticity: min={arr.min():.2f}, median={np.median(arr):.2f}, max={arr.max():.2f}")
    if stats['stellarity']:
        arr = np.array(stats['stellarity'])
        print(f"    stellarity: min={arr.min():.2f}, median={np.median(arr):.2f}, max={arr.max():.2f}")

    print(f"    Classified {len(star_catalog)} sources as stars (stellarity >= {stellarity_threshold})")

    # Dilate star masks to cover PSF wings and halos
    if len(star_catalog) > 0:
        expand_radius = int(PSF_FWHM_PIX * mask_dilation_factor)
        structure = np.ones((expand_radius * 2 + 1, expand_radius * 2 + 1))
        star_mask = binary_dilation(star_mask, structure=structure)
        print(f"    Dilated masks by {expand_radius} pixels")

    print(f"    Star mask: {np.sum(star_mask)} pixels ({100*np.mean(star_mask):.2f}%)")

    return star_mask, star_catalog


def create_combined_mask(
    new_fits_path: Path = Path("fits/b_full.fits"),
    old_fits_path: Path = Path("fits/b.fits"),
    old_mask_path: Path = Path("data/planet_mask2.fits"),
    output_path: Path = Path("data/combined_mask.fits"),
    star_mask_output: Path = Path("data/star_mask.fits"),
    use_old_mask: bool = False,  # Set to False to only use star detection
) -> tuple[np.ndarray, dict]:
    """
    Create a combined mask including:
    1. WCS-transformed original mask (optional, controlled by use_old_mask)
    2. Morphologically detected stars across the full image

    Parameters
    ----------
    use_old_mask : bool
        If True, include the WCS-transformed original mask.
        If False, only use star detection (default).

    Returns
    -------
    combined_mask : np.ndarray
        Boolean mask
    stats : dict
        Statistics about mask components
    """
    print("\n" + "=" * 60)
    print("CREATING COMBINED MASK")
    print("=" * 60)
    print(f"Mode: {'Include old mask + star detection' if use_old_mask else 'Star detection ONLY'}")

    # Load new image
    with fits.open(new_fits_path) as hdul:
        new_data = hdul[0].data.astype(float)
        new_header = hdul[0].header

    shape = new_data.shape
    print(f"Target image shape: {shape}")

    stats = {"shape": shape, "use_old_mask": use_old_mask}
    combined_mask = np.zeros(shape, dtype=bool)

    # 1. Transform original mask via WCS (only if use_old_mask is True)
    if use_old_mask:
        print("\n[1/2] Transforming original mask via WCS...")
        if old_mask_path.exists() and old_fits_path.exists():
            with fits.open(old_mask_path) as hdul:
                old_mask = hdul[0].data.astype(bool)

            transformed_mask = transform_mask_wcs(
                old_mask, old_fits_path, new_fits_path, dilation_radius=3
            )
            combined_mask |= transformed_mask
            stats["original_transformed_pixels"] = int(np.sum(transformed_mask))
            print(f"    Original mask transformed: {stats['original_transformed_pixels']} pixels")
        else:
            print("    Original mask not found, skipping")
            stats["original_transformed_pixels"] = 0
    else:
        print("\n[1/2] Skipping old mask (star detection only mode)")
        stats["original_transformed_pixels"] = 0

    # 2. Detect stars morphologically
    # Stars in HDF are rare but need to be masked
    # Multi-criteria detection based on professional research:
    # - r_half < 3.0 pix (compact, near PSF size of ~2.25 pix)
    # - Concentration C = r50/r90 > 0.42 (SDSS-style, stars are concentrated)
    # - Ellipticity < 0.3 (round, not elongated)
    # - Bright sources (top 30%)
    # - Stellarity score > 0.6 (multi-criteria 0-1 index)
    # References: SExtractor CLASS_STAR, SDSS DR17, COSMOS2020, MU_MAX method
    print("\n[2/2] Detecting stars via professional multi-criteria morphology...")
    star_mask, star_catalog = detect_stars_morphology(
        new_data,
        detection_sigma=2.0,
        star_r_half_max=3.0,           # Compact sources (PSF FWHM ~2.25 pix)
        star_concentration_min=0.42,   # High concentration C = r50/r90 (SDSS-style)
        star_ellipticity_max=0.3,      # Round sources (stars have circular PSF)
        min_flux_percentile=70.0,      # Bright sources (top 30%)
        mask_dilation_factor=5.0,      # Mask PSF wings and halos
        stellarity_threshold=0.6,      # Multi-criteria stellarity score threshold
    )

    combined_mask |= star_mask
    stats["star_pixels"] = int(np.sum(star_mask))
    stats["n_stars"] = len(star_catalog)

    # Save star mask separately
    if star_mask_output:
        star_header = new_header.copy()
        star_header["COMMENT"] = "Star mask from morphological detection"
        hdu = fits.PrimaryHDU(star_mask.astype(np.uint8), header=star_header)
        hdu.writeto(star_mask_output, overwrite=True)
        print(f"  Saved star mask: {star_mask_output}")

    # Final stats
    stats["combined_pixels"] = int(np.sum(combined_mask))
    stats["combined_fraction"] = float(np.mean(combined_mask))

    print("\n" + "-" * 40)
    print("MASK SUMMARY:")
    print(f"  Original mask (transformed): {stats['original_transformed_pixels']:,} pixels")
    print(f"  Star mask: {stats['star_pixels']:,} pixels ({stats['n_stars']} stars)")
    print(f"  Combined: {stats['combined_pixels']:,} pixels ({100*stats['combined_fraction']:.2f}%)")

    # Save combined mask
    if output_path:
        mask_header = new_header.copy()
        mask_header["BUNIT"] = "MASK"
        mask_header["COMMENT"] = "Combined mask: WCS-transformed original + morphological stars"
        hdu = fits.PrimaryHDU(combined_mask.astype(np.uint8), header=mask_header)
        hdu.writeto(output_path, overwrite=True)
        print(f"\nSaved combined mask: {output_path}")

    return combined_mask, stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create star mask for HDF images")
    parser.add_argument(
        "--use-old-mask",
        action="store_true",
        help="Include the WCS-transformed original mask (default: star detection only)"
    )
    args = parser.parse_args()

    combined_mask, stats = create_combined_mask(use_old_mask=args.use_old_mask)
    print("\nDone!")
