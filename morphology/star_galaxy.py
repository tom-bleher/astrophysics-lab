"""Star/galaxy classification based on morphological parameters.

This module provides tools for separating point sources (stars)
from extended sources (galaxies) using concentration, size, and
other morphological parameters.

Methods (basic):
- Concentration index (stars are more concentrated)
- Size relative to PSF (stars are unresolved)
- Half-light radius comparison
- ML-based classification (Random Forest with combined features)

For professional-grade classification, use the professional_classification
module which implements:
- Multi-tier classification (Gaia → SPREAD_MODEL → ML → Color)
- Empirical PSF measurement from Gaia stars
- SPREAD_MODEL morphology comparison
- Magnitude-dependent thresholds
- Comprehensive validation metrics

References:
- Odewahn et al. 1992, AJ, 103, 318
- Abraham et al. 1994, ApJ, 432, 75
- Baqui et al. 2021, A&A, 645, A87 (miniJPAS ML classification)
- Cook et al. 2024, MNRAS (WAVES UMAP+HDBSCAN)
- Desai et al. 2012, ApJ (SPREAD_MODEL)
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from astroquery.gaia import Gaia
from numpy.typing import NDArray

from morphology.ml_classifier import (
    MLStarGalaxyClassifier,
    extract_features_from_catalog,
)

if TYPE_CHECKING:
    pass


def get_stellarity_index(
    flux_radius: float,
    psf_fwhm: float = 2.5,
) -> float:
    """Calculate a simple stellarity index.

    Compares the source's flux radius to the PSF FWHM.
    Stars should have flux_radius approximately equal to PSF.

    Parameters
    ----------
    flux_radius : float
        Source half-light radius in pixels
    psf_fwhm : float
        PSF FWHM in pixels (default: 2.5 for HST/WFPC2)

    Returns
    -------
    float
        Stellarity index (0 = extended galaxy, 1 = point source)
    """
    # Ratio of source size to PSF size
    size_ratio = flux_radius / (psf_fwhm / 2.35)  # Convert FWHM to sigma

    # Convert to stellarity (0 to 1)
    # Stars have size_ratio ~ 1, galaxies have size_ratio > 1
    stellarity = np.exp(-(size_ratio - 1) ** 2 / 0.5)

    return np.clip(stellarity, 0, 1)


def classify_star_galaxy(
    image: NDArray,
    catalog: pd.DataFrame,
    x_col: str = "xcentroid",
    y_col: str = "ycentroid",
    c_threshold: float = 0.5,
    method: str = "concentration",
    psf_fwhm: float = 2.5,
) -> pd.Series:
    """Classify sources as stars or galaxies.

    Parameters
    ----------
    image : NDArray
        2D image array
    catalog : pd.DataFrame
        Source catalog with position columns
    x_col, y_col : str
        Column names for centroid positions
    c_threshold : float
        Concentration threshold for classification
    method : str
        Classification method: 'concentration', 'size', or 'combined'
    psf_fwhm : float
        PSF FWHM in pixels

    Returns
    -------
    pd.Series
        Boolean series: True = galaxy, False = star
    """
    from morphology.concentration import (
        concentration_index_batch,
        half_light_radius_batch,
    )

    # Extract coordinates as numpy arrays for vectorized processing
    x_coords = catalog[x_col].values
    y_coords = catalog[y_col].values
    psf_sigma = psf_fwhm / 2.35

    if method == "concentration":
        # Vectorized concentration calculation
        c_values = concentration_index_batch(image, x_coords, y_coords)
        # Stars are MORE concentrated (lower concentration index)
        is_galaxy = np.where(np.isfinite(c_values), c_values > c_threshold, True)

    elif method == "size":
        # Vectorized half-light radius calculation
        r_half_values = half_light_radius_batch(image, x_coords, y_coords)
        # Galaxies are larger than PSF
        is_galaxy = np.where(
            np.isfinite(r_half_values), r_half_values > 1.5 * psf_sigma, True
        )

    elif method == "combined":
        # Vectorized computation of both metrics
        c_values = concentration_index_batch(image, x_coords, y_coords)
        r_half_values = half_light_radius_batch(image, x_coords, y_coords)

        c_galaxy = np.where(np.isfinite(c_values), c_values > c_threshold, True)
        size_galaxy = np.where(
            np.isfinite(r_half_values), r_half_values > 1.5 * psf_sigma, True
        )

        # Both criteria must agree for confident classification
        is_galaxy = c_galaxy | size_galaxy

    else:
        raise ValueError(f"Unknown method: {method}")

    return pd.Series(is_galaxy, index=catalog.index, dtype=bool)


def classify_star_galaxy_batch(
    image: NDArray,
    catalog: pd.DataFrame,
    x_col: str = "xcentroid",
    y_col: str = "ycentroid",
    psf_fwhm: float = 2.5,
    _snr_threshold: float = 5.0,
    method: str = "combined",
    ml_model_path: Path | str | None = None,
    flux_cols: dict[str, str] | None = None,
    error_cols: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Classify stars/galaxies with multiple parameters and confidence.

    Parameters
    ----------
    image : NDArray
        2D image array
    catalog : pd.DataFrame
        Source catalog
    x_col, y_col : str
        Position column names
    psf_fwhm : float
        PSF FWHM in pixels
    snr_threshold : float
        Minimum SNR for reliable classification
    method : str
        Classification method: 'concentration', 'size', 'combined', or 'ml'
        - 'concentration': Uses concentration index C > 2.8 threshold
        - 'size': Uses half-light radius vs PSF comparison
        - 'combined': Uses both concentration and size (default)
        - 'ml': Uses trained Random Forest classifier (requires ml_model_path)
    ml_model_path : Path or str, optional
        Path to trained ML model file (required if method='ml')
    flux_cols : dict, optional
        Mapping of band name to flux column (for ML method)
    error_cols : dict, optional
        Mapping of band name to error column (for ML method)

    Returns
    -------
    pd.DataFrame
        Classification results with columns:
        - is_galaxy: Boolean classification
        - stellarity: Stellarity index (0-1)
        - concentration: Concentration index
        - half_light_radius: Half-light radius
        - confidence: Classification confidence
        - ml_prob_galaxy: (ML method only) Probability of being a galaxy
    """
    # Handle ML method
    if method == "ml":
        if ml_model_path is None:
            raise ValueError("ml_model_path required for method='ml'")

        # Load trained model
        classifier = MLStarGalaxyClassifier.load(ml_model_path)

        # Extract features
        features = extract_features_from_catalog(
            catalog, image, x_col, y_col, flux_cols, error_cols
        )

        # Get predictions
        results = classifier.predict(features)

        # Build output DataFrame matching expected format
        output = []
        for idx, r in zip(catalog.index, results, strict=False):
            output.append(
                {
                    "source_id": idx,
                    "is_galaxy": r.is_galaxy,
                    "stellarity": r.probability_star,
                    "concentration": features.loc[idx, "concentration_c"],
                    "concentration_c": features.loc[idx, "concentration_c"],
                    "half_light_radius": features.loc[idx, "half_light_radius"],
                    "confidence": r.confidence,
                    "ml_prob_galaxy": r.probability_galaxy,
                }
            )
        return pd.DataFrame(output)

    # Classical methods - use vectorized batch processing
    from morphology.concentration import compute_morphology_batch

    # Extract coordinates as numpy arrays
    x_coords = catalog[x_col].values
    y_coords = catalog[y_col].values
    n_sources = len(catalog)

    # Compute all morphological parameters at once (much faster than iterrows)
    morphology = compute_morphology_batch(image, x_coords, y_coords)

    c_values = morphology['concentration']
    c_full_values = morphology['concentration_c']
    r_half_values = morphology['half_light_radius']

    # Vectorized stellarity calculation
    stellarity_values = np.full(n_sources, np.nan)
    valid_r_half = np.isfinite(r_half_values)

    stellarity_values[valid_r_half] = np.array([
        get_stellarity_index(r_half_values[i], psf_fwhm)
        for i in np.where(valid_r_half)[0]
    ])

    # Vectorized classification logic
    psf_sigma = psf_fwhm / 2.35

    # Size criterion (stars are smaller than PSF)
    size_star = np.where(np.isfinite(r_half_values), r_half_values < 1.5 * psf_sigma, False)

    # Concentration criterion (stars have C > 2.8 typically)
    conc_star = np.where(np.isfinite(c_full_values), c_full_values > 2.8, False)

    # Vectorized classification
    is_galaxy = np.full(n_sources, True)  # Default to galaxy
    confidence = np.full(n_sources, 0.5)

    # Both criteria agree: star
    both_star = size_star & conc_star
    is_galaxy[both_star] = False
    confidence[both_star] = 0.9

    # Both criteria agree: galaxy
    both_galaxy = ~size_star & ~conc_star
    confidence[both_galaxy] = 0.9

    # Disagreement - use size as primary
    disagree = size_star != conc_star
    is_galaxy[disagree] = ~size_star[disagree]
    confidence[disagree] = 0.6

    # Build results DataFrame
    results = pd.DataFrame({
        "source_id": catalog.index,
        "is_galaxy": is_galaxy,
        "stellarity": stellarity_values,
        "concentration": c_values,
        "concentration_c": c_full_values,
        "half_light_radius": r_half_values,
        "confidence": confidence,
    })

    return results


def apply_star_mask(
    catalog: pd.DataFrame,
    is_galaxy: pd.Series,
    verbose: bool = True,
) -> pd.DataFrame:
    """Filter catalog to only include galaxies.

    Parameters
    ----------
    catalog : pd.DataFrame
        Source catalog
    is_galaxy : pd.Series
        Boolean classification from classify_star_galaxy()
    verbose : bool
        Print summary statistics

    Returns
    -------
    pd.DataFrame
        Catalog with only galaxy sources
    """
    n_total = len(catalog)
    n_galaxies = is_galaxy.sum()
    n_stars = n_total - n_galaxies

    if verbose:
        print("Star/galaxy classification:")
        print(f"  Total sources: {n_total}")
        print(f"  Stars:         {n_stars} ({100*n_stars/n_total:.1f}%)")
        print(f"  Galaxies:      {n_galaxies} ({100*n_galaxies/n_total:.1f}%)")

    return catalog[is_galaxy].copy()


def plot_star_galaxy_separation(
    classification_results: pd.DataFrame,
    figsize: tuple = (12, 5),
):
    """Plot star/galaxy separation diagnostics.

    Parameters
    ----------
    classification_results : pd.DataFrame
        Results from classify_star_galaxy_batch()
    figsize : tuple
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    df = classification_results

    # Left: Size vs concentration
    ax1 = axes[0]
    galaxies = df[df["is_galaxy"]]
    stars = df[~df["is_galaxy"]]

    ax1.scatter(
        galaxies["half_light_radius"],
        galaxies["concentration_c"],
        alpha=0.5,
        s=20,
        c="blue",
        label=f"Galaxies ({len(galaxies)})",
    )
    ax1.scatter(
        stars["half_light_radius"],
        stars["concentration_c"],
        alpha=0.5,
        s=20,
        c="red",
        label=f"Stars ({len(stars)})",
    )

    ax1.axhline(2.8, color="k", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Half-light radius (pixels)")
    ax1.set_ylabel("Concentration C")
    ax1.set_title("Star/Galaxy Separation")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Stellarity histogram
    ax2 = axes[1]
    valid = df["stellarity"].notna()
    ax2.hist(
        df.loc[valid & df["is_galaxy"], "stellarity"],
        bins=20,
        range=(0, 1),
        alpha=0.7,
        label="Galaxies",
        color="blue",
    )
    ax2.hist(
        df.loc[valid & ~df["is_galaxy"], "stellarity"],
        bins=20,
        range=(0, 1),
        alpha=0.7,
        label="Stars",
        color="red",
    )
    ax2.set_xlabel("Stellarity Index")
    ax2.set_ylabel("Count")
    ax2.set_title("Stellarity Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


# =============================================================================
# Professional Classification Integration
# =============================================================================

def classify_professional(
    catalog: pd.DataFrame,
    image: np.ndarray,
    wcs=None,
    gaia_catalog: pd.DataFrame | None = None,
    ml_classifier=None,
    x_col: str = "xcentroid",
    y_col: str = "ycentroid",
    ra_col: str = "ra",
    dec_col: str = "dec",
    mag_col: str = "mag_auto",
    flux_cols: dict | None = None,
    error_cols: dict | None = None,
    pixel_scale: float = 0.04,
    completeness_priority: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Professional multi-tier star-galaxy classification.

    This is a convenience wrapper around the professional_classification module.
    It implements survey-standard classification using:

    Tier 1: Gaia cross-match (100% confidence for bright stars)
    Tier 2: SPREAD_MODEL morphology (high confidence)
    Tier 3: Classical morphology (concentration + size)
    Tier 4: ML classifier (if available)
    Tier 5: Color-color stellar locus

    Parameters
    ----------
    catalog : pd.DataFrame
        Source catalog with positions and photometry
    image : NDArray
        2D image array for morphological measurements
    wcs : WCS, optional
        WCS for coordinate transformation
    gaia_catalog : pd.DataFrame, optional
        Gaia DR3 catalog for cross-matching
    ml_classifier : optional
        Trained ML classifier (MLStarGalaxyClassifier)
    x_col, y_col : str
        Column names for pixel positions
    ra_col, dec_col : str
        Column names for sky coordinates
    mag_col : str
        Column name for magnitude (for thresholds)
    flux_cols : dict, optional
        Mapping of band name to flux column
    error_cols : dict, optional
        Mapping of band name to error column
    pixel_scale : float
        Pixel scale in arcsec/pixel
    completeness_priority : bool, optional
        If True, use relaxed thresholds that prioritize completeness over
        purity. Useful for deep surveys where maximizing galaxy detection
        is more important than minimizing stellar contamination. Default False.
    verbose : bool
        Print progress information

    Returns
    -------
    pd.DataFrame
        Classification results with columns:
        - is_galaxy: Final classification
        - is_star: Inverse of is_galaxy
        - probability_galaxy: Probability (0-1)
        - confidence: Classification confidence (0-1)
        - classification_method: Primary method used
        - classification_tier: Tier of classification
        - spread_model: SPREAD_MODEL value
        - gaia_confirmed_star: Confirmed star from Gaia
        - concentration_c: Concentration index
        - half_light_radius: Half-light radius

    Example
    -------
    >>> from morphology.star_galaxy import classify_professional
    >>> from download_hdf_data import query_gaia_stars
    >>>
    >>> # Query Gaia stars
    >>> gaia_stars = query_gaia_stars(ra=189.2, dec=62.2, radius_arcmin=3.0)
    >>> gaia_df = pd.DataFrame(gaia_stars)
    >>>
    >>> # Run professional classification
    >>> results = classify_professional(
    ...     catalog, image, wcs=wcs, gaia_catalog=gaia_df
    ... )
    >>>
    >>> # Filter to galaxies only
    >>> galaxies = catalog[results['is_galaxy']]
    """
    from morphology.professional_classification import (
        classify_professional as _classify_professional,
    )

    return _classify_professional(
        catalog=catalog,
        image=image,
        wcs=wcs,
        gaia_catalog=gaia_catalog,
        ml_classifier=ml_classifier,
        x_col=x_col,
        y_col=y_col,
        ra_col=ra_col,
        dec_col=dec_col,
        mag_col=mag_col,
        flux_cols=flux_cols,
        error_cols=error_cols,
        pixel_scale=pixel_scale,
        completeness_priority=completeness_priority,
        verbose=verbose,
    )


def query_gaia_for_classification(
    ra_center: float,
    dec_center: float,
    radius_arcmin: float = 3.0,
    magnitude_limit: float = 21.0,
) -> pd.DataFrame:
    """Query Gaia DR3 for star-galaxy classification.

    Convenience function to query Gaia and return a DataFrame
    suitable for use with classify_professional().

    Parameters
    ----------
    ra_center, dec_center : float
        Field center in degrees
    radius_arcmin : float
        Search radius in arcminutes
    magnitude_limit : float
        Faint magnitude limit (Gaia completeness drops at G > 21)

    Returns
    -------
    pd.DataFrame
        Gaia catalog with columns needed for classification
    """
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
            CIRCLE('ICRS', {ra_center}, {dec_center}, {radius_arcmin / 60.0})
        ) = 1
        AND phot_g_mean_mag < {magnitude_limit}
    ORDER BY phot_g_mean_mag ASC
    """

    try:
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        Gaia.ROW_LIMIT = -1

        job = Gaia.launch_job(query)
        results = job.get_results()

        return results.to_pandas()

    except Exception as e:
        print(f"Error querying Gaia: {e}")
        return pd.DataFrame()


def run_full_classification_pipeline(
    catalog: pd.DataFrame,
    image: np.ndarray,
    header: dict,
    field_ra: float = 189.228621,
    field_dec: float = 62.212572,
    field_radius_arcmin: float = 3.0,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Run the complete professional classification pipeline.

    This is the main entry point for professional star-galaxy classification.
    It handles:
    1. Querying Gaia DR3 for foreground stars
    2. Measuring empirical PSF from Gaia stars
    3. Running multi-tier classification
    4. Generating diagnostic plots
    5. Computing validation metrics (if reference available)

    Parameters
    ----------
    catalog : pd.DataFrame
        Source catalog with positions (xcentroid, ycentroid) and
        sky coordinates (ra, dec) if available
    image : NDArray
        2D science image
    header : dict
        FITS header with WCS information
    field_ra, field_dec : float
        Field center for Gaia query (degrees)
    field_radius_arcmin : float
        Gaia query radius (arcminutes)
    output_dir : Path, optional
        Directory for diagnostic outputs
    verbose : bool
        Print progress information

    Returns
    -------
    results : pd.DataFrame
        Classification results merged with input catalog
    diagnostics : dict
        Dictionary with PSF model, metrics, and other diagnostics
    """
    from astropy.wcs import WCS

    if verbose:
        print("\n" + "=" * 60)
        print("PROFESSIONAL STAR-GALAXY CLASSIFICATION PIPELINE")
        print("=" * 60)

    diagnostics = {}

    # Get WCS
    try:
        wcs = WCS(header)
        if not wcs.has_celestial:
            wcs = None
    except Exception:
        wcs = None

    # Query Gaia
    if verbose:
        print(f"\nQuerying Gaia DR3 at RA={field_ra:.4f}, Dec={field_dec:.4f}...")

    gaia_catalog = query_gaia_for_classification(
        ra_center=field_ra,
        dec_center=field_dec,
        radius_arcmin=field_radius_arcmin,
        magnitude_limit=21.0,
    )

    if len(gaia_catalog) > 0:
        if verbose:
            print(f"Retrieved {len(gaia_catalog)} Gaia sources")
        diagnostics['gaia_sources'] = len(gaia_catalog)
    else:
        if verbose:
            print("Warning: No Gaia sources retrieved")
        diagnostics['gaia_sources'] = 0

    # Run classification
    results = classify_professional(
        catalog=catalog,
        image=image,
        wcs=wcs,
        gaia_catalog=gaia_catalog,
        x_col="xcentroid",
        y_col="ycentroid",
        verbose=verbose,
    )

    # Store diagnostics
    diagnostics['n_galaxies'] = results['is_galaxy'].sum()
    diagnostics['n_stars'] = results['is_star'].sum()
    diagnostics['gaia_confirmed_stars'] = results['gaia_confirmed_star'].sum() if 'gaia_confirmed_star' in results.columns else 0

    # Merge results with input catalog
    # Keep only classification columns
    classification_cols = [
        'is_galaxy', 'is_star', 'probability_galaxy', 'confidence',
        'classification_method', 'classification_tier', 'classification_flags',
        'spread_model', 'spread_model_err', 'gaia_match', 'gaia_confirmed_star',
        'gaia_parallax', 'gaia_pmtot', 'concentration_c', 'half_light_radius',
        'stellar_locus_distance'
    ]

    for col in classification_cols:
        if col in results.columns and col not in catalog.columns:
            catalog[col] = results[col].values

    # Generate diagnostic plot
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        from morphology.professional_classification import plot_classification_diagnostics
        plot_classification_diagnostics(
            results,
            output_path=output_dir / "star_galaxy_diagnostics.pdf"
        )

    if verbose:
        print("\n" + "=" * 60)
        print("CLASSIFICATION COMPLETE")
        print("=" * 60)
        print(f"Galaxies: {diagnostics['n_galaxies']}")
        print(f"Stars:    {diagnostics['n_stars']}")
        print(f"Gaia-confirmed stars: {diagnostics['gaia_confirmed_stars']}")

    return catalog, diagnostics
