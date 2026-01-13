"""Star/galaxy classification based on morphological parameters.

This module provides tools for separating point sources (stars)
from extended sources (galaxies) using concentration, size, and
other morphological parameters.

Methods:
- Concentration index (stars are more concentrated)
- Size relative to PSF (stars are unresolved)
- Half-light radius comparison
- ML-based classification (Random Forest with combined features)

References:
- Odewahn et al. 1992, AJ, 103, 318
- Abraham et al. 1994, ApJ, 432, 75
- Baqui et al. 2021, A&A, 645, A87 (miniJPAS ML classification)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def get_stellarity_index(
    flux_radius: float,
    peak_flux: float,
    psf_fwhm: float = 2.5,
) -> float:
    """Calculate a simple stellarity index.

    Compares the source's flux radius to the PSF FWHM.
    Stars should have flux_radius approximately equal to PSF.

    Parameters
    ----------
    flux_radius : float
        Source half-light radius in pixels
    peak_flux : float
        Peak flux (not used directly but can be for SNR weighting)
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
    from morphology.concentration import concentration_index, half_light_radius

    is_galaxy = pd.Series(index=catalog.index, dtype=bool)

    if method == "concentration":
        for idx, row in catalog.iterrows():
            x, y = row[x_col], row[y_col]
            c = concentration_index(image, x, y)
            # Stars are MORE concentrated (lower concentration index)
            is_galaxy[idx] = c > c_threshold if np.isfinite(c) else True

    elif method == "size":
        psf_sigma = psf_fwhm / 2.35
        for idx, row in catalog.iterrows():
            x, y = row[x_col], row[y_col]
            r_half = half_light_radius(image, x, y)
            # Galaxies are larger than PSF
            is_galaxy[idx] = r_half > 1.5 * psf_sigma if np.isfinite(r_half) else True

    elif method == "combined":
        for idx, row in catalog.iterrows():
            x, y = row[x_col], row[y_col]

            c = concentration_index(image, x, y)
            r_half = half_light_radius(image, x, y)

            c_galaxy = c > c_threshold if np.isfinite(c) else True

            psf_sigma = psf_fwhm / 2.35
            size_galaxy = r_half > 1.5 * psf_sigma if np.isfinite(r_half) else True

            # Both criteria must agree for confident classification
            is_galaxy[idx] = c_galaxy or size_galaxy

    else:
        raise ValueError(f"Unknown method: {method}")

    return is_galaxy


def classify_star_galaxy_batch(
    image: NDArray,
    catalog: pd.DataFrame,
    x_col: str = "xcentroid",
    y_col: str = "ycentroid",
    psf_fwhm: float = 2.5,
    snr_threshold: float = 5.0,
    method: str = "combined",
    ml_model_path: Optional[Path | str] = None,
    flux_cols: Optional[dict[str, str]] = None,
    error_cols: Optional[dict[str, str]] = None,
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
        try:
            from morphology.ml_classifier import (
                MLStarGalaxyClassifier,
                extract_features_from_catalog,
            )
        except ImportError as e:
            raise ImportError(
                "ML classification requires scikit-learn. "
                "Install with: pip install scikit-learn joblib"
            ) from e

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
        for idx, r in zip(catalog.index, results):
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

    # Classical methods
    from morphology.concentration import (
        calculate_concentration_c,
        concentration_index,
        half_light_radius,
    )

    results = []

    for idx, row in catalog.iterrows():
        x, y = row[x_col], row[y_col]

        # Measure parameters
        c = concentration_index(image, x, y)
        c_full = calculate_concentration_c(image, x, y)
        r_half = half_light_radius(image, x, y)

        # Calculate stellarity
        if np.isfinite(r_half):
            stellarity = get_stellarity_index(r_half, image[int(y), int(x)], psf_fwhm)
        else:
            stellarity = np.nan

        # Classification logic
        psf_sigma = psf_fwhm / 2.35

        # Size criterion
        size_star = r_half < 1.5 * psf_sigma if np.isfinite(r_half) else False

        # Concentration criterion (stars have C > 2.8 typically)
        conc_star = c_full > 2.8 if np.isfinite(c_full) else False

        # Combined classification
        if size_star and conc_star:
            is_galaxy = False
            confidence = 0.9
        elif not size_star and not conc_star:
            is_galaxy = True
            confidence = 0.9
        elif size_star != conc_star:
            # Disagreement - use size as primary
            is_galaxy = not size_star
            confidence = 0.6
        else:
            is_galaxy = True  # Default to galaxy
            confidence = 0.5

        results.append({
            "source_id": idx,
            "is_galaxy": is_galaxy,
            "stellarity": stellarity,
            "concentration": c,
            "concentration_c": c_full,
            "half_light_radius": r_half,
            "confidence": confidence,
        })

    return pd.DataFrame(results)


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
        print(f"Star/galaxy classification:")
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
