#!/usr/bin/env python3
"""
Professional Star-Galaxy Classification Pipeline

This script demonstrates how to use the professional-grade star-galaxy
classification following research standards from major surveys.

The classification uses a multi-tier approach:
1. Gaia DR3 cross-match: Definitive foreground star identification
2. SPREAD_MODEL: PSF vs extended source morphology comparison
3. ML classifier: Random Forest with photometric+morphological features
4. Color-color stellar locus: Supplementary color-based check

Usage:
    python run_professional_classification.py

This will run on the existing catalog and produce:
- Classified catalog with star/galaxy labels
- Diagnostic plots showing classification quality
- Validation metrics (if reference catalog available)

Requirements:
    pip install astropy astroquery pandas numpy scipy matplotlib

Author: Professional classification implementation based on:
- Cook et al. 2024, MNRAS (WAVES UMAP+HDBSCAN)
- Desai et al. 2012, ApJ (SPREAD_MODEL)
- Baqui et al. 2021, A&A (miniJPAS ML classification)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

# Configuration
HDF_CENTER_RA = 189.228621  # degrees
HDF_CENTER_DEC = 62.212572  # degrees
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output/full")


def load_existing_catalog(catalog_path: Path) -> pd.DataFrame:
    """Load an existing galaxy catalog."""
    if catalog_path.suffix == '.csv':
        return pd.read_csv(catalog_path)
    elif catalog_path.suffix == '.fits':
        from astropy.table import Table
        return Table.read(catalog_path).to_pandas()
    else:
        raise ValueError(f"Unsupported catalog format: {catalog_path.suffix}")


def load_reference_image(band: str = 'i') -> tuple[np.ndarray, dict]:
    """Load reference image and header for morphological analysis."""
    science_path = DATA_DIR / f"{band}_full.fits"

    if not science_path.exists():
        # Try alternative paths
        science_path = DATA_DIR / f"hdf_{band}.fits"

    if not science_path.exists():
        raise FileNotFoundError(f"Science image not found: {science_path}")

    with fits.open(science_path) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = dict(hdul[0].header)

    return data, header


def run_professional_classification(
    catalog: pd.DataFrame,
    image: np.ndarray,
    header: dict,
    output_dir: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full professional classification pipeline.

    Parameters
    ----------
    catalog : pd.DataFrame
        Source catalog with xcentroid, ycentroid, and ideally ra, dec
    image : np.ndarray
        Reference image for morphological measurements
    header : dict
        FITS header for WCS
    output_dir : Path
        Output directory for results
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        Catalog with professional classification columns added
    """
    from morphology.star_galaxy import run_full_classification_pipeline

    # Run the pipeline
    classified_catalog, diagnostics = run_full_classification_pipeline(
        catalog=catalog,
        image=image,
        header=header,
        field_ra=HDF_CENTER_RA,
        field_dec=HDF_CENTER_DEC,
        field_radius_arcmin=3.0,
        output_dir=output_dir,
        verbose=verbose,
    )

    return classified_catalog, diagnostics


def compare_with_basic_classification(
    catalog: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """
    Compare professional classification with basic FLAG_PSF_LIKE approach.

    Returns statistics on how many sources changed classification.
    """
    # Basic classification (from quality flags)
    FLAG_PSF_LIKE = 64
    if 'quality_flag' in catalog.columns:
        basic_is_star = (catalog['quality_flag'].astype(int) & FLAG_PSF_LIKE) != 0
    else:
        basic_is_star = pd.Series([False] * len(catalog))

    # Professional classification
    if 'is_star' in catalog.columns:
        pro_is_star = catalog['is_star']
    else:
        pro_is_star = pd.Series([False] * len(catalog))

    # Compare
    agree = basic_is_star == pro_is_star
    disagree = ~agree

    # Stars that professional found but basic missed
    missed_stars = pro_is_star & ~basic_is_star

    # Stars that basic found but professional rejected
    false_stars = basic_is_star & ~pro_is_star

    stats = {
        'total_sources': len(catalog),
        'basic_stars': basic_is_star.sum(),
        'professional_stars': pro_is_star.sum(),
        'agree': agree.sum(),
        'disagree': disagree.sum(),
        'missed_by_basic': missed_stars.sum(),
        'false_positives_basic': false_stars.sum(),
    }

    if verbose:
        print("\n" + "=" * 60)
        print("COMPARISON: Basic vs Professional Classification")
        print("=" * 60)
        print(f"Total sources:          {stats['total_sources']}")
        print(f"Basic method stars:     {stats['basic_stars']}")
        print(f"Professional stars:     {stats['professional_stars']}")
        print(f"Agreement:              {stats['agree']} ({100*stats['agree']/stats['total_sources']:.1f}%)")
        print(f"Disagreement:           {stats['disagree']} ({100*stats['disagree']/stats['total_sources']:.1f}%)")
        print(f"Stars missed by basic:  {stats['missed_by_basic']}")
        print(f"False stars (basic):    {stats['false_positives_basic']}")
        print("=" * 60)

    return stats


def print_classification_summary(catalog: pd.DataFrame) -> None:
    """Print detailed classification summary."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)

    n_total = len(catalog)

    if 'is_galaxy' in catalog.columns:
        n_galaxies = catalog['is_galaxy'].sum()
        n_stars = catalog['is_star'].sum() if 'is_star' in catalog.columns else n_total - n_galaxies
    else:
        n_galaxies = n_total
        n_stars = 0

    print(f"\nTotal sources:  {n_total}")
    print(f"Galaxies:       {n_galaxies} ({100*n_galaxies/n_total:.1f}%)")
    print(f"Stars:          {n_stars} ({100*n_stars/n_total:.1f}%)")

    # Classification by tier
    if 'classification_tier' in catalog.columns:
        print("\nClassification by tier:")
        tier_names = {
            0: "Unclassified (default)",
            1: "Gaia cross-match",
            2: "SPREAD_MODEL",
            3: "Morphology (C+size)",
            4: "ML classifier",
            5: "Color-color locus",
        }
        for tier in range(6):
            n_tier = (catalog['classification_tier'] == tier).sum()
            if n_tier > 0:
                print(f"  Tier {tier} ({tier_names.get(tier, 'Unknown')}): {n_tier}")

    # Confidence distribution
    if 'confidence' in catalog.columns:
        conf = catalog['confidence']
        print("\nConfidence statistics:")
        print(f"  High (>0.8):    {(conf > 0.8).sum()}")
        print(f"  Medium (0.5-0.8): {((conf >= 0.5) & (conf <= 0.8)).sum()}")
        print(f"  Low (<0.5):     {(conf < 0.5).sum()}")

    # Gaia statistics
    if 'gaia_confirmed_star' in catalog.columns:
        n_gaia = catalog['gaia_confirmed_star'].sum()
        print(f"\nGaia-confirmed stars: {n_gaia}")

    print("=" * 60)


def main():
    """Main entry point."""
    print("=" * 60)
    print("PROFESSIONAL STAR-GALAXY CLASSIFICATION")
    print("Following research standards from major surveys")
    print("=" * 60)

    output_dir = OUTPUT_DIR / "professional_classification"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if we have an existing catalog
    catalog_path = OUTPUT_DIR / "galaxy_catalog_full.csv"

    if catalog_path.exists():
        print(f"\nLoading existing catalog: {catalog_path}")
        catalog = load_existing_catalog(catalog_path)
        print(f"Loaded {len(catalog)} sources")
    else:
        print(f"\nNo existing catalog found at {catalog_path}")
        print("Please run the main analysis first: python run_analysis.py")
        sys.exit(1)

    # Load reference image
    print("\nLoading reference image (I-band)...")
    try:
        image, header = load_reference_image('i')
        print(f"Image shape: {image.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data files are in the data/ directory")
        sys.exit(1)

    # Check if we have coordinates
    has_coords = 'ra' in catalog.columns and 'dec' in catalog.columns
    if not has_coords:
        print("\nWarning: Catalog missing RA/Dec coordinates")
        print("Gaia cross-matching will be skipped")

    # Run professional classification
    print("\nRunning professional classification...")
    classified_catalog, _diagnostics = run_professional_classification(
        catalog=catalog,
        image=image,
        header=header,
        output_dir=output_dir,
        verbose=True,
    )

    # Print summary
    print_classification_summary(classified_catalog)

    # Compare with basic classification
    compare_with_basic_classification(classified_catalog)

    # Save results
    output_catalog_path = output_dir / "galaxy_catalog_professional.csv"

    # Filter to galaxies only
    galaxies_only = classified_catalog[classified_catalog['is_galaxy']].copy()
    galaxies_only.to_csv(output_catalog_path, index=False)
    print(f"\nSaved galaxy catalog: {output_catalog_path}")
    print(f"  {len(galaxies_only)} galaxies (stars removed)")

    # Save full catalog with classifications
    full_output_path = output_dir / "full_catalog_with_classification.csv"
    classified_catalog.to_csv(full_output_path, index=False)
    print(f"Saved full catalog: {full_output_path}")

    print("\nDone!")
    print(f"Diagnostic plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
