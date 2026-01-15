"""Cross-match with external catalogs for star-galaxy validation.

Uses the 3D-HST/CANDELS catalog (Skelton et al. 2014) which provides:
- star_flag: Star/galaxy classification from SExtractor + visual inspection
- class_star: SExtractor stellarity parameter
- z_spec: Spectroscopic redshifts from literature compilation
- z_peak: Photometric redshifts from EAZY

VizieR catalog: J/ApJS/214/24
Reference: Skelton et al. 2014, ApJS, 214, 24

This module downloads the GOODS-N subset and cross-matches with the
local HDF catalog to:
1. Identify misclassified stars using the 3D-HST star_flag
2. Validate photometric redshifts against spectroscopic redshifts
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Optional imports - graceful fallback if not available
try:
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    print("Warning: astropy not available. Cross-matching disabled.")

try:
    from astroquery.vizier import Vizier
    ASTROQUERY_AVAILABLE = True
except ImportError:
    ASTROQUERY_AVAILABLE = False
    print("Warning: astroquery not available. VizieR queries disabled.")


# Constants
VIZIER_3DHST_CATALOG = "J/ApJS/214/24"  # Skelton et al. 2014
GOODS_N_CENTER_RA = 189.2  # degrees (12h 36m 49s)
GOODS_N_CENTER_DEC = 62.22  # degrees (+62° 13' 58")
GOODS_N_RADIUS = 0.15  # degrees (~9 arcmin, covers HDF-N)

# Cache directory for downloaded catalogs
CACHE_DIR = Path(__file__).parent.parent / "data" / "external"


def download_3dhst_goodsn(
    force_download: bool = False,
    cache_file: str = "3dhst_goodsn.csv"
) -> pd.DataFrame | None:
    """Download 3D-HST catalog for GOODS-N field from VizieR.

    Parameters
    ----------
    force_download : bool
        If True, re-download even if cached file exists
    cache_file : str
        Name of cached CSV file

    Returns
    -------
    pd.DataFrame or None
        Catalog with columns: ra, dec, star_flag, class_star, z_spec, z_peak, use_phot
        Returns None if download fails or dependencies missing
    """
    if not ASTROQUERY_AVAILABLE or not ASTROPY_AVAILABLE:
        print("Error: astroquery and astropy required for VizieR queries")
        print("Install with: pip install astroquery astropy")
        return None

    cache_path = CACHE_DIR / cache_file

    # Check cache first
    if cache_path.exists() and not force_download:
        print(f"Loading cached 3D-HST catalog from {cache_path}")
        return pd.read_csv(cache_path)

    print(f"Downloading 3D-HST GOODS-N catalog from VizieR ({VIZIER_3DHST_CATALOG})...")

    try:
        # Configure Vizier query
        v = Vizier(
            columns=['RAJ2000', 'DEJ2000', 'S/G', 'St', 'Use', 'zsp', 'zpk', 'Field'],
            row_limit=-1,  # No row limit
            timeout=120
        )

        # Query region around GOODS-N center
        center = SkyCoord(ra=GOODS_N_CENTER_RA, dec=GOODS_N_CENTER_DEC, unit='deg')
        result = v.query_region(
            center,
            radius=GOODS_N_RADIUS * u.deg,
            catalog=VIZIER_3DHST_CATALOG
        )

        if not result or len(result) == 0:
            print("Warning: No results from VizieR query")
            return None

        # Convert to pandas DataFrame
        table = result[0]
        df = table.to_pandas()

        # Rename columns for clarity
        column_map = {
            'RAJ2000': 'ra',
            'DEJ2000': 'dec',
            'S/G': 'star_flag',      # 0=galaxy, 1/2=star
            'St': 'near_star',       # 1=close to bright star
            'Use': 'use_phot',       # 1=use, 0=don't use
            'zsp': 'z_spec',         # Spectroscopic redshift
            'zpk': 'z_peak',         # Photometric redshift (EAZY)
            'Field': 'field'
        }
        df = df.rename(columns=column_map)

        # Filter to GOODS-N field only (field name contains 'goodsn' or 'GOODS-N')
        if 'field' in df.columns:
            goods_n_mask = df['field'].str.lower().str.contains('goods', na=False)
            df = df[goods_n_mask]
            print(f"  Filtered to GOODS-N: {len(df)} sources")

        # Cache the result
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"  Cached to {cache_path}")

        print(f"  Downloaded {len(df)} sources")
        print(f"  Stars (star_flag > 0): {(df['star_flag'] > 0).sum()}")
        print(f"  With spec-z: {df['z_spec'].notna().sum()}")

        return df

    except Exception as e:
        print(f"Error downloading from VizieR: {e}")
        return None


def crossmatch_with_3dhst(
    catalog: pd.DataFrame,
    match_radius_arcsec: float = 1.0,
    ra_col: str = 'ra',
    dec_col: str = 'dec'
) -> pd.DataFrame:
    """Cross-match local catalog with 3D-HST to identify stars and get spec-z.

    Parameters
    ----------
    catalog : pd.DataFrame
        Local catalog with ra, dec columns
    match_radius_arcsec : float
        Maximum separation for matching (arcseconds)
    ra_col, dec_col : str
        Column names for coordinates

    Returns
    -------
    pd.DataFrame
        Input catalog with added columns:
        - match_3dhst: True if matched to 3D-HST source
        - star_3dhst: True if 3D-HST classifies as star (star_flag > 0)
        - z_spec_3dhst: Spectroscopic redshift from 3D-HST
        - z_peak_3dhst: EAZY photo-z from 3D-HST
        - sep_3dhst: Separation in arcsec to matched source
    """
    if not ASTROPY_AVAILABLE:
        print("Error: astropy required for cross-matching")
        return catalog

    # Download or load 3D-HST catalog
    ref_catalog = download_3dhst_goodsn()
    if ref_catalog is None or len(ref_catalog) == 0:
        print("Warning: Could not load 3D-HST catalog. Skipping cross-match.")
        return catalog

    print(f"\nCross-matching {len(catalog)} sources with 3D-HST ({len(ref_catalog)} sources)...")

    # Create SkyCoord objects
    coords_local = SkyCoord(
        ra=catalog[ra_col].values * u.deg,
        dec=catalog[dec_col].values * u.deg
    )
    coords_3dhst = SkyCoord(
        ra=ref_catalog['ra'].values * u.deg,
        dec=ref_catalog['dec'].values * u.deg
    )

    # Find nearest neighbor for each local source
    idx, sep2d, _ = coords_local.match_to_catalog_sky(coords_3dhst)

    # Apply match radius
    matched = sep2d.arcsec < match_radius_arcsec

    # Initialize new columns
    catalog = catalog.copy()
    catalog['match_3dhst'] = matched
    catalog['star_3dhst'] = False
    catalog['z_spec_3dhst'] = np.nan
    catalog['z_peak_3dhst'] = np.nan
    catalog['sep_3dhst'] = sep2d.arcsec

    # Fill matched values
    catalog.loc[matched, 'star_3dhst'] = (ref_catalog.iloc[idx[matched]]['star_flag'].values > 0)
    catalog.loc[matched, 'z_spec_3dhst'] = ref_catalog.iloc[idx[matched]]['z_spec'].values
    catalog.loc[matched, 'z_peak_3dhst'] = ref_catalog.iloc[idx[matched]]['z_peak'].values

    # Summary statistics
    n_matched = matched.sum()
    n_stars_3dhst = catalog['star_3dhst'].sum()
    n_with_specz = catalog['z_spec_3dhst'].notna().sum()

    print(f"  Matched: {n_matched} ({100*n_matched/len(catalog):.1f}%)")
    print(f"  3D-HST stars in sample: {n_stars_3dhst}")
    print(f"  With spec-z: {n_with_specz}")

    return catalog


def validate_star_classification(
    catalog: pd.DataFrame,
    our_star_col: str = 'pro_is_star',
    match_radius_arcsec: float = 1.0
) -> dict:
    """Validate our star classification against 3D-HST.

    Parameters
    ----------
    catalog : pd.DataFrame
        Catalog with our star classification and coordinates
    our_star_col : str
        Column name for our star classification (True=star)
    match_radius_arcsec : float
        Match radius in arcseconds

    Returns
    -------
    dict
        Validation metrics including:
        - n_matched: Number of matched sources
        - n_agree_star: Both classify as star
        - n_agree_galaxy: Both classify as galaxy
        - n_our_star_their_galaxy: We say star, they say galaxy
        - n_our_galaxy_their_star: We say galaxy, they say star (CONTAMINATION)
        - accuracy: Overall agreement rate
        - contamination_rate: Fraction of our galaxies that are actually stars
    """
    # Cross-match if not already done
    if 'star_3dhst' not in catalog.columns:
        catalog = crossmatch_with_3dhst(catalog, match_radius_arcsec)

    # Filter to matched sources
    matched = catalog[catalog['match_3dhst']].copy()

    if len(matched) == 0:
        print("No matched sources for validation")
        return {}

    # Get classifications
    our_star = matched[our_star_col].fillna(False).astype(bool)
    their_star = matched['star_3dhst'].fillna(False).astype(bool)

    # Calculate metrics
    n_agree_star = (our_star & their_star).sum()
    n_agree_galaxy = (~our_star & ~their_star).sum()
    n_our_star_their_galaxy = (our_star & ~their_star).sum()
    n_our_galaxy_their_star = (~our_star & their_star).sum()  # CONTAMINATION

    n_matched = len(matched)
    n_agree = n_agree_star + n_agree_galaxy
    accuracy = n_agree / n_matched if n_matched > 0 else 0

    # Contamination: galaxies in our sample that 3D-HST says are stars
    n_our_galaxies = (~our_star).sum()
    contamination_rate = n_our_galaxy_their_star / n_our_galaxies if n_our_galaxies > 0 else 0

    results = {
        'n_matched': n_matched,
        'n_agree_star': int(n_agree_star),
        'n_agree_galaxy': int(n_agree_galaxy),
        'n_our_star_their_galaxy': int(n_our_star_their_galaxy),
        'n_our_galaxy_their_star': int(n_our_galaxy_their_star),
        'accuracy': accuracy,
        'contamination_rate': contamination_rate
    }

    print("\n=== Star Classification Validation (vs 3D-HST) ===")
    print(f"  Matched sources: {n_matched}")
    print(f"  Agreement (star): {n_agree_star}")
    print(f"  Agreement (galaxy): {n_agree_galaxy}")
    print(f"  We=star, 3D-HST=galaxy: {n_our_star_their_galaxy}")
    print(f"  We=galaxy, 3D-HST=star: {n_our_galaxy_their_star} (CONTAMINATION)")
    print(f"  Accuracy: {100*accuracy:.1f}%")
    print(f"  Galaxy contamination rate: {100*contamination_rate:.1f}%")

    return results


def validate_photoz(
    catalog: pd.DataFrame,
    our_z_col: str = 'redshift',
    match_radius_arcsec: float = 1.0
) -> dict:
    """Validate our photometric redshifts against 3D-HST spectroscopic redshifts.

    Parameters
    ----------
    catalog : pd.DataFrame
        Catalog with our photo-z and coordinates
    our_z_col : str
        Column name for our photometric redshift
    match_radius_arcsec : float
        Match radius in arcseconds

    Returns
    -------
    dict
        Photo-z validation metrics:
        - nmad: Normalized median absolute deviation
        - bias: Median (z_phot - z_spec) / (1 + z_spec)
        - outlier_rate: Fraction with |dz/(1+z)| > 0.15
        - catastrophic_rate: Fraction with |dz/(1+z)| > 0.5
    """
    # Cross-match if not already done
    if 'z_spec_3dhst' not in catalog.columns:
        catalog = crossmatch_with_3dhst(catalog, match_radius_arcsec)

    # Filter to sources with spec-z
    has_specz = catalog['z_spec_3dhst'].notna() & (catalog['z_spec_3dhst'] > 0)
    matched = catalog[has_specz].copy()

    if len(matched) < 5:
        print(f"Too few sources with spec-z ({len(matched)}). Skipping validation.")
        return {}

    z_phot = matched[our_z_col].values
    z_spec = matched['z_spec_3dhst'].values

    # Calculate residuals
    dz = (z_phot - z_spec) / (1 + z_spec)

    # Metrics
    nmad = 1.48 * np.median(np.abs(dz - np.median(dz)))
    bias = np.median(dz)
    outlier_rate = np.mean(np.abs(dz) > 0.15)
    catastrophic_rate = np.mean(np.abs(dz) > 0.5)

    results = {
        'n_specz': len(matched),
        'nmad': nmad,
        'bias': bias,
        'sigma_68': np.percentile(np.abs(dz), 68),
        'outlier_rate': outlier_rate,
        'catastrophic_rate': catastrophic_rate
    }

    print("\n=== Photo-z Validation (vs 3D-HST spec-z) ===")
    print(f"  Sources with spec-z: {len(matched)}")
    print(f"  NMAD: {nmad:.4f}")
    print(f"  Bias: {bias:+.4f}")
    print(f"  σ_68: {results['sigma_68']:.4f}")
    print(f"  Outlier rate (|Δz/(1+z)| > 0.15): {100*outlier_rate:.1f}%")
    print(f"  Catastrophic rate (|Δz/(1+z)| > 0.5): {100*catastrophic_rate:.1f}%")

    # Identify low-z catastrophic outliers
    low_z_phot = z_phot < 0.2
    if low_z_phot.sum() > 0:
        low_z_catastrophic = (np.abs(dz) > 0.5) & low_z_phot
        print(f"\n  Low-z (z_phot < 0.2) sources: {low_z_phot.sum()}")
        print(f"  Low-z catastrophic outliers: {low_z_catastrophic.sum()}")
        if low_z_catastrophic.sum() > 0:
            print("    These may be misclassified high-z galaxies!")

    return results


def apply_spectroscopic_redshifts(
    catalog: pd.DataFrame,
    z_col: str = 'redshift',
    match_radius_arcsec: float = 1.0,
    flag_catastrophic: bool = True,
    catastrophic_threshold: float = 0.3
) -> pd.DataFrame:
    """Replace photometric redshifts with spectroscopic when available.

    This fixes catastrophic photo-z outliers by using trusted spec-z values
    from the 3D-HST compilation.

    Parameters
    ----------
    catalog : pd.DataFrame
        Catalog with photometric redshifts
    z_col : str
        Column name for redshift (will be updated)
    match_radius_arcsec : float
        Match radius for cross-matching
    flag_catastrophic : bool
        If True, add 'catastrophic_photoz' column for sources with |dz/(1+z)| > threshold
    catastrophic_threshold : float
        Threshold for catastrophic outlier (default 0.3 = 30% error in 1+z)

    Returns
    -------
    pd.DataFrame
        Catalog with updated redshifts and optional flags
    """
    # Cross-match if not already done
    if 'z_spec_3dhst' not in catalog.columns:
        catalog = crossmatch_with_3dhst(catalog, match_radius_arcsec)

    catalog = catalog.copy()

    # Find sources with valid spec-z
    has_specz = catalog['z_spec_3dhst'].notna() & (catalog['z_spec_3dhst'] > 0)
    n_with_specz = has_specz.sum()

    if n_with_specz == 0:
        print("  No sources with spectroscopic redshifts to apply.")
        return catalog

    # Calculate photo-z residuals for sources with spec-z
    z_phot = catalog.loc[has_specz, z_col].values
    z_spec = catalog.loc[has_specz, 'z_spec_3dhst'].values
    dz_norm = np.abs(z_phot - z_spec) / (1 + z_spec)

    # Identify catastrophic outliers
    is_catastrophic = dz_norm > catastrophic_threshold

    # Store original photo-z before replacement
    catalog['z_phot_original'] = catalog[z_col].copy()

    # Replace photo-z with spec-z for matched sources
    catalog.loc[has_specz, z_col] = catalog.loc[has_specz, 'z_spec_3dhst']

    # Track which redshifts are spectroscopic
    catalog['z_is_spectroscopic'] = False
    catalog.loc[has_specz, 'z_is_spectroscopic'] = True

    # Flag catastrophic outliers
    if flag_catastrophic:
        catalog['catastrophic_photoz'] = False
        catastrophic_indices = catalog.index[has_specz][is_catastrophic]
        catalog.loc[catastrophic_indices, 'catastrophic_photoz'] = True

    n_catastrophic = is_catastrophic.sum()
    print(f"\n  Applied {n_with_specz} spectroscopic redshifts from 3D-HST")
    print(f"  Catastrophic photo-z outliers corrected: {n_catastrophic}")

    if n_catastrophic > 0:
        # Show the corrected sources
        corrected = catalog[catalog.get('catastrophic_photoz', False)]
        print("\n  Corrected catastrophic outliers:")
        for idx, row in corrected.iterrows():
            z_old = row['z_phot_original']
            z_new = row[z_col]
            print(f"    Source {idx}: z_phot={z_old:.2f} -> z_spec={z_new:.2f}")

    return catalog


def flag_stars_from_3dhst(
    catalog: pd.DataFrame,
    match_radius_arcsec: float = 1.0
) -> pd.DataFrame:
    """Add star flags based on 3D-HST classification.

    This is the main function to call to improve star classification.
    It cross-matches with 3D-HST and flags sources they identify as stars.

    Parameters
    ----------
    catalog : pd.DataFrame
        Local catalog
    match_radius_arcsec : float
        Match radius in arcseconds

    Returns
    -------
    pd.DataFrame
        Catalog with 'star_3dhst' column added
    """
    return crossmatch_with_3dhst(catalog, match_radius_arcsec)


# Main execution for testing
if __name__ == "__main__":
    # Test with the full catalog
    catalog_path = Path(__file__).parent.parent / "output" / "full_HDF" / "full_z" / "galaxy_catalog.csv"

    if catalog_path.exists():
        print(f"Loading catalog from {catalog_path}")
        cat = pd.read_csv(catalog_path)

        # Cross-match and validate
        cat = crossmatch_with_3dhst(cat)

        if 'pro_is_star' in cat.columns:
            validate_star_classification(cat)

        if 'redshift' in cat.columns:
            validate_photoz(cat)

        # Show misclassified sources
        if 'star_3dhst' in cat.columns:
            misclassified = cat[cat['star_3dhst'] & ~cat.get('pro_is_star', pd.Series([False]*len(cat)))]
            if len(misclassified) > 0:
                print(f"\n=== Sources we classify as galaxy but 3D-HST says star ({len(misclassified)}) ===")
                cols = ['ra', 'dec', 'redshift', 'r_half_arcsec', 'stellarity', 'galaxy_type']
                cols = [c for c in cols if c in misclassified.columns]
                print(misclassified[cols].head(10))
    else:
        print(f"Test catalog not found: {catalog_path}")
        print("Run: python run_analysis.py full")
