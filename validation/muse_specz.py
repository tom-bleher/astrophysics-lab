"""Load and validate against MUSE HUDF spectroscopic redshifts.

The MUSE (Multi Unit Spectroscopic Explorer) survey provides high-quality
spectroscopic redshifts for 1,338 sources in the HUDF, making it ideal
for validating photometric redshifts.

Reference: Bacon et al. 2017, A&A, 608, A1
VizieR: J/A+A/608/A1
"""

import numpy as np
import pandas as pd


def load_muse_catalog(
    local_path: str | None = None,
    confidence_min: int = 2,
) -> pd.DataFrame:
    """Load MUSE HUDF spectroscopic redshift catalog.

    Parameters
    ----------
    local_path : str, optional
        Path to local FITS file. If None, queries VizieR.
    confidence_min : int
        Minimum confidence level (1-3, where 3 is most secure)

    Returns
    -------
    pd.DataFrame
        Catalog with columns: ra, dec, z_spec, confidence
    """
    df = _load_muse_local(local_path) if local_path is not None else _query_muse_vizier()

    if len(df) == 0:
        return df

    # Filter by confidence
    if "confidence" in df.columns:
        mask = df["confidence"] >= confidence_min
        df = df[mask].copy()
        print(f"  After confidence filter (>={confidence_min}): {len(df)} sources")

    return df


def _load_muse_local(local_path: str) -> pd.DataFrame:
    """Load MUSE catalog from local FITS file."""
    from pathlib import Path

    from astropy.io import fits

    path = Path(local_path)

    if not path.exists():
        print(f"MUSE catalog not found at {path}")
        return pd.DataFrame()

    try:
        with fits.open(path) as hdul:
            data = hdul[1].data
            colnames = data.names

        # Build DataFrame
        df = pd.DataFrame()

        # Coordinates
        for ra_col in ["RAJ2000", "RA", "ra"]:
            if ra_col in colnames:
                df["ra"] = data[ra_col]
                break

        for dec_col in ["DEJ2000", "DEC", "Dec", "dec"]:
            if dec_col in colnames:
                df["dec"] = data[dec_col]
                break

        # Redshift
        for z_col in ["zMUSE", "z", "Z", "REDSHIFT", "z_spec"]:
            if z_col in colnames:
                df["z_spec"] = data[z_col]
                break

        # Confidence
        for conf_col in ["Conf", "CONF", "confidence", "Quality"]:
            if conf_col in colnames:
                df["confidence"] = data[conf_col]
                break

        print(f"Loaded MUSE catalog: {len(df)} sources")
        return df

    except Exception as e:
        print(f"Error loading MUSE catalog: {e}")
        return pd.DataFrame()


def _query_muse_vizier() -> pd.DataFrame:
    """Query MUSE catalog from VizieR."""
    try:
        from astroquery.vizier import Vizier
    except ImportError:
        print("astroquery not installed")
        return pd.DataFrame()

    print("Querying MUSE HUDF catalog from VizieR (J/A+A/608/A1)...")

    try:
        Vizier.ROW_LIMIT = -1
        result = Vizier.get_catalogs("J/A+A/608/A1")

        if result and len(result) > 0:
            df = result[0].to_pandas()

            # Standardize column names
            col_mapping = {
                "RAJ2000": "ra",
                "DEJ2000": "dec",
                "zMUSE": "z_spec",
                "Conf": "confidence",
            }

            for old_col, new_col in col_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})

            print(f"  Loaded {len(df)} sources")
            return df

    except Exception as e:
        print(f"  VizieR query failed: {e}")

    return pd.DataFrame()


def validate_with_specz(
    our_catalog: pd.DataFrame,
    muse_catalog: pd.DataFrame,
    match_radius: float = 1.0,
    our_z_col: str = "redshift",
    our_ra_col: str = "ra",
    our_dec_col: str = "dec",
) -> dict:
    """Validate our photo-z against MUSE spectroscopic redshifts.

    Parameters
    ----------
    our_catalog : pd.DataFrame
        Our catalog with photo-z estimates
    muse_catalog : pd.DataFrame
        MUSE catalog with spec-z
    match_radius : float
        Maximum separation in arcseconds
    our_z_col : str
        Column name for our redshift estimates
    our_ra_col, our_dec_col : str
        Column names for coordinates

    Returns
    -------
    dict
        Validation results including:
        - matched_catalog: Cross-matched sources
        - n_matched: Number of matches
        - metrics: Photo-z quality metrics
        - outliers: Sources with |Δz/(1+z)| > 0.15
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord, match_coordinates_sky

    # Check required columns
    if our_z_col not in our_catalog.columns:
        return {"error": f"Column '{our_z_col}' not found in our catalog"}

    if "z_spec" not in muse_catalog.columns:
        return {"error": "Column 'z_spec' not found in MUSE catalog"}

    # Build coordinates
    our_coords = SkyCoord(
        ra=our_catalog[our_ra_col].values * u.deg,
        dec=our_catalog[our_dec_col].values * u.deg,
    )
    muse_coords = SkyCoord(
        ra=muse_catalog["ra"].values * u.deg,
        dec=muse_catalog["dec"].values * u.deg,
    )

    # Cross-match
    idx, sep, _ = match_coordinates_sky(our_coords, muse_coords)
    matched = sep.arcsec < match_radius

    n_matched = matched.sum()
    print(f"Matched {n_matched} sources within {match_radius} arcsec")

    if n_matched < 5:
        return {
            "n_matched": n_matched,
            "error": "Too few matches for statistics",
        }

    # Extract matched data
    z_phot = our_catalog.loc[matched, our_z_col].values
    z_spec = muse_catalog.iloc[idx[matched]]["z_spec"].values

    # Filter valid redshifts
    valid = (z_phot > 0) & (z_spec > 0) & np.isfinite(z_phot) & np.isfinite(z_spec)
    z_phot = z_phot[valid]
    z_spec = z_spec[valid]

    if len(z_phot) < 5:
        return {
            "n_matched": n_matched,
            "n_valid": len(z_phot),
            "error": "Too few valid redshifts",
        }

    # Compute metrics
    dz = (z_phot - z_spec) / (1 + z_spec)
    nmad = 1.48 * np.median(np.abs(dz - np.median(dz)))
    bias = np.median(dz)
    outlier_mask = np.abs(dz) > 0.15
    outlier_frac = np.mean(outlier_mask)
    catastrophic_mask = np.abs(dz) > 0.5
    catastrophic_frac = np.mean(catastrophic_mask)

    # Build matched catalog
    matched_catalog = our_catalog[matched].copy()
    matched_catalog["z_spec"] = muse_catalog.iloc[idx[matched]]["z_spec"].values
    matched_catalog["sep_arcsec"] = sep.arcsec[matched]
    if "confidence" in muse_catalog.columns:
        matched_catalog["spec_confidence"] = muse_catalog.iloc[idx[matched]][
            "confidence"
        ].values

    # Identify outliers
    outlier_indices = np.where(outlier_mask)[0]

    return {
        "n_matched": n_matched,
        "n_valid": len(z_phot),
        "matched_catalog": matched_catalog,
        "z_phot": z_phot,
        "z_spec": z_spec,
        "dz_norm": dz,
        "metrics": {
            "nmad": nmad,
            "bias": bias,
            "sigma": np.std(dz),
            "outlier_frac": outlier_frac,
            "catastrophic_frac": catastrophic_frac,
        },
        "outlier_indices": outlier_indices,
    }


def print_validation_summary(validation_result: dict) -> None:
    """Print a summary of the validation results.

    Parameters
    ----------
    validation_result : dict
        Output from validate_with_specz()
    """
    if "error" in validation_result:
        print(f"Validation error: {validation_result['error']}")
        return

    print("\n" + "=" * 50)
    print("Photo-z Validation Summary (vs MUSE spec-z)")
    print("=" * 50)

    print(f"\nMatched sources: {validation_result['n_matched']}")
    print(f"Valid redshifts: {validation_result['n_valid']}")

    metrics = validation_result["metrics"]
    print("\nMetrics:")
    print(f"  NMAD (sigma_NMAD): {metrics['nmad']:.4f}")
    print(f"  Bias (median Δz):  {metrics['bias']:.4f}")
    print(f"  Scatter (sigma):   {metrics['sigma']:.4f}")
    print(f"  Outlier fraction:  {metrics['outlier_frac']:.1%} (|Δz/(1+z)| > 0.15)")
    print(f"  Catastrophic:      {metrics['catastrophic_frac']:.1%} (|Δz/(1+z)| > 0.5)")

    # Quality assessment
    print("\nQuality Assessment:")
    if metrics["nmad"] < 0.02:
        print("  ★★★ Excellent photo-z accuracy (NMAD < 0.02)")
    elif metrics["nmad"] < 0.05:
        print("  ★★☆ Good photo-z accuracy (NMAD < 0.05)")
    elif metrics["nmad"] < 0.10:
        print("  ★☆☆ Moderate photo-z accuracy (NMAD < 0.10)")
    else:
        print("  ☆☆☆ Poor photo-z accuracy (NMAD >= 0.10)")

    if metrics["outlier_frac"] < 0.05:
        print("  ★★★ Low outlier rate (<5%)")
    elif metrics["outlier_frac"] < 0.15:
        print("  ★★☆ Moderate outlier rate (<15%)")
    else:
        print("  ★☆☆ High outlier rate (>=15%)")
