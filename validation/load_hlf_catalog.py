"""Load and cross-match with Hubble Legacy Fields (HLF) reference catalog.

The HLF catalog provides validated photometry and photo-z for 103,098 sources
in 13 bands, making it ideal for validating our detection and classification.

Reference: Illingworth et al. 2019
URL: https://archive.stsci.edu/prepds/hlf/
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_hlf_catalog(local_path: str | Path) -> pd.DataFrame:
    """Load HLF GOODS-S photometric catalog.

    Parameters
    ----------
    local_path : str or Path
        Path to the HLF catalog FITS file

    Returns
    -------
    pd.DataFrame
        Catalog with columns: id, ra, dec, fluxes, photo_z, etc.
    """
    from astropy.io import fits

    local_path = Path(local_path)

    if not local_path.exists():
        raise FileNotFoundError(
            f"HLF catalog not found at {local_path}. "
            "Run: python -m download.fetch_all_data"
        )

    with fits.open(local_path) as hdul:
        data = hdul[1].data
        colnames = data.names

    # Build DataFrame with available columns
    catalog = pd.DataFrame({"id": data["ID"]})

    # Coordinates
    if "RA" in colnames:
        catalog["ra"] = data["RA"]
    if "DEC" in colnames:
        catalog["dec"] = data["DEC"]

    # Fluxes - try common column name patterns
    flux_mappings = {
        "f435w": ["F435W_FLUX", "F435W", "FLUX_F435W"],
        "f606w": ["F606W_FLUX", "F606W", "FLUX_F606W"],
        "f775w": ["F775W_FLUX", "F775W", "FLUX_F775W"],
        "f814w": ["F814W_FLUX", "F814W", "FLUX_F814W"],
        "f850lp": ["F850LP_FLUX", "F850LP", "FLUX_F850LP"],
    }

    for output_col, possible_cols in flux_mappings.items():
        for col in possible_cols:
            if col in colnames:
                catalog[output_col] = data[col]
                break

    # Photo-z
    photoz_cols = ["Z_BEST", "ZBEST", "Z_PHOT", "PHOTO_Z", "ZP"]
    for col in photoz_cols:
        if col in colnames:
            catalog["photo_z"] = data[col]
            break

    # Photo-z uncertainties
    for suffix in ["_LO", "_LOW", "_16"]:
        col = f"Z_BEST{suffix}" if "Z_BEST" in colnames else f"ZBEST{suffix}"
        if col in colnames:
            catalog["photo_z_lo"] = data[col]
            break

    for suffix in ["_HI", "_HIGH", "_84"]:
        col = f"Z_BEST{suffix}" if "Z_BEST" in colnames else f"ZBEST{suffix}"
        if col in colnames:
            catalog["photo_z_hi"] = data[col]
            break

    # Spec-z if available
    specz_cols = ["Z_SPEC", "ZSPEC", "SPEC_Z"]
    for col in specz_cols:
        if col in colnames:
            catalog["spec_z"] = data[col]
            break

    # Stellar mass
    mass_cols = ["MASS", "LMASS", "LOG_MASS", "STELLAR_MASS"]
    for col in mass_cols:
        if col in colnames:
            catalog["log_mass"] = data[col]
            break

    print(f"Loaded HLF catalog: {len(catalog)} sources")
    print(f"Columns available: {list(catalog.columns)}")

    return catalog


def cross_match_catalogs(
    our_catalog: pd.DataFrame,
    reference_catalog: pd.DataFrame,
    max_sep: float = 1.0,
    our_ra_col: str = "ra",
    our_dec_col: str = "dec",
    ref_ra_col: str = "ra",
    ref_dec_col: str = "dec",
) -> pd.DataFrame:
    """Cross-match our detections with a reference catalog.

    Parameters
    ----------
    our_catalog : pd.DataFrame
        Our source catalog (must have ra/dec columns)
    reference_catalog : pd.DataFrame
        Reference catalog to match against
    max_sep : float
        Maximum separation in arcseconds
    our_ra_col, our_dec_col : str
        Column names for coordinates in our catalog
    ref_ra_col, ref_dec_col : str
        Column names for coordinates in reference catalog

    Returns
    -------
    pd.DataFrame
        Matched catalog with reference columns added
    """
    from astropy.coordinates import SkyCoord, match_coordinates_sky
    import astropy.units as u

    # Build coordinate arrays
    our_coords = SkyCoord(
        ra=our_catalog[our_ra_col].values * u.degree,
        dec=our_catalog[our_dec_col].values * u.degree,
    )
    ref_coords = SkyCoord(
        ra=reference_catalog[ref_ra_col].values * u.degree,
        dec=reference_catalog[ref_dec_col].values * u.degree,
    )

    # Find nearest matches
    idx, sep2d, _ = match_coordinates_sky(our_coords, ref_coords)

    # Filter by separation
    matched = sep2d.arcsec < max_sep

    # Build result DataFrame
    result = our_catalog[matched].copy()
    result["ref_idx"] = idx[matched]
    result["sep_arcsec"] = sep2d.arcsec[matched]

    # Add reference catalog columns
    matched_ref = reference_catalog.iloc[idx[matched]].reset_index(drop=True)

    for col in matched_ref.columns:
        if col not in [ref_ra_col, ref_dec_col]:
            result[f"ref_{col}"] = matched_ref[col].values

    print(f"Cross-matched {matched.sum()} sources out of {len(our_catalog)}")
    print(f"Match rate: {100 * matched.sum() / len(our_catalog):.1f}%")
    print(f"Median separation: {np.median(sep2d.arcsec[matched]):.3f} arcsec")

    return result


def validate_photometry(
    matched_catalog: pd.DataFrame,
    our_flux_col: str,
    ref_flux_col: str,
) -> dict:
    """Compare our photometry with reference catalog.

    Parameters
    ----------
    matched_catalog : pd.DataFrame
        Cross-matched catalog
    our_flux_col : str
        Column name for our flux measurement
    ref_flux_col : str
        Column name for reference flux

    Returns
    -------
    dict
        Statistics comparing the two flux measurements
    """
    our_flux = matched_catalog[our_flux_col].values
    ref_flux = matched_catalog[ref_flux_col].values

    # Filter valid measurements
    valid = (our_flux > 0) & (ref_flux > 0) & np.isfinite(our_flux) & np.isfinite(ref_flux)

    if valid.sum() < 10:
        return {"error": "Too few valid measurements"}

    our_valid = our_flux[valid]
    ref_valid = ref_flux[valid]

    # Compute flux ratio
    ratio = our_valid / ref_valid
    log_ratio = np.log10(ratio)

    return {
        "n_valid": valid.sum(),
        "median_ratio": np.median(ratio),
        "std_ratio": np.std(ratio),
        "median_log_ratio": np.median(log_ratio),
        "nmad_log_ratio": 1.48 * np.median(np.abs(log_ratio - np.median(log_ratio))),
        "offset_mag": -2.5 * np.median(log_ratio),  # Magnitude offset
    }


def summarize_hlf_validation(
    our_catalog: pd.DataFrame,
    hlf_catalog: pd.DataFrame,
    max_sep: float = 1.0,
) -> dict:
    """Run full HLF validation and return summary.

    Parameters
    ----------
    our_catalog : pd.DataFrame
        Our source catalog with ra, dec, flux columns
    hlf_catalog : pd.DataFrame
        HLF reference catalog
    max_sep : float
        Maximum match separation in arcseconds

    Returns
    -------
    dict
        Validation summary statistics
    """
    # Cross-match
    matched = cross_match_catalogs(our_catalog, hlf_catalog, max_sep=max_sep)

    summary = {
        "n_our_sources": len(our_catalog),
        "n_hlf_sources": len(hlf_catalog),
        "n_matched": len(matched),
        "match_fraction": len(matched) / len(our_catalog),
        "median_separation_arcsec": np.median(matched["sep_arcsec"]),
    }

    # Photo-z comparison if available
    if "redshift" in matched.columns and "ref_photo_z" in matched.columns:
        z_our = matched["redshift"].values
        z_ref = matched["ref_photo_z"].values

        valid = (z_our > 0) & (z_ref > 0) & np.isfinite(z_our) & np.isfinite(z_ref)

        if valid.sum() > 10:
            dz = (z_our[valid] - z_ref[valid]) / (1 + z_ref[valid])
            summary["photoz_comparison"] = {
                "n_valid": valid.sum(),
                "median_dz": np.median(dz),
                "nmad": 1.48 * np.median(np.abs(dz)),
                "outlier_frac": np.mean(np.abs(dz) > 0.15),
            }

    return summary
