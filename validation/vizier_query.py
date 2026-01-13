"""Query VizieR for HUDF and HDF catalog data.

VizieR provides access to published astronomical catalogs including:
- HUDF photometry (II/258)
- Fernández-Soto 1999 HDF photo-z (J/ApJ/513/34)
- Surface photometry (J/A+AS/129/583)
- MUSE spectroscopic redshifts (J/A+A/608/A1)

Reference: https://vizier.cds.unistra.fr/
"""

import pandas as pd


def query_vizier_catalog(
    catalog_id: str,
    center_ra: float | None = None,
    center_dec: float | None = None,
    radius_arcmin: float = 3.0,
    row_limit: int = -1,
) -> pd.DataFrame:
    """Query a VizieR catalog, optionally around a position.

    Parameters
    ----------
    catalog_id : str
        VizieR catalog identifier (e.g., "J/ApJ/513/34")
    center_ra : float, optional
        Center RA in degrees for cone search
    center_dec : float, optional
        Center Dec in degrees for cone search
    radius_arcmin : float
        Search radius in arcminutes
    row_limit : int
        Maximum rows to return (-1 for all)

    Returns
    -------
    pd.DataFrame
        Catalog data
    """
    try:
        from astroquery.vizier import Vizier
    except ImportError:
        print("astroquery not installed. Run: pip install astroquery")
        return pd.DataFrame()

    Vizier.ROW_LIMIT = row_limit

    try:
        if center_ra is not None and center_dec is not None:
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            coord = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg)
            result = Vizier.query_region(
                coord,
                radius=radius_arcmin * u.arcmin,
                catalog=catalog_id,
            )
        else:
            result = Vizier.get_catalogs(catalog_id)

        if result and len(result) > 0:
            return result[0].to_pandas()

        return pd.DataFrame()

    except Exception as e:
        print(f"VizieR query failed: {e}")
        return pd.DataFrame()


def query_hudf_vizier(
    center_ra: float = 53.1625,  # HUDF center
    center_dec: float = -27.7914,
    radius_arcmin: float = 3.0,
) -> pd.DataFrame:
    """Query VizieR HUDF catalog (II/258) around a position.

    Parameters
    ----------
    center_ra : float
        Right ascension in degrees (default: HUDF center)
    center_dec : float
        Declination in degrees
    radius_arcmin : float
        Search radius in arcminutes

    Returns
    -------
    pd.DataFrame
        HUDF catalog data
    """
    print(f"Querying HUDF catalog around RA={center_ra:.4f}, Dec={center_dec:.4f}")

    df = query_vizier_catalog(
        "II/258",
        center_ra=center_ra,
        center_dec=center_dec,
        radius_arcmin=radius_arcmin,
    )

    if len(df) > 0:
        print(f"  Found {len(df)} sources")
    else:
        print("  No sources found")

    return df


def query_fernandez_soto(row_limit: int = -1) -> pd.DataFrame:
    """Query Fernández-Soto et al. 1999 HDF-N photo-z catalog.

    This is the primary reference catalog for HDF-N photometric redshifts.
    Contains 1,067 galaxies with 7-band photometry (UBVIJHK).

    Reference: ApJ 513, 34 (1999)

    Parameters
    ----------
    row_limit : int
        Maximum rows to return (-1 for all)

    Returns
    -------
    pd.DataFrame
        Catalog with id, ra, dec, magnitudes, photo-z
    """
    print("Querying Fernández-Soto 1999 catalog (J/ApJ/513/34)...")

    df = query_vizier_catalog("J/ApJ/513/34", row_limit=row_limit)

    if len(df) == 0:
        print("  VizieR query failed, trying direct download...")
        df = load_fernandez_soto_local()

    if len(df) > 0:
        print(f"  Loaded {len(df)} sources")

        # Standardize column names
        col_mapping = {
            "RAJ2000": "ra",
            "DEJ2000": "dec",
            "_RA": "ra",
            "_DE": "dec",
            "zphot": "photo_z",
            "zspec": "spec_z",
            "Umag": "U_mag",
            "Bmag": "B_mag",
            "Vmag": "V_mag",
            "Imag": "I_mag",
        }

        for old_col, new_col in col_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

    return df


def load_fernandez_soto_local(
    local_path: str = "./data/external/fernandez_soto_1999.dat",
) -> pd.DataFrame:
    """Load Fernández-Soto catalog from local file.

    Parameters
    ----------
    local_path : str
        Path to downloaded catalog file

    Returns
    -------
    pd.DataFrame
        Catalog data
    """
    from pathlib import Path
    import numpy as np

    path = Path(local_path)

    if not path.exists():
        print(f"  Local file not found: {path}")
        return pd.DataFrame()

    try:
        # Read the fixed-width format file
        # Format from ReadMe: columns are fixed-width
        data = np.genfromtxt(
            path,
            dtype=None,
            encoding="utf-8",
            names=True,
            skip_header=0,
        )

        return pd.DataFrame(data)

    except Exception as e:
        print(f"  Error loading local file: {e}")
        return pd.DataFrame()


def query_hdf_surface_photometry() -> pd.DataFrame:
    """Query HDF surface photometry catalog (J/A+AS/129/583).

    This catalog provides surface photometry parameters useful for
    validating size measurements.

    Reference: Fasano et al. 1998, A&AS 129, 583

    Returns
    -------
    pd.DataFrame
        Surface photometry catalog
    """
    print("Querying HDF surface photometry catalog...")

    df = query_vizier_catalog("J/A+AS/129/583")

    if len(df) > 0:
        print(f"  Found {len(df)} sources")

    return df


def query_multiple_catalogs(
    center_ra: float,
    center_dec: float,
    radius_arcmin: float = 3.0,
) -> dict[str, pd.DataFrame]:
    """Query multiple relevant VizieR catalogs around a position.

    Parameters
    ----------
    center_ra : float
        Right ascension in degrees
    center_dec : float
        Declination in degrees
    radius_arcmin : float
        Search radius in arcminutes

    Returns
    -------
    dict
        Mapping of catalog name to DataFrame
    """
    catalogs = {
        "hudf": "II/258",
        "fernandez_soto": "J/ApJ/513/34",
        "surface_photometry": "J/A+AS/129/583",
    }

    results = {}

    for name, cat_id in catalogs.items():
        print(f"\nQuerying {name}...")
        df = query_vizier_catalog(
            cat_id,
            center_ra=center_ra,
            center_dec=center_dec,
            radius_arcmin=radius_arcmin,
        )
        results[name] = df
        print(f"  {len(df)} sources")

    return results
