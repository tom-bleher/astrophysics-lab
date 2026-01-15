"""Query MAST (Mikulski Archive for Space Telescopes) for HDF data.

MAST is the primary archive for HST observations including:
- Original HDF-N and HDF-S images
- Follow-up observations
- Processed data products

This module provides utilities for querying and downloading
HDF-related data from MAST.

Usage:
    python -m download.query_mast
"""

from pathlib import Path

import pandas as pd


def search_hdf_observations(
    target: str = "HDF",
    filters: list[str] | None = None,
    instrument: str = "WFPC2",
) -> "pd.DataFrame":
    """Search MAST for Hubble Deep Field observations.

    Parameters
    ----------
    target : str
        Target name pattern (e.g., "HDF", "GOODS")
    filters : list of str, optional
        Filter names to search for (e.g., ['F300W', 'F450W'])
    instrument : str
        Instrument name (e.g., "WFPC2", "ACS/WFC")

    Returns
    -------
    pd.DataFrame
        Table of matching observations
    """
    import pandas as pd

    try:
        from astroquery.mast import Observations
    except ImportError:
        print("astroquery not installed. Run: pip install astroquery")
        return pd.DataFrame()

    if filters is None:
        filters = ["F300W", "F450W", "F606W", "F814W"]

    print(f"Searching MAST for {target} observations...")
    print(f"  Instrument: {instrument}")
    print(f"  Filters: {filters}")

    try:
        obs = Observations.query_criteria(
            obs_collection="HST",
            target_name=f"{target}*",
            filters=filters,
            instrument_name=instrument,
            dataproduct_type="image",
        )

        if obs is None or len(obs) == 0:
            print("  No observations found")
            return pd.DataFrame()

        print(f"  Found {len(obs)} observations")
        return obs.to_pandas()

    except Exception as e:
        print(f"  Error querying MAST: {e}")
        return pd.DataFrame()


def search_hdf_region(
    ra: float = 189.2286,  # HDF-N center
    dec: float = 62.2161,
    radius_arcmin: float = 3.0,
) -> "pd.DataFrame":
    """Search MAST for observations in a region around HDF coordinates.

    Parameters
    ----------
    ra : float
        Right ascension in degrees (default: HDF-N center)
    dec : float
        Declination in degrees
    radius_arcmin : float
        Search radius in arcminutes

    Returns
    -------
    pd.DataFrame
        Table of matching observations
    """
    import pandas as pd

    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astroquery.mast import Observations
    except ImportError:
        print("astroquery/astropy not installed")
        return pd.DataFrame()

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    print(f"Searching MAST around RA={ra:.4f}, Dec={dec:.4f}")
    print(f"  Radius: {radius_arcmin} arcmin")

    try:
        obs = Observations.query_region(
            coord,
            radius=radius_arcmin * u.arcmin,
        )

        if obs is None or len(obs) == 0:
            print("  No observations found")
            return pd.DataFrame()

        # Filter to HST only
        hst_mask = obs["obs_collection"] == "HST"
        hst_obs = obs[hst_mask]

        print(f"  Found {len(hst_obs)} HST observations")
        return hst_obs.to_pandas()

    except Exception as e:
        print(f"  Error querying MAST: {e}")
        return pd.DataFrame()


def download_hdf_products(
    obs_table,
    output_dir: str | Path = "./data/hst/",
    product_type: str = "SCIENCE",
    extension: str = "fits",
    max_products: int = 10,
) -> list[Path]:
    """Download data products for selected observations.

    Parameters
    ----------
    obs_table : astropy Table or pandas DataFrame
        Table of observations from search functions
    output_dir : str or Path
        Directory to save downloaded files
    product_type : str
        Type of products to download ('SCIENCE', 'CALIBRATED', etc.)
    extension : str
        File extension filter
    max_products : int
        Maximum number of products to download

    Returns
    -------
    list of Path
        Paths to downloaded files
    """
    try:
        from astroquery.mast import Observations
    except ImportError:
        print("astroquery not installed")
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if len(obs_table) == 0:
        print("No observations to download")
        return []

    print(f"Getting product list for {len(obs_table)} observations...")

    try:
        # Get product list
        products = Observations.get_product_list(obs_table)

        if products is None or len(products) == 0:
            print("  No products found")
            return []

        # Filter products
        filtered = Observations.filter_products(
            products,
            productType=product_type,
            extension=extension,
        )

        if filtered is None or len(filtered) == 0:
            print(f"  No {product_type} products with .{extension} extension")
            return []

        # Limit number of downloads
        if len(filtered) > max_products:
            print(f"  Limiting to first {max_products} products")
            filtered = filtered[:max_products]

        print(f"  Downloading {len(filtered)} products...")

        # Download
        manifest = Observations.download_products(
            filtered,
            download_dir=str(output_path),
        )

        downloaded = []
        if manifest is not None:
            for row in manifest:
                if row["Status"] == "COMPLETE":
                    downloaded.append(Path(row["Local Path"]))

        print(f"  Downloaded {len(downloaded)} files")
        return downloaded

    except Exception as e:
        print(f"  Error downloading products: {e}")
        return []


def get_hdf_coordinates() -> dict:
    """Get coordinates for HDF-N and HDF-S fields.

    Returns
    -------
    dict
        Dictionary with field coordinates
    """
    return {
        "HDF-N": {
            "ra": 189.2286,  # 12h 36m 49.4s
            "dec": 62.2161,  # +62° 12' 58"
            "description": "Hubble Deep Field North",
        },
        "HDF-S": {
            "ra": 338.2342,  # 22h 32m 56.2s
            "dec": -60.5508,  # -60° 33' 02.7"
            "description": "Hubble Deep Field South",
        },
        "GOODS-N": {
            "ra": 189.2286,
            "dec": 62.2161,
            "description": "Great Observatories Origins Deep Survey North",
        },
        "GOODS-S": {
            "ra": 53.1225,
            "dec": -27.8053,
            "description": "Great Observatories Origins Deep Survey South",
        },
    }


def print_hdf_info():
    """Print information about available HDF data."""
    print("=" * 60)
    print("Hubble Deep Field Data Information")
    print("=" * 60)

    coords = get_hdf_coordinates()

    for field, info in coords.items():
        print(f"\n{field}: {info['description']}")
        print(f"  RA:  {info['ra']:.4f}° ({info['ra']/15:.4f}h)")
        print(f"  Dec: {info['dec']:.4f}°")

    print("\n" + "-" * 60)
    print("Original HDF-N Observations (Williams et al. 1996):")
    print("  F300W (U): 46,900s exposure, limiting mag ~27.0 AB")
    print("  F450W (B): 93,700s exposure, limiting mag ~28.4 AB")
    print("  F606W (V): 93,600s exposure, limiting mag ~28.8 AB")
    print("  F814W (I): 93,400s exposure, limiting mag ~28.1 AB")

    print("\n" + "-" * 60)
    print("To search for HDF data:")
    print("  from download.query_mast import search_hdf_observations")
    print("  obs = search_hdf_observations(target='HDF')")


if __name__ == "__main__":
    print_hdf_info()

    print("\n" + "=" * 60)
    print("Searching MAST for HDF-N observations...")
    print("=" * 60)

    obs = search_hdf_observations(target="HDF", instrument="WFPC2")

    if len(obs) > 0:
        print("\nFirst 5 observations:")
        cols = ["target_name", "filters", "t_exptime", "instrument_name"]
        available_cols = [c for c in cols if c in obs.columns]
        print(obs[available_cols].head())
