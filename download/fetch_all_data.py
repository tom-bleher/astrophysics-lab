"""Download all external data for the HDF analysis project.

This script downloads:
- HLF photometric catalog (103,098 sources)
- EAZY templates and example catalogs
- Fernández-Soto 1999 catalog
- MUSE spectroscopic redshifts

Usage:
    python -m download.fetch_all_data
"""

import urllib.request
import urllib.error
from pathlib import Path
import sys

# Data sources with metadata
DATA_SOURCES: dict[str, dict] = {
    "hlf_catalog": {
        "url": "https://archive.stsci.edu/hlsps/hlf/hlsp_hlf_hst_goodss_v2.1_catalog.fits",
        "local": "./data/external/hlf_catalog.fits",
        "description": "Hubble Legacy Fields GOODS-S catalog (103,098 sources)",
        "size_mb": 45,
    },
    "fernandez_soto_vizier": {
        "url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/513/34/table1.dat",
        "local": "./data/external/fernandez_soto_1999.dat",
        "description": "Fernández-Soto et al. 1999 HDF-N photo-z catalog",
        "size_mb": 0.1,
    },
    "fernandez_soto_readme": {
        "url": "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/513/34/ReadMe",
        "local": "./data/external/fernandez_soto_1999_readme.txt",
        "description": "Fernández-Soto catalog format description",
        "size_mb": 0.01,
    },
}

# VizieR catalog IDs for astroquery access
VIZIER_CATALOGS = {
    "fernandez_soto": "J/ApJ/513/34",
    "hudf_photometry": "II/258",
    "muse_hudf": "J/A+A/608/A1",
    "hdf_surface_photometry": "J/A+AS/129/583",
}


def download_file(
    url: str,
    local_path: str | Path,
    description: str = "",
    timeout: int = 60,
) -> bool:
    """Download a file with progress indication.

    Parameters
    ----------
    url : str
        URL to download from
    local_path : str or Path
        Local path to save the file
    description : str
        Human-readable description of the file
    timeout : int
        Timeout in seconds

    Returns
    -------
    bool
        True if download succeeded, False otherwise
    """
    local_path = Path(local_path)

    if local_path.exists():
        print(f"  [SKIP] Already exists: {local_path}")
        return True

    # Create parent directories
    local_path.parent.mkdir(parents=True, exist_ok=True)

    desc = description or url
    print(f"  Downloading: {desc}")
    print(f"    URL: {url}")
    print(f"    -> {local_path}")

    try:
        # Download with progress callback
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 / total_size)
                sys.stdout.write(f"\r    Progress: {percent:.1f}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, local_path, reporthook=progress_hook)
        print()  # Newline after progress
        print(f"    [OK] Saved to: {local_path}")
        return True

    except urllib.error.HTTPError as e:
        print(f"\n    [ERROR] HTTP {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"\n    [ERROR] URL Error: {e.reason}")
        return False
    except TimeoutError:
        print(f"\n    [ERROR] Timeout after {timeout}s")
        return False
    except Exception as e:
        print(f"\n    [ERROR] {type(e).__name__}: {e}")
        return False


def download_vizier_catalog(
    catalog_id: str,
    output_path: str | Path,
    row_limit: int = -1,
) -> bool:
    """Download a catalog from VizieR using astroquery.

    Parameters
    ----------
    catalog_id : str
        VizieR catalog identifier (e.g., "J/ApJ/513/34")
    output_path : str or Path
        Path to save the catalog
    row_limit : int
        Maximum number of rows (-1 for all)

    Returns
    -------
    bool
        True if download succeeded
    """
    output_path = Path(output_path)

    if output_path.exists():
        print(f"  [SKIP] Already exists: {output_path}")
        return True

    try:
        from astroquery.vizier import Vizier

        print(f"  Querying VizieR catalog: {catalog_id}")

        Vizier.ROW_LIMIT = row_limit
        result = Vizier.get_catalogs(catalog_id)

        if not result:
            print(f"    [WARN] No data returned for {catalog_id}")
            return False

        # Save the first table
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as FITS for better compatibility
        table = result[0]
        fits_path = output_path.with_suffix(".fits")
        table.write(fits_path, format="fits", overwrite=True)
        print(f"    [OK] Saved {len(table)} rows to: {fits_path}")

        return True

    except ImportError:
        print("    [ERROR] astroquery not installed. Run: pip install astroquery")
        return False
    except Exception as e:
        print(f"    [ERROR] {type(e).__name__}: {e}")
        return False


def download_all(include_vizier: bool = True) -> dict[str, bool]:
    """Download all external data sources.

    Parameters
    ----------
    include_vizier : bool
        Whether to include VizieR catalog downloads (requires astroquery)

    Returns
    -------
    dict
        Mapping of data source name to download success status
    """
    print("=" * 60)
    print("HDF Analysis: External Data Download")
    print("=" * 60)

    results = {}

    # Download direct URL sources
    print("\n[1/2] Downloading from direct URLs...")
    print("-" * 40)

    for name, info in DATA_SOURCES.items():
        success = download_file(
            info["url"],
            info["local"],
            info["description"],
        )
        results[name] = success

    # Download VizieR catalogs
    if include_vizier:
        print("\n[2/2] Downloading from VizieR...")
        print("-" * 40)

        vizier_dir = Path("./data/external/vizier")

        for name, catalog_id in VIZIER_CATALOGS.items():
            output_path = vizier_dir / f"{name}.fits"
            success = download_vizier_catalog(catalog_id, output_path)
            results[f"vizier_{name}"] = success

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)

    success_count = sum(results.values())
    total_count = len(results)

    for name, success in results.items():
        status = "[OK]" if success else "[FAILED]"
        print(f"  {status} {name}")

    print(f"\nTotal: {success_count}/{total_count} successful")

    if success_count < total_count:
        print("\nNote: Some downloads failed. This may be due to:")
        print("  - Network issues")
        print("  - Missing dependencies (astroquery)")
        print("  - Temporary server unavailability")
        print("You can re-run this script to retry failed downloads.")

    print("\nNext steps:")
    print("  1. Run validation scripts to compare with external catalogs")
    print("  2. Use benchmark/eazy_comparison.py to compare photo-z methods")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download external HDF data")
    parser.add_argument(
        "--no-vizier",
        action="store_true",
        help="Skip VizieR downloads (useful if astroquery is not installed)",
    )
    args = parser.parse_args()

    download_all(include_vizier=not args.no_vizier)
