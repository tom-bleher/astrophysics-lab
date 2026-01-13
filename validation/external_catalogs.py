"""External catalog access and validation for HDF/HUDF analysis.

This module provides functions to:
1. Query/download external reference catalogs (VizieR, HLF, Fernández-Soto)
2. Cross-match your detections against reference catalogs
3. Compute validation metrics (photo-z accuracy, completeness, etc.)

References:
- Fernández-Soto et al. 1999, ApJ, 513, 34 (HDF-N photo-z catalog)
- VizieR II/258: HUDF catalog (Beckwith et al. 2006)
- Hubble Legacy Fields: Illingworth et al. 2016
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import urllib.request

import numpy as np
import pandas as pd

# Optional imports - gracefully handle if not installed
try:
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord, match_coordinates_sky
    import astropy.units as u
    HAS_ASTROQUERY = True
except ImportError:
    HAS_ASTROQUERY = False
    warnings.warn("astroquery not installed. Install with: pip install astroquery")

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


# Default data directory
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "external"


@dataclass
class ValidationReport:
    """Results from photo-z validation against reference catalog."""
    n_matched: int
    n_total_ours: int
    n_total_reference: int

    # Photo-z metrics
    nmad: float  # Normalized median absolute deviation
    bias: float  # Median offset
    outlier_fraction: float  # |Δz/(1+z)| > 0.15
    catastrophic_fraction: float  # |Δz/(1+z)| > 0.5

    # Arrays for plotting
    z_ours: np.ndarray
    z_reference: np.ndarray
    separation_arcsec: np.ndarray

    def __str__(self) -> str:
        return f"""
=== Photo-z Validation Report ===
Matched sources: {self.n_matched} / {self.n_total_ours} ({100*self.n_matched/self.n_total_ours:.1f}%)
Reference catalog: {self.n_total_reference} sources

Photo-z accuracy:
  NMAD (σ):     {self.nmad:.4f}
  Bias:         {self.bias:+.4f}
  Outliers:     {100*self.outlier_fraction:.1f}% (|Δz/(1+z)| > 0.15)
  Catastrophic: {100*self.catastrophic_fraction:.1f}% (|Δz/(1+z)| > 0.5)
"""


# =============================================================================
# VizieR HUDF Catalog (II/258)
# =============================================================================

def load_vizier_hudf(
    center_ra: float = 53.1625,  # HUDF center
    center_dec: float = -27.7914,
    radius_arcmin: float = 3.0,
    cache_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Query VizieR for HUDF catalog (II/258) around a position.

    Parameters
    ----------
    center_ra : float
        Right ascension of field center in degrees
    center_dec : float
        Declination of field center in degrees
    radius_arcmin : float
        Search radius in arcminutes
    cache_path : Path, optional
        If provided, cache results to this file

    Returns
    -------
    pd.DataFrame
        Catalog with columns: id, ra, dec, mag_*, flux_*, etc.
    """
    if not HAS_ASTROQUERY:
        raise ImportError("astroquery required. Install with: pip install astroquery")

    # Check cache first
    if cache_path and cache_path.exists():
        print(f"Loading cached VizieR HUDF catalog from {cache_path}")
        return pd.read_csv(cache_path)

    print(f"Querying VizieR II/258 (HUDF) at RA={center_ra:.4f}, Dec={center_dec:.4f}...")

    Vizier.ROW_LIMIT = -1  # No row limit

    coord = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg)

    result = Vizier.query_region(
        coord,
        radius=radius_arcmin * u.arcmin,
        catalog="II/258"
    )

    if not result or len(result) == 0:
        print("No results from VizieR. Check coordinates or try larger radius.")
        return pd.DataFrame()

    # Convert to pandas
    df = result[0].to_pandas()

    # Standardize column names
    df = df.rename(columns={
        '_RAJ2000': 'ra',
        '_DEJ2000': 'dec',
        'UDF': 'id',
    })

    print(f"Retrieved {len(df)} sources from VizieR HUDF catalog")

    # Cache if requested
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"Cached to {cache_path}")

    return df


# =============================================================================
# Fernández-Soto et al. 1999 (HDF-N)
# =============================================================================

FERNANDEZ_SOTO_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/513/34/table4.dat"
FERNANDEZ_SOTO_README = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/513/34/ReadMe"


def load_fernandez_soto(
    data_dir: Optional[Path] = None,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Load Fernández-Soto et al. 1999 HDF-N photometric redshift catalog.

    This catalog contains 1,067 galaxies with photo-z from the original
    Hubble Deep Field North observations.

    Parameters
    ----------
    data_dir : Path, optional
        Directory to store downloaded data
    force_download : bool
        Re-download even if cached file exists

    Returns
    -------
    pd.DataFrame
        Catalog with columns: id, ra, dec, z_phot, mag_*, etc.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    local_path = data_dir / "fernandez_soto_1999.dat"

    # Download if needed
    if not local_path.exists() or force_download:
        print(f"Downloading Fernández-Soto 1999 catalog from CDS...")
        try:
            urllib.request.urlretrieve(FERNANDEZ_SOTO_URL, local_path)
            print(f"Saved to {local_path}")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Trying alternative parsing...")
            return _load_fernandez_soto_fallback(data_dir)

    # Parse the fixed-width format catalog
    # Format from ReadMe: http://cdsarc.u-strasbg.fr/ftp/J/ApJ/513/34/ReadMe
    try:
        colspecs = [
            (0, 4),    # ID
            (5, 7),    # RAh
            (8, 10),   # RAm
            (11, 16),  # RAs
            (17, 18),  # DE-
            (19, 21),  # DEd
            (22, 24),  # DEm
            (25, 29),  # DEs
            (30, 35),  # Imag (I814 magnitude)
            (36, 41),  # z (photometric redshift)
            (42, 47),  # e_z (redshift error)
            (48, 53),  # zsp (spectroscopic redshift, -1 if none)
        ]

        df = pd.read_fwf(
            local_path,
            colspecs=colspecs,
            names=['id', 'RAh', 'RAm', 'RAs', 'DE_sign', 'DEd', 'DEm', 'DEs',
                   'Imag', 'z_phot', 'z_phot_err', 'z_spec'],
            comment='#'
        )

        # Convert RA/Dec to degrees
        df['ra'] = 15 * (df['RAh'] + df['RAm']/60 + df['RAs']/3600)

        # Handle declination sign
        dec_sign = df['DE_sign'].apply(lambda x: -1 if x == '-' else 1)
        df['dec'] = dec_sign * (df['DEd'] + df['DEm']/60 + df['DEs']/3600)

        # Clean up
        df['z_spec'] = df['z_spec'].replace(-1, np.nan)

        # Select useful columns
        df = df[['id', 'ra', 'dec', 'Imag', 'z_phot', 'z_phot_err', 'z_spec']].copy()

        print(f"Loaded {len(df)} sources from Fernández-Soto 1999 catalog")
        print(f"  With spectroscopic z: {df['z_spec'].notna().sum()}")

        return df

    except Exception as e:
        print(f"Error parsing catalog: {e}")
        return _load_fernandez_soto_fallback(data_dir)


def _load_fernandez_soto_fallback(data_dir: Path) -> pd.DataFrame:
    """Fallback: create a minimal test catalog if download fails."""
    print("Using fallback: returning empty DataFrame")
    return pd.DataFrame(columns=['id', 'ra', 'dec', 'z_phot', 'z_spec'])


# =============================================================================
# Hubble Legacy Fields (HLF) Catalog
# =============================================================================

HLF_CATALOG_URL = "https://archive.stsci.edu/hlsps/hlf/hlsp_hlf_hst_acs-30mas_goodss_f814w_v2.0_catalog.fits"
HLF_CATALOG_FULL_URL = "https://archive.stsci.edu/hlsps/hlf/hlsp_hlf_hst_goodss_v2.1_catalog.fits"


def download_hlf_catalog(
    data_dir: Optional[Path] = None,
    force_download: bool = False,
    use_full_catalog: bool = True
) -> Path:
    """
    Download the Hubble Legacy Fields GOODS-S catalog.

    Parameters
    ----------
    data_dir : Path, optional
        Directory to save the catalog
    force_download : bool
        Re-download even if file exists
    use_full_catalog : bool
        Download full v2.1 catalog (larger) vs F814W-only

    Returns
    -------
    Path
        Path to downloaded FITS file
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    url = HLF_CATALOG_FULL_URL if use_full_catalog else HLF_CATALOG_URL
    filename = "hlf_goodss_v2.1_catalog.fits" if use_full_catalog else "hlf_goodss_f814w_catalog.fits"
    local_path = data_dir / filename

    if local_path.exists() and not force_download:
        print(f"HLF catalog already exists at {local_path}")
        return local_path

    print(f"Downloading HLF catalog ({filename})...")
    print(f"This may take a few minutes (large file)...")

    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"Downloaded to {local_path}")
        return local_path
    except Exception as e:
        print(f"Download failed: {e}")
        print("You can manually download from: https://archive.stsci.edu/prepds/hlf/")
        raise


def load_hlf_catalog(
    catalog_path: Optional[Path] = None,
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load the HLF GOODS-S photometric catalog.

    Parameters
    ----------
    catalog_path : Path, optional
        Direct path to catalog FITS file
    data_dir : Path, optional
        Directory containing downloaded catalog

    Returns
    -------
    pd.DataFrame
        Catalog with photometry and photo-z
    """
    if not HAS_ASTROPY:
        raise ImportError("astropy required for FITS reading")

    if catalog_path is None:
        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR
        catalog_path = data_dir / "hlf_goodss_v2.1_catalog.fits"

    if not catalog_path.exists():
        print(f"Catalog not found at {catalog_path}")
        print("Run download_hlf_catalog() first, or download manually from:")
        print("https://archive.stsci.edu/prepds/hlf/")
        return pd.DataFrame()

    print(f"Loading HLF catalog from {catalog_path}...")

    with fits.open(catalog_path) as hdul:
        data = hdul[1].data

        # Extract key columns
        columns = {
            'id': data['ID'] if 'ID' in data.names else data['NUMBER'],
            'ra': data['RA'],
            'dec': data['DEC'],
        }

        # Try to get photo-z if available
        for z_col in ['Z_BEST', 'Z_PHOT', 'ZBEST', 'z_best', 'z_phot']:
            if z_col in data.names:
                columns['z_phot'] = data[z_col]
                break

        # Get flux columns (try different naming conventions)
        for band in ['F435W', 'F606W', 'F775W', 'F814W', 'F850LP']:
            for suffix in ['_FLUX', '_flux', '_FLUXAPER', '']:
                col = f'{band}{suffix}'
                if col in data.names:
                    columns[f'flux_{band.lower()}'] = data[col]
                    break

        df = pd.DataFrame(columns)

    print(f"Loaded {len(df)} sources from HLF catalog")

    return df


# =============================================================================
# Cross-matching
# =============================================================================

def cross_match_catalogs(
    catalog_ours: pd.DataFrame,
    catalog_reference: pd.DataFrame,
    max_separation_arcsec: float = 1.0,
    ra_col_ours: str = 'ra',
    dec_col_ours: str = 'dec',
    ra_col_ref: str = 'ra',
    dec_col_ref: str = 'dec'
) -> pd.DataFrame:
    """
    Cross-match two catalogs by position.

    Parameters
    ----------
    catalog_ours : pd.DataFrame
        Our detected catalog (must have ra, dec columns)
    catalog_reference : pd.DataFrame
        Reference catalog to match against
    max_separation_arcsec : float
        Maximum separation for a match in arcseconds
    ra_col_ours, dec_col_ours : str
        Column names for coordinates in our catalog
    ra_col_ref, dec_col_ref : str
        Column names for coordinates in reference catalog

    Returns
    -------
    pd.DataFrame
        Matched catalog with columns from both catalogs
    """
    if not HAS_ASTROQUERY:
        raise ImportError("astropy required for coordinate matching")

    # Handle case where our catalog uses pixel coordinates
    if ra_col_ours not in catalog_ours.columns:
        if 'xcentroid' in catalog_ours.columns:
            print("Warning: Our catalog has pixel coords, not RA/Dec.")
            print("Cross-matching requires sky coordinates.")
            print("You may need to add WCS transformation.")
            return pd.DataFrame()

    # Create SkyCoord objects
    coords_ours = SkyCoord(
        ra=catalog_ours[ra_col_ours].values * u.deg,
        dec=catalog_ours[dec_col_ours].values * u.deg
    )

    coords_ref = SkyCoord(
        ra=catalog_reference[ra_col_ref].values * u.deg,
        dec=catalog_reference[dec_col_ref].values * u.deg
    )

    # Match
    idx, sep2d, _ = match_coordinates_sky(coords_ours, coords_ref)

    # Filter by separation
    matched_mask = sep2d.arcsec < max_separation_arcsec

    # Build matched catalog
    matched_ours = catalog_ours[matched_mask].copy()
    matched_ref_idx = idx[matched_mask]

    # Add reference columns with suffix
    for col in catalog_reference.columns:
        matched_ours[f'{col}_ref'] = catalog_reference.iloc[matched_ref_idx][col].values

    matched_ours['separation_arcsec'] = sep2d.arcsec[matched_mask]

    print(f"Matched {len(matched_ours)} / {len(catalog_ours)} sources "
          f"(within {max_separation_arcsec}\")")

    return matched_ours


# =============================================================================
# Photo-z Validation
# =============================================================================

def validate_photoz(
    catalog_ours: pd.DataFrame,
    catalog_reference: pd.DataFrame,
    z_col_ours: str = 'redshift',
    z_col_ref: str = 'z_phot',
    max_separation_arcsec: float = 1.0,
    ra_col_ours: str = 'ra',
    dec_col_ours: str = 'dec',
) -> ValidationReport:
    """
    Validate our photo-z against a reference catalog.

    Parameters
    ----------
    catalog_ours : pd.DataFrame
        Our catalog with photo-z
    catalog_reference : pd.DataFrame
        Reference catalog with photo-z or spec-z
    z_col_ours : str
        Redshift column name in our catalog
    z_col_ref : str
        Redshift column name in reference catalog
    max_separation_arcsec : float
        Maximum separation for matching

    Returns
    -------
    ValidationReport
        Validation metrics and matched data
    """
    # Cross-match
    matched = cross_match_catalogs(
        catalog_ours, catalog_reference,
        max_separation_arcsec=max_separation_arcsec,
        ra_col_ours=ra_col_ours,
        dec_col_ours=dec_col_ours
    )

    if len(matched) == 0:
        print("No matches found. Check coordinate systems.")
        return None

    # Get redshifts
    z_ours = matched[z_col_ours].values
    z_ref = matched[f'{z_col_ref}_ref'].values

    # Filter valid redshifts
    valid = np.isfinite(z_ours) & np.isfinite(z_ref) & (z_ref > 0)
    z_ours = z_ours[valid]
    z_ref = z_ref[valid]
    sep = matched['separation_arcsec'].values[valid]

    if len(z_ours) == 0:
        print("No valid redshift pairs found.")
        return None

    # Compute metrics
    dz = (z_ours - z_ref) / (1 + z_ref)

    nmad = 1.48 * np.median(np.abs(dz - np.median(dz)))
    bias = np.median(dz)
    outlier_frac = np.mean(np.abs(dz) > 0.15)
    catastrophic_frac = np.mean(np.abs(dz) > 0.5)

    report = ValidationReport(
        n_matched=len(z_ours),
        n_total_ours=len(catalog_ours),
        n_total_reference=len(catalog_reference),
        nmad=nmad,
        bias=bias,
        outlier_fraction=outlier_frac,
        catastrophic_fraction=catastrophic_frac,
        z_ours=z_ours,
        z_reference=z_ref,
        separation_arcsec=sep
    )

    return report


def plot_photoz_comparison(report: ValidationReport, save_path: Optional[Path] = None):
    """
    Create diagnostic plots for photo-z validation.

    Parameters
    ----------
    report : ValidationReport
        Output from validate_photoz()
    save_path : Path, optional
        Save figure to this path
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: z_ours vs z_ref
    ax1 = axes[0]
    ax1.scatter(report.z_reference, report.z_ours, alpha=0.5, s=10)

    z_max = max(report.z_reference.max(), report.z_ours.max()) * 1.1
    ax1.plot([0, z_max], [0, z_max], 'k--', lw=1, label='1:1')

    # Outlier boundaries
    z_line = np.linspace(0, z_max, 100)
    ax1.fill_between(z_line, z_line - 0.15*(1+z_line), z_line + 0.15*(1+z_line),
                     alpha=0.2, color='gray', label='±0.15(1+z)')

    ax1.set_xlabel('Reference z')
    ax1.set_ylabel('Our z')
    ax1.set_title(f'Photo-z Comparison (N={report.n_matched})')
    ax1.legend()
    ax1.set_xlim(0, z_max)
    ax1.set_ylim(0, z_max)

    # Right: Δz/(1+z) histogram
    ax2 = axes[1]
    dz = (report.z_ours - report.z_reference) / (1 + report.z_reference)

    ax2.hist(dz, bins=50, range=(-0.5, 0.5), alpha=0.7, edgecolor='white')
    ax2.axvline(0, color='k', linestyle='--')
    ax2.axvline(report.nmad, color='red', linestyle=':', label=f'NMAD={report.nmad:.3f}')
    ax2.axvline(-report.nmad, color='red', linestyle=':')

    ax2.set_xlabel('Δz / (1+z)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Outliers: {100*report.outlier_fraction:.1f}%')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    return fig


# =============================================================================
# HDF-Specific VizieR Catalogs
# =============================================================================

# HDF-N center coordinates
HDF_N_CENTER_RA = 189.2058  # 12h 36m 49.4s in degrees
HDF_N_CENTER_DEC = 62.2161  # +62° 12' 58"

# HDF-S center coordinates
HDF_S_CENTER_RA = 338.2342  # 22h 32m 56.2s in degrees
HDF_S_CENTER_DEC = -60.5508  # -60° 33' 02.7"


def load_vizier_hdf_photoz(
    cache_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load Fernández-Soto 1999 HDF photo-z catalog from VizieR.

    VizieR Catalog: J/ApJ/513/34

    Returns
    -------
    pd.DataFrame
        1,067 sources with photo-z and some spec-z
    """
    if not HAS_ASTROQUERY:
        raise ImportError("astroquery required. Install with: pip install astroquery")

    # Check cache
    if cache_path and cache_path.exists():
        print(f"Loading cached catalog from {cache_path}")
        return pd.read_csv(cache_path)

    print("Querying VizieR J/ApJ/513/34 (Fernández-Soto 1999 HDF photo-z)...")

    Vizier.ROW_LIMIT = -1

    result = Vizier.get_catalogs("J/ApJ/513/34")

    if not result or len(result) == 0:
        print("No results from VizieR. Catalog may have moved.")
        return pd.DataFrame()

    # Get the main table
    df = result[0].to_pandas()

    # Standardize columns
    col_mapping = {
        '_RAJ2000': 'ra',
        '_DEJ2000': 'dec',
        'zph': 'z_phot',
        'e_zph': 'z_phot_err',
        'zsp': 'z_spec',
        'Imag': 'I_mag',
    }

    for old, new in col_mapping.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    # Replace -1 spec-z with NaN
    if 'z_spec' in df.columns:
        df['z_spec'] = df['z_spec'].replace(-1, np.nan)
        df['z_spec'] = df['z_spec'].replace(-1.0, np.nan)

    print(f"Loaded {len(df)} sources from Fernández-Soto 1999")
    if 'z_spec' in df.columns:
        print(f"  With spectroscopic z: {df['z_spec'].notna().sum()}")

    # Cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    return df


def load_vizier_hdf_surface_photometry(
    cache_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load Fasano+ 1998 HDF surface photometry catalog from VizieR.

    VizieR Catalog: J/A+AS/129/583

    This catalog contains surface photometry parameters useful for
    validating half-light radius measurements.

    Returns
    -------
    pd.DataFrame
        Surface photometry parameters (effective radius, surface brightness, etc.)
    """
    if not HAS_ASTROQUERY:
        raise ImportError("astroquery required. Install with: pip install astroquery")

    if cache_path and cache_path.exists():
        print(f"Loading cached catalog from {cache_path}")
        return pd.read_csv(cache_path)

    print("Querying VizieR J/A+AS/129/583 (Fasano+ 1998 HDF surface photometry)...")

    Vizier.ROW_LIMIT = -1

    result = Vizier.get_catalogs("J/A+AS/129/583")

    if not result or len(result) == 0:
        print("No results from VizieR.")
        return pd.DataFrame()

    df = result[0].to_pandas()

    # Standardize
    col_mapping = {
        '_RAJ2000': 'ra',
        '_DEJ2000': 'dec',
        'Re': 'r_eff',  # Effective radius
        'SBe': 'mu_eff',  # Surface brightness at r_eff
        'n': 'sersic_n',  # Sérsic index
    }

    for old, new in col_mapping.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    print(f"Loaded {len(df)} sources from Fasano+ 1998 surface photometry")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    return df


def load_vizier_hdf_ugrk(
    cache_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load Hogg+ 2000 UGRK photometry catalog from VizieR.

    VizieR Catalog: J/ApJS/127/1

    Returns
    -------
    pd.DataFrame
        UGRK photometry for HDF region
    """
    if not HAS_ASTROQUERY:
        raise ImportError("astroquery required. Install with: pip install astroquery")

    if cache_path and cache_path.exists():
        print(f"Loading cached catalog from {cache_path}")
        return pd.read_csv(cache_path)

    print("Querying VizieR J/ApJS/127/1 (Hogg+ 2000 HDF UGRK photometry)...")

    Vizier.ROW_LIMIT = -1

    result = Vizier.get_catalogs("J/ApJS/127/1")

    if not result or len(result) == 0:
        print("No results from VizieR.")
        return pd.DataFrame()

    df = result[0].to_pandas()

    col_mapping = {
        '_RAJ2000': 'ra',
        '_DEJ2000': 'dec',
    }

    for old, new in col_mapping.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    print(f"Loaded {len(df)} sources from Hogg+ 2000 UGRK photometry")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    return df


# =============================================================================
# Hawaii H-HDF-N Catalog (Yang+ 2014)
# =============================================================================

HAWAII_HDFN_URL = "https://www.astro.caltech.edu/~capak/hdf/hdfn_public_v1.0.cat.gz"


def load_hawaii_hdfn(
    data_dir: Optional[Path] = None,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Load Hawaii H-HDF-N catalog (Yang et al. 2014).

    This is the most comprehensive HDF-N catalog with 131,678 sources
    and 15-band photometry from U to IRAC 4.5μm.

    Reference: ApJS 215, 27 (2014)

    Returns
    -------
    pd.DataFrame
        Large catalog with photo-z and multi-band photometry
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    local_path = data_dir / "hawaii_hdfn_v1.0.cat"
    local_path_gz = data_dir / "hawaii_hdfn_v1.0.cat.gz"

    # Download if needed
    if not local_path.exists() and not local_path_gz.exists():
        if not force_download:
            print(f"Hawaii H-HDF-N catalog not found at {local_path}")
            print(f"Download manually from: {HAWAII_HDFN_URL}")
            print("Or set force_download=True")
            return pd.DataFrame()

        print("Downloading Hawaii H-HDF-N catalog...")
        try:
            urllib.request.urlretrieve(HAWAII_HDFN_URL, local_path_gz)
            print(f"Downloaded to {local_path_gz}")
        except Exception as e:
            print(f"Download failed: {e}")
            print(f"Download manually from: {HAWAII_HDFN_URL}")
            return pd.DataFrame()

    # Decompress if needed
    if local_path_gz.exists() and not local_path.exists():
        import gzip
        import shutil
        print("Decompressing...")
        with gzip.open(local_path_gz, 'rb') as f_in:
            with open(local_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    if not local_path.exists():
        print("Catalog file not found after download attempt.")
        return pd.DataFrame()

    # Parse the catalog
    print(f"Loading Hawaii H-HDF-N catalog from {local_path}...")

    try:
        # The catalog is space-separated
        df = pd.read_csv(local_path, sep=r'\s+', comment='#')

        # Standardize column names (depends on actual format)
        col_mapping = {
            'RA': 'ra',
            'DEC': 'dec',
            'ra': 'ra',
            'dec': 'dec',
            'z_best': 'z_phot',
            'z_phot': 'z_phot',
            'zphot': 'z_phot',
        }

        for old, new in col_mapping.items():
            if old in df.columns and old != new:
                df = df.rename(columns={old: new})

        print(f"Loaded {len(df)} sources from Hawaii H-HDF-N")

        return df

    except Exception as e:
        print(f"Error parsing catalog: {e}")
        return pd.DataFrame()


# =============================================================================
# Spectroscopic Redshift Validation
# =============================================================================

def validate_with_specz(
    catalog_ours: pd.DataFrame,
    catalog_reference: pd.DataFrame,
    z_col_ours: str = 'redshift',
    z_col_spec: str = 'z_spec',
    max_separation_arcsec: float = 1.0,
) -> ValidationReport:
    """
    Validate photo-z against SPECTROSCOPIC redshifts (ground truth).

    This is more valuable than comparing to other photo-z catalogs.

    Parameters
    ----------
    catalog_ours : pd.DataFrame
        Our catalog with photo-z (must have ra, dec, redshift)
    catalog_reference : pd.DataFrame
        Reference catalog with spectroscopic redshifts
    z_col_ours : str
        Photo-z column in our catalog
    z_col_spec : str
        Spec-z column in reference catalog
    max_separation_arcsec : float
        Match radius

    Returns
    -------
    ValidationReport
        Validation against ground truth
    """
    # First filter reference to only sources with spec-z
    if z_col_spec not in catalog_reference.columns:
        print(f"Column '{z_col_spec}' not found in reference catalog")
        return None

    ref_with_specz = catalog_reference[catalog_reference[z_col_spec].notna()].copy()

    if len(ref_with_specz) == 0:
        print("No spectroscopic redshifts in reference catalog")
        return None

    print(f"Reference catalog has {len(ref_with_specz)} sources with spec-z")

    # Use the standard validation but with spec-z column
    return validate_photoz(
        catalog_ours,
        ref_with_specz,
        z_col_ours=z_col_ours,
        z_col_ref=z_col_spec,
        max_separation_arcsec=max_separation_arcsec
    )


def get_hdf_specz_from_fernandez_soto() -> pd.DataFrame:
    """
    Extract sources with spectroscopic redshifts from Fernández-Soto catalog.

    The Fernández-Soto 1999 catalog includes ~50 sources with ground-based
    spectroscopic confirmation.

    Returns
    -------
    pd.DataFrame
        Subset with spectroscopic redshifts only
    """
    df = load_vizier_hdf_photoz()

    if df.empty:
        return df

    specz_subset = df[df['z_spec'].notna()].copy()

    print(f"Extracted {len(specz_subset)} sources with spectroscopic redshifts")
    print(f"  z_spec range: {specz_subset['z_spec'].min():.2f} - {specz_subset['z_spec'].max():.2f}")

    return specz_subset


# =============================================================================
# Size/Morphology Validation
# =============================================================================

@dataclass
class SizeValidationReport:
    """Results from half-light radius validation."""
    n_matched: int
    n_total_ours: int
    n_total_reference: int

    # Size metrics
    median_ratio: float  # median(r_ours / r_ref)
    scatter: float  # std of log ratio
    bias_percent: float  # systematic offset in percent

    # Arrays
    r_ours: np.ndarray
    r_reference: np.ndarray
    separation_arcsec: np.ndarray

    def __str__(self) -> str:
        return f"""
=== Size Validation Report ===
Matched sources: {self.n_matched} / {self.n_total_ours}
Reference catalog: {self.n_total_reference} sources

Size comparison:
  Median ratio (ours/ref): {self.median_ratio:.3f}
  Scatter (dex):           {self.scatter:.3f}
  Bias:                    {self.bias_percent:+.1f}%
"""


def validate_sizes(
    catalog_ours: pd.DataFrame,
    catalog_reference: pd.DataFrame,
    r_col_ours: str = 'r_half_arcsec',
    r_col_ref: str = 'r_eff',
    max_separation_arcsec: float = 1.0,
) -> SizeValidationReport:
    """
    Validate half-light radius measurements against reference catalog.

    Parameters
    ----------
    catalog_ours : pd.DataFrame
        Our catalog with size measurements
    catalog_reference : pd.DataFrame
        Reference catalog with sizes (e.g., Fasano+ 1998)
    r_col_ours : str
        Size column in our catalog (arcsec)
    r_col_ref : str
        Size column in reference (arcsec)

    Returns
    -------
    SizeValidationReport
    """
    matched = cross_match_catalogs(
        catalog_ours, catalog_reference,
        max_separation_arcsec=max_separation_arcsec
    )

    if len(matched) == 0:
        print("No matches found")
        return None

    r_ours = matched[r_col_ours].values
    r_ref = matched[f'{r_col_ref}_ref'].values
    sep = matched['separation_arcsec'].values

    # Filter valid sizes
    valid = (r_ours > 0) & (r_ref > 0) & np.isfinite(r_ours) & np.isfinite(r_ref)
    r_ours = r_ours[valid]
    r_ref = r_ref[valid]
    sep = sep[valid]

    if len(r_ours) == 0:
        print("No valid size pairs")
        return None

    # Compute metrics
    ratio = r_ours / r_ref
    log_ratio = np.log10(ratio)

    report = SizeValidationReport(
        n_matched=len(r_ours),
        n_total_ours=len(catalog_ours),
        n_total_reference=len(catalog_reference),
        median_ratio=np.median(ratio),
        scatter=np.std(log_ratio),
        bias_percent=100 * (np.median(ratio) - 1),
        r_ours=r_ours,
        r_reference=r_ref,
        separation_arcsec=sep
    )

    return report
