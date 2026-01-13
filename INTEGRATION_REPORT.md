# Integration Report: External Data Sources and Tools

**Project:** Hubble Deep Field Galaxy Classification and Cosmological Analysis
**Author:** Generated for Itay Feldman
**Date:** 2026-01-13

---

## Executive Summary

This report outlines how to integrate external data sources, catalogs, and tools discovered through research into the existing HDF analysis pipeline. The current project performs:

1. Multi-band source detection (U, B, V, I) using photutils
2. Cross-matching sources across bands
3. Galaxy classification via SED template fitting
4. Photometric redshift estimation
5. Angular size-redshift analysis comparing Static vs ΛCDM cosmology

The integrations proposed below will enable validation, improve accuracy, and extend the scientific capabilities of the analysis.

---

## Table of Contents

1. [Current Pipeline Overview](#1-current-pipeline-overview)
2. [Catalog Integration for Validation](#2-catalog-integration-for-validation)
3. [Template Spectra Enhancement](#3-template-spectra-enhancement)
4. [Photometric Redshift Benchmarking](#4-photometric-redshift-benchmarking)
5. [Spectroscopic Redshift Validation](#5-spectroscopic-redshift-validation)
6. [Morphological Measurement Improvements](#6-morphological-measurement-improvements)
7. [Cosmological Analysis Extensions](#7-cosmological-analysis-extensions)
8. [Data Download Scripts](#8-data-download-scripts)
9. [Recommended Project Structure](#9-recommended-project-structure)
10. [References](#10-references)

---

## 1. Current Pipeline Overview

### Existing Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  FITS Images    │────▶│  Source Detection │────▶│  Cross-Match    │
│  (U, B, V, I)   │     │  (photutils)      │     │  (4 bands)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  θ(z) Analysis  │◀────│  Photo-z + Type   │◀────│  SED Fitting    │
│  (scientific.py)│     │  (classify.py)    │     │  (templates)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `astro.qmd` | Main analysis notebook |
| `classify.py` | SED template fitting, photo-z estimation |
| `scientific.py` | Cosmological models (θ_static, θ_ΛCDM) |
| `spectra/*.dat` | Galaxy template spectra (CWW-style) |

---

## 2. Catalog Integration for Validation

### 2.1 Hubble Legacy Fields (HLF) Catalog

The HLF provides a validated reference catalog with 103,098 sources in 13 bands.

**Integration approach:**

```python
# validation/load_hlf_catalog.py
"""Load and cross-match with HLF reference catalog."""

import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u

HLF_CATALOG_URL = "https://archive.stsci.edu/hlsps/hlf/hlsp_hlf_hst_goodss_v2.1_catalog.fits"

def load_hlf_catalog(local_path: str = "./data/hlf_catalog.fits") -> pd.DataFrame:
    """Load HLF GOODS-S photometric catalog."""
    with fits.open(local_path) as hdul:
        data = hdul[1].data

    # Extract relevant columns
    catalog = pd.DataFrame({
        'id': data['ID'],
        'ra': data['RA'],
        'dec': data['DEC'],
        'f435w': data['F435W_FLUX'],  # B-band equivalent
        'f606w': data['F606W_FLUX'],  # V-band equivalent
        'f775w': data['F775W_FLUX'],  # i-band
        'f814w': data['F814W_FLUX'],  # I-band equivalent
        'f850lp': data['F850LP_FLUX'],
        'photo_z': data['Z_BEST'] if 'Z_BEST' in data.names else None,
    })
    return catalog

def cross_match_catalogs(
    our_catalog: pd.DataFrame,
    reference_catalog: pd.DataFrame,
    max_sep: float = 1.0  # arcseconds
) -> pd.DataFrame:
    """Cross-match our detections with reference catalog."""
    our_coords = SkyCoord(
        ra=our_catalog['ra'].values * u.degree,
        dec=our_catalog['dec'].values * u.degree
    )
    ref_coords = SkyCoord(
        ra=reference_catalog['ra'].values * u.degree,
        dec=reference_catalog['dec'].values * u.degree
    )

    idx, sep2d, _ = match_coordinates_sky(our_coords, ref_coords)

    # Filter by separation
    matched = sep2d.arcsec < max_sep

    result = our_catalog[matched].copy()
    result['ref_idx'] = idx[matched]
    result['sep_arcsec'] = sep2d.arcsec[matched]
    result['ref_photo_z'] = reference_catalog.iloc[idx[matched]]['photo_z'].values

    return result
```

### 2.2 VizieR HUDF Catalog (II/258)

**Integration using astroquery:**

```python
# validation/vizier_query.py
"""Query VizieR for HUDF catalog data."""

from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

def query_hudf_vizier(
    center_ra: float,
    center_dec: float,
    radius_arcmin: float = 3.0
) -> pd.DataFrame:
    """Query VizieR HUDF catalog around a position."""

    Vizier.ROW_LIMIT = -1  # No row limit

    coord = SkyCoord(ra=center_ra*u.deg, dec=center_dec*u.deg)

    result = Vizier.query_region(
        coord,
        radius=radius_arcmin * u.arcmin,
        catalog="II/258"  # HUDF catalog
    )

    if result:
        return result[0].to_pandas()
    return pd.DataFrame()
```

### 2.3 Fernández-Soto 1999 Catalog (Original HDF-N)

This catalog is included with EAZY and provides 1,067 galaxies with photo-z:

```python
# validation/fernandez_soto.py
"""Load Fernández-Soto et al. 1999 HDF-N catalog from EAZY."""

import numpy as np
from pathlib import Path

def load_fernandez_soto_catalog(eazy_path: str) -> pd.DataFrame:
    """
    Load the HDF-N catalog from EAZY distribution.

    The catalog is at: eazy-photoz/inputs/hdfn_fs99/
    """
    catalog_path = Path(eazy_path) / "inputs" / "hdfn_fs99" / "hdfn_fs99.cat"

    # Catalog format from EAZY
    data = np.loadtxt(catalog_path, comments='#')

    return pd.DataFrame({
        'id': data[:, 0].astype(int),
        'ra': data[:, 1],
        'dec': data[:, 2],
        # Fluxes in various bands follow
        'z_phot': data[:, -2],  # Photometric redshift
        'z_spec': data[:, -1],  # Spectroscopic redshift (-1 if unavailable)
    })
```

---

## 3. Template Spectra Enhancement

### 3.1 Current Templates

The project uses CWW-style templates in `spectra/`:
- `elliptical.dat`, `S0.dat`, `Sa.dat`, `Sb.dat`
- `sbt1.dat` through `sbt6.dat` (starburst templates)

### 3.2 EAZY Template Integration

Download and use the extended EAZY templates:

```python
# templates/download_eazy_templates.py
"""Download EAZY template spectra for comparison."""

import urllib.request
from pathlib import Path

EAZY_TEMPLATE_BASE = "https://raw.githubusercontent.com/gbrammer/eazy-photoz/master/templates"

TEMPLATE_SETS = {
    'cww_kin': [
        'CWW+KIN/cww_E_ext.sed',
        'CWW+KIN/cww_Sbc_ext.sed',
        'CWW+KIN/cww_Scd_ext.sed',
        'CWW+KIN/cww_Im_ext.sed',
        'CWW+KIN/kinney_starb1.sed',
        'CWW+KIN/kinney_starb2.sed',
    ],
    'eazy_v1': [
        'EAZY_v1.0/eazy_v1.0_sed1.dat',
        'EAZY_v1.0/eazy_v1.0_sed2.dat',
        'EAZY_v1.0/eazy_v1.0_sed3.dat',
        'EAZY_v1.0/eazy_v1.0_sed4.dat',
        'EAZY_v1.0/eazy_v1.0_sed5.dat',
    ],
}

def download_templates(output_dir: str = "./spectra/eazy"):
    """Download EAZY template spectra."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for set_name, templates in TEMPLATE_SETS.items():
        set_path = output_path / set_name
        set_path.mkdir(exist_ok=True)

        for template in templates:
            url = f"{EAZY_TEMPLATE_BASE}/{template}"
            filename = template.split('/')[-1]
            local_path = set_path / filename

            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, local_path)

    print("Template download complete.")
```

### 3.3 Template Comparison Module

```python
# templates/compare_templates.py
"""Compare classification results using different template sets."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from classify import classify_galaxy

def compare_template_sets(
    fluxes: list,
    errors: list,
    template_paths: dict[str, str]
) -> dict:
    """
    Run classification with multiple template sets and compare.

    Parameters
    ----------
    fluxes : list
        Observed fluxes [B, I, U, V]
    errors : list
        Flux errors
    template_paths : dict
        Mapping of template set name to path

    Returns
    -------
    dict
        Results from each template set
    """
    results = {}

    for name, path in template_paths.items():
        try:
            galaxy_type, redshift = classify_galaxy(
                fluxes, errors, spectra_path=path
            )
            results[name] = {
                'type': galaxy_type,
                'redshift': redshift,
            }
        except Exception as e:
            results[name] = {'error': str(e)}

    return results

def plot_template_comparison(results: dict, true_z: float = None):
    """Plot comparison of redshift estimates from different templates."""
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(results.keys())
    redshifts = [r.get('redshift', np.nan) for r in results.values()]

    ax.barh(names, redshifts, color='steelblue')

    if true_z is not None:
        ax.axvline(true_z, color='red', linestyle='--',
                   label=f'Spectroscopic z={true_z:.3f}')
        ax.legend()

    ax.set_xlabel('Photometric Redshift')
    ax.set_title('Template Set Comparison')

    return fig
```

---

## 4. Photometric Redshift Benchmarking

### 4.1 EAZY-py Integration

Install and run EAZY as a benchmark:

```bash
pip install eazy
python -c "import eazy; eazy.fetch_eazy_photoz()"
```

**Benchmark script:**

```python
# benchmark/eazy_comparison.py
"""Benchmark our photo-z against EAZY."""

import eazy
import numpy as np
import pandas as pd
from pathlib import Path

def run_eazy_photoz(
    catalog: pd.DataFrame,
    filter_names: list = ['F435W', 'F606W', 'F775W', 'F814W'],
    output_dir: str = './eazy_output'
) -> pd.DataFrame:
    """
    Run EAZY on our photometry and compare results.

    Parameters
    ----------
    catalog : pd.DataFrame
        Must contain flux and error columns for each filter
    filter_names : list
        HST filter names
    output_dir : str
        Output directory for EAZY results
    """
    # Prepare EAZY input catalog
    eazy_cat = eazy.photoz.PhotoZ(
        param_file=None,
        translate_file=None,
        zeropoint_file=None,
        load_prior=True,
        load_products=False
    )

    # Set up photometry
    # ... (EAZY-specific setup)

    # Run fitting
    eazy_cat.fit_catalog()

    # Extract results
    results = pd.DataFrame({
        'z_eazy': eazy_cat.zbest,
        'z_eazy_lo': eazy_cat.zlo,
        'z_eazy_hi': eazy_cat.zhi,
        'chi2': eazy_cat.chi2_best,
    })

    return results

def compare_photoz_methods(
    our_results: pd.DataFrame,
    eazy_results: pd.DataFrame,
    spec_z: pd.Series = None
) -> dict:
    """
    Compare photo-z accuracy between methods.

    Returns scatter, outlier fraction, and bias metrics.
    """
    def photoz_metrics(z_phot, z_true):
        dz = (z_phot - z_true) / (1 + z_true)

        return {
            'nmad': 1.48 * np.median(np.abs(dz - np.median(dz))),
            'bias': np.median(dz),
            'outlier_frac': np.sum(np.abs(dz) > 0.15) / len(dz),
            'sigma': np.std(dz),
        }

    metrics = {}

    if spec_z is not None:
        mask = spec_z > 0  # Valid spectroscopic redshifts

        metrics['our_method'] = photoz_metrics(
            our_results['redshift'][mask], spec_z[mask]
        )
        metrics['eazy'] = photoz_metrics(
            eazy_results['z_eazy'][mask], spec_z[mask]
        )

    return metrics
```

### 4.2 Photo-z Quality Metrics

Add ODDS parameter calculation (already in `classify.py` as `classify_galaxy_with_pdf`):

```python
# In astro.qmd, use the PDF version for quality assessment:

from classify import classify_galaxy_with_pdf

# Replace simple classification with PDF version
for idx, row in sed_catalog.iterrows():
    fluxes = [row[f'flux_{band}'] for band in ['b', 'i', 'u', 'v']]
    errors = [row[f'error_{band}'] for band in ['b', 'i', 'u', 'v']]

    result = classify_galaxy_with_pdf(fluxes, errors, spectra_path=R".\spectra")

    sed_catalog.at[idx, 'redshift'] = result.redshift
    sed_catalog.at[idx, 'z_lo'] = result.z_lo
    sed_catalog.at[idx, 'z_hi'] = result.z_hi
    sed_catalog.at[idx, 'galaxy_type'] = result.galaxy_type
    sed_catalog.at[idx, 'odds'] = result.odds  # Quality flag
    sed_catalog.at[idx, 'chi2'] = result.chi_sq_min

# Filter by quality
high_quality = sed_catalog[sed_catalog['odds'] > 0.9]
print(f"High-quality photo-z: {len(high_quality)} / {len(sed_catalog)}")
```

---

## 5. Spectroscopic Redshift Validation

### 5.1 MUSE HUDF Spectroscopic Catalog

The MUSE survey provides 1,338 high-quality spectroscopic redshifts:

```python
# validation/muse_specz.py
"""Load and use MUSE spectroscopic redshifts for validation."""

from astroquery.vizier import Vizier
import pandas as pd

def load_muse_catalog() -> pd.DataFrame:
    """
    Load MUSE HUDF spectroscopic redshift catalog.

    Reference: Bacon et al. 2017, A&A, 608, A1
    VizieR catalog: J/A+A/608/A1
    """
    Vizier.ROW_LIMIT = -1

    result = Vizier.get_catalogs("J/A+A/608/A1")

    if result:
        muse_cat = result[0].to_pandas()
        return muse_cat[['RAJ2000', 'DEJ2000', 'zMUSE', 'Conf']]

    return pd.DataFrame()

def validate_with_specz(
    our_catalog: pd.DataFrame,
    muse_catalog: pd.DataFrame,
    match_radius: float = 1.0  # arcsec
) -> dict:
    """
    Validate our photo-z against MUSE spec-z.

    Returns
    -------
    dict with:
        - matched_catalog: Cross-matched sources
        - metrics: Accuracy statistics
        - outliers: Sources with |Δz/(1+z)| > 0.15
    """
    from astropy.coordinates import SkyCoord, match_coordinates_sky
    import astropy.units as u

    # Cross-match
    our_coords = SkyCoord(
        ra=our_catalog['ra'].values * u.deg,
        dec=our_catalog['dec'].values * u.deg
    )
    muse_coords = SkyCoord(
        ra=muse_catalog['RAJ2000'].values * u.deg,
        dec=muse_catalog['DEJ2000'].values * u.deg
    )

    idx, sep, _ = match_coordinates_sky(our_coords, muse_coords)
    matched = sep.arcsec < match_radius

    # Compute metrics
    z_phot = our_catalog.loc[matched, 'redshift'].values
    z_spec = muse_catalog.iloc[idx[matched]]['zMUSE'].values

    dz = (z_phot - z_spec) / (1 + z_spec)

    return {
        'n_matched': matched.sum(),
        'nmad': 1.48 * np.median(np.abs(dz)),
        'bias': np.median(dz),
        'outlier_frac': np.mean(np.abs(dz) > 0.15),
        'catastrophic_frac': np.mean(np.abs(dz) > 0.5),
    }
```

### 5.2 Validation Visualization

```python
# validation/plot_photoz_vs_specz.py
"""Visualization for photo-z vs spec-z comparison."""

import matplotlib.pyplot as plt
import numpy as np

def plot_photoz_vs_specz(
    z_phot: np.ndarray,
    z_spec: np.ndarray,
    z_err_lo: np.ndarray = None,
    z_err_hi: np.ndarray = None,
    title: str = "Photometric vs Spectroscopic Redshift"
):
    """
    Create diagnostic plot comparing photo-z to spec-z.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: z_phot vs z_spec
    ax1 = axes[0]

    ax1.scatter(z_spec, z_phot, alpha=0.5, s=20, c='steelblue')

    # 1:1 line
    z_range = [0, max(z_spec.max(), z_phot.max()) * 1.1]
    ax1.plot(z_range, z_range, 'k--', lw=1, label='1:1')

    # ±0.15(1+z) outlier lines
    z_arr = np.linspace(0, z_range[1], 100)
    ax1.fill_between(z_arr, z_arr - 0.15*(1+z_arr), z_arr + 0.15*(1+z_arr),
                     alpha=0.2, color='gray', label='±0.15(1+z)')

    ax1.set_xlabel('Spectroscopic Redshift')
    ax1.set_ylabel('Photometric Redshift')
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlim(z_range)
    ax1.set_ylim(z_range)

    # Right panel: Δz/(1+z) histogram
    ax2 = axes[1]

    dz = (z_phot - z_spec) / (1 + z_spec)
    nmad = 1.48 * np.median(np.abs(dz))
    outlier_frac = np.mean(np.abs(dz) > 0.15)

    ax2.hist(dz, bins=50, range=(-0.5, 0.5),
             color='steelblue', alpha=0.7, edgecolor='white')
    ax2.axvline(0, color='k', linestyle='--')
    ax2.axvline(nmad, color='red', linestyle=':', label=f'NMAD={nmad:.3f}')
    ax2.axvline(-nmad, color='red', linestyle=':')

    ax2.set_xlabel('Δz / (1+z)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Outlier fraction: {outlier_frac:.1%}')
    ax2.legend()

    plt.tight_layout()
    return fig
```

---

## 6. Morphological Measurement Improvements

### 6.1 PetroFit Integration for Half-Light Radius

The current `fluxfrac_radius(0.5)` from photutils can be improved:

```python
# morphology/petrofit_analysis.py
"""Use PetroFit for robust Sérsic fitting and half-light radius."""

# pip install petrofit

from petrofit.modeling import PSFConvolvedModel2D, model_to_image
from petrofit.petrosian import Petrosian
from petrofit.segmentation import make_catalog, plot_segments
from astropy.modeling.models import Sersic2D
import numpy as np

def measure_sersic_params(
    image: np.ndarray,
    segm_map,
    source_id: int,
    psf: np.ndarray = None
) -> dict:
    """
    Fit Sérsic profile to get robust effective radius.

    Returns
    -------
    dict with:
        - r_eff: Effective (half-light) radius in pixels
        - n: Sérsic index
        - ellip: Ellipticity
        - pa: Position angle
        - chi2: Fit quality
    """
    from petrofit.fitting import fit_model

    # Get source cutout
    # ... (extract source region)

    # Initial Sérsic model
    sersic_model = Sersic2D(
        amplitude=1,
        r_eff=5,  # Initial guess
        n=2.5,    # Initial guess
        x_0=cutout.shape[1]/2,
        y_0=cutout.shape[0]/2,
        ellip=0.3,
        theta=0,
    )

    # Fit with PSF convolution if available
    if psf is not None:
        model = PSFConvolvedModel2D(sersic_model, psf=psf)
    else:
        model = sersic_model

    fitted_model, fit_info = fit_model(
        cutout, model,
        maxiter=500,
        epsilon=1e-6
    )

    return {
        'r_eff': fitted_model.r_eff.value,
        'r_eff_arcsec': fitted_model.r_eff.value * PIXEL_SCALE,
        'n': fitted_model.n.value,
        'ellip': fitted_model.ellip.value,
        'pa': np.degrees(fitted_model.theta.value),
    }
```

### 6.2 Concentration Index for Star/Galaxy Separation

```python
# morphology/concentration.py
"""Calculate concentration index for star/galaxy classification."""

import numpy as np
from photutils.aperture import CircularAperture, aperture_photometry

def concentration_index(
    image: np.ndarray,
    x: float, y: float,
    r_inner: float = 3.0,  # pixels
    r_outer: float = 10.0  # pixels
) -> float:
    """
    Calculate concentration index C = 5 * log10(r_80 / r_20).

    Simplified version using fixed apertures.
    Stars typically have C > 3.5, galaxies C < 3.0
    """
    aper_inner = CircularAperture((x, y), r=r_inner)
    aper_outer = CircularAperture((x, y), r=r_outer)

    phot_inner = aperture_photometry(image, aper_inner)
    phot_outer = aperture_photometry(image, aper_outer)

    flux_inner = phot_inner['aperture_sum'][0]
    flux_outer = phot_outer['aperture_sum'][0]

    if flux_outer > 0 and flux_inner > 0:
        # Approximate C using ratio
        return -2.5 * np.log10(flux_inner / flux_outer)
    return np.nan

def classify_star_galaxy(
    image: np.ndarray,
    catalog: pd.DataFrame,
    c_threshold: float = 0.5  # Calibrate based on your data
) -> pd.Series:
    """
    Classify sources as stars or galaxies based on concentration.

    Returns boolean Series: True = galaxy, False = star
    """
    concentrations = []

    for _, row in catalog.iterrows():
        c = concentration_index(image, row['xcentroid'], row['ycentroid'])
        concentrations.append(c)

    c_arr = np.array(concentrations)

    # Stars are more concentrated (higher C)
    is_galaxy = c_arr < c_threshold

    return pd.Series(is_galaxy, index=catalog.index, name='is_galaxy')
```

### 6.3 Zoobot ML Classification (Optional)

```python
# morphology/zoobot_classify.py
"""Use Zoobot for deep learning morphology classification."""

# pip install zoobot

def classify_with_zoobot(image_paths: list[str]) -> pd.DataFrame:
    """
    Run Zoobot CNN classifier on galaxy images.

    Returns morphology predictions including:
    - smooth_or_featured
    - disk_edge_on
    - has_spiral_arms
    - bar
    - bulge_size
    """
    from zoobot.pytorch.training import finetune
    from zoobot.pytorch.predictions import predict_on_catalog

    # Load pretrained model
    model = finetune.load_pretrained_zoobot(
        checkpoint_loc='path/to/zoobot_checkpoint.ckpt'
    )

    # Run predictions
    predictions = predict_on_catalog(
        model=model,
        catalog=image_paths,
        # ... configuration
    )

    return predictions
```

---

## 7. Cosmological Analysis Extensions

### 7.1 Improved Angular Diameter Distance

Your `scientific.py` already uses astropy.cosmology. Add uncertainty propagation:

```python
# scientific_extended.py
"""Extended cosmological analysis with uncertainties."""

import numpy as np
from astropy.cosmology import FlatLambdaCDM, Planck18
import astropy.units as u
from scipy.optimize import minimize

def fit_cosmology_mcmc(
    z: np.ndarray,
    theta_arcsec: np.ndarray,
    theta_error: np.ndarray,
    n_samples: int = 10000
) -> dict:
    """
    Fit cosmological parameters using MCMC.

    Fits for:
    - R: Characteristic galaxy size [kpc]
    - Omega_m: Matter density parameter

    Returns posterior samples and best-fit values.
    """
    import emcee

    def log_likelihood(params, z, theta, theta_err):
        R_kpc, Omega_m = params

        if R_kpc <= 0 or Omega_m <= 0 or Omega_m >= 1:
            return -np.inf

        # Convert R to Mpc
        R_Mpc = R_kpc / 1000

        # Compute model
        cosmo = FlatLambdaCDM(H0=70, Om0=Omega_m)
        D_A = cosmo.angular_diameter_distance(z).to(u.Mpc).value

        theta_model = R_Mpc / D_A  # radians
        theta_model_arcsec = np.degrees(theta_model) * 3600

        # Chi-squared
        chi2 = np.sum(((theta - theta_model_arcsec) / theta_err)**2)

        return -0.5 * chi2

    def log_prior(params):
        R_kpc, Omega_m = params
        if 0.1 < R_kpc < 50 and 0.1 < Omega_m < 0.5:
            return 0.0
        return -np.inf

    def log_probability(params, z, theta, theta_err):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params, z, theta, theta_err)

    # Initialize walkers
    ndim = 2
    nwalkers = 32
    p0 = np.array([5.0, 0.3]) + 0.1 * np.random.randn(nwalkers, ndim)

    # Run MCMC
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=(z, theta_arcsec, theta_error)
    )
    sampler.run_mcmc(p0, n_samples, progress=True)

    # Extract results
    samples = sampler.get_chain(discard=1000, flat=True)

    return {
        'R_kpc': np.median(samples[:, 0]),
        'R_kpc_err': np.std(samples[:, 0]),
        'Omega_m': np.median(samples[:, 1]),
        'Omega_m_err': np.std(samples[:, 1]),
        'samples': samples,
    }
```

### 7.2 Tolman Surface Brightness Test

Add surface brightness analysis to test expansion:

```python
# cosmology/tolman_test.py
"""Implement Tolman surface brightness test."""

import numpy as np

def surface_brightness_vs_redshift(
    catalog: pd.DataFrame,
    flux_col: str,
    area_col: str,  # in pixels
    pixel_scale: float = 0.04  # arcsec/pixel
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate surface brightness vs redshift.

    In expanding universe: SB ∝ (1+z)^{-4}
    In static universe: SB = constant
    """
    # Surface brightness in flux per square arcsec
    area_arcsec2 = catalog[area_col] * pixel_scale**2
    sb = catalog[flux_col] / area_arcsec2

    z = catalog['redshift']

    return z.values, sb.values

def fit_tolman_exponent(z: np.ndarray, sb: np.ndarray) -> tuple[float, float]:
    """
    Fit SB ∝ (1+z)^n and return n with uncertainty.

    Expected: n ≈ -4 for expanding universe (before evolution correction)
    """
    from scipy.optimize import curve_fit

    def model(z, A, n):
        return A * (1 + z)**n

    # Initial guess
    p0 = [np.median(sb), -2]

    popt, pcov = curve_fit(model, z, sb, p0=p0)

    n = popt[1]
    n_err = np.sqrt(pcov[1, 1])

    return n, n_err
```

---

## 8. Data Download Scripts

### 8.1 Complete Download Script

```python
# download/fetch_all_data.py
"""Download all external data for the project."""

import os
import urllib.request
from pathlib import Path

DATA_SOURCES = {
    'hlf_catalog': {
        'url': 'https://archive.stsci.edu/hlsps/hlf/hlsp_hlf_hst_goodss_v2.1_catalog.fits',
        'local': './data/external/hlf_catalog.fits',
        'description': 'Hubble Legacy Fields GOODS-S catalog (103,098 sources)'
    },
    'eazy_templates': {
        'url': 'https://github.com/gbrammer/eazy-photoz/archive/refs/heads/master.zip',
        'local': './data/external/eazy-photoz.zip',
        'description': 'EAZY templates and example catalogs'
    },
}

def download_file(url: str, local_path: str, description: str = ""):
    """Download a file with progress indication."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        print(f"Already exists: {local_path}")
        return

    print(f"Downloading {description or url}...")
    urllib.request.urlretrieve(url, local_path)
    print(f"Saved to: {local_path}")

def download_all():
    """Download all external data sources."""
    for name, info in DATA_SOURCES.items():
        download_file(info['url'], info['local'], info['description'])

    print("\nAll downloads complete!")
    print("\nNext steps:")
    print("1. Unzip eazy-photoz.zip to access templates")
    print("2. Run validation scripts to compare with external catalogs")

if __name__ == '__main__':
    download_all()
```

### 8.2 MAST Query Script

```python
# download/query_mast.py
"""Query MAST for HDF data products."""

from astroquery.mast import Observations
import pandas as pd

def search_hdf_observations(
    target: str = "HDF",
    filters: list = None
) -> pd.DataFrame:
    """
    Search MAST for Hubble Deep Field observations.
    """
    if filters is None:
        filters = ['F300W', 'F450W', 'F606W', 'F814W']

    obs = Observations.query_criteria(
        obs_collection='HST',
        target_name=f'{target}*',
        filters=filters,
        dataproduct_type='image'
    )

    return obs.to_pandas()

def download_hdf_products(
    obs_table,
    output_dir: str = './data/hst/',
    product_type: str = 'SCIENCE'
):
    """Download data products for selected observations."""
    products = Observations.get_product_list(obs_table)

    # Filter to science products
    filtered = Observations.filter_products(
        products,
        productType=product_type,
        extension='fits'
    )

    # Download
    Observations.download_products(
        filtered,
        download_dir=output_dir
    )
```

---

## 9. Recommended Project Structure

```
astro_labb1/
├── astro.qmd                    # Main analysis notebook
├── classify.py                  # SED fitting (existing)
├── scientific.py                # Cosmology models (existing)
├── INTEGRATION_REPORT.md        # This document
│
├── spectra/                     # Template spectra
│   ├── elliptical.dat          # Existing templates
│   ├── S0.dat
│   └── eazy/                   # NEW: EAZY templates
│       ├── cww_kin/
│       └── eazy_v1/
│
├── data/
│   ├── fits/                   # Your FITS images
│   └── external/               # NEW: External catalogs
│       ├── hlf_catalog.fits
│       ├── muse_specz.fits
│       └── fernandez_soto.cat
│
├── validation/                  # NEW: Validation scripts
│   ├── __init__.py
│   ├── load_hlf_catalog.py
│   ├── vizier_query.py
│   ├── muse_specz.py
│   └── plot_photoz_vs_specz.py
│
├── benchmark/                   # NEW: Benchmarking
│   ├── __init__.py
│   └── eazy_comparison.py
│
├── morphology/                  # NEW: Morphology tools
│   ├── __init__.py
│   ├── petrofit_analysis.py
│   ├── concentration.py
│   └── zoobot_classify.py
│
├── cosmology/                   # NEW: Extended cosmology
│   ├── __init__.py
│   ├── scientific_extended.py
│   └── tolman_test.py
│
├── download/                    # NEW: Data acquisition
│   ├── fetch_all_data.py
│   ├── query_mast.py
│   └── download_eazy_templates.py
│
├── output/                      # Analysis outputs
│   ├── catalogs/
│   ├── figures/
│   └── validation_results/
│
└── tests/                       # Unit tests
    ├── test_classify.py
    ├── test_scientific.py
    └── test_validation.py
```

---

## 10. References

### Primary Data Sources

1. **Hubble Legacy Fields (HLF)**
   - Illingworth et al. 2019
   - https://archive.stsci.edu/prepds/hlf/

2. **MUSE HUDF Survey**
   - Bacon et al. 2017, A&A, 608, A1
   - https://www.aanda.org/articles/aa/full_html/2017/12/aa30833-17/aa30833-17.html

3. **Fernández-Soto et al. 1999**
   - ApJ, 513, 34
   - https://iopscience.iop.org/article/10.1086/306847

### Software & Tools

4. **EAZY**
   - Brammer, van Dokkum & Coppi 2008, ApJ, 686, 1503
   - https://github.com/gbrammer/eazy-py

5. **Photutils**
   - https://photutils.readthedocs.io/

6. **PetroFit**
   - https://github.com/PetroFit/petrofit

7. **Zoobot**
   - Walmsley et al. 2023
   - https://github.com/mwalmsley/zoobot

### Scientific Background

8. **Galaxy Size Evolution**
   - Bouwens et al. 2004, ApJL, 611, L1
   - https://arxiv.org/abs/astro-ph/0406562

9. **Tolman Surface Brightness Test**
   - Lubin & Sandage 2001, AJ, 122, 1084
   - https://ui.adsabs.harvard.edu/abs/2001AJ....122.1084L/abstract

10. **CWW Templates**
    - Coleman, Wu & Weedman 1980, ApJS, 43, 393

---

## Appendix A: Quick Start Commands

```bash
# Install required packages
pip install eazy petrofit astroquery emcee

# Download external data
python download/fetch_all_data.py

# Run EAZY benchmark
python benchmark/eazy_comparison.py

# Validate against spectroscopic redshifts
python validation/muse_specz.py
```

---

## Appendix B: Filter Mapping

Your project uses generic U, B, V, I band names. Here's the mapping to HST filters:

| Your Band | HST/WFPC2 | HST/ACS | Central λ (Å) |
|-----------|-----------|---------|---------------|
| U | F300W | - | 2940 |
| B | F450W | F435W | 4520 |
| V | F606W | F606W | 5940 |
| I | F814W | F814W | 7920 |

Ensure your conversion factors in `astro.qmd` match the appropriate zeropoints:

```python
# Updated conversion factors (verify against your data source)
conversion_factors = {
    'b': 8.8e-18,   # F450W
    'i': 2.45e-18,  # F814W
    'u': 5.99e-17,  # F300W
    'v': 1.89e-18,  # F606W
}
```

---

*Report generated 2026-01-13*
