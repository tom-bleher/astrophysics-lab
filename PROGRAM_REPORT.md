# Astrophysics Lab Program Report

## Overview

This is an **Angular Size Test Analysis** pipeline for cosmological modeling. It analyzes galaxy data from the Hubble Deep Field North (HDF-N) to compare observed galaxy angular sizes against two cosmological models:

1. **Linear Hubble Law (Static)** - Assumes a naive Euclidean universe
2. **Flat ΛCDM Model** - Standard concordance cosmology with dark matter and dark energy

---

## Core Components

| File | Purpose |
|------|---------|
| `run_analysis.py` | Main orchestration pipeline (4079 lines) |
| `classify.py` | Photo-z estimation via SED template fitting |
| `scientific.py` | Cosmological distance calculations and model fitting |
| `generate_filtered_binning.py` | Redshift binning strategies and analysis |
| `resource_config.py` | Hardware profile detection and resource management |

---

## Pipeline Architecture (6 Stages)

```
FITS IMAGE (4096×4096)
    │
    ├─[1] LOAD FITS ─────────────── Read science + weight maps
    │
    ├─[2] DETECT SOURCES ─────────── Background estimation, source detection, deblending
    │
    ├─[3] MEASURE MORPHOLOGY ─────── Concentration, Sérsic index, half-light radius
    │
    ├─[4] CLASSIFY PHOTO-Z ───────── SED template fitting (10 galaxy templates)
    │                                IGM absorption, dust attenuation, χ² minimization
    │
    ├─[5] FILTER QUALITY ─────────── 12 quality flags, ODDS confidence, outlier removal
    │
    └─[6] BINNING & FITTING ──────── 3 binning strategies, Static vs ΛCDM model fitting
```

---

## Key Algorithms

### Photo-z Estimation (`classify.py`)
- **Method**: Chi-squared SED template fitting over redshift grid
- **Templates**: 10 galaxy types (elliptical → starburst)
- **Physics**: Madau IGM absorption + Calzetti dust attenuation
- **Output**: Redshift, PDF, ODDS confidence, galaxy type
- **Performance**: Vectorized with optional Numba JIT (50x speedup)

### Cosmological Fitting (`scientific.py`)
- **Static Model**: `θ = R / [(c/H₀)z]` (1 free parameter: R)
- **ΛCDM Model**: `θ = R / D_A(z; Ω_m)` (2 free parameters: R, Ω_m)
- **Constants**: H₀ = 70 km/s/Mpc, flat universe constraint (Ω_Λ = 1 - Ω_m)

### Binning Strategies (`generate_filtered_binning.py`)
1. **Equal Width** - Linear spacing in redshift
2. **Percentile** - Equal galaxy count per bin
3. **Bayesian Blocks** - Data-driven adaptive binning

---

## Quality Control

The pipeline uses 12 bit-packed quality flags:

| Flag | Description |
|------|-------------|
| BLENDED | Deblended overlapping sources |
| EDGE | Near image boundary |
| SATURATED | Saturated pixels |
| LOW_SNR | SNR < 5 |
| BAD_PHOTOZ | ODDS < 0.6 |
| PSF_LIKE | Point source (star) |
| UNRELIABLE_Z | σ_z > z |
| CATASTROPHIC_PHOTOZ | Outlier vs spec-z |

**High-quality mode** retains ~40% of sources with strict filtering.

---

## Data Flow

```
Input:  HDF-N FITS mosaics (F300W, F450W, F606W, F814W bands)
        11 Galaxy template spectra
        External spec-z catalogs for validation

Output: galaxy_catalog_full.csv (all sources)
        galaxy_catalog.csv (quality-filtered)
        angular_size_vs_redshift.pdf (main result)
        Per-type analysis plots
        Model fit parameters with χ²/p-values
```

---

## Supporting Modules

| Module | Function |
|--------|----------|
| `morphology/concentration.py` | Statmorph measurements (C, A, S, Gini, M20) |
| `morphology/star_classifier.py` | ML star/galaxy separation |
| `detection/sep_detection.py` | SExtractor-compatible source detection |
| `validation/specz_validation.py` | Cross-match with spectroscopic redshifts |

---

## Usage

```bash
python run_analysis.py full    # Full 4096×4096 image
python run_analysis.py chip3   # Extract Chip3 (2048×2048)
python run_analysis.py both    # Both analyses
```

Resource profiles auto-detect hardware or can be set via `ASTRO_RESOURCE_PROFILE=[low|medium|high]`.

---

## Scientific Goal

Compare angular size-redshift relationships to determine which cosmological model better fits observations. The ΛCDM model predicts angular sizes that reach a minimum around z~1.5 then increase, while the Static model predicts monotonic decrease. This test probes the geometry of the universe.
