# Comprehensive Hubble Deep Field Data Inventory

**Generated:** 2026-01-13
**Purpose:** Complete catalog of available data products for HDF-N and HDF-S analysis

---

## Table of Contents

1. [Original HDF Observations](#1-original-hdf-observations)
2. [Photometric Catalogs](#2-photometric-catalogs)
3. [Spectroscopic Redshift Catalogs](#3-spectroscopic-redshift-catalogs)
4. [Infrared Data](#4-infrared-data)
5. [Multi-Wavelength Data](#5-multi-wavelength-data)
6. [Morphology Catalogs](#6-morphology-catalogs)
7. [VizieR Catalogs](#7-vizier-catalogs)
8. [Data Access Links](#8-data-access-links)
9. [Key References](#9-key-references)

---

## 1. Original HDF Observations

### HDF-North (HDF-N)

| Parameter | Value |
|-----------|-------|
| **Observation Dates** | December 18-30, 1995 |
| **Coordinates** | RA: 12h 36m 49.4s, Dec: +62° 12' 58" (J2000) |
| **Instrument** | WFPC2 |
| **Field of View** | 5.3 arcmin² |
| **Pixel Scale** | 0.04 arcsec/pixel (WF), 0.0996 arcsec/pixel (PC) |
| **Total Exposure** | ~150 orbits (~10 days continuous) |

**Filters & Exposure Times:**

| Filter | Central λ | Bandwidth | Exposure Time | Limiting Mag (5σ) |
|--------|-----------|-----------|---------------|-------------------|
| F300W (U) | 2940 Å | 740 Å | 46,900 s | 27.0 AB |
| F450W (B) | 4520 Å | 1030 Å | 93,700 s | 28.4 AB |
| F606W (V) | 5940 Å | 2340 Å | 93,600 s | 28.8 AB |
| F814W (I) | 7920 Å | 2510 Å | 93,400 s | 28.1 AB |

**Sources Detected:** ~3,000 galaxies to I < 28.5

### HDF-South (HDF-S)

| Parameter | Value |
|-----------|-------|
| **Observation Dates** | September-October 1998 |
| **Coordinates** | RA: 22h 32m 56.2s, Dec: -60° 33' 02.7" (J2000) |
| **Instrument** | WFPC2 + STIS + NICMOS |
| **Field of View** | 5.3 arcmin² (WFPC2) |
| **Total Exposure** | 150 orbits |

**Key Difference:** Includes QSO J2233-606 (z=2.24) in the field

---

## 2. Photometric Catalogs

### Primary Catalogs

| Catalog | Sources | Bands | Photo-z? | Reference |
|---------|---------|-------|----------|-----------|
| **Williams et al. 1996** | ~3,000 | U B V I | No | [AJ 112, 1335](https://ui.adsabs.harvard.edu/abs/1996AJ....112.1335W/abstract) |
| **Fernández-Soto et al. 1999** | 1,067 | U B V I J H K | Yes | [ApJ 513, 34](https://ui.adsabs.harvard.edu/abs/1999ApJ...513...34F/abstract) |
| **Hawaii HDF-N (Yang+ 2014)** | 131,678 | 15 bands (U to IRAC 4.5μm) | Yes | [ApJS 215, 27](https://arxiv.org/abs/1410.6860) |
| **HDF-S ISAAC (Labbé+ 2003)** | 833 | U B V I J H K | Yes | [AJ 125, 1107](https://ui.adsabs.harvard.edu/abs/2003AJ....125.1107L) |

### Fernández-Soto 1999 Details

The most widely-used HDF photo-z catalog:
- **1,067 galaxies** with photo-z
- Uses 7-band photometry (UBVIJHK)
- Accuracy: Δz/(1+z) ≈ 0.1 for AB(8140) < 26.0
- Includes high-z candidates at z ≈ 5-6
- **Download:** Included with [EAZY code](https://github.com/gbrammer/eazy-photoz) as example

### Hawaii HDF-N (H-HDF-N)

Most comprehensive modern catalog:
- **131,678 sources** over 0.4 deg²
- 15 bands from U to IRAC 4.5μm
- Photo-z with EAZY
- Star/galaxy classification
- **Download:** [ifa.hawaii.edu/~capak/hdf/](https://www.astro.caltech.edu/~capak/hdf/)

---

## 3. Spectroscopic Redshift Catalogs

### Ground-Based Spectroscopy

| Survey | Instrument | Sources | z Range | Reference |
|--------|------------|---------|---------|-----------|
| **Cohen et al. 2000** | Keck/LRIS | ~150 (HDF proper) | 0.09-5.6 | [ApJ 538, 29](https://ui.adsabs.harvard.edu/abs/2000ApJ...538...29C) |
| **Steidel et al. 1996** | Keck/LRIS | 8 LBGs | 2.6-3.2 | [AJ 112, 352](https://ui.adsabs.harvard.edu/abs/1996AJ....112..352S) |
| **Spinrad et al. 1998** | Keck | 2 | z=5.34, 5.60 | Confirmed high-z |

### Spectroscopic Completeness

- **92% complete** for R < 24 in central HDF-N
- **92% complete** for R < 23 in flanking fields
- **~700 spec-z** in HDF-N region total
- **30 redshifts per arcmin²** - highest density field survey

### High-Redshift Confirmed Sources

| Object | z_spec | Method |
|--------|--------|--------|
| HDF 3-951.1 | 5.34 | Keck spectroscopy |
| HDF 4-473.0 | 5.60 | Keck spectroscopy |
| Multiple LBGs | 2.6-3.5 | Lyman break selection |

---

## 4. Infrared Data

### Near-Infrared (J, H, K)

| Dataset | Instrument | Depth | Coverage | Reference |
|---------|------------|-------|----------|-----------|
| **KPNO IRIM** | KPNO 4m | J~24, H~23, K~23 | HDF + flanking | [STScI clearinghouse](https://www.stsci.edu/ftp/science/hdf/clearinghouse/irim/irim_hdf.html) |
| **VLT/ISAAC (HDF-S)** | VLT | Ks~26 AB | 2.5'×2.5' | Labbé+ 2003 |
| **NICMOS (HDF-N)** | HST | H~28 | Central | Thompson+ |
| **Subaru/MOIRCS** | Subaru | J~25, K~24 | Wide field | Hawaii surveys |

### Mid/Far-Infrared

| Dataset | Instrument | Bands | Reference |
|---------|------------|-------|-----------|
| **Spitzer/IRAC** | Spitzer | 3.6, 4.5, 5.8, 8.0 μm | GOODS program |
| **ISO** | ISO | 15 μm | [artemis.ph.ic.ac.uk/hdf/](http://artemis.ph.ic.ac.uk/hdf/) |
| **SCUBA** | JCMT | 450, 850 μm | UK Sub-mm Consortium |

---

## 5. Multi-Wavelength Data

### X-Ray (Chandra)

| Survey | Exposure | Sources | Coverage |
|--------|----------|---------|----------|
| **Chandra 166 ks** | 166 ks | 6 in HDF | Initial observation |
| **Chandra 1 Ms** | 1 Ms | 370 | Extended HDF-N |
| **Chandra 2 Ms** | 2 Ms | 503 (20 in HDF) | Full GOODS-N |

**Download:** [HEASARC Chandra catalogs](https://heasarc.gsfc.nasa.gov/docs/cgro/db-perl/W3Browse/w3table.pl?MissionHelp=chandra)

### Radio

| Survey | Instrument | Frequency | Depth |
|--------|------------|-----------|-------|
| **VLA HDF** | VLA | 1.4 GHz | ~8 μJy |
| **MERLIN** | MERLIN | 1.4 GHz | High resolution |
| **WSRT** | Westerbork | 1.4 GHz | Wide field |
| **EVN VLBI** | European VLBI | 1.6 GHz | Milliarcsec resolution |

---

## 6. Morphology Catalogs

### Visual Classifications

| Catalog | Method | Sources | Reference |
|---------|--------|---------|-----------|
| **van den Bergh 1996** | Visual (eyeball) | HDF galaxies | [AJ 112, 359](https://ui.adsabs.harvard.edu/abs/1996AJ....112..359V) |
| **Abraham et al. 1996** | Concentration/Asymmetry | I < 25 | MDS extension |

### Quantitative Morphology

| Catalog | Parameters | Reference |
|---------|------------|-----------|
| **Fasano+ 1998** | Surface photometry | VizieR J/A+AS/129/583 |
| **Abraham+ 1996** | C (concentration), A (asymmetry) | |

### Key Finding

At I ~ 25 mag:
- **40% irregular/merging/peculiar** systems
- Barred spirals become rare beyond z > 0.5
- Hubble classification breaks down at high-z

---

## 7. VizieR Catalogs

Direct VizieR catalog IDs for HDF:

| Catalog ID | Description | Access |
|------------|-------------|--------|
| **J/ApJS/127/1** | UGRK Photometry (Hogg+ 2000) | [VizieR](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJS/127/1) |
| **J/A+AS/129/583** | Surface photometry (Fasano+ 1998) | [VizieR](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+AS/129/583) |
| **J/ApJ/513/34** | Photo-z catalog (Fernández-Soto 1999) | [VizieR](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJ/513/34) |

### Query Example (Python)

```python
from astroquery.vizier import Vizier

# Query Fernández-Soto 1999 HDF catalog
Vizier.ROW_LIMIT = -1
result = Vizier.get_catalogs("J/ApJ/513/34")
hdf_catalog = result[0].to_pandas()
print(f"Loaded {len(hdf_catalog)} sources")
```

---

## 8. Data Access Links

### Primary Archives

| Archive | URL | Content |
|---------|-----|---------|
| **STScI HDF Main** | [stsci.edu/ftp/science/hdf/](https://www.stsci.edu/ftp/science/hdf/hdf.html) | Original images, catalogs |
| **STScI Clearinghouse** | [clearinghouse](https://www.stsci.edu/ftp/science/hdf/clearinghouse/clearinghouse.html) | All follow-up data |
| **MAST** | [archive.stsci.edu](https://archive.stsci.edu/) | HST data products |
| **HDF-S Archive** | [hdfsouth](https://www.stsci.edu/ftp/science/hdfsouth/) | HDF-S specific |

### Specialized Resources

| Resource | URL | Content |
|----------|-----|---------|
| **Hawaii Active Catalog** | [ifa.hawaii.edu/~cowie/tts/](http://www.ifa.hawaii.edu/~cowie/tts/tts.html) | Interactive HDF catalog |
| **Hawaii HDF-N** | [astro.caltech.edu/~capak/hdf/](https://www.astro.caltech.edu/~capak/hdf/) | H-HDF-N data |
| **Chandra HDF** | [ifa.hawaii.edu/users/cowie/chandra/](http://www.ifa.hawaii.edu/users/cowie/chandra/chandra_bk_4.html) | X-ray sources |
| **EAZY (includes HDF-N)** | [github.com/gbrammer/eazy-photoz](https://github.com/gbrammer/eazy-photoz) | Photo-z code + HDF example |

---

## 9. Key References

### Foundational Papers

1. **Williams et al. (1996)** - "The Hubble Deep Field: Observations, Data Reduction, and Galaxy Photometry"
   - [AJ 112, 1335](https://ui.adsabs.harvard.edu/abs/1996AJ....112.1335W/abstract) | [arXiv:astro-ph/9607174](https://arxiv.org/abs/astro-ph/9607174)
   - Original HDF catalog

2. **Fernández-Soto et al. (1999)** - "A New Catalog of Photometric Redshifts in the Hubble Deep Field"
   - [ApJ 513, 34](https://ui.adsabs.harvard.edu/abs/1999ApJ...513...34F/abstract)
   - Primary photo-z reference

3. **Cohen et al. (2000)** - "Redshift Clustering in the HDF"
   - [ApJ 538, 29](https://ui.adsabs.harvard.edu/abs/2000ApJ...538...29C)
   - Spectroscopic survey

4. **Ferguson et al. (2000)** - "The Hubble Deep Fields" (Review)
   - [ARA&A 38, 667](https://www.astro.ucla.edu/~malkan/astro278/hdf.pdf)
   - Comprehensive review

### High-Redshift Studies

5. **Steidel et al. (1996)** - "Spectroscopy of Lyman Break Galaxies"
   - [AJ 112, 352](https://ui.adsabs.harvard.edu/abs/1996AJ....112..352S)
   - LBG discovery in HDF

6. **Madau et al. (1996)** - "High-redshift galaxies in the HDF"
   - [MNRAS 283, 1388](https://academic.oup.com/mnras/article/283/4/1388/1071226)
   - Star formation history

### Morphology Studies

7. **Abraham et al. (1996)** - "Galaxy morphology to I=25 mag"
   - [MNRAS 279, L47](https://academic.oup.com/mnras/article/279/3/L47/967812)
   - Concentration/asymmetry

8. **van den Bergh et al. (1996)** - "A Morphological Catalog"
   - [AJ 112, 359](https://ui.adsabs.harvard.edu/abs/1996AJ....112..359V)
   - Visual classification

---

## Summary: Most Useful Catalogs for Your Project

Based on your current analysis (photo-z + angular size vs redshift), prioritize:

| Priority | Catalog | Why |
|----------|---------|-----|
| **1** | Fernández-Soto 1999 | Direct photo-z validation for HDF |
| **2** | Cohen et al. 2000 spec-z | Ground truth redshifts |
| **3** | Hawaii H-HDF-N | 131k sources, modern photo-z |
| **4** | VizieR J/A+AS/129/583 | Surface photometry for size validation |
| **5** | Abraham et al. morphology | Concentration/asymmetry metrics |

---

## Appendix: Coordinate Reference

**HDF-N Center:** RA 12:36:49.4, Dec +62:12:58 (J2000)
**HDF-S Center:** RA 22:32:56.2, Dec -60:33:02.7 (J2000)

To check if your data is from HDF-N or HDF-S, examine the WCS in your FITS headers.

---

*This inventory was compiled from STScI archives, NASA/ADS, VizieR, and published literature.*
