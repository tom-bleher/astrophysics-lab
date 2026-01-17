"""Spectroscopic redshift validation and improvement module.

This module provides functions to:
1. Load external spectroscopic catalogs (3D-HST, Hawaii Yang 2014, Fernandez-Soto 1999)
2. Cross-match with photo-z catalog using spherical coordinates
3. Compute standard photo-z validation metrics (NMAD, outlier rate, bias)
4. Optionally replace unreliable photo-z with spec-z
5. Generate validation plots

References:
- 3D-HST: Skelton et al. 2014, ApJS, 214, 24
- Hawaii HDFN: Yang et al. 2014, ApJ, 793, 40
- Fernandez-Soto et al. 1999, ApJ, 513, 34
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

# Quality flag for catastrophic outliers
FLAG_CATASTROPHIC_PHOTOZ = 2048


def _sexagesimal_to_decimal(ra_str: str, dec_str: str) -> tuple[float, float]:
    """Convert sexagesimal coordinates to decimal degrees.

    Args:
        ra_str: Right ascension in "HH MM SS.sss" format
        dec_str: Declination in "+DD MM SS.ss" format

    Returns:
        (ra_deg, dec_deg) tuple of decimal degrees
    """
    # Parse RA: "HH MM SS.sss" -> hours, minutes, seconds
    ra_parts = ra_str.split()
    ra_h = float(ra_parts[0])
    ra_m = float(ra_parts[1])
    ra_s = float(ra_parts[2])
    ra_deg = 15.0 * (ra_h + ra_m / 60.0 + ra_s / 3600.0)  # 15 deg/hour

    # Parse Dec: "+DD MM SS.ss" -> degrees, arcmin, arcsec
    dec_parts = dec_str.split()
    dec_d = float(dec_parts[0])
    dec_m = float(dec_parts[1])
    dec_s = float(dec_parts[2])
    sign = -1 if dec_d < 0 or dec_str.strip().startswith('-') else 1
    dec_deg = sign * (abs(dec_d) + dec_m / 60.0 + dec_s / 3600.0)

    return ra_deg, dec_deg


def load_specz_catalogs(data_dir: str = "./data/external") -> dict[str, pd.DataFrame]:
    """Load all three external spectroscopic catalogs.

    Args:
        data_dir: Directory containing the external catalog files

    Returns:
        Dictionary mapping catalog names to DataFrames with columns:
        - ra: Right ascension in decimal degrees
        - dec: Declination in decimal degrees
        - z_spec: Spectroscopic redshift
    """
    data_path = Path(data_dir)
    catalogs = {}

    # 1. 3D-HST GOODS-N catalog
    try:
        hst_path = data_path / "3dhst_goodsn.csv"
        if hst_path.exists():
            df = pd.read_csv(hst_path)
            # Filter to sources with valid spec-z
            df = df[df["z_spec"].notna() & (df["z_spec"] > 0)]
            catalogs["3D-HST"] = pd.DataFrame({
                "ra": df["ra"],
                "dec": df["dec"],
                "z_spec": df["z_spec"],
            })
            print(f"  Loaded 3D-HST: {len(catalogs['3D-HST'])} sources with spec-z")
    except Exception as e:
        print(f"  Warning: Could not load 3D-HST catalog: {e}")

    # 2. Hawaii HDFN Yang 2014 catalog
    try:
        hawaii_path = data_path / "hawaii_hdfn_yang2014.csv"
        if hawaii_path.exists():
            df = pd.read_csv(hawaii_path, low_memory=False)
            # Filter to sources with valid spec-z (column 'zsp')
            df = df[df["zsp"].notna() & (df["zsp"] > 0)]
            catalogs["Hawaii-Yang2014"] = pd.DataFrame({
                "ra": df["ra"],
                "dec": df["dec"],
                "z_spec": df["zsp"],
            })
            print(f"  Loaded Hawaii-Yang2014: {len(catalogs['Hawaii-Yang2014'])} sources with spec-z")
    except Exception as e:
        print(f"  Warning: Could not load Hawaii-Yang2014 catalog: {e}")

    # 3. Fernandez-Soto 1999 catalog (needs coordinate conversion)
    try:
        fs_path = data_path / "fernandez_soto_1999.csv"
        if fs_path.exists():
            df = pd.read_csv(fs_path)
            # Filter to sources with valid spec-z
            df = df[df["z_spec"].notna() & (df["z_spec"] > 0)]

            # Convert sexagesimal to decimal degrees using astropy (vectorized)
            try:
                # Use astropy's SkyCoord for efficient batch parsing
                coords = SkyCoord(
                    df["RAJ2000"].values,
                    df["DEJ2000"].values,
                    unit=(u.hourangle, u.deg),
                )
                catalogs["Fernandez-Soto1999"] = pd.DataFrame({
                    "ra": coords.ra.deg,
                    "dec": coords.dec.deg,
                    "z_spec": df["z_spec"].values,
                })
                print(f"  Loaded Fernandez-Soto1999: {len(catalogs['Fernandez-Soto1999'])} sources with spec-z")
            except Exception:
                # Fallback to row-by-row conversion if batch fails
                ra_dec_list = []
                for _, row in df.iterrows():
                    try:
                        ra_deg, dec_deg = _sexagesimal_to_decimal(
                            row["RAJ2000"], row["DEJ2000"]
                        )
                        ra_dec_list.append({
                            "ra": ra_deg,
                            "dec": dec_deg,
                            "z_spec": row["z_spec"],
                        })
                    except (ValueError, IndexError):
                        continue
                if ra_dec_list:
                    catalogs["Fernandez-Soto1999"] = pd.DataFrame(ra_dec_list)
                    print(f"  Loaded Fernandez-Soto1999: {len(catalogs['Fernandez-Soto1999'])} sources with spec-z")
    except Exception as e:
        print(f"  Warning: Could not load Fernandez-Soto1999 catalog: {e}")

    # 4. Team Keck Treasury Redshift Survey (TKRS) - Wirth et al. 2004
    # High-quality Keck/DEIMOS spectroscopy in GOODS-N, quality 3-4 only
    try:
        tkrs_path = data_path / "tkrs_goodsn.csv"
        if tkrs_path.exists():
            df = pd.read_csv(tkrs_path)
            df = df[df["z_spec"].notna() & (df["z_spec"] > 0)]
            catalogs["TKRS"] = pd.DataFrame({
                "ra": df["ra"],
                "dec": df["dec"],
                "z_spec": df["z_spec"],
            })
            print(f"  Loaded TKRS: {len(catalogs['TKRS'])} sources with spec-z")
    except Exception as e:
        print(f"  Warning: Could not load TKRS catalog: {e}")

    # 5. JADES DR4 - JWST/NIRSpec spectroscopy (D'Eugenio et al. 2024)
    # Deep JWST spectroscopy, especially valuable for high-z validation
    try:
        jades_path = data_path / "jades_dr4_goodsn.csv"
        if jades_path.exists():
            df = pd.read_csv(jades_path)
            df = df[df["z_spec"].notna() & (df["z_spec"] > 0)]
            catalogs["JADES-DR4"] = pd.DataFrame({
                "ra": df["ra"],
                "dec": df["dec"],
                "z_spec": df["z_spec"],
            })
            print(f"  Loaded JADES-DR4: {len(catalogs['JADES-DR4'])} sources with spec-z")
    except Exception as e:
        print(f"  Warning: Could not load JADES-DR4 catalog: {e}")

    # 6. Chandra Deep Field North - X-ray selected sources with spec-z
    # Useful for AGN identification and validation
    try:
        chandra_path = data_path / "chandra_cdfn.csv"
        if chandra_path.exists():
            df = pd.read_csv(chandra_path)
            df = df[df["z_spec"].notna() & (df["z_spec"] > 0)]
            catalogs["Chandra-CDFN"] = pd.DataFrame({
                "ra": df["ra"],
                "dec": df["dec"],
                "z_spec": df["z_spec"],
            })
            print(f"  Loaded Chandra-CDFN: {len(catalogs['Chandra-CDFN'])} sources with spec-z")
    except Exception as e:
        print(f"  Warning: Could not load Chandra-CDFN catalog: {e}")

    return catalogs


def cross_match_with_specz(
    catalog: pd.DataFrame,
    specz_catalogs: dict[str, pd.DataFrame],
    match_radius: float = 1.0,
) -> pd.DataFrame:
    """Cross-match catalog with spectroscopic catalogs using spherical matching.

    Args:
        catalog: Input catalog with 'ra' and 'dec' columns
        specz_catalogs: Dictionary of spec-z catalogs from load_specz_catalogs()
        match_radius: Maximum separation in arcsec for a valid match

    Returns:
        Input catalog with added columns:
        - z_spec: Spectroscopic redshift (NaN if no match)
        - z_spec_source: Name of source catalog
        - z_spec_sep_arcsec: Match separation in arcsec
    """
    # Make a copy to avoid modifying the original
    result = catalog.copy()

    # Initialize new columns
    result["z_spec"] = np.nan
    result["z_spec_source"] = ""
    result["z_spec_sep_arcsec"] = np.nan

    if not specz_catalogs:
        print("  No spectroscopic catalogs available for cross-matching")
        return result

    # Check for required columns
    if "ra" not in catalog.columns or "dec" not in catalog.columns:
        print("  Warning: Catalog missing 'ra' and/or 'dec' columns, skipping cross-match")
        return result

    # Create SkyCoord for input catalog
    try:
        catalog_coords = SkyCoord(
            ra=catalog["ra"].values * u.degree,
            dec=catalog["dec"].values * u.degree,
        )
    except Exception as e:
        print(f"  Warning: Could not create coordinates for input catalog: {e}")
        return result

    total_matches = 0

    # Match against each spec-z catalog
    for cat_name, specz_df in specz_catalogs.items():
        try:
            specz_coords = SkyCoord(
                ra=specz_df["ra"].values * u.degree,
                dec=specz_df["dec"].values * u.degree,
            )

            # Find nearest matches
            idx, sep2d, _ = catalog_coords.match_to_catalog_sky(specz_coords)
            sep_arcsec = sep2d.arcsec

            # Apply match radius criterion
            # Only update sources that don't already have a closer match
            for i in range(len(catalog)):
                if sep_arcsec[i] <= match_radius:
                    # Check if this is a better match than existing
                    existing_sep = result.iloc[i]["z_spec_sep_arcsec"]
                    if pd.isna(existing_sep) or sep_arcsec[i] < existing_sep:
                        result.iloc[i, result.columns.get_loc("z_spec")] = specz_df.iloc[idx[i]]["z_spec"]
                        result.iloc[i, result.columns.get_loc("z_spec_source")] = cat_name
                        result.iloc[i, result.columns.get_loc("z_spec_sep_arcsec")] = sep_arcsec[i]

            n_matches = (sep_arcsec <= match_radius).sum()
            total_matches += n_matches
            print(f"  Matched {n_matches} sources to {cat_name} (radius={match_radius}\")")

        except Exception as e:
            print(f"  Warning: Could not match against {cat_name}: {e}")

    # Report unique matches (after deduplication)
    n_unique = result["z_spec"].notna().sum()
    print(f"  Total unique spec-z matches: {n_unique}")

    return result


def compute_photoz_metrics(z_phot: np.ndarray, z_spec: np.ndarray) -> dict:
    """Compute standard photo-z validation metrics.

    Args:
        z_phot: Array of photometric redshifts
        z_spec: Array of spectroscopic redshifts

    Returns:
        Dictionary with metrics:
        - nmad: Normalized median absolute deviation
        - bias: Median of delta_z / (1+z_spec)
        - outlier_015: Fraction with |delta_z/(1+z)| > 0.15
        - outlier_030: Fraction with |delta_z/(1+z)| > 0.3 (catastrophic)
        - n_matched: Number of matched sources
    """
    # Handle array-like inputs
    z_phot = np.asarray(z_phot)
    z_spec = np.asarray(z_spec)

    # Filter to valid pairs
    valid = np.isfinite(z_phot) & np.isfinite(z_spec) & (z_spec > 0)
    z_phot = z_phot[valid]
    z_spec = z_spec[valid]

    if len(z_phot) == 0:
        return {
            "nmad": np.nan,
            "bias": np.nan,
            "outlier_015": np.nan,
            "outlier_030": np.nan,
            "n_matched": 0,
        }

    # Compute delta_z / (1+z_spec)
    delta_z_norm = (z_phot - z_spec) / (1 + z_spec)

    # NMAD: 1.48 * median(|delta_z/(1+z)|)
    nmad = 1.48 * np.median(np.abs(delta_z_norm))

    # Bias: median(delta_z/(1+z))
    bias = np.median(delta_z_norm)

    # Outlier fractions
    outlier_015 = np.mean(np.abs(delta_z_norm) > 0.15)
    outlier_030 = np.mean(np.abs(delta_z_norm) > 0.30)

    return {
        "nmad": nmad,
        "bias": bias,
        "outlier_015": outlier_015,
        "outlier_030": outlier_030,
        "n_matched": len(z_phot),
    }


def apply_spectroscopic_redshifts(
    catalog: pd.DataFrame,
    replace_threshold: float = 0.15,
) -> pd.DataFrame:
    """Replace unreliable photo-z with spec-z when available.

    For sources with spec-z matches:
    - If |z_phot - z_spec|/(1+z_spec) > threshold: flag as catastrophic outlier
    - Replace photo-z with spec-z
    - Update redshift_err to typical spec-z error (~0.001)
    - Add 'redshift_source' column ('spec' vs 'phot')

    Args:
        catalog: Catalog with z_spec column from cross_match_with_specz()
        replace_threshold: Threshold for flagging catastrophic outliers

    Returns:
        Updated catalog with spec-z applied
    """
    result = catalog.copy()

    # Preserve original photo-z before replacement
    if "redshift" in result.columns:
        result["redshift_phot_original"] = result["redshift"].copy()

    # Initialize new columns
    result["redshift_source"] = "phot"
    result["catastrophic_outlier"] = False

    # Find sources with spec-z matches
    has_specz = result["z_spec"].notna()
    n_specz = has_specz.sum()

    if n_specz == 0:
        print("  No spec-z matches to apply")
        return result

    # Identify catastrophic outliers before replacement
    if "redshift" in result.columns:
        z_phot = result.loc[has_specz, "redshift"]
        z_spec = result.loc[has_specz, "z_spec"]
        delta_z_norm = np.abs((z_phot - z_spec) / (1 + z_spec))

        catastrophic_mask = delta_z_norm > replace_threshold
        n_catastrophic = catastrophic_mask.sum()

        # Mark catastrophic outliers
        catastrophic_indices = result.index[has_specz][catastrophic_mask]
        result.loc[catastrophic_indices, "catastrophic_outlier"] = True

        print(f"  Identified {n_catastrophic} catastrophic outliers ({100*n_catastrophic/n_specz:.1f}% of matched)")

    # Replace photo-z with spec-z
    result.loc[has_specz, "redshift"] = result.loc[has_specz, "z_spec"]
    result.loc[has_specz, "redshift_source"] = "spec"

    # Update redshift errors for spec-z sources (typical spec-z precision)
    if "redshift_err" in result.columns:
        result.loc[has_specz, "redshift_err"] = 0.001

    print(f"  Replaced {n_specz} photo-z values with spec-z")

    return result


def flag_catastrophic_outliers(
    catalog: pd.DataFrame,
    threshold: float = 0.15,
) -> pd.DataFrame:
    """Add FLAG_CATASTROPHIC_PHOTOZ to quality flags for outliers.

    Only flags sources where:
    - spec-z is available
    - |z_phot - z_spec|/(1+z_spec) > threshold

    Args:
        catalog: Catalog with z_spec and quality_flag columns
        threshold: Threshold for catastrophic outlier classification

    Returns:
        Catalog with updated quality_flag column
    """
    result = catalog.copy()

    if "quality_flag" not in result.columns:
        print("  Warning: No quality_flag column, cannot flag outliers")
        return result

    if "z_spec" not in result.columns or "redshift_phot_original" not in result.columns:
        # Use current redshift if original not available
        z_phot_col = "redshift_phot_original" if "redshift_phot_original" in result.columns else "redshift"
        if z_phot_col not in result.columns:
            print("  Warning: No redshift column available for outlier flagging")
            return result
    else:
        z_phot_col = "redshift_phot_original"

    # Find catastrophic outliers
    has_specz = result["z_spec"].notna()
    if has_specz.sum() == 0:
        return result

    z_phot = result.loc[has_specz, z_phot_col]
    z_spec = result.loc[has_specz, "z_spec"]
    delta_z_norm = np.abs((z_phot - z_spec) / (1 + z_spec))

    catastrophic_mask = delta_z_norm > threshold
    catastrophic_indices = result.index[has_specz][catastrophic_mask]

    # Update quality flags
    result.loc[catastrophic_indices, "quality_flag"] = (
        result.loc[catastrophic_indices, "quality_flag"].astype(int) | FLAG_CATASTROPHIC_PHOTOZ
    )

    n_flagged = len(catastrophic_indices)
    print(f"  Added FLAG_CATASTROPHIC_PHOTOZ to {n_flagged} sources")

    return result


def generate_validation_plots(
    catalog: pd.DataFrame,
    output_dir: str,
    prefix: str = "photoz",
) -> None:
    """Generate validation plots for photo-z quality assessment.

    Creates:
    1. z_phot vs z_spec scatter plot with 1:1 line
    2. delta_z/(1+z) histogram with NMAD annotation
    3. Outlier fraction vs redshift
    4. validation_summary.txt with all metrics

    Args:
        catalog: Catalog with z_spec and redshift columns
        output_dir: Directory to save plots
        prefix: Prefix for output filenames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get photo-z column (original if available)
    if "redshift_phot_original" in catalog.columns:
        z_phot_col = "redshift_phot_original"
    else:
        z_phot_col = "redshift"

    # Filter to sources with spec-z
    has_specz = catalog["z_spec"].notna()
    matched = catalog[has_specz].copy()

    if len(matched) < 5:
        print(f"  Too few spec-z matches ({len(matched)}) for validation plots")
        return

    z_phot = matched[z_phot_col].values
    z_spec = matched["z_spec"].values

    # Compute metrics
    metrics = compute_photoz_metrics(z_phot, z_spec)

    # Plot 1: z_phot vs z_spec scatter
    fig, ax = plt.subplots(figsize=(8, 8))

    # Compute delta_z_norm for coloring
    delta_z_norm = np.abs((z_phot - z_spec) / (1 + z_spec))
    is_outlier = delta_z_norm > 0.15

    # Plot non-outliers
    ax.scatter(
        z_spec[~is_outlier], z_phot[~is_outlier],
        s=20, alpha=0.6, c="steelblue", label=f"Good ({(~is_outlier).sum()})"
    )
    # Plot outliers
    ax.scatter(
        z_spec[is_outlier], z_phot[is_outlier],
        s=30, alpha=0.8, c="crimson", marker="x", label=f"Outliers ({is_outlier.sum()})"
    )

    # 1:1 line and outlier boundaries
    z_range = np.array([0, max(z_spec.max(), z_phot.max()) * 1.1])
    ax.plot(z_range, z_range, "k-", lw=1.5, label="1:1")
    ax.plot(z_range, z_range + 0.15 * (1 + z_range), "k--", lw=1, alpha=0.5)
    ax.plot(z_range, z_range - 0.15 * (1 + z_range), "k--", lw=1, alpha=0.5, label=r"$\pm$0.15(1+z)")

    ax.set_xlabel(r"$z_{\rm spec}$", fontsize=12)
    ax.set_ylabel(r"$z_{\rm phot}$", fontsize=12)
    ax.set_xlim(0, z_range[1])
    ax.set_ylim(0, z_range[1])
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=10)

    # Add metrics annotation
    metrics_text = (
        f"N = {metrics['n_matched']}\n"
        f"NMAD = {metrics['nmad']:.4f}\n"
        f"Bias = {metrics['bias']:.4f}\n"
        f"Outliers (>0.15) = {100*metrics['outlier_015']:.1f}%\n"
        f"Catastrophic (>0.3) = {100*metrics['outlier_030']:.1f}%"
    )
    ax.text(
        0.95, 0.05, metrics_text,
        transform=ax.transAxes, fontsize=10,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    ax.set_title("Photo-z vs Spec-z Comparison", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path / f"{prefix}_vs_specz.pdf", dpi=150)
    plt.close()
    print(f"  Saved {prefix}_vs_specz.pdf")

    # Plot 2: Residual histogram
    fig, ax = plt.subplots(figsize=(8, 6))

    residuals = (z_phot - z_spec) / (1 + z_spec)

    bins = np.linspace(-0.5, 0.5, 51)
    ax.hist(residuals, bins=bins, color="steelblue", alpha=0.7, edgecolor="black")

    # Mark outlier boundaries
    ax.axvline(-0.15, color="crimson", linestyle="--", lw=1.5, label=r"$\pm$0.15")
    ax.axvline(0.15, color="crimson", linestyle="--", lw=1.5)
    ax.axvline(0, color="black", linestyle="-", lw=1, alpha=0.5)

    ax.set_xlabel(r"$(z_{\rm phot} - z_{\rm spec}) / (1 + z_{\rm spec})$", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlim(-0.5, 0.5)
    ax.legend(loc="upper right", fontsize=10)

    # Add NMAD annotation
    ax.text(
        0.05, 0.95, f"NMAD = {metrics['nmad']:.4f}",
        transform=ax.transAxes, fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    ax.set_title("Photo-z Residual Distribution", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path / f"{prefix}_residuals.pdf", dpi=150)
    plt.close()
    print(f"  Saved {prefix}_residuals.pdf")

    # Plot 3: Outlier fraction vs redshift
    fig, ax = plt.subplots(figsize=(8, 6))

    # Bin by spec-z
    z_bins = np.arange(0, z_spec.max() + 0.5, 0.5)
    if len(z_bins) < 3:
        z_bins = np.linspace(0, z_spec.max() * 1.1, 5)

    z_centers = []
    outlier_fractions = []
    counts = []

    for i in range(len(z_bins) - 1):
        mask = (z_spec >= z_bins[i]) & (z_spec < z_bins[i + 1])
        if mask.sum() >= 5:  # Require at least 5 sources per bin
            z_centers.append((z_bins[i] + z_bins[i + 1]) / 2)
            outlier_fractions.append(is_outlier[mask].mean())
            counts.append(mask.sum())

    if z_centers:
        ax.bar(z_centers, outlier_fractions, width=0.4, color="steelblue", alpha=0.7, edgecolor="black")

        # Add count annotations
        for zc, frac, cnt in zip(z_centers, outlier_fractions, counts):
            ax.text(zc, frac + 0.02, f"n={cnt}", ha="center", fontsize=9)

    ax.axhline(0.15, color="crimson", linestyle="--", lw=1.5, label="15% threshold")

    ax.set_xlabel(r"$z_{\rm spec}$", fontsize=12)
    ax.set_ylabel("Outlier Fraction", fontsize=12)
    ax.set_ylim(0, min(1.0, max(outlier_fractions) * 1.5 if outlier_fractions else 0.5))
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Outlier Fraction vs Redshift", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path / f"{prefix}_outlier_vs_z.pdf", dpi=150)
    plt.close()
    print(f"  Saved {prefix}_outlier_vs_z.pdf")

    # Write summary text file
    summary_path = output_path / f"{prefix}_validation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Photo-z Validation Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write("Metrics:\n")
        f.write(f"  N matched: {metrics['n_matched']}\n")
        f.write(f"  NMAD: {metrics['nmad']:.4f}\n")
        f.write(f"  Bias: {metrics['bias']:.4f}\n")
        f.write(f"  Outlier rate (>0.15): {100*metrics['outlier_015']:.2f}%\n")
        f.write(f"  Catastrophic rate (>0.3): {100*metrics['outlier_030']:.2f}%\n\n")

        # Band count breakdown (if available)
        if "n_valid_bands" in matched.columns:
            f.write("Metrics by band count:\n")
            f.write("  (Sources with <4 bands have color-redshift degeneracy)\n")
            for n_bands in sorted(matched["n_valid_bands"].unique()):
                band_mask = matched["n_valid_bands"] == n_bands
                if band_mask.sum() >= 5:
                    band_metrics = compute_photoz_metrics(
                        matched.loc[band_mask, z_phot_col].values,
                        matched.loc[band_mask, "z_spec"].values
                    )
                    reliability = "reliable" if n_bands >= 4 else "degenerate"
                    f.write(f"  {int(n_bands)} bands ({reliability}): N={band_metrics['n_matched']}, "
                           f"NMAD={band_metrics['nmad']:.3f}, "
                           f"outlier={100*band_metrics['outlier_015']:.0f}%\n")
            f.write("\n")

        # Source breakdown
        if "z_spec_source" in matched.columns:
            f.write("Matches by source catalog:\n")
            for source in matched["z_spec_source"].unique():
                if source:
                    n = (matched["z_spec_source"] == source).sum()
                    f.write(f"  {source}: {n}\n")

        f.write("\n")
        f.write("Quality interpretation:\n")
        if metrics['nmad'] < 0.05:
            f.write("  NMAD < 0.05: Excellent photo-z precision\n")
        elif metrics['nmad'] < 0.10:
            f.write("  NMAD 0.05-0.10: Good photo-z precision\n")
        else:
            f.write("  NMAD > 0.10: Moderate photo-z precision\n")

        if metrics['outlier_015'] < 0.10:
            f.write("  Outlier rate < 10%: Excellent\n")
        elif metrics['outlier_015'] < 0.20:
            f.write("  Outlier rate 10-20%: Acceptable\n")
        else:
            f.write("  Outlier rate > 20%: High, consider flagging\n")

        # Add recommendation if many sources lack U-band
        if "n_valid_bands" in matched.columns:
            n_3band = (matched["n_valid_bands"] < 4).sum()
            pct_3band = 100 * n_3band / len(matched)
            if pct_3band > 30:
                f.write(f"\n  WARNING: {pct_3band:.0f}% of sources have <4 bands.\n")
                f.write("  Recommend filtering to photoz_reliable=True for science.\n")

    print(f"  Saved {prefix}_validation_summary.txt")


def run_full_validation(
    catalog: pd.DataFrame,
    output_dir: str,
    match_radius: float = 1.0,
    apply_corrections: bool = True,
    data_dir: str = "./data/external",
) -> pd.DataFrame:
    """Run the complete spec-z validation and improvement pipeline.

    This is a convenience function that runs all steps:
    1. Load spec-z catalogs
    2. Cross-match with input catalog
    3. Compute and report metrics
    4. Optionally apply spec-z corrections
    5. Generate validation plots

    Args:
        catalog: Input catalog with ra, dec, redshift columns
        output_dir: Directory for output plots
        match_radius: Cross-match radius in arcsec
        apply_corrections: If True, replace photo-z with spec-z
        data_dir: Directory containing external catalogs

    Returns:
        Updated catalog with validation columns
    """
    print("\n  === Spectroscopic Validation ===")

    # Load external catalogs
    specz_catalogs = load_specz_catalogs(data_dir)

    if not specz_catalogs:
        print("  No spectroscopic catalogs loaded, skipping validation")
        return catalog

    # Cross-match
    result = cross_match_with_specz(catalog, specz_catalogs, match_radius)

    # Compute metrics
    matched = result[result["z_spec"].notna()]
    if len(matched) >= 5:
        metrics = compute_photoz_metrics(matched["redshift"], matched["z_spec"])
        print(f"\n  Photo-z metrics:")
        print(f"    NMAD = {metrics['nmad']:.4f}")
        print(f"    Bias = {metrics['bias']:.4f}")
        print(f"    Outlier rate (>0.15) = {100*metrics['outlier_015']:.1f}%")
        print(f"    Catastrophic rate (>0.3) = {100*metrics['outlier_030']:.1f}%")

    # Apply corrections if requested
    if apply_corrections:
        result = apply_spectroscopic_redshifts(result)
        result = flag_catastrophic_outliers(result)

    # Generate plots
    generate_validation_plots(result, output_dir)

    return result
