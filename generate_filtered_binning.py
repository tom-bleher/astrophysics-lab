#!/usr/bin/env python3
"""
Generate binning comparison plots for angular size vs redshift analysis.
"""

import matplotlib

matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure matplotlib with robust font settings
# LaTeX disabled for compatibility (requires system latex installation)
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "axes.labelsize": 12,
    "font.size": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

import builtins
import contextlib

from astropy.stats import bayesian_blocks
from scipy.stats import chi2 as chi2_dist

from scientific import (
    get_radius,
    theta_lcdm,
    theta_static,
)
from validation.muse_specz import load_muse_catalog, print_validation_summary, validate_with_specz

RAD_TO_ARCSEC = 180.0 * 3600.0 / np.pi
ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
Z_MIN_CUT = 0.3  # Filter out low-z where selection effects dominate (PSF-limited sample)
Z_MAX_CUT = 2.0  # Filter out sparse high-z (too few galaxies for statistics)
MIN_GALAXIES_PER_TYPE = 30  # Minimum galaxies needed for per-type statistics (combined types)
MIN_GALAXIES_PER_TYPE_BINNING = 9  # Lower threshold for per-type binning comparison plots

# Combined galaxy type groups for better statistics
COMBINED_TYPE_GROUPS = {
    "low_dust_starburst": ["sbt1", "sbt2"],           # E(B-V) = 0.0-0.21
    "high_dust_starburst": ["sbt3", "sbt4", "sbt5", "sbt6"],  # E(B-V) = 0.25-0.70
}

# Physical size limits for realistic galaxy selection (kpc)
R_PHYS_MIN_KPC = 0.3   # Minimum half-light radius
R_PHYS_MAX_KPC = 15.0  # Maximum half-light radius (excludes obvious outliers)


def compute_chi2_stats(z_data, theta_data, theta_err, model_func, n_params=1):
    """Compute chi2/ndf and p-value for a model fit."""
    theta_model = model_func(z_data)
    residuals = (theta_data - theta_model) / theta_err
    chi2 = np.sum(residuals**2)
    ndf = len(z_data) - n_params
    if ndf <= 0:
        return {'chi2': chi2, 'ndf': ndf, 'chi2_ndf': np.nan, 'p_value': np.nan}
    chi2_ndf = chi2 / ndf
    p_value = 1.0 - chi2_dist.cdf(chi2, ndf)
    return {'chi2': chi2, 'ndf': ndf, 'chi2_ndf': chi2_ndf, 'p_value': p_value}


# =============================================================================
# Quality Filters and Robust Statistics
# =============================================================================

def compute_physical_size(z, theta_arcsec, Omega_m=0.3):
    """Compute physical half-light radius in kpc assuming LCDM cosmology."""
    from scientific import D_A_LCDM_vectorized
    z = np.atleast_1d(z)
    theta_arcsec = np.atleast_1d(theta_arcsec)
    D_A = D_A_LCDM_vectorized(z, Omega_m, 1.0 - Omega_m)  # Mpc
    R_kpc = theta_arcsec * ARCSEC_TO_RAD * D_A * 1000  # kpc
    return R_kpc


def apply_physical_size_filter(df, z_col='redshift', theta_col='r_half_arcsec',
                                r_min=R_PHYS_MIN_KPC, r_max=R_PHYS_MAX_KPC):
    """Filter galaxies to have realistic physical sizes.

    This removes obvious outliers where the inferred physical size is
    unrealistic for typical galaxies (e.g., < 0.3 kpc or > 15 kpc).

    Parameters
    ----------
    df : pd.DataFrame
        Galaxy catalog
    z_col, theta_col : str
        Column names for redshift and angular size
    r_min, r_max : float
        Minimum and maximum allowed physical radius in kpc

    Returns
    -------
    pd.DataFrame
        Filtered catalog with realistic physical sizes
    """
    df = df.copy()
    R_kpc = compute_physical_size(df[z_col].values, df[theta_col].values)
    df['R_phys_kpc'] = R_kpc

    n_before = len(df)
    valid_mask = (R_kpc >= r_min) & (R_kpc <= r_max) & np.isfinite(R_kpc)
    df_filtered = df[valid_mask].copy()
    n_after = len(df_filtered)

    if n_before > n_after:
        print(f"    Physical size filter: {n_before} -> {n_after} "
              f"(removed {n_before - n_after} outliers outside {r_min}-{r_max} kpc)")

    return df_filtered


def sigma_clip_by_redshift_bin(df, z_col='redshift', theta_col='r_half_arcsec',
                                sigma=2.5, n_bins=5):
    """Apply sigma-clipping within redshift bins for robust outlier rejection.

    Parameters
    ----------
    df : pd.DataFrame
        Galaxy catalog
    z_col, theta_col : str
        Column names
    sigma : float
        Number of standard deviations for clipping
    n_bins : int
        Number of redshift bins for local outlier detection

    Returns
    -------
    pd.DataFrame
        Catalog with outliers removed
    """
    from astropy.stats import sigma_clip

    df = df.copy()
    z_bins = pd.qcut(df[z_col], q=n_bins, duplicates='drop')

    keep_mask = np.ones(len(df), dtype=bool)

    for bin_label in z_bins.unique():
        bin_mask = z_bins == bin_label
        if bin_mask.sum() < 5:
            continue

        theta_in_bin = df.loc[bin_mask, theta_col].values
        clipped = sigma_clip(theta_in_bin, sigma=sigma, maxiters=3, masked=True)

        # Update keep_mask for this bin
        outlier_in_bin = clipped.mask
        keep_mask[bin_mask] = ~outlier_in_bin

    n_before = len(df)
    df_filtered = df[keep_mask].copy()
    n_after = len(df_filtered)

    if n_before > n_after:
        print(f"    Sigma-clipping ({sigma}σ): {n_before} -> {n_after} "
              f"(removed {n_before - n_after} outliers)")

    return df_filtered


def assign_combined_types(df, type_col='galaxy_type'):
    """Assign combined type groups to galaxies.

    Creates a new column 'combined_type' based on COMBINED_TYPE_GROUPS.

    Parameters
    ----------
    df : pd.DataFrame
        Galaxy catalog with galaxy_type column
    type_col : str
        Column name for original galaxy type

    Returns
    -------
    pd.DataFrame
        Catalog with added 'combined_type' column
    """
    df = df.copy()
    df['combined_type'] = 'other'

    for combined_name, member_types in COMBINED_TYPE_GROUPS.items():
        mask = df[type_col].isin(member_types)
        df.loc[mask, 'combined_type'] = combined_name

    return df


def get_adaptive_n_bins(n_sources: int, min_per_bin: int = 8) -> int:
    """Calculate adaptive number of bins based on data size."""
    if n_sources < 100:
        target_per_bin = 10
    elif n_sources < 300:
        target_per_bin = 15
    elif n_sources < 1000:
        target_per_bin = 20
    else:
        target_per_bin = 25
    n_bins_target = n_sources // target_per_bin
    n_bins_max = n_sources // min_per_bin
    if n_sources < 200:
        absolute_max = 12
    elif n_sources < 1000:
        absolute_max = 20
    elif n_sources < 5000:
        absolute_max = 35
    else:
        absolute_max = 50
    n_bins = max(5, min(n_bins_target, n_bins_max, absolute_max))
    return n_bins


def standard_error_median(x):
    """Standard error of the median using analytical approximation."""
    x = x.dropna() if hasattr(x, 'dropna') else x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    return 1.2533 * np.std(x, ddof=1) / np.sqrt(len(x))


def aggregate_bins(sed_catalog):
    """Aggregate binned data into summary statistics."""
    binned = (
        sed_catalog.groupby("z_bin", observed=False)
        .agg(
            z_mid=("redshift", "median"),
            theta_med=("r_half_arcsec", "median"),
            theta_err=("r_half_arcsec", standard_error_median),
            theta_lo=("r_half_arcsec", lambda x: np.percentile(x, 16) if len(x) >= 2 else np.nan),
            theta_hi=("r_half_arcsec", lambda x: np.percentile(x, 84) if len(x) >= 2 else np.nan),
            n=("redshift", "count"),
        )
        .reset_index(drop=True)
    )
    binned = binned.dropna(subset=["z_mid", "theta_med", "theta_err"])
    binned = binned[(binned.n > 0) & (binned.z_mid > 0)]
    return binned


def bin_equal_width(sed_catalog, n_bins=None):
    if n_bins is None:
        n_bins = get_adaptive_n_bins(len(sed_catalog))
    z_min, z_max = sed_catalog.redshift.min(), sed_catalog.redshift.max()
    bins = np.linspace(z_min, z_max, n_bins + 1)
    sed_catalog = sed_catalog.copy()
    sed_catalog["z_bin"] = pd.cut(sed_catalog.redshift, bins, include_lowest=True)
    return aggregate_bins(sed_catalog)


def bin_percentile(sed_catalog, n_bins=None):
    if n_bins is None:
        n_bins = get_adaptive_n_bins(len(sed_catalog))
    sed_catalog = sed_catalog.copy()
    sed_catalog["z_bin"] = pd.qcut(sed_catalog.redshift, q=n_bins, duplicates="drop")
    return aggregate_bins(sed_catalog)


def bin_bayesian_blocks(sed_catalog, p0=0.05):
    """Bayesian Blocks adaptive binning (Scargle et al. 2013).

    Uses astropy's bayesian_blocks algorithm to find optimal bin edges
    based on the data structure. This is data-driven binning that creates
    more bins where data is dense/variable and fewer where sparse/uniform.
    """
    sed_catalog = sed_catalog.copy()
    z = sed_catalog.redshift.values
    theta = sed_catalog.r_half_arcsec.values

    # Use point measures fitness with angular sizes as values
    sigma = np.median(np.abs(theta - np.median(theta))) * 1.4826
    sigma = max(sigma, 0.01 * np.median(theta))

    try:
        edges = bayesian_blocks(z, theta, sigma=sigma, fitness='measures', p0=p0)
        if len(edges) < 4:
            edges = np.linspace(z.min(), z.max(), 4)
        if len(edges) > 16:
            edges = bayesian_blocks(z, theta, sigma=sigma, fitness='measures', p0=0.2)
            if len(edges) > 16:
                edges = np.linspace(z.min(), z.max(), 12)
    except Exception:
        n_bins = get_adaptive_n_bins(len(sed_catalog))
        edges = np.linspace(z.min(), z.max(), n_bins + 1)

    sed_catalog["z_bin"] = pd.cut(sed_catalog.redshift, edges, include_lowest=True)
    return aggregate_bins(sed_catalog)


def generate_binning_plot(data_dir, output_path, dataset_name):
    """Generate binning comparison plot for a dataset.

    Parameters
    ----------
    data_dir : str
        Directory containing galaxy_catalog.csv
    output_path : str
        Path to save the output PDF
    dataset_name : str
        Name for plot title
    """
    print(f"\n{'='*60}")
    print(f"Generating binning plot for: {dataset_name}")
    print(f"{'='*60}")

    # Load catalog
    catalog_path = f"{data_dir}/galaxy_catalog.csv"
    print(f"\nLoading catalog from: {catalog_path}")
    sed_catalog = pd.read_csv(catalog_path)
    sed_catalog = sed_catalog.dropna(subset=["redshift", "r_half_arcsec", "r_half_arcsec_error"])
    sed_catalog = sed_catalog[np.isfinite(sed_catalog["r_half_arcsec_error"])]
    n_total = len(sed_catalog)
    print(f"  Loaded {n_total} galaxies")

    n_used = n_total

    if n_used < 10:
        print(f"  ERROR: Too few galaxies ({n_used}). Skipping.")
        return

    # Binning strategies (1x3 layout: Equal Width, Percentile, Bayesian Blocks)
    binning_strategies = {
        "Equal Width": bin_equal_width,
        "Percentile": bin_percentile,
        "Bayesian Blocks": bin_bayesian_blocks,
    }

    binned_results = {}
    for name, bin_func in binning_strategies.items():
        try:
            binned_results[name] = bin_func(sed_catalog)
            print(f"  {name}: {len(binned_results[name])} bins")
        except Exception as e:
            print(f"  {name} failed: {e}")

    # Axis limits
    z_data_min, z_data_max = sed_catalog.redshift.min(), sed_catalog.redshift.max()
    theta_data_min, theta_data_max = sed_catalog.r_half_arcsec.min(), sed_catalog.r_half_arcsec.max()
    z_padding = (z_data_max - z_data_min) * 0.08
    theta_padding = (theta_data_max - theta_data_min) * 0.1
    z_lim = (max(0, z_data_min - z_padding), z_data_max + z_padding)
    theta_lim = (max(0, theta_data_min - theta_padding), theta_data_max + theta_padding)

    colors = {
        "Equal Width": "green",
        "Percentile": "blue",
        "Bayesian Blocks": "purple",
    }
    markers = {
        "Equal Width": "o",
        "Percentile": "o",
        "Bayesian Blocks": "o",
    }

    # Create plot (1x3 grid with residuals)
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[3, 1, 1], hspace=0.08, wspace=0.15)

    strategy_names = list(binned_results.keys())

    for idx, name in enumerate(strategy_names):
        binned_data = binned_results[name]
        col = idx

        ax_main = fig.add_subplot(gs[0, col])
        ax_resid_lcdm = fig.add_subplot(gs[1, col], sharex=ax_main)
        ax_resid_static = fig.add_subplot(gs[2, col], sharex=ax_main)

        # Fit models
        R_static_i = get_radius(binned_data.z_mid.values, binned_data.theta_med.values,
                                binned_data.theta_err.values, model="static")
        R_lcdm_i = get_radius(binned_data.z_mid.values, binned_data.theta_med.values,
                              binned_data.theta_err.values, model="lcdm")

        z_model_i = np.linspace(max(0.05, z_lim[0]), z_lim[1], 200)
        theta_static_i = theta_static(z_model_i, R_static_i) * RAD_TO_ARCSEC
        theta_lcdm_i = theta_lcdm(z_model_i, R_lcdm_i, 0.3, 0.7) * RAD_TO_ARCSEC

        # Chi2 stats
        def lcdm_func_i(z, R=R_lcdm_i):
            return theta_lcdm(z, R, 0.3, 0.7) * RAD_TO_ARCSEC

        stats_lcdm_i = compute_chi2_stats(
            binned_data.z_mid.values, binned_data.theta_med.values,
            binned_data.theta_err.values, lcdm_func_i, n_params=1
        )

        # Main plot
        ax_main.scatter(sed_catalog["redshift"], sed_catalog["r_half_arcsec"],
                       c="lightgray", alpha=0.4, s=15, zorder=1)
        ax_main.errorbar(binned_data["z_mid"], binned_data["theta_med"], yerr=binned_data["theta_err"],
                        fmt=markers[name], capsize=3, capthick=0.8, markersize=6, markeredgewidth=0.8,
                        elinewidth=0.8, color=colors[name],
                        label="Binned", zorder=3)
        ax_main.plot(z_model_i, theta_static_i, "--", color="gray", linewidth=1.5, label=r"Static", zorder=2)
        ax_main.plot(z_model_i, theta_lcdm_i, "-", color="darkblue", linewidth=1.5, label=r"$\Lambda$CDM", zorder=2)

        ax_main.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=11)
        ax_main.set_title(
            f"{name} ({len(binned_data)} bins)\n"
            + r"$R_{\Lambda{\rm CDM}}$" + f"={R_lcdm_i*1000:.1f} kpc, "
            + r"$\chi^2/{\rm ndf}$" + f"={stats_lcdm_i['chi2_ndf']:.2f}, "
            + f"P={stats_lcdm_i['p_value']:.3f}",
            fontsize=11
        )
        ax_main.legend(fontsize=8, loc="upper right")
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xlim(z_lim)
        ax_main.set_ylim(theta_lim)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # Residuals
        theta_lcdm_at_data = theta_lcdm(binned_data["z_mid"].values, R_lcdm_i, 0.3, 0.7) * RAD_TO_ARCSEC
        theta_static_at_data = theta_static(binned_data["z_mid"].values, R_static_i) * RAD_TO_ARCSEC
        residuals_lcdm = binned_data["theta_med"].values - theta_lcdm_at_data
        residuals_static = binned_data["theta_med"].values - theta_static_at_data

        # LCDM residual
        ax_resid_lcdm.axhline(0, color="darkblue", linestyle="-", linewidth=1.5, alpha=0.7)
        ax_resid_lcdm.errorbar(binned_data["z_mid"], residuals_lcdm, yerr=binned_data["theta_err"],
                               fmt=markers[name], capsize=2, capthick=0.6, markersize=5,
                               markeredgewidth=0.6, elinewidth=0.6, color=colors[name], zorder=3)
        ax_resid_lcdm.set_ylabel(r"$\theta_{\rm bin} - \theta_{\rm fit,\Lambda CDM}$", fontsize=9)
        ax_resid_lcdm.grid(True, alpha=0.3)
        ax_resid_lcdm.set_xlim(z_lim)
        max_resid_lcdm = max(0.05, np.max(np.abs(residuals_lcdm) + binned_data["theta_err"].values) * 1.2)
        ax_resid_lcdm.set_ylim(-max_resid_lcdm, max_resid_lcdm)
        plt.setp(ax_resid_lcdm.get_xticklabels(), visible=False)

        # Static residual
        ax_resid_static.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax_resid_static.errorbar(binned_data["z_mid"], residuals_static, yerr=binned_data["theta_err"],
                                 fmt=markers[name], capsize=2, capthick=0.6, markersize=5,
                                 markeredgewidth=0.6, elinewidth=0.6, color=colors[name], zorder=3)
        ax_resid_static.set_xlabel(r"Redshift $z$", fontsize=11)
        ax_resid_static.set_ylabel(r"$\theta_{\rm bin} - \theta_{\rm fit,stat}$", fontsize=9)
        ax_resid_static.grid(True, alpha=0.3)
        ax_resid_static.set_xlim(z_lim)
        max_resid_static = max(0.05, np.max(np.abs(residuals_static) + binned_data["theta_err"].values) * 1.2)
        ax_resid_static.set_ylim(-max_resid_static, max_resid_static)

    fig.suptitle(f"Binning Strategies - {dataset_name}",
                 fontsize=14, fontweight="bold")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close()


def generate_overlay_plot(data_dir, output_path, dataset_name):
    """Generate binning overlay plot (all strategies on one plot)."""
    print(f"\n  Generating overlay plot for: {dataset_name}")

    # Load catalog
    catalog_path = f"{data_dir}/galaxy_catalog.csv"
    sed_catalog = pd.read_csv(catalog_path)
    sed_catalog = sed_catalog.dropna(subset=["redshift", "r_half_arcsec", "r_half_arcsec_error"])
    sed_catalog = sed_catalog[np.isfinite(sed_catalog["r_half_arcsec_error"])]

    if len(sed_catalog) < 10:
        print("    Too few galaxies. Skipping.")
        return

    # Binning strategies (1x3 layout: Equal Width, Percentile, Bayesian Blocks)
    binning_strategies = {
        "Equal Width": bin_equal_width,
        "Percentile": bin_percentile,
        "Bayesian Blocks": bin_bayesian_blocks,
    }

    binned_results = {}
    for name, bin_func in binning_strategies.items():
        with contextlib.suppress(builtins.BaseException):
            binned_results[name] = bin_func(sed_catalog)

    # Use percentile for model fitting
    binned = binned_results.get("Percentile", bin_equal_width(sed_catalog))
    R_lcdm = get_radius(binned.z_mid.values, binned.theta_med.values, binned.theta_err.values, model="lcdm")
    R_static = get_radius(binned.z_mid.values, binned.theta_med.values, binned.theta_err.values, model="static")

    z_model = np.linspace(max(0.05, sed_catalog.redshift.min()), sed_catalog.redshift.max(), 200)
    theta_lcdm_model = theta_lcdm(z_model, R_lcdm, 0.3, 0.7) * RAD_TO_ARCSEC
    theta_static_model = theta_static(z_model, R_static) * RAD_TO_ARCSEC

    # Axis limits
    z_lim = (max(0, sed_catalog.redshift.min() * 0.9), sed_catalog.redshift.max() * 1.05)
    theta_lim = (max(0, sed_catalog.r_half_arcsec.min() * 0.9), sed_catalog.r_half_arcsec.max() * 1.1)

    colors = {"Equal Width": "green", "Percentile": "blue", "Bayesian Blocks": "purple"}
    markers = {"Equal Width": "o", "Percentile": "o", "Bayesian Blocks": "o"}

    # Create plot
    _fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(sed_catalog["redshift"], sed_catalog["r_half_arcsec"], c="lightgray", alpha=0.3, s=15,
               label="Individual galaxies", zorder=1)
    for name, binned_data in binned_results.items():
        ax.errorbar(binned_data["z_mid"], binned_data["theta_med"], yerr=binned_data["theta_err"],
                    fmt=f"{markers[name]}-", capsize=2, capthick=0.6, markersize=5, markeredgewidth=0.6,
                    elinewidth=0.6, linewidth=1.0, color=colors[name],
                    label=f"{name} ({len(binned_data)} bins)", alpha=0.8, zorder=2)
    ax.plot(z_model, theta_lcdm_model, "-", color="black", linewidth=2, label=r"$\Lambda$CDM model", zorder=3)
    ax.plot(z_model, theta_static_model, "--", color="black", linewidth=2, label=r"Static model", zorder=3)

    ax.set_xlabel(r"Redshift $z$", fontsize=12)
    ax.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=12)
    ax.set_title(f"All Dynamic Binning Strategies - {dataset_name}", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(z_lim)
    ax.set_ylim(theta_lim)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"    Saved: {output_path}")
    plt.close()


def get_types_with_enough_statistics(sed_catalog, min_count=MIN_GALAXIES_PER_TYPE):
    """Get galaxy types that have enough sources for statistical analysis."""
    if 'galaxy_type' not in sed_catalog.columns:
        return []
    type_counts = sed_catalog['galaxy_type'].value_counts()
    valid_types = type_counts[type_counts >= min_count].index.tolist()
    return valid_types


def generate_angular_size_plot(data_dir, output_path, dataset_name):
    """Generate the main angular size vs redshift plot with binned medians.

    This is the key science plot showing θ(z) with ΛCDM and static model fits.

    Parameters
    ----------
    data_dir : str
        Directory containing galaxy_catalog.csv
    output_path : str
        Path to save the output PDF
    dataset_name : str
        Name for plot title (e.g., "Full Field", "Chip3")
    """
    print("\n  Generating angular size vs redshift plot...")

    # Load catalog
    catalog_path = f"{data_dir}/galaxy_catalog.csv"
    sed_catalog = pd.read_csv(catalog_path)
    sed_catalog = sed_catalog.dropna(subset=["redshift", "r_half_arcsec", "r_half_arcsec_error"])
    sed_catalog = sed_catalog[np.isfinite(sed_catalog["r_half_arcsec_error"])]
    n_used = len(sed_catalog)

    if n_used < 10:
        print(f"    ERROR: Too few galaxies ({n_used}). Skipping.")
        return

    # Use Bayesian Blocks binning for adaptive data-driven bins
    binned = bin_bayesian_blocks(sed_catalog)

    # Fit models
    R_lcdm = get_radius(binned.z_mid.values, binned.theta_med.values,
                        binned.theta_err.values, model="lcdm", Omega_m=0.3)
    R_lcdm_kpc = R_lcdm * 1000

    # Generate model curves
    z_model = np.linspace(max(0.01, sed_catalog.redshift.min() * 0.9),
                          sed_catalog.redshift.max() * 1.05, 300)
    theta_lcdm_model = theta_lcdm(z_model, R_lcdm, 0.3, 0.7) * RAD_TO_ARCSEC
    theta_static_model = theta_static(z_model, R_lcdm) * RAD_TO_ARCSEC

    # Compute chi2 stats for LCDM
    def lcdm_func(z):
        return theta_lcdm(z, R_lcdm, 0.3, 0.7) * RAD_TO_ARCSEC
    chi2_stats = compute_chi2_stats(binned.z_mid.values, binned.theta_med.values,
                                    binned.theta_err.values, lcdm_func, n_params=1)

    # Axis limits
    z_data_min, z_data_max = sed_catalog.redshift.min(), sed_catalog.redshift.max()
    theta_data_min = sed_catalog.r_half_arcsec.min()
    theta_data_max = sed_catalog.r_half_arcsec.max()
    z_padding = (z_data_max - z_data_min) * 0.08
    theta_padding = (theta_data_max - theta_data_min) * 0.12
    z_lim = (max(0, z_data_min - z_padding), z_data_max + z_padding)
    theta_lim = (max(0, theta_data_min - theta_padding), theta_data_max + theta_padding)

    # Create plot
    _fig, ax = plt.subplots(figsize=(10, 7))

    # Plot model curves
    ax.plot(z_model, theta_static_model, "--", color="gray", linewidth=2,
            label=r"Linear Hubble law (Euclidean)", zorder=1)
    ax.plot(z_model, theta_lcdm_model, "-", color="blue", linewidth=2,
            label=r"$\Lambda$CDM ($\Omega_m$=0.30)", zorder=2)

    # Plot binned data with error bars
    ax.errorbar(binned.z_mid, binned.theta_med, yerr=binned.theta_err,
                fmt="o", capsize=3, capthick=1, markersize=8, markeredgewidth=1,
                elinewidth=1.5, color="black", markerfacecolor="black",
                label=f"Median angular size (N={n_used})", zorder=3)

    ax.set_xlabel(r"Redshift $z$", fontsize=12)
    ax.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=12)

    # Title with filter info
    ax.set_title(f"Galaxy Angular Size vs Redshift\n({dataset_name})", fontsize=14)

    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(z_lim)
    ax.set_ylim(theta_lim)

    # Add fit statistics text box
    stats_text = (f"$R_{{\\Lambda CDM}}$ = {R_lcdm_kpc:.1f} kpc\n"
                  f"$\\chi^2$/ndf = {chi2_stats['chi2_ndf']:.1f}\n"
                  f"P = {chi2_stats['p_value']:.3f}")
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"    Saved: {output_path}")
    plt.close()


def generate_size_distribution_by_type(data_dir, output_dir, dataset_name):
    """Generate size distribution plots for each galaxy type with enough statistics."""
    print("\n  Generating size distribution by galaxy type...")

    # Load catalog
    catalog_path = f"{data_dir}/galaxy_catalog.csv"
    sed_catalog = pd.read_csv(catalog_path)
    sed_catalog = sed_catalog.dropna(subset=["redshift", "r_half_arcsec", "r_half_arcsec_error"])
    sed_catalog = sed_catalog[np.isfinite(sed_catalog["r_half_arcsec_error"])]

    if 'galaxy_type' not in sed_catalog.columns:
        print("    ERROR: No galaxy_type column in catalog")
        return

    # Apply z range filter
    sed_catalog = sed_catalog[(sed_catalog.redshift >= Z_MIN_CUT) & (sed_catalog.redshift <= Z_MAX_CUT)]
    n_total = len(sed_catalog)
    print(f"    Total galaxies after z cut: {n_total}")

    # Get types with enough statistics
    valid_types = get_types_with_enough_statistics(sed_catalog)
    print(f"    Types with N >= {MIN_GALAXIES_PER_TYPE}: {valid_types}")

    if len(valid_types) == 0:
        print("    No types have enough statistics. Skipping.")
        return

    # Color palette for types
    type_colors = {
        'Elliptical': 'red',
        'S0': 'orange',
        'Spiral': 'blue',
        'Sbc': 'cyan',
        'Scd': 'green',
        'Irregular': 'purple',
        'Starburst': 'magenta',
        'Unknown': 'gray',
        # SBT spectral types
        'sbt1': '#e41a1c',  # red
        'sbt2': '#377eb8',  # blue
        'sbt3': '#4daf4a',  # green
        'sbt4': '#984ea3',  # purple
        'sbt5': '#ff7f00',  # orange
        'sbt6': '#a65628',  # brown
        'sbt7': '#f781bf',  # pink
    }

    # Create combined histogram plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Angular size histogram by type
    ax1 = axes[0, 0]
    for gtype in valid_types:
        subset = sed_catalog[sed_catalog['galaxy_type'] == gtype]
        color = type_colors.get(gtype, 'gray')
        ax1.hist(subset['r_half_arcsec'], bins=30, alpha=0.5, label=f"{gtype} (N={len(subset)})",
                 color=color, density=True)
    ax1.set_xlabel(r"Angular size $\theta$ (arcsec)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("Angular Size Distribution by Type", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Box plot of angular sizes by type
    ax2 = axes[0, 1]
    data_for_box = [sed_catalog[sed_catalog['galaxy_type'] == gtype]['r_half_arcsec'].values
                    for gtype in valid_types]
    bp = ax2.boxplot(data_for_box, labels=valid_types, patch_artist=True)
    for patch, gtype in zip(bp['boxes'], valid_types, strict=False):
        patch.set_facecolor(type_colors.get(gtype, 'gray'))
        patch.set_alpha(0.6)
    ax2.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=11)
    ax2.set_title("Angular Size Distribution by Type", fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: Redshift histogram by type
    ax3 = axes[1, 0]
    for gtype in valid_types:
        subset = sed_catalog[sed_catalog['galaxy_type'] == gtype]
        color = type_colors.get(gtype, 'gray')
        ax3.hist(subset['redshift'], bins=30, alpha=0.5, label=f"{gtype} (N={len(subset)})",
                 color=color, density=True)
    ax3.set_xlabel(r"Redshift $z$", fontsize=11)
    ax3.set_ylabel("Density", fontsize=11)
    ax3.set_title("Redshift Distribution by Type", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Compute summary statistics
    summary_data = []
    for gtype in valid_types:
        subset = sed_catalog[sed_catalog['galaxy_type'] == gtype]
        # Compute R for each type
        binned = bin_percentile(subset)
        R_lcdm = get_radius(binned.z_mid.values, binned.theta_med.values, binned.theta_err.values, model="lcdm")

        def lcdm_func(z, R=R_lcdm):
            return theta_lcdm(z, R, 0.3, 0.7) * RAD_TO_ARCSEC
        stats = compute_chi2_stats(binned.z_mid.values, binned.theta_med.values,
                                   binned.theta_err.values, lcdm_func, n_params=1)

        summary_data.append([
            gtype,
            len(subset),
            f"{subset['r_half_arcsec'].median():.2f}",
            f"{subset['r_half_arcsec'].std():.2f}",
            f"{R_lcdm*1000:.1f}",
            f"{stats['chi2_ndf']:.2f}",
            f"{stats['p_value']:.3f}"
        ])

    col_labels = ['Type', 'N', r'$\tilde{\theta}$ (")', r'$\sigma_\theta$ (")', 'R (kpc)', r'$\chi^2$/ndf', 'P']
    table = ax4.table(cellText=summary_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title("Summary Statistics by Galaxy Type", fontsize=12, pad=20)

    fig.suptitle(f"Size Distributions by Galaxy Type - {dataset_name}\n"
                 + f"(${Z_MIN_CUT} \\leq z \\leq {Z_MAX_CUT}$)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = f"{output_dir}/size_distribution_by_type.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"    Saved: {output_path}")
    plt.close()


def generate_type_comparison_overlay(data_dir, output_dir, dataset_name):
    """Generate overlay plot comparing fits for different galaxy types."""
    print("\n  Generating type comparison overlay plot...")

    # Load catalog
    catalog_path = f"{data_dir}/galaxy_catalog.csv"
    sed_catalog = pd.read_csv(catalog_path)
    sed_catalog = sed_catalog.dropna(subset=["redshift", "r_half_arcsec", "r_half_arcsec_error"])
    sed_catalog = sed_catalog[np.isfinite(sed_catalog["r_half_arcsec_error"])]

    if 'galaxy_type' not in sed_catalog.columns:
        print("    ERROR: No galaxy_type column in catalog")
        return

    # Apply z range filter
    sed_catalog = sed_catalog[(sed_catalog.redshift >= Z_MIN_CUT) & (sed_catalog.redshift <= Z_MAX_CUT)]

    # Get types with enough statistics
    valid_types = get_types_with_enough_statistics(sed_catalog)

    if len(valid_types) == 0:
        print("    No types have enough statistics. Skipping.")
        return

    # Color palette for types
    type_colors = {
        'Elliptical': 'red',
        'S0': 'orange',
        'Spiral': 'blue',
        'Sbc': 'cyan',
        'Scd': 'green',
        'Irregular': 'purple',
        'Starburst': 'magenta',
        'Unknown': 'gray',
        # SBT spectral types
        'sbt1': '#e41a1c',  # red
        'sbt2': '#377eb8',  # blue
        'sbt3': '#4daf4a',  # green
        'sbt4': '#984ea3',  # purple
        'sbt5': '#ff7f00',  # orange
        'sbt6': '#a65628',  # brown
        'sbt7': '#f781bf',  # pink
    }

    # Create plot
    _fig, ax = plt.subplots(figsize=(12, 8))

    z_min_all = sed_catalog.redshift.min()
    z_max_all = sed_catalog.redshift.max()
    z_model = np.linspace(max(0.05, z_min_all), z_max_all, 200)

    # First compute and plot all type fits
    fit_results = {}
    for gtype in valid_types:
        subset = sed_catalog[sed_catalog['galaxy_type'] == gtype]
        color = type_colors.get(gtype, 'gray')

        # Scatter points
        ax.scatter(subset["redshift"], subset["r_half_arcsec"], c=color, alpha=0.3, s=15, zorder=1)

        # Bin and fit
        binned = bin_percentile(subset)
        R_lcdm = get_radius(binned.z_mid.values, binned.theta_med.values, binned.theta_err.values, model="lcdm")

        # Compute chi2 stats
        def lcdm_func(z, R=R_lcdm):
            return theta_lcdm(z, R, 0.3, 0.7) * RAD_TO_ARCSEC
        stats = compute_chi2_stats(binned.z_mid.values, binned.theta_med.values,
                                   binned.theta_err.values, lcdm_func, n_params=1)

        fit_results[gtype] = {'R': R_lcdm, 'stats': stats, 'N': len(subset)}

        # Plot fit curve
        theta_model = theta_lcdm(z_model, R_lcdm, 0.3, 0.7) * RAD_TO_ARCSEC
        ax.plot(z_model, theta_model, "-", color=color, linewidth=2.5,
                label=f"{gtype}: R={R_lcdm*1000:.1f} kpc, $\\chi^2$/ndf={stats['chi2_ndf']:.2f} (N={len(subset)})",
                zorder=3)

        # Plot binned data
        ax.errorbar(binned["z_mid"], binned["theta_med"], yerr=binned["theta_err"],
                    fmt='o', capsize=3, capthick=1, markersize=6, markeredgewidth=1,
                    elinewidth=1, color=color, zorder=4, alpha=0.8)

    ax.set_xlabel(r"Redshift $z$", fontsize=12)
    ax.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=12)
    ax.set_title(f"Per-Type $\\Lambda$CDM Fits Comparison - {dataset_name}\n"
                 + f"(${Z_MIN_CUT} \\leq z \\leq {Z_MAX_CUT}$)",
                 fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Set axis limits
    z_lim = (max(0, z_min_all * 0.9), z_max_all * 1.05)
    theta_lim = (max(0, sed_catalog.r_half_arcsec.min() * 0.9), sed_catalog.r_half_arcsec.max() * 1.1)
    ax.set_xlim(z_lim)
    ax.set_ylim(theta_lim)

    plt.tight_layout()

    output_path = f"{output_dir}/type_comparison_overlay.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"    Saved: {output_path}")
    plt.close()

    return fit_results


def generate_per_type_fit_plot(data_dir, output_dir, dataset_name, galaxy_type):
    """Generate individual θ(z) fit plot for a specific galaxy type.

    Creates a detailed plot showing all three binning strategies side-by-side
    (matching the style of binning_comparison.pdf):
    - 3 columns: Equal Width, Percentile, Bayesian Blocks
    - Each column has: main plot, ΛCDM residuals, Static residuals
    - Individual galaxies as scatter points
    - Binned medians with error bars
    - ΛCDM and Static model fits
    - Fit statistics (R, χ²/ndf, p-value)
    """
    print(f"    Generating fit plot for type: {galaxy_type}")

    # Load catalog
    catalog_path = f"{data_dir}/galaxy_catalog.csv"
    sed_catalog = pd.read_csv(catalog_path)
    sed_catalog = sed_catalog.dropna(subset=["redshift", "r_half_arcsec", "r_half_arcsec_error"])
    sed_catalog = sed_catalog[np.isfinite(sed_catalog["r_half_arcsec_error"])]

    if 'galaxy_type' not in sed_catalog.columns:
        print(f"      ERROR: No galaxy_type column. Skipping {galaxy_type}.")
        return None

    # Apply z range filter
    sed_catalog = sed_catalog[(sed_catalog.redshift >= Z_MIN_CUT) & (sed_catalog.redshift <= Z_MAX_CUT)]

    # Filter to specific type
    type_catalog = sed_catalog[sed_catalog['galaxy_type'] == galaxy_type]
    n_galaxies = len(type_catalog)

    if n_galaxies < MIN_GALAXIES_PER_TYPE:
        print(f"      Too few galaxies ({n_galaxies} < {MIN_GALAXIES_PER_TYPE}). Skipping.")
        return None

    # Binning strategies (same order as binning_comparison.pdf)
    binning_strategies = {
        "Equal Width": bin_equal_width,
        "Percentile": bin_percentile,
        "Bayesian Blocks": bin_bayesian_blocks,
    }

    # Apply all binning strategies
    binned_results = {}
    for name, bin_func in binning_strategies.items():
        try:
            binned_results[name] = bin_func(type_catalog)
        except Exception as e:
            print(f"      {name} binning failed for {galaxy_type}: {e}")

    if len(binned_results) == 0:
        print("      No binning strategies succeeded. Skipping.")
        return None

    # Check if any binning has enough bins
    valid_binnings = {k: v for k, v in binned_results.items() if len(v) >= 3}
    if len(valid_binnings) == 0:
        print("      Too few bins in all strategies. Skipping.")
        return None

    # Axis limits based on data
    z_data_min, z_data_max = type_catalog.redshift.min(), type_catalog.redshift.max()
    theta_data_min, theta_data_max = type_catalog.r_half_arcsec.min(), type_catalog.r_half_arcsec.max()
    z_padding = (z_data_max - z_data_min) * 0.08
    theta_padding = (theta_data_max - theta_data_min) * 0.1
    z_lim = (max(0, z_data_min - z_padding), z_data_max + z_padding)
    theta_lim = (max(0, theta_data_min - theta_padding), theta_data_max + theta_padding)

    # Colors for each binning strategy (same as binning_comparison.pdf)
    colors = {
        "Equal Width": "green",
        "Percentile": "blue",
        "Bayesian Blocks": "purple",
    }
    markers = {
        "Equal Width": "o",
        "Percentile": "o",
        "Bayesian Blocks": "o",
    }

    # Create figure (3x3 grid like binning_comparison.pdf)
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[3, 1, 1], hspace=0.08, wspace=0.15)

    strategy_names = list(binned_results.keys())
    fit_stats = {}

    for idx, name in enumerate(strategy_names):
        binned_data = binned_results[name]
        col = idx

        ax_main = fig.add_subplot(gs[0, col])
        ax_resid_lcdm = fig.add_subplot(gs[1, col], sharex=ax_main)
        ax_resid_static = fig.add_subplot(gs[2, col], sharex=ax_main)

        # Fit models
        R_static_i = get_radius(binned_data.z_mid.values, binned_data.theta_med.values,
                                binned_data.theta_err.values, model="static")
        R_lcdm_i = get_radius(binned_data.z_mid.values, binned_data.theta_med.values,
                              binned_data.theta_err.values, model="lcdm")

        z_model_i = np.linspace(max(0.05, z_lim[0]), z_lim[1], 200)
        theta_static_i = theta_static(z_model_i, R_static_i) * RAD_TO_ARCSEC
        theta_lcdm_i = theta_lcdm(z_model_i, R_lcdm_i, 0.3, 0.7) * RAD_TO_ARCSEC

        # Chi2 stats
        def lcdm_func_i(z, R=R_lcdm_i):
            return theta_lcdm(z, R, 0.3, 0.7) * RAD_TO_ARCSEC

        def static_func_i(z, R=R_static_i):
            return theta_static(z, R) * RAD_TO_ARCSEC

        stats_lcdm_i = compute_chi2_stats(
            binned_data.z_mid.values, binned_data.theta_med.values,
            binned_data.theta_err.values, lcdm_func_i, n_params=1
        )
        stats_static_i = compute_chi2_stats(
            binned_data.z_mid.values, binned_data.theta_med.values,
            binned_data.theta_err.values, static_func_i, n_params=1
        )

        # Store stats for return value (use Percentile as primary)
        if name == "Percentile":
            fit_stats = {
                'R_lcdm_kpc': R_lcdm_i * 1000,
                'R_static_kpc': R_static_i * 1000,
                'chi2_ndf_lcdm': stats_lcdm_i['chi2_ndf'],
                'p_value_lcdm': stats_lcdm_i['p_value'],
                'chi2_ndf_static': stats_static_i['chi2_ndf'],
                'p_value_static': stats_static_i['p_value'],
                'n_bins': len(binned_data),
            }

        # Main plot
        ax_main.scatter(type_catalog["redshift"], type_catalog["r_half_arcsec"],
                       c="lightgray", alpha=0.4, s=15, zorder=1)
        ax_main.errorbar(binned_data["z_mid"], binned_data["theta_med"], yerr=binned_data["theta_err"],
                        fmt=markers[name], capsize=3, capthick=0.8, markersize=6, markeredgewidth=0.8,
                        elinewidth=0.8, color=colors[name],
                        label="Binned", zorder=3)
        ax_main.plot(z_model_i, theta_static_i, "--", color="gray", linewidth=1.5, label=r"Static", zorder=2)
        ax_main.plot(z_model_i, theta_lcdm_i, "-", color="darkblue", linewidth=1.5, label=r"$\Lambda$CDM", zorder=2)

        ax_main.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=11)
        ax_main.set_title(
            f"{name} ({len(binned_data)} bins) - ${Z_MIN_CUT} \\leq z \\leq {Z_MAX_CUT}$\n"
            + r"$\Lambda$CDM: $R$" + f"={R_lcdm_i*1000:.1f} kpc, "
            + r"$\chi^2/{\rm ndf}$" + f"={stats_lcdm_i['chi2_ndf']:.2f}, "
            + f"P={stats_lcdm_i['p_value']:.3f}\n"
            + r"Static: $R$" + f"={R_static_i*1000:.1f} kpc, "
            + r"$\chi^2/{\rm ndf}$" + f"={stats_static_i['chi2_ndf']:.2f}, "
            + f"P={stats_static_i['p_value']:.3f}",
            fontsize=10
        )
        ax_main.legend(fontsize=8, loc="upper right")
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xlim(z_lim)
        ax_main.set_ylim(theta_lim)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # Calculate residuals
        theta_lcdm_at_data = theta_lcdm(binned_data["z_mid"].values, R_lcdm_i, 0.3, 0.7) * RAD_TO_ARCSEC
        theta_static_at_data = theta_static(binned_data["z_mid"].values, R_static_i) * RAD_TO_ARCSEC
        residuals_lcdm = binned_data["theta_med"].values - theta_lcdm_at_data
        residuals_static = binned_data["theta_med"].values - theta_static_at_data

        # LCDM residual plot
        ax_resid_lcdm.axhline(0, color="darkblue", linestyle="-", linewidth=1.5, alpha=0.7)
        ax_resid_lcdm.errorbar(binned_data["z_mid"], residuals_lcdm, yerr=binned_data["theta_err"],
                               fmt=markers[name], capsize=2, capthick=0.6, markersize=5,
                               markeredgewidth=0.6, elinewidth=0.6, color=colors[name], zorder=3)
        ax_resid_lcdm.set_ylabel(r"$\theta_{\rm bin} - \theta_{\rm fit,\Lambda CDM}$", fontsize=9)
        ax_resid_lcdm.grid(True, alpha=0.3)
        ax_resid_lcdm.set_xlim(z_lim)
        max_resid_lcdm = max(0.05, np.max(np.abs(residuals_lcdm) + binned_data["theta_err"].values) * 1.2)
        ax_resid_lcdm.set_ylim(-max_resid_lcdm, max_resid_lcdm)
        plt.setp(ax_resid_lcdm.get_xticklabels(), visible=False)

        # Static residual plot
        ax_resid_static.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax_resid_static.errorbar(binned_data["z_mid"], residuals_static, yerr=binned_data["theta_err"],
                                 fmt=markers[name], capsize=2, capthick=0.6, markersize=5,
                                 markeredgewidth=0.6, elinewidth=0.6, color=colors[name], zorder=3)
        ax_resid_static.set_xlabel(r"Redshift $z$", fontsize=11)
        ax_resid_static.set_ylabel(r"$\theta_{\rm bin} - \theta_{\rm fit,stat}$", fontsize=9)
        ax_resid_static.grid(True, alpha=0.3)
        ax_resid_static.set_xlim(z_lim)
        max_resid_static = max(0.05, np.max(np.abs(residuals_static) + binned_data["theta_err"].values) * 1.2)
        ax_resid_static.set_ylim(-max_resid_static, max_resid_static)

    fig.suptitle(f"Binning Strategies - {galaxy_type} ({dataset_name}, N={n_galaxies})",
                 fontsize=14, fontweight="bold")

    # Save with type name in filename (sanitize for filesystem)
    safe_type = galaxy_type.replace('/', '_').replace(' ', '_')
    output_path = f"{output_dir}/fit_{safe_type}.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"      Saved: {output_path}")
    plt.close()

    return {
        'type': galaxy_type,
        'N': n_galaxies,
        **fit_stats,
    }


def generate_per_type_binning_comparison(data_dir, output_dir, dataset_name, galaxy_type,
                                          min_galaxies=MIN_GALAXIES_PER_TYPE_BINNING):
    """Generate binning comparison plot for a specific galaxy type.

    Creates a plot matching the style of binning_comparison.pdf but filtered
    to a single galaxy type. Uses a lower threshold than the standard
    per-type analysis to allow plotting for types with fewer galaxies.

    Parameters
    ----------
    data_dir : str
        Directory containing galaxy_catalog.csv
    output_dir : str
        Directory to save output plot
    dataset_name : str
        Name for plot title
    galaxy_type : str
        Galaxy type to filter (e.g., 'sbt2', 'sbt6')
    min_galaxies : int
        Minimum number of galaxies required (default: 9)

    Returns
    -------
    dict or None
        Fit statistics or None if insufficient data
    """
    print(f"    Generating binning comparison for type: {galaxy_type}")

    # Load catalog
    catalog_path = f"{data_dir}/galaxy_catalog.csv"
    sed_catalog = pd.read_csv(catalog_path)
    sed_catalog = sed_catalog.dropna(subset=["redshift", "r_half_arcsec", "r_half_arcsec_error"])
    sed_catalog = sed_catalog[np.isfinite(sed_catalog["r_half_arcsec_error"])]

    if 'galaxy_type' not in sed_catalog.columns:
        print(f"      ERROR: No galaxy_type column. Skipping {galaxy_type}.")
        return None

    # Filter to specific type (no z range cut for binning comparison)
    type_catalog = sed_catalog[sed_catalog['galaxy_type'] == galaxy_type].copy()
    n_galaxies = len(type_catalog)

    if n_galaxies < min_galaxies:
        print(f"      Too few galaxies ({n_galaxies} < {min_galaxies}). Skipping.")
        return None

    # Binning strategies (same order as binning_comparison.pdf)
    binning_strategies = {
        "Equal Width": bin_equal_width,
        "Percentile": bin_percentile,
        "Bayesian Blocks": bin_bayesian_blocks,
    }

    # Apply all binning strategies
    binned_results = {}
    for name, bin_func in binning_strategies.items():
        try:
            binned_results[name] = bin_func(type_catalog)
            print(f"      {name}: {len(binned_results[name])} bins")
        except Exception as e:
            print(f"      {name} binning failed for {galaxy_type}: {e}")

    if len(binned_results) == 0:
        print("      No binning strategies succeeded. Skipping.")
        return None

    # Check if any binning has enough bins
    valid_binnings = {k: v for k, v in binned_results.items() if len(v) >= 3}
    if len(valid_binnings) == 0:
        print("      Too few bins in all strategies. Skipping.")
        return None

    # Axis limits based on data
    z_data_min, z_data_max = type_catalog.redshift.min(), type_catalog.redshift.max()
    theta_data_min, theta_data_max = type_catalog.r_half_arcsec.min(), type_catalog.r_half_arcsec.max()
    z_padding = (z_data_max - z_data_min) * 0.08
    theta_padding = (theta_data_max - theta_data_min) * 0.1
    z_lim = (max(0, z_data_min - z_padding), z_data_max + z_padding)
    theta_lim = (max(0, theta_data_min - theta_padding), theta_data_max + theta_padding)

    # Colors for each binning strategy (same as binning_comparison.pdf)
    colors = {
        "Equal Width": "green",
        "Percentile": "blue",
        "Bayesian Blocks": "purple",
    }

    # Create figure (3x3 grid like binning_comparison.pdf)
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[3, 1, 1], hspace=0.08, wspace=0.15)

    strategy_names = list(binned_results.keys())
    fit_stats = {}

    for idx, name in enumerate(strategy_names):
        binned_data = binned_results[name]
        col = idx

        ax_main = fig.add_subplot(gs[0, col])
        ax_resid_lcdm = fig.add_subplot(gs[1, col], sharex=ax_main)
        ax_resid_static = fig.add_subplot(gs[2, col], sharex=ax_main)

        # Fit models
        R_static_i = get_radius(binned_data.z_mid.values, binned_data.theta_med.values,
                                binned_data.theta_err.values, model="static")
        R_lcdm_i = get_radius(binned_data.z_mid.values, binned_data.theta_med.values,
                              binned_data.theta_err.values, model="lcdm")

        z_model_i = np.linspace(max(0.05, z_lim[0]), z_lim[1], 200)
        theta_static_i = theta_static(z_model_i, R_static_i) * RAD_TO_ARCSEC
        theta_lcdm_i = theta_lcdm(z_model_i, R_lcdm_i, 0.3, 0.7) * RAD_TO_ARCSEC

        # Chi2 stats
        def lcdm_func_i(z, R=R_lcdm_i):
            return theta_lcdm(z, R, 0.3, 0.7) * RAD_TO_ARCSEC

        stats_lcdm_i = compute_chi2_stats(
            binned_data.z_mid.values, binned_data.theta_med.values,
            binned_data.theta_err.values, lcdm_func_i, n_params=1
        )

        # Store stats for return value (use Percentile as primary)
        if name == "Percentile":
            fit_stats = {
                'R_lcdm_kpc': R_lcdm_i * 1000,
                'chi2_ndf_lcdm': stats_lcdm_i['chi2_ndf'],
                'p_value_lcdm': stats_lcdm_i['p_value'],
                'n_bins': len(binned_data),
            }

        # Main plot
        ax_main.scatter(type_catalog["redshift"], type_catalog["r_half_arcsec"],
                       c="lightgray", alpha=0.4, s=15, zorder=1)
        ax_main.errorbar(binned_data["z_mid"], binned_data["theta_med"], yerr=binned_data["theta_err"],
                        fmt='o', capsize=3, capthick=0.8, markersize=6, markeredgewidth=0.8,
                        elinewidth=0.8, color=colors[name],
                        label="Binned", zorder=3)
        ax_main.plot(z_model_i, theta_static_i, "--", color="gray", linewidth=1.5, label=r"Static", zorder=2)
        ax_main.plot(z_model_i, theta_lcdm_i, "-", color="darkblue", linewidth=1.5, label=r"$\Lambda$CDM", zorder=2)

        ax_main.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=11)
        ax_main.set_title(
            f"{name} ({len(binned_data)} bins)\n"
            + r"$R_{\Lambda{\rm CDM}}$" + f"={R_lcdm_i*1000:.1f} kpc, "
            + r"$\chi^2/{\rm ndf}$" + f"={stats_lcdm_i['chi2_ndf']:.2f}, "
            + f"P={stats_lcdm_i['p_value']:.3f}",
            fontsize=11
        )
        ax_main.legend(fontsize=8, loc="upper right")
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xlim(z_lim)
        ax_main.set_ylim(theta_lim)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # Calculate residuals
        theta_lcdm_at_data = theta_lcdm(binned_data["z_mid"].values, R_lcdm_i, 0.3, 0.7) * RAD_TO_ARCSEC
        theta_static_at_data = theta_static(binned_data["z_mid"].values, R_static_i) * RAD_TO_ARCSEC
        residuals_lcdm = binned_data["theta_med"].values - theta_lcdm_at_data
        residuals_static = binned_data["theta_med"].values - theta_static_at_data

        # LCDM residual plot
        ax_resid_lcdm.axhline(0, color="darkblue", linestyle="-", linewidth=1.5, alpha=0.7)
        ax_resid_lcdm.errorbar(binned_data["z_mid"], residuals_lcdm, yerr=binned_data["theta_err"],
                               fmt='o', capsize=2, capthick=0.6, markersize=5,
                               markeredgewidth=0.6, elinewidth=0.6, color=colors[name], zorder=3)
        ax_resid_lcdm.set_ylabel(r"$\theta_{\rm data} - \theta_{\Lambda{\rm CDM}}$", fontsize=9)
        ax_resid_lcdm.grid(True, alpha=0.3)
        ax_resid_lcdm.set_xlim(z_lim)
        max_resid_lcdm = max(0.05, np.max(np.abs(residuals_lcdm) + binned_data["theta_err"].values) * 1.2)
        ax_resid_lcdm.set_ylim(-max_resid_lcdm, max_resid_lcdm)
        plt.setp(ax_resid_lcdm.get_xticklabels(), visible=False)

        # Static residual plot
        ax_resid_static.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax_resid_static.errorbar(binned_data["z_mid"], residuals_static, yerr=binned_data["theta_err"],
                                 fmt='o', capsize=2, capthick=0.6, markersize=5,
                                 markeredgewidth=0.6, elinewidth=0.6, color=colors[name], zorder=3)
        ax_resid_static.set_xlabel(r"Redshift $z$", fontsize=11)
        ax_resid_static.set_ylabel(r"$\theta_{\rm data} - \theta_{\rm Static}$", fontsize=9)
        ax_resid_static.grid(True, alpha=0.3)
        ax_resid_static.set_xlim(z_lim)
        max_resid_static = max(0.05, np.max(np.abs(residuals_static) + binned_data["theta_err"].values) * 1.2)
        ax_resid_static.set_ylim(-max_resid_static, max_resid_static)

    fig.suptitle(f"Binning Strategies - {galaxy_type} ({dataset_name}, N={n_galaxies})",
                 fontsize=14, fontweight="bold")

    # Save with type name in filename (sanitize for filesystem)
    safe_type = galaxy_type.replace('/', '_').replace(' ', '_')
    output_path = f"{output_dir}/{safe_type}_binning_comparison.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"      Saved: {output_path}")
    plt.close()

    return {
        'type': galaxy_type,
        'N': n_galaxies,
        **fit_stats,
    }


def generate_combined_type_fit_plot(data_dir, output_dir, dataset_name, combined_type_name,
                                     apply_quality_filters=True):
    """Generate θ(z) fit plot for a combined galaxy type group.

    This combines multiple individual SBT types for better statistics
    and applies quality filters (physical size, sigma-clipping).

    Parameters
    ----------
    data_dir : str
        Directory containing galaxy_catalog.csv
    output_dir : str
        Directory to save output plot
    dataset_name : str
        Name for plot title
    combined_type_name : str
        Name of combined type group (e.g., 'low_dust_starburst')
    apply_quality_filters : bool
        If True, apply physical size filter and sigma-clipping

    Returns
    -------
    dict or None
        Fit statistics or None if insufficient data
    """
    if combined_type_name not in COMBINED_TYPE_GROUPS:
        print(f"    Unknown combined type: {combined_type_name}")
        return None

    member_types = COMBINED_TYPE_GROUPS[combined_type_name]
    print(f"    Generating combined fit plot for: {combined_type_name}")
    print(f"      Member types: {member_types}")

    # Load catalog
    catalog_path = f"{data_dir}/galaxy_catalog.csv"
    sed_catalog = pd.read_csv(catalog_path)
    sed_catalog = sed_catalog.dropna(subset=["redshift", "r_half_arcsec", "r_half_arcsec_error"])
    sed_catalog = sed_catalog[np.isfinite(sed_catalog["r_half_arcsec_error"])]

    if 'galaxy_type' not in sed_catalog.columns:
        print("      ERROR: No galaxy_type column. Skipping.")
        return None

    # Apply z range filter
    sed_catalog = sed_catalog[(sed_catalog.redshift >= Z_MIN_CUT) & (sed_catalog.redshift <= Z_MAX_CUT)]

    # Filter to combined type
    type_catalog = sed_catalog[sed_catalog['galaxy_type'].isin(member_types)].copy()
    n_before_filters = len(type_catalog)
    print(f"      Initial count: {n_before_filters}")

    if n_before_filters < MIN_GALAXIES_PER_TYPE:
        print(f"      Too few galaxies ({n_before_filters} < {MIN_GALAXIES_PER_TYPE}). Skipping.")
        return None

    # Apply quality filters
    if apply_quality_filters:
        # Physical size filter
        type_catalog = apply_physical_size_filter(type_catalog)

        # Sigma-clipping
        if len(type_catalog) >= 20:
            type_catalog = sigma_clip_by_redshift_bin(type_catalog, sigma=2.5, n_bins=5)

    n_galaxies = len(type_catalog)
    print(f"      After quality filters: {n_galaxies}")

    if n_galaxies < MIN_GALAXIES_PER_TYPE:
        print("      Too few galaxies after filtering. Skipping.")
        return None

    # Binning strategies
    binning_strategies = {
        "Equal Width": bin_equal_width,
        "Percentile": bin_percentile,
        "Bayesian Blocks": bin_bayesian_blocks,
    }

    binned_results = {}
    for name, bin_func in binning_strategies.items():
        try:
            binned_results[name] = bin_func(type_catalog)
        except Exception as e:
            print(f"      {name} binning failed: {e}")

    if len(binned_results) == 0:
        print("      No binning strategies succeeded. Skipping.")
        return None

    # Check if any binning has enough bins
    valid_binnings = {k: v for k, v in binned_results.items() if len(v) >= 3}
    if len(valid_binnings) == 0:
        print("      Too few bins in all strategies. Skipping.")
        return None

    # Axis limits based on data
    z_data_min, z_data_max = type_catalog.redshift.min(), type_catalog.redshift.max()
    theta_data_min, theta_data_max = type_catalog.r_half_arcsec.min(), type_catalog.r_half_arcsec.max()
    z_padding = (z_data_max - z_data_min) * 0.08
    theta_padding = (theta_data_max - theta_data_min) * 0.1
    z_lim = (max(0, z_data_min - z_padding), z_data_max + z_padding)
    theta_lim = (max(0, theta_data_min - theta_padding), theta_data_max + theta_padding)

    # Colors
    colors = {"Equal Width": "green", "Percentile": "blue", "Bayesian Blocks": "purple"}
    markers = {"Equal Width": "o", "Percentile": "o", "Bayesian Blocks": "o"}

    # Create figure
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[3, 1, 1], hspace=0.08, wspace=0.15)

    strategy_names = list(binned_results.keys())
    fit_stats = {}

    for idx, name in enumerate(strategy_names):
        binned_data = binned_results[name]
        col = idx

        ax_main = fig.add_subplot(gs[0, col])
        ax_resid_lcdm = fig.add_subplot(gs[1, col], sharex=ax_main)
        ax_resid_static = fig.add_subplot(gs[2, col], sharex=ax_main)

        # Fit models
        R_static_i = get_radius(binned_data.z_mid.values, binned_data.theta_med.values,
                                binned_data.theta_err.values, model="static")
        R_lcdm_i = get_radius(binned_data.z_mid.values, binned_data.theta_med.values,
                              binned_data.theta_err.values, model="lcdm")

        z_model_i = np.linspace(max(0.05, z_lim[0]), z_lim[1], 200)
        theta_static_i = theta_static(z_model_i, R_static_i) * RAD_TO_ARCSEC
        theta_lcdm_i = theta_lcdm(z_model_i, R_lcdm_i, 0.3, 0.7) * RAD_TO_ARCSEC

        # Chi2 stats
        def lcdm_func_i(z, R=R_lcdm_i):
            return theta_lcdm(z, R, 0.3, 0.7) * RAD_TO_ARCSEC

        def static_func_i(z, R=R_static_i):
            return theta_static(z, R) * RAD_TO_ARCSEC

        stats_lcdm_i = compute_chi2_stats(
            binned_data.z_mid.values, binned_data.theta_med.values,
            binned_data.theta_err.values, lcdm_func_i, n_params=1
        )
        stats_static_i = compute_chi2_stats(
            binned_data.z_mid.values, binned_data.theta_med.values,
            binned_data.theta_err.values, static_func_i, n_params=1
        )

        # Store stats
        if name == "Percentile":
            fit_stats = {
                'R_lcdm_kpc': R_lcdm_i * 1000,
                'R_static_kpc': R_static_i * 1000,
                'chi2_ndf_lcdm': stats_lcdm_i['chi2_ndf'],
                'p_value_lcdm': stats_lcdm_i['p_value'],
                'chi2_ndf_static': stats_static_i['chi2_ndf'],
                'p_value_static': stats_static_i['p_value'],
                'n_bins': len(binned_data),
            }

        # Main plot
        ax_main.scatter(type_catalog["redshift"], type_catalog["r_half_arcsec"],
                       c="lightgray", alpha=0.4, s=15, zorder=1)
        ax_main.errorbar(binned_data["z_mid"], binned_data["theta_med"], yerr=binned_data["theta_err"],
                        fmt=markers[name], capsize=3, capthick=0.8, markersize=6, markeredgewidth=0.8,
                        elinewidth=0.8, color=colors[name], label="Binned", zorder=3)
        ax_main.plot(z_model_i, theta_static_i, "--", color="gray", linewidth=1.5, label=r"Static", zorder=2)
        ax_main.plot(z_model_i, theta_lcdm_i, "-", color="darkblue", linewidth=1.5, label=r"$\Lambda$CDM", zorder=2)

        ax_main.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=11)
        filter_note = " [filtered]" if apply_quality_filters else ""
        ax_main.set_title(
            f"{name} ({len(binned_data)} bins) - ${Z_MIN_CUT} \\leq z \\leq {Z_MAX_CUT}${filter_note}\n"
            + r"$R_{\Lambda{\rm CDM}}$" + f"={R_lcdm_i*1000:.1f} kpc, "
            + r"$\chi^2/{\rm ndf}$" + f"={stats_lcdm_i['chi2_ndf']:.2f}, "
            + f"P={stats_lcdm_i['p_value']:.3f}",
            fontsize=11
        )
        ax_main.legend(fontsize=8, loc="upper right")
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xlim(z_lim)
        ax_main.set_ylim(theta_lim)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # Residuals
        theta_lcdm_at_data = theta_lcdm(binned_data["z_mid"].values, R_lcdm_i, 0.3, 0.7) * RAD_TO_ARCSEC
        theta_static_at_data = theta_static(binned_data["z_mid"].values, R_static_i) * RAD_TO_ARCSEC
        residuals_lcdm = binned_data["theta_med"].values - theta_lcdm_at_data
        residuals_static = binned_data["theta_med"].values - theta_static_at_data

        ax_resid_lcdm.axhline(0, color="darkblue", linestyle="-", linewidth=1.5, alpha=0.7)
        ax_resid_lcdm.errorbar(binned_data["z_mid"], residuals_lcdm, yerr=binned_data["theta_err"],
                               fmt=markers[name], capsize=2, capthick=0.6, markersize=5,
                               markeredgewidth=0.6, elinewidth=0.6, color=colors[name], zorder=3)
        ax_resid_lcdm.set_ylabel(r"$\theta_{\rm bin} - \theta_{\rm fit,\Lambda CDM}$", fontsize=9)
        ax_resid_lcdm.grid(True, alpha=0.3)
        ax_resid_lcdm.set_xlim(z_lim)
        max_resid_lcdm = max(0.05, np.max(np.abs(residuals_lcdm) + binned_data["theta_err"].values) * 1.2)
        ax_resid_lcdm.set_ylim(-max_resid_lcdm, max_resid_lcdm)
        plt.setp(ax_resid_lcdm.get_xticklabels(), visible=False)

        ax_resid_static.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax_resid_static.errorbar(binned_data["z_mid"], residuals_static, yerr=binned_data["theta_err"],
                                 fmt=markers[name], capsize=2, capthick=0.6, markersize=5,
                                 markeredgewidth=0.6, elinewidth=0.6, color=colors[name], zorder=3)
        ax_resid_static.set_xlabel(r"Redshift $z$", fontsize=11)
        ax_resid_static.set_ylabel(r"$\theta_{\rm bin} - \theta_{\rm fit,stat}$", fontsize=9)
        ax_resid_static.grid(True, alpha=0.3)
        ax_resid_static.set_xlim(z_lim)
        max_resid_static = max(0.05, np.max(np.abs(residuals_static) + binned_data["theta_err"].values) * 1.2)
        ax_resid_static.set_ylim(-max_resid_static, max_resid_static)

    # Title with member types
    member_str = " + ".join(member_types)
    quality_str = " (quality filtered)" if apply_quality_filters else ""
    fig.suptitle(f"Combined Type: {combined_type_name} ({member_str})\n"
                 f"{dataset_name}, N={n_galaxies}{quality_str}",
                 fontsize=14, fontweight="bold")

    # Save
    safe_name = combined_type_name.replace('/', '_').replace(' ', '_')
    output_path = f"{output_dir}/fit_combined_{safe_name}.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"      Saved: {output_path}")
    plt.close()

    return {
        'type': combined_type_name,
        'member_types': member_types,
        'N': n_galaxies,
        'N_before_filters': n_before_filters,
        **fit_stats,
    }


def generate_validation_report(data_dir, output_dir):
    """Generate a validation report comparing photo-z to external catalogs.

    Parameters
    ----------
    data_dir : str
        Directory containing galaxy_catalog.csv
    output_dir : str
        Directory to save output files
    """
    print("\n  === Generating Validation Report ===")

    catalog_path = f"{data_dir}/galaxy_catalog.csv"
    sed_catalog = pd.read_csv(catalog_path)
    sed_catalog = sed_catalog.dropna(subset=["redshift", "r_half_arcsec", "ra", "dec"])

    # MUSE validation
    try:
        print("  Attempting MUSE spectroscopic validation...")
        muse_catalog = load_muse_catalog(confidence_min=2)

        if len(muse_catalog) > 0:
            result = validate_with_specz(sed_catalog, muse_catalog, match_radius=1.0)
            print_validation_summary(result)

            # Save validation metrics
            if 'metrics' in result:
                metrics_path = f"{output_dir}/validation_muse_metrics.txt"
                with open(metrics_path, 'w') as f:
                    f.write("MUSE Spectroscopic Validation Results\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Matched sources: {result['n_matched']}\n")
                    f.write(f"Valid redshifts: {result['n_valid']}\n\n")
                    f.write("Metrics:\n")
                    f.writelines(f"  {key}: {val:.4f}\n" for key, val in result['metrics'].items())
                print(f"    Saved: {metrics_path}")
        else:
            print("    No MUSE catalog available")

    except Exception as e:
        print(f"    MUSE validation failed: {e}")

    # Try Fernandez-Soto validation
    fs_path = f"{os.path.dirname(data_dir)}/../../data/external/fernandez_soto_1999.csv"
    fs_path = os.path.normpath(fs_path)

    if os.path.exists(fs_path):
        print("  Loading Fernandez-Soto 1999 catalog...")
        try:
            fs_catalog = pd.read_csv(fs_path)
            print(f"    Loaded {len(fs_catalog)} sources from Fernandez-Soto 1999")

            # This catalog has RA/Dec in sexagesimal format, would need conversion
            # For now, just report availability
            print("    External validation with Fernandez-Soto catalog available")
            print("    (Full cross-match requires coordinate conversion)")

        except Exception as e:
            print(f"    Failed to load Fernandez-Soto catalog: {e}")
    else:
        print(f"    Fernandez-Soto catalog not found at {fs_path}")


def generate_redshift_histogram(data_dir, output_path, dataset_name):
    """Generate redshift distribution histogram."""
    print("  Generating redshift histogram...")

    # Load catalog
    catalog_path = f"{data_dir}/galaxy_catalog.csv"
    sed_catalog = pd.read_csv(catalog_path)
    sed_catalog = sed_catalog.dropna(subset=["redshift", "r_half_arcsec"])
    n_used = len(sed_catalog)
    if n_used < 5:
        print(f"    Too few galaxies ({n_used}). Skipping.")
        return

    _fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    n_bins = min(50, max(15, n_used // 10))
    ax.hist(sed_catalog['redshift'], bins=n_bins,
            color='steelblue', edgecolor='black',
            linewidth=0.5, alpha=0.7)

    # Statistics
    z_median = sed_catalog['redshift'].median()
    z_mean = sed_catalog['redshift'].mean()
    z_std = sed_catalog['redshift'].std()

    # Add vertical lines for statistics
    ax.axvline(z_median, color='red', linestyle='--', linewidth=2, label=f'Median: {z_median:.3f}')
    ax.axvline(z_mean, color='orange', linestyle=':', linewidth=2, label=f'Mean: {z_mean:.3f}')

    ax.set_xlabel(r"Redshift $z$", fontsize=12)
    ax.set_ylabel("Number of Galaxies", fontsize=12)

    ax.set_title(f"Redshift Distribution - {dataset_name}\n"
                 f"(N = {n_used})", fontsize=14)

    # Add statistics text box
    stats_text = (f"N = {n_used}\n"
                  f"Median = {z_median:.3f}\n"
                  f"Mean = {z_mean:.3f}\n"
                  f"$\\sigma$ = {z_std:.3f}")
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"    Saved: {output_path}")
    plt.close()


def generate_galaxy_type_histogram(data_dir, output_path, dataset_name):
    """Generate galaxy type distribution histogram."""
    print("  Generating galaxy type histogram...")

    # Load catalog
    catalog_path = f"{data_dir}/galaxy_catalog.csv"
    sed_catalog = pd.read_csv(catalog_path)
    sed_catalog = sed_catalog.dropna(subset=["redshift", "r_half_arcsec"])

    if 'galaxy_type' not in sed_catalog.columns:
        print("    ERROR: No galaxy_type column. Skipping.")
        return

    n_used = len(sed_catalog)
    if n_used < 5:
        print(f"    Too few galaxies ({n_used}). Skipping.")
        return

    # Count by type
    type_counts = sed_catalog['galaxy_type'].value_counts().sort_index()

    # Color palette
    type_colors = {
        'Ell': '#e41a1c',      # red - Elliptical
        'S0': '#ff7f00',       # orange
        'Sa': '#377eb8',       # blue
        'Sb': '#4daf4a',       # green
        'Sbc': '#984ea3',      # purple
        'Scd': '#00CED1',      # dark cyan
        'Sdm': '#a65628',      # brown
        'sbt1': '#e41a1c',     # red (starburst templates)
        'sbt2': '#377eb8',
        'sbt3': '#4daf4a',
        'sbt4': '#984ea3',
        'sbt5': '#ff7f00',
        'sbt6': '#a65628',
    }

    _fig, ax = plt.subplots(figsize=(12, 6))

    colors = [type_colors.get(t, 'gray') for t in type_counts.index]
    bars = ax.bar(type_counts.index, type_counts.values, color=colors,
                  edgecolor='black', linewidth=0.5, alpha=0.8)

    # Add count labels on bars
    for bar, count in zip(bars, type_counts.values):
        height = bar.get_height()
        ax.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel("Galaxy Type", fontsize=12)
    ax.set_ylabel("Number of Galaxies", fontsize=12)

    ax.set_title(f"Galaxy Type Distribution - {dataset_name}\n"
                 f"(N = {n_used})", fontsize=14)

    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"    Saved: {output_path}")
    plt.close()


def generate_final_sources_plot(data_dir, output_path, dataset_name):
    """Generate visualization of final sources used in analysis."""
    print("  Generating final sources plot...")

    # Load catalog
    catalog_path = f"{data_dir}/galaxy_catalog.csv"
    sed_catalog = pd.read_csv(catalog_path)
    sed_catalog = sed_catalog.dropna(subset=["redshift", "r_half_arcsec"])
    n_used = len(sed_catalog)
    if n_used < 5:
        print(f"    Too few galaxies ({n_used}). Skipping.")
        return

    # Create 2x2 summary plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Color by galaxy type if available
    if 'galaxy_type' in sed_catalog.columns:
        type_colors = {
            'Ell': '#e41a1c', 'S0': '#ff7f00', 'Sa': '#377eb8', 'Sb': '#4daf4a',
            'Sbc': '#984ea3', 'Scd': '#00CED1', 'Sdm': '#a65628',
            'sbt1': '#e41a1c', 'sbt2': '#377eb8', 'sbt3': '#4daf4a',
            'sbt4': '#984ea3', 'sbt5': '#ff7f00', 'sbt6': '#a65628',
        }
        colors = [type_colors.get(t, 'gray') for t in sed_catalog['galaxy_type']]
    else:
        colors = 'steelblue'

    # Plot 1: Spatial distribution (x, y coordinates if available)
    ax1 = axes[0, 0]
    if 'x' in sed_catalog.columns and 'y' in sed_catalog.columns:
        ax1.scatter(sed_catalog['x'], sed_catalog['y'], c=colors, alpha=0.5, s=10)
        ax1.set_xlabel("X (pixels)", fontsize=11)
        ax1.set_ylabel("Y (pixels)", fontsize=11)
        ax1.set_title(f"Spatial Distribution (N = {n_used})", fontsize=12)
    else:
        # If no x,y, show redshift vs angular size
        ax1.scatter(sed_catalog['redshift'], sed_catalog['r_half_arcsec'], c=colors, alpha=0.5, s=15)
        ax1.set_xlabel(r"Redshift $z$", fontsize=11)
        ax1.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=11)
        ax1.set_title(f"Redshift vs Angular Size (N = {n_used})", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Redshift vs Angular Size (color by type)
    ax2 = axes[0, 1]
    ax2.scatter(sed_catalog['redshift'], sed_catalog['r_half_arcsec'],
                c=colors, alpha=0.6, s=20)
    ax2.set_xlabel(r"Redshift $z$", fontsize=11)
    ax2.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=11)
    ax2.set_title("Redshift vs Angular Size", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Photo-z quality (ODDS distribution if available)
    ax3 = axes[1, 0]
    if 'photo_z_odds' in sed_catalog.columns:
        odds = sed_catalog['photo_z_odds'].dropna()
        ax3.hist(odds, bins=30, color='steelblue', edgecolor='black',
                 linewidth=0.5, alpha=0.7)
        ax3.axvline(0.9, color='green', linestyle='--', linewidth=2,
                    label='Excellent (0.9)')
        ax3.axvline(0.6, color='orange', linestyle='--', linewidth=2,
                    label='Good (0.6)')
        ax3.set_xlabel("Photo-z ODDS", fontsize=11)
        ax3.set_ylabel("Count", fontsize=11)
        ax3.set_title(f"Photo-z Quality Distribution\n(median ODDS = {odds.median():.3f})", fontsize=12)
        ax3.legend(fontsize=9)
    elif 'chi_sq_min' in sed_catalog.columns:
        chi2 = sed_catalog['chi_sq_min'].dropna()
        chi2_clipped = chi2[chi2 < chi2.quantile(0.99)]  # Clip outliers
        ax3.hist(chi2_clipped, bins=30, color='steelblue', edgecolor='black',
                 linewidth=0.5, alpha=0.7)
        ax3.set_xlabel(r"$\chi^2_{\rm min}$", fontsize=11)
        ax3.set_ylabel("Count", fontsize=11)
        ax3.set_title(f"SED Fit Quality\n(median $\\chi^2$ = {chi2.median():.2f})", fontsize=12)
    else:
        ax3.text(0.5, 0.5, "No quality metrics available", transform=ax3.transAxes,
                 ha='center', va='center', fontsize=12)
        ax3.set_title("Photo-z Quality", fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Compute summary statistics
    stats_data = [
        ["Total sources (after quality cuts)", f"{n_used}"],
        ["Sources in z range", f"{n_used}"],
        ["Redshift range", f"{sed_catalog['redshift'].min():.3f} - {sed_catalog['redshift'].max():.3f}"],
        ["Median redshift", f"{sed_catalog['redshift'].median():.3f}"],
        ["Median angular size", f"{sed_catalog['r_half_arcsec'].median():.3f} arcsec"],
    ]

    if 'galaxy_type' in sed_catalog.columns:
        n_types = sed_catalog['galaxy_type'].nunique()
        most_common = sed_catalog['galaxy_type'].value_counts().index[0]
        stats_data.append(["Number of galaxy types", f"{n_types}"])
        stats_data.append(["Most common type", f"{most_common}"])

    if 'photo_z_odds' in sed_catalog.columns:
        odds_median = sed_catalog['photo_z_odds'].median()
        odds_excellent = (sed_catalog['photo_z_odds'] >= 0.9).sum()
        stats_data.append(["Median ODDS", f"{odds_median:.3f}"])
        stats_data.append(["Excellent ODDS (>=0.9)", f"{odds_excellent} ({100*odds_excellent/n_used:.1f}%)"])

    table = ax4.table(cellText=stats_data, colLabels=['Metric', 'Value'],
                      loc='center', cellLoc='left', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    fig.suptitle(f"Final Source Summary - {dataset_name}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"    Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    base_dir = "/home/tomble/astrophysics-lab/output"

    for dataset, data_dir in [
        ("Full Field", f"{base_dir}/full_HDF"),
        ("Chip3", f"{base_dir}/chip3_HDF")
    ]:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset}")
        print(f"{'='*60}")

        # Check if catalog exists
        catalog_path = f"{data_dir}/galaxy_catalog.csv"
        if not os.path.exists(catalog_path):
            print(f"  WARNING: Catalog not found at {catalog_path}. Skipping {dataset}.")
            continue

        # ==================================================================
        # PART 1: All galaxy types combined
        # ==================================================================
        print("\n--- All Galaxy Types ---")

        # Binning comparison plots (all types)
        generate_binning_plot(data_dir=data_dir,
            output_path=f"{data_dir}/binning_comparison.pdf",
            dataset_name=dataset)

        # Overlay plots (all types)
        generate_overlay_plot(data_dir=data_dir,
            output_path=f"{data_dir}/binning_overlay.pdf",
            dataset_name=dataset)

        # Redshift distribution histograms
        generate_redshift_histogram(data_dir=data_dir,
            output_path=f"{data_dir}/redshift_histogram.pdf",
            dataset_name=dataset)

        # Galaxy type histograms
        generate_galaxy_type_histogram(data_dir=data_dir,
            output_path=f"{data_dir}/galaxy_types.pdf",
            dataset_name=dataset)

        # ==================================================================
        # PART 2: Per-galaxy-type analysis
        # ==================================================================

        # Load catalog to get valid types
        sed_catalog = pd.read_csv(catalog_path)
        sed_catalog = sed_catalog.dropna(subset=["redshift", "r_half_arcsec", "r_half_arcsec_error"])
        sed_catalog = sed_catalog[np.isfinite(sed_catalog["r_half_arcsec_error"])]

        print(f"\n--- Per-Galaxy-Type Analysis ({dataset}) ---")
        valid_types = get_types_with_enough_statistics(sed_catalog)
        print(f"  Galaxy types with N >= {MIN_GALAXIES_PER_TYPE}: {valid_types}")
        if len(valid_types) > 0:
            generate_size_distribution_by_type(data_dir, data_dir, dataset)
            generate_type_comparison_overlay(data_dir, data_dir, dataset)
            print(f"  Generating individual θ(z) fits for {len(valid_types)} types...")
            for gtype in valid_types:
                generate_per_type_fit_plot(data_dir, data_dir, dataset, gtype)
        else:
            print("  No galaxy types have enough statistics for per-type analysis.")

        # ==================================================================
        # Per-type binning comparison plots (lower threshold)
        # ==================================================================
        print(f"\n--- Per-Type Binning Comparison Plots ({dataset}) ---")
        types_for_binning = get_types_with_enough_statistics(sed_catalog, min_count=MIN_GALAXIES_PER_TYPE_BINNING)
        print(f"  Galaxy types with N >= {MIN_GALAXIES_PER_TYPE_BINNING}: {types_for_binning}")
        if len(types_for_binning) > 0:
            binning_results = {}
            for gtype in types_for_binning:
                result = generate_per_type_binning_comparison(data_dir, data_dir, dataset, gtype)
                if result:
                    binning_results[gtype] = result

            # Print summary
            if binning_results:
                print("\n  === Per-Type Binning Comparison Summary ===")
                print(f"  {'Type':<15} {'N':>5} {'R_LCDM':>10} {'χ²/ndf':>10} {'P-value':>10}")
                print("  " + "-" * 55)
                for gtype, res in binning_results.items():
                    r_kpc = res.get('R_lcdm_kpc', np.nan)
                    chi2 = res.get('chi2_ndf_lcdm', np.nan)
                    pval = res.get('p_value_lcdm', np.nan)
                    n = res.get('N', 0)
                    print(f"  {gtype:<15} {n:>5} {r_kpc:>10.1f} {chi2:>10.2f} {pval:>10.3f}")
        else:
            print(f"  No galaxy types have N >= {MIN_GALAXIES_PER_TYPE_BINNING} for binning comparison.")

        # ==================================================================
        # PART 3: Combined galaxy type analysis
        # ==================================================================
        print(f"\n--- Combined Galaxy Type Analysis ({dataset}) ---")
        print("  Combining galaxy types for better statistics with quality filters...")

        combined_results = {}
        for combined_name in COMBINED_TYPE_GROUPS:
            result = generate_combined_type_fit_plot(
                data_dir=data_dir,
                output_dir=data_dir,
                dataset_name=dataset,
                combined_type_name=combined_name,
                apply_quality_filters=True
            )
            if result:
                combined_results[combined_name] = result

        # Print summary of combined type results
        if combined_results:
            print("\n  === Combined Type Fit Summary ===")
            print(f"  {'Type':<25} {'N':>5} {'R_LCDM':>10} {'χ²/ndf':>10} {'P-value':>10}")
            print("  " + "-" * 65)
            for name, res in combined_results.items():
                r_kpc = res.get('R_lcdm_kpc', np.nan)
                chi2 = res.get('chi2_ndf_lcdm', np.nan)
                pval = res.get('p_value_lcdm', np.nan)
                n = res.get('N', 0)
                print(f"  {name:<25} {n:>5} {r_kpc:>10.1f} {chi2:>10.2f} {pval:>10.3f}")

        # ==================================================================
        # PART 4: Validation against external catalogs
        # ==================================================================
        generate_validation_report(data_dir, data_dir)

    print("\n" + "="*60)
    print("DONE - All plots generated (individual + combined + validation)")
    print("="*60)
