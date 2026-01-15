#!/usr/bin/env python3
"""
Lightweight binning analysis - uses existing catalog to test binning strategies.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

from scientific import (  # noqa: E402
    choose_redshift_bin_edges,
    compute_chi2_stats,
    get_radius,
    get_radius_and_omega,
    theta_lcdm,
    theta_lcdm_flat,
    theta_static,
)

OUTPUT_DIR = "./output"
RAD_TO_ARCSEC = 180.0 * 3600.0 / np.pi


def get_adaptive_n_bins(n_sources: int, min_per_bin: int = 8) -> int:
    """Calculate adaptive number of bins based on data size.

    Balances two competing needs:
    - Statistical robustness: more sources per bin → better medians
    - Curve fitting: more bins → better resolution of θ(z) shape

    Parameters
    ----------
    n_sources : int
        Total number of sources in the sample.
    min_per_bin : int
        Absolute minimum sources per bin. Default 8.
        - 8-10: Minimum for meaningful median + error
        - 15-20: Good robustness
        - 25+: Ideal for large samples

    Returns
    -------
    int
        Recommended number of bins.

    Examples
    --------
    N=75  → 6 bins (~12 per bin) - enough for curve shape
    N=200 → 10 bins (~20 per bin) - good balance
    N=500 → 15 bins (~33 per bin) - robust statistics
    """
    # Adaptive target: more sources/bin for larger samples
    if n_sources < 100:
        target_per_bin = 10  # Small samples: accept 10 per bin
    elif n_sources < 300:
        target_per_bin = 15  # Medium samples
    elif n_sources < 1000:
        target_per_bin = 20  # Larger samples
    else:
        target_per_bin = 25  # Large samples: full robustness

    n_bins_target = n_sources // target_per_bin

    # Hard floor: need at least min_per_bin sources
    n_bins_max = n_sources // min_per_bin

    # Scale max bins with sample size
    if n_sources < 200:
        absolute_max = 12
    elif n_sources < 1000:
        absolute_max = 20
    elif n_sources < 5000:
        absolute_max = 35
    else:
        absolute_max = 50

    # Minimum 5 bins for curve fitting, max based on data
    n_bins = max(5, min(n_bins_target, n_bins_max, absolute_max))
    return n_bins


def standard_error_median(x):
    """Standard error of the median using analytical approximation.

    For approximately normal data: SE(median) ~ 1.2533 * sigma / sqrt(n)
    This is the standard approach used in professional astronomy surveys.
    Reference: Euclid Consortium methodology, Astropy robust statistics.
    """
    x = x.dropna() if hasattr(x, 'dropna') else x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    return 1.2533 * np.std(x, ddof=1) / np.sqrt(len(x))


def aggregate_bins(sed_catalog):
    """Aggregate binned data into summary statistics.

    Uses professional astronomy standards:
    - Median for central tendency (robust to outliers)
    - Standard error of the median for error bars (not RMS of individual errors)
    - 16th/84th percentiles for 1-sigma spread visualization

    References:
    - Euclid tomographic binning methodology
    - JWST angular size-redshift test (arXiv:2507.19651)
    - Astropy robust statistical estimators
    """
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
    """Equal-width bins in redshift space."""
    if n_bins is None:
        n_bins = get_adaptive_n_bins(len(sed_catalog))
    z_min, z_max = sed_catalog.redshift.min(), sed_catalog.redshift.max()
    bins = np.linspace(z_min, z_max, n_bins + 1)
    sed_catalog = sed_catalog.copy()
    sed_catalog["z_bin"] = pd.cut(sed_catalog.redshift, bins, include_lowest=True)
    return aggregate_bins(sed_catalog)


def bin_equal_count(sed_catalog, n_bins=None):
    """Equal-count bins (quantile-based)."""
    if n_bins is None:
        n_bins = get_adaptive_n_bins(len(sed_catalog))
    sed_catalog = sed_catalog.copy()
    sed_catalog["z_bin"] = pd.qcut(sed_catalog.redshift, q=n_bins, duplicates="drop")
    return aggregate_bins(sed_catalog)


def bin_percentile(sed_catalog, n_bins=None):
    """Percentile-based bins using quantile edges for robust statistics."""
    if n_bins is None:
        n_bins = get_adaptive_n_bins(len(sed_catalog))
    sed_catalog = sed_catalog.copy()
    bins = choose_redshift_bin_edges(
        sed_catalog.redshift.values,
        target_count_per_bin=len(sed_catalog) // n_bins,
        min_bins=max(3, n_bins - 2),
        max_bins=n_bins + 2,
        method="quantile",
    )
    sed_catalog["z_bin"] = pd.cut(sed_catalog.redshift, bins, include_lowest=True)
    return aggregate_bins(sed_catalog)


def main():
    print("=" * 60)
    print("DYNAMIC BINNING ANALYSIS")
    print("=" * 60)

    # Load existing catalog
    print("\nLoading existing catalog...")
    sed_catalog = pd.read_csv(f"{OUTPUT_DIR}/galaxy_catalog.csv")
    sed_catalog = sed_catalog.dropna(subset=["redshift", "r_half_arcsec", "r_half_arcsec_error"])
    sed_catalog = sed_catalog[np.isfinite(sed_catalog["r_half_arcsec_error"])]
    print(f"  Loaded {len(sed_catalog)} galaxies")

    # Filter out unreliable low-z photometric redshifts (z < 0.1 often contaminated by stars/errors)
    z_min_cut = 0.1
    n_before = len(sed_catalog)
    sed_catalog = sed_catalog[sed_catalog.redshift >= z_min_cut]
    print(f"  After z >= {z_min_cut} cut: {len(sed_catalog)} galaxies (removed {n_before - len(sed_catalog)} low-z)")

    # Calculate adaptive bins
    n_bins_adaptive = get_adaptive_n_bins(len(sed_catalog))
    print(f"\n  Adaptive binning: {n_bins_adaptive} bins for {len(sed_catalog)} galaxies")

    # Define strategies
    binning_strategies = {
        "Equal Count": bin_equal_count,
        "Equal Width": bin_equal_width,
        "Percentile": bin_percentile,
    }

    # Apply strategies
    binned_results = {}
    for name, bin_func in binning_strategies.items():
        try:
            binned_results[name] = bin_func(sed_catalog)
            print(f"\n  {name} binning ({len(binned_results[name])} bins):")
            print(binned_results[name][["z_mid", "theta_med", "theta_err", "n"]].to_string())
        except Exception as e:
            print(f"\n  {name} binning failed: {e}")

    # Use equal-count as primary
    binned = binned_results.get("Equal Count", bin_equal_width(sed_catalog))

    # Fit models
    z_min = max(1e-4, binned.z_mid.min())
    z_model = np.linspace(z_min, binned.z_mid.max(), 300)

    # Fit R with standard cosmology (fixed Omega_m = 0.3)
    R_static = get_radius(binned.z_mid.values, binned.theta_med.values, binned.theta_err.values, model="static")
    R_lcdm = get_radius(binned.z_mid.values, binned.theta_med.values, binned.theta_err.values, model="lcdm")

    # Also try joint fit of R and Omega_m (may hit boundaries if data doesn't constrain well)
    R_joint, Omega_m_joint = get_radius_and_omega(binned.z_mid.values, binned.theta_med.values, binned.theta_err.values)

    # Use fixed cosmology for main plots (more physically meaningful)
    Omega_m_std = 0.3
    Omega_L_std = 0.7

    # Compute goodness-of-fit statistics
    z_data = binned.z_mid.values
    theta_data = binned.theta_med.values
    theta_err = binned.theta_err.values

    # Model functions for chi2 calculation (return arcsec)
    def static_model_arcsec(z):
        return theta_static(z, R_static) * RAD_TO_ARCSEC

    def lcdm_model_arcsec(z):
        return theta_lcdm(z, R_lcdm, Omega_m_std, Omega_L_std) * RAD_TO_ARCSEC

    def lcdm_joint_model_arcsec(z):
        return theta_lcdm_flat(z, R_joint, Omega_m_joint) * RAD_TO_ARCSEC

    # Static model: 1 free parameter (R)
    stats_static = compute_chi2_stats(z_data, theta_data, theta_err, static_model_arcsec, n_params=1)

    # LCDM fixed Omega_m: 1 free parameter (R)
    stats_lcdm = compute_chi2_stats(z_data, theta_data, theta_err, lcdm_model_arcsec, n_params=1)

    # LCDM joint fit: 2 free parameters (R, Omega_m)
    stats_joint = compute_chi2_stats(z_data, theta_data, theta_err, lcdm_joint_model_arcsec, n_params=2)

    print("\n  Fitted parameters and goodness-of-fit:")
    print(f"    {'Model':<25} {'R (kpc)':<10} {'χ²/ndf':<12} {'P-value':<10}")
    print(f"    {'-'*57}")
    print(f"    {'Static (Euclidean)':<25} {R_static*1000:>7.2f}   "
          f"{stats_static['chi2_ndf']:>6.2f} ({stats_static['ndf']})   {stats_static['p_value']:.4f}")
    print(f"    {'ΛCDM (Ωm=0.3 fixed)':<25} {R_lcdm*1000:>7.2f}   "
          f"{stats_lcdm['chi2_ndf']:>6.2f} ({stats_lcdm['ndf']})   {stats_lcdm['p_value']:.4f}")
    print(f"    {'ΛCDM (Ωm={:.2f} fit)':<25} {R_joint*1000:>7.2f}   "
          f"{stats_joint['chi2_ndf']:>6.2f} ({stats_joint['ndf']})   {stats_joint['p_value']:.4f}".format(Omega_m_joint))

    # Interpretation
    print("\n  Interpretation:")
    print("    χ²/ndf ≈ 1: good fit | χ²/ndf >> 1: poor fit | χ²/ndf << 1: overfitting or overestimated errors")
    print("    P-value > 0.05: model consistent with data | P-value < 0.05: model may be rejected")

    if stats_lcdm['p_value'] > stats_static['p_value']:
        print("    → ΛCDM provides better fit (higher P-value)")
    else:
        print("    → Static model provides better fit (higher P-value)")

    theta_static_model = theta_static(z_model, R_static) * RAD_TO_ARCSEC
    theta_lcdm_model = theta_lcdm(z_model, R_lcdm, Omega_m_std, Omega_L_std) * RAD_TO_ARCSEC

    # Dynamic axis limits
    z_data_min, z_data_max = sed_catalog.redshift.min(), sed_catalog.redshift.max()
    theta_data_min, theta_data_max = sed_catalog.r_half_arcsec.min(), sed_catalog.r_half_arcsec.max()
    z_padding = (z_data_max - z_data_min) * 0.08
    theta_padding = (theta_data_max - theta_data_min) * 0.1
    z_lim = (max(0, z_data_min - z_padding), z_data_max + z_padding)
    theta_lim = (max(0, theta_data_min - theta_padding), theta_data_max + theta_padding)

    colors = {
        "Equal Count": "blue",
        "Equal Width": "green",
        "Percentile": "purple",
    }
    markers = {
        "Equal Count": "o",
        "Equal Width": "o",
        "Percentile": "o",
    }

    # Plot 1: Main angular size vs redshift
    print("\nCreating plots...")
    _fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(binned["z_mid"], binned["theta_med"], yerr=binned["theta_err"], fmt="o",
                capsize=4, capthick=1.5, markersize=10, color="black",
                label=f"Median angular size (N={len(sed_catalog)}, bins={len(binned)})")
    ax.plot(z_model, theta_static_model, "--", color="gray", linewidth=2, label=r"Linear Hubble law (Euclidean)")
    ax.plot(z_model, theta_lcdm_model, "-", color="blue", linewidth=2,
            label=r"$\Lambda$CDM ($\Omega_m=0.3$)")
    ax.set_xlabel(r"Redshift $z$", fontsize=12)
    ax.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=12)
    ax.set_title(r"Galaxy Angular Size vs Redshift (Dynamic Binning)", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(z_lim)
    ax.set_ylim(theta_lim)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/angular_size_vs_redshift.pdf", dpi=300, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR}/angular_size_vs_redshift.pdf")
    plt.close()

    # Plot 2: Individual galaxies
    _fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(sed_catalog["redshift"], sed_catalog["r_half_arcsec"],
               c=sed_catalog["galaxy_type"].astype("category").cat.codes, cmap="tab10", alpha=0.6, s=20)
    ax.plot(z_model, theta_static_model, "--", color="gray", linewidth=2, label=r"Linear Hubble law", alpha=0.8)
    ax.plot(z_model, theta_lcdm_model, "-", color="blue", linewidth=2, label=r"$\Lambda$CDM model", alpha=0.8)
    ax.set_xlabel(r"Redshift $z$", fontsize=12)
    ax.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=12)
    ax.set_title(r"Individual Galaxy Angular Sizes", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(z_lim)
    ax.set_ylim(theta_lim)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/individual_galaxies.pdf", dpi=300, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR}/individual_galaxies.pdf")
    plt.close()

    # Plot 3: Binning comparison (1x3 grid) with two residual subplots each (ΛCDM and Static)
    # Layout: 3 strategies in a 1x3 grid
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[3, 1, 1], hspace=0.08, wspace=0.15)

    strategy_names = list(binned_results.keys())

    for idx, name in enumerate(strategy_names):
        binned_data = binned_results[name]
        # Grid position: strategies in 1 row of 3, each with main + 2 residual plots
        col = idx

        ax_main = fig.add_subplot(gs[0, col])
        ax_resid_lcdm = fig.add_subplot(gs[1, col], sharex=ax_main)
        ax_resid_static = fig.add_subplot(gs[2, col], sharex=ax_main)

        # Fit R with fixed standard cosmology (Omega_m = 0.3)
        R_static_i = get_radius(binned_data.z_mid.values, binned_data.theta_med.values,
                                binned_data.theta_err.values, model="static")
        R_lcdm_i = get_radius(binned_data.z_mid.values, binned_data.theta_med.values,
                              binned_data.theta_err.values, model="lcdm")

        z_model_i = np.linspace(max(0.05, z_lim[0]), z_lim[1], 200)
        theta_static_i = theta_static(z_model_i, R_static_i) * RAD_TO_ARCSEC
        theta_lcdm_i = theta_lcdm(z_model_i, R_lcdm_i, 0.3, 0.7) * RAD_TO_ARCSEC

        # Compute chi2/ndf for this binning strategy
        def static_func_i(z, R=R_static_i):
            return theta_static(z, R) * RAD_TO_ARCSEC
        def lcdm_func_i(z, R=R_lcdm_i):
            return theta_lcdm(z, R, 0.3, 0.7) * RAD_TO_ARCSEC

        stats_static_i = compute_chi2_stats(
            binned_data.z_mid.values, binned_data.theta_med.values,
            binned_data.theta_err.values, static_func_i, n_params=1
        )
        stats_lcdm_i = compute_chi2_stats(
            binned_data.z_mid.values, binned_data.theta_med.values,
            binned_data.theta_err.values, lcdm_func_i, n_params=1
        )

        # Main plot
        ax_main.scatter(sed_catalog["redshift"], sed_catalog["r_half_arcsec"], c="lightgray", alpha=0.4, s=15, zorder=1)
        ax_main.errorbar(binned_data["z_mid"], binned_data["theta_med"], yerr=binned_data["theta_err"],
                    fmt=markers[name], capsize=3, capthick=0.8, markersize=6, markeredgewidth=0.8,
                    elinewidth=0.8, color=colors[name],
                    label="Binned", zorder=3)
        ax_main.plot(z_model_i, theta_static_i, "--", color="gray", linewidth=1.5, label=r"Static", zorder=2)
        ax_main.plot(z_model_i, theta_lcdm_i, "-", color="darkblue", linewidth=1.5, label=r"$\Lambda$CDM", zorder=2)

        ax_main.set_ylabel(r"Angular size $\theta$ (arcsec)", fontsize=11)
        ax_main.set_title(
            f"{name} ({len(binned_data)} bins)\n"
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

        # Calculate residuals: (data - model) in arcsec
        theta_lcdm_at_data = theta_lcdm(binned_data["z_mid"].values, R_lcdm_i, 0.3, 0.7) * RAD_TO_ARCSEC
        theta_static_at_data = theta_static(binned_data["z_mid"].values, R_static_i) * RAD_TO_ARCSEC

        residuals_lcdm = binned_data["theta_med"].values - theta_lcdm_at_data
        residuals_static = binned_data["theta_med"].values - theta_static_at_data

        # ΛCDM residual plot
        ax_resid_lcdm.axhline(0, color="darkblue", linestyle="-", linewidth=1.5, alpha=0.7)
        ax_resid_lcdm.errorbar(binned_data["z_mid"], residuals_lcdm, yerr=binned_data["theta_err"],
                               fmt=markers[name], capsize=2, capthick=0.6, markersize=5,
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
                                 fmt=markers[name], capsize=2, capthick=0.6, markersize=5,
                                 markeredgewidth=0.6, elinewidth=0.6, color=colors[name], zorder=3)
        ax_resid_static.set_xlabel(r"Redshift $z$", fontsize=11)
        ax_resid_static.set_ylabel(r"$\theta_{\rm data} - \theta_{\rm Static}$", fontsize=9)
        ax_resid_static.grid(True, alpha=0.3)
        ax_resid_static.set_xlim(z_lim)
        max_resid_static = max(0.05, np.max(np.abs(residuals_static) + binned_data["theta_err"].values) * 1.2)
        ax_resid_static.set_ylim(-max_resid_static, max_resid_static)

    fig.suptitle(r"Binning Strategies: $\Lambda$CDM vs Static Model Comparison", fontsize=14, fontweight="bold")
    plt.savefig(f"{OUTPUT_DIR}/binning_comparison.pdf", dpi=300, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR}/binning_comparison.pdf")
    plt.close()

    # Plot 4: All strategies overlaid
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
    ax.set_title(r"All Dynamic Binning Strategies", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(z_lim)
    ax.set_ylim(theta_lim)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/binning_overlay.pdf", dpi=300, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR}/binning_overlay.pdf")
    plt.close()

    # Plot 5: Redshift histogram with dynamic y-axis
    _fig, ax = plt.subplots(figsize=(8, 5))
    hist_bins = np.linspace(z_data_min, z_data_max, 15)
    counts, _, _ = ax.hist(sed_catalog["redshift"], bins=hist_bins, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel(r"Redshift $z$", fontsize=12)
    ax.set_ylabel(r"Number of galaxies", fontsize=12)
    ax.set_title(f"Redshift Distribution (N={len(sed_catalog)})", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(z_lim)
    ax.set_ylim(0, np.max(counts) * 1.1)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/redshift_histogram.pdf", dpi=300, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR}/redshift_histogram.pdf")
    plt.close()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return sed_catalog, binned_results


if __name__ == "__main__":
    sed_catalog, binned_results = main()
