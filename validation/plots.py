"""Visualization tools for photo-z validation.

This module provides publication-quality plots for:
- Photo-z vs spec-z comparison
- Δz/(1+z) distribution histograms
- Binned metrics plots
- Outlier visualization
"""

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_photoz_vs_specz(
    z_phot: ArrayLike,
    z_spec: ArrayLike,
    z_err_lo: ArrayLike | None = None,
    z_err_hi: ArrayLike | None = None,
    title: str = "Photometric vs Spectroscopic Redshift",
    show_metrics: bool = True,
    figsize: tuple = (8, 8),
) -> Figure:
    """Create diagnostic plot comparing photo-z to spec-z.

    Parameters
    ----------
    z_phot : array-like
        Photometric redshifts
    z_spec : array-like
        Spectroscopic redshifts
    z_err_lo : array-like, optional
        Lower error bars (z_phot - z_lo)
    z_err_hi : array-like, optional
        Upper error bars (z_hi - z_phot)
    title : str
        Plot title
    show_metrics : bool
        Whether to show metrics in the plot
    figsize : tuple
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    z_phot = np.asarray(z_phot)
    z_spec = np.asarray(z_spec)

    # Filter valid data
    valid = (z_phot > 0) & (z_spec > 0) & np.isfinite(z_phot) & np.isfinite(z_spec)
    z_phot = z_phot[valid]
    z_spec = z_spec[valid]

    if z_err_lo is not None:
        z_err_lo = np.asarray(z_err_lo)[valid]
    if z_err_hi is not None:
        z_err_hi = np.asarray(z_err_hi)[valid]

    fig, ax = plt.subplots(figsize=figsize)

    # Determine plot range
    z_max = max(z_spec.max(), z_phot.max()) * 1.1
    z_range = [0, z_max]

    # Plot data points
    if z_err_lo is not None and z_err_hi is not None:
        ax.errorbar(
            z_spec,
            z_phot,
            yerr=[z_err_lo, z_err_hi],
            fmt="o",
            markersize=4,
            alpha=0.5,
            color="steelblue",
            ecolor="lightgray",
            elinewidth=0.5,
            label="Data",
        )
    else:
        ax.scatter(z_spec, z_phot, alpha=0.5, s=20, c="steelblue", label="Data")

    # 1:1 line
    ax.plot(z_range, z_range, "k--", lw=1.5, label="1:1")

    # ±0.15(1+z) outlier boundaries
    z_arr = np.linspace(0, z_max, 100)
    ax.fill_between(
        z_arr,
        z_arr - 0.15 * (1 + z_arr),
        z_arr + 0.15 * (1 + z_arr),
        alpha=0.15,
        color="gray",
        label="±0.15(1+z)",
    )

    # Compute and display metrics
    if show_metrics and len(z_phot) > 3:
        dz = (z_phot - z_spec) / (1 + z_spec)
        nmad = 1.48 * np.median(np.abs(dz - np.median(dz)))
        bias = np.median(dz)
        outlier_frac = np.mean(np.abs(dz) > 0.15)

        metrics_text = (
            f"N = {len(z_phot)}\n"
            f"NMAD = {nmad:.4f}\n"
            f"Bias = {bias:+.4f}\n"
            f"Outliers = {outlier_frac:.1%}"
        )
        ax.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel("Spectroscopic Redshift", fontsize=12)
    ax.set_ylabel("Photometric Redshift", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.set_xlim(z_range)
    ax.set_ylim(z_range)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_delta_z_histogram(
    z_phot: ArrayLike,
    z_spec: ArrayLike,
    title: str = "Photo-z Residuals",
    figsize: tuple = (8, 6),
    bins: int = 50,
    range_sigma: float = 0.5,
) -> Figure:
    """Plot histogram of Δz/(1+z) residuals.

    Parameters
    ----------
    z_phot : array-like
        Photometric redshifts
    z_spec : array-like
        Spectroscopic redshifts
    title : str
        Plot title
    figsize : tuple
        Figure size
    bins : int
        Number of histogram bins
    range_sigma : float
        Plot range in Δz/(1+z)

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    z_phot = np.asarray(z_phot)
    z_spec = np.asarray(z_spec)

    # Filter valid data
    valid = (z_phot > 0) & (z_spec > 0) & np.isfinite(z_phot) & np.isfinite(z_spec)
    z_phot = z_phot[valid]
    z_spec = z_spec[valid]

    dz = (z_phot - z_spec) / (1 + z_spec)

    fig, ax = plt.subplots(figsize=figsize)

    # Histogram
    ax.hist(
        dz,
        bins=bins,
        range=(-range_sigma, range_sigma),
        color="steelblue",
        alpha=0.7,
        edgecolor="white",
    )

    # Reference lines
    ax.axvline(0, color="k", linestyle="--", lw=1.5, label="Zero")

    # NMAD lines
    nmad = 1.48 * np.median(np.abs(dz - np.median(dz)))
    bias = np.median(dz)

    ax.axvline(bias, color="red", linestyle="-", lw=1.5, label=f"Bias = {bias:.4f}")
    ax.axvline(
        bias + nmad, color="red", linestyle=":", lw=1, label=f"NMAD = {nmad:.4f}"
    )
    ax.axvline(bias - nmad, color="red", linestyle=":", lw=1)

    # Outlier thresholds
    ax.axvline(-0.15, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0.15, color="gray", linestyle="--", alpha=0.5, label="±0.15 outlier")

    # Statistics text
    outlier_frac = np.mean(np.abs(dz) > 0.15)
    stats_text = f"N = {len(dz)}\nOutliers = {outlier_frac:.1%}"
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Δz / (1+z)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_validation_panel(
    z_phot: ArrayLike,
    z_spec: ArrayLike,
    z_err_lo: ArrayLike | None = None,
    z_err_hi: ArrayLike | None = None,
    title: str = "Photo-z Validation",
    figsize: tuple = (14, 6),
) -> Figure:
    """Create a two-panel validation plot.

    Left panel: z_phot vs z_spec scatter plot
    Right panel: Δz/(1+z) histogram

    Parameters
    ----------
    z_phot : array-like
        Photometric redshifts
    z_spec : array-like
        Spectroscopic redshifts
    z_err_lo : array-like, optional
        Lower error bars
    z_err_hi : array-like, optional
        Upper error bars
    title : str
        Overall title
    figsize : tuple
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    z_phot = np.asarray(z_phot)
    z_spec = np.asarray(z_spec)

    valid = (z_phot > 0) & (z_spec > 0) & np.isfinite(z_phot) & np.isfinite(z_spec)
    z_phot_v = z_phot[valid]
    z_spec_v = z_spec[valid]

    if z_err_lo is not None:
        z_err_lo = np.asarray(z_err_lo)[valid]
    if z_err_hi is not None:
        z_err_hi = np.asarray(z_err_hi)[valid]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left panel: z_phot vs z_spec
    ax1 = axes[0]
    z_max = max(z_spec_v.max(), z_phot_v.max()) * 1.1

    ax1.scatter(z_spec_v, z_phot_v, alpha=0.5, s=20, c="steelblue")
    ax1.plot([0, z_max], [0, z_max], "k--", lw=1.5, label="1:1")

    z_arr = np.linspace(0, z_max, 100)
    ax1.fill_between(
        z_arr,
        z_arr - 0.15 * (1 + z_arr),
        z_arr + 0.15 * (1 + z_arr),
        alpha=0.15,
        color="gray",
        label="±0.15(1+z)",
    )

    ax1.set_xlabel("Spectroscopic Redshift")
    ax1.set_ylabel("Photometric Redshift")
    ax1.set_title("z_phot vs z_spec")
    ax1.legend(loc="lower right")
    ax1.set_xlim([0, z_max])
    ax1.set_ylim([0, z_max])
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Right panel: histogram
    ax2 = axes[1]
    dz = (z_phot_v - z_spec_v) / (1 + z_spec_v)

    ax2.hist(
        dz, bins=50, range=(-0.5, 0.5), color="steelblue", alpha=0.7, edgecolor="white"
    )

    nmad = 1.48 * np.median(np.abs(dz - np.median(dz)))
    bias = np.median(dz)
    outlier_frac = np.mean(np.abs(dz) > 0.15)

    ax2.axvline(0, color="k", linestyle="--", lw=1.5)
    ax2.axvline(bias, color="red", linestyle="-", lw=1.5, label=f"Bias = {bias:.4f}")
    ax2.axvline(bias + nmad, color="red", linestyle=":", lw=1)
    ax2.axvline(bias - nmad, color="red", linestyle=":", lw=1, label=f"NMAD = {nmad:.4f}")

    metrics_text = f"N = {len(dz)}\nOutliers = {outlier_frac:.1%}"
    ax2.text(
        0.95,
        0.95,
        metrics_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax2.set_xlabel("Δz / (1+z)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_binned_metrics(
    binned_results: dict,
    metric: str = "nmad",
    figsize: tuple = (10, 6),
) -> Figure:
    """Plot metrics as function of redshift or magnitude bins.

    Parameters
    ----------
    binned_results : dict
        Output from binned_metrics() or magnitude_dependent_metrics()
    metric : str
        Metric to plot ('nmad', 'bias', 'sigma', 'outlier_frac')
    figsize : tuple
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    bins = binned_results["bins"]

    if len(bins) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    # Determine x-axis (redshift or magnitude)
    if "z_median" in bins[0]:
        x_values = [b["z_median"] for b in bins]
        x_label = "Redshift"
        x_lo = [b["z_min"] for b in bins]
        x_hi = [b["z_max"] for b in bins]
    elif "mag_median" in bins[0]:
        x_values = [b["mag_median"] for b in bins]
        x_label = "Magnitude"
        x_lo = [b["mag_min"] for b in bins]
        x_hi = [b["mag_max"] for b in bins]
    else:
        x_values = list(range(len(bins)))
        x_label = "Bin"
        x_lo = x_hi = None

    # Get metric values
    y_values = []
    for b in bins:
        if "error" in b:
            y_values.append(np.nan)
        else:
            y_values.append(b.get(metric, np.nan))

    # Plot
    ax.plot(x_values, y_values, "o-", markersize=8, color="steelblue")

    # Add horizontal bars for bin widths
    if x_lo is not None and x_hi is not None:
        for x, y, lo, hi in zip(x_values, y_values, x_lo, x_hi):
            if np.isfinite(y):
                ax.hlines(y, lo, hi, color="steelblue", alpha=0.3, linewidth=2)

    # Labels
    metric_labels = {
        "nmad": "NMAD",
        "bias": "Bias",
        "sigma": "Standard Deviation",
        "outlier_frac": "Outlier Fraction",
    }

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
    ax.set_title(f"{metric_labels.get(metric, metric)} vs {x_label}", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
