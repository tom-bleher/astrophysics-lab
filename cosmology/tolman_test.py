"""Tolman surface brightness test for cosmological expansion.

The Tolman test compares observed surface brightness vs redshift
to distinguish between expanding and static universe models:

- Expanding universe: SB ∝ (1+z)^{-4} (cosmological dimming)
- Static universe: SB = constant (no dimming)

In practice, galaxy evolution complicates this test, but it remains
a useful cosmological probe.

References:
- Tolman 1930, PNAS, 16, 511
- Lubin & Sandage 2001, AJ, 122, 1084
- Lerner et al. 2014, IJMPD, 23, 1450058
"""

import numpy as np
from numpy.typing import NDArray
import pandas as pd


def surface_brightness_vs_redshift(
    catalog: pd.DataFrame,
    flux_col: str,
    area_col: str,
    z_col: str = "redshift",
    pixel_scale: float = 0.04,
) -> tuple[NDArray, NDArray]:
    """Calculate surface brightness vs redshift.

    Parameters
    ----------
    catalog : pd.DataFrame
        Source catalog
    flux_col : str
        Column name for total flux
    area_col : str
        Column name for source area (in pixels)
    z_col : str
        Column name for redshift
    pixel_scale : float
        Pixel scale in arcsec/pixel

    Returns
    -------
    tuple
        (redshift array, surface brightness array)
    """
    z = catalog[z_col].values
    flux = catalog[flux_col].values
    area_pix = catalog[area_col].values

    # Convert area to square arcseconds
    area_arcsec2 = area_pix * pixel_scale**2

    # Surface brightness (flux per square arcsec)
    sb = flux / area_arcsec2

    # Filter valid values
    valid = (z > 0) & (sb > 0) & np.isfinite(z) & np.isfinite(sb)

    return z[valid], sb[valid]


def fit_tolman_exponent(
    z: NDArray,
    sb: NDArray,
    sb_error: NDArray | None = None,
) -> tuple[float, float, dict]:
    """Fit SB ∝ (1+z)^n and return n with uncertainty.

    In the expanding universe with no evolution:
        n = -4 (Tolman dimming)

    Parameters
    ----------
    z : NDArray
        Redshift values
    sb : NDArray
        Surface brightness values
    sb_error : NDArray, optional
        Uncertainties on surface brightness

    Returns
    -------
    tuple
        (n, n_error, fit_info)
    """
    from scipy.optimize import curve_fit

    # Filter valid data
    valid = (z > 0) & (sb > 0) & np.isfinite(z) & np.isfinite(sb)
    z = z[valid]
    sb = sb[valid]

    if sb_error is not None:
        sb_error = sb_error[valid]

    if len(z) < 3:
        return np.nan, np.nan, {"error": "Too few valid points"}

    # Model: SB = A * (1+z)^n
    def model(z, A, n):
        return A * (1 + z) ** n

    # Initial guesses
    A0 = np.median(sb)
    n0 = -2  # Between static (0) and expanding (-4)

    try:
        if sb_error is not None:
            popt, pcov = curve_fit(
                model,
                z,
                sb,
                p0=[A0, n0],
                sigma=sb_error,
                absolute_sigma=True,
                maxfev=5000,
            )
        else:
            popt, pcov = curve_fit(
                model,
                z,
                sb,
                p0=[A0, n0],
                maxfev=5000,
            )

        A_fit = popt[0]
        n_fit = popt[1]
        n_err = np.sqrt(pcov[1, 1])

        # Compute chi-squared
        sb_model = model(z, A_fit, n_fit)
        if sb_error is not None:
            chi2 = np.sum(((sb - sb_model) / sb_error) ** 2)
        else:
            chi2 = np.sum((sb - sb_model) ** 2) / np.var(sb)

        chi2_red = chi2 / (len(z) - 2)

        fit_info = {
            "A": A_fit,
            "n": n_fit,
            "n_err": n_err,
            "chi2": chi2,
            "chi2_red": chi2_red,
            "n_points": len(z),
        }

        return n_fit, n_err, fit_info

    except Exception as e:
        return np.nan, np.nan, {"error": str(e)}


def interpret_tolman_result(n: float, n_err: float) -> str:
    """Interpret the Tolman exponent result.

    Parameters
    ----------
    n : float
        Fitted exponent
    n_err : float
        Uncertainty on exponent

    Returns
    -------
    str
        Interpretation of the result
    """
    if np.isnan(n):
        return "Fit failed - cannot interpret"

    # Expected values
    n_expanding = -4.0  # Pure Tolman dimming
    n_static = 0.0  # No cosmological dimming

    # Check consistency
    sigma_from_expanding = abs(n - n_expanding) / n_err if n_err > 0 else np.inf
    sigma_from_static = abs(n - n_static) / n_err if n_err > 0 else np.inf

    lines = []
    lines.append(f"Fitted exponent: n = {n:.2f} ± {n_err:.2f}")
    lines.append("")
    lines.append("Expected values:")
    lines.append(f"  Expanding universe (pure): n = -4")
    lines.append(f"  Static universe:           n = 0")
    lines.append("")

    if sigma_from_expanding < 2:
        lines.append(f"Result CONSISTENT with expanding universe ({sigma_from_expanding:.1f}σ from -4)")
    elif sigma_from_static < 2:
        lines.append(f"Result CONSISTENT with static universe ({sigma_from_static:.1f}σ from 0)")
    elif -4 < n < 0:
        lines.append("Result between pure models - likely affected by galaxy evolution")
        lines.append("(Galaxies were brighter in the past, reducing dimming effect)")
    else:
        lines.append("Result outside expected range - check for systematics")

    return "\n".join(lines)


def plot_tolman_test(
    z: NDArray,
    sb: NDArray,
    sb_error: NDArray | None = None,
    n_fit: float | None = None,
    n_err: float | None = None,
    figsize: tuple = (10, 8),
):
    """Plot Tolman surface brightness test.

    Parameters
    ----------
    z : NDArray
        Redshift values
    sb : NDArray
        Surface brightness values
    sb_error : NDArray, optional
        Uncertainties
    n_fit : float, optional
        Fitted exponent (if already computed)
    n_err : float, optional
        Uncertainty on exponent
    figsize : tuple
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left panel: SB vs z (linear scale)
    ax1 = axes[0]

    if sb_error is not None:
        ax1.errorbar(z, sb, yerr=sb_error, fmt="o", alpha=0.5, markersize=4,
                     color="steelblue", ecolor="lightgray")
    else:
        ax1.scatter(z, sb, alpha=0.5, s=20, c="steelblue")

    # Plot model predictions
    z_model = np.linspace(0.01, z.max() * 1.1, 100)
    sb_ref = np.median(sb[z < 0.5]) if np.any(z < 0.5) else np.median(sb)

    ax1.plot(z_model, sb_ref * (1 + z_model) ** (-4), "r--", lw=2,
             label="Expanding (n=-4)")
    ax1.plot(z_model, sb_ref * np.ones_like(z_model), "g--", lw=2,
             label="Static (n=0)")

    if n_fit is not None:
        ax1.plot(z_model, sb_ref * (1 + z_model) ** n_fit, "k-", lw=2,
                 label=f"Fit (n={n_fit:.2f}±{n_err:.2f})" if n_err else f"Fit (n={n_fit:.2f})")

    ax1.set_xlabel("Redshift")
    ax1.set_ylabel("Surface Brightness")
    ax1.set_title("Tolman Surface Brightness Test")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right panel: SB vs (1+z) in log-log
    ax2 = axes[1]

    ax2.scatter(1 + z, sb, alpha=0.5, s=20, c="steelblue")

    # Power law slopes
    one_plus_z = 1 + z_model
    ax2.plot(one_plus_z, sb_ref * one_plus_z ** (-4), "r--", lw=2,
             label="n=-4 (expanding)")
    ax2.plot(one_plus_z, sb_ref * one_plus_z ** 0, "g--", lw=2,
             label="n=0 (static)")
    ax2.plot(one_plus_z, sb_ref * one_plus_z ** (-2), "b:", lw=1,
             label="n=-2")

    if n_fit is not None and np.isfinite(n_fit):
        ax2.plot(one_plus_z, sb_ref * one_plus_z ** n_fit, "k-", lw=2,
                 label=f"n={n_fit:.2f}")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("1 + z")
    ax2.set_ylabel("Surface Brightness")
    ax2.set_title("Log-Log View")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def binned_tolman_test(
    z: NDArray,
    sb: NDArray,
    n_bins: int = 5,
) -> dict:
    """Perform Tolman test in redshift bins.

    Parameters
    ----------
    z : NDArray
        Redshift values
    sb : NDArray
        Surface brightness values
    n_bins : int
        Number of redshift bins

    Returns
    -------
    dict
        Binned results with median SB in each bin
    """
    # Create equal-count bins
    z_edges = np.percentile(z, np.linspace(0, 100, n_bins + 1))
    z_edges = np.unique(z_edges)

    results = {
        "z_edges": z_edges,
        "z_centers": [],
        "sb_median": [],
        "sb_lo": [],
        "sb_hi": [],
        "n_sources": [],
    }

    for i in range(len(z_edges) - 1):
        mask = (z >= z_edges[i]) & (z < z_edges[i + 1])
        if mask.sum() < 3:
            continue

        sb_bin = sb[mask]

        results["z_centers"].append(np.median(z[mask]))
        results["sb_median"].append(np.median(sb_bin))
        results["sb_lo"].append(np.percentile(sb_bin, 16))
        results["sb_hi"].append(np.percentile(sb_bin, 84))
        results["n_sources"].append(mask.sum())

    # Convert to arrays
    for key in ["z_centers", "sb_median", "sb_lo", "sb_hi", "n_sources"]:
        results[key] = np.array(results[key])

    return results
