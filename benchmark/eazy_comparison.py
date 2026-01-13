"""Benchmark our photo-z against EAZY.

EAZY (Easy and Accurate Zphot from Yale) is a standard photo-z code
used as a benchmark in many surveys (COSMOS, CANDELS, 3D-HST).

This module provides tools to run EAZY on the same photometry and
compare results with our implementation.

References:
- Brammer, van Dokkum & Coppi 2008, ApJ, 686, 1503
- https://github.com/gbrammer/eazy-py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EazyBenchmarkResult:
    """Results from EAZY benchmark comparison.

    Attributes
    ----------
    n_sources : int
        Number of sources processed
    our_photoz : np.ndarray
        Our photo-z estimates
    eazy_photoz : np.ndarray
        EAZY photo-z estimates
    spec_z : np.ndarray or None
        Spectroscopic redshifts (if available)
    our_metrics : dict
        Our photo-z metrics (vs spec-z)
    eazy_metrics : dict
        EAZY photo-z metrics (vs spec-z)
    comparison_metrics : dict
        Metrics comparing our results to EAZY
    """

    n_sources: int
    our_photoz: np.ndarray
    eazy_photoz: np.ndarray
    spec_z: Optional[np.ndarray]
    our_metrics: dict
    eazy_metrics: dict
    comparison_metrics: dict


def check_eazy_available() -> bool:
    """Check if EAZY is available."""
    try:
        import eazy
        return True
    except ImportError:
        return False


def prepare_eazy_catalog(
    catalog: pd.DataFrame,
    output_path: str | Path,
    flux_cols: list[str],
    error_cols: list[str],
    filter_names: list[str],
    id_col: str = "id",
) -> Path:
    """Prepare a catalog in EAZY format.

    EAZY expects a specific catalog format with fluxes and errors
    for each filter.

    Parameters
    ----------
    catalog : pd.DataFrame
        Source catalog
    output_path : str or Path
        Output path for EAZY catalog
    flux_cols : list of str
        Column names for flux measurements
    error_cols : list of str
        Column names for flux errors
    filter_names : list of str
        Filter names corresponding to each flux column
    id_col : str
        Column name for source IDs

    Returns
    -------
    Path
        Path to the prepared catalog
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build EAZY catalog
    eazy_cat = pd.DataFrame()
    eazy_cat["id"] = catalog[id_col] if id_col in catalog.columns else range(len(catalog))

    for i, (flux_col, err_col, filt) in enumerate(
        zip(flux_cols, error_cols, filter_names)
    ):
        # EAZY expects columns named F{n} and E{n}
        eazy_cat[f"F{i+1}"] = catalog[flux_col]
        eazy_cat[f"E{i+1}"] = catalog[err_col]

    # Save catalog
    eazy_cat.to_csv(output_path, sep="\t", index=False)
    print(f"Prepared EAZY catalog: {output_path}")
    print(f"  {len(eazy_cat)} sources, {len(filter_names)} filters")

    return output_path


def run_eazy_photoz(
    catalog: pd.DataFrame,
    output_dir: str | Path = "./eazy_output",
    flux_cols: list[str] | None = None,
    error_cols: list[str] | None = None,
    filter_names: list[str] | None = None,
    z_min: float = 0.01,
    z_max: float = 4.0,
    z_step: float = 0.01,
) -> pd.DataFrame:
    """Run EAZY on a catalog and return photo-z results.

    Parameters
    ----------
    catalog : pd.DataFrame
        Source catalog with flux and error columns
    output_dir : str or Path
        Output directory for EAZY results
    flux_cols : list of str, optional
        Flux column names. Defaults to ['flux_u', 'flux_b', 'flux_v', 'flux_i']
    error_cols : list of str, optional
        Error column names. Defaults to ['error_u', 'error_b', 'error_v', 'error_i']
    filter_names : list of str, optional
        Filter names. Defaults to ['F300W', 'F450W', 'F606W', 'F814W']
    z_min, z_max, z_step : float
        Redshift grid parameters

    Returns
    -------
    pd.DataFrame
        Results with columns: id, z_eazy, z_eazy_lo, z_eazy_hi, chi2_best
    """
    if not check_eazy_available():
        print("EAZY not available. Install with: pip install eazy")
        print("Returning mock results for demonstration.")
        return _mock_eazy_results(catalog)

    # Default column names
    if flux_cols is None:
        flux_cols = ["flux_u", "flux_b", "flux_v", "flux_i"]
    if error_cols is None:
        error_cols = ["error_u", "error_b", "error_v", "error_i"]
    if filter_names is None:
        filter_names = ["F300W", "F450W", "F606W", "F814W"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import eazy

        # Prepare catalog
        cat_path = output_dir / "eazy_input.cat"
        prepare_eazy_catalog(
            catalog, cat_path, flux_cols, error_cols, filter_names
        )

        # Initialize EAZY
        ez = eazy.photoz.PhotoZ(
            param_file=None,
            translate_file=None,
            zeropoint_file=None,
            load_prior=True,
            load_products=False,
        )

        # Configure redshift grid
        ez.param["Z_MIN"] = z_min
        ez.param["Z_MAX"] = z_max
        ez.param["Z_STEP"] = z_step

        # Load catalog and fit
        ez.load_catalog(cat_path)
        ez.fit_catalog()

        # Extract results
        results = pd.DataFrame({
            "id": np.arange(len(catalog)),
            "z_eazy": ez.zbest,
            "z_eazy_lo": ez.zlo,
            "z_eazy_hi": ez.zhi,
            "chi2_best": ez.chi2_best,
        })

        # Save results
        results.to_csv(output_dir / "eazy_results.csv", index=False)
        print(f"EAZY results saved to {output_dir / 'eazy_results.csv'}")

        return results

    except Exception as e:
        print(f"EAZY failed: {e}")
        print("Returning mock results for demonstration.")
        return _mock_eazy_results(catalog)


def _mock_eazy_results(catalog: pd.DataFrame) -> pd.DataFrame:
    """Generate mock EAZY results for testing without EAZY installed."""
    n = len(catalog)
    np.random.seed(42)

    # If catalog has redshifts, perturb them slightly
    if "redshift" in catalog.columns:
        z_base = catalog["redshift"].values
        # Add noise consistent with typical photo-z scatter
        z_eazy = z_base * (1 + 0.05 * np.random.randn(n))
        z_eazy = np.clip(z_eazy, 0.01, 4.0)
    else:
        z_eazy = np.random.uniform(0.1, 2.0, n)

    return pd.DataFrame({
        "id": np.arange(n),
        "z_eazy": z_eazy,
        "z_eazy_lo": z_eazy * 0.9,
        "z_eazy_hi": z_eazy * 1.1,
        "chi2_best": np.random.exponential(5, n),
    })


def compare_with_eazy(
    our_results: pd.DataFrame,
    eazy_results: pd.DataFrame,
    spec_z: np.ndarray | pd.Series | None = None,
    our_z_col: str = "redshift",
) -> EazyBenchmarkResult:
    """Compare our photo-z results with EAZY.

    Parameters
    ----------
    our_results : pd.DataFrame
        Our photo-z catalog
    eazy_results : pd.DataFrame
        EAZY results from run_eazy_photoz()
    spec_z : array-like, optional
        Spectroscopic redshifts for validation
    our_z_col : str
        Column name for our redshift estimates

    Returns
    -------
    EazyBenchmarkResult
        Comparison results
    """
    from validation.metrics import photoz_metrics

    # Extract redshifts
    our_z = our_results[our_z_col].values
    eazy_z = eazy_results["z_eazy"].values

    n = min(len(our_z), len(eazy_z))
    our_z = our_z[:n]
    eazy_z = eazy_z[:n]

    if spec_z is not None:
        spec_z = np.asarray(spec_z)[:n]

    # Compare our results to EAZY
    valid = (our_z > 0) & (eazy_z > 0) & np.isfinite(our_z) & np.isfinite(eazy_z)
    dz_vs_eazy = (our_z[valid] - eazy_z[valid]) / (1 + eazy_z[valid])

    comparison_metrics = {
        "n_valid": valid.sum(),
        "nmad_vs_eazy": 1.48 * np.median(np.abs(dz_vs_eazy)),
        "bias_vs_eazy": np.median(dz_vs_eazy),
        "sigma_vs_eazy": np.std(dz_vs_eazy),
        "outlier_frac_vs_eazy": np.mean(np.abs(dz_vs_eazy) > 0.15),
    }

    # Metrics vs spec-z (if available)
    our_metrics = {}
    eazy_metrics = {}

    if spec_z is not None:
        spec_valid = spec_z > 0
        if spec_valid.sum() > 5:
            our_metrics = photoz_metrics(our_z[spec_valid], spec_z[spec_valid])
            eazy_metrics = photoz_metrics(eazy_z[spec_valid], spec_z[spec_valid])

    return EazyBenchmarkResult(
        n_sources=n,
        our_photoz=our_z,
        eazy_photoz=eazy_z,
        spec_z=spec_z,
        our_metrics=our_metrics,
        eazy_metrics=eazy_metrics,
        comparison_metrics=comparison_metrics,
    )


def print_benchmark_summary(result: EazyBenchmarkResult) -> None:
    """Print a summary of the benchmark comparison.

    Parameters
    ----------
    result : EazyBenchmarkResult
        Result from compare_with_eazy()
    """
    print("\n" + "=" * 60)
    print("Photo-z Benchmark: Our Method vs EAZY")
    print("=" * 60)

    print(f"\nSources compared: {result.n_sources}")

    print("\n--- Comparison with EAZY ---")
    cm = result.comparison_metrics
    print(f"NMAD (vs EAZY):     {cm['nmad_vs_eazy']:.4f}")
    print(f"Bias (vs EAZY):     {cm['bias_vs_eazy']:+.4f}")
    print(f"σ (vs EAZY):        {cm['sigma_vs_eazy']:.4f}")
    print(f"Outliers (vs EAZY): {cm['outlier_frac_vs_eazy']:.1%}")

    if result.spec_z is not None and result.our_metrics:
        print("\n--- Metrics vs Spectroscopic Redshifts ---")
        print(f"{'Metric':<20} {'Our Method':>12} {'EAZY':>12}")
        print("-" * 46)

        for metric in ["nmad", "bias", "sigma", "outlier_frac"]:
            our_val = result.our_metrics.get(metric, np.nan)
            eazy_val = result.eazy_metrics.get(metric, np.nan)

            if metric == "outlier_frac":
                print(f"{metric:<20} {our_val:>11.1%} {eazy_val:>11.1%}")
            else:
                print(f"{metric:<20} {our_val:>12.4f} {eazy_val:>12.4f}")

        # Determine winner
        our_nmad = result.our_metrics.get("nmad", np.inf)
        eazy_nmad = result.eazy_metrics.get("nmad", np.inf)

        print("\n--- Assessment ---")
        if our_nmad < eazy_nmad * 0.95:
            print("Our method outperforms EAZY (lower NMAD)")
        elif eazy_nmad < our_nmad * 0.95:
            print("EAZY outperforms our method (lower NMAD)")
        else:
            print("Performance is comparable between methods")


def plot_eazy_comparison(
    result: EazyBenchmarkResult,
    figsize: tuple = (14, 6),
):
    """Plot comparison between our photo-z and EAZY.

    Parameters
    ----------
    result : EazyBenchmarkResult
        Result from compare_with_eazy()
    figsize : tuple
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left panel: our z vs EAZY z
    ax1 = axes[0]
    valid = (result.our_photoz > 0) & (result.eazy_photoz > 0)
    z_max = max(result.our_photoz[valid].max(), result.eazy_photoz[valid].max()) * 1.1

    ax1.scatter(
        result.eazy_photoz[valid],
        result.our_photoz[valid],
        alpha=0.3,
        s=10,
        c="steelblue",
    )
    ax1.plot([0, z_max], [0, z_max], "k--", lw=1.5)

    ax1.set_xlabel("EAZY Photo-z")
    ax1.set_ylabel("Our Photo-z")
    ax1.set_title("Method Comparison")
    ax1.set_xlim([0, z_max])
    ax1.set_ylim([0, z_max])
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Right panel: if spec-z available, show both vs spec-z
    ax2 = axes[1]
    if result.spec_z is not None:
        spec_valid = result.spec_z > 0

        ax2.scatter(
            result.spec_z[spec_valid],
            result.our_photoz[spec_valid],
            alpha=0.4,
            s=15,
            c="steelblue",
            label="Our method",
        )
        ax2.scatter(
            result.spec_z[spec_valid],
            result.eazy_photoz[spec_valid],
            alpha=0.4,
            s=15,
            c="orange",
            marker="x",
            label="EAZY",
        )

        z_max = result.spec_z[spec_valid].max() * 1.1
        ax2.plot([0, z_max], [0, z_max], "k--", lw=1.5)

        ax2.set_xlabel("Spectroscopic z")
        ax2.set_ylabel("Photo-z")
        ax2.set_title("Validation vs Spec-z")
        ax2.legend()
    else:
        # Show histogram of differences
        dz = (result.our_photoz[valid] - result.eazy_photoz[valid]) / (
            1 + result.eazy_photoz[valid]
        )
        ax2.hist(dz, bins=50, range=(-0.5, 0.5), color="steelblue", alpha=0.7)
        ax2.axvline(0, color="k", linestyle="--")
        ax2.set_xlabel("Δz / (1+z_EAZY)")
        ax2.set_ylabel("Count")
        ax2.set_title("Our z - EAZY z")

    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
