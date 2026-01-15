"""Compare classification results using different template sets.

This module allows testing how different galaxy template libraries
affect photo-z accuracy and galaxy classification.

Template sets available:
- Our CWW-style templates (spectra/)
- EAZY CWW+KIN templates
- EAZY v1.0 optimized templates
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def compare_template_sets(
    fluxes: list,
    errors: list,
    template_paths: dict[str, str | Path],
    z_min: float = 0.0,
    z_max: float = 3.5,
) -> dict[str, dict]:
    """Run classification with multiple template sets and compare.

    Parameters
    ----------
    fluxes : list
        Observed fluxes [B, I, U, V]
    errors : list
        Flux errors
    template_paths : dict
        Mapping of template set name to spectra directory path
    z_min, z_max : float
        Redshift search range

    Returns
    -------
    dict
        Results from each template set with keys:
        - 'type': Best-fit galaxy type
        - 'redshift': Best-fit redshift
        - 'chi2': Minimum chi-squared
        - 'z_lo', 'z_hi': Uncertainty bounds
    """
    from classify import classify_galaxy_with_pdf

    results = {}

    for name, path in template_paths.items():
        try:
            result = classify_galaxy_with_pdf(
                fluxes,
                errors,
                spectra_path=path,
                z_min=z_min,
                z_max=z_max,
            )
            results[name] = {
                "type": result.galaxy_type,
                "redshift": result.redshift,
                "z_lo": result.z_lo,
                "z_hi": result.z_hi,
                "chi2": result.chi_sq_min,
                "odds": result.odds,
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return results


def compare_template_sets_batch(
    catalog: pd.DataFrame,
    template_paths: dict[str, str | Path],
    flux_cols: list[str] | None = None,
    error_cols: list[str] | None = None,
    z_min: float = 0.0,
    z_max: float = 3.5,
    progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run classification on a catalog with multiple template sets.

    Parameters
    ----------
    catalog : pd.DataFrame
        Source catalog with flux and error columns
    template_paths : dict
        Mapping of template set name to spectra directory
    flux_cols : list of str, optional
        Flux column names [B, I, U, V]
    error_cols : list of str, optional
        Error column names
    z_min, z_max : float
        Redshift search range
    progress : bool
        Show progress bar

    Returns
    -------
    dict
        Mapping of template set name to results DataFrame
    """
    if flux_cols is None:
        flux_cols = ["flux_b", "flux_i", "flux_u", "flux_v"]
    if error_cols is None:
        error_cols = ["error_b", "error_i", "error_u", "error_v"]

    from classify import classify_galaxy_with_pdf

    results = {name: [] for name in template_paths}

    iterator = catalog.iterrows()
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(
                iterator, total=len(catalog), desc="Comparing templates"
            )
        except ImportError:
            pass

    for idx, row in iterator:
        fluxes = [row[col] for col in flux_cols]
        errors = [row[col] for col in error_cols]

        for name, path in template_paths.items():
            try:
                result = classify_galaxy_with_pdf(
                    fluxes,
                    errors,
                    spectra_path=path,
                    z_min=z_min,
                    z_max=z_max,
                )
                results[name].append({
                    "id": idx,
                    "type": result.galaxy_type,
                    "redshift": result.redshift,
                    "z_lo": result.z_lo,
                    "z_hi": result.z_hi,
                    "chi2": result.chi_sq_min,
                    "odds": result.odds,
                })
            except Exception as e:
                results[name].append({
                    "id": idx,
                    "error": str(e),
                })

    # Convert to DataFrames
    return {name: pd.DataFrame(data) for name, data in results.items()}


def summarize_template_comparison(
    results: dict[str, pd.DataFrame],
    spec_z: np.ndarray | pd.Series | None = None,
) -> dict[str, dict]:
    """Summarize results from template set comparison.

    Parameters
    ----------
    results : dict
        Results from compare_template_sets_batch()
    spec_z : array-like, optional
        Spectroscopic redshifts for validation

    Returns
    -------
    dict
        Summary statistics for each template set
    """
    from validation.metrics import photoz_metrics

    summary = {}

    for name, df in results.items():
        if "error" in df.columns and df["error"].notna().any():
            n_errors = df["error"].notna().sum()
        else:
            n_errors = 0

        valid_mask = df["redshift"].notna() if "redshift" in df.columns else pd.Series([])
        n_valid = valid_mask.sum()

        stats = {
            "n_sources": len(df),
            "n_valid": n_valid,
            "n_errors": n_errors,
        }

        if n_valid > 0:
            z_arr = df.loc[valid_mask, "redshift"].values
            stats["z_median"] = np.median(z_arr)
            stats["z_mean"] = np.mean(z_arr)
            stats["z_std"] = np.std(z_arr)

            if "chi2" in df.columns:
                stats["chi2_median"] = np.median(df.loc[valid_mask, "chi2"])

            if "odds" in df.columns:
                stats["odds_median"] = np.median(df.loc[valid_mask, "odds"])
                stats["high_odds_frac"] = np.mean(df.loc[valid_mask, "odds"] > 0.9)

        # Compare to spec-z if available
        if spec_z is not None and n_valid > 0:
            spec_z_arr = np.asarray(spec_z)
            if len(spec_z_arr) == len(df):
                spec_valid = (spec_z_arr > 0) & valid_mask.values
                if spec_valid.sum() > 5:
                    metrics = photoz_metrics(
                        z_arr[spec_valid[valid_mask]],
                        spec_z_arr[spec_valid],
                    )
                    stats["vs_specz"] = metrics

        summary[name] = stats

    return summary


def plot_template_comparison(
    results: dict[str, Any],
    true_z: float | None = None,
    figsize: tuple = (10, 6),
):
    """Plot comparison of redshift estimates from different templates.

    Parameters
    ----------
    results : dict
        Results from compare_template_sets() for a single source
    true_z : float, optional
        True (spectroscopic) redshift
    figsize : tuple
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    names = []
    redshifts = []
    errors_lo = []
    errors_hi = []

    for name, result in results.items():
        if "error" in result:
            continue

        names.append(name)
        z = result["redshift"]
        redshifts.append(z)
        errors_lo.append(z - result.get("z_lo", z))
        errors_hi.append(result.get("z_hi", z) - z)

    if not names:
        ax.text(0.5, 0.5, "No valid results", ha="center", va="center")
        return fig

    y_pos = np.arange(len(names))

    ax.barh(
        y_pos,
        redshifts,
        xerr=[errors_lo, errors_hi],
        color="steelblue",
        alpha=0.7,
        capsize=5,
    )

    if true_z is not None:
        ax.axvline(
            true_z, color="red", linestyle="--", lw=2, label=f"Spec-z = {true_z:.3f}"
        )
        ax.legend()

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Photometric Redshift")
    ax.set_title("Template Set Comparison")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig


def plot_template_comparison_scatter(
    results: dict[str, pd.DataFrame],
    reference_name: str,
    figsize: tuple = (12, 10),
):
    """Plot scatter comparison between template sets.

    Parameters
    ----------
    results : dict
        Results from compare_template_sets_batch()
    reference_name : str
        Name of reference template set for x-axis
    figsize : tuple
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    if reference_name not in results:
        raise ValueError(f"Reference '{reference_name}' not in results")

    other_names = [n for n in results if n != reference_name]

    if not other_names:
        raise ValueError("Need at least 2 template sets to compare")

    n_plots = len(other_names)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    ref_z = results[reference_name]["redshift"].values

    for i, name in enumerate(other_names):
        ax = axes[i // n_cols, i % n_cols]

        other_z = results[name]["redshift"].values

        valid = (ref_z > 0) & (other_z > 0) & np.isfinite(ref_z) & np.isfinite(other_z)

        ax.scatter(ref_z[valid], other_z[valid], alpha=0.3, s=10, c="steelblue")

        # Dynamic limits with epsilon padding
        z_min = min(ref_z[valid].min(), other_z[valid].min())
        z_max = max(ref_z[valid].max(), other_z[valid].max())
        z_padding = (z_max - z_min) * 0.08
        z_lo = max(0, z_min - z_padding)
        z_hi = z_max + z_padding
        ax.plot([z_lo, z_hi], [z_lo, z_hi], "k--", lw=1.5)

        # Compute scatter
        dz = (other_z[valid] - ref_z[valid]) / (1 + ref_z[valid])
        nmad = 1.48 * np.median(np.abs(dz))

        ax.set_xlabel(f"{reference_name} z")
        ax.set_ylabel(f"{name} z")
        ax.set_title(f"NMAD = {nmad:.4f}")
        ax.set_xlim([z_lo, z_hi])
        ax.set_ylim([z_lo, z_hi])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_plots, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    fig.suptitle(f"Template Comparison (Reference: {reference_name})", fontsize=14)
    plt.tight_layout()
    return fig


def get_default_template_paths() -> dict[str, Path]:
    """Get paths to available template sets.

    Returns
    -------
    dict
        Mapping of template set name to path
    """
    base_path = Path(__file__).parent.parent

    paths = {
        "our_templates": base_path / "spectra",
    }

    # Check for EAZY templates
    eazy_path = base_path / "spectra" / "eazy"
    if eazy_path.exists():
        for subdir in eazy_path.iterdir():
            if subdir.is_dir():
                paths[f"eazy_{subdir.name}"] = subdir

    # Check for converted templates
    converted_path = base_path / "spectra" / "eazy_converted"
    if converted_path.exists():
        paths["eazy_converted"] = converted_path

    return paths
