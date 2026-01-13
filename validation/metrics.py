"""Photo-z quality metrics and comparison tools.

This module provides standard metrics for evaluating photometric
redshift accuracy, following conventions from the literature.

Key metrics:
- NMAD: Normalized Median Absolute Deviation
- Bias: Systematic offset
- Outlier fraction: Fraction with |Δz/(1+z)| > threshold

References:
- Ilbert et al. 2006 (COSMOS photo-z methodology)
- Brammer et al. 2008 (EAZY)
- Dahlen et al. 2013 (CANDELS photo-z comparison)
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def photoz_metrics(
    z_phot: ArrayLike,
    z_true: ArrayLike,
    outlier_threshold: float = 0.15,
) -> dict:
    """Compute standard photo-z quality metrics.

    Parameters
    ----------
    z_phot : array-like
        Photometric redshifts
    z_true : array-like
        True (spectroscopic) redshifts
    outlier_threshold : float
        Threshold for outlier definition in Δz/(1+z)

    Returns
    -------
    dict
        Metrics including nmad, bias, sigma, outlier_frac
    """
    z_phot = np.asarray(z_phot)
    z_true = np.asarray(z_true)

    # Filter valid values
    valid = (z_phot > 0) & (z_true > 0) & np.isfinite(z_phot) & np.isfinite(z_true)

    if valid.sum() < 3:
        return {
            "error": "Too few valid points",
            "n_valid": valid.sum(),
        }

    z_phot = z_phot[valid]
    z_true = z_true[valid]

    # Normalized residual: Δz / (1 + z_spec)
    dz = (z_phot - z_true) / (1 + z_true)

    # NMAD: Normalized Median Absolute Deviation
    # σ_NMAD = 1.48 × median(|Δz - median(Δz)|)
    nmad = 1.48 * np.median(np.abs(dz - np.median(dz)))

    # Bias: systematic offset
    bias = np.median(dz)

    # Standard deviation
    sigma = np.std(dz)

    # Outlier fraction
    outlier_mask = np.abs(dz) > outlier_threshold
    outlier_frac = np.mean(outlier_mask)

    # Catastrophic outlier fraction (|Δz/(1+z)| > 0.5)
    catastrophic_mask = np.abs(dz) > 0.5
    catastrophic_frac = np.mean(catastrophic_mask)

    # 68% confidence interval
    percentiles = np.percentile(dz, [16, 50, 84])
    sigma_68 = (percentiles[2] - percentiles[0]) / 2

    return {
        "n_valid": len(z_phot),
        "nmad": nmad,
        "bias": bias,
        "sigma": sigma,
        "sigma_68": sigma_68,
        "outlier_frac": outlier_frac,
        "catastrophic_frac": catastrophic_frac,
        "outlier_threshold": outlier_threshold,
        "percentiles": {
            "p16": percentiles[0],
            "p50": percentiles[1],
            "p84": percentiles[2],
        },
    }


def compare_photoz_methods(
    z_true: ArrayLike,
    methods: dict[str, ArrayLike],
    outlier_threshold: float = 0.15,
) -> dict[str, dict]:
    """Compare photo-z accuracy between multiple methods.

    Parameters
    ----------
    z_true : array-like
        True (spectroscopic) redshifts
    methods : dict
        Mapping of method name to photo-z array
    outlier_threshold : float
        Threshold for outlier definition

    Returns
    -------
    dict
        Metrics for each method
    """
    z_true = np.asarray(z_true)
    results = {}

    for name, z_phot in methods.items():
        z_phot = np.asarray(z_phot)
        results[name] = photoz_metrics(z_phot, z_true, outlier_threshold)

    return results


def binned_metrics(
    z_phot: ArrayLike,
    z_true: ArrayLike,
    z_bins: ArrayLike | None = None,
    n_bins: int = 5,
) -> dict:
    """Compute photo-z metrics in redshift bins.

    Parameters
    ----------
    z_phot : array-like
        Photometric redshifts
    z_true : array-like
        True redshifts
    z_bins : array-like, optional
        Bin edges. If None, uses equal-count bins.
    n_bins : int
        Number of bins if z_bins not provided

    Returns
    -------
    dict
        Metrics for each bin
    """
    z_phot = np.asarray(z_phot)
    z_true = np.asarray(z_true)

    valid = (z_phot > 0) & (z_true > 0) & np.isfinite(z_phot) & np.isfinite(z_true)
    z_phot = z_phot[valid]
    z_true = z_true[valid]

    if z_bins is None:
        # Equal-count bins
        z_bins = np.percentile(z_true, np.linspace(0, 100, n_bins + 1))
        z_bins = np.unique(z_bins)

    results = {
        "bin_edges": z_bins,
        "bins": [],
    }

    for i in range(len(z_bins) - 1):
        mask = (z_true >= z_bins[i]) & (z_true < z_bins[i + 1])

        if mask.sum() < 3:
            continue

        bin_metrics = photoz_metrics(z_phot[mask], z_true[mask])
        bin_metrics["z_min"] = z_bins[i]
        bin_metrics["z_max"] = z_bins[i + 1]
        bin_metrics["z_median"] = np.median(z_true[mask])

        results["bins"].append(bin_metrics)

    return results


def magnitude_dependent_metrics(
    z_phot: ArrayLike,
    z_true: ArrayLike,
    magnitude: ArrayLike,
    mag_bins: ArrayLike | None = None,
    n_bins: int = 5,
) -> dict:
    """Compute photo-z metrics as function of magnitude.

    Parameters
    ----------
    z_phot : array-like
        Photometric redshifts
    z_true : array-like
        True redshifts
    magnitude : array-like
        Source magnitudes
    mag_bins : array-like, optional
        Magnitude bin edges
    n_bins : int
        Number of bins if mag_bins not provided

    Returns
    -------
    dict
        Metrics for each magnitude bin
    """
    z_phot = np.asarray(z_phot)
    z_true = np.asarray(z_true)
    magnitude = np.asarray(magnitude)

    valid = (
        (z_phot > 0)
        & (z_true > 0)
        & np.isfinite(z_phot)
        & np.isfinite(z_true)
        & np.isfinite(magnitude)
    )

    z_phot = z_phot[valid]
    z_true = z_true[valid]
    magnitude = magnitude[valid]

    if mag_bins is None:
        mag_bins = np.percentile(magnitude, np.linspace(0, 100, n_bins + 1))
        mag_bins = np.unique(mag_bins)

    results = {
        "bin_edges": mag_bins,
        "bins": [],
    }

    for i in range(len(mag_bins) - 1):
        mask = (magnitude >= mag_bins[i]) & (magnitude < mag_bins[i + 1])

        if mask.sum() < 3:
            continue

        bin_metrics = photoz_metrics(z_phot[mask], z_true[mask])
        bin_metrics["mag_min"] = mag_bins[i]
        bin_metrics["mag_max"] = mag_bins[i + 1]
        bin_metrics["mag_median"] = np.median(magnitude[mask])

        results["bins"].append(bin_metrics)

    return results


def odds_quality_metrics(
    z_phot: ArrayLike,
    z_true: ArrayLike,
    odds: ArrayLike,
    odds_thresholds: list[float] | None = None,
) -> dict:
    """Compute photo-z metrics for different ODDS quality cuts.

    ODDS is a quality parameter measuring how concentrated the
    redshift PDF is around the peak.

    Parameters
    ----------
    z_phot : array-like
        Photometric redshifts
    z_true : array-like
        True redshifts
    odds : array-like
        ODDS quality parameter (0-1)
    odds_thresholds : list of float, optional
        ODDS thresholds to test

    Returns
    -------
    dict
        Metrics for each ODDS threshold
    """
    z_phot = np.asarray(z_phot)
    z_true = np.asarray(z_true)
    odds = np.asarray(odds)

    if odds_thresholds is None:
        odds_thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]

    valid = (
        (z_phot > 0)
        & (z_true > 0)
        & np.isfinite(z_phot)
        & np.isfinite(z_true)
        & np.isfinite(odds)
    )

    z_phot = z_phot[valid]
    z_true = z_true[valid]
    odds = odds[valid]

    results = {
        "all_sources": photoz_metrics(z_phot, z_true),
        "thresholds": [],
    }

    for thresh in odds_thresholds:
        mask = odds >= thresh
        n_pass = mask.sum()
        fraction = n_pass / len(odds) if len(odds) > 0 else 0

        if n_pass >= 3:
            thresh_metrics = photoz_metrics(z_phot[mask], z_true[mask])
        else:
            thresh_metrics = {"error": "Too few sources"}

        thresh_metrics["odds_threshold"] = thresh
        thresh_metrics["n_pass"] = n_pass
        thresh_metrics["fraction"] = fraction

        results["thresholds"].append(thresh_metrics)

    return results


def format_metrics_table(metrics_list: list[dict], method_names: list[str]) -> str:
    """Format metrics as a text table for display.

    Parameters
    ----------
    metrics_list : list of dict
        List of metrics dictionaries
    method_names : list of str
        Names for each method

    Returns
    -------
    str
        Formatted table string
    """
    header = (
        f"{'Method':<20} {'N':>6} {'NMAD':>8} {'Bias':>8} {'σ':>8} "
        f"{'Out%':>8} {'Cat%':>8}"
    )
    separator = "-" * len(header)

    lines = [header, separator]

    for name, metrics in zip(method_names, metrics_list):
        if "error" in metrics:
            lines.append(f"{name:<20} {'ERROR':>6}")
            continue

        line = (
            f"{name:<20} {metrics['n_valid']:>6} {metrics['nmad']:>8.4f} "
            f"{metrics['bias']:>8.4f} {metrics['sigma']:>8.4f} "
            f"{metrics['outlier_frac']*100:>7.1f}% "
            f"{metrics['catastrophic_frac']*100:>7.1f}%"
        )
        lines.append(line)

    return "\n".join(lines)


# =============================================================================
# Star-Galaxy Classification Metrics
# =============================================================================


def confusion_matrix_star_galaxy(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: list[str] | None = None,
) -> dict:
    """Compute confusion matrix for star-galaxy classification.

    Parameters
    ----------
    y_true : array-like
        True labels (1=galaxy, 0=star)
    y_pred : array-like
        Predicted labels
    labels : list, optional
        Label names (default: ['Star', 'Galaxy'])

    Returns
    -------
    dict
        Confusion matrix and derived metrics including:
        - confusion_matrix: 2x2 array [[TN, FP], [FN, TP]]
        - accuracy, precision, recall, F1 for each class
        - contamination rates
    """
    if labels is None:
        labels = ["Star", "Galaxy"]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Compute confusion matrix components
    # Convention: 1 = galaxy (positive), 0 = star (negative)
    tp = ((y_true == 1) & (y_pred == 1)).sum()  # Galaxy correctly classified
    tn = ((y_true == 0) & (y_pred == 0)).sum()  # Star correctly classified
    fp = ((y_true == 0) & (y_pred == 1)).sum()  # Star misclassified as galaxy
    fn = ((y_true == 1) & (y_pred == 0)).sum()  # Galaxy misclassified as star

    total = len(y_true)

    # Overall metrics
    accuracy = (tp + tn) / total if total > 0 else 0

    # Galaxy metrics (positive class)
    precision_galaxy = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_galaxy = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_galaxy = (
        2 * precision_galaxy * recall_galaxy / (precision_galaxy + recall_galaxy)
        if (precision_galaxy + recall_galaxy) > 0
        else 0
    )

    # Star metrics (negative class)
    precision_star = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_star = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_star = (
        2 * precision_star * recall_star / (precision_star + recall_star)
        if (precision_star + recall_star) > 0
        else 0
    )

    # Contamination rates
    # Fraction of predicted "galaxies" that are actually stars
    galaxy_contamination = fp / (tp + fp) if (tp + fp) > 0 else 0
    # Fraction of predicted "stars" that are actually galaxies
    star_contamination = fn / (tn + fn) if (tn + fn) > 0 else 0

    # Completeness rates
    galaxy_completeness = recall_galaxy  # Fraction of true galaxies recovered
    star_completeness = recall_star  # Fraction of true stars recovered

    return {
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
        "labels": labels,
        "n_total": total,
        "n_galaxies_true": int(tp + fn),
        "n_stars_true": int(tn + fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "accuracy": accuracy,
        "precision_galaxy": precision_galaxy,
        "recall_galaxy": recall_galaxy,
        "f1_galaxy": f1_galaxy,
        "precision_star": precision_star,
        "recall_star": recall_star,
        "f1_star": f1_star,
        "galaxy_contamination": galaxy_contamination,
        "star_contamination": star_contamination,
        "galaxy_completeness": galaxy_completeness,
        "star_completeness": star_completeness,
    }


def binned_metrics_2d(
    z_phot: ArrayLike,
    z_true: ArrayLike,
    magnitude: ArrayLike,
    z_bins: ArrayLike | None = None,
    mag_bins: ArrayLike | None = None,
) -> list[dict]:
    """Compute photo-z metrics in 2D bins of redshift and magnitude.

    This allows assessing photo-z performance as a function of both
    redshift and brightness simultaneously.

    Parameters
    ----------
    z_phot : array-like
        Photometric redshifts
    z_true : array-like
        True redshifts
    magnitude : array-like
        Source magnitudes
    z_bins : array-like, optional
        Redshift bin edges (default: [0, 0.5, 1.0, 1.5, 2.0, 3.0])
    mag_bins : array-like, optional
        Magnitude bin edges (default: [18, 22, 24, 26, 28, 30])

    Returns
    -------
    list[dict]
        List of metrics dictionaries for each 2D bin
    """
    z_phot = np.asarray(z_phot)
    z_true = np.asarray(z_true)
    magnitude = np.asarray(magnitude)

    if z_bins is None:
        z_bins = np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0])
    else:
        z_bins = np.asarray(z_bins)

    if mag_bins is None:
        mag_bins = np.array([18, 22, 24, 26, 28, 30])
    else:
        mag_bins = np.asarray(mag_bins)

    # Filter valid values
    valid = (
        (z_phot > 0)
        & (z_true > 0)
        & np.isfinite(z_phot)
        & np.isfinite(z_true)
        & np.isfinite(magnitude)
    )

    z_phot = z_phot[valid]
    z_true = z_true[valid]
    magnitude = magnitude[valid]

    results = []

    for i in range(len(z_bins) - 1):
        for j in range(len(mag_bins) - 1):
            mask = (
                (z_true >= z_bins[i])
                & (z_true < z_bins[i + 1])
                & (magnitude >= mag_bins[j])
                & (magnitude < mag_bins[j + 1])
            )

            n = mask.sum()
            if n < 5:
                continue

            metrics = photoz_metrics(z_phot[mask], z_true[mask])

            # Add bin information
            metrics["z_min"] = z_bins[i]
            metrics["z_max"] = z_bins[i + 1]
            metrics["z_center"] = (z_bins[i] + z_bins[i + 1]) / 2
            metrics["mag_min"] = mag_bins[j]
            metrics["mag_max"] = mag_bins[j + 1]
            metrics["mag_center"] = (mag_bins[j] + mag_bins[j + 1]) / 2
            metrics["n_sources"] = n

            results.append(metrics)

    return results


def format_confusion_matrix(cm_dict: dict) -> str:
    """Format confusion matrix for display.

    Parameters
    ----------
    cm_dict : dict
        Output from confusion_matrix_star_galaxy()

    Returns
    -------
    str
        Formatted string representation
    """
    cm = cm_dict["confusion_matrix"]
    labels = cm_dict["labels"]

    lines = [
        "=" * 60,
        "Star-Galaxy Classification Confusion Matrix",
        "=" * 60,
        "",
        f"{'':20} Predicted {labels[0]:>10}  Predicted {labels[1]:>10}",
        f"Actual {labels[0]:>12}     {cm[0,0]:>10}        {cm[0,1]:>10}",
        f"Actual {labels[1]:>12}     {cm[1,0]:>10}        {cm[1,1]:>10}",
        "",
        f"Total samples: {cm_dict['n_total']}",
        f"True galaxies: {cm_dict['n_galaxies_true']}",
        f"True stars:    {cm_dict['n_stars_true']}",
        "",
        "Performance Metrics:",
        f"  Accuracy:            {cm_dict['accuracy']:.4f}",
        "",
        f"  Galaxy Precision:    {cm_dict['precision_galaxy']:.4f}",
        f"  Galaxy Recall:       {cm_dict['recall_galaxy']:.4f}",
        f"  Galaxy F1:           {cm_dict['f1_galaxy']:.4f}",
        "",
        f"  Star Precision:      {cm_dict['precision_star']:.4f}",
        f"  Star Recall:         {cm_dict['recall_star']:.4f}",
        f"  Star F1:             {cm_dict['f1_star']:.4f}",
        "",
        "Contamination & Completeness:",
        f"  Galaxy sample contamination (stars in galaxy sample): {cm_dict['galaxy_contamination']:.1%}",
        f"  Star sample contamination (galaxies in star sample):  {cm_dict['star_contamination']:.1%}",
        f"  Galaxy completeness (recovered fraction):             {cm_dict['galaxy_completeness']:.1%}",
        f"  Star completeness (recovered fraction):               {cm_dict['star_completeness']:.1%}",
        "=" * 60,
    ]

    return "\n".join(lines)
