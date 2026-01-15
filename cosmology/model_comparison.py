"""Model comparison statistics for cosmological fits.

This module provides tools for comparing different cosmological
models using information criteria:

- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- Chi-squared statistics

These help determine which model best describes the data while
accounting for model complexity.

References:
- Akaike 1974, IEEE Trans Auto Control
- Schwarz 1978, Annals of Statistics
- Liddle 2007, MNRAS, 377, L74
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def compute_chi2(
    model: Callable,
    data: NDArray,
    errors: NDArray,
    params: tuple,
) -> float:
    """Compute chi-squared for a model.

    Parameters
    ----------
    model : callable
        Model function that takes params and returns predictions
    data : NDArray
        Observed data
    errors : NDArray
        Measurement errors
    params : tuple
        Model parameters

    Returns
    -------
    float
        Chi-squared value
    """
    prediction = model(*params)
    return np.sum(((data - prediction) / errors) ** 2)


def compute_aic(chi2: float, k: int, n: int) -> float:
    """Compute Akaike Information Criterion.

    AIC = χ² + 2k

    Lower AIC indicates a better model, balancing fit quality
    against model complexity.

    Parameters
    ----------
    chi2 : float
        Chi-squared value
    k : int
        Number of model parameters
    n : int
        Number of data points

    Returns
    -------
    float
        AIC value
    """
    # Standard AIC
    aic = chi2 + 2 * k

    # Corrected AIC for small samples (AICc)
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else aic

    return aicc


def compute_bic(chi2: float, k: int, n: int) -> float:
    """Compute Bayesian Information Criterion.

    BIC = χ² + k * ln(n)

    BIC penalizes model complexity more strongly than AIC
    for large datasets.

    Parameters
    ----------
    chi2 : float
        Chi-squared value
    k : int
        Number of model parameters
    n : int
        Number of data points

    Returns
    -------
    float
        BIC value
    """
    return chi2 + k * np.log(n)


def compare_models(
    chi2_values: dict[str, float],
    n_params: dict[str, int],
    n_data: int,
) -> dict:
    """Compare multiple models using information criteria.

    Parameters
    ----------
    chi2_values : dict
        Chi-squared for each model
    n_params : dict
        Number of parameters for each model
    n_data : int
        Number of data points

    Returns
    -------
    dict
        Comparison results with AIC, BIC, and model probabilities
    """
    models = list(chi2_values.keys())

    results = {
        "models": models,
        "chi2": {},
        "aic": {},
        "bic": {},
        "delta_aic": {},
        "delta_bic": {},
        "aic_weights": {},
        "bic_weights": {},
    }

    # Compute AIC and BIC for each model
    aic_values = {}
    bic_values = {}

    for model in models:
        chi2 = chi2_values[model]
        k = n_params[model]

        aic = compute_aic(chi2, k, n_data)
        bic = compute_bic(chi2, k, n_data)

        results["chi2"][model] = chi2
        results["aic"][model] = aic
        results["bic"][model] = bic
        aic_values[model] = aic
        bic_values[model] = bic

    # Compute delta values relative to minimum
    aic_min = min(aic_values.values())
    bic_min = min(bic_values.values())

    for model in models:
        results["delta_aic"][model] = aic_values[model] - aic_min
        results["delta_bic"][model] = bic_values[model] - bic_min

    # Compute Akaike weights
    delta_aic = np.array([results["delta_aic"][m] for m in models])
    aic_weights = np.exp(-delta_aic / 2)
    aic_weights /= aic_weights.sum()

    for i, model in enumerate(models):
        results["aic_weights"][model] = aic_weights[i]

    # Compute BIC weights (Bayes factors approximation)
    delta_bic = np.array([results["delta_bic"][m] for m in models])
    bic_weights = np.exp(-delta_bic / 2)
    bic_weights /= bic_weights.sum()

    for i, model in enumerate(models):
        results["bic_weights"][model] = bic_weights[i]

    # Determine best model
    results["best_aic"] = min(aic_values, key=aic_values.get)
    results["best_bic"] = min(bic_values, key=bic_values.get)

    return results


def interpret_delta_aic(delta_aic: float) -> str:
    """Interpret the delta AIC value.

    Following Burnham & Anderson (2002):
    - Δ < 2: Substantial support for model
    - 4 < Δ < 7: Considerably less support
    - Δ > 10: Essentially no support

    Parameters
    ----------
    delta_aic : float
        AIC difference from best model

    Returns
    -------
    str
        Interpretation
    """
    if delta_aic < 2:
        return "Substantial support"
    elif delta_aic < 4:
        return "Moderate support"
    elif delta_aic < 7:
        return "Considerably less support"
    elif delta_aic < 10:
        return "Little support"
    else:
        return "Essentially no support"


def print_model_comparison(results: dict) -> None:
    """Print model comparison summary.

    Parameters
    ----------
    results : dict
        Output from compare_models()
    """
    print("\n" + "=" * 70)
    print("Model Comparison Summary")
    print("=" * 70)

    # Table header
    print(f"\n{'Model':<15} {'χ²':>8} {'AIC':>10} {'ΔAIC':>8} {'BIC':>10} {'ΔBIC':>8}")
    print("-" * 70)

    for model in results["models"]:
        chi2 = results["chi2"][model]
        aic = results["aic"][model]
        d_aic = results["delta_aic"][model]
        bic = results["bic"][model]
        d_bic = results["delta_bic"][model]

        print(f"{model:<15} {chi2:>8.2f} {aic:>10.2f} {d_aic:>8.2f} {bic:>10.2f} {d_bic:>8.2f}")

    print("\n" + "-" * 70)
    print("Model Weights (probability of being best):")
    print(f"\n{'Model':<15} {'AIC weight':>12} {'BIC weight':>12} {'Interpretation':<25}")
    print("-" * 70)

    for model in results["models"]:
        aic_w = results["aic_weights"][model]
        bic_w = results["bic_weights"][model]
        interp = interpret_delta_aic(results["delta_aic"][model])

        print(f"{model:<15} {aic_w:>12.3f} {bic_w:>12.3f} {interp:<25}")

    print("\n" + "-" * 70)
    print(f"Best model (AIC): {results['best_aic']}")
    print(f"Best model (BIC): {results['best_bic']}")


def bayes_factor(bic1: float, bic2: float) -> float:
    """Compute approximate Bayes factor from BIC values.

    B_{12} ≈ exp(-Δ_{12}/2) where Δ_{12} = BIC_1 - BIC_2

    Interpretation (Kass & Raftery 1995):
    - B < 1: Evidence for model 2
    - 1 < B < 3: Weak evidence for model 1
    - 3 < B < 20: Positive evidence for model 1
    - 20 < B < 150: Strong evidence for model 1
    - B > 150: Very strong evidence for model 1

    Parameters
    ----------
    bic1 : float
        BIC for model 1
    bic2 : float
        BIC for model 2

    Returns
    -------
    float
        Approximate Bayes factor B_{12}
    """
    return np.exp(-(bic1 - bic2) / 2)


def interpret_bayes_factor(bf: float) -> str:
    """Interpret Bayes factor.

    Parameters
    ----------
    bf : float
        Bayes factor

    Returns
    -------
    str
        Interpretation
    """
    if bf < 1:
        bf_inv = 1 / bf
        if bf_inv > 150:
            return f"Very strong evidence against (B = 1/{bf_inv:.1f})"
        elif bf_inv > 20:
            return f"Strong evidence against (B = 1/{bf_inv:.1f})"
        elif bf_inv > 3:
            return f"Positive evidence against (B = 1/{bf_inv:.1f})"
        else:
            return f"Weak evidence against (B = 1/{bf_inv:.1f})"
    else:
        if bf > 150:
            return f"Very strong evidence for (B = {bf:.1f})"
        elif bf > 20:
            return f"Strong evidence for (B = {bf:.1f})"
        elif bf > 3:
            return f"Positive evidence for (B = {bf:.1f})"
        else:
            return f"Weak evidence for (B = {bf:.1f})"
