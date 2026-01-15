"""Extended cosmological analysis module.

This module extends the base scientific.py with:
- MCMC parameter estimation with emcee
- Uncertainty propagation
- Tolman surface brightness test
- Model comparison statistics

Example usage:
    from cosmology import fit_cosmology_mcmc, tolman_test
    from cosmology.plotting import plot_corner, plot_theta_z_fit

    # Fit cosmological parameters
    result = fit_cosmology_mcmc(z, theta, theta_err)
    print(f"R = {result['R_kpc']:.2f} ± {result['R_kpc_err']:.2f} kpc")

    # Tolman test
    n, n_err = tolman_test.fit_tolman_exponent(z, surface_brightness)
"""

from cosmology.mcmc_fitting import (
    MCMCResult,
    fit_cosmology_mcmc,
    fit_radius_mcmc,
)
from cosmology.model_comparison import (
    compare_models,
    compute_aic,
    compute_bic,
)
from cosmology.tolman_test import (
    fit_tolman_exponent,
    plot_tolman_test,
    surface_brightness_vs_redshift,
)

__all__ = [
    "MCMCResult",
    "compare_models",
    "compute_aic",
    "compute_bic",
    "fit_cosmology_mcmc",
    "fit_radius_mcmc",
    "fit_tolman_exponent",
    "plot_tolman_test",
    "surface_brightness_vs_redshift",
]
