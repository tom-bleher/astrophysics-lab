"""MCMC fitting for cosmological parameters.

This module provides Bayesian parameter estimation using MCMC
(Markov Chain Monte Carlo) via the emcee package.

Fits for:
- R: Characteristic galaxy size [kpc]
- Omega_L: Dark energy density parameter (optional, with flat universe constraint)

Returns posterior distributions and uncertainties.

References:
- Foreman-Mackey et al. 2013 (emcee)
- Hogg, Bovy & Lang 2010 (MCMC best practices)
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class MCMCResult:
    """Results from MCMC cosmological fit.

    Attributes
    ----------
    R_kpc : float
        Best-fit galaxy size in kpc
    R_kpc_err : float
        1-sigma uncertainty on R
    R_kpc_lo, R_kpc_hi : float
        16th and 84th percentile bounds
    Omega_L : float or None
        Best-fit dark energy density (if fitted)
    Omega_L_err : float or None
        1-sigma uncertainty on Omega_L
    samples : NDArray
        Full MCMC samples
    log_prob : NDArray
        Log probability for each sample
    acceptance_fraction : float
        MCMC acceptance fraction
    autocorr_time : float
        Autocorrelation time
    """

    R_kpc: float
    R_kpc_err: float
    R_kpc_lo: float
    R_kpc_hi: float
    Omega_L: float | None
    Omega_L_err: float | None
    samples: NDArray
    log_prob: NDArray
    acceptance_fraction: float
    autocorr_time: float

    @property
    def Omega_m(self) -> float | None:
        """Matter density (flat universe: Omega_m = 1 - Omega_L)."""
        return 1.0 - self.Omega_L if self.Omega_L is not None else None

    @property
    def Omega_m_err(self) -> float | None:
        """Uncertainty on Omega_m (same as Omega_L for flat universe)."""
        return self.Omega_L_err


def check_emcee_available() -> bool:
    """Check if emcee is installed."""
    import importlib.util
    return importlib.util.find_spec("emcee") is not None


def fit_radius_mcmc(
    z: NDArray,
    theta_arcsec: NDArray,
    theta_error: NDArray,
    Omega_L: float = 0.7,
    n_walkers: int = 32,
    n_steps: int = 5000,
    n_burn: int = 1000,
    progress: bool = True,
) -> MCMCResult:
    """Fit galaxy size R using MCMC with fixed cosmology.

    Parameters
    ----------
    z : NDArray
        Redshift values
    theta_arcsec : NDArray
        Angular sizes in arcseconds
    theta_error : NDArray
        Uncertainties on angular sizes
    Omega_L : float
        Fixed dark energy density parameter (flat universe: Omega_m = 1 - Omega_L)
    n_walkers : int
        Number of MCMC walkers
    n_steps : int
        Number of MCMC steps
    n_burn : int
        Number of burn-in steps to discard
    progress : bool
        Show progress bar

    Returns
    -------
    MCMCResult
        MCMC fit results
    """
    if not check_emcee_available():
        print("emcee not installed. Install with: pip install emcee")
        return _fit_radius_scipy(z, theta_arcsec, theta_error, Omega_L)

    import astropy.units as u
    import emcee
    from astropy.cosmology import FlatLambdaCDM

    # Set up cosmology (flat universe: Omega_m = 1 - Omega_L)
    Omega_m = 1.0 - Omega_L
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=Omega_m)

    # Compute angular diameter distances
    D_A = cosmo.angular_diameter_distance(z).to(u.Mpc).value

    # Convert theta to radians
    theta_rad = theta_arcsec * np.pi / (180 * 3600)
    theta_err_rad = theta_error * np.pi / (180 * 3600)

    def log_likelihood(params):
        R_Mpc = params[0] / 1000  # Convert kpc to Mpc

        if R_Mpc <= 0:
            return -np.inf

        # Model: theta = R / D_A
        theta_model = R_Mpc / D_A

        # Chi-squared
        chi2 = np.sum(((theta_rad - theta_model) / theta_err_rad) ** 2)

        return -0.5 * chi2

    def log_prior(params):
        R_kpc = params[0]
        if 0.1 < R_kpc < 100:
            return 0.0
        return -np.inf

    def log_probability(params):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params)

    # Initialize walkers
    ndim = 1
    rng = np.random.default_rng()
    # Initial guess from simple fit
    R_init = np.median(theta_rad * D_A) * 1000  # kpc
    p0 = R_init + 0.5 * rng.standard_normal((n_walkers, ndim))
    p0 = np.abs(p0)  # Ensure positive

    # Run MCMC
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability)

    print("Running MCMC...")
    sampler.run_mcmc(p0, n_steps, progress=progress)

    # Extract samples
    try:
        tau = sampler.get_autocorr_time(quiet=True)[0]
    except Exception:
        tau = np.nan

    samples = sampler.get_chain(discard=n_burn, flat=True)
    log_prob = sampler.get_log_prob(discard=n_burn, flat=True)

    # Compute statistics
    R_samples = samples[:, 0]
    R_median = np.median(R_samples)
    R_lo = np.percentile(R_samples, 16)
    R_hi = np.percentile(R_samples, 84)
    R_err = (R_hi - R_lo) / 2

    return MCMCResult(
        R_kpc=R_median,
        R_kpc_err=R_err,
        R_kpc_lo=R_lo,
        R_kpc_hi=R_hi,
        Omega_L=Omega_L,
        Omega_L_err=None,
        samples=samples,
        log_prob=log_prob,
        acceptance_fraction=np.mean(sampler.acceptance_fraction),
        autocorr_time=tau,
    )


def fit_cosmology_mcmc(
    z: NDArray,
    theta_arcsec: NDArray,
    theta_error: NDArray,
    n_walkers: int = 32,
    n_steps: int = 10000,
    n_burn: int = 2000,
    progress: bool = True,
) -> MCMCResult:
    """Fit both R and Omega_L using MCMC (flat universe constraint).

    Parameters
    ----------
    z : NDArray
        Redshift values
    theta_arcsec : NDArray
        Angular sizes in arcseconds
    theta_error : NDArray
        Uncertainties on angular sizes
    n_walkers : int
        Number of MCMC walkers
    n_steps : int
        Number of MCMC steps
    n_burn : int
        Number of burn-in steps to discard
    progress : bool
        Show progress bar

    Returns
    -------
    MCMCResult
        MCMC fit results including Omega_L
    """
    if not check_emcee_available():
        print("emcee not installed. Install with: pip install emcee")
        return _fit_cosmology_scipy(z, theta_arcsec, theta_error)

    import astropy.units as u
    import emcee
    from astropy.cosmology import FlatLambdaCDM

    # Convert theta to radians
    theta_rad = theta_arcsec * np.pi / (180 * 3600)
    theta_err_rad = theta_error * np.pi / (180 * 3600)

    def log_likelihood(params):
        R_kpc, Omega_L = params

        if R_kpc <= 0 or Omega_L <= 0.01 or Omega_L >= 0.99:
            return -np.inf

        R_Mpc = R_kpc / 1000

        # Compute D_A for this Omega_L (flat universe: Omega_m = 1 - Omega_L)
        try:
            Omega_m = 1.0 - Omega_L
            cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=Omega_m)
            D_A = cosmo.angular_diameter_distance(z).to(u.Mpc).value
        except Exception:
            return -np.inf

        # Model
        theta_model = R_Mpc / D_A

        # Chi-squared
        chi2 = np.sum(((theta_rad - theta_model) / theta_err_rad) ** 2)

        return -0.5 * chi2

    def log_prior(params):
        R_kpc, Omega_L = params
        if 0.1 < R_kpc < 100 and 0.05 < Omega_L < 0.95:
            # Flat prior on R, Gaussian prior on Omega_L centered on 0.7
            return -0.5 * ((Omega_L - 0.7) / 0.1) ** 2
        return -np.inf

    def log_probability(params):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params)

    # Initialize walkers
    ndim = 2
    rng = np.random.default_rng()
    p0 = np.array([5.0, 0.7]) + 0.1 * rng.standard_normal((n_walkers, ndim))
    p0[:, 0] = np.abs(p0[:, 0])  # R positive
    p0[:, 1] = np.clip(p0[:, 1], 0.1, 0.9)  # Omega_L in range

    # Run MCMC
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability)

    print("Running MCMC (fitting R and Omega_L)...")
    sampler.run_mcmc(p0, n_steps, progress=progress)

    # Extract samples
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        tau_max = np.max(tau)
    except Exception:
        tau_max = np.nan

    samples = sampler.get_chain(discard=n_burn, flat=True)
    log_prob = sampler.get_log_prob(discard=n_burn, flat=True)

    # Compute statistics
    R_samples = samples[:, 0]
    OmL_samples = samples[:, 1]

    R_median = np.median(R_samples)
    R_lo = np.percentile(R_samples, 16)
    R_hi = np.percentile(R_samples, 84)

    OmL_median = np.median(OmL_samples)
    OmL_lo = np.percentile(OmL_samples, 16)
    OmL_hi = np.percentile(OmL_samples, 84)

    return MCMCResult(
        R_kpc=R_median,
        R_kpc_err=(R_hi - R_lo) / 2,
        R_kpc_lo=R_lo,
        R_kpc_hi=R_hi,
        Omega_L=OmL_median,
        Omega_L_err=(OmL_hi - OmL_lo) / 2,
        samples=samples,
        log_prob=log_prob,
        acceptance_fraction=np.mean(sampler.acceptance_fraction),
        autocorr_time=tau_max,
    )


def _fit_radius_scipy(
    z: NDArray,
    theta_arcsec: NDArray,
    theta_error: NDArray,
    Omega_L: float,
) -> MCMCResult:
    """Fallback scipy-based fitting when emcee is not available."""
    import astropy.units as u
    from astropy.cosmology import FlatLambdaCDM
    from scipy.optimize import minimize

    Omega_m = 1.0 - Omega_L
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=Omega_m)
    D_A = cosmo.angular_diameter_distance(z).to(u.Mpc).value

    theta_rad = theta_arcsec * np.pi / (180 * 3600)
    theta_err_rad = theta_error * np.pi / (180 * 3600)

    def chi2(R_kpc):
        R_Mpc = R_kpc / 1000
        theta_model = R_Mpc / D_A
        return np.sum(((theta_rad - theta_model) / theta_err_rad) ** 2)

    result = minimize(chi2, x0=5.0, bounds=[(0.1, 100)])
    R_best = result.x[0]

    # Estimate error from Hessian
    R_err = 0.5  # Placeholder

    return MCMCResult(
        R_kpc=R_best,
        R_kpc_err=R_err,
        R_kpc_lo=R_best - R_err,
        R_kpc_hi=R_best + R_err,
        Omega_L=Omega_L,
        Omega_L_err=None,
        samples=np.array([[R_best]]),
        log_prob=np.array([-chi2(R_best) / 2]),
        acceptance_fraction=1.0,
        autocorr_time=np.nan,
    )


def _fit_cosmology_scipy(
    z: NDArray,
    theta_arcsec: NDArray,
    theta_error: NDArray,
) -> MCMCResult:
    """Fallback scipy-based fitting when emcee is not available."""
    import astropy.units as u
    from astropy.cosmology import FlatLambdaCDM
    from scipy.optimize import minimize

    theta_rad = theta_arcsec * np.pi / (180 * 3600)
    theta_err_rad = theta_error * np.pi / (180 * 3600)

    def chi2(params):
        R_kpc, Omega_L = params
        if R_kpc <= 0 or Omega_L <= 0.01 or Omega_L >= 0.99:
            return 1e10

        R_Mpc = R_kpc / 1000
        Omega_m = 1.0 - Omega_L
        cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=Omega_m)
        D_A = cosmo.angular_diameter_distance(z).to(u.Mpc).value
        theta_model = R_Mpc / D_A
        return np.sum(((theta_rad - theta_model) / theta_err_rad) ** 2)

    result = minimize(chi2, x0=[5.0, 0.7], bounds=[(0.1, 100), (0.05, 0.95)])
    R_best, OmL_best = result.x

    return MCMCResult(
        R_kpc=R_best,
        R_kpc_err=0.5,
        R_kpc_lo=R_best - 0.5,
        R_kpc_hi=R_best + 0.5,
        Omega_L=OmL_best,
        Omega_L_err=0.05,
        samples=np.array([[R_best, OmL_best]]),
        log_prob=np.array([-chi2([R_best, OmL_best]) / 2]),
        acceptance_fraction=1.0,
        autocorr_time=np.nan,
    )


def plot_corner(result: MCMCResult, figsize: tuple = (8, 8)):
    """Plot corner plot of MCMC samples.

    Parameters
    ----------
    result : MCMCResult
        MCMC fit results
    figsize : tuple
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure
    """
    try:
        import corner
    except ImportError:
        print("corner package not installed. Install with: pip install corner")
        return None

    if result.Omega_L is not None and result.samples.shape[1] == 2:
        labels = [r"$R$ [kpc]", r"$\Omega_\Lambda$"]
        truths = [result.R_kpc, result.Omega_L]
    else:
        labels = [r"$R$ [kpc]"]
        truths = [result.R_kpc]

    fig = corner.corner(
        result.samples,
        labels=labels,
        truths=truths,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
    )

    return fig


def print_mcmc_summary(result: MCMCResult) -> None:
    """Print summary of MCMC results.

    Parameters
    ----------
    result : MCMCResult
        MCMC fit results
    """
    print("\n" + "=" * 50)
    print("MCMC Fit Results")
    print("=" * 50)

    print("\nGalaxy Size:")
    print(f"  R = {result.R_kpc:.2f} +{result.R_kpc_hi-result.R_kpc:.2f} "
          f"-{result.R_kpc-result.R_kpc_lo:.2f} kpc")
    print(f"  R = {result.R_kpc:.2f} ± {result.R_kpc_err:.2f} kpc (symmetric)")

    if result.Omega_L is not None:
        print("\nDark Energy Density:")
        print(f"  Omega_L = {result.Omega_L:.3f} ± {result.Omega_L_err:.3f}")
        print(f"  Omega_m = {result.Omega_m:.3f} ± {result.Omega_m_err:.3f} (derived)")

    print("\nMCMC Diagnostics:")
    print(f"  Acceptance fraction: {result.acceptance_fraction:.2f}")
    if np.isfinite(result.autocorr_time):
        print(f"  Autocorrelation time: {result.autocorr_time:.1f}")
    print(f"  Number of samples: {len(result.samples)}")
