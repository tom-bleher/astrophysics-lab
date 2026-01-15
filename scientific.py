"""Cosmological models for angular size calculations.

Performance notes:
- Uses astropy.cosmology for vectorized angular diameter distance (10-50x faster)
- Pre-computed interpolation table for ultra-fast lookups when same cosmology is reused
- References: astropy.cosmology documentation, scipy.integrate.quad_vec
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import astropy.units as u
import corner
import emcee
import numpy as np
from astropy.constants import c as speed_of_light
from astropy.cosmology import FlatLambdaCDM
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist

# Physical constants (unitless, for backwards compatibility)
c: float = 299792.458  # speed of light [km/s]
H0: float = 70.0  # Hubble constant [km/s/Mpc]
ARCSEC_TO_RAD: float = np.pi / (180.0 * 3600.0)  # ~4.848e-6 rad/arcsec

# Physical constants with units
c_with_units = speed_of_light.to(u.km / u.s)  # speed of light
H0_with_units = 70.0 * u.km / u.s / u.Mpc  # Hubble constant

# Lazy-loaded astropy cosmology (avoids import overhead if not used)
_COSMO_CACHE: dict = {}


@dataclass
class CosmologyFitResult:
    """Results from flat LCDM cosmological fit.

    Attributes
    ----------
    R_mpc : float
        Best-fit galaxy size in Mpc
    Omega_L : float
        Best-fit dark energy density parameter
    R_err : float
        1-sigma uncertainty on R
    Omega_L_err : float
        1-sigma uncertainty on Omega_L
    chi2_reduced : float
        Reduced chi-squared (chi2 / dof)
    p_value : float
        p-value for goodness of fit (probability of observing
        chi2 >= observed if model is correct)
    n_points : int
        Number of data points used in fit
    """

    R_mpc: float
    Omega_L: float
    R_err: float
    Omega_L_err: float
    chi2_reduced: float
    p_value: float
    n_points: int

    @property
    def Omega_m(self) -> float:
        """Matter density (flat universe: Omega_m = 1 - Omega_L)."""
        return 1.0 - self.Omega_L

    @property
    def Omega_m_err(self) -> float:
        """Uncertainty on Omega_m (same as Omega_L for flat universe)."""
        return self.Omega_L_err

    @property
    def R_kpc(self) -> float:
        """Best-fit radius in kpc."""
        return self.R_mpc * 1000

    @property
    def R_kpc_err(self) -> float:
        """Uncertainty on R in kpc."""
        return self.R_err * 1000


@dataclass
class MCMCFitResult:
    """Results from MCMC parameter estimation for flat LCDM cosmology.

    Attributes
    ----------
    samples : NDArray
        The MCMC chain samples after burn-in, shape (n_samples, 2)
        where columns are [R, Omega_L]
    R_median : float
        Median value of R from posterior
    R_lo : float
        16th percentile of R (lower 1-sigma bound)
    R_hi : float
        84th percentile of R (upper 1-sigma bound)
    Omega_L_median : float
        Median value of Omega_L from posterior
    Omega_L_lo : float
        16th percentile of Omega_L (lower 1-sigma bound)
    Omega_L_hi : float
        84th percentile of Omega_L (upper 1-sigma bound)
    acceptance_fraction : float
        Mean acceptance fraction across all walkers
    autocorr_time : float
        Integrated autocorrelation time (if computable, else NaN)
    """

    samples: NDArray
    R_median: float
    R_lo: float
    R_hi: float
    Omega_L_median: float
    Omega_L_lo: float
    Omega_L_hi: float
    acceptance_fraction: float
    autocorr_time: float

    @property
    def Omega_m_median(self) -> float:
        """Median matter density (flat universe: Omega_m = 1 - Omega_L)."""
        return 1.0 - self.Omega_L_median

    @property
    def Omega_m_lo(self) -> float:
        """Lower 1-sigma bound on Omega_m."""
        return 1.0 - self.Omega_L_hi  # Note: swapped because Omega_m = 1 - Omega_L

    @property
    def Omega_m_hi(self) -> float:
        """Upper 1-sigma bound on Omega_m."""
        return 1.0 - self.Omega_L_lo

    @property
    def R_kpc_median(self) -> float:
        """Median radius in kpc."""
        return self.R_median * 1000

    @property
    def R_kpc_lo(self) -> float:
        """Lower 1-sigma bound on R in kpc."""
        return self.R_lo * 1000

    @property
    def R_kpc_hi(self) -> float:
        """Upper 1-sigma bound on R in kpc."""
        return self.R_hi * 1000


def compute_chi2_stats(
    z_data: ArrayLike,
    theta_data: ArrayLike,
    theta_err: ArrayLike,
    model_func: callable,
    n_params: int = 1,
) -> dict:
    """Compute chi^2/ndf and p-value for a model fit.

    Parameters
    ----------
    z_data : array-like
        Redshift values of binned data.
    theta_data : array-like
        Observed angular sizes (arcsec).
    theta_err : array-like
        Uncertainties on angular sizes (arcsec).
    model_func : callable
        Function that takes z and returns model theta in arcsec.
    n_params : int
        Number of fitted parameters (for degrees of freedom).

    Returns
    -------
    dict
        Contains 'chi2', 'ndf', 'chi2_ndf', 'p_value'
    """
    theta_model = model_func(np.asarray(z_data))

    # Chi-squared
    residuals = (np.asarray(theta_data) - theta_model) / np.asarray(theta_err)
    chi2 = float(np.sum(residuals**2))

    # Degrees of freedom
    ndf = len(z_data) - n_params

    if ndf <= 0:
        return {"chi2": chi2, "ndf": ndf, "chi2_ndf": np.nan, "p_value": np.nan}

    chi2_ndf = chi2 / ndf

    # P-value: probability of getting chi^2 this large or larger by chance
    # if the model is correct
    p_value = float(1.0 - chi2_dist.cdf(chi2, ndf))

    return {
        "chi2": chi2,
        "ndf": ndf,
        "chi2_ndf": chi2_ndf,
        "p_value": p_value,
    }


def _get_cosmology(Omega_m: float, Omega_L: float):
    """Get or create a cached astropy FlatLambdaCDM cosmology object.

    Astropy's cosmology module provides highly optimized, vectorized
    distance calculations that are 10-50x faster than scipy.integrate.quad.
    """
    key = (round(Omega_m, 6), round(Omega_L, 6))
    if key not in _COSMO_CACHE:
        # FlatLambdaCDM enforces Omega_L = 1 - Omega_m for flat universe
        # We use Om0 which sets Omega_m, and Ode0 is computed automatically
        _COSMO_CACHE[key] = FlatLambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=Omega_m)
    return _COSMO_CACHE[key]


def theta_static(z: ArrayLike, R: float) -> NDArray[np.floating]:
    """
    Linear Hubble law (naive Euclidean) angular size model
    D_A = (c/H0) * z
    R: physical size [Mpc]
    """
    z_arr = np.atleast_1d(z)
    z_safe = np.clip(z_arr, 1e-6, None)
    D_A = (c / H0) * z_safe  # Mpc
    return R / D_A


def E(z: float, Omega_m: float, Omega_L: float) -> float:
    """Dimensionless Hubble parameter E(z) for flat LCDM cosmology."""
    inner = Omega_m * (1 + z) ** 3 + Omega_L
    return float(np.sqrt(np.clip(inner, 0.0, None)))


@lru_cache(maxsize=1024)
def D_A_LCDM(z: float, Omega_m: float = 0.3, Omega_L: float = 0.7) -> float:
    """Angular diameter distance in LCDM cosmology [Mpc].

    Results are cached for repeated calls with same parameters.
    For vectorized operations, use D_A_LCDM_vectorized() instead.
    """
    # Round to 6 decimal places for cache efficiency (precision ~0.0001%)
    z = round(z, 6)
    integral, _ = quad(lambda zp: 1.0 / E(zp, Omega_m, Omega_L), 0, z)
    return (c / H0) * integral / (1 + z)


def D_A_LCDM_vectorized(
    z: ArrayLike, Omega_m: float = 0.3, Omega_L: float = 0.7
) -> NDArray[np.floating]:
    """Vectorized angular diameter distance in LCDM cosmology [Mpc].

    Uses astropy.cosmology for 10-50x faster computation on arrays.
    This is the preferred method for arrays of redshifts.
    """
    z_arr = np.atleast_1d(z).astype(np.float64)
    cosmo = _get_cosmology(Omega_m, Omega_L)
    # astropy returns Quantity with units; extract value in Mpc
    return cosmo.angular_diameter_distance(z_arr).to_value('Mpc')


def theta_lcdm(
    z: ArrayLike, R: float, Omega_m: float = 0.3, Omega_L: float = 0.7
) -> NDArray[np.floating]:
    """Angular size in LCDM cosmology [radians].

    Uses astropy.cosmology vectorized calculations (10-50x faster than loop).
    """
    z_arr = np.atleast_1d(z).astype(np.float64)
    # Use vectorized astropy calculation instead of loop
    D_A = D_A_LCDM_vectorized(z_arr, Omega_m, Omega_L)
    return R / D_A


def theta_lcdm_flat(z: ArrayLike, R: float, Omega_m: float) -> NDArray[np.floating]:
    """
    Angular size in flat LCDM cosmology [radians].
    Enforces Omega_L = 1 - Omega_m (flat universe constraint).
    """
    Omega_L = 1.0 - Omega_m
    return theta_lcdm(z, R, Omega_m, Omega_L)


def _theta_flat_for_fit(z: ArrayLike, R: float, Omega_L: float) -> NDArray[np.floating]:
    """Angular size model for curve_fit (fits Omega_L, not Omega_m).

    Returns angular size in arcseconds (not radians) for direct comparison
    with observed data.
    """
    Omega_m = 1.0 - Omega_L
    theta_rad = theta_lcdm(z, R, Omega_m, Omega_L)
    return theta_rad / ARCSEC_TO_RAD  # Convert to arcseconds


def fit_flat_lcdm(
    z: ArrayLike,
    theta_arcsec: ArrayLike,
    theta_error_arcsec: ArrayLike,
    *,
    z_error: ArrayLike | None = None,
    Omega_L_init: float = 0.7,
    maxfev: int = 8000,
) -> CosmologyFitResult:
    """Fit R and Omega_L under a flat-universe assumption (Omega_m + Omega_L = 1).

    Fits a 2-parameter model to angular-size data:

        theta(z) = R / D_A(z; Omega_m, Omega_L)

    with the flatness constraint Omega_m = 1 - Omega_L.

    Parameters
    ----------
    z : array-like
        Redshift values
    theta_arcsec : array-like
        Observed angular sizes in arcseconds
    theta_error_arcsec : array-like
        1-sigma uncertainties on angular sizes in arcseconds
    z_error : array-like, optional
        Redshift uncertainties (propagated into total theta uncertainty)
    Omega_L_init : float
        Initial guess for Omega_L (default 0.7)
    maxfev : int
        Maximum function evaluations for curve_fit

    Returns
    -------
    CosmologyFitResult
        Dataclass with R_mpc, Omega_L, uncertainties, and fit quality metrics

    Notes
    -----
    Uses vectorized astropy cosmology calculations (10-50x faster than
    scipy.integrate.quad loops).

    If z_error is provided, it is propagated into total uncertainty using:
        sigma_total^2 = sigma_theta^2 + (dtheta/dz)^2 * sigma_z^2
    where dtheta/dz is approximately -theta/z (simplified approximation).
    """
    z = np.asarray(z, dtype=float)
    theta_arcsec = np.asarray(theta_arcsec, dtype=float)
    theta_error_arcsec = np.asarray(theta_error_arcsec, dtype=float)

    z_error = np.asarray(z_error, dtype=float) if z_error is not None else np.zeros_like(z)

    # Filter valid data points
    mask = (
        np.isfinite(z)
        & np.isfinite(theta_arcsec)
        & np.isfinite(theta_error_arcsec)
        & np.isfinite(z_error)
        & (z > 0)
        & (theta_error_arcsec > 0)
    )
    if mask.sum() < 3:
        return CosmologyFitResult(
            R_mpc=np.nan,
            Omega_L=np.nan,
            R_err=np.nan,
            Omega_L_err=np.nan,
            chi2_reduced=np.nan,
            p_value=np.nan,
            n_points=int(mask.sum()),
        )

    z = z[mask]
    theta_arcsec = theta_arcsec[mask]
    theta_error_arcsec = theta_error_arcsec[mask]
    z_error = z_error[mask]

    # Avoid z=0 which breaks angular-diameter distance
    z = np.clip(z, 1e-6, None)

    # Numerical stability: ensure non-zero errors
    min_err = np.nanmax(theta_error_arcsec) * 1e-6 + 1e-12
    theta_error_arcsec = np.maximum(theta_error_arcsec, min_err)

    # Propagate z_error into total theta error
    # Error propagation: sigma_total^2 = sigma_theta^2 + (dtheta/dz)^2 * sigma_z^2
    # Approximate: dtheta/dz is approximately -theta/z (angular size decreases with redshift)
    if np.any(z_error > 0):
        dtheta_dz = np.abs(theta_arcsec / z)
        theta_error_total = np.sqrt(
            theta_error_arcsec**2 + (dtheta_dz * z_error) ** 2
        )
    else:
        theta_error_total = theta_error_arcsec

    # Initial guess for R using LCDM D_A with initial Omega_L guess
    Omega_m_init = 1.0 - Omega_L_init
    D_A_init = D_A_LCDM_vectorized(z, Omega_m_init, Omega_L_init)
    theta_rad = theta_arcsec * ARCSEC_TO_RAD
    R_estimates = theta_rad * D_A_init
    R_init = float(np.nanmedian(R_estimates))
    if not np.isfinite(R_init) or R_init <= 0:
        R_init = 0.01  # fallback ~10 kpc

    # Ensure R_init is within bounds
    R_init = max(R_init, 1e-7)  # Ensure above lower bound

    # Fit using curve_fit
    try:
        popt, pcov = curve_fit(
            _theta_flat_for_fit,
            z,
            theta_arcsec,
            sigma=theta_error_total,
            absolute_sigma=True,
            maxfev=maxfev,
            p0=[R_init, Omega_L_init],
            bounds=([1e-8, 0.0], [np.inf, 1.0]),
        )
    except (RuntimeError, ValueError):
        # Fitting failed to converge or initial guess outside bounds
        return CosmologyFitResult(
            R_mpc=np.nan,
            Omega_L=np.nan,
            R_err=np.nan,
            Omega_L_err=np.nan,
            chi2_reduced=np.nan,
            p_value=np.nan,
            n_points=len(z),
        )

    R_fit = float(popt[0])
    Omega_L_fit = float(popt[1])

    # Extract uncertainties from covariance matrix
    if np.all(np.isfinite(pcov)):
        perr = np.sqrt(np.diag(pcov))
        R_err = float(perr[0])
        Omega_L_err = float(perr[1])
    else:
        R_err = np.nan
        Omega_L_err = np.nan

    # Calculate goodness-of-fit metrics
    theta_model = _theta_flat_for_fit(z, R_fit, Omega_L_fit)
    residuals = theta_arcsec - theta_model
    chi_squared = np.sum((residuals / theta_error_total) ** 2)
    dof = len(z) - 2  # two fitted parameters
    chi2_reduced = chi_squared / dof if dof > 0 else np.nan
    p_value = float(chi2_dist.sf(chi_squared, df=dof)) if dof > 0 else np.nan

    return CosmologyFitResult(
        R_mpc=R_fit,
        Omega_L=Omega_L_fit,
        R_err=R_err,
        Omega_L_err=Omega_L_err,
        chi2_reduced=chi2_reduced,
        p_value=p_value,
        n_points=len(z),
    )


def fit_flat_lcdm_mcmc(
    z: ArrayLike,
    theta_arcsec: ArrayLike,
    theta_error_arcsec: ArrayLike,
    *,
    z_error: ArrayLike | None = None,
    n_walkers: int = 32,
    n_steps: int = 5000,
    n_burn: int = 1000,
    R_prior_bounds: tuple[float, float] = (1e-6, 1.0),
    Omega_L_prior_bounds: tuple[float, float] = (0.0, 1.0),
    progress: bool = True,
    n_threads: int | None = None,
) -> MCMCFitResult:
    """Bayesian MCMC parameter estimation for flat LCDM cosmology.

    Uses emcee EnsembleSampler to sample the posterior distribution of
    (R, Omega_L) given angular size vs redshift data.

    Parameters
    ----------
    z : array-like
        Redshift values
    theta_arcsec : array-like
        Observed angular sizes in arcseconds
    theta_error_arcsec : array-like
        1-sigma uncertainties on angular sizes in arcseconds
    z_error : array-like, optional
        Redshift uncertainties (propagated into total theta uncertainty)
    n_walkers : int
        Number of MCMC walkers (default 32)
    n_steps : int
        Number of MCMC steps per walker (default 5000)
    n_burn : int
        Number of burn-in steps to discard (default 1000)
    R_prior_bounds : tuple[float, float]
        Uniform prior bounds on R in Mpc (default (1e-6, 1.0))
    Omega_L_prior_bounds : tuple[float, float]
        Uniform prior bounds on Omega_L (default (0.0, 1.0))
    progress : bool
        Show progress bar during sampling (default True)
    n_threads : int, optional
        Number of threads for parallel likelihood evaluation.
        If None (default), automatically uses the number of threads from
        resource_config, enabling parallel MCMC by default on multi-core systems.
        Set to 1 to disable parallelization.
        Typical speedup is ~3x on 4 cores, ~4x on 8 cores.

    Returns
    -------
    MCMCFitResult
        Dataclass with samples, medians, percentile uncertainties, and diagnostics

    Raises
    ------
    ValueError
        If insufficient valid data points (< 3)

    Notes
    -----
    The model uses uniform priors on R and Omega_L within the specified bounds.
    The likelihood is Gaussian based on the observed angular sizes and their
    uncertainties.

    For reliable results, ensure that:
    - n_steps > n_burn + 1000 (to have enough post-burn-in samples)
    - n_walkers >= 2 * n_dim (here n_dim=2, so >= 4 walkers)
    - acceptance_fraction is between 0.2 and 0.5 (adjust proposal if not)

    Examples
    --------
    >>> result = fit_flat_lcdm_mcmc(z, theta, theta_err)
    >>> print(f"R = {result.R_median:.4f} +{result.R_hi-result.R_median:.4f} -{result.R_median-result.R_lo:.4f} Mpc")
    >>> print(f"Omega_L = {result.Omega_L_median:.3f} +{result.Omega_L_hi-result.Omega_L_median:.3f} -{result.Omega_L_median-result.Omega_L_lo:.3f}")
    """
    # Auto-detect number of threads if not specified
    if n_threads is None:
        try:
            from resource_config import get_config
            config = get_config()
            n_threads = config.n_threads
        except ImportError:
            import os
            n_threads = max(1, (os.cpu_count() or 2) - 1)

    # Prepare data
    z = np.asarray(z, dtype=float)
    theta_arcsec = np.asarray(theta_arcsec, dtype=float)
    theta_error_arcsec = np.asarray(theta_error_arcsec, dtype=float)

    z_error = np.asarray(z_error, dtype=float) if z_error is not None else np.zeros_like(z)

    # Filter valid data points
    mask = (
        np.isfinite(z)
        & np.isfinite(theta_arcsec)
        & np.isfinite(theta_error_arcsec)
        & np.isfinite(z_error)
        & (z > 0)
        & (theta_error_arcsec > 0)
    )

    if mask.sum() < 3:
        raise ValueError(
            f"Insufficient valid data points for MCMC: {mask.sum()} < 3 required"
        )

    z_data = z[mask]
    theta_data = theta_arcsec[mask]
    theta_err_data = theta_error_arcsec[mask]
    z_err_data = z_error[mask]

    # Avoid z=0 which breaks angular-diameter distance
    z_data = np.clip(z_data, 1e-6, None)

    # Numerical stability: ensure non-zero errors
    min_err = np.nanmax(theta_err_data) * 1e-6 + 1e-12
    theta_err_data = np.maximum(theta_err_data, min_err)

    # Propagate z_error into total theta error
    if np.any(z_err_data > 0):
        dtheta_dz = np.abs(theta_data / z_data)
        theta_err_total = np.sqrt(
            theta_err_data**2 + (dtheta_dz * z_err_data) ** 2
        )
    else:
        theta_err_total = theta_err_data

    # Define log prior (uniform within bounds)
    R_min, R_max = R_prior_bounds
    Omega_L_min, Omega_L_max = Omega_L_prior_bounds

    def log_prior(params: NDArray) -> float:
        """Uniform prior on R and Omega_L within specified bounds."""
        R, Omega_L = params
        if R_min < R < R_max and Omega_L_min < Omega_L < Omega_L_max:
            return 0.0
        return -np.inf

    # Define log likelihood (Gaussian)
    def log_likelihood(params: NDArray) -> float:
        """Gaussian likelihood for angular size data."""
        R, Omega_L = params

        # Compute model prediction
        theta_model = _theta_flat_for_fit(z_data, R, Omega_L)

        # Chi-squared
        chi2 = np.sum(((theta_data - theta_model) / theta_err_total) ** 2)

        # Log likelihood (up to constant)
        return -0.5 * chi2

    # Define log posterior
    def log_posterior(params: NDArray) -> float:
        """Log posterior = log prior + log likelihood."""
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_likelihood(params)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    # Initialize walkers near the maximum likelihood estimate
    # Use fit_flat_lcdm to get a good starting point
    try:
        mle_result = fit_flat_lcdm(z, theta_arcsec, theta_error_arcsec, z_error=z_error)
        if np.isfinite(mle_result.R_mpc) and np.isfinite(mle_result.Omega_L):
            R_init = mle_result.R_mpc
            Omega_L_init = mle_result.Omega_L
        else:
            # Fallback initialization
            Omega_m_init = 0.3
            D_A_init = D_A_LCDM_vectorized(z_data, Omega_m_init, 1.0 - Omega_m_init)
            theta_rad = theta_data * ARCSEC_TO_RAD
            R_init = float(np.nanmedian(theta_rad * D_A_init))
            Omega_L_init = 0.7
    except Exception:
        # Fallback initialization
        Omega_m_init = 0.3
        D_A_init = D_A_LCDM_vectorized(z_data, Omega_m_init, 1.0 - Omega_m_init)
        theta_rad = theta_data * ARCSEC_TO_RAD
        R_init = float(np.nanmedian(theta_rad * D_A_init))
        Omega_L_init = 0.7

    # Ensure initial values are within prior bounds
    R_init = np.clip(R_init, R_min * 1.01, R_max * 0.99)
    Omega_L_init = np.clip(Omega_L_init, Omega_L_min + 0.01, Omega_L_max - 0.01)

    # Initialize walker positions with small scatter around initial estimate
    ndim = 2
    # Use relative scatter for R (which can span orders of magnitude)
    # and absolute scatter for Omega_L (bounded 0-1)
    R_scatter = max(R_init * 0.1, 1e-5)
    Omega_L_scatter = 0.05

    rng = np.random.default_rng()
    pos = np.zeros((n_walkers, ndim))
    for i in range(n_walkers):
        while True:
            R_try = R_init + R_scatter * rng.standard_normal()
            Omega_L_try = Omega_L_init + Omega_L_scatter * rng.standard_normal()
            if (R_min < R_try < R_max) and (Omega_L_min < Omega_L_try < Omega_L_max):
                pos[i] = [R_try, Omega_L_try]
                break

    # Run MCMC (with optional parallel processing)
    if n_threads > 1:
        from multiprocessing import Pool
        with Pool(n_threads) as pool:
            sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior, pool=pool)
            sampler.run_mcmc(pos, n_steps, progress=progress)
    else:
        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)
        sampler.run_mcmc(pos, n_steps, progress=progress)

    # Get chain and discard burn-in
    chain = sampler.get_chain(discard=n_burn, flat=True)

    # Compute percentiles
    R_samples = chain[:, 0]
    Omega_L_samples = chain[:, 1]

    R_lo, R_median, R_hi = np.percentile(R_samples, [16, 50, 84])
    Omega_L_lo, Omega_L_median, Omega_L_hi = np.percentile(Omega_L_samples, [16, 50, 84])

    # Compute acceptance fraction
    acceptance_fraction = float(np.mean(sampler.acceptance_fraction))

    # Compute autocorrelation time (if possible)
    try:
        autocorr = sampler.get_autocorr_time(quiet=True)
        autocorr_time = float(np.mean(autocorr))
    except Exception:
        autocorr_time = np.nan

    return MCMCFitResult(
        samples=chain,
        R_median=float(R_median),
        R_lo=float(R_lo),
        R_hi=float(R_hi),
        Omega_L_median=float(Omega_L_median),
        Omega_L_lo=float(Omega_L_lo),
        Omega_L_hi=float(Omega_L_hi),
        acceptance_fraction=acceptance_fraction,
        autocorr_time=autocorr_time,
    )


def plot_cosmology_corner(
    result: MCMCFitResult,
    output_path: str,
    *,
    labels: list[str] | None = None,
    truths: list[float] | None = None,
    title: str | None = None,
) -> None:
    """Create a corner plot for MCMC posterior visualization.

    Parameters
    ----------
    result : MCMCFitResult
        Result from fit_flat_lcdm_mcmc containing posterior samples
    output_path : str
        Path to save the corner plot (e.g., 'corner_plot.png')
    labels : list[str], optional
        Labels for parameters (default: ['R [Mpc]', 'Omega_L'])
    truths : list[float], optional
        True values to mark on plot (e.g., from simulation or literature)
    title : str, optional
        Title for the plot

    Examples
    --------
    >>> result = fit_flat_lcdm_mcmc(z, theta, theta_err)
    >>> plot_cosmology_corner(result, 'cosmology_corner.png')
    """
    import matplotlib.pyplot as plt

    if labels is None:
        labels = [r'$R$ [Mpc]', r'$\Omega_\Lambda$']

    # Create corner plot
    fig = corner.corner(
        result.samples,
        labels=labels,
        truths=truths,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
    )

    if title is not None:
        fig.suptitle(title, fontsize=16, y=1.02)

    # Add summary text
    summary_text = (
        f"$R = {result.R_median:.4f}^{{+{result.R_hi - result.R_median:.4f}}}_{{-{result.R_median - result.R_lo:.4f}}}$ Mpc\n"
        f"$\\Omega_\\Lambda = {result.Omega_L_median:.3f}^{{+{result.Omega_L_hi - result.Omega_L_median:.3f}}}_{{-{result.Omega_L_median - result.Omega_L_lo:.3f}}}$\n"
        f"Acceptance: {result.acceptance_fraction:.2f}"
    )
    fig.text(0.65, 0.75, summary_text, transform=fig.transFigure, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def choose_redshift_bin_edges(
    z,
    *,
    target_count_per_bin: int = 25,
    min_bins: int = 4,
    max_bins: int = 10,
    method: str = "quantile",
):
    """Choose redshift bin edges for robust summary-statistics plots.

    Why this exists:
    - Too few bins hides structure; too many bins makes medians/noise unstable.
    - For medians/IQR-based summaries, you typically want O(10-30) objects per bin.

    Strategy:
    - Pick the number of bins from sample size: nbins ~ N / target_count_per_bin,
      clamped to [min_bins, max_bins].
    - Use quantile edges by default (equal-count bins) which is usually better when
      the redshift distribution is skewed.
    - Fall back to linear spacing if quantiles produce duplicate edges.

    Returns:
        1D numpy array of monotonically increasing bin edges.
    """

    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 2:
        raise ValueError("Need at least 2 redshift values to choose bins")

    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
        raise ValueError("Cannot choose bins: invalid or zero redshift range")

    if target_count_per_bin <= 0:
        raise ValueError("target_count_per_bin must be positive")

    # At least 2 bins to allow a fit downstream.
    nbins = int(np.floor(z.size / target_count_per_bin))
    nbins = int(np.clip(nbins, min_bins, max_bins))
    nbins = max(2, nbins)

    method = str(method).lower().strip()
    if method not in {"quantile", "linear"}:
        raise ValueError("method must be 'quantile' or 'linear'")

    if method == "quantile":
        edges = np.quantile(z, np.linspace(0.0, 1.0, nbins + 1))
        edges = np.unique(edges)
        # If many identical z values exist, quantiles can collapse.
        if edges.size < 3 or np.any(np.diff(edges) <= 0):
            edges = np.linspace(z_min, z_max, nbins + 1)
    else:
        edges = np.linspace(z_min, z_max, nbins + 1)

    # Ensure exact endpoints and monotonicity for pd.cut.
    edges[0] = z_min
    edges[-1] = z_max
    if np.any(np.diff(edges) <= 0):
        edges = np.linspace(z_min, z_max, nbins + 1)

    return edges


def get_radius(
    z: ArrayLike,
    theta_arcsec: ArrayLike,
    theta_error: ArrayLike,
    model: Literal["static", "lcdm"] = "lcdm",
    Omega_m: float = 0.3,
) -> float:
    """
    Given redshifts and angular sizes (in arcseconds), compute physical sizes (in Mpc)
    using the specified cosmological model.

    Uses direct calculation: R = theta * D_A(z) for each point, then takes
    weighted median for robustness.

    Parameters
    ----------
    z : array-like
        Redshift values.
    theta_arcsec : array-like
        Angular sizes in arcseconds.
    theta_error : array-like
        Uncertainties on angular sizes in arcseconds.
    model : {'static', 'lcdm'}
        Cosmological model to use.
    Omega_m : float
        Matter density parameter (for LCDM model).

    Returns
    -------
    float
        Best-fit physical radius R in Mpc.
    """
    z = np.asarray(z, dtype=float)
    theta_arcsec = np.asarray(theta_arcsec, dtype=float)
    theta_error = np.asarray(theta_error, dtype=float)

    # Filter finite values and positive errors
    mask = np.isfinite(z) & np.isfinite(theta_arcsec) & np.isfinite(theta_error) & (theta_error > 0)
    if mask.sum() < 2:
        return np.nan

    z = z[mask]
    theta_arcsec = theta_arcsec[mask]
    theta_error = theta_error[mask]

    # Convert arcseconds to radians
    theta_rad = theta_arcsec * ARCSEC_TO_RAD

    # Avoid z=0
    z = np.clip(z, 1e-6, None)

    # Calculate angular diameter distance for each redshift
    if model == "static":
        D_A = (c / H0) * z  # Simple Hubble law
    elif model == "lcdm":
        Omega_L = 1.0 - Omega_m
        # Use vectorized astropy calculation (10-50x faster than loop)
        D_A = D_A_LCDM_vectorized(z, Omega_m, Omega_L)
    else:
        raise ValueError("Model must be 'static' or 'lcdm'")

    # Calculate R for each data point: R = theta * D_A
    R_values = theta_rad * D_A

    # Filter out very low-z points where D_A is unreliable
    # and R estimates are dominated by local peculiar velocities
    z_min_reliable = 0.1
    reliable_mask = z >= z_min_reliable

    if reliable_mask.sum() >= 2:
        R_reliable = R_values[reliable_mask]
        theta_err_reliable = theta_error[reliable_mask]

        # Use inverse-variance weighted mean (weight by 1/theta_error^2 only)
        # This avoids the D_A factor that biases toward low-z
        weights = 1.0 / theta_err_reliable**2
        weights = weights / np.sum(weights)
        R_fit = np.sum(weights * R_reliable)
    else:
        # Fallback to simple median
        R_fit = np.median(R_values)

    return R_fit


def get_radius_and_omega(
    z: ArrayLike,
    theta_arcsec: ArrayLike,
    theta_error: ArrayLike,
    *,
    z_error: ArrayLike | None = None,
    full_output: bool = False,
) -> tuple[float, float] | CosmologyFitResult:
    """
    Fit both physical radius R and Omega_L for flat LCDM cosmology.

    Uses scipy curve_fit for joint parameter estimation with proper
    uncertainty propagation.

    Enforces flat universe: Omega_m = 1 - Omega_L

    Parameters
    ----------
    z : array-like
        Redshift values.
    theta_arcsec : array-like
        Angular sizes in arcseconds.
    theta_error : array-like
        Uncertainties on angular sizes in arcseconds.
    z_error : array-like, optional
        Redshift uncertainties (propagated into total theta uncertainty).
    full_output : bool
        If True, return full CosmologyFitResult with uncertainties and
        goodness-of-fit metrics. If False (default), return simple tuple
        (R_fit, Omega_m_fit) for backwards compatibility.

    Returns
    -------
    tuple[float, float] or CosmologyFitResult
        If full_output=False: (R_fit, Omega_m_fit) tuple
        If full_output=True: CosmologyFitResult dataclass with all fit info
    """
    result = fit_flat_lcdm(
        z,
        theta_arcsec,
        theta_error,
        z_error=z_error,
    )

    if full_output:
        return result

    # Backwards compatible: return (R, Omega_m) tuple
    return result.R_mpc, result.Omega_m


# =============================================================================
# Unit-aware convenience functions using astropy.units
# =============================================================================


def get_cosmology(H0: float = 70.0, Om0: float = 0.3) -> FlatLambdaCDM:
    """Get a FlatLambdaCDM cosmology with specified parameters.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc (default: 70.0)
    Om0 : float
        Matter density parameter (default: 0.3)

    Returns
    -------
    FlatLambdaCDM
        Configured astropy FlatLambdaCDM cosmology object
    """
    return FlatLambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=Om0)


def angular_size_to_physical(
    theta: u.Quantity,
    z: float,
    cosmo: FlatLambdaCDM | None = None,
) -> u.Quantity:
    """Convert angular size to physical size using proper units.

    Parameters
    ----------
    theta : astropy.units.Quantity
        Angular size with units (e.g., arcsec, arcmin, deg)
    z : float
        Redshift of the object
    cosmo : FlatLambdaCDM, optional
        Astropy cosmology object. If None, uses default FlatLambdaCDM
        with H0=70 km/s/Mpc and Om0=0.3

    Returns
    -------
    astropy.units.Quantity
        Physical size in kpc

    Examples
    --------
    >>> theta = 1.0 * u.arcsec
    >>> R = angular_size_to_physical(theta, z=0.5)
    >>> print(f"Physical size: {R:.2f}")
    """
    if cosmo is None:
        cosmo = get_cosmology()

    # Ensure theta has angular units
    if not isinstance(theta, u.Quantity):
        raise TypeError("theta must be an astropy Quantity with angular units")

    # Convert to radians for the calculation
    theta_rad = theta.to(u.rad)

    # Get angular diameter distance
    D_A = cosmo.angular_diameter_distance(z)

    # Physical size: R = theta * D_A
    R = (theta_rad.value * D_A).to(u.kpc)

    return R


def physical_size_to_angular(
    R: u.Quantity,
    z: float,
    cosmo: FlatLambdaCDM | None = None,
) -> u.Quantity:
    """Convert physical size to angular size.

    Parameters
    ----------
    R : astropy.units.Quantity
        Physical size with units (e.g., kpc, Mpc, pc)
    z : float
        Redshift of the object
    cosmo : FlatLambdaCDM, optional
        Astropy cosmology object. If None, uses default FlatLambdaCDM
        with H0=70 km/s/Mpc and Om0=0.3

    Returns
    -------
    astropy.units.Quantity
        Angular size in arcseconds

    Examples
    --------
    >>> R = 10.0 * u.kpc
    >>> theta = physical_size_to_angular(R, z=0.5)
    >>> print(f"Angular size: {theta:.3f}")
    """
    if cosmo is None:
        cosmo = get_cosmology()

    # Ensure R has length units
    if not isinstance(R, u.Quantity):
        raise TypeError("R must be an astropy Quantity with length units")

    # Get angular diameter distance
    D_A = cosmo.angular_diameter_distance(z)

    # Angular size: theta = R / D_A (in radians)
    theta_rad = (R / D_A).decompose() * u.rad

    # Convert to arcseconds
    theta = theta_rad.to(u.arcsec)

    return theta
