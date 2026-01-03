import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit

c = 299792.458  # speed of light [km/s]
H0 = 70.0  # Hubble constant [km/s/Mpc]


def theta_static(z, R):
    """
    Static universe angular size model
    R: physical size [Mpc]
    """
    D = (c / H0) * z  # Mpc
    return R / D


def E(z, Omega_m, Omega_L):
    inner = Omega_m * (1 + z) ** 3 + Omega_L
    return np.sqrt(np.clip(inner, 0.0, None))


def D_A_LCDM(z, Omega_m=0.3, Omega_L=0.7):
    integral, _ = quad(lambda zp: 1.0 / E(zp, Omega_m, Omega_L), 0, z)
    return (c / H0) * integral / (1 + z)


def theta_lcdm(z, R, Omega_m=0.3, Omega_L=0.7):
    z_arr = np.atleast_1d(z)
    D_A = np.array([D_A_LCDM(float(zi), Omega_m, Omega_L) for zi in z_arr])
    return R / D_A


def get_radius(z, theta_arcsec, theta_error, model="lcdm", normalize=True):
    """
    Given redshifts and angular sizes (in arcseconds), compute physical sizes (in Mpc)
    using the specified cosmological model.

    normalize: if True, divide angles by their median for numerical stability, then
    rescale the fitted R accordingly.
    """

    z = np.asarray(z, dtype=float)
    theta_arcsec = np.asarray(theta_arcsec, dtype=float)
    theta_error = np.asarray(theta_error, dtype=float)

    # Filter finite values and positive errors
    mask = (
        np.isfinite(z)
        & np.isfinite(theta_arcsec)
        & np.isfinite(theta_error)
        & (theta_error > 0)
    )
    if mask.sum() < 2:
        return np.nan

    z = z[mask]
    theta_arcsec = theta_arcsec[mask]
    theta_error = theta_error[mask]

    # Avoid z=0 which breaks angular-diameter distance; clip to tiny positive
    z = np.clip(z, 1e-6, None)

    # Numerical stability: avoid zero errors
    theta_error = np.maximum(theta_error, np.nanmax(theta_error) * 1e-6 + 1e-12)

    if normalize:
        norm = np.nanmedian(theta_arcsec)
        if not np.isfinite(norm) or norm <= 0:
            norm = 1.0
        theta_fit = theta_arcsec / norm
        theta_err_fit = theta_error / norm
    else:
        norm = 1.0
        theta_fit = theta_arcsec
        theta_err_fit = theta_error

    if model == "static":
        func = theta_static
    elif model == "lcdm":
        func = theta_lcdm
    else:
        raise ValueError("Model must be 'static' or 'lcdm'")

    # Rough initial guess for R using small-z approximation: R ~ theta * (c/H0) * z
    D_approx = (c / H0) * z
    R0 = np.nanmedian(theta_arcsec * D_approx)
    if not np.isfinite(R0) or R0 <= 0:
        R0 = 1.0

    try:
        p, _ = curve_fit(
            func,
            z,
            theta_fit,
            sigma=theta_err_fit,
            absolute_sigma=True,
            maxfev=5000,
            p0=[R0],
            bounds=(1e-6, np.inf)
        )
        R_fit = p[0] * norm
    except Exception:
        R_fit = np.nan

    return R_fit
