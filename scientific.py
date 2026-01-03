import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit

c = 299792.458  # speed of light [km/s]
H0 = 70.0       # Hubble constant [km/s/Mpc]


def theta_static(z, R):
    """
    Static universe angular size model
    R: physical size [Mpc]
    """
    D = (c / H0) * z  # Mpc
    return R / D


def E(z, Omega_m, Omega_L):
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_L)


def D_A_LCDM(z, Omega_m=0.3, Omega_L=0.7):
    integral, _ = quad(lambda zp: 1.0 / E(zp, Omega_m, Omega_L), 0, z)
    return (c / H0) * integral / (1 + z)


def theta_lcdm(z, R, Omega_m=0.3, Omega_L=0.7):
    D_A = np.array([D_A_LCDM(zi, Omega_m, Omega_L) for zi in z])
    return R / D_A


def get_radius(z, theta_arcsec, theta_error, model='lcdm'):
    """
    Given redshifts and angular sizes (in arcseconds), compute physical sizes (in Mpc)
    using the specified cosmological model.
    """
    theta_rad = theta_arcsec / 3600.0 * (np.pi / 180.0)  # convert to radians
    theta_error_rad = theta_error / 3600.0 * (np.pi / 180.0)

    if model == 'static':
        func = theta_static
    elif model == 'lcdm':
        func = theta_lcdm
    else:
        raise ValueError("Model must be 'static' or 'lcdm'")
    p, cov = curve_fit(
        func,
        z,
        theta_rad,
        sigma=theta_error_rad,
        absolute_sigma=True
    )

    return p[0]
