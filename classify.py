import numpy as np
import os
from scipy.optimize import minimize_scalar


def prep_filters():
    wl_filter_small = np.arange(2200, 9500, 1)
    filter_centers = np.array([3000, 4500, 6060, 8140])
    filter_widths = np.array([1521, 1501, 951, 766]) / 2
    filters = []
    for i in range(len(filter_centers)):
        center = filter_centers[i]
        width = filter_widths[i]
        filter_spec = np.zeros(len(wl_filter_small))
        filter_spec[
            (wl_filter_small >= (center - width)) & (wl_filter_small <= (center + width))
        ] = 1.0
        filters.append(filter_spec)
    return filters


def classify_galaxy(fluxes, errors, spectra_path=R".\spectra"):
    (
        B,
        Ir,
        U,
        V,
    ) = fluxes
    dB, dIr, dU, dV = errors
    galaxies = [
        "elliptical",
        "S0",
        "Sa",
        "Sb",
        "sbt1",
        "sbt2",
        "sbt3",
        "sbt4",
        "sbt5",
        "sbt6",
    ]
    filters = prep_filters()

    # Measurement data (normalized)
    meas_photo = np.array([U, B, V, Ir])
    meas_errs = np.array([dU, dB, dV, dIr])
    meas_photo_norm = meas_photo / np.median(meas_photo)
    meas_photo_err_norm = meas_errs / np.median(meas_photo)

    def compute_chi_square(redshift, galaxy_type):
        """Compute chi-square for a given redshift and galaxy type."""
        # load the relevant galaxy
        path = os.path.join(spectra_path, f"{galaxy_type}.dat")
        wl, spec = np.loadtxt(path, usecols=[0, 1], unpack=True)

        # apply a redshift transformation to the galaxy spectrum
        wl_redshifted = wl * (1 + redshift)

        # move to a common wavelength grid
        wl_small = np.arange(2200, 9500, 1)
        spec_small = np.interp(wl_small, wl_redshifted, spec)
        spec_small_norm = spec_small / np.median(spec_small)

        # create the synthetic photometry according to this redshift
        syn_photometry = []
        for j in range(len(filters)):
            filter_arr = filters[j]
            syn_phot = np.median(
                filter_arr[filter_arr != 0] * spec_small_norm[filter_arr != 0]
            )
            syn_photometry.append(syn_phot)
        syn_photometry = np.array(syn_photometry)
        syn_photometry_norm = syn_photometry / np.median(syn_photometry)

        # compute the chi-square of the fit
        chi_square = np.sum(
            (meas_photo_norm - syn_photometry_norm) ** 2 / meas_photo_err_norm**2
        )
        return chi_square

    chimins = []
    zmins = []

    for galaxy_type in galaxies:
        # Use continuous optimization to find best redshift (0 to 1)
        result = minimize_scalar(
            lambda z: compute_chi_square(z, galaxy_type), bounds=(0, 1), method="bounded"
        )
        zmins.append(result.x)
        chimins.append(result.fun)  # type: ignore

    galaxy_type = galaxies[chimins.index(min(chimins))]
    redshift = zmins[chimins.index(min(chimins))]
    return galaxy_type, redshift
