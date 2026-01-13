"""Tests for scientific.py cosmological calculations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scientific import ARCSEC_TO_RAD, D_A_LCDM, H0, E, c, get_radius, theta_lcdm, theta_static


class TestConstants:
    """Test that physical constants are correct."""

    def test_speed_of_light(self):
        assert c == pytest.approx(299792.458, rel=1e-6)

    def test_hubble_constant(self):
        assert pytest.approx(70.0, rel=1e-6) == H0


class TestStaticModel:
    """Tests for the static universe (linear Hubble law) model."""

    def test_theta_static_basic(self):
        """Test angular size calculation for static model."""
        # At z=1, D_A = c/H0 ≈ 4283 Mpc
        # For R=1 Mpc, theta should be ~1/4283 rad
        z = 1.0
        R = 1.0  # Mpc
        theta = theta_static(z, R)
        expected_D_A = c / H0  # ~4282.75 Mpc
        expected_theta = R / expected_D_A
        assert_allclose(theta, expected_theta, rtol=1e-6)

    def test_theta_static_array_input(self):
        """Test that function handles array inputs."""
        z = np.array([0.5, 1.0, 2.0])
        R = 1.0
        theta = theta_static(z, R)
        assert theta.shape == (3,)
        # Higher z should give smaller theta (more distant)
        assert theta[0] > theta[1] > theta[2]

    def test_theta_static_near_zero(self):
        """Test behavior near z=0 (should not produce inf/nan)."""
        z = 0.001
        R = 1.0
        theta = theta_static(z, R)
        assert np.isfinite(theta)

    def test_theta_static_proportional_to_R(self):
        """Angular size should scale linearly with physical size."""
        z = 1.0
        theta1 = theta_static(z, 1.0)
        theta2 = theta_static(z, 2.0)
        assert_allclose(theta2, 2.0 * theta1, rtol=1e-10)


class TestLCDMModel:
    """Tests for the Lambda-CDM cosmological model."""

    def test_E_function_matter_dominated(self):
        """Test E(z) at z=0 equals 1."""
        result = E(0.0, 0.3, 0.7)
        # At z=0: E = sqrt(0.3 + 0.7) = 1.0
        assert_allclose(result, 1.0, rtol=1e-10)

    def test_E_function_high_z(self):
        """At high z, matter dominates."""
        z = 10.0
        result = E(z, 0.3, 0.7)
        # E ≈ sqrt(0.3 * (1+10)^3) = sqrt(0.3 * 1331) ≈ 19.98
        expected = np.sqrt(0.3 * (1 + z) ** 3 + 0.7)
        assert_allclose(result, expected, rtol=1e-10)

    def test_D_A_LCDM_positive(self):
        """Angular diameter distance should be positive."""
        for z in [0.1, 0.5, 1.0, 2.0, 3.0]:
            D_A = D_A_LCDM(z)
            assert D_A > 0

    def test_D_A_LCDM_maximum(self):
        """D_A has a maximum around z~1.6 for standard cosmology."""
        z_values = np.linspace(0.1, 3.0, 30)
        D_A_values = [D_A_LCDM(z) for z in z_values]

        # Find the maximum
        max_idx = np.argmax(D_A_values)
        z_max = z_values[max_idx]

        # Maximum should be roughly between z=1.4 and z=1.8
        assert 1.3 < z_max < 2.0

    def test_theta_lcdm_minimum(self):
        """Angular size has a minimum (objects appear larger at high z)."""
        z_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        R = 0.01  # 10 kpc in Mpc
        theta = theta_lcdm(z_values, R)

        # Find minimum
        min_idx = np.argmin(theta)
        z_min = z_values[min_idx]

        # Minimum should be around z~1.6
        assert 1.0 < z_min < 2.5

    def test_theta_lcdm_vs_static_difference(self):
        """LCDM and static models diverge at high z."""
        z = 2.0
        R = 0.01
        theta_s = theta_static(z, R)
        theta_l = theta_lcdm(z, R)

        # At high z, LCDM gives larger angular sizes
        assert theta_l > theta_s


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_very_small_z(self):
        """Both models should give finite results for tiny z."""
        z = 1e-6
        R = 1.0
        assert np.isfinite(theta_static(z, R))
        assert np.isfinite(theta_lcdm(z, R))

    def test_array_of_different_types(self):
        """Function should handle list and tuple inputs."""
        z_list = [0.5, 1.0, 1.5]
        R = 1.0
        result = theta_static(z_list, R)
        assert len(result) == 3


@pytest.mark.parametrize(
    "z,expected_range",
    [
        (0.1, (0, 500)),
        (1.0, (1500, 2000)),
        (3.0, (1500, 2000)),
    ],
)
def test_D_A_LCDM_ranges(z, expected_range):
    """Test that D_A values are in expected ranges for standard cosmology."""
    D_A = D_A_LCDM(z)
    assert expected_range[0] < D_A < expected_range[1]


class TestGetRadius:
    """Tests for the get_radius fitting function."""

    def test_get_radius_returns_physical_units(self):
        """Fitted R should be in physically reasonable range (kpc)."""
        # Typical data for distant galaxies
        z = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        theta_arcsec = np.array([0.6, 0.4, 0.35, 0.38, 0.42])
        theta_error = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

        R_lcdm = get_radius(z, theta_arcsec, theta_error, model="lcdm")
        R_kpc = R_lcdm * 1000  # Convert Mpc to kpc

        # Galaxy half-light radii are typically 1-15 kpc
        assert 0.5 < R_kpc < 20, f"R = {R_kpc:.2f} kpc is outside expected range"

    def test_get_radius_static_vs_lcdm(self):
        """Static model should give larger R than LCDM for same data."""
        z = np.array([0.5, 1.0, 1.5, 2.0])
        theta_arcsec = np.array([0.5, 0.4, 0.35, 0.38])
        theta_error = np.array([0.1, 0.1, 0.1, 0.1])

        R_static = get_radius(z, theta_arcsec, theta_error, model="static")
        R_lcdm = get_radius(z, theta_arcsec, theta_error, model="lcdm")

        # Static model underestimates D_A at high z, so needs larger R to fit
        assert R_static > R_lcdm

    def test_get_radius_handles_insufficient_data(self):
        """Should return nan if fewer than 2 valid points."""
        z = np.array([1.0])
        theta = np.array([0.5])
        error = np.array([0.1])

        result = get_radius(z, theta, error)
        assert np.isnan(result)

    def test_arcsec_to_rad_constant(self):
        """Verify the arcsec to radian conversion constant."""
        expected = np.pi / (180.0 * 3600.0)
        assert_allclose(ARCSEC_TO_RAD, expected, rtol=1e-10)
        # Should be approximately 4.848e-6
        assert_allclose(ARCSEC_TO_RAD, 4.848136811e-6, rtol=1e-6)
