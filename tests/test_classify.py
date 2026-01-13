"""Tests for classify.py galaxy classification."""

import numpy as np
import pytest

from classify import prep_filters


class TestFilters:
    """Tests for filter preparation."""

    def test_prep_filters_returns_list(self):
        """Should return a list of 4 filters."""
        filters = prep_filters()
        assert isinstance(filters, list)
        assert len(filters) == 4

    def test_filter_shapes(self):
        """All filters should have the same shape."""
        filters = prep_filters()
        expected_length = len(np.arange(2200, 9500, 1))
        for f in filters:
            assert len(f) == expected_length

    def test_filters_are_binary(self):
        """Filters should only contain 0 and 1."""
        filters = prep_filters()
        for f in filters:
            unique_values = np.unique(f)
            assert all(v in [0.0, 1.0] for v in unique_values)

    def test_filter_centers_approximate(self):
        """Filters should peak around their expected centers."""
        filters = prep_filters()
        wl = np.arange(2200, 9500, 1)
        expected_centers = [3000, 4500, 6060, 8140]

        for i, (filt, center) in enumerate(zip(filters, expected_centers, strict=True)):
            # Find the center of the passband
            passband_wl = wl[filt > 0]
            actual_center = np.mean(passband_wl)
            # Allow 10% tolerance
            assert abs(actual_center - center) < center * 0.1, f"Filter {i} center mismatch"


class TestClassifyGalaxy:
    """Tests for galaxy classification function."""

    @pytest.fixture
    def sample_fluxes(self):
        """Sample flux values for testing."""
        return np.array([1.0, 1.2, 1.1, 0.9])

    @pytest.fixture
    def sample_errors(self):
        """Sample error values for testing."""
        return np.array([0.1, 0.1, 0.1, 0.1])

    @pytest.mark.skipif(
        not pytest.importorskip("scipy", reason="scipy required"), reason="scipy not available"
    )
    def test_classify_returns_tuple(self, sample_fluxes, sample_errors):
        """Classification should return (galaxy_type, redshift) tuple."""
        from classify import classify_galaxy

        # This test requires spectra files to exist
        try:
            result = classify_galaxy(sample_fluxes, sample_errors)
            assert isinstance(result, tuple)
            assert len(result) == 2
            galaxy_type, redshift = result
            assert isinstance(galaxy_type, str)
            assert isinstance(redshift, (int, float))
        except FileNotFoundError:
            pytest.skip("Spectra files not found")

    def test_classify_redshift_bounds(self, sample_fluxes, sample_errors):
        """Redshift should be in reasonable bounds."""
        from classify import classify_galaxy

        try:
            _, redshift = classify_galaxy(sample_fluxes, sample_errors)
            assert 0.0 <= redshift <= 3.5
        except FileNotFoundError:
            pytest.skip("Spectra files not found")
