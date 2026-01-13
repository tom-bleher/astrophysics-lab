"""Tests for ML star-galaxy classifier."""

import numpy as np
import pandas as pd
import pytest

# Skip all tests if sklearn not available
sklearn = pytest.importorskip("sklearn")


class TestMLStarGalaxyClassifier:
    """Tests for the MLStarGalaxyClassifier class."""

    @pytest.fixture
    def sample_features(self):
        """Generate sample feature data."""
        np.random.seed(42)
        n_samples = 100

        # Create features with some separation between classes
        features = pd.DataFrame(
            {
                "flux_u": np.random.exponential(100, n_samples),
                "flux_b": np.random.exponential(150, n_samples),
                "flux_v": np.random.exponential(200, n_samples),
                "flux_i": np.random.exponential(250, n_samples),
                "color_ub": np.random.normal(0.5, 0.3, n_samples),
                "color_bv": np.random.normal(0.3, 0.2, n_samples),
                "color_vi": np.random.normal(0.2, 0.2, n_samples),
                "mag_i": np.random.uniform(20, 28, n_samples),
                "concentration_c": np.random.uniform(1.5, 4.0, n_samples),
                "half_light_radius": np.random.exponential(5, n_samples),
                "gini": np.random.uniform(0.3, 0.7, n_samples),
                "m20": np.random.uniform(-2.5, -1.0, n_samples),
                "asymmetry": np.random.uniform(0, 0.3, n_samples),
                "flux_u_err": np.random.exponential(10, n_samples),
                "flux_b_err": np.random.exponential(15, n_samples),
                "flux_v_err": np.random.exponential(20, n_samples),
                "flux_i_err": np.random.exponential(25, n_samples),
                "radius_err": np.random.exponential(1, n_samples),
            }
        )

        return features

    @pytest.fixture
    def sample_labels(self, sample_features):
        """Generate labels based on concentration (proxy for star/galaxy)."""
        # Stars have higher concentration
        concentration = sample_features["concentration_c"].values
        prob_star = 1 / (1 + np.exp(-(concentration - 2.5)))
        labels = (np.random.random(len(concentration)) > prob_star).astype(int)
        return labels

    def test_classifier_initialization(self):
        """Test classifier can be initialized."""
        from morphology.ml_classifier import MLStarGalaxyClassifier

        clf = MLStarGalaxyClassifier()
        assert clf.is_fitted is False
        assert len(clf.feature_names) == 18

    def test_classifier_fit(self, sample_features, sample_labels):
        """Test classifier can be trained."""
        from morphology.ml_classifier import MLStarGalaxyClassifier

        clf = MLStarGalaxyClassifier(n_estimators=10)  # Small for speed
        metrics = clf.fit(sample_features, sample_labels, test_size=0.3)

        assert clf.is_fitted is True
        assert metrics.n_train > 0
        assert metrics.n_test > 0
        assert 0 <= metrics.auc_roc <= 1
        assert len(metrics.feature_importances) == 18

    def test_classifier_predict(self, sample_features, sample_labels):
        """Test classifier predictions."""
        from morphology.ml_classifier import MLStarGalaxyClassifier

        clf = MLStarGalaxyClassifier(n_estimators=10)
        clf.fit(sample_features, sample_labels)

        results = clf.predict(sample_features.iloc[:10])

        assert len(results) == 10
        for r in results:
            assert isinstance(r.is_galaxy, bool)
            assert 0 <= r.probability_galaxy <= 1
            assert 0 <= r.confidence <= 1

    def test_classifier_predict_proba(self, sample_features, sample_labels):
        """Test probability predictions."""
        from morphology.ml_classifier import MLStarGalaxyClassifier

        clf = MLStarGalaxyClassifier(n_estimators=10)
        clf.fit(sample_features, sample_labels)

        probs = clf.predict_proba(sample_features.iloc[:10])

        assert len(probs) == 10
        assert all(0 <= p <= 1 for p in probs)

    def test_classifier_save_load(self, sample_features, sample_labels, tmp_path):
        """Test model persistence."""
        from morphology.ml_classifier import MLStarGalaxyClassifier

        clf = MLStarGalaxyClassifier(n_estimators=10)
        clf.fit(sample_features, sample_labels)

        # Save
        model_path = tmp_path / "test_model.joblib"
        clf.save(model_path)

        assert model_path.exists()

        # Load
        clf2 = MLStarGalaxyClassifier.load(model_path)

        assert clf2.is_fitted is True

        # Predictions should match
        pred1 = clf.predict_proba(sample_features.iloc[:5])
        pred2 = clf2.predict_proba(sample_features.iloc[:5])

        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_classifier_unfitted_error(self, sample_features):
        """Test that unfitted classifier raises error."""
        from morphology.ml_classifier import MLStarGalaxyClassifier

        clf = MLStarGalaxyClassifier()

        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(sample_features)

        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict_proba(sample_features)

    def test_classifier_missing_features_error(self, sample_labels):
        """Test that missing features raises error."""
        from morphology.ml_classifier import MLStarGalaxyClassifier

        clf = MLStarGalaxyClassifier(n_estimators=10)

        # Create features with missing columns
        incomplete_features = pd.DataFrame(
            {
                "flux_u": np.random.exponential(100, 50),
                "flux_b": np.random.exponential(150, 50),
                # Missing other required features
            }
        )

        with pytest.raises(ValueError, match="Missing features"):
            clf.fit(incomplete_features, sample_labels[:50])

    def test_auc_on_separable_data(self):
        """Test that AUC is high on well-separated data."""
        from morphology.ml_classifier import MLStarGalaxyClassifier

        # Create well-separated data
        n = 200
        np.random.seed(42)

        # Stars: high concentration, small radius
        stars = pd.DataFrame(
            {
                "flux_u": np.random.exponential(100, n // 2),
                "flux_b": np.random.exponential(150, n // 2),
                "flux_v": np.random.exponential(200, n // 2),
                "flux_i": np.random.exponential(250, n // 2),
                "color_ub": np.random.normal(0.2, 0.1, n // 2),
                "color_bv": np.random.normal(0.1, 0.1, n // 2),
                "color_vi": np.random.normal(0.1, 0.1, n // 2),
                "mag_i": np.random.uniform(18, 22, n // 2),
                "concentration_c": np.random.uniform(3.0, 4.5, n // 2),  # High
                "half_light_radius": np.random.uniform(1, 3, n // 2),  # Small
                "gini": np.random.uniform(0.5, 0.7, n // 2),
                "m20": np.random.uniform(-2.5, -2.0, n // 2),
                "asymmetry": np.random.uniform(0, 0.1, n // 2),
                "flux_u_err": np.random.exponential(10, n // 2),
                "flux_b_err": np.random.exponential(15, n // 2),
                "flux_v_err": np.random.exponential(20, n // 2),
                "flux_i_err": np.random.exponential(25, n // 2),
                "radius_err": np.random.exponential(0.5, n // 2),
            }
        )

        # Galaxies: lower concentration, larger radius
        galaxies = pd.DataFrame(
            {
                "flux_u": np.random.exponential(80, n // 2),
                "flux_b": np.random.exponential(120, n // 2),
                "flux_v": np.random.exponential(180, n // 2),
                "flux_i": np.random.exponential(220, n // 2),
                "color_ub": np.random.normal(0.6, 0.2, n // 2),
                "color_bv": np.random.normal(0.4, 0.2, n // 2),
                "color_vi": np.random.normal(0.3, 0.2, n // 2),
                "mag_i": np.random.uniform(22, 28, n // 2),
                "concentration_c": np.random.uniform(1.5, 2.8, n // 2),  # Low
                "half_light_radius": np.random.uniform(4, 15, n // 2),  # Large
                "gini": np.random.uniform(0.3, 0.5, n // 2),
                "m20": np.random.uniform(-1.8, -1.2, n // 2),
                "asymmetry": np.random.uniform(0.1, 0.3, n // 2),
                "flux_u_err": np.random.exponential(10, n // 2),
                "flux_b_err": np.random.exponential(15, n // 2),
                "flux_v_err": np.random.exponential(20, n // 2),
                "flux_i_err": np.random.exponential(25, n // 2),
                "radius_err": np.random.exponential(1, n // 2),
            }
        )

        features = pd.concat([stars, galaxies], ignore_index=True)
        labels = np.array([0] * (n // 2) + [1] * (n // 2))  # 0=star, 1=galaxy

        clf = MLStarGalaxyClassifier(n_estimators=100)
        metrics = clf.fit(features, labels)

        # With well-separated data, should achieve high AUC
        assert metrics.auc_roc > 0.90, f"AUC {metrics.auc_roc} below 0.90"

    def test_cross_validation_scores(self, sample_features, sample_labels):
        """Test cross-validation produces reasonable scores."""
        from morphology.ml_classifier import MLStarGalaxyClassifier

        clf = MLStarGalaxyClassifier(n_estimators=10)
        metrics = clf.fit(sample_features, sample_labels, cv_folds=3)

        assert len(metrics.cross_val_scores) == 3
        assert all(0 <= s <= 1 for s in metrics.cross_val_scores)

    def test_confusion_matrix_shape(self, sample_features, sample_labels):
        """Test confusion matrix has correct shape."""
        from morphology.ml_classifier import MLStarGalaxyClassifier

        clf = MLStarGalaxyClassifier(n_estimators=10)
        metrics = clf.fit(sample_features, sample_labels)

        assert metrics.confusion_matrix.shape == (2, 2)
        # Sum should equal test set size
        assert metrics.confusion_matrix.sum() == metrics.n_test


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_result_creation(self):
        """Test ClassificationResult can be created."""
        from morphology.ml_classifier import ClassificationResult

        result = ClassificationResult(
            is_galaxy=True,
            probability_galaxy=0.85,
            probability_star=0.15,
            confidence=0.7,
            features_used=["flux_i", "concentration_c"],
        )

        assert result.is_galaxy is True
        assert result.probability_galaxy == 0.85
        assert result.confidence == 0.7
        assert len(result.features_used) == 2

    def test_result_probabilities_sum(self):
        """Test that probabilities sum to approximately 1."""
        from morphology.ml_classifier import ClassificationResult

        result = ClassificationResult(
            is_galaxy=True,
            probability_galaxy=0.85,
            probability_star=0.15,
            confidence=0.7,
        )

        assert abs(result.probability_galaxy + result.probability_star - 1.0) < 0.01


class TestFeatureExtraction:
    """Tests for feature extraction functions."""

    def test_extract_from_photometry(self):
        """Test feature extraction from photometry catalog."""
        from morphology.ml_classifier import extract_features_from_photometry

        catalog = pd.DataFrame(
            {
                "flux_u": [100, 200],
                "flux_b": [150, 250],
                "flux_v": [200, 300],
                "flux_i": [250, 350],
                "flux_u_err": [10, 20],
                "flux_b_err": [15, 25],
                "flux_v_err": [20, 30],
                "flux_i_err": [25, 35],
                "concentration_c": [2.5, 3.0],
                "half_light_radius": [5.0, 3.0],
                "gini": [0.5, 0.6],
                "m20": [-1.5, -2.0],
                "asymmetry": [0.1, 0.05],
            }
        )

        features = extract_features_from_photometry(catalog)

        assert len(features) == 2
        assert "color_ub" in features.columns
        assert "mag_i" in features.columns
        # Check color calculation: -2.5 * log10(100/150) for first row
        expected_color_ub = -2.5 * np.log10(100 / 150)
        assert abs(features.iloc[0]["color_ub"] - expected_color_ub) < 0.01

    def test_extract_handles_missing_flux(self):
        """Test feature extraction handles missing/zero fluxes."""
        from morphology.ml_classifier import extract_features_from_photometry

        catalog = pd.DataFrame(
            {
                "flux_u": [0, 100],  # Zero flux
                "flux_b": [150, 250],
                "flux_v": [200, 300],
                "flux_i": [250, 350],
                "flux_u_err": [10, 20],
                "flux_b_err": [15, 25],
                "flux_v_err": [20, 30],
                "flux_i_err": [25, 35],
                "concentration_c": [2.5, 3.0],
                "half_light_radius": [5.0, 3.0],
                "gini": [0.5, 0.6],
                "m20": [-1.5, -2.0],
                "asymmetry": [0.1, 0.05],
            }
        )

        features = extract_features_from_photometry(catalog)

        # First row should have NaN for U-band color
        assert np.isnan(features.iloc[0]["color_ub"])
        # Second row should have valid color
        assert not np.isnan(features.iloc[1]["color_ub"])


class TestHasSklearn:
    """Test sklearn availability flag."""

    def test_has_sklearn_flag(self):
        """Test HAS_SKLEARN is True when sklearn is available."""
        from morphology.ml_classifier import HAS_SKLEARN

        assert HAS_SKLEARN is True
