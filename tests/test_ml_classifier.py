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
        rng = np.random.default_rng(42)
        n_samples = 100

        # Create features with some separation between classes
        features = pd.DataFrame(
            {
                "flux_u": rng.exponential(100, n_samples),
                "flux_b": rng.exponential(150, n_samples),
                "flux_v": rng.exponential(200, n_samples),
                "flux_i": rng.exponential(250, n_samples),
                "color_ub": rng.normal(0.5, 0.3, n_samples),
                "color_bv": rng.normal(0.3, 0.2, n_samples),
                "color_vi": rng.normal(0.2, 0.2, n_samples),
                "mag_i": rng.uniform(20, 28, n_samples),
                "concentration_c": rng.uniform(1.5, 4.0, n_samples),
                "half_light_radius": rng.exponential(5, n_samples),
                "gini": rng.uniform(0.3, 0.7, n_samples),
                "m20": rng.uniform(-2.5, -1.0, n_samples),
                "asymmetry": rng.uniform(0, 0.3, n_samples),
                "flux_u_err": rng.exponential(10, n_samples),
                "flux_b_err": rng.exponential(15, n_samples),
                "flux_v_err": rng.exponential(20, n_samples),
                "flux_i_err": rng.exponential(25, n_samples),
                "radius_err": rng.exponential(1, n_samples),
            }
        )

        return features

    @pytest.fixture
    def sample_labels(self, sample_features):
        """Generate labels based on concentration (proxy for star/galaxy)."""
        rng = np.random.default_rng(42)
        # Stars have higher concentration
        concentration = sample_features["concentration_c"].values
        prob_star = 1 / (1 + np.exp(-(concentration - 2.5)))
        labels = (rng.random(len(concentration)) > prob_star).astype(int)
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

        rng = np.random.default_rng(42)
        clf = MLStarGalaxyClassifier(n_estimators=10)

        # Create features with missing columns
        incomplete_features = pd.DataFrame(
            {
                "flux_u": rng.exponential(100, 50),
                "flux_b": rng.exponential(150, 50),
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
        rng = np.random.default_rng(42)

        # Stars: high concentration, small radius
        stars = pd.DataFrame(
            {
                "flux_u": rng.exponential(100, n // 2),
                "flux_b": rng.exponential(150, n // 2),
                "flux_v": rng.exponential(200, n // 2),
                "flux_i": rng.exponential(250, n // 2),
                "color_ub": rng.normal(0.2, 0.1, n // 2),
                "color_bv": rng.normal(0.1, 0.1, n // 2),
                "color_vi": rng.normal(0.1, 0.1, n // 2),
                "mag_i": rng.uniform(18, 22, n // 2),
                "concentration_c": rng.uniform(3.0, 4.5, n // 2),  # High
                "half_light_radius": rng.uniform(1, 3, n // 2),  # Small
                "gini": rng.uniform(0.5, 0.7, n // 2),
                "m20": rng.uniform(-2.5, -2.0, n // 2),
                "asymmetry": rng.uniform(0, 0.1, n // 2),
                "flux_u_err": rng.exponential(10, n // 2),
                "flux_b_err": rng.exponential(15, n // 2),
                "flux_v_err": rng.exponential(20, n // 2),
                "flux_i_err": rng.exponential(25, n // 2),
                "radius_err": rng.exponential(0.5, n // 2),
            }
        )

        # Galaxies: lower concentration, larger radius
        galaxies = pd.DataFrame(
            {
                "flux_u": rng.exponential(80, n // 2),
                "flux_b": rng.exponential(120, n // 2),
                "flux_v": rng.exponential(180, n // 2),
                "flux_i": rng.exponential(220, n // 2),
                "color_ub": rng.normal(0.6, 0.2, n // 2),
                "color_bv": rng.normal(0.4, 0.2, n // 2),
                "color_vi": rng.normal(0.3, 0.2, n // 2),
                "mag_i": rng.uniform(22, 28, n // 2),
                "concentration_c": rng.uniform(1.5, 2.8, n // 2),  # Low
                "half_light_radius": rng.uniform(4, 15, n // 2),  # Large
                "gini": rng.uniform(0.3, 0.5, n // 2),
                "m20": rng.uniform(-1.8, -1.2, n // 2),
                "asymmetry": rng.uniform(0.1, 0.3, n // 2),
                "flux_u_err": rng.exponential(10, n // 2),
                "flux_b_err": rng.exponential(15, n // 2),
                "flux_v_err": rng.exponential(20, n // 2),
                "flux_i_err": rng.exponential(25, n // 2),
                "radius_err": rng.exponential(1, n // 2),
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
            is_galaxy=True, probability_galaxy=0.8, probability_star=0.2, confidence=0.9
        )

        assert result.is_galaxy is True
        assert result.probability_galaxy == 0.8
        assert result.probability_star == 0.2
        assert result.confidence == 0.9


class TestFeatureExtraction:
    """Tests for feature extraction utilities."""

    def test_extract_features_from_photometry(self):
        """Test feature extraction from photometry data."""
        from morphology.ml_classifier import extract_features_from_photometry

        rng = np.random.default_rng(42)
        # Create mock catalog with all required columns
        catalog = pd.DataFrame(
            {
                "flux_u": rng.exponential(100, 50),
                "flux_b": rng.exponential(150, 50),
                "flux_v": rng.exponential(200, 50),
                "flux_i": rng.exponential(250, 50),
                "flux_u_err": rng.exponential(10, 50),
                "flux_b_err": rng.exponential(15, 50),
                "flux_v_err": rng.exponential(20, 50),
                "flux_i_err": rng.exponential(25, 50),
                "concentration_c": rng.uniform(1.5, 4.0, 50),
                "half_light_radius": rng.exponential(5, 50),
                "gini": rng.uniform(0.3, 0.7, 50),
                "m20": rng.uniform(-2.5, -1.0, 50),
                "asymmetry": rng.uniform(0, 0.3, 50),
            }
        )

        features = extract_features_from_photometry(
            catalog,
            flux_cols={"u": "flux_u", "b": "flux_b", "v": "flux_v", "i": "flux_i"},
            error_cols={
                "u": "flux_u_err",
                "b": "flux_b_err",
                "v": "flux_v_err",
                "i": "flux_i_err",
            },
        )

        assert len(features) == 50
        assert "flux_u" in features.columns
        assert "color_ub" in features.columns
        assert "mag_i" in features.columns
