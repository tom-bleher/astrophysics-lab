"""ML-based star-galaxy classification using Random Forest and XGBoost.

This module implements machine learning classifiers for separating stars from
galaxies using combined photometric and morphological features. Following
the miniJPAS approach (Baqui et al. 2021), measurement errors are included
as features to improve classification accuracy.

Classifiers available:
- MLStarGalaxyClassifier: Random Forest (default, robust)
- XGBoostStarGalaxyClassifier: XGBoost (often higher accuracy, faster)

Features used (18 total):
- Photometric: 4-band fluxes (U, B, V, I), colors, magnitude
- Morphological: concentration (C), half-light radius, Gini, M20, asymmetry
- Errors: flux errors, size errors (miniJPAS finding)

Target AUC: >0.98

References:
- Baqui et al. 2021, A&A, 645, A87 (miniJPAS star-galaxy)
- Aguilar-Argüello et al. 2025, arXiv:2501.06340 (XGBoost for morphology)
- Fadely et al. 2012, ApJ, 760, 15 (photometric star-galaxy)
- Abraham et al. 1994, ApJ, 432, 75 (concentration index)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler


@dataclass
class ClassificationResult:
    """Result of ML star-galaxy classification.

    Attributes
    ----------
    is_galaxy : bool
        True if classified as galaxy, False if star
    probability_galaxy : float
        Probability of being a galaxy (0-1)
    probability_star : float
        Probability of being a star (0-1)
    confidence : float
        Classification confidence: abs(p_galaxy - 0.5) * 2
    features_used : list[str]
        Names of features used for classification
    """

    is_galaxy: bool
    probability_galaxy: float
    probability_star: float
    confidence: float
    features_used: list[str] = field(default_factory=list)


@dataclass
class ClassifierMetrics:
    """Performance metrics for the classifier.

    Attributes
    ----------
    auc_roc : float
        Area under ROC curve
    accuracy : float
        Overall classification accuracy
    precision_galaxy : float
        Precision for galaxy class
    recall_galaxy : float
        Recall for galaxy class
    precision_star : float
        Precision for star class
    recall_star : float
        Recall for star class
    n_train : int
        Number of training samples
    n_test : int
        Number of test samples
    feature_importances : dict
        Feature importance scores from Random Forest
    confusion_matrix : NDArray
        Confusion matrix [[TN, FP], [FN, TP]]
    cross_val_scores : NDArray
        Cross-validation AUC scores
    """

    auc_roc: float
    accuracy: float
    precision_galaxy: float
    recall_galaxy: float
    precision_star: float
    recall_star: float
    n_train: int
    n_test: int
    feature_importances: dict
    confusion_matrix: NDArray
    cross_val_scores: NDArray


class MLStarGalaxyClassifier:
    """Random Forest classifier for star-galaxy separation.

    This classifier uses a combination of photometric, morphological, and
    error features to separate stars from galaxies. Following the miniJPAS
    methodology, including measurement errors as features significantly
    improves classification accuracy.

    Attributes
    ----------
    model : RandomForestClassifier
        The trained Random Forest model
    scaler : StandardScaler
        Feature scaler for normalization
    feature_names : list
        Names of features in order
    is_fitted : bool
        Whether the model has been trained
    metrics : ClassifierMetrics
        Performance metrics from training

    Examples
    --------
    >>> clf = MLStarGalaxyClassifier()
    >>> metrics = clf.fit(features_df, labels)
    >>> print(f"AUC: {metrics.auc_roc:.3f}")
    >>> results = clf.predict(new_features)
    """

    # Default feature set following miniJPAS approach
    DEFAULT_FEATURES: ClassVar[list[str]] = [
        # Fluxes (4)
        "flux_u",
        "flux_b",
        "flux_v",
        "flux_i",
        # Colors (3)
        "color_ub",
        "color_bv",
        "color_vi",
        # Magnitude (1)
        "mag_i",
        # Morphology (5)
        "concentration_c",
        "half_light_radius",
        "gini",
        "m20",
        "asymmetry",
        # Errors - miniJPAS finding: errors improve classification (5)
        "flux_u_err",
        "flux_b_err",
        "flux_v_err",
        "flux_i_err",
        "radius_err",
    ]

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 15,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """Initialize the classifier.

        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest (default: 200)
        max_depth : int
            Maximum depth of trees (default: 15)
        min_samples_split : int
            Minimum samples to split a node (default: 10)
        min_samples_leaf : int
            Minimum samples in a leaf (default: 5)
        class_weight : str
            'balanced' to handle class imbalance (default)
        random_state : int
            Random seed for reproducibility (default: 42)
        n_jobs : int
            Number of parallel jobs, -1 for all cores (default: -1)
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            oob_score=True,
        )
        self.scaler = StandardScaler()
        self.feature_names = self.DEFAULT_FEATURES.copy()
        self.is_fitted = False
        self.metrics: ClassifierMetrics | None = None
        self._calibrated_model = None

    def fit(
        self,
        features: pd.DataFrame,
        labels: NDArray,
        test_size: float = 0.2,
        calibrate: bool = True,
        cv_folds: int = 5,
    ) -> ClassifierMetrics:
        """Train the classifier.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix with columns matching feature_names
        labels : NDArray
            Binary labels (1 = galaxy, 0 = star)
        test_size : float
            Fraction of data for testing (default: 0.2)
        calibrate : bool
            Apply probability calibration (default: True)
        cv_folds : int
            Number of cross-validation folds (default: 5)

        Returns
        -------
        ClassifierMetrics
            Training and validation metrics
        """
        # Validate features
        missing = set(self.feature_names) - set(features.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")

        X = features[self.feature_names].values
        y = np.asarray(labels).astype(int)

        # Handle missing values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=cv, scoring="roc_auc"
        )

        # Probability calibration (isotonic regression)
        # Note: For sklearn >= 1.3, we use a simpler approach since cv='prefit' was removed
        if calibrate:
            # Calibrate using cross-validation on full training data
            from sklearn.calibration import CalibratedClassifierCV as CalCV

            # Create a new calibrated classifier trained from scratch
            self._calibrated_model = CalCV(
                RandomForestClassifier(
                    n_estimators=self.model.n_estimators,
                    max_depth=self.model.max_depth,
                    min_samples_split=self.model.min_samples_split,
                    min_samples_leaf=self.model.min_samples_leaf,
                    class_weight=self.model.class_weight,
                    random_state=42,
                    n_jobs=self.model.n_jobs,
                ),
                cv=3,
                method="isotonic",
            )
            self._calibrated_model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self._get_probabilities(X_test_scaled)

        # Compute metrics
        auc = roc_auc_score(y_test, y_prob)
        accuracy = (y_pred == y_test).mean()

        cm = confusion_matrix(y_test, y_pred)

        # Per-class metrics
        report = classification_report(y_test, y_pred, output_dict=True)

        # Feature importances
        importances = dict(zip(self.feature_names, self.model.feature_importances_, strict=False))

        self.metrics = ClassifierMetrics(
            auc_roc=auc,
            accuracy=accuracy,
            precision_galaxy=report["1"]["precision"],
            recall_galaxy=report["1"]["recall"],
            precision_star=report["0"]["precision"],
            recall_star=report["0"]["recall"],
            n_train=len(X_train),
            n_test=len(X_test),
            feature_importances=importances,
            confusion_matrix=cm,
            cross_val_scores=cv_scores,
        )

        self.is_fitted = True
        return self.metrics

    def _get_probabilities(self, X_scaled: NDArray) -> NDArray:
        """Get calibrated probabilities if available."""
        if self._calibrated_model is not None:
            return self._calibrated_model.predict_proba(X_scaled)[:, 1]
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(
        self,
        features: pd.DataFrame,
    ) -> list[ClassificationResult]:
        """Classify sources as stars or galaxies.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix for sources to classify

        Returns
        -------
        list[ClassificationResult]
            Classification results for each source
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = features[self.feature_names].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X_scaled = self.scaler.transform(X)

        probabilities = self._get_probabilities(X_scaled)
        predictions = probabilities > 0.5

        results = []
        for is_gal, prob in zip(predictions, probabilities, strict=False):
            results.append(
                ClassificationResult(
                    is_galaxy=bool(is_gal),
                    probability_galaxy=float(prob),
                    probability_star=float(1 - prob),
                    confidence=float(abs(prob - 0.5) * 2),
                    features_used=self.feature_names,
                )
            )

        return results

    def predict_proba(self, features: pd.DataFrame) -> NDArray:
        """Get probability of being a galaxy for each source.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix for sources

        Returns
        -------
        NDArray
            Probability of being a galaxy for each source
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = features[self.feature_names].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X_scaled = self.scaler.transform(X)

        return self._get_probabilities(X_scaled)

    def save(self, path: Path | str) -> None:
        """Save the trained model to disk.

        Parameters
        ----------
        path : Path or str
            Output path for the model file (.joblib)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        path = Path(path)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "metrics": self.metrics,
                "calibrated_model": self._calibrated_model,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "MLStarGalaxyClassifier":
        """Load a trained model from disk.

        Parameters
        ----------
        path : Path or str
            Path to saved model file

        Returns
        -------
        MLStarGalaxyClassifier
            Loaded classifier instance
        """
        path = Path(path)
        data = joblib.load(path)

        instance = cls.__new__(cls)
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.feature_names = data["feature_names"]
        instance.metrics = data["metrics"]
        instance._calibrated_model = data.get("calibrated_model")
        instance.is_fitted = True

        return instance

    def print_metrics(self) -> None:
        """Print formatted classification metrics."""
        if self.metrics is None:
            print("No metrics available. Train the model first.")
            return

        m = self.metrics
        print("\n" + "=" * 50)
        print("ML Star-Galaxy Classifier Performance")
        print("=" * 50)
        print(f"Training samples:    {m.n_train}")
        print(f"Test samples:        {m.n_test}")
        print(f"AUC-ROC:             {m.auc_roc:.4f}")
        print(f"Accuracy:            {m.accuracy:.4f}")
        print(f"Cross-val AUC:       {m.cross_val_scores.mean():.4f} +/- {m.cross_val_scores.std():.4f}")
        print()
        print("Per-class metrics:")
        print(f"  Galaxy precision:  {m.precision_galaxy:.4f}")
        print(f"  Galaxy recall:     {m.recall_galaxy:.4f}")
        print(f"  Star precision:    {m.precision_star:.4f}")
        print(f"  Star recall:       {m.recall_star:.4f}")
        print()
        print("Top 5 feature importances:")
        sorted_imp = sorted(
            m.feature_importances.items(), key=lambda x: x[1], reverse=True
        )
        for name, imp in sorted_imp[:5]:
            print(f"  {name:20s}: {imp:.4f}")
        print("=" * 50)


def _safe_color_vectorized(f1: NDArray, f2: NDArray) -> NDArray:
    """Vectorized safe color computation."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where((f1 > 0) & (f2 > 0), -2.5 * np.log10(f1 / f2), np.nan)
    return result


def extract_features_from_catalog(
    catalog: pd.DataFrame,
    image: NDArray,
    x_col: str = "xcentroid",
    y_col: str = "ycentroid",
    flux_cols: dict[str, str] | None = None,
    error_cols: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Extract ML features from a source catalog.

    Parameters
    ----------
    catalog : pd.DataFrame
        Source catalog with positions and photometry
    image : NDArray
        Reference image for morphological measurements
    x_col, y_col : str
        Position column names
    flux_cols : dict
        Mapping of band name to flux column name
    error_cols : dict
        Mapping of band name to error column name

    Returns
    -------
    pd.DataFrame
        Feature matrix ready for classifier
    """
    from morphology.concentration import (
        asymmetry_index,
        compute_morphology_batch,
        gini_coefficient,
        m20_statistic,
    )

    if flux_cols is None:
        flux_cols = {"u": "flux_u", "b": "flux_b", "v": "flux_v", "i": "flux_i"}

    if error_cols is None:
        error_cols = {
            "u": "flux_u_err",
            "b": "flux_b_err",
            "v": "flux_v_err",
            "i": "flux_i_err",
        }

    n_sources = len(catalog)

    # Extract coordinates as arrays
    x_coords = catalog[x_col].values
    y_coords = catalog[y_col].values

    # Vectorized photometric feature extraction
    f_u = catalog[flux_cols.get("u", "flux_u")].values if flux_cols.get("u", "flux_u") in catalog.columns else np.full(n_sources, np.nan)
    f_b = catalog[flux_cols.get("b", "flux_b")].values if flux_cols.get("b", "flux_b") in catalog.columns else np.full(n_sources, np.nan)
    f_v = catalog[flux_cols.get("v", "flux_v")].values if flux_cols.get("v", "flux_v") in catalog.columns else np.full(n_sources, np.nan)
    f_i = catalog[flux_cols.get("i", "flux_i")].values if flux_cols.get("i", "flux_i") in catalog.columns else np.full(n_sources, np.nan)

    # Vectorized color computation
    color_ub = _safe_color_vectorized(f_u, f_b)
    color_bv = _safe_color_vectorized(f_b, f_v)
    color_vi = _safe_color_vectorized(f_v, f_i)

    # Vectorized magnitude
    with np.errstate(divide='ignore', invalid='ignore'):
        mag_i = np.where(f_i > 0, -2.5 * np.log10(f_i), np.nan)

    # Batch morphological features (concentration_c, half_light_radius)
    morphology = compute_morphology_batch(image, x_coords, y_coords)
    c_values = morphology['concentration_c']
    r_half_values = morphology['half_light_radius']

    # Gini, M20, and asymmetry still need per-source computation (cutouts)
    gini_values = np.full(n_sources, np.nan)
    m20_values = np.full(n_sources, np.nan)
    asym_values = np.full(n_sources, np.nan)

    cutout_size = 30
    for i in range(n_sources):
        x, y = x_coords[i], y_coords[i]
        x_int, y_int = int(x), int(y)
        y_lo = max(0, y_int - cutout_size)
        y_hi = min(image.shape[0], y_int + cutout_size)
        x_lo = max(0, x_int - cutout_size)
        x_hi = min(image.shape[1], x_int + cutout_size)
        cutout = image[y_lo:y_hi, x_lo:x_hi]

        gini_values[i] = gini_coefficient(cutout)
        m20_values[i] = m20_statistic(cutout, cutout_size, cutout_size)
        asym_values[i] = asymmetry_index(image, x, y)

    # Vectorized error extraction
    e_u = catalog[error_cols.get("u", "flux_u_err")].values if error_cols.get("u", "flux_u_err") in catalog.columns else np.full(n_sources, np.nan)
    e_b = catalog[error_cols.get("b", "flux_b_err")].values if error_cols.get("b", "flux_b_err") in catalog.columns else np.full(n_sources, np.nan)
    e_v = catalog[error_cols.get("v", "flux_v_err")].values if error_cols.get("v", "flux_v_err") in catalog.columns else np.full(n_sources, np.nan)
    e_i = catalog[error_cols.get("i", "flux_i_err")].values if error_cols.get("i", "flux_i_err") in catalog.columns else np.full(n_sources, np.nan)

    # Vectorized radius error estimate
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(e_i > 0, f_i / e_i, np.nan)
        radius_err = np.where(snr > 0, r_half_values / snr, np.nan)

    # Build DataFrame directly (much faster than appending dicts)
    features_df = pd.DataFrame({
        "flux_u": f_u,
        "flux_b": f_b,
        "flux_v": f_v,
        "flux_i": f_i,
        "color_ub": color_ub,
        "color_bv": color_bv,
        "color_vi": color_vi,
        "mag_i": mag_i,
        "concentration_c": c_values,
        "half_light_radius": r_half_values,
        "gini": gini_values,
        "m20": m20_values,
        "asymmetry": asym_values,
        "flux_u_err": e_u,
        "flux_b_err": e_b,
        "flux_v_err": e_v,
        "flux_i_err": e_i,
        "radius_err": radius_err,
    }, index=catalog.index)

    return features_df


def extract_features_from_photometry(
    catalog: pd.DataFrame,
    flux_cols: dict[str, str] | None = None,
    error_cols: dict[str, str] | None = None,
    morphology_cols: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Extract ML features from catalog with pre-computed morphology.

    Use this when morphological parameters are already in the catalog
    (e.g., from SExtractor or previous processing).

    Parameters
    ----------
    catalog : pd.DataFrame
        Source catalog with photometry and morphology columns
    flux_cols : dict
        Mapping of band name to flux column name
    error_cols : dict
        Mapping of band name to error column name
    morphology_cols : dict
        Mapping of morphology parameter to column name

    Returns
    -------
    pd.DataFrame
        Feature matrix ready for classifier
    """
    if flux_cols is None:
        flux_cols = {"u": "flux_u", "b": "flux_b", "v": "flux_v", "i": "flux_i"}

    if error_cols is None:
        error_cols = {
            "u": "flux_u_err",
            "b": "flux_b_err",
            "v": "flux_v_err",
            "i": "flux_i_err",
        }

    if morphology_cols is None:
        morphology_cols = {
            "concentration_c": "concentration_c",
            "half_light_radius": "half_light_radius",
            "gini": "gini",
            "m20": "m20",
            "asymmetry": "asymmetry",
        }

    n_sources = len(catalog)

    # Helper to safely get column values
    def get_col(col_name: str) -> NDArray:
        if col_name in catalog.columns:
            return catalog[col_name].values
        return np.full(n_sources, np.nan)

    # Vectorized photometric feature extraction
    f_u = get_col(flux_cols.get("u", "flux_u"))
    f_b = get_col(flux_cols.get("b", "flux_b"))
    f_v = get_col(flux_cols.get("v", "flux_v"))
    f_i = get_col(flux_cols.get("i", "flux_i"))

    # Vectorized color computation
    color_ub = _safe_color_vectorized(f_u, f_b)
    color_bv = _safe_color_vectorized(f_b, f_v)
    color_vi = _safe_color_vectorized(f_v, f_i)

    # Vectorized magnitude
    with np.errstate(divide='ignore', invalid='ignore'):
        mag_i = np.where(f_i > 0, -2.5 * np.log10(f_i), np.nan)

    # Vectorized morphological feature extraction from columns
    c = get_col(morphology_cols.get("concentration_c", "concentration_c"))
    r_half = get_col(morphology_cols.get("half_light_radius", "half_light_radius"))
    gini = get_col(morphology_cols.get("gini", "gini"))
    m20 = get_col(morphology_cols.get("m20", "m20"))
    asym = get_col(morphology_cols.get("asymmetry", "asymmetry"))

    # Vectorized error extraction
    e_u = get_col(error_cols.get("u", "flux_u_err"))
    e_b = get_col(error_cols.get("b", "flux_b_err"))
    e_v = get_col(error_cols.get("v", "flux_v_err"))
    e_i = get_col(error_cols.get("i", "flux_i_err"))

    # Vectorized radius error estimate
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(e_i > 0, f_i / e_i, np.nan)
        radius_err = np.where(snr > 0, r_half / snr, np.nan)

    # Build DataFrame directly (much faster than appending dicts)
    features_df = pd.DataFrame({
        "flux_u": f_u,
        "flux_b": f_b,
        "flux_v": f_v,
        "flux_i": f_i,
        "color_ub": color_ub,
        "color_bv": color_bv,
        "color_vi": color_vi,
        "mag_i": mag_i,
        "concentration_c": c,
        "half_light_radius": r_half,
        "gini": gini,
        "m20": m20,
        "asymmetry": asym,
        "flux_u_err": e_u,
        "flux_b_err": e_b,
        "flux_v_err": e_v,
        "flux_i_err": e_i,
        "radius_err": radius_err,
    }, index=catalog.index)

    return features_df


# =============================================================================
# XGBoost Classifier (Alternative to Random Forest)
# =============================================================================


class XGBoostStarGalaxyClassifier:
    """XGBoost classifier for star-galaxy separation.

    XGBoost often outperforms Random Forest for morphological classification
    (Aguilar-Argüello et al. 2025). It provides:
    - Faster training and inference
    - Better handling of feature interactions
    - Built-in regularization to prevent overfitting

    Attributes
    ----------
    model : XGBClassifier
        The trained XGBoost model
    scaler : StandardScaler
        Feature scaler for normalization
    feature_names : list
        Names of features in order
    is_fitted : bool
        Whether the model has been trained
    metrics : ClassifierMetrics
        Performance metrics from training

    Examples
    --------
    >>> clf = XGBoostStarGalaxyClassifier()
    >>> metrics = clf.fit(features_df, labels)
    >>> print(f"AUC: {metrics.auc_roc:.3f}")
    >>> results = clf.predict(new_features)
    """

    # Same feature set as Random Forest
    DEFAULT_FEATURES: ClassVar[list[str]] = MLStarGalaxyClassifier.DEFAULT_FEATURES

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        use_gpu: bool = False,
    ):
        """Initialize the XGBoost classifier.

        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds (default: 200)
        max_depth : int
            Maximum tree depth (default: 6, lower than RF to prevent overfitting)
        learning_rate : float
            Boosting learning rate (default: 0.1)
        subsample : float
            Subsample ratio of training instances (default: 0.8)
        colsample_bytree : float
            Subsample ratio of columns when constructing each tree (default: 0.8)
        reg_alpha : float
            L1 regularization term (default: 0.1)
        reg_lambda : float
            L2 regularization term (default: 1.0)
        random_state : int
            Random seed for reproducibility (default: 42)
        n_jobs : int
            Number of parallel jobs, -1 for all cores (default: -1)
        use_gpu : bool
            Use GPU acceleration if available (default: False)
        """
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError(
                "XGBoost is required for XGBoostStarGalaxyClassifier. "
                "Install with: pip install xgboost"
            ) from None

        # Configure tree method based on GPU availability
        tree_method = "hist"  # Default CPU method (fast)
        if use_gpu:
            try:
                import xgboost as xgb
                # Check if GPU is available
                if xgb.build_info().get("USE_CUDA", False):
                    tree_method = "gpu_hist"
            except Exception:
                pass

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            tree_method=tree_method,
            objective="binary:logistic",
            eval_metric="auc",
        )
        self.scaler = StandardScaler()
        self.feature_names = self.DEFAULT_FEATURES.copy()
        self.is_fitted = False
        self.metrics: ClassifierMetrics | None = None
        self._use_gpu = use_gpu

    def fit(
        self,
        features: pd.DataFrame,
        labels: NDArray,
        test_size: float = 0.2,
        early_stopping_rounds: int | None = 20,
        cv_folds: int = 5,
        verbose: bool = False,
    ) -> ClassifierMetrics:
        """Train the XGBoost classifier.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix with columns matching feature_names
        labels : NDArray
            Binary labels (1 = galaxy, 0 = star)
        test_size : float
            Fraction of data for testing (default: 0.2)
        early_stopping_rounds : int, optional
            Stop training if validation AUC doesn't improve for this many rounds.
            Set to None to disable (default: 20)
        cv_folds : int
            Number of cross-validation folds (default: 5)
        verbose : bool
            Print training progress (default: False)

        Returns
        -------
        ClassifierMetrics
            Training and validation metrics
        """
        # Validate features
        missing = set(self.feature_names) - set(features.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")

        X = features[self.feature_names].values
        y = np.asarray(labels).astype(int)

        # Handle missing values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Handle class imbalance by computing scale_pos_weight
        n_neg = np.sum(y == 0)
        n_pos = np.sum(y == 1)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        self.model.set_params(scale_pos_weight=scale_pos_weight)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model with early stopping
        if early_stopping_rounds:
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=verbose,
            )
        else:
            self.model.fit(X_train_scaled, y_train)

        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=cv, scoring="roc_auc"
        )

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        # Compute metrics
        auc = roc_auc_score(y_test, y_prob)
        accuracy = (y_pred == y_test).mean()

        cm = confusion_matrix(y_test, y_pred)

        # Per-class metrics
        report = classification_report(y_test, y_pred, output_dict=True)

        # Feature importances
        importances = dict(zip(self.feature_names, self.model.feature_importances_, strict=False))

        self.metrics = ClassifierMetrics(
            auc_roc=auc,
            accuracy=accuracy,
            precision_galaxy=report["1"]["precision"],
            recall_galaxy=report["1"]["recall"],
            precision_star=report["0"]["precision"],
            recall_star=report["0"]["recall"],
            n_train=len(X_train),
            n_test=len(X_test),
            feature_importances=importances,
            confusion_matrix=cm,
            cross_val_scores=cv_scores,
        )

        self.is_fitted = True
        return self.metrics

    def predict(
        self,
        features: pd.DataFrame,
    ) -> list[ClassificationResult]:
        """Classify sources as stars or galaxies."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = features[self.feature_names].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X_scaled = self.scaler.transform(X)

        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        predictions = probabilities > 0.5

        results = []
        for is_gal, prob in zip(predictions, probabilities, strict=False):
            results.append(
                ClassificationResult(
                    is_galaxy=bool(is_gal),
                    probability_galaxy=float(prob),
                    probability_star=float(1 - prob),
                    confidence=float(abs(prob - 0.5) * 2),
                    features_used=self.feature_names,
                )
            )

        return results

    def predict_proba(self, features: pd.DataFrame) -> NDArray:
        """Get probability of being a galaxy for each source."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = features[self.feature_names].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X_scaled = self.scaler.transform(X)

        return self.model.predict_proba(X_scaled)[:, 1]

    def save(self, path: Path | str) -> None:
        """Save the trained model to disk."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        path = Path(path)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "metrics": self.metrics,
                "classifier_type": "xgboost",
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "XGBoostStarGalaxyClassifier":
        """Load a trained model from disk."""
        path = Path(path)
        data = joblib.load(path)

        instance = cls.__new__(cls)
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.feature_names = data["feature_names"]
        instance.metrics = data["metrics"]
        instance.is_fitted = True
        instance._use_gpu = False

        return instance

    def print_metrics(self) -> None:
        """Print formatted classification metrics."""
        if self.metrics is None:
            print("No metrics available. Train the model first.")
            return

        m = self.metrics
        print("\n" + "=" * 50)
        print("XGBoost Star-Galaxy Classifier Performance")
        print("=" * 50)
        print(f"Training samples:    {m.n_train}")
        print(f"Test samples:        {m.n_test}")
        print(f"AUC-ROC:             {m.auc_roc:.4f}")
        print(f"Accuracy:            {m.accuracy:.4f}")
        print(f"Cross-val AUC:       {m.cross_val_scores.mean():.4f} +/- {m.cross_val_scores.std():.4f}")
        print()
        print("Per-class metrics:")
        print(f"  Galaxy precision:  {m.precision_galaxy:.4f}")
        print(f"  Galaxy recall:     {m.recall_galaxy:.4f}")
        print(f"  Star precision:    {m.precision_star:.4f}")
        print(f"  Star recall:       {m.recall_star:.4f}")
        print()
        print("Top 5 feature importances:")
        sorted_imp = sorted(
            m.feature_importances.items(), key=lambda x: x[1], reverse=True
        )
        for name, imp in sorted_imp[:5]:
            print(f"  {name:20s}: {imp:.4f}")
        print("=" * 50)


def create_classifier(
    classifier_type: str = "random_forest",
    **kwargs,
) -> MLStarGalaxyClassifier | XGBoostStarGalaxyClassifier:
    """Factory function to create a star-galaxy classifier.

    Parameters
    ----------
    classifier_type : str
        Type of classifier: 'random_forest' or 'xgboost'
    **kwargs
        Additional arguments passed to the classifier constructor

    Returns
    -------
    classifier
        Classifier instance (MLStarGalaxyClassifier or XGBoostStarGalaxyClassifier)

    Examples
    --------
    >>> clf = create_classifier('xgboost', n_estimators=300, max_depth=8)
    >>> metrics = clf.fit(features, labels)
    """
    if classifier_type.lower() in ("random_forest", "rf"):
        return MLStarGalaxyClassifier(**kwargs)
    elif classifier_type.lower() in ("xgboost", "xgb"):
        return XGBoostStarGalaxyClassifier(**kwargs)
    else:
        raise ValueError(
            f"Unknown classifier type: {classifier_type}. "
            "Choose 'random_forest' or 'xgboost'."
        )


