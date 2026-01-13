"""ML-based star-galaxy classification using Random Forest.

This module implements a Random Forest classifier for separating stars from
galaxies using combined photometric and morphological features. Following
the miniJPAS approach (Baqui et al. 2021), measurement errors are included
as features to improve classification accuracy.

Features used (18 total):
- Photometric: 4-band fluxes (U, B, V, I), colors, magnitude
- Morphological: concentration (C), half-light radius, Gini, M20, asymmetry
- Errors: flux errors, size errors (miniJPAS finding)

Target AUC: >0.98

References:
- Baqui et al. 2021, A&A, 645, A87 (miniJPAS star-galaxy)
- Fadely et al. 2012, ApJ, 760, 15 (photometric star-galaxy)
- Abraham et al. 1994, ApJ, 432, 75 (concentration index)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Optional sklearn import with graceful fallback
try:
    from sklearn.calibration import CalibratedClassifierCV
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

    import joblib

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


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
    DEFAULT_FEATURES = [
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
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn required for ML classifier. "
                "Install with: pip install scikit-learn joblib"
            )

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
        self.metrics: Optional[ClassifierMetrics] = None
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
        importances = dict(zip(self.feature_names, self.model.feature_importances_))

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
        for is_gal, prob in zip(predictions, probabilities):
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
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn required. Install with: pip install scikit-learn joblib"
            )

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
        calculate_concentration_c,
        gini_coefficient,
        half_light_radius,
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

    features = []

    for idx, row in catalog.iterrows():
        x, y = row[x_col], row[y_col]

        # Photometric features
        f_u = row.get(flux_cols.get("u", "flux_u"), np.nan)
        f_b = row.get(flux_cols.get("b", "flux_b"), np.nan)
        f_v = row.get(flux_cols.get("v", "flux_v"), np.nan)
        f_i = row.get(flux_cols.get("i", "flux_i"), np.nan)

        # Colors (magnitude differences)
        def safe_color(f1, f2):
            if f1 > 0 and f2 > 0:
                return -2.5 * np.log10(f1 / f2)
            return np.nan

        color_ub = safe_color(f_u, f_b)
        color_bv = safe_color(f_b, f_v)
        color_vi = safe_color(f_v, f_i)

        # Magnitude
        mag_i = -2.5 * np.log10(f_i) if f_i > 0 else np.nan

        # Morphological features from image
        c = calculate_concentration_c(image, x, y)
        r_half = half_light_radius(image, x, y)

        # Gini and M20 require cutout
        cutout_size = 30
        x_int, y_int = int(x), int(y)
        y_lo = max(0, y_int - cutout_size)
        y_hi = min(image.shape[0], y_int + cutout_size)
        x_lo = max(0, x_int - cutout_size)
        x_hi = min(image.shape[1], x_int + cutout_size)
        cutout = image[y_lo:y_hi, x_lo:x_hi]

        gini = gini_coefficient(cutout)
        m20 = m20_statistic(cutout, cutout_size, cutout_size)
        asym = asymmetry_index(image, x, y)

        # Errors
        e_u = row.get(error_cols.get("u", "flux_u_err"), np.nan)
        e_b = row.get(error_cols.get("b", "flux_b_err"), np.nan)
        e_v = row.get(error_cols.get("v", "flux_v_err"), np.nan)
        e_i = row.get(error_cols.get("i", "flux_i_err"), np.nan)

        # Radius error estimate (from SNR)
        snr = f_i / e_i if e_i > 0 else np.nan
        radius_err = r_half / snr if snr > 0 else np.nan

        features.append(
            {
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
            }
        )

    return pd.DataFrame(features, index=catalog.index)


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

    features = []

    for idx, row in catalog.iterrows():
        # Photometric features
        f_u = row.get(flux_cols.get("u", "flux_u"), np.nan)
        f_b = row.get(flux_cols.get("b", "flux_b"), np.nan)
        f_v = row.get(flux_cols.get("v", "flux_v"), np.nan)
        f_i = row.get(flux_cols.get("i", "flux_i"), np.nan)

        # Colors
        def safe_color(f1, f2):
            if f1 > 0 and f2 > 0:
                return -2.5 * np.log10(f1 / f2)
            return np.nan

        color_ub = safe_color(f_u, f_b)
        color_bv = safe_color(f_b, f_v)
        color_vi = safe_color(f_v, f_i)

        # Magnitude
        mag_i = -2.5 * np.log10(f_i) if f_i > 0 else np.nan

        # Morphological features from columns
        c = row.get(morphology_cols.get("concentration_c", "concentration_c"), np.nan)
        r_half = row.get(
            morphology_cols.get("half_light_radius", "half_light_radius"), np.nan
        )
        gini = row.get(morphology_cols.get("gini", "gini"), np.nan)
        m20 = row.get(morphology_cols.get("m20", "m20"), np.nan)
        asym = row.get(morphology_cols.get("asymmetry", "asymmetry"), np.nan)

        # Errors
        e_u = row.get(error_cols.get("u", "flux_u_err"), np.nan)
        e_b = row.get(error_cols.get("b", "flux_b_err"), np.nan)
        e_v = row.get(error_cols.get("v", "flux_v_err"), np.nan)
        e_i = row.get(error_cols.get("i", "flux_i_err"), np.nan)

        # Radius error estimate
        snr = f_i / e_i if e_i > 0 else np.nan
        radius_err = r_half / snr if snr > 0 else np.nan

        features.append(
            {
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
            }
        )

    return pd.DataFrame(features, index=catalog.index)


def train_from_external_catalogs(
    our_catalog: pd.DataFrame,
    our_image: NDArray | None = None,
    hlf_catalog: pd.DataFrame | None = None,
    match_radius_arcsec: float = 1.0,
    output_model_path: Path | str | None = None,
    flux_cols: dict[str, str] | None = None,
    error_cols: dict[str, str] | None = None,
    star_flag_col: str = "CLASS_STAR",
    star_threshold: float = 0.9,
    verbose: bool = True,
) -> MLStarGalaxyClassifier:
    """Train classifier using external reference catalogs.

    This function cross-matches your catalog with HLF or similar reference
    catalogs to obtain ground-truth star/galaxy labels for training.

    Parameters
    ----------
    our_catalog : pd.DataFrame
        Your source catalog with RA, DEC and photometry
    our_image : NDArray, optional
        Image for morphological feature extraction
    hlf_catalog : pd.DataFrame, optional
        Reference catalog with star/galaxy classification
    match_radius_arcsec : float
        Maximum match radius in arcseconds (default: 1.0)
    output_model_path : Path, optional
        Save trained model to this path
    flux_cols : dict
        Flux column name mapping
    error_cols : dict
        Error column name mapping
    star_flag_col : str
        Column name for star probability in reference catalog
    star_threshold : float
        Threshold above which sources are classified as stars
    verbose : bool
        Print progress information

    Returns
    -------
    MLStarGalaxyClassifier
        Trained classifier
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord, match_coordinates_sky

    if verbose:
        print("Training ML Star-Galaxy Classifier from external catalogs...")

    # Load HLF catalog if not provided
    if hlf_catalog is None:
        try:
            from validation.external_catalogs import load_hlf_catalog

            if verbose:
                print("Loading HLF reference catalog...")
            hlf_catalog = load_hlf_catalog()
        except Exception as e:
            raise ValueError(
                f"Could not load HLF catalog: {e}. "
                "Please provide hlf_catalog parameter."
            ) from e

    # Cross-match catalogs
    if verbose:
        print(f"Cross-matching {len(our_catalog)} sources with {len(hlf_catalog)} reference sources...")

    our_coords = SkyCoord(
        ra=our_catalog["ra"].values * u.deg,
        dec=our_catalog["dec"].values * u.deg,
    )
    ref_coords = SkyCoord(
        ra=hlf_catalog["ra"].values * u.deg,
        dec=hlf_catalog["dec"].values * u.deg,
    )

    idx, sep, _ = match_coordinates_sky(our_coords, ref_coords)
    matched_mask = sep.arcsec < match_radius_arcsec

    if matched_mask.sum() < 50:
        raise ValueError(
            f"Only {matched_mask.sum()} matches found. "
            "Check coordinate systems or increase match_radius_arcsec."
        )

    if verbose:
        print(f"Found {matched_mask.sum()} matches within {match_radius_arcsec} arcsec")

    # Get matched sources
    matched_our = our_catalog[matched_mask].copy()
    matched_ref = hlf_catalog.iloc[idx[matched_mask]].copy()

    # Extract labels from reference catalog
    if star_flag_col in matched_ref.columns:
        # CLASS_STAR > threshold means star
        labels = (matched_ref[star_flag_col].values < star_threshold).astype(int)
    else:
        raise ValueError(
            f"Star flag column '{star_flag_col}' not found in reference catalog. "
            f"Available columns: {list(matched_ref.columns)}"
        )

    n_galaxies = labels.sum()
    n_stars = len(labels) - n_galaxies
    if verbose:
        print(f"Training set: {n_galaxies} galaxies, {n_stars} stars")

    # Extract features
    if verbose:
        print("Extracting features...")

    if our_image is not None:
        features = extract_features_from_catalog(
            matched_our,
            our_image,
            flux_cols=flux_cols,
            error_cols=error_cols,
        )
    else:
        features = extract_features_from_photometry(
            matched_our,
            flux_cols=flux_cols,
            error_cols=error_cols,
        )

    # Train classifier
    if verbose:
        print("Training Random Forest classifier...")

    clf = MLStarGalaxyClassifier()
    metrics = clf.fit(features, labels)

    if verbose:
        clf.print_metrics()

    # Save model if path provided
    if output_model_path is not None:
        clf.save(output_model_path)
        if verbose:
            print(f"Model saved to: {output_model_path}")

    return clf
