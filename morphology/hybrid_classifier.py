"""Hybrid star-galaxy classifier combining multiple methods.

This module provides ensemble classifiers that combine:
- Classical morphology (SPREAD_MODEL, concentration)
- Machine learning (Random Forest / XGBoost)
- Deep learning (Zoobot, if available)
- Color-based classification

The hybrid approach provides more robust classifications by leveraging
the strengths of each method:
- SPREAD_MODEL: Best for bright sources with good S/N
- ML: Good for intermediate magnitudes with training data
- Deep learning: Best for complex morphologies
- Colors: Useful fallback for faint sources

References
----------
- Sevilla-Noarbe et al. (2018) - DES Y1 star-galaxy separation
- Cabayol et al. (2023) - miniJPAS machine learning approach
- Walmsley et al. (2023) - Zoobot deep learning framework
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class ClassifierWeight(IntEnum):
    """Weights for different classification methods in ensemble voting."""

    GAIA = 100      # Highest weight - definitive for bright stars
    SPREAD_MODEL = auto()
    ML_CLASSIFIER = auto()
    DEEP_LEARNING = auto()
    COLOR_LOCUS = auto()
    CONCENTRATION = auto()


@dataclass
class ClassificationVote:
    """A single classification vote from one method."""

    method: str
    is_galaxy: bool
    probability: float  # 0-1 probability of being a galaxy
    confidence: float   # 0-1 confidence in this vote
    weight: float       # Weight for ensemble voting


@dataclass
class HybridClassifierConfig:
    """Configuration for the hybrid classifier."""

    # Method weights for ensemble voting
    weights: dict[str, float] = field(default_factory=lambda: {
        'gaia': 10.0,           # Highest - definitive matches
        'spread_model': 5.0,    # High - reliable morphology
        'ml_classifier': 3.0,   # Medium - trained model
        'deep_learning': 4.0,   # High if available
        'color_locus': 1.5,     # Lower - supplementary
        'concentration': 2.0,   # Medium - classical method
    })

    # Minimum confidence threshold to include a vote
    min_confidence: float = 0.3

    # Minimum number of agreeing methods for high confidence
    min_agreeing_methods: int = 2

    # Magnitude ranges where each method is most reliable
    magnitude_reliability: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        'gaia': (0.0, 21.0),           # Bright end only
        'spread_model': (0.0, 24.0),   # Good for most magnitudes
        'ml_classifier': (18.0, 25.0), # Trained on intermediate
        'deep_learning': (18.0, 26.0), # Works on fainter sources
        'color_locus': (20.0, 27.0),   # Useful for faint sources
        'concentration': (0.0, 23.0),  # Reliable for brighter
    })

    # Use Bayesian combination instead of weighted voting
    use_bayesian: bool = False


class HybridStarGalaxyClassifier:
    """Hybrid classifier combining multiple star-galaxy separation methods.

    This classifier uses weighted voting or Bayesian combination to merge
    classifications from multiple methods, providing more robust results
    than any single method alone.

    Parameters
    ----------
    config : HybridClassifierConfig, optional
        Configuration for the classifier. Uses defaults if not provided.
    ml_classifier : optional
        Pre-trained ML classifier (MLStarGalaxyClassifier or XGBoostStarGalaxyClassifier)
    deep_learning_classifier : optional
        Pre-trained deep learning classifier (ZoobotStarGalaxyClassifier)

    Attributes
    ----------
    config : HybridClassifierConfig
        Current configuration
    ml_classifier : optional
        ML classifier instance
    dl_classifier : optional
        Deep learning classifier instance

    Examples
    --------
    >>> from morphology.hybrid_classifier import HybridStarGalaxyClassifier
    >>> from morphology.ml_classifier import create_classifier
    >>>
    >>> # Create hybrid classifier with ML backend
    >>> ml_clf = create_classifier('xgboost')
    >>> ml_clf.load('models/star_galaxy_xgb.joblib')
    >>>
    >>> hybrid = HybridStarGalaxyClassifier(ml_classifier=ml_clf)
    >>>
    >>> # Run hybrid classification
    >>> results = hybrid.classify(
    ...     catalog=sources,
    ...     spread_model=spread_model_values,
    ...     spread_model_err=spread_model_errors,
    ...     magnitudes=mag_auto,
    ... )
    """

    # Class variables
    AVAILABLE_METHODS: ClassVar[list[str]] = [
        'gaia', 'spread_model', 'ml_classifier',
        'deep_learning', 'color_locus', 'concentration'
    ]

    def __init__(
        self,
        config: HybridClassifierConfig | None = None,
        ml_classifier=None,
        deep_learning_classifier=None,
    ):
        """Initialize the hybrid classifier."""
        self.config = config or HybridClassifierConfig()
        self.ml_classifier = ml_classifier
        self.dl_classifier = deep_learning_classifier

        # Track which methods are available
        self._available_methods = ['spread_model', 'concentration']
        if ml_classifier is not None:
            self._available_methods.append('ml_classifier')
        if deep_learning_classifier is not None:
            self._available_methods.append('deep_learning')

    def _get_weight_for_magnitude(
        self,
        method: str,
        magnitude: float,
    ) -> float:
        """Get the weight for a method at a given magnitude.

        Weights are adjusted based on the magnitude reliability range
        for each method.
        """
        base_weight = self.config.weights.get(method, 1.0)
        mag_range = self.config.magnitude_reliability.get(method, (0.0, 30.0))

        # Full weight within reliable range
        if mag_range[0] <= magnitude <= mag_range[1]:
            return base_weight

        # Reduce weight outside reliable range
        if magnitude < mag_range[0]:
            distance = mag_range[0] - magnitude
        else:
            distance = magnitude - mag_range[1]

        # Exponential decay outside reliable range
        decay_factor = np.exp(-0.5 * distance)
        return base_weight * decay_factor

    def _classify_spread_model(
        self,
        spread_model: NDArray,
        spread_model_err: NDArray,
        magnitudes: NDArray,
        threshold: float = 0.003,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Classify using SPREAD_MODEL.

        Returns
        -------
        is_galaxy : NDArray
            Boolean array of galaxy classifications
        probability : NDArray
            Probability of being a galaxy (0-1)
        confidence : NDArray
            Confidence in the classification (0-1)
        """
        n = len(spread_model)
        is_galaxy = np.ones(n, dtype=bool)
        probability = np.full(n, 0.5)
        confidence = np.zeros(n)

        valid = np.isfinite(spread_model) & np.isfinite(spread_model_err)
        if not np.any(valid):
            return is_galaxy, probability, confidence

        sm = spread_model[valid]
        sm_err = spread_model_err[valid]

        # Adaptive threshold based on error
        adaptive_thresh = np.sqrt(threshold**2 + (3.0 * sm_err)**2)

        # Stars: SPREAD_MODEL < -threshold (point-like)
        # Galaxies: SPREAD_MODEL > threshold (extended)
        # Ambiguous: in between

        stars = sm < -adaptive_thresh
        galaxies = sm > adaptive_thresh

        is_galaxy[valid] = galaxies

        # Convert to probability using sigmoid-like function
        # Centered at 0, with width determined by threshold
        x = sm / adaptive_thresh
        prob = 1.0 / (1.0 + np.exp(-2.0 * x))
        probability[valid] = prob

        # Confidence based on signal-to-noise of SPREAD_MODEL
        snr = np.abs(sm) / np.maximum(sm_err, 1e-6)
        conf = 1.0 - np.exp(-0.5 * snr)
        confidence[valid] = conf

        return is_galaxy, probability, confidence

    def _classify_concentration(
        self,
        concentration: NDArray,
        threshold: float = 2.5,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Classify using concentration index.

        Returns
        -------
        is_galaxy : NDArray
            Boolean array of galaxy classifications
        probability : NDArray
            Probability of being a galaxy (0-1)
        confidence : NDArray
            Confidence in the classification (0-1)
        """
        n = len(concentration)
        is_galaxy = np.ones(n, dtype=bool)
        probability = np.full(n, 0.5)
        confidence = np.zeros(n)

        valid = np.isfinite(concentration)
        if not np.any(valid):
            return is_galaxy, probability, confidence

        c = concentration[valid]

        # Stars: C > threshold (centrally concentrated)
        # Galaxies: C < threshold (more extended)
        stars = c > threshold
        is_galaxy[valid] = ~stars

        # Probability using sigmoid
        x = (threshold - c) / 0.5  # Width of ~0.5 in C
        prob = 1.0 / (1.0 + np.exp(-x))
        probability[valid] = prob

        # Confidence based on distance from threshold
        dist = np.abs(c - threshold)
        conf = 1.0 - np.exp(-2.0 * dist)
        confidence[valid] = conf

        return is_galaxy, probability, confidence

    def _classify_ml(
        self,
        catalog: pd.DataFrame,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Classify using ML classifier.

        Returns
        -------
        is_galaxy : NDArray
            Boolean array of galaxy classifications
        probability : NDArray
            Probability of being a galaxy (0-1)
        confidence : NDArray
            Confidence in the classification (0-1)
        """
        n = len(catalog)
        is_galaxy = np.ones(n, dtype=bool)
        probability = np.full(n, 0.5)
        confidence = np.zeros(n)

        if self.ml_classifier is None:
            return is_galaxy, probability, confidence

        try:
            # Get predictions and probabilities
            predictions = self.ml_classifier.predict(catalog)
            probabilities = self.ml_classifier.predict_proba(catalog)

            # Handle different output formats
            if probabilities.ndim == 2:
                # Assume column 1 is galaxy probability
                prob_galaxy = probabilities[:, 1]
            else:
                prob_galaxy = probabilities

            is_galaxy = predictions == 1  # Assuming 1 = galaxy
            probability = prob_galaxy

            # Confidence from how far from 0.5 the probability is
            confidence = 2.0 * np.abs(prob_galaxy - 0.5)

        except Exception:
            pass

        return is_galaxy, probability, confidence

    def _classify_deep_learning(
        self,
        images: NDArray | None = None,
        catalog: pd.DataFrame | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Classify using deep learning classifier.

        Returns
        -------
        is_galaxy : NDArray
            Boolean array of galaxy classifications
        probability : NDArray
            Probability of being a galaxy (0-1)
        confidence : NDArray
            Confidence in the classification (0-1)
        """
        if catalog is not None:
            n = len(catalog)
        elif images is not None:
            n = len(images)
        else:
            return np.array([]), np.array([]), np.array([])

        is_galaxy = np.ones(n, dtype=bool)
        probability = np.full(n, 0.5)
        confidence = np.zeros(n)

        if self.dl_classifier is None:
            return is_galaxy, probability, confidence

        try:
            if images is not None:
                predictions = self.dl_classifier.predict(images)
            elif catalog is not None and hasattr(self.dl_classifier, 'predict_from_catalog'):
                predictions = self.dl_classifier.predict_from_catalog(catalog)
            else:
                return is_galaxy, probability, confidence

            # Extract results
            if isinstance(predictions, dict):
                prob_galaxy = predictions.get('prob_galaxy', probability)
                confidence = predictions.get('confidence', confidence)
            else:
                prob_galaxy = predictions
                confidence = 2.0 * np.abs(prob_galaxy - 0.5)

            is_galaxy = prob_galaxy > 0.5
            probability = prob_galaxy

        except Exception:
            pass

        return is_galaxy, probability, confidence

    def _combine_votes_weighted(
        self,
        votes: list[dict[str, NDArray]],
        magnitudes: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Combine votes using weighted averaging.

        Parameters
        ----------
        votes : list of dicts
            Each dict contains: method, is_galaxy, probability, confidence
        magnitudes : NDArray
            Source magnitudes for weight adjustment

        Returns
        -------
        is_galaxy : NDArray
            Final boolean classifications
        probability : NDArray
            Final probabilities
        confidence : NDArray
            Final confidences
        """
        n = len(magnitudes)

        # Accumulate weighted probabilities
        weighted_prob_sum = np.zeros(n)
        weight_sum = np.zeros(n)
        n_methods = np.zeros(n, dtype=int)

        for vote in votes:
            method = vote['method']
            prob = vote['probability']
            conf = vote['confidence']

            # Get magnitude-dependent weights
            weights = np.array([
                self._get_weight_for_magnitude(method, m)
                for m in magnitudes
            ])

            # Only count votes with sufficient confidence
            valid = conf >= self.config.min_confidence

            # Weight by both method weight and confidence
            effective_weight = weights * conf
            effective_weight[~valid] = 0

            weighted_prob_sum += effective_weight * prob
            weight_sum += effective_weight
            n_methods[valid] += 1

        # Compute final probability
        probability = np.where(
            weight_sum > 0,
            weighted_prob_sum / weight_sum,
            0.5  # Default to uncertain
        )

        # Final classification
        is_galaxy = probability > 0.5

        # Confidence based on agreement and probability distance from 0.5
        prob_confidence = 2.0 * np.abs(probability - 0.5)
        agreement_factor = np.minimum(n_methods / self.config.min_agreeing_methods, 1.0)
        confidence = prob_confidence * agreement_factor

        return is_galaxy, probability, confidence

    def _combine_votes_bayesian(
        self,
        votes: list[dict[str, NDArray]],
        magnitudes: NDArray,
        prior_galaxy: float = 0.7,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Combine votes using Bayesian inference.

        Uses naive Bayes assumption that classifiers are independent.

        Parameters
        ----------
        votes : list of dicts
            Each dict contains: method, is_galaxy, probability, confidence
        magnitudes : NDArray
            Source magnitudes for weight adjustment
        prior_galaxy : float
            Prior probability of a source being a galaxy

        Returns
        -------
        is_galaxy : NDArray
            Final boolean classifications
        probability : NDArray
            Final probabilities (posterior)
        confidence : NDArray
            Final confidences
        """
        n = len(magnitudes)

        # Start with log-odds from prior
        log_odds = np.full(n, np.log(prior_galaxy / (1 - prior_galaxy)))
        n_methods = np.zeros(n, dtype=int)

        for vote in votes:
            conf = vote['confidence']
            prob = vote['probability']

            # Only use confident votes
            valid = conf >= self.config.min_confidence

            # Clip probabilities to avoid log(0)
            prob_clipped = np.clip(prob, 0.01, 0.99)

            # Add log-likelihood ratio
            llr = np.log(prob_clipped / (1 - prob_clipped))

            # Weight by confidence
            log_odds[valid] += conf[valid] * llr[valid]
            n_methods[valid] += 1

        # Convert back to probability
        probability = 1.0 / (1.0 + np.exp(-log_odds))

        # Final classification
        is_galaxy = probability > 0.5

        # Confidence
        prob_confidence = 2.0 * np.abs(probability - 0.5)
        agreement_factor = np.minimum(n_methods / self.config.min_agreeing_methods, 1.0)
        confidence = prob_confidence * agreement_factor

        return is_galaxy, probability, confidence

    def classify(
        self,
        catalog: pd.DataFrame,
        spread_model: NDArray | None = None,
        spread_model_err: NDArray | None = None,
        concentration: NDArray | None = None,
        magnitudes: NDArray | None = None,
        images: NDArray | None = None,
        gaia_stars: NDArray | None = None,
        color_locus_distance: NDArray | None = None,
        mag_col: str = 'mag_auto',
    ) -> pd.DataFrame:
        """Run hybrid classification combining multiple methods.

        Parameters
        ----------
        catalog : pd.DataFrame
            Source catalog with features for ML classifier
        spread_model : NDArray, optional
            SPREAD_MODEL values
        spread_model_err : NDArray, optional
            SPREAD_MODEL errors
        concentration : NDArray, optional
            Concentration index values
        magnitudes : NDArray, optional
            Source magnitudes (uses mag_col from catalog if not provided)
        images : NDArray, optional
            Cutout images for deep learning classifier
        gaia_stars : NDArray, optional
            Boolean array of Gaia-confirmed stars
        color_locus_distance : NDArray, optional
            Distance from stellar locus in color-color space
        mag_col : str
            Column name for magnitudes in catalog

        Returns
        -------
        pd.DataFrame
            Classification results with columns:
            - is_galaxy: Final classification
            - is_star: Inverse of is_galaxy
            - probability_galaxy: Combined probability (0-1)
            - confidence: Classification confidence (0-1)
            - n_methods: Number of methods that contributed
            - method_agreement: Fraction of methods agreeing
            - primary_method: Method with highest weight contribution
        """
        n = len(catalog)

        # Get magnitudes
        if magnitudes is None:
            if mag_col in catalog.columns:
                magnitudes = catalog[mag_col].values
            else:
                magnitudes = np.full(n, 22.0)

        # Collect votes from each method
        votes = []

        # Gaia (if available) - highest confidence
        if gaia_stars is not None:
            gaia_vote = {
                'method': 'gaia',
                'is_galaxy': ~gaia_stars,
                'probability': np.where(gaia_stars, 0.0, 0.5),
                'confidence': np.where(gaia_stars, 1.0, 0.0),
            }
            votes.append(gaia_vote)

        # SPREAD_MODEL
        if spread_model is not None:
            if spread_model_err is None:
                spread_model_err = np.full(n, 0.003)

            sm_galaxy, sm_prob, sm_conf = self._classify_spread_model(
                spread_model, spread_model_err, magnitudes
            )
            votes.append({
                'method': 'spread_model',
                'is_galaxy': sm_galaxy,
                'probability': sm_prob,
                'confidence': sm_conf,
            })

        # Concentration
        if concentration is not None:
            c_galaxy, c_prob, c_conf = self._classify_concentration(concentration)
            votes.append({
                'method': 'concentration',
                'is_galaxy': c_galaxy,
                'probability': c_prob,
                'confidence': c_conf,
            })

        # ML classifier
        if self.ml_classifier is not None:
            ml_galaxy, ml_prob, ml_conf = self._classify_ml(catalog)
            votes.append({
                'method': 'ml_classifier',
                'is_galaxy': ml_galaxy,
                'probability': ml_prob,
                'confidence': ml_conf,
            })

        # Deep learning
        if self.dl_classifier is not None and images is not None:
            dl_galaxy, dl_prob, dl_conf = self._classify_deep_learning(
                images=images, catalog=catalog
            )
            votes.append({
                'method': 'deep_learning',
                'is_galaxy': dl_galaxy,
                'probability': dl_prob,
                'confidence': dl_conf,
            })

        # Color locus (if provided)
        if color_locus_distance is not None:
            # Stars are close to stellar locus (distance < threshold)
            cl_threshold = 0.2  # Typical threshold in color space
            cl_stars = color_locus_distance < cl_threshold
            cl_prob = 1.0 / (1.0 + np.exp(-10.0 * (color_locus_distance - cl_threshold)))
            cl_conf = 1.0 - np.exp(-5.0 * np.abs(color_locus_distance - cl_threshold))
            cl_conf[~np.isfinite(color_locus_distance)] = 0.0

            votes.append({
                'method': 'color_locus',
                'is_galaxy': ~cl_stars,
                'probability': cl_prob,
                'confidence': cl_conf,
            })

        # Combine votes
        if len(votes) == 0:
            # No methods available - return uncertain classification
            results = pd.DataFrame({
                'is_galaxy': np.ones(n, dtype=bool),
                'is_star': np.zeros(n, dtype=bool),
                'probability_galaxy': np.full(n, 0.5),
                'confidence': np.zeros(n),
                'n_methods': np.zeros(n, dtype=int),
                'method_agreement': np.zeros(n),
                'primary_method': ['none'] * n,
            })
            return results

        if self.config.use_bayesian:
            is_galaxy, probability, confidence = self._combine_votes_bayesian(
                votes, magnitudes
            )
        else:
            is_galaxy, probability, confidence = self._combine_votes_weighted(
                votes, magnitudes
            )

        # Calculate method agreement
        n_methods = np.zeros(n, dtype=int)
        n_agreeing = np.zeros(n, dtype=int)
        primary_method = ['unknown'] * n
        primary_weight = np.zeros(n)

        for vote in votes:
            valid = vote['confidence'] >= self.config.min_confidence
            n_methods[valid] += 1
            agrees = vote['is_galaxy'] == is_galaxy
            n_agreeing[valid & agrees] += 1

            # Track primary method (highest weighted contribution)
            method = vote['method']
            weights = np.array([
                self._get_weight_for_magnitude(method, m) * vote['confidence'][i]
                for i, m in enumerate(magnitudes)
            ])
            for i in range(n):
                if weights[i] > primary_weight[i] and vote['confidence'][i] >= self.config.min_confidence:
                    primary_weight[i] = weights[i]
                    primary_method[i] = method

        method_agreement = np.where(n_methods > 0, n_agreeing / n_methods, 0.0)

        # Build results DataFrame
        results = pd.DataFrame({
            'is_galaxy': is_galaxy,
            'is_star': ~is_galaxy,
            'probability_galaxy': probability,
            'confidence': confidence,
            'n_methods': n_methods,
            'method_agreement': method_agreement,
            'primary_method': primary_method,
        })

        return results

    def get_method_diagnostics(
        self,
        catalog: pd.DataFrame,
        spread_model: NDArray | None = None,
        spread_model_err: NDArray | None = None,
        concentration: NDArray | None = None,
        magnitudes: NDArray | None = None,
        mag_col: str = 'mag_auto',
    ) -> pd.DataFrame:
        """Get individual classifications from each method for diagnostics.

        Useful for understanding how each method performs and where they
        disagree.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for each method's classification
        """
        n = len(catalog)

        if magnitudes is None:
            if mag_col in catalog.columns:
                magnitudes = catalog[mag_col].values
            else:
                magnitudes = np.full(n, 22.0)

        diagnostics = pd.DataFrame(index=range(n))
        diagnostics['magnitude'] = magnitudes

        # SPREAD_MODEL
        if spread_model is not None:
            if spread_model_err is None:
                spread_model_err = np.full(n, 0.003)
            sm_galaxy, sm_prob, sm_conf = self._classify_spread_model(
                spread_model, spread_model_err, magnitudes
            )
            diagnostics['spread_model_galaxy'] = sm_galaxy
            diagnostics['spread_model_prob'] = sm_prob
            diagnostics['spread_model_conf'] = sm_conf

        # Concentration
        if concentration is not None:
            c_galaxy, c_prob, c_conf = self._classify_concentration(concentration)
            diagnostics['concentration_galaxy'] = c_galaxy
            diagnostics['concentration_prob'] = c_prob
            diagnostics['concentration_conf'] = c_conf

        # ML classifier
        if self.ml_classifier is not None:
            ml_galaxy, ml_prob, ml_conf = self._classify_ml(catalog)
            diagnostics['ml_galaxy'] = ml_galaxy
            diagnostics['ml_prob'] = ml_prob
            diagnostics['ml_conf'] = ml_conf

        return diagnostics


def create_hybrid_classifier(
    ml_type: str = 'xgboost',
    ml_model_path: str | None = None,
    dl_model_path: str | None = None,
    config: HybridClassifierConfig | None = None,
) -> HybridStarGalaxyClassifier:
    """Factory function to create a hybrid classifier with optional models.

    Parameters
    ----------
    ml_type : str
        Type of ML classifier: 'random_forest' or 'xgboost'
    ml_model_path : str, optional
        Path to saved ML model. If not provided, ML classifier is not loaded.
    dl_model_path : str, optional
        Path to saved deep learning model. If not provided, DL is not used.
    config : HybridClassifierConfig, optional
        Configuration for the hybrid classifier

    Returns
    -------
    HybridStarGalaxyClassifier
        Configured hybrid classifier

    Examples
    --------
    >>> # Create hybrid with XGBoost
    >>> hybrid = create_hybrid_classifier(
    ...     ml_type='xgboost',
    ...     ml_model_path='models/star_galaxy_xgb.joblib'
    ... )
    >>>
    >>> # Create hybrid with both ML and DL
    >>> hybrid = create_hybrid_classifier(
    ...     ml_type='random_forest',
    ...     ml_model_path='models/star_galaxy_rf.joblib',
    ...     dl_model_path='models/zoobot_finetuned.pt'
    ... )
    """
    ml_classifier = None
    dl_classifier = None

    # Load ML classifier if path provided
    if ml_model_path is not None:
        try:
            from morphology.ml_classifier import create_classifier
            ml_classifier = create_classifier(ml_type)
            ml_classifier.load(ml_model_path)
        except Exception as e:
            print(f"Warning: Could not load ML classifier from {ml_model_path}: {e}")

    # Load DL classifier if path provided
    if dl_model_path is not None:
        try:
            from morphology.deep_learning_classifier import ZoobotStarGalaxyClassifier
            dl_classifier = ZoobotStarGalaxyClassifier()
            dl_classifier.load_pretrained(dl_model_path)
        except Exception as e:
            print(f"Warning: Could not load DL classifier from {dl_model_path}: {e}")

    return HybridStarGalaxyClassifier(
        config=config,
        ml_classifier=ml_classifier,
        deep_learning_classifier=dl_classifier,
    )
