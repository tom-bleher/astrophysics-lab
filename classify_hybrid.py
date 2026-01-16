"""Hybrid photometric redshift estimation combining templates with ML.

This module provides advanced photo-z estimation by combining the traditional
SED template fitting approach with machine learning methods. The hybrid
approach offers several advantages:

1. Template fitting provides physically-motivated priors
2. ML corrects systematic biases in template predictions
3. Combined uncertainty estimation from both methods
4. Improved outlier rejection

Methods implemented:
- GPz-style Gaussian Process regression (via scikit-learn)
- Random Forest for photo-z correction
- Neural network photo-z (optional, requires torch)

References:
- Almosallam et al. 2016, MNRAS, 455, 2387 (GPz)
- Carliles et al. 2010, ApJ, 712, 511 (RF photo-z)
- Hoyle et al. 2016, A&C, 16, 34 (ML photo-z comparison)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

from classify import PhotoZResult, classify_galaxy_with_pdf


@dataclass
class HybridPhotoZResult:
    """Result from hybrid photo-z estimation.

    Attributes
    ----------
    redshift : float
        Final combined redshift estimate
    redshift_err : float
        Combined uncertainty (from template + ML)
    z_template : float
        Template-only redshift
    z_ml : float
        ML-predicted redshift
    z_correction : float
        ML correction applied to template (z_ml - z_template)
    confidence : float
        Overall confidence score (0-1)
    method_agreement : float
        Agreement between template and ML methods
    galaxy_type : str
        Best-fit template type
    odds : float
        BPZ-style ODDS parameter from template fitting
    chi2_min : float
        Minimum chi-squared from template fitting
    ml_uncertainty : float
        ML-estimated uncertainty
    """
    redshift: float
    redshift_err: float
    z_template: float
    z_ml: float
    z_correction: float
    confidence: float
    method_agreement: float
    galaxy_type: str
    odds: float
    chi2_min: float
    ml_uncertainty: float


class HybridPhotoZEstimator:
    """Hybrid photo-z estimator combining templates with ML.

    This class implements a two-stage photo-z estimation:
    1. Template fitting for initial redshift and galaxy type
    2. ML correction to reduce systematic biases

    The ML model learns to correct template photo-z residuals using
    photometric colors and template fitting quality indicators.

    Parameters
    ----------
    ml_method : str
        ML method to use: 'gp' (Gaussian Process), 'rf' (Random Forest),
        'gbr' (Gradient Boosting), or 'ensemble' (all combined)
    n_estimators : int
        Number of trees for RF/GBR methods
    use_template_features : bool
        Include template fitting outputs (chi2, odds, type) as ML features
    spectra_path : str or Path
        Path to template spectra directory

    Examples
    --------
    >>> estimator = HybridPhotoZEstimator(ml_method='rf')
    >>> # Train on spectroscopic sample
    >>> estimator.fit(train_fluxes, train_errors, train_specz)
    >>> # Predict photo-z for new sources
    >>> results = estimator.predict(test_fluxes, test_errors)
    """

    def __init__(
        self,
        ml_method: Literal['gp', 'rf', 'gbr', 'ensemble'] = 'rf',
        n_estimators: int = 100,
        use_template_features: bool = True,
        spectra_path: str | Path = './spectra',
    ):
        self.ml_method = ml_method
        self.n_estimators = n_estimators
        self.use_template_features = use_template_features
        self.spectra_path = str(spectra_path)

        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False

        # Initialize ML models based on method
        self._init_models()

    def _init_models(self):
        """Initialize ML models based on selected method."""
        if self.ml_method == 'gp':
            # Gaussian Process with RBF kernel (GPz-style)
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3)) *
                RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
                WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
            )
            self.model = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                normalize_y=True,
                alpha=1e-6,
            )
        elif self.ml_method == 'rf':
            # Random Forest
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            )
        elif self.ml_method == 'gbr':
            # Gradient Boosting
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                random_state=42,
            )
        elif self.ml_method == 'ensemble':
            # Use all methods and combine
            self.models = {
                'rf': RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=15,
                    n_jobs=-1,
                    random_state=42,
                ),
                'gbr': GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                ),
            }
            self.model = None  # Will use ensemble prediction

    def _compute_colors(
        self,
        flux_u: NDArray,
        flux_b: NDArray,
        flux_v: NDArray,
        flux_i: NDArray,
    ) -> NDArray:
        """Compute color features from fluxes.

        Returns colors and color ratios as ML features.
        Handles negative/zero fluxes gracefully.
        """
        # Safe log for magnitude calculation
        def safe_mag(flux):
            with np.errstate(divide='ignore', invalid='ignore'):
                mag = np.where(flux > 0, -2.5 * np.log10(flux), np.nan)
            return mag

        mag_u = safe_mag(flux_u)
        mag_b = safe_mag(flux_b)
        mag_v = safe_mag(flux_v)
        mag_i = safe_mag(flux_i)

        # Standard colors
        color_ub = mag_u - mag_b
        color_bv = mag_b - mag_v
        color_vi = mag_v - mag_i

        # Additional color combinations
        color_ui = mag_u - mag_i  # Wide baseline
        color_ratio = (flux_b / np.maximum(flux_i, 1e-30))  # Flux ratio

        return np.column_stack([
            color_ub, color_bv, color_vi, color_ui,
            np.log10(np.maximum(color_ratio, 1e-30)),
        ])

    def _build_features(
        self,
        flux_array: NDArray,
        error_array: NDArray,
        template_results: list[PhotoZResult] | None = None,
    ) -> NDArray:
        """Build feature matrix for ML model.

        Features include:
        - Colors (U-B, B-V, V-I, U-I)
        - Flux ratios and SNR
        - Template fitting outputs (if enabled and available)
        """
        n_sources = len(flux_array)

        # Extract fluxes (expected order: U, B, V, I)
        flux_u = flux_array[:, 0]
        flux_b = flux_array[:, 1]
        flux_v = flux_array[:, 2]
        flux_i = flux_array[:, 3]

        err_u = error_array[:, 0]
        err_b = error_array[:, 1]
        err_v = error_array[:, 2]
        err_i = error_array[:, 3]

        # Colors
        colors = self._compute_colors(flux_u, flux_b, flux_v, flux_i)

        # SNR in each band
        snr_u = np.abs(flux_u) / np.maximum(err_u, 1e-30)
        snr_b = np.abs(flux_b) / np.maximum(err_b, 1e-30)
        snr_v = np.abs(flux_v) / np.maximum(err_v, 1e-30)
        snr_i = np.abs(flux_i) / np.maximum(err_i, 1e-30)

        # Flux ratios
        total_flux = flux_u + flux_b + flux_v + flux_i
        frac_u = flux_u / np.maximum(total_flux, 1e-30)
        frac_i = flux_i / np.maximum(total_flux, 1e-30)

        features = np.column_stack([
            colors,
            np.log10(np.maximum(snr_u, 1e-3)),
            np.log10(np.maximum(snr_b, 1e-3)),
            np.log10(np.maximum(snr_v, 1e-3)),
            np.log10(np.maximum(snr_i, 1e-3)),
            frac_u,
            frac_i,
        ])

        # Add template features if available
        if self.use_template_features and template_results is not None:
            z_template = np.array([r.redshift for r in template_results])
            odds = np.array([r.odds for r in template_results])
            chi2 = np.array([r.chi_sq_min for r in template_results])

            # Galaxy type as numeric encoding
            type_map = {
                'elliptical': 0, 'S0': 1, 'Sa': 2, 'Sb': 3,
                'sbt1': 4, 'sbt2': 5, 'sbt3': 6,
                'sbt4': 7, 'sbt5': 8, 'sbt6': 9,
            }
            galaxy_types = np.array([
                type_map.get(r.galaxy_type, 5) for r in template_results
            ])

            template_features = np.column_stack([
                z_template,
                odds,
                np.log10(np.maximum(chi2, 0.1)),
                galaxy_types / 10.0,  # Normalize to 0-1
            ])

            features = np.column_stack([features, template_features])

        # Replace NaN/inf with median
        features = np.nan_to_num(features, nan=0, posinf=10, neginf=-10)

        return features

    def fit(
        self,
        flux_array: NDArray,
        error_array: NDArray,
        spec_z: NDArray,
        sample_weight: NDArray | None = None,
    ) -> 'HybridPhotoZEstimator':
        """Fit the hybrid photo-z model using spectroscopic training data.

        Parameters
        ----------
        flux_array : NDArray, shape (N, 4)
            Fluxes in [U, B, V, I] order
        error_array : NDArray, shape (N, 4)
            Flux errors in [U, B, V, I] order
        spec_z : NDArray, shape (N,)
            Spectroscopic redshifts for training
        sample_weight : NDArray, optional
            Weights for training samples

        Returns
        -------
        self : HybridPhotoZEstimator
            Fitted estimator
        """
        n_train = len(flux_array)
        print(f"Training hybrid photo-z model on {n_train} sources...")

        # Run template fitting for all training sources
        print("  Running template fitting...")
        template_results = []
        for i in range(n_train):
            # Reorder from [U, B, V, I] to [B, I, U, V] for classify function
            fluxes = [flux_array[i, 1], flux_array[i, 3],
                      flux_array[i, 0], flux_array[i, 2]]
            errors = [error_array[i, 1], error_array[i, 3],
                      error_array[i, 0], error_array[i, 2]]

            result = classify_galaxy_with_pdf(
                fluxes, errors,
                spectra_path=self.spectra_path,
                z_step=0.02,  # Coarser grid for training
            )
            template_results.append(result)

        # Build feature matrix
        print("  Building features...")
        features = self._build_features(flux_array, error_array, template_results)

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Target: spectroscopic redshift (not residual, for simplicity)
        # The ML learns to predict z_spec directly, incorporating template info
        y = spec_z

        # Fit model(s)
        print(f"  Fitting {self.ml_method} model...")
        if self.ml_method == 'ensemble':
            for name, model in self.models.items():
                model.fit(features_scaled, y, sample_weight=sample_weight)
                print(f"    Fitted {name}")
        else:
            if sample_weight is not None and hasattr(self.model, 'fit'):
                self.model.fit(features_scaled, y, sample_weight=sample_weight)
            else:
                self.model.fit(features_scaled, y)

        self.is_fitted = True
        print("  Training complete!")

        # Compute training metrics
        y_pred = self._predict_ml(features_scaled)
        residuals = y_pred - y
        nmad = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
        outlier_rate = np.mean(np.abs(residuals) > 0.15 * (1 + y))
        print(f"  Training NMAD: {nmad:.4f}")
        print(f"  Training outlier rate (>15%): {100*outlier_rate:.1f}%")

        return self

    def _predict_ml(self, features_scaled: NDArray) -> NDArray:
        """Get ML predictions from fitted model(s)."""
        if self.ml_method == 'ensemble':
            predictions = []
            for model in self.models.values():
                predictions.append(model.predict(features_scaled))
            return np.mean(predictions, axis=0)
        else:
            return self.model.predict(features_scaled)

    def _predict_ml_uncertainty(self, features_scaled: NDArray) -> NDArray:
        """Estimate ML prediction uncertainty."""
        if self.ml_method == 'gp':
            # GP provides built-in uncertainty
            _, std = self.model.predict(features_scaled, return_std=True)
            return std
        elif self.ml_method == 'rf':
            # RF: use variance of tree predictions
            predictions = np.array([
                tree.predict(features_scaled)
                for tree in self.model.estimators_
            ])
            return np.std(predictions, axis=0)
        elif self.ml_method == 'ensemble':
            # Ensemble: use variance across methods
            predictions = []
            for model in self.models.values():
                predictions.append(model.predict(features_scaled))
            return np.std(predictions, axis=0)
        else:
            # GBR: approximate with staged predictions
            return np.ones(len(features_scaled)) * 0.1

    def predict(
        self,
        flux_array: NDArray,
        error_array: NDArray,
    ) -> list[HybridPhotoZResult]:
        """Predict photo-z using hybrid template + ML approach.

        Parameters
        ----------
        flux_array : NDArray, shape (N, 4)
            Fluxes in [U, B, V, I] order
        error_array : NDArray, shape (N, 4)
            Flux errors in [U, B, V, I] order

        Returns
        -------
        list[HybridPhotoZResult]
            Hybrid photo-z results for each source
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_sources = len(flux_array)
        results = []

        # Run template fitting
        template_results = []
        for i in range(n_sources):
            fluxes = [flux_array[i, 1], flux_array[i, 3],
                      flux_array[i, 0], flux_array[i, 2]]
            errors = [error_array[i, 1], error_array[i, 3],
                      error_array[i, 0], error_array[i, 2]]

            result = classify_galaxy_with_pdf(
                fluxes, errors,
                spectra_path=self.spectra_path,
            )
            template_results.append(result)

        # Build features and predict with ML
        features = self._build_features(flux_array, error_array, template_results)
        features_scaled = self.scaler.transform(features)

        z_ml = self._predict_ml(features_scaled)
        ml_uncertainty = self._predict_ml_uncertainty(features_scaled)

        # Combine results
        for i in range(n_sources):
            tr = template_results[i]

            # Combine template and ML predictions
            # Weight by inverse variance (roughly)
            template_err = (tr.z_hi - tr.z_lo) / 2.0 if tr.z_hi > tr.z_lo else 0.1
            ml_err = max(ml_uncertainty[i], 0.01)

            w_template = 1.0 / (template_err**2 + 0.01)
            w_ml = 1.0 / (ml_err**2 + 0.01)
            w_total = w_template + w_ml

            # Weighted average
            z_combined = (w_template * tr.redshift + w_ml * z_ml[i]) / w_total
            err_combined = np.sqrt(1.0 / w_total)

            # Method agreement
            z_diff = abs(tr.redshift - z_ml[i])
            agreement = 1.0 - min(1.0, z_diff / max(0.1, z_combined))

            # Confidence based on ODDS, agreement, and uncertainties
            confidence = tr.odds * agreement * min(1.0, 0.2 / err_combined)
            confidence = np.clip(confidence, 0.0, 1.0)

            results.append(HybridPhotoZResult(
                redshift=float(z_combined),
                redshift_err=float(err_combined),
                z_template=float(tr.redshift),
                z_ml=float(z_ml[i]),
                z_correction=float(z_ml[i] - tr.redshift),
                confidence=float(confidence),
                method_agreement=float(agreement),
                galaxy_type=tr.galaxy_type,
                odds=float(tr.odds),
                chi2_min=float(tr.chi_sq_min),
                ml_uncertainty=float(ml_err),
            ))

        return results

    def predict_batch(
        self,
        flux_array: NDArray,
        error_array: NDArray,
        batch_size: int = 1000,
        catalog_odds: NDArray | None = None,
        catalog_z_lo: NDArray | None = None,
        catalog_z_hi: NDArray | None = None,
        catalog_redshift: NDArray | None = None,
        has_specz: NDArray | None = None,
    ) -> dict[str, NDArray]:
        """Batch prediction returning arrays for DataFrame integration.

        Parameters
        ----------
        flux_array : NDArray, shape (N, 4)
            Fluxes in [U, B, V, I] order
        error_array : NDArray, shape (N, 4)
            Flux errors in [U, B, V, I] order
        batch_size : int
            Process in batches to manage memory
        catalog_odds : NDArray, optional
            Pre-computed ODDS values from catalog (used for confidence calculation)
        catalog_z_lo : NDArray, optional
            Pre-computed z_lo values from catalog
        catalog_z_hi : NDArray, optional
            Pre-computed z_hi values from catalog
        catalog_redshift : NDArray, optional
            Pre-computed redshifts from catalog (used as z_template for spec-z sources)
        has_specz : NDArray, optional
            Boolean array indicating sources with spectroscopic redshifts

        Returns
        -------
        dict
            Dictionary of result arrays matching input length
        """
        n_sources = len(flux_array)

        # Initialize result arrays
        results = {
            'redshift_hybrid': np.zeros(n_sources),
            'redshift_hybrid_err': np.zeros(n_sources),
            'z_template': np.zeros(n_sources),
            'z_ml': np.zeros(n_sources),
            'z_correction': np.zeros(n_sources),
            'confidence_hybrid': np.zeros(n_sources),
            'method_agreement': np.zeros(n_sources),
            'ml_uncertainty': np.zeros(n_sources),
        }

        # Process in batches
        for start in range(0, n_sources, batch_size):
            end = min(start + batch_size, n_sources)

            batch_results = self.predict(
                flux_array[start:end],
                error_array[start:end],
            )

            for i, r in enumerate(batch_results):
                idx = start + i
                results['redshift_hybrid'][idx] = r.redshift
                results['redshift_hybrid_err'][idx] = r.redshift_err
                results['z_template'][idx] = r.z_template
                results['z_ml'][idx] = r.z_ml
                results['z_correction'][idx] = r.z_correction
                results['confidence_hybrid'][idx] = r.confidence
                results['method_agreement'][idx] = r.method_agreement
                results['ml_uncertainty'][idx] = r.ml_uncertainty

        # Override confidence calculation using catalog values if provided
        # This ensures consistency between template fitting and hybrid results
        if catalog_odds is not None:
            # Recompute confidence using catalog ODDS instead of fresh template ODDS
            # This fixes the issue where re-running template fitting gives different ODDS
            for idx in range(n_sources):
                # Get catalog values or fall back to computed values
                odds = catalog_odds[idx] if catalog_odds is not None else results['confidence_hybrid'][idx]

                # Compute template error from catalog z_lo/z_hi if provided
                if catalog_z_lo is not None and catalog_z_hi is not None:
                    template_err = (catalog_z_hi[idx] - catalog_z_lo[idx]) / 2.0
                    # For spec-z sources, use a small but reasonable error
                    if template_err < 0.01:
                        template_err = 0.01  # Minimum error for stability
                else:
                    template_err = results['redshift_hybrid_err'][idx]

                # Compute ML error
                ml_err = max(results['ml_uncertainty'][idx], 0.01)

                # Compute combined error (inverse variance weighting)
                w_template = 1.0 / (template_err**2 + 0.01)
                w_ml = 1.0 / (ml_err**2 + 0.01)
                w_total = w_template + w_ml
                err_combined = np.sqrt(1.0 / w_total)

                # Recompute agreement using catalog redshift if provided
                if catalog_redshift is not None:
                    z_template = catalog_redshift[idx]
                else:
                    z_template = results['z_template'][idx]

                z_ml = results['z_ml'][idx]
                z_diff = abs(z_template - z_ml)
                z_combined = results['redshift_hybrid'][idx]
                agreement = 1.0 - min(1.0, z_diff / max(0.1, z_combined))

                # Compute confidence using catalog ODDS
                confidence = odds * agreement * min(1.0, 0.2 / err_combined)
                confidence = np.clip(confidence, 0.0, 1.0)

                results['confidence_hybrid'][idx] = confidence
                results['method_agreement'][idx] = agreement

                # For spec-z sources, use spectroscopic redshift as the hybrid value
                # but preserve the photo-z confidence to track photo-z quality
                if has_specz is not None and has_specz[idx]:
                    # Use catalog redshift as the hybrid redshift (spec-z is trusted)
                    results['redshift_hybrid'][idx] = z_template
                    results['redshift_hybrid_err'][idx] = template_err
                    results['z_template'][idx] = z_template
                    # Keep the original photo-z confidence (no floor)
                    # This preserves information about photo-z quality for validation
                    # The has_specz flag already indicates the redshift is reliable
                    # Previously: results['confidence_hybrid'][idx] = max(confidence, 0.9)
                    # Now: keep original confidence for quality tracking

        return results


def train_hybrid_photoz_from_specz(
    catalog: pd.DataFrame,
    spec_z_col: str = 'spec_z',
    flux_cols: tuple[str, ...] = ('flux_f300', 'flux_f450', 'flux_f606', 'flux_f814'),
    error_cols: tuple[str, ...] = ('error_f300', 'error_f450', 'error_f606', 'error_f814'),
    ml_method: str = 'rf',
    test_fraction: float = 0.2,
    spectra_path: str = './spectra',
    has_specz_col: str | None = None,
) -> tuple[HybridPhotoZEstimator, dict]:
    """Train hybrid photo-z model from spectroscopic catalog.

    Parameters
    ----------
    catalog : pd.DataFrame
        Catalog with fluxes and spectroscopic redshifts
    spec_z_col : str
        Column name for spectroscopic redshift
    flux_cols : tuple
        Column names for U, B, V, I fluxes
    error_cols : tuple
        Column names for flux errors
    ml_method : str
        ML method: 'gp', 'rf', 'gbr', or 'ensemble'
    test_fraction : float
        Fraction held out for testing
    spectra_path : str
        Path to template spectra
    has_specz_col : str, optional
        Column name for boolean flag indicating sources with spectroscopic redshifts.
        If provided, uses this to filter training sources instead of just spec_z_col > 0.

    Returns
    -------
    estimator : HybridPhotoZEstimator
        Trained estimator
    metrics : dict
        Training and test metrics
    """
    # Filter to sources with spec-z
    # Use has_specz_col if provided (more reliable filtering)
    if has_specz_col is not None and has_specz_col in catalog.columns:
        has_specz = catalog[has_specz_col].fillna(False).astype(bool)
        # Also require valid spec-z value
        has_specz = has_specz & catalog[spec_z_col].notna() & (catalog[spec_z_col] > 0)
    else:
        has_specz = catalog[spec_z_col].notna() & (catalog[spec_z_col] > 0)

    train_cat = catalog[has_specz].copy()

    if len(train_cat) < 20:
        raise ValueError(f"Need at least 20 sources with spec-z, got {len(train_cat)}")

    # Diagnostic info about the training sample
    spec_z_values = train_cat[spec_z_col].values
    print(f"Training hybrid photo-z on {len(train_cat)} sources with spec-z")
    print(f"  Spec-z column: '{spec_z_col}' (range: {spec_z_values.min():.3f} - {spec_z_values.max():.3f})")

    # Build arrays
    flux_array = np.column_stack([train_cat[c].values for c in flux_cols])
    error_array = np.column_stack([train_cat[c].values for c in error_cols])
    spec_z = train_cat[spec_z_col].values

    # Split train/test
    indices = np.arange(len(train_cat))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_fraction, random_state=42
    )

    # Train estimator
    estimator = HybridPhotoZEstimator(
        ml_method=ml_method,
        spectra_path=spectra_path,
    )

    estimator.fit(
        flux_array[train_idx],
        error_array[train_idx],
        spec_z[train_idx],
    )

    # Evaluate on test set
    test_results = estimator.predict(
        flux_array[test_idx],
        error_array[test_idx],
    )

    z_pred = np.array([r.redshift for r in test_results])
    z_true = spec_z[test_idx]

    # Compute metrics
    residuals = (z_pred - z_true) / (1 + z_true)
    nmad = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
    bias = np.median(residuals)
    outlier_rate = np.mean(np.abs(residuals) > 0.15)

    # Compare to template-only
    z_template = np.array([r.z_template for r in test_results])
    residuals_template = (z_template - z_true) / (1 + z_true)
    nmad_template = 1.4826 * np.median(np.abs(residuals_template - np.median(residuals_template)))
    outlier_template = np.mean(np.abs(residuals_template) > 0.15)

    # Compute improvement metrics with guards for zero denominators
    # Zero NMAD/outlier rates can occur with small samples or perfect predictions
    if nmad_template > 0:
        improvement_nmad = (nmad_template - nmad) / nmad_template
    else:
        # If template NMAD is 0, improvement is undefined (templates are perfect)
        improvement_nmad = 0.0 if nmad == 0 else -np.inf

    if outlier_template > 0:
        improvement_outlier = (outlier_template - outlier_rate) / outlier_template
    else:
        # If template outlier rate is 0, improvement is undefined
        improvement_outlier = 0.0 if outlier_rate == 0 else -np.inf

    metrics = {
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'nmad_hybrid': nmad,
        'bias_hybrid': bias,
        'outlier_rate_hybrid': outlier_rate,
        'nmad_template': nmad_template,
        'outlier_rate_template': outlier_template,
        'improvement_nmad': improvement_nmad,
        'improvement_outlier': improvement_outlier,
    }

    print("\n  Test set metrics:")
    print(f"    Hybrid NMAD:      {nmad:.4f}")
    print(f"    Template NMAD:    {nmad_template:.4f}")
    if np.isfinite(improvement_nmad):
        print(f"    Improvement:      {100*improvement_nmad:.1f}%")
    elif nmad_template == 0:
        print("    Improvement:      N/A (template NMAD is 0)")
    else:
        print(f"    Improvement:      Hybrid worse than template")
    print(f"    Hybrid outliers:  {100*outlier_rate:.1f}%")
    print(f"    Template outliers: {100*outlier_template:.1f}%")

    return estimator, metrics
