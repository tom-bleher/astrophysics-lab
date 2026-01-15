"""Comprehensive quality control pipeline for galaxy classification.

This module provides:
1. Quality flag computation and assessment
2. External catalog validation
3. Bias correction estimation and application
4. Zoobot morphology cross-validation
5. Template confusion matrix analysis
6. Unified quality report generation

Usage:
    from quality_control import QualityControlPipeline

    qc = QualityControlPipeline(catalog, output_dir='./output/qc')
    report = qc.run_full_validation()
    qc.save_report()
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Local imports
from classify import compute_chi2_flag, compute_odds_flag

# External catalog validation removed
HAS_EXTERNAL_CATALOGS = False
ValidationReport = None


try:
    from validation.zoobot_morphology import ZoobotValidationReport
    HAS_ZOOBOT = True
except ImportError:
    HAS_ZOOBOT = False
    ZoobotValidationReport = None


# Galaxy types from classify.py
GALAXY_TYPES = [
    "Ell", "S0", "Sa", "Sb", "sbt1", "sbt2", "sbt3", "sbt4", "sbt5", "sbt6"
]


@dataclass
class TemplateConfusionReport:
    """Report on template fitting confusion/degeneracy.

    Attributes
    ----------
    confusion_matrix : NDArray
        Shape (n_templates, n_templates): counts of primary vs secondary template
    template_names : list[str]
        Template names in order
    degeneracy_pairs : list[tuple[str, str, float]]
        Most confused template pairs with frequency
    mean_delta_chi2 : dict[str, float]
        Mean chi2 difference to second-best by primary type
    """
    confusion_matrix: NDArray
    template_names: list[str]
    degeneracy_pairs: list[tuple]
    mean_delta_chi2: dict

    def __str__(self) -> str:
        lines = [
            "=== Template Confusion Analysis ===",
            f"Templates analyzed: {len(self.template_names)}",
            "",
            "Most confused pairs (primary -> secondary, frequency):",
        ]
        for t1, t2, freq in self.degeneracy_pairs[:5]:
            lines.append(f"  {t1} -> {t2}: {100*freq:.1f}%")
        return "\n".join(lines)


@dataclass
class TypeValidationReport:
    """Report on galaxy type classification validation.

    Cross-validates SED-based types with colors and morphology.
    """
    n_validated: int = 0

    # Color-type agreement
    color_agreement: float = 0.0  # Fraction where colors match expected for type
    color_disagreements: dict = field(default_factory=dict)  # By type

    # Morphology-type agreement
    morphology_agreement: float = 0.0
    morphology_disagreements: dict = field(default_factory=dict)

    # Type confidence
    mean_type_confidence: float = 0.0
    low_confidence_fraction: float = 0.0  # Fraction with confidence < 0.5

    # Suspicious classifications (color/morphology mismatch)
    n_suspicious: int = 0
    suspicious_fraction: float = 0.0

    def __str__(self) -> str:
        return f"""
=== Type Validation Report ===
Sources validated: {self.n_validated}

Color-Type Agreement: {100*self.color_agreement:.1f}%
Morphology-Type Agreement: {100*self.morphology_agreement:.1f}%

Mean Type Confidence: {self.mean_type_confidence:.2f}
Low Confidence Fraction: {100*self.low_confidence_fraction:.1f}%

Suspicious Classifications: {self.n_suspicious} ({100*self.suspicious_fraction:.1f}%)
"""


# Expected color ranges for each galaxy type (U-B, B-V, V-I)
# Based on Coleman, Wu & Weedman (1980) and Kinney et al. (1996)
EXPECTED_COLORS = {
    'elliptical': {'U-B': (0.3, 0.7), 'B-V': (0.8, 1.1), 'V-I': (1.0, 1.4)},
    'S0': {'U-B': (0.2, 0.6), 'B-V': (0.7, 1.0), 'V-I': (0.9, 1.3)},
    'Sa': {'U-B': (0.0, 0.4), 'B-V': (0.5, 0.9), 'V-I': (0.8, 1.2)},
    'Sb': {'U-B': (-0.2, 0.3), 'B-V': (0.4, 0.8), 'V-I': (0.6, 1.1)},
    'sbt1': {'U-B': (-0.5, 0.1), 'B-V': (0.2, 0.6), 'V-I': (0.4, 0.9)},
    'sbt2': {'U-B': (-0.6, 0.0), 'B-V': (0.1, 0.5), 'V-I': (0.3, 0.8)},
    'sbt3': {'U-B': (-0.7, -0.1), 'B-V': (0.0, 0.4), 'V-I': (0.2, 0.7)},
    'sbt4': {'U-B': (-0.8, -0.2), 'B-V': (-0.1, 0.3), 'V-I': (0.1, 0.6)},
    'sbt5': {'U-B': (-0.9, -0.3), 'B-V': (-0.2, 0.2), 'V-I': (0.0, 0.5)},
    'sbt6': {'U-B': (-1.0, -0.4), 'B-V': (-0.3, 0.1), 'V-I': (-0.1, 0.4)},
}

# Expected concentration ranges (C = 5 * log10(r80/r20))
# Ellipticals are more concentrated, starbursts are more extended
EXPECTED_CONCENTRATION = {
    'elliptical': (2.8, 4.5),  # High concentration
    'S0': (2.5, 4.0),
    'Sa': (2.2, 3.5),
    'Sb': (2.0, 3.2),
    'sbt1': (1.8, 3.0),
    'sbt2': (1.6, 2.8),
    'sbt3': (1.5, 2.7),
    'sbt4': (1.4, 2.6),
    'sbt5': (1.3, 2.5),
    'sbt6': (1.2, 2.4),  # Low concentration (extended)
}


def validate_type_with_colors(
    catalog: pd.DataFrame,
    type_col: str = 'galaxy_type',
    tolerance: float = 0.3,
) -> tuple[NDArray, dict]:
    """Validate galaxy types using observed colors.

    Parameters
    ----------
    catalog : pd.DataFrame
        Must contain flux columns and galaxy_type
    type_col : str
        Column with galaxy type
    tolerance : float
        Extra tolerance on color ranges (magnitudes)

    Returns
    -------
    agreement_mask : NDArray[bool]
        True where colors agree with type
    disagreement_details : dict
        Details of disagreements by type
    """
    n = len(catalog)
    agreement = np.ones(n, dtype=bool)
    details = {}

    # Compute colors from fluxes (if available)
    has_colors = all(col in catalog.columns for col in ['flux_u', 'flux_b', 'flux_v', 'flux_i'])

    if not has_colors:
        # Check for pre-computed colors
        has_colors = all(col in catalog.columns for col in ['U-B', 'B-V', 'V-I'])
        if has_colors:
            ub = catalog['U-B'].values
            bv = catalog['B-V'].values
            vi = catalog['V-I'].values
        else:
            return agreement, {'error': 'No flux or color columns available'}
    else:
        # Compute AB magnitudes and colors
        # Avoid log of zero/negative
        flux_u = np.maximum(catalog['flux_u'].values, 1e-10)
        flux_b = np.maximum(catalog['flux_b'].values, 1e-10)
        flux_v = np.maximum(catalog['flux_v'].values, 1e-10)
        flux_i = np.maximum(catalog['flux_i'].values, 1e-10)

        mag_u = -2.5 * np.log10(flux_u)
        mag_b = -2.5 * np.log10(flux_b)
        mag_v = -2.5 * np.log10(flux_v)
        mag_i = -2.5 * np.log10(flux_i)

        ub = mag_u - mag_b
        bv = mag_b - mag_v
        vi = mag_v - mag_i

    # Check each source
    for gtype in EXPECTED_COLORS:
        mask = catalog[type_col] == gtype
        if mask.sum() == 0:
            continue

        expected = EXPECTED_COLORS[gtype]
        type_agreement = np.ones(mask.sum(), dtype=bool)

        # Check U-B
        ub_lo, ub_hi = expected['U-B']
        ub_ok = (ub[mask] >= ub_lo - tolerance) & (ub[mask] <= ub_hi + tolerance)

        # Check B-V
        bv_lo, bv_hi = expected['B-V']
        bv_ok = (bv[mask] >= bv_lo - tolerance) & (bv[mask] <= bv_hi + tolerance)

        # Check V-I
        vi_lo, vi_hi = expected['V-I']
        vi_ok = (vi[mask] >= vi_lo - tolerance) & (vi[mask] <= vi_hi + tolerance)

        # Require at least 2 of 3 colors to agree
        n_agree = ub_ok.astype(int) + bv_ok.astype(int) + vi_ok.astype(int)
        type_agreement = n_agree >= 2

        # Update overall agreement
        agreement[mask] = type_agreement

        # Record disagreements
        n_disagree = (~type_agreement).sum()
        if n_disagree > 0:
            details[gtype] = {
                'n_total': int(mask.sum()),
                'n_disagree': int(n_disagree),
                'disagree_fraction': float(n_disagree / mask.sum()),
            }

    return agreement, details


def validate_type_with_morphology(
    catalog: pd.DataFrame,
    type_col: str = 'galaxy_type',
    concentration_col: str = 'concentration',
    tolerance: float = 0.5,
) -> tuple[NDArray, dict]:
    """Validate galaxy types using morphological concentration.

    Parameters
    ----------
    catalog : pd.DataFrame
        Must contain concentration and galaxy_type
    type_col : str
        Column with galaxy type
    concentration_col : str
        Column with concentration index
    tolerance : float
        Extra tolerance on concentration ranges

    Returns
    -------
    agreement_mask : NDArray[bool]
        True where morphology agrees with type
    disagreement_details : dict
        Details of disagreements by type
    """
    n = len(catalog)
    agreement = np.ones(n, dtype=bool)
    details = {}

    if concentration_col not in catalog.columns:
        return agreement, {'error': 'No concentration column available'}

    conc = catalog[concentration_col].values

    for gtype, (c_lo, c_hi) in EXPECTED_CONCENTRATION.items():
        mask = catalog[type_col] == gtype
        if mask.sum() == 0:
            continue

        # Check if concentration is in expected range
        type_agreement = (conc[mask] >= c_lo - tolerance) & (conc[mask] <= c_hi + tolerance)

        agreement[mask] = type_agreement

        n_disagree = (~type_agreement).sum()
        if n_disagree > 0:
            details[gtype] = {
                'n_total': int(mask.sum()),
                'n_disagree': int(n_disagree),
                'disagree_fraction': float(n_disagree / mask.sum()),
                'mean_concentration': float(conc[mask].mean()),
                'expected_range': (c_lo, c_hi),
            }

    return agreement, details


def compute_type_confidence(
    catalog: pd.DataFrame,
    chi2_col: str = 'chi_sq_min',
    odds_col: str = 'photo_z_odds',
    ambiguity_col: str = 'template_ambiguity',
    delta_chi2_col: str = 'delta_chi2_templates',
) -> NDArray:
    """Compute confidence score for galaxy type classification.

    Combines multiple quality metrics into a single confidence score (0-1).

    Parameters
    ----------
    catalog : pd.DataFrame
        Classification results

    Returns
    -------
    confidence : NDArray[float]
        Type confidence scores (0-1)
    """
    n = len(catalog)
    confidence = np.ones(n, dtype=np.float64)

    # Factor 1: Chi-squared quality (reduced chi2 < 5 is good)
    if chi2_col in catalog.columns:
        reduced_chi2 = catalog[chi2_col].values / 3  # DOF = 4-1 = 3
        chi2_factor = np.clip(1.0 - (reduced_chi2 - 1) / 10, 0.0, 1.0)
        confidence *= chi2_factor

    # Factor 2: ODDS (PDF concentration)
    if odds_col in catalog.columns:
        odds = catalog[odds_col].values
        odds_factor = np.clip(odds, 0.0, 1.0)
        confidence *= odds_factor

    # Factor 3: Template ambiguity (low is better)
    if ambiguity_col in catalog.columns:
        ambiguity = catalog[ambiguity_col].values
        ambiguity_factor = np.clip(1.0 - ambiguity, 0.0, 1.0)
        confidence *= ambiguity_factor

    # Factor 4: Delta chi2 to second-best template (high is better)
    if delta_chi2_col in catalog.columns:
        delta_chi2 = catalog[delta_chi2_col].values
        # delta_chi2 > 4 gives confidence, < 2 is ambiguous
        delta_factor = np.clip((delta_chi2 - 2) / 6, 0.0, 1.0)
        confidence *= (0.5 + 0.5 * delta_factor)  # Weight less heavily

    return confidence


@dataclass
class QualityReport:
    """Comprehensive quality control report for galaxy classification.

    Aggregates all validation results into a single report object.
    """

    n_sources: int = 0
    n_quality_filtered: int = 0

    # Flag distributions
    chi2_flag_dist: dict = field(default_factory=dict)
    odds_flag_dist: dict = field(default_factory=dict)
    bimodal_fraction: float = 0.0
    mean_template_ambiguity: float = 0.0

    # External validation
    photoz_validation: ValidationReport | None = None
    specz_validation: ValidationReport | None = None

    # Bias correction
    bias_correction: dict = field(default_factory=dict)
    bias_correction_function: Callable | None = None

    # Template analysis
    template_confusion: TemplateConfusionReport | None = None

    # Morphology validation
    zoobot_validation: ZoobotValidationReport | None = None
    concentration_validation: dict | None = None

    # Type validation
    type_validation: TypeValidationReport | None = None

    # Summary
    overall_quality_score: float = 0.0
    recommendations: list = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "=" * 70,
            "GALAXY CLASSIFICATION QUALITY CONTROL REPORT",
            "=" * 70,
            "",
            f"Total sources: {self.n_sources}",
            f"Quality-filtered: {self.n_quality_filtered} "
            f"({100*self.n_quality_filtered/max(1,self.n_sources):.1f}%)",
            "",
            "--- Quality Flag Distribution ---",
            f"Chi2 flags: good={self.chi2_flag_dist.get(0,0)}, "
            f"marginal={self.chi2_flag_dist.get(1,0)}, "
            f"poor={self.chi2_flag_dist.get(2,0)}",
            f"ODDS flags: excellent={self.odds_flag_dist.get(0,0)}, "
            f"good={self.odds_flag_dist.get(1,0)}, "
            f"poor={self.odds_flag_dist.get(2,0)}",
            f"Bimodal PDF fraction: {100*self.bimodal_fraction:.1f}%",
            f"Mean template ambiguity: {self.mean_template_ambiguity:.3f}",
            "",
        ]

        if self.photoz_validation:
            lines.extend([
                "--- Photo-z External Validation ---",
                f"Matched: {self.photoz_validation.n_matched} sources",
                f"NMAD: {self.photoz_validation.nmad:.4f}",
                f"Bias: {self.photoz_validation.bias:+.4f}",
                f"Outlier fraction: {100*self.photoz_validation.outlier_fraction:.1f}%",
                "",
            ])

        if self.specz_validation:
            lines.extend([
                "--- Spectroscopic Validation (Ground Truth) ---",
                f"Matched: {self.specz_validation.n_matched} sources",
                f"NMAD: {self.specz_validation.nmad:.4f}",
                f"Bias: {self.specz_validation.bias:+.4f}",
                f"Outlier fraction: {100*self.specz_validation.outlier_fraction:.1f}%",
                f"Catastrophic: {100*self.specz_validation.catastrophic_fraction:.1f}%",
                "",
            ])

        if self.template_confusion:
            lines.extend([
                "--- Template Confusion ---",
                "Top confusions:",
            ])
            for t1, t2, freq in self.template_confusion.degeneracy_pairs[:3]:
                lines.append(f"  {t1} -> {t2}: {100*freq:.1f}%")
            lines.append("")

        if self.zoobot_validation and self.zoobot_validation.n_compared > 0:
            lines.extend([
                "--- Morphology Cross-Validation ---",
                f"Sources compared: {self.zoobot_validation.n_compared}",
                f"SED-Morphology agreement: {100*self.zoobot_validation.agreement_fraction:.1f}%",
                f"Elliptical agreement: {100*self.zoobot_validation.elliptical_agreement:.1f}%",
                f"Spiral agreement: {100*self.zoobot_validation.spiral_agreement:.1f}%",
                "",
            ])

        if self.type_validation and self.type_validation.n_validated > 0:
            lines.extend([
                "--- Type Classification Validation ---",
                f"Sources validated: {self.type_validation.n_validated}",
                f"Color-Type agreement: {100*self.type_validation.color_agreement:.1f}%",
                f"Morphology-Type agreement: {100*self.type_validation.morphology_agreement:.1f}%",
                f"Mean type confidence: {self.type_validation.mean_type_confidence:.2f}",
                f"Suspicious classifications: {self.type_validation.n_suspicious} "
                f"({100*self.type_validation.suspicious_fraction:.1f}%)",
                "",
            ])

        if self.recommendations:
            lines.extend([
                "--- Recommendations ---",
                *[f"  - {r}" for r in self.recommendations],
                "",
            ])

        lines.extend([
            f"Overall Quality Score: {self.overall_quality_score:.2f}/1.00",
            "=" * 70,
        ])

        return "\n".join(lines)


def compute_template_confusion_matrix(
    catalog: pd.DataFrame,
    primary_col: str = 'galaxy_type',
    secondary_col: str = 'second_best_template',
    delta_chi2_col: str = 'delta_chi2_templates',
    ambiguity_threshold: float = 0.5,
) -> TemplateConfusionReport:
    """Compute confusion matrix between primary and secondary template fits.

    This identifies which template types are frequently confused,
    helping diagnose systematic issues in the template library.

    Parameters
    ----------
    catalog : pd.DataFrame
        Must contain primary and secondary template columns
    primary_col : str
        Column with best-fit template type
    secondary_col : str
        Column with second-best template type
    delta_chi2_col : str
        Column with chi2 difference
    ambiguity_threshold : float
        Only count pairs where template_ambiguity > threshold

    Returns
    -------
    TemplateConfusionReport
        Confusion analysis results
    """
    template_names = list(GALAXY_TYPES)
    n_templates = len(template_names)

    # Build confusion matrix
    confusion = np.zeros((n_templates, n_templates), dtype=np.int32)

    # Filter to ambiguous cases
    if 'template_ambiguity' in catalog.columns:
        ambiguous = catalog[catalog['template_ambiguity'] > ambiguity_threshold]
    else:
        ambiguous = catalog

    for _, row in ambiguous.iterrows():
        primary = row.get(primary_col, '')
        secondary = row.get(secondary_col, '')
        if primary in template_names and secondary in template_names:
            i = template_names.index(primary)
            j = template_names.index(secondary)
            confusion[i, j] += 1

    # Find most confused pairs
    pairs = []
    for i in range(n_templates):
        for j in range(n_templates):
            if i != j and confusion[i, j] > 0:
                total = max(1, confusion[i, :].sum())
                freq = confusion[i, j] / total
                pairs.append((template_names[i], template_names[j], freq))

    pairs.sort(key=lambda x: -x[2])

    # Mean delta chi2 by primary type
    mean_delta = {}
    if delta_chi2_col in catalog.columns:
        for tname in template_names:
            subset = catalog[catalog[primary_col] == tname]
            if len(subset) > 0:
                mean_delta[tname] = float(subset[delta_chi2_col].mean())

    return TemplateConfusionReport(
        confusion_matrix=confusion,
        template_names=template_names,
        degeneracy_pairs=pairs[:10],
        mean_delta_chi2=mean_delta,
    )


class QualityControlPipeline:
    """Unified quality control pipeline for galaxy classification.

    Orchestrates all validation steps and generates comprehensive reports.

    Parameters
    ----------
    catalog : pd.DataFrame
        Galaxy catalog with classification results
    output_dir : str or Path
        Directory for output files and plots
    reference_catalog : pd.DataFrame, optional
        External reference catalog for validation
    image_data : np.ndarray, optional
        Image data for Zoobot morphology validation

    Examples
    --------
    >>> qc = QualityControlPipeline(catalog, output_dir='./output/qc')
    >>> report = qc.run_full_validation()
    >>> print(report)
    >>> qc.save_report()
    """

    def __init__(
        self,
        catalog: pd.DataFrame,
        output_dir: str | Path = './output/qc',
        reference_catalog: pd.DataFrame | None = None,
        image_data: np.ndarray | None = None,
    ):
        self.catalog = catalog.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reference_catalog = reference_catalog
        self.image_data = image_data
        self.report = QualityReport()

    def compute_quality_flags(self) -> pd.DataFrame:
        """Compute quality flags if not already present.

        Adds columns: chi2_flag, odds_flag
        """
        if 'chi2_flag' not in self.catalog.columns:
            if 'chi_sq_min' in self.catalog.columns:
                self.catalog['chi2_flag'] = compute_chi2_flag(
                    self.catalog['chi_sq_min'].values
                )
            else:
                self.catalog['chi2_flag'] = 0

        if 'odds_flag' not in self.catalog.columns:
            if 'photo_z_odds' in self.catalog.columns:
                self.catalog['odds_flag'] = compute_odds_flag(
                    self.catalog['photo_z_odds'].values
                )
            elif 'odds' in self.catalog.columns:
                self.catalog['odds_flag'] = compute_odds_flag(
                    self.catalog['odds'].values
                )
            else:
                self.catalog['odds_flag'] = 0

        return self.catalog

    def analyze_flag_distribution(self) -> None:
        """Analyze distribution of quality flags."""
        self.report.n_sources = len(self.catalog)

        if 'chi2_flag' in self.catalog.columns:
            self.report.chi2_flag_dist = (
                self.catalog['chi2_flag'].value_counts().to_dict()
            )

        if 'odds_flag' in self.catalog.columns:
            self.report.odds_flag_dist = (
                self.catalog['odds_flag'].value_counts().to_dict()
            )

        if 'bimodal_flag' in self.catalog.columns:
            self.report.bimodal_fraction = float(self.catalog['bimodal_flag'].mean())

        if 'template_ambiguity' in self.catalog.columns:
            self.report.mean_template_ambiguity = float(
                self.catalog['template_ambiguity'].mean()
            )

        # Count quality-filtered sources
        good_chi2 = self.catalog.get('chi2_flag', pd.Series([0]*len(self.catalog))) <= 1
        good_odds = self.catalog.get('odds_flag', pd.Series([0]*len(self.catalog))) <= 1
        self.report.n_quality_filtered = int((good_chi2 & good_odds).sum())

    def validate_against_external(
        self,
        reference_catalog: pd.DataFrame | None = None,
        z_col_ours: str = 'redshift',
        z_col_ref: str = 'z_phot',
        max_separation_arcsec: float = 1.0,
    ) -> ValidationReport | None:
        """Validate photo-z against external catalog.

        Parameters
        ----------
        reference_catalog : pd.DataFrame, optional
            External catalog (uses self.reference_catalog if not provided)
        z_col_ours : str
            Redshift column in our catalog
        z_col_ref : str
            Redshift column in reference catalog
        max_separation_arcsec : float
            Maximum match separation

        Returns
        -------
        ValidationReport or None
            Validation results
        """
        if not HAS_EXTERNAL_CATALOGS:
            warnings.warn("External catalogs module not available", stacklevel=2)
            return None

        if reference_catalog is None:
            reference_catalog = self.reference_catalog

        if reference_catalog is None or reference_catalog.empty:
            warnings.warn("No reference catalog provided", stacklevel=2)
            return None

        # Check for required columns
        if 'ra' not in self.catalog.columns or 'dec' not in self.catalog.columns:
            warnings.warn("Catalog missing RA/Dec columns for cross-matching", stacklevel=2)
            return None

        try:
            # Cross-match catalogs
            matched = cross_match_catalogs(
                self.catalog,
                reference_catalog,
                max_sep_arcsec=max_separation_arcsec,
            )

            if len(matched) == 0:
                warnings.warn("No matches found in cross-matching", stacklevel=2)
                return None

            # Get matched redshifts
            z_ours = matched[z_col_ours].values
            z_ref = matched[z_col_ref].values

            # Filter valid values
            valid = (z_ours > 0) & (z_ref > 0) & np.isfinite(z_ours) & np.isfinite(z_ref)
            z_ours = z_ours[valid]
            z_ref = z_ref[valid]

            if len(z_ours) < 5:
                return None

            # Compute metrics
            dz = (z_ours - z_ref) / (1 + z_ref)
            nmad = 1.48 * np.median(np.abs(dz - np.median(dz)))
            bias = np.median(dz)
            outlier_frac = np.mean(np.abs(dz) > 0.15)
            catastrophic_frac = np.mean(np.abs(dz) > 0.5)

            report = ValidationReport(
                n_matched=len(z_ours),
                n_total_ours=len(self.catalog),
                n_total_reference=len(reference_catalog),
                nmad=nmad,
                bias=bias,
                outlier_fraction=outlier_frac,
                catastrophic_fraction=catastrophic_frac,
                z_ours=z_ours,
                z_reference=z_ref,
                separation_arcsec=matched['separation_arcsec'].values[valid]
                if 'separation_arcsec' in matched.columns
                else np.zeros(len(z_ours)),
            )

            self.report.photoz_validation = report
            return report

        except Exception as e:
            warnings.warn(f"Validation failed: {e}", stacklevel=2)
            return None

    def compute_bias_correction(
        self,
        z_bins: np.ndarray | None = None,
        n_bins: int = 5,
    ) -> dict[str, float]:
        """Compute redshift-dependent bias correction.

        Uses validation against reference redshifts to estimate
        systematic biases that can be corrected.

        Parameters
        ----------
        z_bins : np.ndarray, optional
            Redshift bin edges
        n_bins : int
            Number of bins if z_bins not provided

        Returns
        -------
        dict
            Bias correction by redshift bin center
        """
        validation = self.report.specz_validation or self.report.photoz_validation

        if validation is None:
            warnings.warn("No validation available for bias correction", stacklevel=2)
            return {}

        z_ours = validation.z_ours
        z_ref = validation.z_reference

        if z_bins is None:
            z_bins = np.percentile(z_ref, np.linspace(0, 100, n_bins + 1))

        corrections = {}
        for i in range(len(z_bins) - 1):
            mask = (z_ref >= z_bins[i]) & (z_ref < z_bins[i + 1])
            if mask.sum() >= 3:
                dz = (z_ours[mask] - z_ref[mask]) / (1 + z_ref[mask])
                bias = np.median(dz)
                z_center = (z_bins[i] + z_bins[i + 1]) / 2
                corrections[f"{z_center:.2f}"] = float(bias)

        self.report.bias_correction = corrections
        return corrections

    def apply_bias_correction(
        self,
        z_col: str = 'redshift',
    ) -> pd.DataFrame:
        """Apply computed bias correction to catalog redshifts.

        Returns
        -------
        pd.DataFrame
            Catalog with corrected redshifts in 'redshift_corrected' column
        """
        if not self.report.bias_correction:
            self.catalog['redshift_corrected'] = self.catalog[z_col]
            return self.catalog

        # Interpolate bias correction
        z_centers = np.array([float(k) for k in self.report.bias_correction])
        biases = np.array(list(self.report.bias_correction.values()))

        z_orig = self.catalog[z_col].values
        bias_interp = np.interp(
            z_orig, z_centers, biases,
            left=biases[0] if len(biases) > 0 else 0,
            right=biases[-1] if len(biases) > 0 else 0
        )

        # Correct: z_true ≈ z_phot - bias*(1+z_phot)
        self.catalog['redshift_corrected'] = z_orig - bias_interp * (1 + z_orig)

        return self.catalog

    def validate_morphology_concentration(self) -> dict | None:
        """Validate SED types using concentration index as morphology proxy.

        Elliptical galaxies should be more concentrated (higher C),
        while spiral galaxies should be more extended (lower C).

        Returns
        -------
        dict or None
            Validation results
        """
        if 'concentration' not in self.catalog.columns:
            return None

        if 'galaxy_type' not in self.catalog.columns:
            return None

        # Define concentration thresholds
        # C > 2.8 typically indicates elliptical/compact
        # C < 2.5 typically indicates disk/extended

        results = {'n_sources': len(self.catalog), 'by_type': {}}

        for gtype in GALAXY_TYPES:
            subset = self.catalog[self.catalog['galaxy_type'] == gtype]
            if len(subset) > 0:
                mean_c = subset['concentration'].mean()
                std_c = subset['concentration'].std()
                results['by_type'][gtype] = {
                    'n': len(subset),
                    'mean_concentration': float(mean_c),
                    'std_concentration': float(std_c),
                }

        # Check agreement
        elliptical_types = ['Ell', 'S0']
        spiral_types = ['Sa', 'Sb']

        ell_mask = self.catalog['galaxy_type'].isin(elliptical_types)
        spi_mask = self.catalog['galaxy_type'].isin(spiral_types)

        if ell_mask.sum() > 0:
            ell_compact = (self.catalog.loc[ell_mask, 'concentration'] > 2.8).mean()
            results['elliptical_compact_fraction'] = float(ell_compact)
        else:
            results['elliptical_compact_fraction'] = np.nan

        if spi_mask.sum() > 0:
            spi_extended = (self.catalog.loc[spi_mask, 'concentration'] < 2.8).mean()
            results['spiral_extended_fraction'] = float(spi_extended)
        else:
            results['spiral_extended_fraction'] = np.nan

        self.report.concentration_validation = results

        # Create approximate ZoobotValidationReport for consistency
        if not np.isnan(results.get('elliptical_compact_fraction', np.nan)):
            e_agree = results['elliptical_compact_fraction']
            s_agree = results.get('spiral_extended_fraction', 0)
            if np.isnan(s_agree):
                s_agree = 0

            if HAS_ZOOBOT and ZoobotValidationReport is not None:
                self.report.zoobot_validation = ZoobotValidationReport(
                    n_compared=int((ell_mask | spi_mask).sum()),
                    agreement_fraction=(e_agree + s_agree) / 2,
                    confusion=results['by_type'],
                    elliptical_agreement=e_agree,
                    spiral_agreement=s_agree,
                    starburst_agreement=0.0,
                )

        return results

    def compute_template_confusion(self) -> TemplateConfusionReport | None:
        """Analyze template fitting confusion."""
        if 'second_best_template' not in self.catalog.columns:
            return None

        if 'galaxy_type' not in self.catalog.columns:
            return None

        report = compute_template_confusion_matrix(
            self.catalog,
            primary_col='galaxy_type',
            secondary_col='second_best_template',
        )

        self.report.template_confusion = report
        return report

    def validate_type_classification(self) -> TypeValidationReport | None:
        """Validate galaxy type classifications using colors and morphology.

        Cross-validates SED-based types with:
        1. Observed colors (U-B, B-V, V-I)
        2. Morphological concentration index
        3. Computes type confidence scores

        Returns
        -------
        TypeValidationReport or None
        """
        if 'galaxy_type' not in self.catalog.columns:
            return None

        report = TypeValidationReport()
        report.n_validated = len(self.catalog)

        # Validate with colors
        color_agreement, color_details = validate_type_with_colors(
            self.catalog,
            type_col='galaxy_type',
        )
        report.color_agreement = float(color_agreement.mean())
        report.color_disagreements = color_details

        # Validate with morphology
        morph_agreement, morph_details = validate_type_with_morphology(
            self.catalog,
            type_col='galaxy_type',
        )
        report.morphology_agreement = float(morph_agreement.mean())
        report.morphology_disagreements = morph_details

        # Compute type confidence
        confidence = compute_type_confidence(self.catalog)
        self.catalog['type_confidence'] = confidence
        report.mean_type_confidence = float(confidence.mean())
        report.low_confidence_fraction = float((confidence < 0.5).mean())

        # Flag suspicious classifications (color AND morphology disagree)
        suspicious = (~color_agreement) & (~morph_agreement)
        report.n_suspicious = int(suspicious.sum())
        report.suspicious_fraction = float(suspicious.mean())

        # Add flag to catalog
        self.catalog['type_suspicious'] = suspicious

        self.report.type_validation = report
        return report

    def compute_overall_score(self) -> float:
        """Compute aggregate quality score (0-1).

        Weighted combination of:
        - Flag quality (30%): fraction with good chi2 and ODDS
        - External validation (20%): NMAD and outlier fraction
        - Morphology agreement (15%): SED-morphology consistency
        - Template clarity (10%): low template ambiguity
        - Type validation (25%): color and morphology agreement with type
        """
        scores = []
        weights = []

        # Flag quality (30%)
        n_good = (
            self.report.chi2_flag_dist.get(0, 0) +
            self.report.chi2_flag_dist.get(1, 0)
        )
        n_total = max(1, sum(self.report.chi2_flag_dist.values()))
        flag_score = n_good / n_total
        scores.append(flag_score)
        weights.append(0.30)

        # External validation (20%)
        if self.report.specz_validation:
            val = self.report.specz_validation
            nmad_score = max(0, 1 - val.nmad / 0.1)
            outlier_score = max(0, 1 - val.outlier_fraction / 0.2)
            ext_score = (nmad_score + outlier_score) / 2
            scores.append(ext_score)
            weights.append(0.20)
        elif self.report.photoz_validation:
            val = self.report.photoz_validation
            nmad_score = max(0, 1 - val.nmad / 0.15)
            scores.append(nmad_score)
            weights.append(0.10)

        # Morphology agreement (15%)
        if self.report.zoobot_validation and self.report.zoobot_validation.n_compared > 0:
            morph_score = self.report.zoobot_validation.agreement_fraction
            scores.append(morph_score)
            weights.append(0.15)

        # Template clarity (10%)
        if self.report.mean_template_ambiguity > 0:
            template_score = 1 - self.report.mean_template_ambiguity
            scores.append(template_score)
            weights.append(0.10)

        # Type validation (25%)
        if self.report.type_validation and self.report.type_validation.n_validated > 0:
            tv = self.report.type_validation
            # Combine color agreement, morphology agreement, and confidence
            type_score = (
                tv.color_agreement * 0.4 +
                tv.morphology_agreement * 0.3 +
                tv.mean_type_confidence * 0.3
            )
            scores.append(type_score)
            weights.append(0.25)

        # Weighted average
        if weights:
            total_weight = sum(weights)
            self.report.overall_quality_score = sum(
                s * w for s, w in zip(scores, weights, strict=False)
            ) / total_weight

        return self.report.overall_quality_score

    def generate_recommendations(self) -> list[str]:
        """Generate quality improvement recommendations."""
        recs = []

        # Chi2 quality
        poor_chi2 = self.report.chi2_flag_dist.get(2, 0)
        n_total = max(1, self.report.n_sources)
        poor_chi2_frac = poor_chi2 / n_total
        if poor_chi2_frac > 0.1:
            recs.append(
                f"High poor chi2 fraction ({100*poor_chi2_frac:.0f}%): "
                "Consider template library expansion or photometry calibration review"
            )

        # ODDS quality
        poor_odds = self.report.odds_flag_dist.get(2, 0)
        poor_odds_frac = poor_odds / n_total
        if poor_odds_frac > 0.2:
            recs.append(
                f"High poor ODDS fraction ({100*poor_odds_frac:.0f}%): "
                "Many sources have degenerate photo-z solutions"
            )

        # Bimodal solutions
        if self.report.bimodal_fraction > 0.3:
            recs.append(
                f"High bimodal fraction ({100*self.report.bimodal_fraction:.0f}%): "
                "Consider using broader wavelength coverage or priors"
            )

        # External validation
        if self.report.specz_validation or self.report.photoz_validation:
            val = self.report.specz_validation or self.report.photoz_validation
            if val.outlier_fraction > 0.15:
                recs.append(
                    f"High outlier fraction ({100*val.outlier_fraction:.0f}%): "
                    "Review template set for missing galaxy types"
                )
            if abs(val.bias) > 0.02:
                recs.append(
                    f"Systematic bias detected ({val.bias:+.3f}): "
                    "Apply bias correction"
                )

        # Template confusion
        if self.report.template_confusion:
            for t1, t2, freq in self.report.template_confusion.degeneracy_pairs[:3]:
                if freq > 0.3:
                    recs.append(
                        f"Template confusion: {t1} vs {t2} ({100*freq:.0f}% overlap)"
                    )

        # Morphology
        if self.report.zoobot_validation and self.report.zoobot_validation.agreement_fraction < 0.6:
            recs.append(
                "Low SED-morphology agreement: "
                "Review elliptical/spiral classification criteria"
            )

        # Type validation
        if self.report.type_validation:
            tv = self.report.type_validation

            if tv.color_agreement < 0.7:
                recs.append(
                    f"Low color-type agreement ({100*tv.color_agreement:.0f}%): "
                    "Colors don't match expected for classified types - review template selection"
                )

            if tv.morphology_agreement < 0.7:
                recs.append(
                    f"Low morphology-type agreement ({100*tv.morphology_agreement:.0f}%): "
                    "Concentration doesn't match expected for types - possible misclassification"
                )

            if tv.low_confidence_fraction > 0.3:
                recs.append(
                    f"High low-confidence fraction ({100*tv.low_confidence_fraction:.0f}%): "
                    "Many classifications have low confidence - consider quality filtering"
                )

            if tv.suspicious_fraction > 0.1:
                recs.append(
                    f"High suspicious classification rate ({100*tv.suspicious_fraction:.0f}%): "
                    "These sources have colors AND morphology inconsistent with type"
                )

            # Check for specific type problems
            for gtype, details in tv.color_disagreements.items():
                if details.get('disagree_fraction', 0) > 0.5:
                    recs.append(
                        f"Type '{gtype}' has {100*details['disagree_fraction']:.0f}% "
                        "color disagreement - review template or classification threshold"
                    )

        self.report.recommendations = recs
        return recs

    def run_full_validation(
        self,
        reference_catalog: pd.DataFrame | None = None,
        generate_plots: bool = True,
    ) -> QualityReport:
        """Run complete quality control pipeline.

        Parameters
        ----------
        reference_catalog : pd.DataFrame, optional
            External reference catalog
        generate_plots : bool
            Whether to generate validation plots

        Returns
        -------
        QualityReport
            Complete quality control report
        """
        print("=" * 60)
        print("RUNNING QUALITY CONTROL PIPELINE")
        print("=" * 60)

        # Step 1: Compute quality flags
        print("\n[1/8] Computing quality flags...")
        self.compute_quality_flags()
        self.analyze_flag_distribution()
        print(f"  Sources: {self.report.n_sources}, "
              f"Quality-filtered: {self.report.n_quality_filtered}")

        # Step 2: External photo-z validation
        print("\n[2/8] Validating against external photo-z catalog...")
        self.validate_against_external(reference_catalog)
        if self.report.photoz_validation:
            print(f"  Matched: {self.report.photoz_validation.n_matched}, "
                  f"NMAD: {self.report.photoz_validation.nmad:.4f}")
        else:
            print("  Skipped (no reference catalog)")

        # Step 3: Bias correction
        print("\n[3/8] Computing bias correction...")
        corrections = self.compute_bias_correction()
        if corrections:
            print(f"  Bias corrections by z: {corrections}")
            self.apply_bias_correction()
        else:
            print("  Skipped (no validation data)")

        # Step 4: Template confusion analysis
        print("\n[4/8] Analyzing template confusion...")
        self.compute_template_confusion()
        if self.report.template_confusion:
            top_pairs = self.report.template_confusion.degeneracy_pairs[:3]
            if top_pairs:
                print(f"  Top confusions: {top_pairs}")
            else:
                print("  No significant template confusion detected")
        else:
            print("  Skipped (no secondary template data)")

        # Step 5: Morphology cross-validation
        print("\n[5/8] Cross-validating with morphology...")
        self.validate_morphology_concentration()
        if self.report.zoobot_validation:
            print(f"  Agreement: {100*self.report.zoobot_validation.agreement_fraction:.1f}%")
        elif self.report.concentration_validation:
            print("  Using concentration index as morphology proxy")
        else:
            print("  Skipped (no morphology data)")

        # Step 6: Type classification validation
        print("\n[6/8] Validating galaxy type classifications...")
        self.validate_type_classification()
        if self.report.type_validation:
            tv = self.report.type_validation
            print(f"  Color-type agreement: {100*tv.color_agreement:.1f}%")
            print(f"  Morphology-type agreement: {100*tv.morphology_agreement:.1f}%")
            print(f"  Mean type confidence: {tv.mean_type_confidence:.2f}")
            if tv.n_suspicious > 0:
                print(f"  Suspicious classifications: {tv.n_suspicious} ({100*tv.suspicious_fraction:.1f}%)")
        else:
            print("  Skipped (no galaxy type data)")

        # Step 7: Generate summary
        print("\n[7/8] Computing overall quality score...")
        self.compute_overall_score()
        print(f"  Overall score: {self.report.overall_quality_score:.2f}")

        # Step 8: Generate recommendations
        print("\n[8/8] Generating recommendations...")
        self.generate_recommendations()
        print(f"  Found {len(self.report.recommendations)} recommendation(s)")

        # Generate plots if requested
        if generate_plots:
            self._generate_validation_plots()

        print("\n" + str(self.report))

        return self.report

    def _generate_validation_plots(self) -> None:
        """Generate all validation plots."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not available for plotting", stacklevel=2)
            return

        plot_dir = self.output_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)

        # Quality flag histogram
        if self.report.chi2_flag_dist or self.report.odds_flag_dist:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Chi2 flags
            ax = axes[0]
            labels = ['Good (0)', 'Marginal (1)', 'Poor (2)']
            values = [self.report.chi2_flag_dist.get(i, 0) for i in range(3)]
            colors = ['#2ecc71', '#f39c12', '#e74c3c']
            ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_title('Chi-squared Quality Flags')
            ax.set_ylabel('Count')

            # ODDS flags
            ax = axes[1]
            labels = ['Excellent (0)', 'Good (1)', 'Poor (2)']
            values = [self.report.odds_flag_dist.get(i, 0) for i in range(3)]
            ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_title('ODDS Quality Flags')
            ax.set_ylabel('Count')

            plt.tight_layout()
            fig.savefig(plot_dir / 'quality_flags.pdf', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {plot_dir / 'quality_flags.pdf'}")

        # Photo-z validation plot
        if self.report.photoz_validation or self.report.specz_validation:
            val = self.report.specz_validation or self.report.photoz_validation
            if val and len(val.z_ours) > 0:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Photo-z vs reference
                ax = axes[0]
                ax.scatter(val.z_reference, val.z_ours, s=10, alpha=0.5, c='blue')
                z_range = [0, max(val.z_reference.max(), val.z_ours.max()) * 1.1]
                ax.plot(z_range, z_range, 'k--', lw=2, label='1:1')
                ax.set_xlabel('Reference z')
                ax.set_ylabel('Our z')
                ax.set_title('Photo-z Comparison')
                ax.legend()
                ax.set_xlim(z_range)
                ax.set_ylim(z_range)

                # Residual histogram
                ax = axes[1]
                dz = (val.z_ours - val.z_reference) / (1 + val.z_reference)
                ax.hist(dz, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
                ax.axvline(0, color='black', linestyle='--', lw=2)
                ax.axvline(val.bias, color='red', linestyle='-', lw=2,
                           label=f'Bias={val.bias:.3f}')
                ax.set_xlabel(r'$\Delta z / (1+z)$')
                ax.set_ylabel('Count')
                ax.set_title(f'Photo-z Residuals (NMAD={val.nmad:.3f})')
                ax.legend()

                plt.tight_layout()
                fig.savefig(plot_dir / 'photoz_validation.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved: {plot_dir / 'photoz_validation.pdf'}")

        # Template confusion matrix
        if self.report.template_confusion is not None:
            confusion = self.report.template_confusion.confusion_matrix
            names = self.report.template_confusion.template_names

            # Only plot if there's actual data
            if confusion.sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(confusion, cmap='Blues')
                ax.set_xticks(range(len(names)))
                ax.set_yticks(range(len(names)))
                ax.set_xticklabels(names, rotation=45, ha='right')
                ax.set_yticklabels(names)
                ax.set_xlabel('Second-best Template')
                ax.set_ylabel('Best Template')
                ax.set_title('Template Confusion Matrix')
                plt.colorbar(im, ax=ax, label='Count')

                plt.tight_layout()
                fig.savefig(plot_dir / 'template_confusion.pdf', dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved: {plot_dir / 'template_confusion.pdf'}")

    def save_report(
        self,
        filename: str = 'quality_report.txt',
        save_catalog: bool = True,
    ) -> None:
        """Save quality report and annotated catalog.

        Parameters
        ----------
        filename : str
            Report filename
        save_catalog : bool
            Whether to save catalog with quality flags
        """
        # Save text report
        report_path = self.output_dir / filename
        with open(report_path, 'w') as f:
            f.write(str(self.report))
        print(f"Saved report: {report_path}")

        # Save annotated catalog
        if save_catalog:
            catalog_path = self.output_dir / 'catalog_with_qc.csv'
            self.catalog.to_csv(catalog_path, index=False)
            print(f"Saved catalog: {catalog_path}")

    def get_quality_filtered_catalog(
        self,
        chi2_max_flag: int = 1,
        odds_max_flag: int = 1,
    ) -> pd.DataFrame:
        """Return catalog filtered to high-quality sources only.

        Parameters
        ----------
        chi2_max_flag : int
            Maximum chi2 flag to include (0=good only, 1=good+marginal)
        odds_max_flag : int
            Maximum odds flag to include

        Returns
        -------
        pd.DataFrame
            Filtered catalog
        """
        mask = (
            (self.catalog.get('chi2_flag', 0) <= chi2_max_flag) &
            (self.catalog.get('odds_flag', 0) <= odds_max_flag)
        )
        return self.catalog[mask].copy()
