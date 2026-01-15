"""Zoobot integration for galaxy morphology classification and validation.

Zoobot is a deep learning model trained on Galaxy Zoo volunteer classifications.
It provides morphological predictions including:
- Smooth vs featured (disk-like)
- Edge-on disk
- Spiral arm presence
- Bar presence
- Bulge dominance
- Merger indicators

This module provides:
1. Cutout extraction from FITS images
2. Zoobot prediction wrapper
3. Cross-validation against SED-based classifications

References:
- Walmsley et al. 2023, MNRAS (Zoobot)
- https://github.com/mwalmsley/zoobot

Installation:
    pip install zoobot[pytorch]
"""

import contextlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from zoobot.pytorch.predictions import predict_on_catalog
from zoobot.pytorch.training import finetune


@dataclass
class MorphologyPrediction:
    """Zoobot morphology prediction for a single galaxy."""
    source_id: int

    # Main morphology probabilities (0-1)
    smooth: float           # Probability of smooth/elliptical
    featured: float         # Probability of disk/featured
    artifact: float         # Probability of artifact/star

    # Detailed features (if featured)
    edge_on: float          # Edge-on disk probability
    has_spiral: float       # Spiral arm probability
    has_bar: float          # Bar probability
    bulge_dominant: float   # Bulge-dominated probability

    # Merger/interaction
    merger: float           # Merger probability

    # Derived classification
    morphology_class: str   # E, S0, Sa, Sb, Sc, Sd, Irr, etc.
    confidence: float       # Confidence in classification


@dataclass
class ZoobotValidationReport:
    """Results from cross-validating SED types with Zoobot morphology."""
    n_compared: int
    agreement_fraction: float

    # Confusion matrix as dict
    confusion: dict

    # Specific agreements
    elliptical_agreement: float  # SED elliptical vs Zoobot smooth
    spiral_agreement: float      # SED Sa/Sb vs Zoobot featured+spiral
    starburst_agreement: float   # SED sbt* vs Zoobot featured+blue

    def __str__(self) -> str:
        return f"""
=== Zoobot Morphology Validation ===
Sources compared: {self.n_compared}
Overall agreement: {100*self.agreement_fraction:.1f}%

By type:
  Elliptical (SED) vs Smooth (Zoobot): {100*self.elliptical_agreement:.1f}%
  Spiral (SED) vs Featured+Spiral (Zoobot): {100*self.spiral_agreement:.1f}%
  Starburst (SED) vs Featured (Zoobot): {100*self.starburst_agreement:.1f}%
"""


def extract_cutouts(
    image_data: np.ndarray,
    catalog: pd.DataFrame,
    size: int = 128,
    x_col: str = 'xcentroid',
    y_col: str = 'ycentroid',
    output_dir: Path | None = None,
    normalize: bool = True,
) -> list[tuple[int, np.ndarray, Path | None]]:
    """
    Extract square cutouts centered on each source for Zoobot.

    Parameters
    ----------
    image_data : np.ndarray
        Full image array
    catalog : pd.DataFrame
        Source catalog with centroid positions
    size : int
        Cutout size in pixels (will be size x size)
    x_col, y_col : str
        Column names for centroid positions
    output_dir : Path, optional
        If provided, save cutouts as PNG files
    normalize : bool
        Apply asinh stretch normalization for visualization

    Returns
    -------
    list of (source_id, cutout_array, path) tuples
    """
    cutouts = []

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    half_size = size // 2
    ny, nx = image_data.shape

    # Vectorized coordinate extraction and boundary calculation
    x_coords = catalog[x_col].values.astype(int)
    y_coords = catalog[y_col].values.astype(int)
    indices = catalog.index.values

    # Vectorized boundary calculation
    x_lo = np.maximum(0, x_coords - half_size)
    x_hi = np.minimum(nx, x_coords + half_size)
    y_lo = np.maximum(0, y_coords - half_size)
    y_hi = np.minimum(ny, y_coords + half_size)

    # Process cutouts (extraction still requires loop but boundary calc is vectorized)
    for i, idx in enumerate(indices):
        # Extract cutout using pre-computed boundaries
        cutout = image_data[y_lo[i]:y_hi[i], x_lo[i]:x_hi[i]].copy()

        # Pad if necessary to get exact size
        if cutout.shape != (size, size):
            padded = np.zeros((size, size), dtype=cutout.dtype)
            dy = (size - cutout.shape[0]) // 2
            dx = (size - cutout.shape[1]) // 2
            padded[dy:dy+cutout.shape[0], dx:dx+cutout.shape[1]] = cutout
            cutout = padded

        # Normalize for visualization
        if normalize:
            cutout = asinh_stretch(cutout)

        # Save as PNG if requested
        file_path = None
        if output_dir:
            file_path = output_dir / f"galaxy_{idx:05d}.png"
            save_cutout_as_png(cutout, file_path)

        cutouts.append((idx, cutout, file_path))

    return cutouts


def asinh_stretch(data: np.ndarray, scale: float = 0.1) -> np.ndarray:
    """
    Apply asinh stretch for better visualization of galaxy structure.

    Parameters
    ----------
    data : np.ndarray
        Input image data
    scale : float
        Softening parameter

    Returns
    -------
    np.ndarray
        Stretched image normalized to 0-255 uint8
    """
    # Handle NaN and negative values
    data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)

    # Subtract background (use median)
    data = data - np.median(data)

    # Asinh stretch
    stretched = np.arcsinh(data / scale)

    # Normalize to 0-255
    vmin, vmax = np.percentile(stretched, [1, 99])
    if vmax > vmin:
        stretched = (stretched - vmin) / (vmax - vmin)
    stretched = np.clip(stretched * 255, 0, 255).astype(np.uint8)

    return stretched


def save_cutout_as_png(cutout: np.ndarray, path: Path):
    """Save a cutout as a PNG image for Zoobot."""
    # Convert to RGB (Zoobot expects 3-channel images)
    rgb = np.stack([cutout, cutout, cutout], axis=-1) if cutout.ndim == 2 else cutout

    img = Image.fromarray(rgb.astype(np.uint8))
    img.save(path)


def run_zoobot_predictions(
    image_paths: list[Path],
    model_name: str = 'hf_hub:mwalmsley/zoobot-encoder-convnext_nano-greyscale',
    batch_size: int = 32,
    device: str = 'auto',
) -> pd.DataFrame:
    """
    Run Zoobot predictions on a list of galaxy images.

    Parameters
    ----------
    image_paths : list of Path
        Paths to PNG images of galaxies
    model_name : str
        HuggingFace model identifier
    batch_size : int
        Batch size for inference
    device : str
        'auto', 'cuda', or 'cpu'

    Returns
    -------
    pd.DataFrame
        Predictions with morphology probabilities
    """
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Running Zoobot on {len(image_paths)} images using {device}...")

    # Create catalog DataFrame for Zoobot
    catalog_df = pd.DataFrame({
        'file_loc': [str(p) for p in image_paths],
        'id_str': [p.stem for p in image_paths],
    })

    # Load model
    model = finetune.FinetuneableZoobotClassifier.load_from_name(model_name)
    model = model.to(device)
    model.eval()

    # Run predictions
    # Zoobot returns Galaxy Zoo decision tree question responses
    predictions = predict_on_catalog.predict(
        catalog_df,
        model,
        n_samples=1,  # Single prediction per image
        batch_size=batch_size,
    )

    return predictions


def interpret_zoobot_predictions(predictions: pd.DataFrame) -> list[MorphologyPrediction]:
    """
    Interpret raw Zoobot predictions into morphology classifications.

    Galaxy Zoo decision tree questions:
    - smooth-or-featured: smooth / featured-or-disk / artifact
    - disk-edge-on: yes / no
    - has-spiral-arms: yes / no
    - bar: strong / weak / no
    - bulge-size: dominant / large / moderate / small / none
    - merging: merger / major-disturbance / minor-disturbance / none

    Returns
    -------
    list of MorphologyPrediction
    """
    n_sources = len(predictions)

    # Helper to safely get column values with default
    def get_col(col_name: str, default: float = 0.0) -> np.ndarray:
        if col_name in predictions.columns:
            return predictions[col_name].values
        return np.full(n_sources, default)

    # Vectorized extraction of all morphology features
    smooth = get_col('smooth-or-featured_smooth', 0.5)
    featured = get_col('smooth-or-featured_featured-or-disk', 0.5)
    artifact = get_col('smooth-or-featured_artifact', 0.0)

    edge_on = get_col('disk-edge-on_yes', 0.0)
    has_spiral = get_col('has-spiral-arms_yes', 0.0)
    bar_strong = get_col('bar_strong', 0.0)
    bar_weak = get_col('bar_weak', 0.0)
    has_bar = bar_strong + bar_weak

    bulge_dom = get_col('bulge-size_dominant', 0.0)
    merger_main = get_col('merging_merger', 0.0)
    merger_major = get_col('merging_major-disturbance', 0.0)
    merger = merger_main + merger_major

    # Extract source IDs
    indices = predictions.index.values
    id_strs = predictions['id_str'].values if 'id_str' in predictions.columns else None

    # Build results list
    results = []
    for i in range(n_sources):
        # Derive classification
        morph_class, confidence = derive_morphology_class(
            smooth[i], featured[i], edge_on[i], has_spiral[i], has_bar[i], bulge_dom[i]
        )

        # Extract source ID from filename if available
        source_id = indices[i]
        if id_strs is not None:
            with contextlib.suppress(ValueError, AttributeError):
                source_id = int(str(id_strs[i]).replace('galaxy_', ''))

        results.append(MorphologyPrediction(
            source_id=source_id,
            smooth=float(smooth[i]),
            featured=float(featured[i]),
            artifact=float(artifact[i]),
            edge_on=float(edge_on[i]),
            has_spiral=float(has_spiral[i]),
            has_bar=float(has_bar[i]),
            bulge_dominant=float(bulge_dom[i]),
            merger=float(merger[i]),
            morphology_class=morph_class,
            confidence=confidence,
        ))

    return results


def derive_morphology_class(
    smooth: float,
    featured: float,
    edge_on: float,
    has_spiral: float,
    has_bar: float,
    bulge_dom: float,
) -> tuple[str, float]:
    """
    Derive Hubble-type morphology class from Zoobot probabilities.

    Returns
    -------
    (class_name, confidence)
    """
    # Threshold for classification
    if smooth > 0.6:
        if bulge_dom > 0.5:
            return 'E', smooth
        else:
            return 'E/S0', smooth * 0.8

    if featured > 0.5:
        if edge_on > 0.6:
            return 'S_edge', featured * edge_on

        if has_spiral > 0.5:
            if bulge_dom > 0.4:
                return 'Sa', featured * has_spiral * bulge_dom
            elif has_bar > 0.5:
                return 'SBb', featured * has_spiral * has_bar
            else:
                return 'Sb/Sc', featured * has_spiral
        else:
            if bulge_dom > 0.5:
                return 'S0', featured * bulge_dom
            else:
                return 'S0/Sa', featured * 0.7

    # Low confidence - could be irregular or artifact
    return 'Irr/Unknown', max(smooth, featured) * 0.5


def sed_to_morphology_class(sed_type: str) -> str:
    """
    Map SED galaxy type to morphology class for comparison.

    Parameters
    ----------
    sed_type : str
        SED classification (elliptical, S0, Sa, Sb, sbt1-6)

    Returns
    -------
    str
        Simplified morphology class
    """
    sed_type = sed_type.lower()

    if sed_type == 'elliptical':
        return 'E'
    elif sed_type == 's0':
        return 'S0'
    elif sed_type in ['sa', 'sb']:
        return 'Spiral'
    elif sed_type.startswith('sbt'):
        return 'Starburst'
    else:
        return 'Unknown'


def validate_morphology_with_zoobot(
    sed_catalog: pd.DataFrame,
    zoobot_predictions: list[MorphologyPrediction],
    type_col: str = 'galaxy_type',
) -> ZoobotValidationReport:
    """
    Cross-validate SED classifications against Zoobot morphology.

    Parameters
    ----------
    sed_catalog : pd.DataFrame
        Our catalog with SED-based classifications
    zoobot_predictions : list
        Zoobot predictions from interpret_zoobot_predictions()
    type_col : str
        Column name for galaxy type in sed_catalog

    Returns
    -------
    ZoobotValidationReport
    """
    # Convert Zoobot predictions to DataFrame for vectorized merge
    zoobot_df = pd.DataFrame([
        {
            'source_id': p.source_id,
            'zoobot_class': p.morphology_class,
            'zoobot_smooth': p.smooth,
            'zoobot_featured': p.featured,
            'zoobot_spiral': p.has_spiral,
            'zoobot_confidence': p.confidence,
        }
        for p in zoobot_predictions
    ]).set_index('source_id')

    # Vectorized merge instead of iterrows + lookup
    sed_subset = sed_catalog[[type_col]].copy()
    sed_subset['sed_type'] = sed_catalog[type_col]

    # Vectorized SED class mapping
    sed_subset['sed_class'] = sed_subset['sed_type'].apply(sed_to_morphology_class)

    # Merge using pandas (much faster than iterrows + lookup)
    comp_df = sed_subset.join(zoobot_df, how='inner')

    if len(comp_df) == 0:
        return ZoobotValidationReport(
            n_compared=0,
            agreement_fraction=0.0,
            confusion={},
            elliptical_agreement=0.0,
            spiral_agreement=0.0,
            starburst_agreement=0.0,
        )

    # Add source_id column from index
    comp_df = comp_df.reset_index()
    comp_df = comp_df.rename(columns={'index': 'source_id'})

    # Calculate agreements
    # Elliptical: SED elliptical should be Zoobot smooth
    elliptical_mask = comp_df['sed_class'] == 'E'
    if elliptical_mask.any():
        elliptical_agree = (comp_df.loc[elliptical_mask, 'zoobot_smooth'] > 0.5).mean()
    else:
        elliptical_agree = np.nan

    # Spiral: SED Sa/Sb should be Zoobot featured with spiral
    spiral_mask = comp_df['sed_class'] == 'Spiral'
    if spiral_mask.any():
        spiral_agree = (
            (comp_df.loc[spiral_mask, 'zoobot_featured'] > 0.5) &
            (comp_df.loc[spiral_mask, 'zoobot_spiral'] > 0.3)
        ).mean()
    else:
        spiral_agree = np.nan

    # Starburst: SED sbt* should be Zoobot featured (usually irregular/blue)
    starburst_mask = comp_df['sed_class'] == 'Starburst'
    if starburst_mask.any():
        starburst_agree = (comp_df.loc[starburst_mask, 'zoobot_featured'] > 0.4).mean()
    else:
        starburst_agree = np.nan

    # Build confusion matrix
    confusion = {}
    for sed_class in comp_df['sed_class'].unique():
        mask = comp_df['sed_class'] == sed_class
        zoobot_classes = comp_df.loc[mask, 'zoobot_class'].value_counts()
        confusion[sed_class] = zoobot_classes.to_dict()

    # Overall agreement (simplified - E matches smooth, featured matches disk)
    def simple_match(row):
        if row['sed_class'] == 'E' and row['zoobot_smooth'] > 0.5:
            return True
        return bool(row['sed_class'] in ['Spiral', 'S0', 'Starburst'] and row['zoobot_featured'] > 0.5)

    agreement_frac = comp_df.apply(simple_match, axis=1).mean()

    return ZoobotValidationReport(
        n_compared=len(comp_df),
        agreement_fraction=agreement_frac,
        confusion=confusion,
        elliptical_agreement=elliptical_agree if not np.isnan(elliptical_agree) else 0.0,
        spiral_agreement=spiral_agree if not np.isnan(spiral_agree) else 0.0,
        starburst_agreement=starburst_agree if not np.isnan(starburst_agree) else 0.0,
    )


# =============================================================================
# Simplified interface for users without Zoobot installed
# =============================================================================

def get_morphology_from_concentration(
    catalog: pd.DataFrame,
    concentration_col: str = 'concentration',
    c_elliptical_threshold: float = 1.5,
    c_spiral_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Estimate morphology from concentration index (fallback without Zoobot).

    This is a simple proxy - ellipticals are more concentrated than spirals.

    Parameters
    ----------
    catalog : pd.DataFrame
        Must have concentration column
    concentration_col : str
        Name of concentration column
    c_elliptical_threshold : float
        C > this suggests elliptical
    c_spiral_threshold : float
        C > this suggests compact/elliptical, < suggests extended/spiral

    Returns
    -------
    pd.DataFrame
        Original catalog with 'morph_estimate' column
    """
    result = catalog.copy()

    c = catalog[concentration_col]

    # Simple classification based on concentration
    conditions = [
        c < c_elliptical_threshold,  # Very extended - likely spiral/irregular
        c < c_spiral_threshold,       # Extended - spiral
        c >= c_spiral_threshold,      # Compact - elliptical/S0
    ]
    choices = ['Spiral/Irr', 'Spiral', 'E/S0']

    result['morph_estimate'] = np.select(conditions, choices, default='Unknown')

    return result


def cross_validate_sed_with_concentration(
    catalog: pd.DataFrame,
    type_col: str = 'galaxy_type',
    concentration_col: str = 'concentration',
) -> dict:
    """
    Simple cross-validation using concentration as morphology proxy.

    Parameters
    ----------
    catalog : pd.DataFrame
        Must have galaxy_type and concentration columns

    Returns
    -------
    dict
        Agreement statistics
    """
    if concentration_col not in catalog.columns:
        return {'error': f'Column {concentration_col} not found'}

    # Add morphology estimate
    cat_with_morph = get_morphology_from_concentration(catalog, concentration_col)

    # Check agreement
    results = {}

    for sed_type in catalog[type_col].unique():
        mask = catalog[type_col] == sed_type
        morph_dist = cat_with_morph.loc[mask, 'morph_estimate'].value_counts(normalize=True)
        results[sed_type] = morph_dist.to_dict()

    # Compute agreement metrics
    elliptical_mask = catalog[type_col].str.lower() == 'elliptical'
    if elliptical_mask.any():
        e_agree = (cat_with_morph.loc[elliptical_mask, 'morph_estimate'] == 'E/S0').mean()
    else:
        e_agree = np.nan

    spiral_mask = catalog[type_col].str.lower().isin(['sa', 'sb'])
    if spiral_mask.any():
        s_agree = cat_with_morph.loc[spiral_mask, 'morph_estimate'].str.contains('Spiral').mean()
    else:
        s_agree = np.nan

    return {
        'by_type': results,
        'elliptical_compact_fraction': e_agree,
        'spiral_extended_fraction': s_agree,
        'n_sources': len(catalog),
    }
