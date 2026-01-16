"""Zoobot galaxy morphology classification.

This module provides:
1. Extraction of Zoobot encoder embeddings from galaxy cutouts
2. Morphology classification using embeddings with simple ML classifiers

The embeddings capture visual morphological features learned from millions
of Galaxy Zoo classifications, enabling transfer learning for morphology.
"""

from pathlib import Path
from typing import NamedTuple
import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    import torch
    from PIL import Image
    from torchvision import transforms
    ZOOBOT_AVAILABLE = True
except ImportError:
    ZOOBOT_AVAILABLE = False

# Morphology classes for classification
MORPHOLOGY_CLASSES = ("elliptical", "spiral", "irregular", "edge-on", "merger")


class MorphologyResult(NamedTuple):
    """Result of morphology classification from embeddings."""
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]


def extract_cutouts(
    image: NDArray,
    catalog: pd.DataFrame,
    output_dir: str | Path,
    size: int = 128,
) -> list[str]:
    """Extract and save cutouts as PNGs. Returns list of saved paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    half, paths = size // 2, []
    ny, nx = image.shape

    for idx, row in catalog.iterrows():
        x, y = int(row["xcentroid"]), int(row["ycentroid"])
        if x < half or x >= nx - half or y < half or y >= ny - half:
            continue

        cutout = image[y - half : y + half, x - half : x + half]
        cutout = ((cutout - np.nanmin(cutout)) / (np.nanmax(cutout) - np.nanmin(cutout) + 1e-10) * 255).astype(np.uint8)

        path = output_dir / f"{idx}.png"
        Image.fromarray(cutout, mode='L').save(path)
        paths.append((idx, str(path)))

    return paths


def classify_morphology(catalog: pd.DataFrame, cutout_paths: list[tuple]) -> pd.DataFrame:
    """Run Zoobot inference on cutouts."""
    if not ZOOBOT_AVAILABLE:
        print("  Zoobot not installed. Run: pip install zoobot[pytorch]")
        return catalog

    # Try the new Zoobot API (v2.0+) first, fall back to legacy API
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = None

    # Method 1: New API (Zoobot v2.0+) using FinetuneableZoobotClassifier
    try:
        from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier
        # Load pretrained encoder - use the classifier's encoder for embeddings
        # num_classes must be > 1, but we only use the encoder anyway
        model = FinetuneableZoobotClassifier(
            name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano',
            num_classes=2,  # Must be > 1, we only use the encoder
        )
        encoder = model.encoder
        encoder = encoder.to(device).eval()
        print("  Zoobot: Using new API (FinetuneableZoobotClassifier)")
    except (ImportError, AttributeError, ValueError):
        pass

    # Method 2: Try timm-based loading (for very recent zoobot versions)
    if encoder is None:
        try:
            import timm
            encoder = timm.create_model(
                'hf_hub:mwalmsley/zoobot-encoder-convnext_nano',
                pretrained=True,
                num_classes=0,  # Remove classification head, get features only
            )
            encoder = encoder.to(device).eval()
            print("  Zoobot: Using timm-based loading")
        except Exception:
            pass

    # Method 3: Legacy API (Zoobot v1.x)
    if encoder is None:
        try:
            from zoobot.pytorch.estimators import define_model
            encoder = define_model.ZoobotEncoder.load_from_name(
                "hf_hub:mwalmsley/zoobot-encoder-efficientnet_b0"
            )
            encoder = encoder.to(device).eval()
            print("  Zoobot: Using legacy API (ZoobotEncoder)")
        except (ImportError, AttributeError) as e:
            raise RuntimeError(
                f"Could not load Zoobot encoder. Tried new API (FinetuneableZoobotClassifier), "
                f"timm, and legacy API (ZoobotEncoder). Error: {e}. "
                f"Try: pip install --upgrade zoobot timm"
            ) from e

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # grayscale -> RGB
    ])

    result = catalog.copy()
    with torch.no_grad():
        for idx, path in cutout_paths:
            img = transform(Image.open(path)).unsqueeze(0).to(device)
            emb = encoder(img).cpu().numpy().flatten()
            for j in range(min(8, len(emb))):
                result.loc[idx, f"zoobot_emb_{j}"] = emb[j]

    print(f"  Zoobot: classified {len(cutout_paths)} galaxies")
    return result


# =============================================================================
# Morphology classification from embeddings
# =============================================================================


def get_embedding_columns(catalog: pd.DataFrame, n_components: int = 8) -> list[str]:
    """Get list of Zoobot embedding column names present in catalog."""
    cols = [f"zoobot_emb_{i}" for i in range(n_components)]
    return [c for c in cols if c in catalog.columns]


def extract_embeddings(catalog: pd.DataFrame, n_components: int = 8) -> NDArray | None:
    """Extract Zoobot embeddings from catalog as numpy array.

    Parameters
    ----------
    catalog : pd.DataFrame
        Catalog with zoobot_emb_* columns
    n_components : int
        Number of embedding dimensions to use (default 8)

    Returns
    -------
    NDArray or None
        Shape (n_galaxies, n_components) array, or None if embeddings not present
    """
    cols = get_embedding_columns(catalog, n_components)
    if not cols:
        return None

    embeddings = catalog[cols].values.astype(np.float64)

    # Handle any NaN values
    valid_mask = ~np.isnan(embeddings).any(axis=1)
    if not valid_mask.all():
        print(f"  Warning: {(~valid_mask).sum()} galaxies have missing embeddings")

    return embeddings


class ZoobotMorphologyClassifier:
    """Morphology classifier using Zoobot embeddings.

    Uses a simple MLP or Random Forest to classify galaxy morphology
    from pre-extracted Zoobot encoder embeddings.
    """

    def __init__(self, classifier_type: str = "mlp", n_components: int = 8):
        """Initialize classifier.

        Parameters
        ----------
        classifier_type : str
            Type of classifier: 'mlp' (neural network) or 'rf' (random forest)
        n_components : int
            Number of embedding dimensions to use
        """
        self.classifier_type = classifier_type
        self.n_components = n_components
        self.model = None
        self.classes_ = None
        self.is_fitted = False

    def fit(
        self,
        embeddings: NDArray,
        labels: NDArray,
        class_names: tuple[str, ...] | None = None,
    ) -> "ZoobotMorphologyClassifier":
        """Train the classifier on labeled embeddings.

        Parameters
        ----------
        embeddings : NDArray
            Shape (n_samples, n_features) embedding vectors
        labels : NDArray
            Shape (n_samples,) integer class labels or string labels
        class_names : tuple of str, optional
            Names for each class. If labels are strings, these are inferred.

        Returns
        -------
        self
        """
        from sklearn.preprocessing import LabelEncoder

        # Handle string labels
        if labels.dtype == object or labels.dtype.kind == 'U':
            le = LabelEncoder()
            y = le.fit_transform(labels)
            self.classes_ = tuple(le.classes_)
        else:
            y = labels.astype(int)
            if class_names is not None:
                self.classes_ = class_names
            else:
                self.classes_ = tuple(str(i) for i in range(int(y.max()) + 1))

        # Remove samples with NaN embeddings
        valid_mask = ~np.isnan(embeddings).any(axis=1)
        X = embeddings[valid_mask]
        y = y[valid_mask]

        if len(X) < 10:
            print(f"  Warning: Only {len(X)} valid training samples")
            return self

        # Create and train classifier
        if self.classifier_type == "mlp":
            from sklearn.neural_network import MLPClassifier
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
            )
        else:  # random forest
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )

        self.model.fit(X, y)
        self.is_fitted = True

        # Report training accuracy
        train_acc = self.model.score(X, y)
        print(f"  Zoobot morphology classifier trained: {len(X)} samples, "
              f"{len(self.classes_)} classes, train accuracy={train_acc:.3f}")

        return self

    def predict(self, embeddings: NDArray) -> NDArray:
        """Predict morphology classes.

        Parameters
        ----------
        embeddings : NDArray
            Shape (n_samples, n_features) embedding vectors

        Returns
        -------
        NDArray
            Shape (n_samples,) predicted class names
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        # Handle NaN embeddings
        valid_mask = ~np.isnan(embeddings).any(axis=1)
        predictions = np.full(len(embeddings), "unknown", dtype=object)

        if valid_mask.any():
            y_pred = self.model.predict(embeddings[valid_mask])
            predictions[valid_mask] = [self.classes_[i] for i in y_pred]

        return predictions

    def predict_proba(self, embeddings: NDArray) -> NDArray:
        """Predict class probabilities.

        Parameters
        ----------
        embeddings : NDArray
            Shape (n_samples, n_features) embedding vectors

        Returns
        -------
        NDArray
            Shape (n_samples, n_classes) probability matrix
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        n_classes = len(self.classes_)
        valid_mask = ~np.isnan(embeddings).any(axis=1)
        proba = np.zeros((len(embeddings), n_classes))

        if valid_mask.any():
            proba[valid_mask] = self.model.predict_proba(embeddings[valid_mask])

        return proba

    def classify_catalog(
        self,
        catalog: pd.DataFrame,
        output_col: str = "zoobot_morphology",
    ) -> pd.DataFrame:
        """Add morphology predictions to catalog.

        Parameters
        ----------
        catalog : pd.DataFrame
            Catalog with zoobot_emb_* columns
        output_col : str
            Name of output column for predictions

        Returns
        -------
        pd.DataFrame
            Catalog with added morphology columns
        """
        embeddings = extract_embeddings(catalog, self.n_components)
        if embeddings is None:
            print("  No Zoobot embeddings found in catalog")
            return catalog

        result = catalog.copy()
        result[output_col] = self.predict(embeddings)

        # Add probability columns
        proba = self.predict_proba(embeddings)
        for i, cls_name in enumerate(self.classes_):
            result[f"{output_col}_prob_{cls_name}"] = proba[:, i]

        # Add confidence (max probability)
        result[f"{output_col}_confidence"] = proba.max(axis=1)

        return result


def train_morphology_classifier(
    catalog: pd.DataFrame,
    label_column: str,
    classifier_type: str = "mlp",
    n_components: int = 8,
) -> ZoobotMorphologyClassifier | None:
    """Train a morphology classifier from a labeled catalog.

    Parameters
    ----------
    catalog : pd.DataFrame
        Catalog with zoobot_emb_* columns and a label column
    label_column : str
        Name of column containing morphology labels
    classifier_type : str
        'mlp' or 'rf'
    n_components : int
        Number of embedding dimensions

    Returns
    -------
    ZoobotMorphologyClassifier or None
        Trained classifier, or None if training failed
    """
    if label_column not in catalog.columns:
        print(f"  Label column '{label_column}' not found")
        return None

    embeddings = extract_embeddings(catalog, n_components)
    if embeddings is None:
        print("  No Zoobot embeddings found")
        return None

    # Get valid samples (have both embeddings and labels)
    labels = catalog[label_column].values
    valid_mask = ~pd.isna(labels) & ~np.isnan(embeddings).any(axis=1)

    if valid_mask.sum() < 10:
        print(f"  Insufficient labeled samples: {valid_mask.sum()}")
        return None

    clf = ZoobotMorphologyClassifier(classifier_type, n_components)
    clf.fit(embeddings[valid_mask], labels[valid_mask])

    return clf


def classify_by_sed_type(catalog: pd.DataFrame) -> pd.DataFrame:
    """Map SED galaxy types to broad morphology classes.

    This provides a baseline morphology estimate from template fitting
    when visual classification is unavailable.

    Parameters
    ----------
    catalog : pd.DataFrame
        Catalog with 'galaxy_type' column from SED fitting

    Returns
    -------
    pd.DataFrame
        Catalog with 'sed_morphology' column
    """
    # Mapping from SED types to morphology
    sed_to_morphology = {
        "elliptical": "elliptical",
        "S0": "elliptical",  # Lenticulars are closer to ellipticals
        "Sa": "spiral",
        "Sb": "spiral",
        "sbt1": "spiral",  # Early starburst - often spirals
        "sbt2": "spiral",
        "sbt3": "irregular",  # Mid starburst - can be irregular
        "sbt4": "irregular",
        "sbt5": "irregular",  # Late starburst - often irregular/merger
        "sbt6": "merger",  # Extreme starburst - often mergers
    }

    result = catalog.copy()
    if "galaxy_type" in catalog.columns:
        result["sed_morphology"] = catalog["galaxy_type"].map(sed_to_morphology)
        result["sed_morphology"] = result["sed_morphology"].fillna("unknown")
    else:
        result["sed_morphology"] = "unknown"

    return result
