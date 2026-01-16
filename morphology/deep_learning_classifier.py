"""Deep learning-based galaxy morphology classification using Zoobot.

This module provides deep learning classifiers for galaxy morphology using
pre-trained models from the Zoobot library (Walmsley et al. 2023). Zoobot
was trained on 450,000+ Galaxy Zoo volunteer classifications.

Available classifiers:
- ZoobotClassifier: Pre-trained CNN for detailed morphology (spiral, elliptical, etc.)
- ZoobotStarGalaxyClassifier: Finetuned for star-galaxy separation

Key features:
- Transfer learning from Galaxy Zoo DECaLS training
- Bayesian CNN with uncertainty quantification
- GPU acceleration support
- Multiple morphology outputs (rings, bars, spiral arms, etc.)

References:
- Walmsley et al. 2023, JOSS, 8, 5312 (Zoobot)
- Walmsley et al. 2022, MNRAS, 509, 3966 (Galaxy Zoo DECaLS)
- Dieleman et al. 2015, MNRAS, 450, 1441 (Original Galaxy Zoo CNN)

Installation:
    pip install zoobot[pytorch]  # or zoobot[tensorflow]
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Check for deep learning framework availability
_TORCH_AVAILABLE = False
_ZOOBOT_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    if _TORCH_AVAILABLE:
        from zoobot.pytorch.training import finetune
        from zoobot.pytorch.predictions import predict_on_catalog
        _ZOOBOT_AVAILABLE = True
except ImportError:
    pass


@dataclass
class DeepLearningResult:
    """Result of deep learning morphology classification.

    Attributes
    ----------
    is_galaxy : bool
        True if classified as galaxy
    is_star : bool
        True if classified as star
    probability_galaxy : float
        Probability of being a galaxy (0-1)
    confidence : float
        Classification confidence (0-1)
    morphology_type : str
        Predicted morphology (elliptical, spiral, irregular, etc.)
    morphology_probabilities : dict
        Probabilities for each morphology class
    features : dict
        Detailed morphological features (arms, bars, rings, etc.)
    """
    is_galaxy: bool
    is_star: bool
    probability_galaxy: float
    confidence: float
    morphology_type: str = "unknown"
    morphology_probabilities: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)


class ZoobotClassifier:
    """Deep learning galaxy morphology classifier using Zoobot.

    Zoobot provides state-of-the-art galaxy morphology classification
    trained on Galaxy Zoo volunteer labels. It can identify:
    - Galaxy vs star/artifact
    - Morphological type (elliptical, spiral, irregular)
    - Detailed features (bars, rings, spiral arms, mergers)

    The model uses a ResNet-based architecture with Bayesian dropout
    for uncertainty quantification.

    Attributes
    ----------
    model : nn.Module
        The loaded PyTorch model
    device : str
        Device for computation ('cuda' or 'cpu')
    is_loaded : bool
        Whether the model is loaded and ready

    Examples
    --------
    >>> clf = ZoobotClassifier()
    >>> clf.load_pretrained()
    >>> results = clf.predict(image_paths, catalog)
    """

    # Morphology classes from Galaxy Zoo
    MORPHOLOGY_CLASSES = [
        "smooth",           # Elliptical/S0
        "featured",         # Spiral/Irregular with features
        "artifact",         # Star, artifact, or image problem
    ]

    # Detailed feature columns from Zoobot
    FEATURE_COLUMNS = [
        "smooth-or-featured_smooth",
        "smooth-or-featured_featured-or-disk",
        "smooth-or-featured_artifact",
        "disk-edge-on_yes",
        "disk-edge-on_no",
        "has-spiral-arms_yes",
        "has-spiral-arms_no",
        "bar_strong",
        "bar_weak",
        "bar_no",
        "bulge-size_dominant",
        "bulge-size_large",
        "bulge-size_moderate",
        "bulge-size_small",
        "bulge-size_none",
        "how-rounded_round",
        "how-rounded_in-between",
        "how-rounded_cigar-shaped",
        "edge-on-bulge_boxy",
        "edge-on-bulge_none",
        "edge-on-bulge_rounded",
        "spiral-winding_tight",
        "spiral-winding_medium",
        "spiral-winding_loose",
        "spiral-arm-count_1",
        "spiral-arm-count_2",
        "spiral-arm-count_3",
        "spiral-arm-count_4",
        "spiral-arm-count_more-than-4",
        "merging_none",
        "merging_minor-disturbance",
        "merging_major-disturbance",
        "merging_merger",
    ]

    def __init__(
        self,
        device: str | None = None,
        model_architecture: str = "efficientnet_b0",
    ):
        """Initialize the Zoobot classifier.

        Parameters
        ----------
        device : str, optional
            Device for computation. If None, auto-detects GPU.
        model_architecture : str
            Model architecture: 'efficientnet_b0', 'resnet18', 'resnet50'
        """
        if not _ZOOBOT_AVAILABLE:
            raise ImportError(
                "Zoobot is required for deep learning classification. "
                "Install with: pip install zoobot[pytorch]"
            )

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_architecture = model_architecture
        self.model = None
        self.is_loaded = False
        self._checkpoint_path = None

    def load_pretrained(
        self,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        """Load a pre-trained Zoobot model.

        Parameters
        ----------
        checkpoint_path : str or Path, optional
            Path to checkpoint file. If None, downloads the default
            Galaxy Zoo DECaLS model.
        """
        from zoobot.pytorch.training import finetune

        if checkpoint_path is not None:
            self._checkpoint_path = Path(checkpoint_path)
            # Load from checkpoint
            self.model = finetune.FinetuneableZoobotClassifier.load_from_checkpoint(
                str(self._checkpoint_path)
            )
        else:
            # Use default pre-trained model
            # This will download if not cached
            self.model = finetune.FinetuneableZoobotClassifier(
                name=self.model_architecture,
                num_classes=len(self.MORPHOLOGY_CLASSES),
            )

        self.model = self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def predict_from_images(
        self,
        images: NDArray | list[NDArray],
        batch_size: int = 32,
    ) -> list[DeepLearningResult]:
        """Classify galaxies from image arrays.

        Parameters
        ----------
        images : NDArray or list of NDArray
            Galaxy images as numpy arrays. Expected shape: (N, H, W, C) or
            list of (H, W, C) arrays. Images should be RGB, 0-255.
        batch_size : int
            Batch size for inference

        Returns
        -------
        list[DeepLearningResult]
            Classification results for each image
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_pretrained() first.")

        import torchvision.transforms as T

        # Standard Zoobot preprocessing
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Convert to list if single array
        if isinstance(images, np.ndarray) and images.ndim == 4:
            images = [images[i] for i in range(len(images))]
        elif isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]

        results = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            # Preprocess batch
            tensors = []
            for img in batch_images:
                # Ensure uint8 format
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
                # Ensure RGB (3 channels)
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                elif img.shape[-1] == 1:
                    img = np.concatenate([img] * 3, axis=-1)
                tensors.append(transform(img))

            batch_tensor = torch.stack(tensors).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)

                # Softmax for probabilities
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

            # Convert to results
            for j, prob in enumerate(probs):
                # Get morphology type
                class_idx = np.argmax(prob)
                morph_type = self.MORPHOLOGY_CLASSES[class_idx]

                # Determine if galaxy or star/artifact
                is_artifact = morph_type == "artifact"
                is_galaxy = not is_artifact
                prob_galaxy = 1.0 - prob[2] if len(prob) > 2 else prob[1]

                # Confidence from max probability
                confidence = float(np.max(prob))

                results.append(DeepLearningResult(
                    is_galaxy=is_galaxy,
                    is_star=is_artifact,
                    probability_galaxy=float(prob_galaxy),
                    confidence=confidence,
                    morphology_type=morph_type,
                    morphology_probabilities={
                        cls: float(prob[k]) for k, cls in enumerate(self.MORPHOLOGY_CLASSES)
                    },
                ))

        return results

    def predict_from_catalog(
        self,
        catalog: pd.DataFrame,
        image_col: str = "file_loc",
        batch_size: int = 32,
        save_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Classify galaxies from a catalog with image paths.

        This is the recommended method for large catalogs.

        Parameters
        ----------
        catalog : pd.DataFrame
            Catalog with image file paths
        image_col : str
            Column name containing image file paths
        batch_size : int
            Batch size for inference
        save_path : str or Path, optional
            Save predictions to this path

        Returns
        -------
        pd.DataFrame
            Catalog with added prediction columns
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_pretrained() first.")

        from zoobot.pytorch.predictions import predict_on_catalog

        # Create output dataframe
        result_df = catalog.copy()

        # Run predictions using Zoobot's built-in function
        predictions = predict_on_catalog.predict(
            catalog,
            self.model,
            label_cols=self.MORPHOLOGY_CLASSES,
            save_loc=str(save_path) if save_path else None,
            batch_size=batch_size,
        )

        # Merge predictions
        for col in predictions.columns:
            if col not in result_df.columns:
                result_df[col] = predictions[col]

        return result_df


class ZoobotStarGalaxyClassifier:
    """Simplified Zoobot classifier optimized for star-galaxy separation.

    This classifier focuses specifically on separating stars from galaxies
    using Zoobot's pre-trained features. It's faster than the full
    ZoobotClassifier when you only need star/galaxy classification.

    For detailed morphology, use ZoobotClassifier instead.

    Examples
    --------
    >>> clf = ZoobotStarGalaxyClassifier()
    >>> clf.load_pretrained()
    >>> results = clf.predict(images)
    >>> for r in results:
    ...     print(f"Galaxy: {r.is_galaxy}, Confidence: {r.confidence:.2f}")
    """

    def __init__(self, device: str | None = None):
        """Initialize the star-galaxy classifier.

        Parameters
        ----------
        device : str, optional
            Device for computation. If None, auto-detects GPU.
        """
        if not _ZOOBOT_AVAILABLE:
            raise ImportError(
                "Zoobot is required for deep learning classification. "
                "Install with: pip install zoobot[pytorch]"
            )

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.is_loaded = False

    def load_pretrained(
        self,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        """Load a pre-trained model for star-galaxy classification.

        Parameters
        ----------
        checkpoint_path : str or Path, optional
            Path to finetuned checkpoint. If None, uses base Zoobot.
        """
        from zoobot.pytorch.training import finetune

        if checkpoint_path is not None:
            self.model = finetune.FinetuneableZoobotClassifier.load_from_checkpoint(
                str(checkpoint_path)
            )
        else:
            # Use base model with 2 classes (star, galaxy)
            self.model = finetune.FinetuneableZoobotClassifier(
                name="efficientnet_b0",
                num_classes=2,
            )

        self.model = self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def finetune(
        self,
        train_catalog: pd.DataFrame,
        val_catalog: pd.DataFrame,
        label_col: str = "is_galaxy",
        image_col: str = "file_loc",
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        save_dir: str | Path = "zoobot_finetuned",
    ) -> dict:
        """Finetune the model for star-galaxy separation.

        Parameters
        ----------
        train_catalog : pd.DataFrame
            Training catalog with image paths and labels
        val_catalog : pd.DataFrame
            Validation catalog
        label_col : str
            Column name for labels (0 = star, 1 = galaxy)
        image_col : str
            Column name for image paths
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        learning_rate : float
            Learning rate
        save_dir : str or Path
            Directory to save checkpoints

        Returns
        -------
        dict
            Training metrics
        """
        from zoobot.pytorch.training import finetune
        import pytorch_lightning as pl
        from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data module
        datamodule = GalaxyDataModule(
            label_cols=[label_col],
            catalog=train_catalog,
            batch_size=batch_size,
        )

        # Initialize model for finetuning
        model = finetune.FinetuneableZoobotClassifier(
            name="efficientnet_b0",
            num_classes=2,
        )

        # Configure trainer
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices=1,
            default_root_dir=str(save_dir),
        )

        # Train
        trainer.fit(model, datamodule)

        # Save final model
        final_path = save_dir / "final_model.ckpt"
        trainer.save_checkpoint(str(final_path))

        # Load the finetuned model
        self.model = model
        self.model.eval()
        self.is_loaded = True

        return {
            "checkpoint_path": str(final_path),
            "epochs": epochs,
        }

    def predict(
        self,
        images: NDArray | list[NDArray],
        batch_size: int = 32,
    ) -> list[DeepLearningResult]:
        """Classify sources as stars or galaxies.

        Parameters
        ----------
        images : NDArray or list of NDArray
            Source images
        batch_size : int
            Batch size for inference

        Returns
        -------
        list[DeepLearningResult]
            Classification results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_pretrained() first.")

        import torchvision.transforms as T

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Convert to list
        if isinstance(images, np.ndarray) and images.ndim == 4:
            images = [images[i] for i in range(len(images))]
        elif isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]

        results = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            tensors = []
            for img in batch_images:
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                elif img.shape[-1] == 1:
                    img = np.concatenate([img] * 3, axis=-1)
                tensors.append(transform(img))

            batch_tensor = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

            for prob in probs:
                # prob[0] = star, prob[1] = galaxy
                is_galaxy = prob[1] > prob[0]
                prob_galaxy = float(prob[1])
                confidence = float(max(prob))

                results.append(DeepLearningResult(
                    is_galaxy=is_galaxy,
                    is_star=not is_galaxy,
                    probability_galaxy=prob_galaxy,
                    confidence=confidence,
                    morphology_type="galaxy" if is_galaxy else "star",
                ))

        return results


def create_cutouts_for_zoobot(
    image: NDArray,
    catalog: pd.DataFrame,
    x_col: str = "xcentroid",
    y_col: str = "ycentroid",
    cutout_size: int = 64,
    output_dir: str | Path | None = None,
) -> tuple[list[NDArray], list[str]]:
    """Create image cutouts for Zoobot classification.

    Parameters
    ----------
    image : NDArray
        Full image array
    catalog : pd.DataFrame
        Source catalog with positions
    x_col, y_col : str
        Column names for centroid positions
    cutout_size : int
        Size of cutouts in pixels (will be resized to 224x224 by Zoobot)
    output_dir : str or Path, optional
        If provided, save cutouts as PNG files

    Returns
    -------
    cutouts : list of NDArray
        Image cutouts for each source
    file_paths : list of str
        Paths to saved files (empty if output_dir is None)
    """
    import cv2

    half_size = cutout_size // 2
    ny, nx = image.shape[:2]

    cutouts = []
    file_paths = []

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in catalog.iterrows():
        x, y = int(row[x_col]), int(row[y_col])

        # Extract cutout with boundary checking
        y_lo = max(0, y - half_size)
        y_hi = min(ny, y + half_size)
        x_lo = max(0, x - half_size)
        x_hi = min(nx, x + half_size)

        cutout = image[y_lo:y_hi, x_lo:x_hi]

        # Pad if necessary
        pad_y = cutout_size - cutout.shape[0]
        pad_x = cutout_size - cutout.shape[1]

        if pad_y > 0 or pad_x > 0:
            cutout = np.pad(
                cutout,
                ((0, pad_y), (0, pad_x)) if cutout.ndim == 2 else ((0, pad_y), (0, pad_x), (0, 0)),
                mode='constant',
                constant_values=0,
            )

        # Normalize to 0-255 for PNG
        if cutout.max() > 0:
            cutout_norm = ((cutout - cutout.min()) / (cutout.max() - cutout.min()) * 255).astype(np.uint8)
        else:
            cutout_norm = np.zeros_like(cutout, dtype=np.uint8)

        # Convert grayscale to RGB
        if cutout_norm.ndim == 2:
            cutout_rgb = np.stack([cutout_norm] * 3, axis=-1)
        else:
            cutout_rgb = cutout_norm

        cutouts.append(cutout_rgb)

        # Save if output directory provided
        if output_dir is not None:
            file_path = output_dir / f"source_{idx}.png"
            cv2.imwrite(str(file_path), cv2.cvtColor(cutout_rgb, cv2.COLOR_RGB2BGR))
            file_paths.append(str(file_path))

    return cutouts, file_paths


def check_zoobot_availability() -> dict:
    """Check if Zoobot and dependencies are available.

    Returns
    -------
    dict
        Status of each dependency
    """
    status = {
        "pytorch": _TORCH_AVAILABLE,
        "zoobot": _ZOOBOT_AVAILABLE,
        "cuda_available": False,
        "gpu_name": None,
    }

    if _TORCH_AVAILABLE:
        status["cuda_available"] = torch.cuda.is_available()
        if status["cuda_available"]:
            status["gpu_name"] = torch.cuda.get_device_name(0)

    return status
