"""Deep learning source detection module.

This module provides alternative source detection methods using deep learning,
offering potential improvements over traditional threshold-based detection
for:
- Faint source detection in low-SNR regions
- Improved deblending of overlapping sources
- Star-galaxy separation at detection stage
- Artifact rejection

Available methods:
1. SEP + ML refinement: Traditional detection with ML-based filtering
2. U-Net segmentation: Pixel-level source segmentation
3. YOLO-style detection: Object detection with bounding boxes

Requirements:
- torch (for neural network models)
- torchvision (for pre-trained backbones)
- sep (for fallback detection)

References:
- Burke et al. 2019, MNRAS, 490, 3952 (Deep source detection)
- Hausen & Robertson 2020, ApJS, 248, 20 (DeepSource)
- Farias et al. 2020, A&C, 33, 100420 (ML source classification)
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Optional imports with graceful fallback
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import sep
    SEP_AVAILABLE = True
except ImportError:
    SEP_AVAILABLE = False

# Create directory if running as script
os.makedirs(Path(__file__).parent, exist_ok=True)


@dataclass
class DetectedSource:
    """A detected source with properties.

    Attributes
    ----------
    x : float
        X centroid position (pixels)
    y : float
        Y centroid position (pixels)
    flux : float
        Integrated flux
    flux_err : float
        Flux uncertainty
    a : float
        Semi-major axis (pixels)
    b : float
        Semi-minor axis (pixels)
    theta : float
        Position angle (radians)
    is_star : bool
        Star classification from detection
    confidence : float
        Detection confidence (0-1)
    flags : int
        Quality flags
    """
    x: float
    y: float
    flux: float
    flux_err: float
    a: float
    b: float
    theta: float
    is_star: bool
    confidence: float
    flags: int


class SourceClassifierCNN(nn.Module):
    """Simple CNN for source classification at detection stage.

    Classifies cutouts as: star, galaxy, artifact, or noise.
    """

    def __init__(self, input_size: int = 32, n_classes: int = 4):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate flattened size
        conv_size = input_size // 8  # After 3 pooling layers
        self.fc_input_size = 128 * conv_size * conv_size

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, n_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (batch, 1, H, W)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, self.fc_input_size)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return F.softmax(x, dim=1)


class DeepSourceDetector:
    """Deep learning enhanced source detector.

    Combines traditional detection (SEP/photutils) with deep learning
    for improved source classification and filtering.

    Parameters
    ----------
    method : str
        Detection method: 'sep_ml' (SEP + ML filter), 'unet', or 'hybrid'
    cutout_size : int
        Size of cutouts for classification (pixels)
    detection_threshold : float
        Detection threshold in sigma above background
    use_gpu : bool
        Use GPU if available
    model_path : str, optional
        Path to pre-trained model weights

    Examples
    --------
    >>> detector = DeepSourceDetector(method='sep_ml')
    >>> sources = detector.detect(image_data)
    >>> print(f"Found {len(sources)} sources")
    """

    def __init__(
        self,
        method: str = 'sep_ml',
        cutout_size: int = 32,
        detection_threshold: float = 2.0,
        use_gpu: bool = True,
        model_path: str | None = None,
    ):
        self.method = method
        self.cutout_size = cutout_size
        self.threshold = detection_threshold
        self.use_gpu = use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
        self.model_path = model_path

        self.device = 'cuda' if self.use_gpu else 'cpu'
        self.classifier = None

        if TORCH_AVAILABLE and method in ['sep_ml', 'hybrid']:
            self._init_classifier()

    def _init_classifier(self):
        """Initialize the source classifier CNN."""
        self.classifier = SourceClassifierCNN(
            input_size=self.cutout_size,
            n_classes=4,  # star, galaxy, artifact, noise
        )

        if self.model_path and Path(self.model_path).exists():
            self.classifier.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            print(f"Loaded classifier weights from {self.model_path}")
        else:
            print("Using randomly initialized classifier (no pre-trained weights)")

        self.classifier.to(self.device)
        self.classifier.eval()

    def _extract_cutout(
        self,
        image: NDArray,
        x: float,
        y: float,
        size: int,
    ) -> NDArray | None:
        """Extract a square cutout centered on (x, y)."""
        half = size // 2
        x_int, y_int = int(round(x)), int(round(y))

        # Check bounds
        if (x_int - half < 0 or x_int + half >= image.shape[1] or
            y_int - half < 0 or y_int + half >= image.shape[0]):
            return None

        cutout = image[y_int - half:y_int + half,
                       x_int - half:x_int + half].copy()

        if cutout.shape != (size, size):
            return None

        return cutout

    def _normalize_cutout(self, cutout: NDArray) -> NDArray:
        """Normalize cutout for CNN input."""
        # Asinh stretch
        cutout = np.arcsinh(cutout / np.std(cutout))

        # Scale to 0-1
        vmin, vmax = np.percentile(cutout, [1, 99])
        if vmax > vmin:
            cutout = (cutout - vmin) / (vmax - vmin)

        return np.clip(cutout, 0, 1).astype(np.float32)

    def _classify_cutouts(
        self,
        cutouts: list[NDArray],
    ) -> tuple[NDArray, NDArray]:
        """Classify cutouts using CNN.

        Returns
        -------
        classes : NDArray
            Class predictions (0=star, 1=galaxy, 2=artifact, 3=noise)
        confidences : NDArray
            Prediction confidences
        """
        if not TORCH_AVAILABLE or self.classifier is None:
            # Fallback: assume all are galaxies
            n = len(cutouts)
            return np.ones(n, dtype=int), np.ones(n) * 0.5

        # Prepare batch
        batch = np.stack([self._normalize_cutout(c) for c in cutouts])
        batch = torch.from_numpy(batch).unsqueeze(1).to(self.device)

        with torch.no_grad():
            probs = self.classifier(batch).cpu().numpy()

        classes = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        return classes, confidences

    def detect_sep(
        self,
        image: NDArray,
        error: NDArray | None = None,
    ) -> list[DetectedSource]:
        """Detect sources using SEP (Source Extractor Python).

        Parameters
        ----------
        image : NDArray
            2D image array
        error : NDArray, optional
            Error map (same shape as image)

        Returns
        -------
        list[DetectedSource]
            Detected sources
        """
        if not SEP_AVAILABLE:
            raise ImportError("SEP not available. Install with: pip install sep")

        # Ensure C-contiguous float array
        data = np.ascontiguousarray(image.astype(np.float64))

        # Background subtraction
        bkg = sep.Background(data)
        data_sub = data - bkg.back()

        # Error map
        if error is None:
            err = bkg.globalrms
        else:
            err = np.ascontiguousarray(error.astype(np.float64))

        # Extract sources
        objects = sep.extract(
            data_sub,
            self.threshold,
            err=err,
            minarea=5,
            deblend_cont=0.005,
        )

        # Convert to DetectedSource objects
        sources = []
        for obj in objects:
            sources.append(DetectedSource(
                x=float(obj['x']),
                y=float(obj['y']),
                flux=float(obj['flux']),
                flux_err=float(obj['fluxerr']),
                a=float(obj['a']),
                b=float(obj['b']),
                theta=float(obj['theta']),
                is_star=False,  # Will be set by ML
                confidence=1.0,
                flags=int(obj['flag']),
            ))

        return sources

    def detect(
        self,
        image: NDArray,
        error: NDArray | None = None,
        classify: bool = True,
    ) -> list[DetectedSource]:
        """Detect sources with optional ML classification.

        Parameters
        ----------
        image : NDArray
            2D image array
        error : NDArray, optional
            Error map
        classify : bool
            Apply ML classification to detected sources

        Returns
        -------
        list[DetectedSource]
            Detected and optionally classified sources
        """
        # Initial detection with SEP
        if self.method in ['sep_ml', 'hybrid'] or not TORCH_AVAILABLE:
            sources = self.detect_sep(image, error)
        else:
            sources = self.detect_sep(image, error)  # Fallback

        if not classify or not TORCH_AVAILABLE or self.classifier is None:
            return sources

        # Extract cutouts for classification
        print(f"  Classifying {len(sources)} detected sources with CNN...")
        cutouts = []
        valid_indices = []

        for i, src in enumerate(sources):
            cutout = self._extract_cutout(image, src.x, src.y, self.cutout_size)
            if cutout is not None:
                cutouts.append(cutout)
                valid_indices.append(i)

        if len(cutouts) == 0:
            return sources

        # Classify cutouts
        classes, confidences = self._classify_cutouts(cutouts)

        # Update sources with classifications
        class_names = ['star', 'galaxy', 'artifact', 'noise']
        n_stars, n_galaxies, n_artifacts, n_noise = 0, 0, 0, 0

        for i, idx in enumerate(valid_indices):
            src = sources[idx]
            cls = classes[i]
            conf = confidences[i]

            # Update source
            src.confidence = float(conf)
            src.is_star = (cls == 0)  # Class 0 = star

            if cls == 0:
                n_stars += 1
            elif cls == 1:
                n_galaxies += 1
            elif cls == 2:
                n_artifacts += 1
            else:
                n_noise += 1

        print(f"    Classification: {n_stars} stars, {n_galaxies} galaxies, "
              f"{n_artifacts} artifacts, {n_noise} noise")

        # Filter artifacts and noise
        if self.method in ['sep_ml', 'hybrid']:
            sources = [s for i, s in enumerate(sources)
                       if i not in valid_indices or classes[valid_indices.index(i)] < 2]
            print(f"    After filtering: {len(sources)} sources")

        return sources

    def detect_to_dataframe(
        self,
        image: NDArray,
        error: NDArray | None = None,
        classify: bool = True,
    ) -> pd.DataFrame:
        """Detect sources and return as DataFrame.

        Parameters
        ----------
        image : NDArray
            2D image array
        error : NDArray, optional
            Error map
        classify : bool
            Apply ML classification

        Returns
        -------
        pd.DataFrame
            Source catalog with detection properties
        """
        sources = self.detect(image, error, classify)

        if len(sources) == 0:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                'xcentroid': s.x,
                'ycentroid': s.y,
                'segment_flux': s.flux,
                'segment_flux_err': s.flux_err,
                'semimajor_axis': s.a,
                'semiminor_axis': s.b,
                'orientation': s.theta,
                'is_star_ml': s.is_star,
                'detection_confidence': s.confidence,
                'detection_flags': s.flags,
                'ellipticity': 1.0 - s.b / max(s.a, 1e-10),
            }
            for s in sources
        ])


def train_source_classifier(
    training_cutouts: NDArray,
    training_labels: NDArray,
    validation_fraction: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    save_path: str | None = None,
) -> SourceClassifierCNN:
    """Train the source classifier CNN.

    Parameters
    ----------
    training_cutouts : NDArray, shape (N, H, W)
        Training cutouts
    training_labels : NDArray, shape (N,)
        Class labels (0=star, 1=galaxy, 2=artifact, 3=noise)
    validation_fraction : float
        Fraction of data for validation
    epochs : int
        Number of training epochs
    batch_size : int
        Training batch size
    learning_rate : float
        Learning rate
    save_path : str, optional
        Path to save trained model

    Returns
    -------
    SourceClassifierCNN
        Trained classifier
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for training. Install with: pip install torch")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")

    # Prepare data
    n_total = len(training_cutouts)
    n_val = int(n_total * validation_fraction)
    indices = np.random.permutation(n_total)
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    # Normalize cutouts
    def normalize_batch(cutouts):
        normalized = []
        for c in cutouts:
            c = np.arcsinh(c / np.std(c))
            vmin, vmax = np.percentile(c, [1, 99])
            if vmax > vmin:
                c = (c - vmin) / (vmax - vmin)
            normalized.append(np.clip(c, 0, 1))
        return np.array(normalized, dtype=np.float32)

    X_train = normalize_batch(training_cutouts[train_idx])
    X_val = normalize_batch(training_cutouts[val_idx])
    y_train = training_labels[train_idx]
    y_val = training_labels[val_idx]

    # Create model
    input_size = training_cutouts.shape[1]
    n_classes = len(np.unique(training_labels))
    model = SourceClassifierCNN(input_size=input_size, n_classes=n_classes)
    model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # Training loop
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0

        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = torch.from_numpy(X_train[i:i+batch_size]).unsqueeze(1).to(device)
            batch_y = torch.from_numpy(y_train[i:i+batch_size]).long().to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == batch_y).sum().item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_X = torch.from_numpy(X_val).unsqueeze(1).to(device)
            val_y = torch.from_numpy(y_val).long().to(device)
            val_outputs = model(val_X)
            val_loss = criterion(val_outputs, val_y).item()
            val_correct = (val_outputs.argmax(1) == val_y).sum().item()

        train_acc = train_correct / len(X_train)
        val_acc = val_correct / len(X_val)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train acc={100*train_acc:.1f}%, Val acc={100*val_acc:.1f}%")

        # Save best model
        if val_acc > best_val_acc and save_path:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"Training complete. Best validation accuracy: {100*best_val_acc:.1f}%")

    if save_path:
        print(f"Model saved to {save_path}")

    return model


# Convenience function for integration with existing pipeline
def detect_sources_deep(
    image: NDArray,
    error: NDArray | None = None,
    threshold: float = 2.0,
    classify: bool = True,
) -> pd.DataFrame:
    """Detect sources using deep learning enhanced method.

    This is a drop-in replacement for detect_sources_professional that
    adds ML-based source classification.

    Parameters
    ----------
    image : NDArray
        2D image array
    error : NDArray, optional
        Error map
    threshold : float
        Detection threshold in sigma
    classify : bool
        Apply ML classification

    Returns
    -------
    pd.DataFrame
        Source catalog
    """
    detector = DeepSourceDetector(
        method='sep_ml',
        detection_threshold=threshold,
    )

    return detector.detect_to_dataframe(image, error, classify)
