"""Detection module with optional deep learning support.

This module provides source detection methods for astronomical images.

Available methods:
- Traditional: photutils/SEP based detection
- Deep learning: CNN-enhanced detection with star/galaxy/artifact classification

Usage:
    # Traditional detection (default)
    from detection import detect_sources_professional

    # Deep learning enhanced detection (optional)
    from detection.deep_detection import DeepSourceDetector, detect_sources_deep

References:
- photutils: https://photutils.readthedocs.io/
- SEP: https://sep.readthedocs.io/
- Burke et al. 2019, MNRAS, 490, 3952 (Deep source detection)
"""

# Re-export deep detection components for convenience
try:
    from .deep_detection import (
        DeepSourceDetector,
        DetectedSource,
        SourceClassifierCNN,
        detect_sources_deep,
        train_source_classifier,
    )
    DEEP_DETECTION_AVAILABLE = True
except ImportError:
    DEEP_DETECTION_AVAILABLE = False

__all__ = [
    'DeepSourceDetector',
    'DetectedSource',
    'SourceClassifierCNN',
    'detect_sources_deep',
    'train_source_classifier',
    'DEEP_DETECTION_AVAILABLE',
]
