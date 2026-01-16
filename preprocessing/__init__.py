"""Preprocessing module for astronomical image analysis.

This module provides preprocessing utilities for astronomical images:

1. Cosmic Ray Removal
   - deepCR: Deep learning based removal (recommended)
   - LACosmic: Traditional edge-detection method (fallback)

2. Pixel-Level Segmentation (Morpheus)
   - Semantic segmentation of astronomical images
   - Per-pixel classification: star/galaxy/background/artifact

Usage
-----
>>> from preprocessing import clean_cosmic_rays, segment_image
>>>
>>> # Clean cosmic rays
>>> cleaned, cr_mask = clean_cosmic_rays(image, instrument='WFPC2')
>>>
>>> # Segment image (star/galaxy/artifact per pixel)
>>> segmentation = segment_image(image)

Installation
------------
For full functionality, install optional dependencies:
    pip install deepCR morpheus-astro astroscrappy
"""

# Cosmic ray removal
from .cosmic_ray_removal import (
    DeepCRCleaner,
    LACosmic,
    CRCleaningResult,
    clean_cosmic_rays,
    clean_image_stack,
    check_deepcr_availability,
)

# Morpheus segmentation (if available)
try:
    from .morpheus_segmentation import (
        MorpheusSegmenter,
        segment_image,
        check_morpheus_availability,
    )
    MORPHEUS_AVAILABLE = True
except ImportError:
    MORPHEUS_AVAILABLE = False

    def check_morpheus_availability():
        return False

__all__ = [
    # Cosmic ray removal
    'DeepCRCleaner',
    'LACosmic',
    'CRCleaningResult',
    'clean_cosmic_rays',
    'clean_image_stack',
    'check_deepcr_availability',
    # Morpheus segmentation
    'MORPHEUS_AVAILABLE',
    'check_morpheus_availability',
]

# Add Morpheus exports if available
if MORPHEUS_AVAILABLE:
    __all__.extend([
        'MorpheusSegmenter',
        'segment_image',
    ])
