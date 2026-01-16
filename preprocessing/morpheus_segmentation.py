"""Pixel-level semantic segmentation using Morpheus.

Morpheus (Hausen & Robertson 2020) performs pixel-level classification of
astronomical images into categories:
- Background: Sky background pixels
- Spheroid: Elliptical/bulge-dominated galaxy pixels
- Disk: Disk-dominated galaxy pixels
- Irregular: Irregular galaxy or merger pixels
- Point Source: Star or compact AGN pixels

This is particularly valuable for:
- Deblending overlapping sources
- Identifying galaxy components (bulge, disk, arms)
- Detecting mergers and irregular structures
- Separating stars from galaxies in crowded fields

References
----------
- Hausen & Robertson 2020, ApJS, 248, 20
- GitHub: https://github.com/morpheus-project/morpheus

Installation
------------
pip install morpheus-astro

Usage
-----
>>> from preprocessing.morpheus_segmentation import segment_image
>>> segmentation = segment_image(image_dict={'H': h_band, 'J': j_band})
>>> galaxy_mask = segmentation['spheroid'] + segmentation['disk'] > 0.5
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray


@dataclass
class SegmentationResult:
    """Result of Morpheus pixel-level segmentation."""

    # Probability maps for each class (same shape as input image)
    background: NDArray      # Probability of background
    spheroid: NDArray        # Probability of spheroid/elliptical
    disk: NDArray            # Probability of disk galaxy
    irregular: NDArray       # Probability of irregular/merger
    point_source: NDArray    # Probability of point source (star)

    # Derived masks
    @property
    def galaxy_mask(self) -> NDArray:
        """Binary mask of likely galaxy pixels."""
        return (self.spheroid + self.disk + self.irregular) > 0.5

    @property
    def star_mask(self) -> NDArray:
        """Binary mask of likely star pixels."""
        return self.point_source > 0.5

    @property
    def source_mask(self) -> NDArray:
        """Binary mask of any source (non-background) pixels."""
        return self.background < 0.5

    @property
    def classification_map(self) -> NDArray:
        """Integer map with most likely class per pixel.

        0 = background, 1 = spheroid, 2 = disk, 3 = irregular, 4 = point_source
        """
        probs = np.stack([
            self.background,
            self.spheroid,
            self.disk,
            self.irregular,
            self.point_source,
        ], axis=-1)
        return np.argmax(probs, axis=-1)

    @property
    def max_probability(self) -> NDArray:
        """Maximum probability at each pixel (confidence map)."""
        probs = np.stack([
            self.background,
            self.spheroid,
            self.disk,
            self.irregular,
            self.point_source,
        ], axis=-1)
        return np.max(probs, axis=-1)


def check_morpheus_availability() -> bool:
    """Check if Morpheus is installed and available."""
    try:
        import morpheus
        return True
    except ImportError:
        return False


class MorpheusSegmenter:
    """Pixel-level semantic segmentation using Morpheus.

    Morpheus is a deep learning framework that classifies each pixel in
    an astronomical image into one of five categories: background,
    spheroid, disk, irregular, or point source.

    Parameters
    ----------
    use_gpu : bool
        Use GPU acceleration if available. Default True.
    batch_size : int
        Batch size for inference. Default 8.

    Attributes
    ----------
    model : morpheus model
        Loaded Morpheus neural network

    Examples
    --------
    >>> segmenter = MorpheusSegmenter()
    >>> result = segmenter.segment({'H': h_image, 'J': j_image})
    >>> galaxy_pixels = result.galaxy_mask
    """

    # Morpheus class labels
    CLASS_NAMES: ClassVar[list[str]] = [
        'background', 'spheroid', 'disk', 'irregular', 'point_source'
    ]

    def __init__(
        self,
        use_gpu: bool = True,
        batch_size: int = 8,
    ):
        """Initialize the Morpheus segmenter."""
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self._model = None
        self._available = check_morpheus_availability()

    def _load_model(self):
        """Load the Morpheus model (lazy loading)."""
        if self._model is not None:
            return

        if not self._available:
            raise ImportError(
                "Morpheus is not installed. Install with: pip install morpheus-astro"
            )

        import morpheus
        from morpheus.classifier import Classifier

        # Initialize classifier (automatically downloads weights if needed)
        self._model = Classifier()

    @property
    def model(self):
        """Get the loaded model (loads on first access)."""
        self._load_model()
        return self._model

    def segment(
        self,
        images: dict[str, NDArray] | NDArray,
        bands: list[str] | None = None,
    ) -> SegmentationResult:
        """Segment an image into morphological classes.

        Parameters
        ----------
        images : dict or NDArray
            Either a dictionary mapping band names to 2D images,
            or a single 2D/3D image array. Morpheus works best with
            multi-band data (H, J bands recommended).
        bands : list of str, optional
            Band names if images is an array. Required if images is 3D.

        Returns
        -------
        SegmentationResult
            Probability maps for each morphological class
        """
        if not self._available:
            # Return dummy result if Morpheus not available
            if isinstance(images, dict):
                shape = list(images.values())[0].shape
            elif images.ndim == 3:
                shape = images.shape[1:]
            else:
                shape = images.shape

            return SegmentationResult(
                background=np.ones(shape),
                spheroid=np.zeros(shape),
                disk=np.zeros(shape),
                irregular=np.zeros(shape),
                point_source=np.zeros(shape),
            )

        self._load_model()

        # Prepare input data
        if isinstance(images, dict):
            # Multi-band dictionary input
            band_data = images
        elif images.ndim == 3:
            # 3D array with bands
            if bands is None:
                raise ValueError("bands must be specified for 3D array input")
            band_data = {b: images[i] for i, b in enumerate(bands)}
        else:
            # Single 2D image - use as H band
            band_data = {'H': images}

        # Run Morpheus classification
        # Morpheus returns a dictionary of probability maps
        output = self._model.classify(band_data)

        return SegmentationResult(
            background=output.get('background', np.zeros_like(list(band_data.values())[0])),
            spheroid=output.get('spheroid', np.zeros_like(list(band_data.values())[0])),
            disk=output.get('disk', np.zeros_like(list(band_data.values())[0])),
            irregular=output.get('irregular', np.zeros_like(list(band_data.values())[0])),
            point_source=output.get('point_source', np.zeros_like(list(band_data.values())[0])),
        )

    def segment_single_band(
        self,
        image: NDArray,
        band: str = 'H',
    ) -> SegmentationResult:
        """Segment a single-band image.

        Note: Morpheus is optimized for multi-band data. Single-band
        results may be less accurate.

        Parameters
        ----------
        image : NDArray
            2D image array
        band : str
            Band name (default 'H')

        Returns
        -------
        SegmentationResult
            Probability maps for each class
        """
        return self.segment({band: image})


def segment_image(
    image: NDArray | dict[str, NDArray],
    bands: list[str] | None = None,
    use_gpu: bool = True,
    verbose: bool = True,
) -> SegmentationResult:
    """Segment an astronomical image into morphological classes.

    This is the main entry point for Morpheus segmentation.

    Parameters
    ----------
    image : NDArray or dict
        2D/3D image array or dictionary of band images
    bands : list of str, optional
        Band names if image is a 3D array
    use_gpu : bool
        Use GPU if available. Default True.
    verbose : bool
        Print progress. Default True.

    Returns
    -------
    SegmentationResult
        Pixel-level classification probabilities

    Examples
    --------
    >>> # Single band
    >>> result = segment_image(i_band_image)
    >>>
    >>> # Multi-band (recommended)
    >>> result = segment_image({'H': h_image, 'J': j_image, 'V': v_image})
    >>>
    >>> # Get galaxy mask
    >>> galaxies = result.galaxy_mask
    """
    if verbose:
        print("    Running Morpheus pixel-level segmentation...")

    segmenter = MorpheusSegmenter(use_gpu=use_gpu)

    if not segmenter._available:
        if verbose:
            print("    Warning: Morpheus not available, returning empty segmentation")
        # Return empty result
        if isinstance(image, dict):
            shape = list(image.values())[0].shape
        elif hasattr(image, 'ndim') and image.ndim == 3:
            shape = image.shape[1:]
        else:
            shape = image.shape

        return SegmentationResult(
            background=np.ones(shape),
            spheroid=np.zeros(shape),
            disk=np.zeros(shape),
            irregular=np.zeros(shape),
            point_source=np.zeros(shape),
        )

    result = segmenter.segment(image, bands=bands)

    if verbose:
        # Print summary statistics
        class_map = result.classification_map
        n_pixels = class_map.size
        print(f"    Segmentation complete:")
        print(f"      Background: {100*(class_map == 0).sum()/n_pixels:.1f}%")
        print(f"      Spheroid:   {100*(class_map == 1).sum()/n_pixels:.1f}%")
        print(f"      Disk:       {100*(class_map == 2).sum()/n_pixels:.1f}%")
        print(f"      Irregular:  {100*(class_map == 3).sum()/n_pixels:.1f}%")
        print(f"      Point src:  {100*(class_map == 4).sum()/n_pixels:.1f}%")

    return result


def create_source_mask_from_segmentation(
    segmentation: SegmentationResult,
    min_probability: float = 0.5,
    exclude_point_sources: bool = False,
) -> NDArray:
    """Create a binary source mask from segmentation results.

    Parameters
    ----------
    segmentation : SegmentationResult
        Morpheus segmentation output
    min_probability : float
        Minimum probability threshold for source detection
    exclude_point_sources : bool
        Exclude point sources (stars) from mask

    Returns
    -------
    NDArray
        Binary mask where True = source pixel
    """
    if exclude_point_sources:
        source_prob = (
            segmentation.spheroid +
            segmentation.disk +
            segmentation.irregular
        )
    else:
        source_prob = 1.0 - segmentation.background

    return source_prob >= min_probability


def get_morphology_fractions(
    segmentation: SegmentationResult,
    source_mask: NDArray | None = None,
) -> dict[str, float]:
    """Calculate morphology fractions for sources in an image.

    Parameters
    ----------
    segmentation : SegmentationResult
        Morpheus segmentation output
    source_mask : NDArray, optional
        Binary mask of source pixels. If None, uses all non-background.

    Returns
    -------
    dict
        Fractions of each morphological class
    """
    if source_mask is None:
        source_mask = segmentation.source_mask

    if not np.any(source_mask):
        return {
            'spheroid': 0.0,
            'disk': 0.0,
            'irregular': 0.0,
            'point_source': 0.0,
        }

    # Calculate mean probabilities within source regions
    total = source_mask.sum()
    return {
        'spheroid': float(segmentation.spheroid[source_mask].sum() / total),
        'disk': float(segmentation.disk[source_mask].sum() / total),
        'irregular': float(segmentation.irregular[source_mask].sum() / total),
        'point_source': float(segmentation.point_source[source_mask].sum() / total),
    }
