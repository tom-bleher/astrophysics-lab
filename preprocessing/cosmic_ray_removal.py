"""Deep learning cosmic ray removal using deepCR.

This module provides cosmic ray detection and removal for astronomical images
using the deepCR neural network (Zhang & Bloom 2020). deepCR achieves superior
performance compared to traditional methods like LACosmic, especially for:
- Single exposures where median rejection isn't possible
- Faint sources near the detection limit
- Complex PSF shapes from HST instruments

Recommended thresholds by instrument (from deepCR documentation):
- ACS/WFC: 0.5
- WFC3/UVIS: 0.1-0.2
- WFPC2: 0.3 (estimated, similar to ACS)

References
----------
- Zhang & Bloom 2020, ApJ, 889, 24 (deepCR paper)
- van Dokkum 2001, PASP, 113, 1420 (LACosmic for comparison)

Installation
------------
pip install deepCR

Usage
-----
>>> from preprocessing.cosmic_ray_removal import clean_cosmic_rays, DeepCRCleaner
>>> cleaned_image, cr_mask = clean_cosmic_rays(image, instrument='ACS-WFC')
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray


class Instrument(Enum):
    """Supported HST instruments with optimized deepCR thresholds."""

    ACS_WFC = "ACS-WFC"
    WFC3_UVIS = "WFC3-UVIS"
    WFC3_IR = "WFC3-IR"
    WFPC2 = "WFPC2"
    GENERIC = "generic"


# Recommended thresholds by instrument
INSTRUMENT_THRESHOLDS: dict[Instrument, float] = {
    Instrument.ACS_WFC: 0.5,
    Instrument.WFC3_UVIS: 0.15,
    Instrument.WFC3_IR: 0.3,
    Instrument.WFPC2: 0.3,
    Instrument.GENERIC: 0.5,
}


@dataclass
class CRCleaningResult:
    """Result of cosmic ray cleaning."""

    cleaned_image: NDArray
    cr_mask: NDArray
    n_pixels_affected: int
    fraction_affected: float
    method: str


def check_deepcr_availability() -> bool:
    """Check if deepCR is installed and available."""
    try:
        import deepCR
        return True
    except ImportError:
        return False


class DeepCRCleaner:
    """Deep learning cosmic ray cleaner using deepCR.

    This class wraps the deepCR neural network for cosmic ray detection
    and removal. It supports multiple HST instruments with optimized
    thresholds and can use GPU acceleration when available.

    Parameters
    ----------
    instrument : str or Instrument
        HST instrument name. Supported: 'ACS-WFC', 'WFC3-UVIS', 'WFC3-IR',
        'WFPC2', 'generic'. Used to set default threshold.
    threshold : float, optional
        Detection threshold (0-1). Higher = fewer false positives but
        may miss faint CRs. If not specified, uses instrument default.
    use_gpu : bool, optional
        Use GPU acceleration if available. Default True.
    inpaint : bool, optional
        Inpaint (fill) detected CR pixels. Default True.

    Attributes
    ----------
    model : deepCR model
        Loaded deepCR neural network
    threshold : float
        Current detection threshold

    Examples
    --------
    >>> cleaner = DeepCRCleaner(instrument='ACS-WFC')
    >>> result = cleaner.clean(image)
    >>> print(f"Removed {result.n_pixels_affected} CR pixels")
    """

    # Pre-trained model names
    PRETRAINED_MODELS: ClassVar[dict[str, str]] = {
        'ACS-WFC': 'ACS-WFC-F606W-2-32',
        'WFC3-UVIS': 'ACS-WFC-F606W-2-32',  # Similar detector
        'WFPC2': 'ACS-WFC-F606W-2-32',  # Best available match
        'generic': 'ACS-WFC-F606W-2-32',
    }

    def __init__(
        self,
        instrument: str | Instrument = Instrument.GENERIC,
        threshold: float | None = None,
        use_gpu: bool = True,
        inpaint: bool = True,
    ):
        """Initialize the deepCR cleaner."""
        if isinstance(instrument, str):
            # Try to match instrument name
            instrument_upper = instrument.upper().replace('_', '-')
            try:
                self.instrument = Instrument(instrument_upper)
            except ValueError:
                # Try matching by name
                for inst in Instrument:
                    if inst.value.upper() == instrument_upper:
                        self.instrument = inst
                        break
                else:
                    self.instrument = Instrument.GENERIC
        else:
            self.instrument = instrument

        # Set threshold
        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = INSTRUMENT_THRESHOLDS.get(
                self.instrument, INSTRUMENT_THRESHOLDS[Instrument.GENERIC]
            )

        self.use_gpu = use_gpu
        self.inpaint = inpaint
        self._model = None
        self._available = check_deepcr_availability()

    def _load_model(self):
        """Load the deepCR model (lazy loading)."""
        if self._model is not None:
            return

        if not self._available:
            raise ImportError(
                "deepCR is not installed. Install with: pip install deepCR"
            )

        import deepCR

        # Get model name for instrument
        model_name = self.PRETRAINED_MODELS.get(
            self.instrument.value,
            self.PRETRAINED_MODELS['generic']
        )

        # Load pre-trained model
        # deepCR will automatically use GPU if available and use_gpu=True
        self._model = deepCR.deepCR(
            mask=model_name,
            inpaint='ACS-WFC-F606W-2-32' if self.inpaint else None,
            device='GPU' if self.use_gpu else 'CPU',
        )

    @property
    def model(self):
        """Get the loaded model (loads on first access)."""
        self._load_model()
        return self._model

    def clean(
        self,
        image: NDArray,
        threshold: float | None = None,
        segment: bool = True,
        patch_size: int = 256,
    ) -> CRCleaningResult:
        """Clean cosmic rays from an image.

        Parameters
        ----------
        image : NDArray
            2D image array to clean
        threshold : float, optional
            Override default threshold for this call
        segment : bool, optional
            Process image in segments to reduce memory usage. Default True.
        patch_size : int, optional
            Size of patches when segmenting. Default 256.

        Returns
        -------
        CRCleaningResult
            Contains cleaned image, CR mask, and statistics
        """
        if not self._available:
            # Return original if deepCR not available
            return CRCleaningResult(
                cleaned_image=image.copy(),
                cr_mask=np.zeros_like(image, dtype=bool),
                n_pixels_affected=0,
                fraction_affected=0.0,
                method='none (deepCR not available)',
            )

        self._load_model()

        thresh = threshold if threshold is not None else self.threshold

        # Run deepCR
        if segment and min(image.shape) > patch_size:
            # Process in patches to reduce memory usage
            cr_mask, cleaned = self._model.clean(
                image,
                threshold=thresh,
                segment=True,
                patch=patch_size,
            )
        else:
            cr_mask, cleaned = self._model.clean(
                image,
                threshold=thresh,
            )

        # Convert mask to boolean
        cr_mask = cr_mask.astype(bool)
        n_affected = np.sum(cr_mask)
        frac_affected = n_affected / image.size

        return CRCleaningResult(
            cleaned_image=cleaned,
            cr_mask=cr_mask,
            n_pixels_affected=int(n_affected),
            fraction_affected=float(frac_affected),
            method=f'deepCR (threshold={thresh:.2f})',
        )

    def detect_only(
        self,
        image: NDArray,
        threshold: float | None = None,
    ) -> NDArray:
        """Detect cosmic rays without inpainting.

        Parameters
        ----------
        image : NDArray
            2D image array
        threshold : float, optional
            Detection threshold

        Returns
        -------
        NDArray
            Boolean mask where True = cosmic ray pixel
        """
        if not self._available:
            return np.zeros_like(image, dtype=bool)

        self._load_model()
        thresh = threshold if threshold is not None else self.threshold

        # Get mask only
        cr_mask = self._model.clean(image, threshold=thresh, inpaint=False)
        if isinstance(cr_mask, tuple):
            cr_mask = cr_mask[0]

        return cr_mask.astype(bool)


class LACosmic:
    """LACosmic cosmic ray removal (fallback when deepCR unavailable).

    Uses the astroscrappy implementation of the LACosmic algorithm
    (van Dokkum 2001). This is a traditional edge-detection based
    method that works well for bright cosmic rays but may miss
    faint ones or create artifacts around bright stars.

    Parameters
    ----------
    sigclip : float
        Laplacian-to-noise limit for cosmic ray detection. Default 4.5.
    sigfrac : float
        Fractional detection limit for neighboring pixels. Default 0.3.
    objlim : float
        Minimum contrast between CR and underlying object. Default 5.0.
    niter : int
        Number of iterations. Default 4.
    """

    def __init__(
        self,
        sigclip: float = 4.5,
        sigfrac: float = 0.3,
        objlim: float = 5.0,
        niter: int = 4,
    ):
        self.sigclip = sigclip
        self.sigfrac = sigfrac
        self.objlim = objlim
        self.niter = niter
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if astroscrappy is available."""
        try:
            import astroscrappy
            return True
        except ImportError:
            return False

    def clean(
        self,
        image: NDArray,
        gain: float = 1.0,
        readnoise: float = 5.0,
    ) -> CRCleaningResult:
        """Clean cosmic rays using LACosmic.

        Parameters
        ----------
        image : NDArray
            2D image array
        gain : float
            Detector gain in e-/ADU
        readnoise : float
            Read noise in electrons

        Returns
        -------
        CRCleaningResult
            Cleaned image and CR mask
        """
        if not self._available:
            return CRCleaningResult(
                cleaned_image=image.copy(),
                cr_mask=np.zeros_like(image, dtype=bool),
                n_pixels_affected=0,
                fraction_affected=0.0,
                method='none (astroscrappy not available)',
            )

        import astroscrappy

        cr_mask, cleaned = astroscrappy.detect_cosmics(
            image,
            gain=gain,
            readnoise=readnoise,
            sigclip=self.sigclip,
            sigfrac=self.sigfrac,
            objlim=self.objlim,
            niter=self.niter,
        )

        n_affected = np.sum(cr_mask)
        frac_affected = n_affected / image.size

        return CRCleaningResult(
            cleaned_image=cleaned,
            cr_mask=cr_mask,
            n_pixels_affected=int(n_affected),
            fraction_affected=float(frac_affected),
            method='LACosmic',
        )


def clean_cosmic_rays(
    image: NDArray,
    instrument: str = 'generic',
    threshold: float | None = None,
    use_deepcr: bool = True,
    use_gpu: bool = True,
    fallback_to_lacosmic: bool = True,
    verbose: bool = True,
) -> tuple[NDArray, NDArray]:
    """Clean cosmic rays from an image.

    This is the main entry point for cosmic ray removal. It uses deepCR
    by default but can fall back to LACosmic if deepCR is unavailable.

    Parameters
    ----------
    image : NDArray
        2D image array to clean
    instrument : str
        HST instrument name for optimized thresholds.
        Options: 'ACS-WFC', 'WFC3-UVIS', 'WFC3-IR', 'WFPC2', 'generic'
    threshold : float, optional
        Detection threshold (0-1). Uses instrument default if not specified.
    use_deepcr : bool
        Use deepCR neural network. Default True.
    use_gpu : bool
        Use GPU acceleration for deepCR. Default True.
    fallback_to_lacosmic : bool
        Fall back to LACosmic if deepCR unavailable. Default True.
    verbose : bool
        Print progress information. Default True.

    Returns
    -------
    cleaned_image : NDArray
        Image with cosmic rays removed
    cr_mask : NDArray
        Boolean mask where True = detected cosmic ray pixel

    Examples
    --------
    >>> # Clean HDF image (WFPC2)
    >>> cleaned, mask = clean_cosmic_rays(image, instrument='WFPC2')
    >>> print(f"Cleaned {mask.sum()} pixels")

    >>> # Clean with custom threshold
    >>> cleaned, mask = clean_cosmic_rays(image, threshold=0.3)
    """
    # Try deepCR first
    if use_deepcr:
        cleaner = DeepCRCleaner(
            instrument=instrument,
            threshold=threshold,
            use_gpu=use_gpu,
        )

        if cleaner._available:
            if verbose:
                print(f"    Cleaning cosmic rays with deepCR (instrument={instrument})...")
            result = cleaner.clean(image)
            if verbose:
                print(f"    Removed {result.n_pixels_affected} CR pixels "
                      f"({100*result.fraction_affected:.3f}% of image)")
            return result.cleaned_image, result.cr_mask

    # Fall back to LACosmic
    if fallback_to_lacosmic:
        if verbose:
            print("    deepCR not available, falling back to LACosmic...")
        lacosmic = LACosmic()
        if lacosmic._available:
            result = lacosmic.clean(image)
            if verbose:
                print(f"    Removed {result.n_pixels_affected} CR pixels "
                      f"({100*result.fraction_affected:.3f}% of image)")
            return result.cleaned_image, result.cr_mask

    # No CR removal available
    if verbose:
        print("    Warning: No cosmic ray removal method available")
    return image.copy(), np.zeros_like(image, dtype=bool)


def clean_image_stack(
    images: list[NDArray],
    instrument: str = 'generic',
    threshold: float | None = None,
    combine_masks: bool = True,
    verbose: bool = True,
) -> tuple[list[NDArray], NDArray]:
    """Clean cosmic rays from a stack of images.

    Parameters
    ----------
    images : list of NDArray
        List of 2D images to clean
    instrument : str
        HST instrument name
    threshold : float, optional
        Detection threshold
    combine_masks : bool
        Return combined mask of all CRs. Default True.
    verbose : bool
        Print progress. Default True.

    Returns
    -------
    cleaned_images : list of NDArray
        Cleaned images
    combined_mask : NDArray
        Combined CR mask (union of all individual masks)
    """
    cleaned_images = []
    combined_mask = None

    for i, image in enumerate(images):
        if verbose:
            print(f"    Processing image {i+1}/{len(images)}...")
        cleaned, mask = clean_cosmic_rays(
            image,
            instrument=instrument,
            threshold=threshold,
            verbose=False,
        )
        cleaned_images.append(cleaned)

        if combine_masks:
            if combined_mask is None:
                combined_mask = mask.copy()
            else:
                combined_mask |= mask

    if combined_mask is None:
        combined_mask = np.zeros_like(images[0], dtype=bool)

    total_pixels = sum(m.sum() for m in [combined_mask])
    if verbose:
        print(f"    Total CR pixels across stack: {total_pixels}")

    return cleaned_images, combined_mask
