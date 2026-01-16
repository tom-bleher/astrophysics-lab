"""Simple ML-based star/galaxy classifier using morphological features."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# HST WFPC2 PSF FWHM in pixels (0.1" at 0.04"/pix)
PSF_FWHM_PIX = 2.25

# Features required for classification (psf_ratio is computed from half_light_radius)
FEATURES = ["concentration", "gini", "half_light_radius", "sersic_n", "psf_ratio"]
REQUIRED_COLUMNS = ["concentration", "gini", "half_light_radius", "sersic_n"]


def classify_stars(catalog: pd.DataFrame, train_col: str = "in_star_mask") -> pd.Series:
    """Train on known stars (e.g., chip3 mask), predict P(star) for all sources.

    Parameters
    ----------
    catalog : pd.DataFrame
        Source catalog with morphological columns
    train_col : str
        Column name containing training labels (True = star)

    Returns
    -------
    pd.Series
        Probability of being a star for each source

    Raises
    ------
    KeyError
        If required morphological columns are missing
    """
    # Check for required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in catalog.columns]
    if missing:
        raise KeyError(f"Missing required morphology columns: {missing}")

    # Add PSF ratio feature: stars have r_half ~ PSF, galaxies have r_half >> PSF
    cat = catalog.copy()
    cat["psf_ratio"] = cat["half_light_radius"] / PSF_FWHM_PIX

    X = cat[FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
    y = cat[train_col].fillna(False).astype(bool)

    # Need both classes to train
    if y.sum() < 2 or (~y).sum() < 2:
        print("  Star classifier: insufficient training data, skipping")
        return pd.Series(np.nan, index=catalog.index)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    proba = clf.predict_proba(X)[:, 1]
    print(f"  Star classifier: trained on {y.sum()} stars, {(~y).sum()} galaxies")
    print(f"  Feature importance: {dict(zip(FEATURES, clf.feature_importances_.round(3)))}")

    return pd.Series(proba, index=catalog.index, name="star_prob_ml")
