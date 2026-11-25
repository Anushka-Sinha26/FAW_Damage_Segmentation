"""Visible leaf (vegetation) mask for the damage-% denominator.

Method (unsupervised, no training): CIELAB a* channel (green < neutral < magenta). The FLIR E8
visual images have magenta/purple-tinted soil and green leaves, so a per-image Otsu threshold
on a* separates vegetation (low a*) from soil (high a*), including bluish-green leaves that green
indices (ExG-ExR, HSV hue) miss. Morphological opening then closing (5x5 ellipse) removes
speckle. Parameters fixed on cross-validation images only (visual check), never on test images.
"""
import cv2
import numpy as np

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def leaf_mask(rgb_u8):
    a = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)[..., 1]
    t, _ = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (a < t).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, KERNEL)
    return m > 0, int(t)


def damage_pct_of_leaf(damage, leaf):
    """Damaged pixels as % of visible leaf area; damaged tissue/holes count as leaf (denominator = leaf OR damage)."""
    den = (leaf | damage).sum()
    return 100.0 * damage.sum() / den if den else 0.0
