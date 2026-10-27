"""Turning a CT volume into RGB slice images for an optional vision-language second reader.

Recipe from Google's ``high_dimensional_ct_hugging_face`` CT notebook:
each axial slice becomes an 8-bit RGB image whose three channels are three CT windows,

    red   = wide window        -1024 .. 1024 HU
    green = soft-tissue window  -135 ..  215 HU
    blue  = brain window           0 ..   80 HU

and the slices are given in order, each followed by a ``SLICE n`` text label.
"""
from __future__ import annotations

import numpy as np


def window_rgb(hu_slice: np.ndarray, windows) -> np.ndarray:
    """(H, W) HU -> (H, W, 3) uint8, one window per channel."""
    chans = []
    for lo, hi in windows:
        x = (np.clip(hu_slice.astype(np.float32), lo, hi) - lo) / float(hi - lo) * 255.0
        chans.append(np.round(x))
    return np.stack(chans, axis=-1).astype(np.uint8)


def select_slices(z_lo: int, z_hi: int, n: int) -> list[int]:
    """n axial indices spread uniformly over [z_lo, z_hi], returned superior -> inferior.

    In the radiological view used across the project, z grows towards the head, so the
    list is in descending z (the order a radiologist scrolls through the liver).
    """
    z_lo, z_hi = int(min(z_lo, z_hi)), int(max(z_lo, z_hi))
    count = z_hi - z_lo + 1
    if count <= n:
        zs = list(range(z_lo, z_hi + 1))
    else:
        zs = sorted({int(round(z)) for z in np.linspace(z_lo, z_hi, n)})
    return sorted(zs, reverse=True)


def liver_z_range(analysis: dict) -> tuple[int, int] | None:
    """Where to sample: the predicted liver extent (falls back to the segmentation slab)."""
    m = analysis.get("measurements", {})
    if m.get("liver_slice_range"):
        return tuple(m["liver_slice_range"])  # type: ignore[return-value]
    seg = analysis.get("segmentation", {})
    if seg.get("found_liver") and seg.get("z_range"):
        z0, z1 = seg["z_range"]
        return int(z0), int(z1) - 1
    return None


def thumbnail(rgb: np.ndarray, size: int) -> np.ndarray:
    import cv2

    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
