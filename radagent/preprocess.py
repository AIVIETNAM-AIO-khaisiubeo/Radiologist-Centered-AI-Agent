"""Intensity windowing, resizing and 2.5D stacking (viewer images and 2.5D plugin segmenters)."""
from __future__ import annotations

import cv2
import numpy as np


def window_u8(hu: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip HU to [lo, hi] and map linearly to 0..255 (uint8)."""
    x = (np.clip(hu.astype(np.float32), lo, hi) - lo) / float(hi - lo) * 255.0
    return x.astype(np.uint8)


def resize_stack(vol: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    """Resize every axial slice of ``vol[z, row, col]`` to ``size x size``."""
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    out = np.empty((vol.shape[0], size, size), dtype=vol.dtype)
    for z in range(vol.shape[0]):
        out[z] = cv2.resize(np.ascontiguousarray(vol[z]), (size, size), interpolation=interp)
    return out


def upsample_slice(img: np.ndarray, shape_hw: tuple[int, int], nearest: bool = False) -> np.ndarray:
    h, w = shape_hw
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(np.ascontiguousarray(img), (w, h), interpolation=interp)


def preprocess_volume(view_hu: np.ndarray, window: tuple[float, float], size: int) -> np.ndarray:
    """CT view (HU) -> windowed uint8 slices at model resolution, shape (Z, size, size)."""
    lo, hi = window
    return resize_stack(window_u8(view_hu, lo, hi), size)


def stacks_25d(img_u8: np.ndarray, z_indices) -> np.ndarray:
    """Return (len(z), 3, S, S) uint8 stacks of slices (z-1, z, z+1), edge-clamped."""
    Z = img_u8.shape[0]
    out = np.empty((len(z_indices), 3) + img_u8.shape[1:], dtype=np.uint8)
    for i, z in enumerate(z_indices):
        out[i, 0] = img_u8[max(0, z - 1)]
        out[i, 1] = img_u8[z]
        out[i, 2] = img_u8[min(Z - 1, z + 1)]
    return out


def to_model_range(x_u8: np.ndarray) -> np.ndarray:
    """uint8 -> float32 in [-1, 1] (the notebook's normalisation)."""
    return (x_u8.astype(np.float32) / 255.0 - 0.5) / 0.5
