"""Segmenter interfaces: every backend is a *tool* that turns a CT into class probabilities.

The agent never depends on a particular network. Two kinds of backend exist:

* :class:`VolumeSegmenter` (3D models such as VISTA3D, the project's default) implements
  ``predict_volume(view_hu, spacing, size) -> (P, Z, 3, size, size)``;
* :class:`Segmenter` (2.5D slice models, used by the test stand-ins) implements
  ``predict_batch(stacks_u8) -> (P, B, 3, S, S)``.

``P`` is the number of stochastic passes (test-time augmentation views); the spread across
passes is the epistemic-uncertainty signal. Classes are 0 background, 1 liver, 2 tumour.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

N_CLASSES = 3


class Segmenter(ABC):
    name: str = "base"
    version: str = "0"

    @property
    @abstractmethod
    def n_passes(self) -> int:
        ...

    @abstractmethod
    def predict_batch(self, stacks_u8: np.ndarray, passes: int | None = None, roi: tuple | None = None,
                      coarse: bool = False) -> np.ndarray:
        """stacks_u8: (B, 3, S, S) uint8 -> probabilities (P, B, 3, S, S) float32.

        ``passes=1`` asks for a single deterministic pass. ``roi=(r0, r1, c0, c1)`` and
        ``coarse=True`` are optional speed hints a backend may ignore (pixels outside the
        ROI may be returned as background).
        """

    def describe(self) -> dict:
        return {"backend": self.name, "version": self.version, "passes": self.n_passes}


class VolumeSegmenter(ABC):
    """A 3D model that segments a whole CT volume at once."""

    name: str = "volume"
    version: str = "0"

    @property
    @abstractmethod
    def n_passes(self) -> int:
        ...

    @abstractmethod
    def predict_volume(self, view_hu: np.ndarray, spacing: tuple, size: int) -> np.ndarray:
        """CT view ``V[z, row, col]`` in HU with spacing ``(dz, drow, dcol)`` mm ->
        probabilities ``(P, Z, 3, size, size)`` (in-plane resized to ``size``)."""

    def describe(self) -> dict:
        return {"backend": self.name, "version": self.version, "passes": self.n_passes}
