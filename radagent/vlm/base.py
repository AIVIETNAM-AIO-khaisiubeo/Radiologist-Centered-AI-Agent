"""Interface of a vision-language second reader.

An adapter receives the prepared slices (RGB uint8, superior -> inferior) and returns

    {"p_yes": P(model answers "yes" to "is there a liver tumour?"),   # from token probabilities
     "answer_mass": share of the next-token probability on yes/no words (format adherence),
     "text": optional free-text read (unverified),
     "seconds": wall time}

``p_yes`` is a score, not a calibrated probability: :mod:`radagent.calibration` calibrates
it on the calib split, alone and fused with the segmenter's score.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VLMReader(ABC):
    name: str = "vlm"

    @abstractmethod
    def read(self, images: list[np.ndarray], describe: bool = True) -> dict:
        ...

    def describe_model(self) -> dict:
        return {"backend": self.name}
