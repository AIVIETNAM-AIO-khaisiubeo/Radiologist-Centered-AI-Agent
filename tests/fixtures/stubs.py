"""Stand-in plugins for automated tests (no GPU, no PyTorch, no Hugging Face download).

They implement the same adapter interfaces as the real models and are loaded through
the plugin mechanism, e.g. ``segmenter: {backend: plugin, plugin: "tests.fixtures.stubs:ThresholdSegmenter"}``.
Their outputs are deliberately crude - they exist to exercise the pipeline, not to segment.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy import ndimage as ndi

from radagent.models.base import N_CLASSES, Segmenter
from radagent.vlm.base import VLMReader


def _gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2)


class ThresholdSegmenter(Segmenter):
    """Intensity-likelihood 'segmenter' on the liver window (-100..250 HU -> 0..255)."""

    name = "threshold-stub"
    version = "test"

    def __init__(self, cfg: dict | None = None):
        lo, hi = (cfg or {}).get("preprocess", {}).get("hu_window", [-100, 250])
        self.u8 = lambda hu: (hu - lo) / (hi - lo) * 255.0

    @property
    def n_passes(self) -> int:
        return 2

    def predict_batch(self, stacks_u8, passes=None, roi=None, coarse=False):
        P = 1 if passes == 1 else self.n_passes
        B, _, S0, S1 = stacks_u8.shape
        out = np.zeros((P, B, N_CLASSES, S0, S1), dtype=np.float32)
        for b in range(B):
            x = ndi.gaussian_filter(stacks_u8[b].astype(np.float32).mean(0), 1.5)
            for p in range(P):
                xx = x + (p - 0.5) * 2.0
                liver = _gauss(xx, self.u8(60), 14)
                tum = 0.8 * _gauss(xx, self.u8(30), 10) * (ndi.uniform_filter(liver, 9) > 0.3)
                bg = 0.25 + 0 * xx
                tot = liver + tum + bg
                out[p, b, 0], out[p, b, 1], out[p, b, 2] = bg / tot, liver / tot, tum / tot
        return out


class StubVLM(VLMReader):
    """Deterministic pseudo-reader: 'sees' dark spots in the soft-tissue channel."""

    name = "stub-vlm"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def read(self, images, describe=True):
        g = np.stack([im[..., 1].astype(np.float32) for im in images])
        dark = float(((g > 90) & (g < 125)).mean())
        h = int(hashlib.sha1(g.tobytes()[:4096]).hexdigest()[:6], 16) / 0xFFFFFF
        p = float(np.clip(0.15 + 8.0 * dark + 0.3 * (h - 0.5), 0.02, 0.98))
        text = (f"Stub read of {len(images)} slices. "
                + ("Hypodense focal lesion suspected. Final Answer: yes" if p >= 0.5 else
                   "No focal lesion. Final Answer: no")) if describe else ""
        return {"p_yes": p, "answer_mass": 0.99, "text": text, "n_images": len(images), "seconds": 0.0}

    def describe_model(self):
        return {"backend": self.name, "model_id": "stub-vlm (tests only)", "precision": "n/a"}
