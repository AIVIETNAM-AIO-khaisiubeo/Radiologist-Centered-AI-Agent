"""Segmenter registry. Torch and MONAI are imported lazily so the website runs without them.

Backends
--------
``vista3d``  NVIDIA VISTA3D (MONAI), pretrained 3D CT foundation model - the project's only model.
             Loaded from Hugging Face ``MONAI/VISTA3D-HF``; nothing is trained in this project.
``plugin``   any adapter class, ``segmenter.plugin: "package.module:ClassName"``, implementing
             :class:`radagent.models.base.VolumeSegmenter` or :class:`radagent.models.base.Segmenter`
             and constructed as ``ClassName(cfg)`` (used by the automated tests).
"""
from __future__ import annotations

from pathlib import Path

from radagent.config import run_dir

BACKENDS = ("vista3d", "plugin")


def model_card_path(cfg: dict) -> Path:
    return run_dir(cfg) / "model" / "model_card.json"


def load_segmenter(cfg: dict):
    backend = cfg["segmenter"]["backend"]
    if backend == "plugin":
        from radagent.plugins import load_class

        return load_class(cfg["segmenter"]["plugin"])(cfg)
    if backend == "vista3d":
        from radagent.models.vista3d import Vista3DSegmenter

        return Vista3DSegmenter(cfg)
    raise ValueError(f"unknown segmenter backend {backend!r} (use vista3d or plugin)")
