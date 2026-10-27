"""Load adapter classes by dotted path: ``"package.module:ClassName"``.

This is how outside models are plugged in without editing the pipeline. A segmentation
adapter implements :class:`radagent.models.base.Segmenter`; a vision-language adapter
implements :class:`radagent.vlm.base.VLMReader`. Point the config at it::

    segmenter: {backend: plugin, plugin: "my_models.medsam:MedSAMSegmenter"}
    vlm:       {backend: plugin, plugin: "my_models.llava:LlavaMedReader"}

The class is constructed with the whole config dict: ``ClassName(cfg)``.
"""
from __future__ import annotations

import importlib


def load_class(spec: str):
    if not spec or ":" not in spec:
        raise ValueError(f"plugin must look like 'package.module:ClassName', got {spec!r}")
    module, name = spec.split(":", 1)
    mod = importlib.import_module(module)
    try:
        return getattr(mod, name)
    except AttributeError as e:
        raise ImportError(f"{module} has no attribute {name!r}") from e
