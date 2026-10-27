"""Optional vision-language second reader - off by default.

The project uses one pretrained model (NVIDIA VISTA3D). This hook stays for research:
``vlm.backend``: ``none`` (default) | ``plugin`` (``vlm.plugin: "package.module:ClassName"``, a
:class:`radagent.vlm.base.VLMReader` constructed as ``ClassName(cfg)``). The automated tests use it
with a stand-in reader.
"""
from __future__ import annotations

import hashlib
import json


def vlm_enabled(cfg: dict) -> bool:
    return cfg.get("vlm", {}).get("backend", "none") not in (None, "none", "")


def vlm_signature(cfg: dict) -> str:
    """Identifies a cached read: reader and slice preparation."""
    v = cfg["vlm"]
    key = {"backend": v["backend"], "plugin": v.get("plugin"), "n": v["n_slices"], "windows": v["windows"],
           "describe": bool(v.get("describe", True))}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def load_vlm(cfg: dict):
    backend = cfg["vlm"]["backend"]
    if backend == "plugin":
        from radagent.plugins import load_class

        return load_class(cfg["vlm"]["plugin"])(cfg)
    raise ValueError(f"unknown vlm backend {backend!r} (use none or plugin)")
