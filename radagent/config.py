"""Configuration: built-in defaults deep-merged with a YAML file.

Every experiment is driven by one YAML file (see ``configs/``). Anything not set in
the YAML falls back to ``DEFAULTS`` below, so a config only needs to state what it
changes.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "run_name": "lits",
    "output_root": "outputs",
    "seed": 13,
    # ------------------------------------------------------------------ data
    "data": {
        "source": "lits",                  # LiTS17 (Kaggle mirror) - the only source
        # Folders that contain volume-*.nii / segmentation-*.nii. Empty -> downloaded
        # (or found in the local cache) with kagglehub from the two Kaggle datasets below.
        "lits_dirs": [],
        "kaggle_datasets": ["andrewmvd/liver-tumor-segmentation",          # masks + volume_pt1..pt5
                            "andrewmvd/liver-tumor-segmentation-part-2"],  # remaining volumes
        "max_cases": None,                 # optional cap: proportional subset of every split
        # Nothing is trained in this project, so every case is either used to calibrate the
        # model's case score or to evaluate it.
        "split_fractions": {"calib": 0.5, "test": 0.5},
    },
    # ------------------------------------------------------------ preprocess
    "preprocess": {
        "hu_window": [-100, 250],          # soft-tissue / liver window (HU) for the viewer and 2.5D plugins
        "img_size": 256,                   # in-plane size of the stored model-resolution maps (LiTS slices are 512)
    },
    # ------------------------------------------------------------- segmenter
    "segmenter": {
        # vista3d: NVIDIA VISTA3D (MONAI), pretrained, loaded from Hugging Face - nothing is trained here
        "backend": "vista3d",             # vista3d | plugin
        "plugin": None,                    # backend: plugin -> "package.module:ClassName" (see radagent/models/__init__.py)
        "hf_repo": "MONAI/VISTA3D-HF",
        "hf_file": "vista3d_pretrained_model/model.safetensors",
        "hf_revision": None,               # pin a commit for exact reproducibility
        "weights": None,                   # or a local path to model.safetensors (offline machines)
        "resample_mm": 1.5,                # VISTA3D works on 1.5 mm isotropic voxels
        "roi_size": [128, 128, 128],       # sliding-window patch
        "overlap": 0.25,
        "sw_batch_size": 1,
        "amp": True,                       # mixed precision on GPU
        "device": "auto",
    },
    # ------------------------------------------------------- vision-language
    # Optional second reader. Off: the project uses one pretrained model (VISTA3D). The hook is kept
    # for research (``backend: plugin``) and is exercised by the automated tests.
    "vlm": {
        "backend": "none",                # none | plugin
        "plugin": None,                    # backend: plugin -> "package.module:ClassName"
        "n_slices": 16,                    # axial slices sampled uniformly through the predicted liver
        "windows": [[-1024, 1024], [-135, 215], [0, 80]],   # RGB channels: wide, soft tissue, narrow
        "describe": True,                  # also ask for a short free-text read (shown as unverified)
        "thumb_size": 160,                 # thumbnails of the slices it saw, for the website
    },
    # ------------------------------------------------------------- inference
    "inference": {
        "tta_flip": True,                  # second pass on the left-right flipped CT (uncertainty)
        "batch_size": 32,                  # slices per batch for 2.5D plugins
        "min_lesion_ml": 0.1,              # smallest lesion reported / looked for by the case score
        "min_lesion_peak_prob": 0.7,       # a lesion candidate must reach this tumour probability somewhere
        "viewer_size": 192,                # stored slice size for the website
        "viewer_max_slices": 64,
        "eval_splits": ["calib", "test"],
    },
    # ------------------------------------------------------------ confidence
    "confidence": {
        # auto: cross-validated choice on the calib split between identity, temperature and
        # Platt scaling of the segmenter score (plus vlm / fused when a second reader is enabled).
        "method": "auto",
        "n_bins": 10,
        "target_min_tumour_ml": 0.0,       # reference "tumour present" if volume > this
    },
    # --------------------------------------------------------------- quality
    "quality": {
        "liver_ml_range": [300.0, 6000.0],
        "spacing_mm_range": [0.3, 8.0],
        "min_slices": 16,
        "max_tumour_outside_liver_frac": 0.25,
        "min_outside_ml": 1.0,             # ... ignored if the volume outside the liver is tiny
    },
    # ---------------------------------------------------------------- triage
    "triage": {
        "suggest_threshold": 0.5,
        "urgent_probability": 0.85,
        "urgent_volume_ml": 5.0,
        "urgent_diameter_mm": 20.0,
        "uncertain_flag": 0.85,            # bits of predictive entropy (p between ~0.28 and ~0.72)
        "volume_ref_ml": 50.0,
        "disagreement_bonus": 0.10,        # priority added when the segmenter and a second reader disagree
        "weights": {"probability": 0.60, "volume": 0.25, "uncertainty": 0.15},
    },
    # ---------------------------------------------------------------- policy
    "policy": {
        "universal_threshold": 0.5,
        "threshold_grid": [round(0.10 + 0.05 * i, 2) for i in range(17)],
        "defer_grid": [None, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        "w_fn": 2.0,                       # a missed tumour costs twice a false alarm
        "w_fp": 1.0,
        "burden_weight": 0.02,             # loss per review action
        "action_cost": {"accept": 1, "reject": 2, "edit": 3, "manual": 4},
        "min_feedback": 10,                # personalise only after this many reads
        "min_improvement": 0.005,          # ... and only if it helps by at least this much
        "improvement_se": 1.0,             # ... and by more than 1 standard error of the paired difference
        "shrink_rho": 0.02,                # pull towards the universal policy
        "trust_prior": {"alpha_mean": 0.0, "alpha_sd": 1.5, "beta_mean": 4.0, "beta_sd": 3.0},
    },
    # --------------------------------------------------------------- readers
    "readers": {
        "seeds": 30,
        "learning_curve_n": [0, 5, 10, 15, 20, 26],
        "profiles": {
            "reader_a": {
                "name": "Reader A - over-reliant",
                "description": "Accepts most AI suggestions, switches to the AI easily.",
                "sens_a": 0.8, "sens_b": 1.5, "sens_c": 0.8, "fpr": 0.12,
                "trust_k": 8.0, "trust_tau": 0.55,
                "lapse": 0.02, "edit_poor": 0.2, "edit_noise": 0.02,
            },
            "reader_b": {
                "name": "Reader B - sceptical",
                "description": "Conservative caller; rejects many suggestions, rarely switches.",
                "sens_a": 0.9, "sens_b": 1.8, "sens_c": 0.8, "fpr": 0.04,
                "trust_k": 6.0, "trust_tau": 0.97,
                "lapse": 0.02, "edit_poor": 0.8, "edit_noise": 0.15,
            },
            "reader_c": {
                "name": "Reader C - inconsistent",
                "description": "Noisy, inconsistent decisions and corrections.",
                "sens_a": 0.7, "sens_b": 1.0, "sens_c": 0.6, "fpr": 0.15,
                "trust_k": 1.5, "trust_tau": 0.6,
                "lapse": 0.15, "edit_poor": 0.5, "edit_noise": 0.3,
            },
        },
    },
    # ----------------------------------------------------------------- agent
    "agent": {
        "planner": "rule",                 # rule | llm
        "max_steps": 12,
        "llm": {
            "base_url": "http://localhost:11434/v1",   # any OpenAI-compatible server (Ollama, vLLM, LM Studio)
            "model": "qwen2.5:7b-instruct",
            "api_key_env": "LLM_API_KEY",
            "timeout_s": 60,
            "temperature": 0.0,
        },
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, user)
        cfg["_config_path"] = str(path)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    if cfg["inference"].get("viewer_size") is None:
        cfg["inference"]["viewer_size"] = cfg["preprocess"]["img_size"]
    return cfg


def run_dir(cfg: dict) -> Path:
    return Path(cfg["output_root"]) / cfg["run_name"]


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(json.loads(json.dumps(clean)), f, sort_keys=False)
