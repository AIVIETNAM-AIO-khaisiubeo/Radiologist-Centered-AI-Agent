"""Uncertainty-aware inference on one CT volume.

3D models (VISTA3D, the default) segment the whole volume in every pass (plain and
left-right flipped); the liver's z-range is then taken from the mean prediction.
2.5D slice models (the test stand-ins) run in two stages: one deterministic pass over every
slice finds the liver's z-range, then all stochastic passes run on the liver slab +/- 10 mm.

Outputs (model resolution, liver slab only): mean class probabilities, every pass's
tumour map, post-processed liver/tumour masks, voxel entropy, and the case-level
*raw* tumour-present score with its spread across passes.

Case-level raw score
--------------------
``p_raw`` = mean of the ``k`` highest tumour probabilities inside the (slightly
dilated) liver, where ``k`` voxels correspond to ``min_lesion_ml`` (0.1 ml by
default). It answers "is there a lesion of at least ~0.1 ml the model believes in?"
and is robust to single noisy voxels. It is a *score*, not a calibrated probability -
:mod:`radagent.calibration` turns it into one.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from radagent.preprocess import preprocess_volume, stacks_25d


def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask.astype(bool)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def fill_holes_slicewise(mask: np.ndarray) -> np.ndarray:
    out = np.empty_like(mask, dtype=bool)
    for z in range(mask.shape[0]):
        out[z] = ndi.binary_fill_holes(mask[z])
    return out


def remove_small(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    if min_voxels <= 1 or not mask.any():
        return mask
    lab, n = ndi.label(mask)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_voxels
    return keep[lab]


def filter_lesions(mask: np.ndarray, prob: np.ndarray, min_voxels: int, min_peak: float) -> np.ndarray:
    """Keep connected components that are large enough AND reach a confident peak probability."""
    if not mask.any():
        return mask
    lab, n = ndi.label(mask)
    idx = np.arange(1, n + 1)
    sizes = ndi.sum(mask, lab, idx)
    peaks = ndi.maximum(prob, lab, idx)
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = (sizes >= min_voxels) & (peaks >= min_peak)
    return keep[lab]


def topk_mean(values: np.ndarray, k: int) -> float:
    if values.size == 0:
        return 0.0
    k = min(k, values.size)
    return float(np.partition(values.ravel(), values.size - k)[values.size - k:].mean())


@dataclass
class SegOutput:
    found_liver: bool
    z0: int = 0
    z1: int = 0
    img_size: int = 0
    eff_spacing: tuple = (1.0, 1.0, 1.0)
    mean: np.ndarray | None = None          # (n, 3, S, S)
    pass_tumour: np.ndarray | None = None   # (P, n, S, S) float16
    liver: np.ndarray | None = None         # (n, S, S) bool, includes tumour
    tumour: np.ndarray | None = None        # (n, S, S) bool
    entropy: np.ndarray | None = None       # (n, S, S) normalised 0..1
    p_raw: float = 0.0
    pass_scores: list = field(default_factory=list)
    k_voxels: int = 1
    n_passes: int = 0
    timings: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"found_liver": self.found_liver, "z_range": [int(self.z0), int(self.z1)],
                "img_size": self.img_size, "n_passes": self.n_passes, "k_voxels": self.k_voxels,
                "tumour_score_raw": round(self.p_raw, 5),
                "pass_scores_raw": [round(float(s), 5) for s in self.pass_scores],
                "timings_s": {k: round(v, 2) for k, v in self.timings.items()}}


def segment_volume(view_hu: np.ndarray, spacing: tuple, segmenter, cfg: dict) -> SegOutput:
    if hasattr(segmenter, "predict_volume"):
        return _segment_volume_3d(view_hu, spacing, segmenter, cfg)
    return _segment_volume_25d(view_hu, spacing, segmenter, cfg)


def _eff_spacing(view_hu: np.ndarray, spacing: tuple, S: int) -> tuple:
    Z, H, W = view_hu.shape
    return (float(spacing[0]), float(spacing[1]) * H / S, float(spacing[2]) * W / S)


def _slab(fg: np.ndarray, eff: tuple) -> tuple[int, int]:
    zidx = np.where(fg.any(axis=(1, 2)))[0]
    m = max(2, int(math.ceil(10.0 / eff[0])))
    return max(0, int(zidx.min()) - m), min(fg.shape[0], int(zidx.max()) + m + 1)


def _segment_volume_3d(view_hu: np.ndarray, spacing: tuple, segmenter, cfg: dict) -> SegOutput:
    S = cfg["preprocess"]["img_size"]
    eff = _eff_spacing(view_hu, spacing, S)
    t0 = time.time()
    probs = segmenter.predict_volume(view_hu, spacing, S)              # (P, Z, 3, S, S)
    t_inf = time.time() - t0
    fg_prob = (probs[:, :, 1].astype(np.float32) + probs[:, :, 2].astype(np.float32)).mean(axis=0)
    fg = largest_component(fg_prob > 0.5)
    del fg_prob
    if not fg.any():
        return SegOutput(found_liver=False, img_size=S, eff_spacing=eff, n_passes=int(probs.shape[0]),
                         timings={"infer": t_inf})
    z0, z1 = _slab(fg, eff)
    mean = probs[:, z0:z1].astype(np.float32).mean(axis=0)
    passes = probs[:, z0:z1, 2].astype(np.float16)
    del probs
    return _finish(mean, passes, z0, z1, S, eff, cfg, {"infer": t_inf})


def _segment_volume_25d(view_hu: np.ndarray, spacing: tuple, segmenter, cfg: dict) -> SegOutput:
    S = cfg["preprocess"]["img_size"]
    bs = int(cfg["inference"]["batch_size"])
    Z, H, W = view_hu.shape
    eff = _eff_spacing(view_hu, spacing, S)
    t0 = time.time()
    img = preprocess_volume(view_hu, tuple(cfg["preprocess"]["hu_window"]), S)
    t_pre = time.time() - t0

    # ---- stage 1: localisation
    t0 = time.time()
    fg_prob = np.zeros((Z, S, S), dtype=np.float32)
    for b0 in range(0, Z, bs):
        zs = list(range(b0, min(Z, b0 + bs)))
        pr = segmenter.predict_batch(stacks_25d(img, zs), passes=1, coarse=True)[0]
        fg_prob[b0:b0 + len(zs)] = pr[:, 1] + pr[:, 2]
    fg = largest_component(fg_prob > 0.5)
    t_loc = time.time() - t0
    if not fg.any():
        return SegOutput(found_liver=False, img_size=S, eff_spacing=eff,
                         timings={"preprocess": t_pre, "localise": t_loc})
    z0, z1 = _slab(fg, eff)

    # ---- stage 2: stochastic passes on the liver slab (in-plane ROI = liver bbox + 15 mm)
    t0 = time.time()
    n = z1 - z0
    rows = np.where(fg.any(axis=(0, 2)))[0]
    cols = np.where(fg.any(axis=(0, 1)))[0]
    pr_, pc_ = int(math.ceil(15.0 / eff[1])), int(math.ceil(15.0 / eff[2]))
    roi = (max(0, int(rows.min()) - pr_), min(S, int(rows.max()) + pr_ + 1),
           max(0, int(cols.min()) - pc_), min(S, int(cols.max()) + pc_ + 1))
    mean = np.zeros((n, 3, S, S), dtype=np.float32)
    passes = None
    for b0 in range(z0, z1, bs):
        zs = list(range(b0, min(z1, b0 + bs)))
        pr = segmenter.predict_batch(stacks_25d(img, zs), roi=roi)   # (P, b, 3, S, S)
        if passes is None:
            passes = np.zeros((pr.shape[0], n, S, S), dtype=np.float16)
        i0 = b0 - z0
        mean[i0:i0 + len(zs)] = pr.mean(axis=0)
        passes[:, i0:i0 + len(zs)] = pr[:, :, 2].astype(np.float16)
    t_inf = time.time() - t0
    return _finish(mean, passes, z0, z1, S, eff, cfg, {"preprocess": t_pre, "localise": t_loc, "infer": t_inf})


def _finish(mean: np.ndarray, passes: np.ndarray, z0: int, z1: int, S: int, eff: tuple, cfg: dict,
            timings: dict) -> SegOutput:
    """Post-processing at model resolution: masks, case-level raw score, per-pass scores, entropy."""
    t0 = time.time()
    eff_vox = eff[0] * eff[1] * eff[2]
    min_ml = float(cfg["inference"]["min_lesion_ml"])
    k = max(1, int(math.ceil(min_ml * 1000.0 / eff_vox)))
    min_comp = max(1, int(round(min_ml * 1000.0 / eff_vox)))
    labels = mean.argmax(axis=1)
    liver = largest_component(labels > 0)
    liver = fill_holes_slicewise(liver)
    tumour = filter_lesions((labels == 2) & liver, mean[:, 2], min_comp, float(cfg["inference"]["min_lesion_peak_prob"]))
    region = ndi.binary_dilation(liver, iterations=1)
    # in-plane box smoothing over ~the diameter of a 0.1 ml sphere: isolated noisy voxels
    # cannot dominate the score, coherent blobs can
    lesion_diam_mm = 2.0 * (3.0 * min_ml * 1000.0 / (4.0 * math.pi)) ** (1.0 / 3.0)   # 5.8 mm for 0.1 ml
    ks = max(1, int(round(lesion_diam_mm / eff[1])))
    ks += 1 - ks % 2

    def smooth(x):
        return ndi.uniform_filter(x, size=(1, ks, ks)) if ks > 1 else x

    p_raw = topk_mean(smooth(mean[:, 2])[region], k)
    pass_scores = [topk_mean(smooth(passes[p].astype(np.float32))[region], k) for p in range(passes.shape[0])]
    eps = 1e-7
    entropy = -(mean * np.log(mean + eps)).sum(axis=1) / math.log(3.0)
    timings = {**timings, "postprocess": time.time() - t0}
    return SegOutput(found_liver=bool(liver.any()), z0=z0, z1=z1, img_size=S, eff_spacing=eff, mean=mean,
                     pass_tumour=passes, liver=liver, tumour=tumour, entropy=entropy.astype(np.float32),
                     p_raw=p_raw, pass_scores=pass_scores, k_voxels=k, n_passes=int(passes.shape[0]),
                     timings=timings)
