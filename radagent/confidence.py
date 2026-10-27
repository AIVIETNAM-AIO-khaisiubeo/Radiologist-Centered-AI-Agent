"""Case-level confidence and uncertainty.

``tumour_probability``  calibrated probability that the case contains a tumour - from the
                        segmenter score (or, with an optional second reader, its P(yes) or both fused), whichever the
                        calibration split selected (see :mod:`radagent.calibration`).
``uncertainty``         predictive entropy of that probability, in bits (0 = certain,
                        1 = p = 0.5). For a calibrated model the expected error of the
                        suggestion is min(p, 1 - p), a monotone function of this entropy.

Components reported so a reviewer can see *why* a case is uncertain:

* expected entropy and mutual information over the segmenter's stochastic passes
  (the CT and its left-right flipped copy): total = aleatoric + epistemic (Depeweg et al. 2018);
* with a second reader: both calibrated probabilities and whether the two AI
  tools disagree (one says tumour, the other does not).
"""
from __future__ import annotations

import math

import numpy as np

from radagent.calibration import Calibrator, binary_entropy


def uncertainty_level(u: float) -> str:
    return "low" if u < 0.5 else ("medium" if u < 0.85 else "high")


def case_confidence(p_raw: float, pass_scores: list, calibrator: Calibrator, cfg: dict,
                    p_vlm: float | None = None) -> dict:
    pv = None if p_vlm is None or (isinstance(p_vlm, float) and math.isnan(p_vlm)) else float(p_vlm)
    p_cal = float(calibrator(p_raw, pv))
    scores = np.asarray(pass_scores, dtype=np.float64) if len(pass_scores) else np.array([p_raw])
    pass_cal = np.asarray(calibrator(scores, pv), dtype=np.float64)
    total = float(binary_entropy(p_cal))
    expected = float(np.mean(binary_entropy(pass_cal)))
    mi = float(max(0.0, binary_entropy(float(pass_cal.mean())) - expected))
    p_seg = float(calibrator.seg_only(p_raw))
    out = {
        "tumour_probability_raw": round(float(p_raw), 5),
        "tumour_probability": round(p_cal, 5),
        "pass_probabilities": [round(float(x), 4) for x in pass_cal.tolist()],
        "pass_std": round(float(pass_cal.std()), 5),
        "uncertainty": round(total, 5),
        "uncertainty_level": uncertainty_level(total),
        "uncertainty_components": {"predictive_entropy": round(total, 5),
                                   "expected_entropy": round(expected, 5),
                                   "mutual_information": round(mi, 5)},
        "segmenter_probability": round(p_seg, 5),
        "calibrator": calibrator.describe(),
        "calibration_method": calibrator.method,
    }
    if pv is not None:
        p_v = float(calibrator.vlm_only(pv))
        out["vlm_probability_raw"] = round(pv, 5)
        out["vlm_probability"] = round(p_v, 5)
        out["tools_disagree"] = bool((p_seg >= 0.5) != (p_v >= 0.5))
    return out
