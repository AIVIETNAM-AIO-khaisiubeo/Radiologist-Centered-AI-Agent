"""Simulated readers for a controlled human-AI experiment on the LiTS cases.

Each reader follows a *sequential reading protocol* (an initial independent impression,
then the AI suggestion, then the final decision), which is how automation bias is
measured in reader studies. The behaviour model:

own read     P(call tumour | tumour of V ml, contrast d HU)
                 = sigmoid(sens_a + sens_b * log10 V + sens_c * (|d| - 20) / 10)
             P(call tumour | no tumour) = fpr
             -> small and faint lesions are missed by readers *and* by the model
                (correlated errors); d is measured from the image and the reference mask
lapse        with probability ``lapse`` the final decision is a coin flip (inconsistency)
agreement    own read == AI suggestion -> keep it
disagreement switch to the AI with probability  sigmoid(trust_k * (c - trust_tau)),
             c = the AI's confidence in its suggested label (>= 0.5)  ("trust curve")
outline      if the AI outline is poor (tumour Dice < 0.5) and the reader keeps the AI
             call, they edit it with probability ``edit_poor``; ``edit_noise`` adds
             unnecessary edits.

Reader A switches readily (over-reliance), Reader B almost never does and is a
conservative caller (rejects many suggestions), Reader C is noisy.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class ReaderProfile:
    reader_id: str
    name: str
    description: str
    sens_a: float
    sens_b: float
    sens_c: float
    fpr: float
    trust_k: float
    trust_tau: float
    lapse: float
    edit_poor: float
    edit_noise: float

    def p_call_positive(self, gt_present: bool, gt_ml: float, contrast_hu: float = 20.0) -> float:
        if gt_present:
            return _sigmoid(self.sens_a + self.sens_b * math.log10(max(gt_ml, 0.01))
                            + self.sens_c * (abs(contrast_hu) - 20.0) / 10.0)
        return self.fpr

    def trust(self, conf: float) -> float:
        return _sigmoid(self.trust_k * (conf - self.trust_tau))

    def to_dict(self) -> dict:
        return asdict(self)


def load_profiles(cfg: dict) -> dict[str, ReaderProfile]:
    out = {}
    for rid, p in cfg["readers"]["profiles"].items():
        out[rid] = ReaderProfile(reader_id=rid, name=p["name"], description=p.get("description", ""),
                                 sens_c=float(p.get("sens_c", 0.0)),
                                 **{k: float(p[k]) for k in ("sens_a", "sens_b", "fpr", "trust_k", "trust_tau",
                                                             "lapse", "edit_poor", "edit_noise")})
    return out


N_DRAWS = 6


def draws_for(seed: int, reader_index: int, phase: int, n: int) -> np.ndarray:
    """Common random numbers: identical draws for every policy compared on the same reads."""
    return np.random.default_rng([seed, reader_index, phase]).random((n, N_DRAWS))


def simulate_read(profile: ReaderProfile, case: dict, decision: dict, u: np.ndarray) -> dict:
    """One sequential read. ``case`` needs gt_present, gt_tumour_ml, poor_outline.

    ``decision``: {"decision": "show"|"defer", "ai_label": 0/1, "confidence": c}
    """
    own = int(u[0] < profile.p_call_positive(case["gt_present"], case["gt_tumour_ml"], case.get("gt_contrast_hu", 20.0)))
    lapsed = bool(u[1] < profile.lapse)
    if decision["decision"] != "show":
        final = int(u[2] < 0.5) if lapsed else own
        return {"initial_read": own, "final_label": final, "action": "manual", "shown": False,
                "switched": None, "lapsed": lapsed}
    ai = int(decision["ai_label"])
    conf = float(decision["confidence"])
    switched = None
    if lapsed:
        final = int(u[2] < 0.5)
    elif own == ai:
        final = ai
    else:
        switched = bool(u[3] < profile.trust(conf))
        final = ai if switched else own
    if final != ai:
        action = "reject"
    elif ai == 1 and case.get("poor_outline", False) and u[4] < profile.edit_poor:
        action = "edit"
    elif u[5] < profile.edit_noise:
        action = "edit"
    else:
        action = "accept"
    return {"initial_read": own, "final_label": final, "action": action, "shown": True, "switched": switched,
            "lapsed": lapsed}
