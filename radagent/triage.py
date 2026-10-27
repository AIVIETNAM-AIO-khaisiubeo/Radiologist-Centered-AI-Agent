"""Worklist triage: a transparent priority score and category for every case.

priority = w_p * p_cal + w_v * volume_term + w_u * uncertainty,   volume_term = log1p(ml) / log1p(ref)
(clipped to [0, 1]). Uncertain cases are pulled *up* the list on purpose: they are the
ones that most need a careful human read. When the two AI tools (segmenter and an optional second reader)
disagree, the case gets a priority bonus and is never labelled LIKELY NORMAL. Cases that
failed QC are placed near the top as well because they cannot be pre-read by the AI at all.
"""
from __future__ import annotations

import math

CATEGORY_ORDER = ["QC FAIL", "URGENT", "SUSPICIOUS", "UNCERTAIN", "LIKELY NORMAL"]
CATEGORY_HELP = {
    "QC FAIL": "Automatic quality control failed - full manual read required.",
    "URGENT": "High-confidence tumour finding with a sizeable lesion.",
    "SUSPICIOUS": "Tumour suspected by the AI.",
    "UNCERTAIN": "AI is uncertain, or two AI tools disagree - read carefully.",
    "LIKELY NORMAL": "No tumour suspected by the AI.",
}


def triage(conf: dict, meas: dict, qc: dict, cfg: dict, threshold: float | None = None) -> dict:
    t = cfg["triage"]
    thr = t["suggest_threshold"] if threshold is None else threshold
    p = float(conf["tumour_probability"])
    u = float(conf["uncertainty"])
    vol = float(meas.get("predicted_tumour_volume_ml", 0.0))
    diam = float(meas.get("largest_lesion_diameter_mm", 0.0))
    w = t["weights"]
    vterm = min(1.0, math.log1p(vol) / math.log1p(t["volume_ref_ml"]))
    priority = w["probability"] * p + w["volume"] * vterm + w["uncertainty"] * u
    reasons = []
    disagree = bool(conf.get("tools_disagree", False))
    if disagree:
        priority += float(t.get("disagreement_bonus", 0.10))
    if not qc.get("passed", True):
        cat = "QC FAIL"
        priority = max(0.75, priority)
        reasons += qc.get("failed_reasons", [])
    elif p >= t["urgent_probability"] and (vol >= t["urgent_volume_ml"] or diam >= t["urgent_diameter_mm"]):
        cat = "URGENT"
        reasons.append(f"p = {p:.2f} >= {t['urgent_probability']} and lesion {vol:.1f} ml / {diam:.0f} mm")
    elif p >= thr:
        cat = "SUSPICIOUS"
        reasons.append(f"p = {p:.2f} >= suggestion threshold {thr:.2f}")
    elif u >= t["uncertain_flag"]:
        cat = "UNCERTAIN"
        reasons.append(f"uncertainty {u:.2f} >= {t['uncertain_flag']}")
    elif disagree:
        cat = "UNCERTAIN"
        reasons.append("segmenter and second reader disagree")
    else:
        cat = "LIKELY NORMAL"
        reasons.append(f"p = {p:.2f} < {thr:.2f}, uncertainty {u:.2f}")
    if disagree and cat not in ("UNCERTAIN", "QC FAIL"):
        reasons.append(f"segmenter ({conf.get('segmenter_probability', 0):.2f}) and second reader "
                       f"({conf.get('vlm_probability', 0):.2f}) disagree")
    return {"priority": round(float(min(1.0, priority)), 4), "category": cat, "threshold": thr,
            "reasons": reasons, "components": {"probability": round(p, 4), "volume_term": round(vterm, 4),
                                               "uncertainty": round(u, 4)}}


def rank_worklist(records: list[dict], key: str = "triage") -> list[dict]:
    ordered = sorted(records, key=lambda r: (-r[key]["priority"], r["case_id"]))
    for i, r in enumerate(ordered, start=1):
        r[key]["rank"] = i
        r[key]["of"] = len(ordered)
    return ordered
