"""Guardrails for free text written by the vision-language model.

The model's text is never used for measurements or for the triage decision. It is shown
as *unverified model text*, and these checks tell the reader where it goes beyond what the
measurement tool produced.
"""
from __future__ import annotations

import re

_FINAL = re.compile(r"final\s*answer\s*[:\-]?\s*\**\s*(yes|no)\b", re.I)
_MEASURE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mm|cm|ml|cc|millimet\w*|centimet\w*)\b", re.I)


def parse_final_answer(text: str) -> int | None:
    m = list(_FINAL.finditer(text or ""))
    if not m:
        return None
    return 1 if m[-1].group(1).lower() == "yes" else 0


def unverified_measurements(text: str, measurements: dict, rel_tol: float = 0.25) -> list[str]:
    """Sizes/volumes quoted by the model that match no measured lesion (within 25 %)."""
    known_mm, known_ml = [], []
    for les in measurements.get("lesions", []) or []:
        known_mm.append(float(les.get("max_axial_diameter_mm", 0.0)))
        known_ml.append(float(les.get("volume_ml", 0.0)))
    known_ml.append(float(measurements.get("predicted_tumour_volume_ml", 0.0)))
    known_ml.append(float(measurements.get("liver_volume_ml", 0.0)))
    flags = []
    for m in _MEASURE.finditer(text or ""):
        val = float(m.group(1).replace(",", "."))
        unit = m.group(2).lower()
        if unit.startswith("c") and unit not in ("cc",):
            val_mm, refs = val * 10.0, known_mm
        elif unit.startswith("mm") or unit.startswith("millimet"):
            val_mm, refs = val, known_mm
        else:
            val_mm, refs = val, known_ml
        if not any(r > 0 and abs(val_mm - r) <= rel_tol * r for r in refs):
            flags.append(m.group(0))
    return flags


def check_text(text: str, measurements: dict, p_yes: float | None) -> dict:
    final = parse_final_answer(text)
    flags = unverified_measurements(text, measurements)
    notes = []
    if flags:
        notes.append("mentions sizes that the measurement tool did not produce: " + ", ".join(flags))
    if final is not None and p_yes is not None and final != int(p_yes >= 0.5):
        notes.append("the written conclusion disagrees with the model's own yes/no probability")
    if text and final is None:
        notes.append("no 'Final Answer' line - conclusion unclear")
    return {"final_answer": final, "unverified_sizes": flags, "notes": notes}
