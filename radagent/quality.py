"""Automatic quality control of a first read.

A failed check never produces a confident suggestion: the agent routes the case to a
full manual read and says why. These checks catch failures that a calibrated
probability cannot see (wrong organ segmented, unusual geometry, ...).
"""
from __future__ import annotations


def quality_check(geometry: dict, measurements: dict, cfg: dict) -> dict:
    q = cfg["quality"]
    checks = []

    def add(name, passed, value, expected, msg):
        checks.append({"name": name, "passed": bool(passed), "value": value, "expected": expected, "message": msg})

    add("liver_found", measurements.get("liver_found", False), measurements.get("liver_found", False), True,
        "a liver was segmented" if measurements.get("liver_found") else "no liver found by the segmentation tool")
    lv = float(measurements.get("liver_volume_ml", 0.0))
    lo, hi = q["liver_ml_range"]
    add("liver_volume_plausible", lo <= lv <= hi, round(lv, 1), [lo, hi],
        f"liver volume {lv:.0f} ml" + ("" if lo <= lv <= hi else f" outside the plausible range {lo:.0f}-{hi:.0f} ml (partial scan or segmentation failure)"))
    sp = geometry.get("spacing_mm", [1, 1, 1])
    slo, shi = q["spacing_mm_range"]
    ok = all(slo <= float(s) <= shi for s in sp)
    add("voxel_spacing_plausible", ok, [round(float(s), 3) for s in sp], [slo, shi],
        "voxel spacing within range" if ok else "unusual voxel spacing - check the header")
    nz = int(geometry.get("shape_view", [0])[0])
    add("enough_slices", nz >= q["min_slices"], nz, f">= {q['min_slices']}",
        f"{nz} axial slices" + ("" if nz >= q["min_slices"] else " - too few for a 3D read"))
    frac = float(measurements.get("tumour_voxels_outside_liver_frac", 0.0))
    out_ml = float(measurements.get("tumour_volume_outside_liver_ml", 0.0))
    ok = frac <= q["max_tumour_outside_liver_frac"] or out_ml < q["min_outside_ml"]
    add("tumour_inside_liver", ok, frac, f"<= {q['max_tumour_outside_liver_frac']} (or < {q['min_outside_ml']} ml)",
        "tumour predictions are confined to the liver" if ok
        else f"{100 * frac:.0f}% ({out_ml:.1f} ml) of tumour-class voxels fall outside the liver - unreliable tumour map")
    passed = all(c["passed"] for c in checks)
    failed = [c["message"] for c in checks if not c["passed"]]
    return {"passed": passed, "checks": checks, "failed_reasons": failed}
