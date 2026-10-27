"""The agent's tools.

Each tool is a plain Python function ``fn(ctx, **args) -> summary`` with a
description (shown to an LLM planner) and preconditions (enforced by the runner -
a planner can *propose* any order, but the runner only executes a tool whose
inputs exist). The only free text about the patient comes from the vision-language
second reader; it is stored verbatim, labelled as unverified and checked by guardrails,
and it never feeds measurements or the triage decision.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from radagent.calibration import Calibrator
from radagent.confidence import case_confidence
from radagent.quality import quality_check as _qc
from radagent.report import draft_report as _draft
from radagent.store import RunStore
from radagent.triage import triage as _triage


def model_signature(cfg: dict) -> str:
    """Identity of the segmenter and its settings; a change invalidates cached results."""
    sc = cfg["segmenter"]
    size = cfg["preprocess"]["img_size"]
    if sc["backend"] == "plugin":
        return f"plugin:{sc.get('plugin')}:{size}"
    tta = int(bool(cfg["inference"].get("tta_flip", True)))
    return (f"{sc['backend']}:{sc.get('hf_repo')}@{sc.get('hf_revision') or 'main'}:{sc.get('resample_mm')}mm:"
            f"roi{sc.get('roi_size')}:ov{sc.get('overlap')}:tta{tta}:{size}")


@dataclass
class CaseContext:
    case_id: str
    row: dict
    cfg: dict
    store: RunStore
    calibrator: Calibrator
    get_segmenter: Callable[[], object]
    get_vlm: Callable[[], object] | None = None
    vlm_mode: str = "cache"               # cache: reuse stored reads only | compute: run the VLM when missing
    research_mode: bool = True
    reader_policy: object | None = None
    force_recompute: bool = False
    persist: bool = True                  # False for ad-hoc re-runs (UI / CLI demo): do not overwrite reports
    # state
    volume: object | None = None
    seg: object | None = None
    analysis: dict = field(default_factory=dict)
    record: dict = field(default_factory=dict)
    done: dict = field(default_factory=dict)

    def cached(self) -> bool:
        return (not self.force_recompute and self.store.has_analysis(self.case_id)
                and self.analysis.get("model_signature") == model_signature(self.cfg)
                and "measurements" in self.analysis and self.store.viewer_path(self.case_id).exists())


# --------------------------------------------------------------------------- tools
def load_study(ctx: CaseContext) -> dict:
    from radagent.geometry import load_volume

    if ctx.store.has_analysis(ctx.case_id):
        ctx.analysis = ctx.store.load_analysis(ctx.case_id)
    if ctx.cached():
        geo = ctx.analysis["geometry"]
        src = "cache"
    else:
        ctx.analysis = {"case_id": ctx.case_id}
        vol = load_volume(ctx.row["ct_path"])
        ctx.volume = vol
        geo = {"shape_on_disk": list(vol.shape_on_disk), "shape_view": list(vol.view.shape),
               "spacing_mm": [round(s, 5) for s in vol.spacing], "orientation_on_disk": list(vol.orientation_on_disk),
               "canonical": "RAS+", "view": "radiological axial: row 0 anterior, col 0 patient right"}
        ctx.analysis["geometry"] = geo
        src = "NIfTI"
    ctx.record["geometry"] = geo
    sp = geo["spacing_mm"]
    return {"source": src, "shape": geo["shape_on_disk"], "orientation": "".join(geo["orientation_on_disk"]),
            "spacing_mm": f"{sp[2]:.2f} x {sp[1]:.2f} x {sp[0]:.2f}"}


def segment_liver_tumour(ctx: CaseContext) -> dict:
    from radagent.inference import segment_volume

    if ctx.cached():
        s = ctx.analysis["segmentation"]
        ctx.record["model"] = ctx.analysis["model"]
        return {"source": "cache", "found_liver": s["found_liver"], "passes": s["n_passes"],
                "raw_tumour_score": s["tumour_score_raw"]}
    seg_model = ctx.get_segmenter()
    vol = ctx.volume
    ctx.seg = segment_volume(vol.view, vol.spacing, seg_model, ctx.cfg)
    ctx.analysis["segmentation"] = ctx.seg.summary()
    ctx.analysis["model"] = {**seg_model.describe(), "signature": model_signature(ctx.cfg)}
    ctx.record["model"] = ctx.analysis["model"]
    return {"source": "model", "found_liver": ctx.seg.found_liver, "passes": ctx.seg.n_passes,
            "raw_tumour_score": round(ctx.seg.p_raw, 4), "seconds": round(sum(ctx.seg.timings.values()), 2)}


def measure_findings(ctx: CaseContext) -> dict:
    from radagent.geometry import load_volume
    from radagent.measure import measure, reference_metrics, viewer_arrays

    if ctx.cached():
        m = ctx.analysis["measurements"]
        src = "cache"
    else:
        vol = ctx.volume
        res = measure(vol.view, vol.spacing, ctx.seg, ctx.cfg)
        m = res["measurements"]
        gt_view = None
        mask_path = ctx.row.get("mask_path")
        if ctx.research_mode and mask_path and os.path.exists(str(mask_path)):
            gt = load_volume(mask_path)
            if gt.view.shape == vol.view.shape:
                gt_view = gt.view
                ctx.analysis["reference"] = reference_metrics(
                    gt_view, vol.spacing, ctx.seg, res["maps"], ctx.cfg["confidence"]["target_min_tumour_ml"], vol.view)
        ctx.store.save_viewer(ctx.case_id, viewer_arrays(vol.view, ctx.seg, res["maps"], ctx.cfg, gt_view))
        ctx.analysis["measurements"] = m
        ctx.analysis["model_signature"] = model_signature(ctx.cfg)
        ctx.store.save_analysis(ctx.case_id, ctx.analysis)
        src = "computed"
    ctx.record["measurements"] = m
    if "reference" in ctx.analysis:
        ctx.record["reference"] = ctx.analysis["reference"]
    return {"source": src, "liver_ml": m.get("liver_volume_ml"), "tumour_ml": m.get("predicted_tumour_volume_ml"),
            "n_lesions": m.get("n_lesions"), "largest_diameter_mm": m.get("largest_lesion_diameter_mm", 0.0)}


def _vlm_summary(v: dict) -> dict:
    if v.get("status") != "ok":
        return {"status": v.get("status"), "reason": v.get("reason")}
    return {"status": "ok", "source": v.get("source"), "p_yes": round(v["p_yes"], 4),
            "final_answer": (v.get("guard") or {}).get("final_answer"), "slices": v.get("n_images"),
            "seconds": v.get("seconds")}


def vlm_second_read(ctx: CaseContext) -> dict:
    """Optional second reader (a VLM adapter, off by default) reads axial slices through the predicted liver."""
    from radagent.vlm import vlm_enabled, vlm_signature

    vcfg = ctx.cfg["vlm"]
    if not vlm_enabled(ctx.cfg):
        ctx.record["vlm"] = {"status": "disabled", "reason": "vlm.backend is none"}
        return _vlm_summary(ctx.record["vlm"])
    sig = vlm_signature(ctx.cfg)
    cached = ctx.analysis.get("vlm") or {}
    seg_sig = ctx.analysis.get("model_signature")
    if (not ctx.force_recompute and cached.get("status") == "ok" and cached.get("signature") == sig
            and cached.get("seg_signature") == seg_sig):
        ctx.record["vlm"] = {**cached, "source": "cache"}
        return _vlm_summary(ctx.record["vlm"])
    if ctx.vlm_mode != "compute":
        ctx.record["vlm"] = {"status": "unavailable", "reason": "no stored second read for this case - run "
                                                                 "`python -m radagent vlm`"}
        return _vlm_summary(ctx.record["vlm"])
    from radagent.geometry import load_volume
    from radagent.vlm.ct_prep import liver_z_range, select_slices, thumbnail, window_rgb
    from radagent.vlm.guard import check_text

    try:
        vol = ctx.volume
        if vol is None:
            if not os.path.exists(str(ctx.row.get("ct_path", ""))):
                raise FileNotFoundError(f"CT file not found: {ctx.row.get('ct_path')}")
            vol = ctx.volume = load_volume(ctx.row["ct_path"])
        zr = liver_z_range(ctx.analysis)
        if zr is None:
            raise ValueError("no liver found by the segmenter - nothing to show the VLM")
        zs = select_slices(zr[0], zr[1], int(vcfg["n_slices"]))
        images = [window_rgb(vol.view[z], vcfg["windows"]) for z in zs]
        reader = ctx.get_vlm()
        out = reader.read(images, describe=bool(vcfg.get("describe", True)))
        guard = check_text(out.get("text", ""), ctx.record.get("measurements", {}), out["p_yes"])
        v = {"status": "ok", "source": "model", "signature": sig, "seg_signature": seg_sig,
             "model": reader.describe_model(), "p_yes": float(out["p_yes"]),
             "answer_mass": float(out.get("answer_mass", float("nan"))), "text": out.get("text", ""),
             "guard": guard, "slices_z": [int(z) for z in zs], "n_images": len(images),
             "seconds": out.get("seconds")}
        if ctx.persist:
            ctx.analysis["vlm"] = v
            ctx.store.save_analysis(ctx.case_id, ctx.analysis)
            vp = ctx.store.viewer_path(ctx.case_id)
            if vp.exists():
                import numpy as np

                arrs = ctx.store.load_viewer(ctx.case_id)
                ts = int(vcfg.get("thumb_size", 160))
                arrs["vlm_rgb"] = np.stack([thumbnail(im, ts) for im in images]).astype(np.uint8)
                arrs["vlm_z"] = np.asarray(zs, dtype=np.int32)
                ctx.store.save_viewer(ctx.case_id, arrs)
    except Exception as e:  # the VLM is a second opinion: its failure must not stop the first read
        v = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
    ctx.record["vlm"] = v
    return _vlm_summary(v)


def quality_check(ctx: CaseContext) -> dict:
    q = _qc(ctx.record["geometry"], ctx.record["measurements"], ctx.cfg)
    ctx.record["quality"] = q
    return {"passed": q["passed"], "failed": q["failed_reasons"]}


def estimate_confidence(ctx: CaseContext) -> dict:
    s = ctx.analysis["segmentation"]
    v = ctx.record.get("vlm") or {}
    p_vlm = v.get("p_yes") if v.get("status") == "ok" else None
    c = case_confidence(s["tumour_score_raw"], s.get("pass_scores_raw", []), ctx.calibrator, ctx.cfg, p_vlm)
    ctx.record["confidence"] = c
    out = {"tumour_probability": c["tumour_probability"], "uncertainty": c["uncertainty"],
           "level": c["uncertainty_level"], "calibrator": c["calibrator"]}
    if "tools_disagree" in c:
        out["tools_disagree"] = c["tools_disagree"]
    return out


def triage_case(ctx: CaseContext) -> dict:
    t = _triage(ctx.record["confidence"], ctx.record["measurements"], ctx.record["quality"], ctx.cfg)
    ctx.record["triage"] = t
    return {"category": t["category"], "priority": t["priority"]}


def apply_reader_policy(ctx: CaseContext) -> dict:
    pol = ctx.reader_policy
    c = ctx.record["confidence"]
    d = pol.decide(c["tumour_probability"], c["uncertainty"], ctx.record["quality"]["passed"])
    ctx.record.setdefault("policy_decisions", {})[pol.reader_id or "universal"] = d
    return {"reader": pol.reader_id, "decision": d["decision"], "threshold": d["threshold"], "reason": d["reason"]}


def draft_report(ctx: CaseContext) -> dict:
    policy = None
    if ctx.reader_policy is not None:
        policy = ctx.record.get("policy_decisions", {}).get(ctx.reader_policy.reader_id or "universal")
    text = _draft(ctx.record, review=None, policy=policy)
    ctx.record["report_text"] = text
    if not ctx.persist:
        return {"lines": text.count("\n") + 1, "path": None}
    path = ctx.store.report_path(ctx.case_id)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)
    ctx.record["report_path"] = str(path)
    return {"lines": text.count("\n") + 1, "path": str(path)}


def finish(ctx: CaseContext) -> dict:
    return {"status": "first read complete"}


@dataclass
class Tool:
    name: str
    fn: Callable
    description: str
    requires: tuple = ()


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("load_study", load_study, "Load the CT study and its header geometry (orientation, voxel spacing)."),
    Tool("segment_liver_tumour", segment_liver_tumour,
         "Run the segmentation tool (NVIDIA VISTA3D: liver + hepatic tumour) with test-time-augmentation "
         "passes for uncertainty.", ("load_study",)),
    Tool("measure_findings", measure_findings,
         "Measure liver and lesion volumes/diameters at original resolution from the voxel spacing.",
         ("segment_liver_tumour",)),
    Tool("quality_check", quality_check, "Run automatic quality-control checks on the first read.", ("measure_findings",)),
    Tool("vlm_second_read", vlm_second_read,
         "Ask the optional second reader (if configured) for an independent read of slices through the "
         "liver: its probability that a liver tumour is present and a short unverified description.",
         ("measure_findings",)),
    Tool("estimate_confidence", estimate_confidence,
         "Turn the segmenter score (and a second read, if any) into one calibrated tumour probability "
         "with an uncertainty estimate.",
         ("segment_liver_tumour", "vlm_second_read")),
    Tool("triage_case", triage_case, "Assign a worklist priority and triage category.",
         ("estimate_confidence", "quality_check")),
    Tool("apply_reader_policy", apply_reader_policy,
         "Apply the current reader's learned policy (suggestion threshold / deferral).",
         ("estimate_confidence", "quality_check")),
    Tool("draft_report", draft_report, "Fill the deterministic structured report template.", ("triage_case",)),
    Tool("finish", finish, "End the first read.", ("draft_report",)),
]}


def requirements(name: str, ctx: CaseContext) -> list[str]:
    req = list(TOOLS[name].requires)
    if name == "draft_report" and ctx.reader_policy is not None:
        req.append("apply_reader_policy")
    return req


def ready_tools(ctx: CaseContext) -> list[str]:
    out = []
    for name in TOOLS:
        if name in ctx.done:
            continue
        if name == "apply_reader_policy" and ctx.reader_policy is None:
            continue
        if all(r in ctx.done for r in requirements(name, ctx)):
            out.append(name)
    return out
