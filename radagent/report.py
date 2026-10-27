"""Deterministic, structured report draft.

Every finding in the draft is a fixed template filled with measured values and model
outputs, so it is reproducible and auditable. The one exception is the clearly separated
SECOND READ section (only with an optional second reader): its own words, quoted verbatim, labelled as unverified model
text and checked by guardrails (sizes the measurement tool did not produce, a written
conclusion that contradicts the model's own yes/no probability). The radiologist's
decision is a separate, clearly attributed section.
"""
from __future__ import annotations

from datetime import datetime, timezone

from radagent import RESEARCH_BANNER

ACTION_TEXT = {
    "accept": "accepted the AI suggestion",
    "edit": "edited the AI suggestion",
    "reject": "rejected the AI suggestion",
    "manual": "read the case without an AI suggestion (AI suggestion withheld)",
    "defer": "deferred the decision (second opinion requested)",
}


def _fmt_ml(x: float) -> str:
    return f"{x:.1f} ml" if x >= 1 else f"{x * 1000:.0f} mm3"


def suggestion_text(p: float, threshold: float) -> str:
    return "Liver tumour suspected" if p >= threshold else "No liver tumour suspected"


def _wrap(text: str, width: int = 96, indent: str = "    ") -> list[str]:
    import textwrap

    out = []
    for para in (text or "").splitlines():
        if para.strip():
            out += textwrap.wrap(para.strip(), width=width, initial_indent=indent, subsequent_indent=indent)
    return out


def draft_report(record: dict, review: dict | None = None, policy: dict | None = None) -> str:
    g, m, c, q, t = (record.get(k, {}) for k in ("geometry", "measurements", "confidence", "quality", "triage"))
    model = record.get("model", {})
    thr = (policy or {}).get("threshold", t.get("threshold", 0.5))
    L = [RESEARCH_BANNER, "=" * len(RESEARCH_BANNER),
         f"Study: {record['case_id']}   |   Source: {record.get('source', '-')}   |   Exam: CT abdomen, axial",
         (f"Geometry: {g.get('shape_on_disk')} voxels on disk ({''.join(g.get('orientation_on_disk', []))}), "
          f"spacing {g.get('spacing_mm', ['?'] * 3)[2]:.2f} x {g.get('spacing_mm', ['?'] * 3)[1]:.2f} x "
          f"{g.get('spacing_mm', ['?'] * 3)[0]:.2f} mm (col x row x slice)") if g else "Geometry: -",
         f"AI tool: segmenter {model.get('version', model.get('backend', '-'))} "
         f"({model.get('passes', '-')} test-time passes)"
         + (f"; second reader {(record.get('vlm') or {}).get('model', {}).get('model_id')}"
            if (record.get("vlm") or {}).get("status") == "ok" else ""),
         f"Confidence: {c.get('calibrator', '-')}", ""]

    L.append("QUALITY CONTROL")
    if q:
        L.append("  Passed." if q.get("passed") else "  FAILED - " + "; ".join(q.get("failed_reasons", [])))
    L.append("")
    L.append("AI FIRST READ (measured values)")
    if not m.get("liver_found", False):
        L.append("  Liver: not identified by the segmentation tool. No measurements available.")
    else:
        L.append(f"  Liver: segmented; volume {m['liver_volume_ml']:.0f} ml; mean attenuation {m.get('liver_mean_hu', float('nan')):.0f} HU.")
        n = m.get("n_lesions", 0)
        if n == 0:
            L.append("  Focal liver lesions (AI): none segmented above the minimum lesion size.")
        else:
            L.append(f"  Focal liver lesions (AI): {n} candidate(s); total volume "
                     f"{_fmt_ml(m['predicted_tumour_volume_ml'])} ({m['tumour_burden_pct']:.2f}% of liver volume).")
            for les in m["lesions"][:8]:
                L.append(f"    {les['id']}. {_fmt_ml(les['volume_ml'])}, max axial diameter {les['max_axial_diameter_mm']:.0f} mm, "
                         f"slices {les['slice_range'][0]}-{les['slice_range'][1]}, mean {les['mean_hu']:.0f} HU, "
                         f"mean model probability {les['mean_probability']:.2f}")
            if n > 8:
                L.append(f"    ... {n - 8} smaller candidate(s) not listed")
    if c:
        L.append(f"  Case-level tumour probability (calibrated): {c['tumour_probability']:.2f}  "
                 f"[segmenter raw score {c['tumour_probability_raw']:.2f}]")
        uc = c["uncertainty_components"]
        L.append(f"  Uncertainty: {c['uncertainty']:.2f} bits ({c['uncertainty_level']}); "
                 f"model disagreement across {len(c.get('pass_probabilities', []))} passes "
                 f"{uc.get('mutual_information', 0.0):.2f} bits")
    if t:
        rank = f", rank {t.get('rank')}/{t.get('of')}" if t.get("rank") else ""
        L.append(f"  Triage: {t['category']} (priority {t['priority']:.2f}{rank}).")
    v = record.get("vlm") or {}
    if v.get("status") == "ok":
        L.append("")
        L.append(f"SECOND READ - {v.get('model', {}).get('model_id', 'vision-language model')} "
                 f"({v.get('n_images', '?')} axial slices)")
        line = f"  P(tumour) answered by the model: {v['p_yes']:.2f}"
        if c.get("vlm_probability") is not None:
            line += f" (calibrated {c['vlm_probability']:.2f})"
        if c.get("tools_disagree") is not None:
            line += "; DISAGREES with the segmenter" if c["tools_disagree"] else "; agrees with the segmenter"
        L.append(line + ".")
        if v.get("text"):
            L.append("  Model text (unverified, quoted verbatim, not used for measurements or triage):")
            L += _wrap(v["text"][:1200] + (" [...]" if len(v["text"]) > 1200 else ""))
        for note in (v.get("guard") or {}).get("notes", []):
            L.append(f"  Guardrail: {note}.")
    elif v.get("status") in ("unavailable", "error"):
        L.append("")
        L.append(f"SECOND READ - not available ({v.get('reason', v.get('status'))}).")
    L.append("")
    L.append("AI SUGGESTION")
    if policy and policy.get("decision") == "defer":
        L.append("  Withheld - the reader-adapted policy deferred this case to an independent read "
                 f"({policy.get('reason', 'uncertain')}).")
    elif q and not q.get("passed", True):
        L.append("  None - quality control failed; full manual read required.")
    elif c:
        L.append(f"  {suggestion_text(c['tumour_probability'], thr)} "
                 f"(threshold {thr:.2f}{', reader-specific' if policy and policy.get('personalised') else ''}).")
    L.append("")
    L.append("RADIOLOGIST REVIEW")
    if not review:
        L.append("  Pending.")
    else:
        when = review.get("timestamp") or datetime.now(timezone.utc).isoformat(timespec="seconds")
        L.append(f"  {review.get('reader_name', review.get('reader_id', 'Reader'))} "
                 f"{ACTION_TEXT.get(review.get('action'), review.get('action'))} at {when}.")
        if review.get("initial_read") is not None:
            L.append(f"  Independent impression before seeing the AI: {'tumour' if review['initial_read'] else 'no tumour'}.")
        if review.get("final_label") is None:
            L.append("  Final finding: not yet decided - deferred for a second opinion.")
        else:
            L.append(f"  Final finding: {'liver tumour present' if review.get('final_label') else 'no liver tumour'}.")
        rec = review.get("recommendation") or {}
        if rec:
            L.append(f"  AI recommendation shown at decision time: {rec.get('text', '-')}"
                     + (f" (policy {review.get('policy_version')}, model set {review.get('model_version')})"
                        if review.get("policy_version") else "") + ".")
        if review.get("edit"):
            e = review["edit"]
            if e.get("kept_lesions") is not None:
                L.append(f"  Lesions retained after edit: {e['kept_lesions'] or 'none'}; "
                         f"edited tumour volume {_fmt_ml(float(e.get('edited_volume_ml', 0.0)))}.")
            if e.get("defer_reason"):
                L.append(f"  Reason for deferral: {e['defer_reason']}.")
            if e.get("note"):
                L.append(f"  Note: {e['note']}")
        if review.get("supersedes"):
            L.append(f"  This decision revises the one saved at {review['supersedes']} (both are kept in the log).")
        L.append("  Status: research decision saved; this report draft is not signed.")
    L.append("")
    L.append("All values above are automatic research measurements from a public, de-identified dataset;")
    L.append("they have not been validated for clinical use. The radiologist is responsible for every finding.")
    return "\n".join(L)
