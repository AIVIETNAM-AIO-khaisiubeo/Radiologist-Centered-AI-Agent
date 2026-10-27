"""Unit tests for geometry, calibration, policies, readers, triage, agent guardrails and the report.

Run:  pytest -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from radagent.agent.llm_client import extract_json  # noqa: E402
from radagent.calibration import Calibrator, brier, ece, fit_calibrator, fit_temperature, sigmoid, logit  # noqa: E402
from radagent.config import load_config  # noqa: E402
from radagent.geometry import (axcodes, canonical_to_view, load_volume, to_canonical,  # noqa: E402
                               view_to_canonical)
from radagent.inference import topk_mean  # noqa: E402
from radagent.io.nifti import _read_nifti1, _write_nifti1, save_nifti  # noqa: E402
from radagent.policy import ReaderPolicy, expected_loss, fit_policy, fit_trust_curve, universal_policy  # noqa: E402
from radagent.readers.profiles import draws_for, load_profiles, simulate_read  # noqa: E402
from radagent.report import draft_report  # noqa: E402
from radagent.triage import rank_worklist, triage  # noqa: E402


# --------------------------------------------------------------------------- geometry
def test_nifti_roundtrip_and_canonical(tmp_path):
    rng = np.random.default_rng(0)
    A = rng.integers(-1000, 1000, size=(7, 5, 4)).astype(np.int16)
    aff = np.diag([-0.7, -0.8, 2.5, 1.0])
    aff[:3, 3] = [10, 20, -30]
    _write_nifti1(A, aff, str(tmp_path / "a.nii.gz"))
    B, aff2 = _read_nifti1(str(tmp_path / "a.nii.gz"))
    assert np.array_equal(A, B) and np.allclose(aff, aff2)
    assert axcodes(aff) == ("L", "P", "S")
    C, affc = to_canonical(A, aff)
    assert axcodes(affc) == ("R", "A", "S")
    assert np.array_equal(C, A[::-1, ::-1, :])
    # the same world point is addressed by both voxel grids
    w1 = aff @ [2, 3, 1, 1]
    w2 = affc @ [6 - 2, 4 - 3, 1, 1]
    assert np.allclose(w1, w2)
    V = canonical_to_view(C)
    assert np.array_equal(view_to_canonical(V), C)
    # LPS storage: radiological view slice z is the transposed stored slice
    assert np.array_equal(V[1], A[:, :, 1].T)


def test_permuted_affine_canonical():
    A = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    P = np.zeros((4, 4))
    P[0, 2], P[1, 0], P[2, 1], P[3, 3] = 1.5, -0.9, 0.8, 1
    C, affc = to_canonical(A, P)
    assert axcodes(affc) == ("R", "A", "S")
    for idx in [(1, 2, 3), (0, 0, 0), (3, 4, 5)]:
        w = P @ [*idx, 1]
        v = np.round(np.linalg.solve(affc[:3, :3], w[:3] - affc[:3, 3])).astype(int)
        assert C[tuple(v)] == A[idx]


def test_volume_spacing_and_ml(tmp_path):
    data = np.zeros((10, 8, 6), dtype=np.uint8)
    data[2:6, 2:5, 1:4] = 1                                      # 4*3*3 = 36 voxels
    aff = np.diag([-0.5, -0.5, 2.0, 1.0])
    save_nifti(data, aff, tmp_path / "m.nii.gz", use_nibabel=False)
    vol = load_volume(tmp_path / "m.nii.gz", use_nibabel=False)
    assert vol.view.shape == (6, 8, 10)
    assert np.allclose(vol.spacing, (2.0, 0.5, 0.5))
    assert math.isclose(vol.volume_ml(vol.view > 0), 36 * 0.5 * 0.5 * 2.0 / 1000.0)


# --------------------------------------------------------------------------- calibration
def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(1)
    z_true = rng.normal(0, 1.5, 4000)
    y = (rng.random(4000) < sigmoid(z_true)).astype(int)
    p_over = sigmoid(3.0 * z_true)                               # over-confident by a factor 3
    T = fit_temperature(p_over, y)
    assert 2.4 < T < 3.6
    cal = Calibrator("temperature", T=T)
    assert ece(cal(p_over), y) < ece(p_over, y)
    assert brier(cal(p_over), y) < brier(p_over, y)


def test_auto_calibrator_and_single_class():
    rng = np.random.default_rng(2)
    p = rng.random(60)
    y = (rng.random(60) < p).astype(int)
    cal, info = fit_calibrator(p, y, "auto", seed=0)
    assert info["selected"] in ("identity", "temperature", "platt")
    cal1, info1 = fit_calibrator(p, np.ones(60, dtype=int))
    assert cal1.method == "identity" and "warning" in info1
    assert np.allclose(Calibrator.from_dict(cal.to_dict())(p), cal(p))


def test_ece_perfect_and_logit():
    p = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0, 0, 1, 1])
    assert ece(p, y) < 1e-9
    assert np.allclose(sigmoid(logit([0.2, 0.7])), [0.2, 0.7])


def test_topk_mean():
    v = np.array([0.1, 0.9, 0.5, 0.8])
    assert math.isclose(topk_mean(v, 2), 0.85)
    assert topk_mean(np.array([]), 3) == 0.0


# --------------------------------------------------------------------------- readers & policies
def _cases(n=40, seed=0):
    rng = np.random.default_rng(seed)
    cases = {}
    for i in range(n):
        g = bool(rng.random() < 0.5)
        p = float(np.clip(rng.normal(0.75 if g else 0.25, 0.2), 0.01, 0.99))
        cases[f"c{i}"] = {"case_id": f"c{i}", "p": p, "u": float(1 - abs(2 * p - 1)), "qc_passed": True,
                          "gt_present": g, "gt_tumour_ml": 5.0 if g else 0.0, "poor_outline": False, "split": "calib"}
    return cases


def test_trust_curve_recovers_direction():
    rng = np.random.default_rng(3)
    c = rng.uniform(0.5, 1.0, 400)
    y = rng.random(400) < 1 / (1 + np.exp(-10 * (c - 0.7)))
    prior = {"alpha_mean": 0.0, "alpha_sd": 1.5, "beta_mean": 4.0, "beta_sd": 3.0}
    a, b = fit_trust_curve(c, y, prior)
    assert b > 4.0                                               # steeper than the prior: data dominates
    a0, b0 = fit_trust_curve(np.array([]), np.array([]), prior)
    assert (a0, b0) == (0.0, 4.0)


def test_policy_decide_and_qc():
    pol = ReaderPolicy(name="x", reader_id="r", threshold=0.6, defer_uncertainty=0.7)
    assert pol.decide(0.9, 0.2)["decision"] == "show" and pol.decide(0.9, 0.2)["ai_label"] == 1
    assert pol.decide(0.5, 0.95)["decision"] == "defer"
    assert pol.decide(0.9, 0.1, qc_passed=False)["decision"] == "defer"
    assert pol.decide(0.55, 0.1)["ai_label"] == 0


def test_fit_policy_personalises_only_with_evidence():
    cfg = load_config(ROOT / "configs" / "lits.yaml")
    cases = _cases()
    prof = load_profiles(cfg)["reader_a"]
    events = []
    d = draws_for(0, 0, 0, len(cases))
    for i, c in enumerate(cases.values()):
        dec = universal_policy(cfg).decide(c["p"], c["u"])
        r = simulate_read(prof, c, dec, d[i])
        events.append({"case_id": c["case_id"], "shown": True, "ai_label": dec["ai_label"],
                       "ai_confidence": dec["confidence"], "initial_read": r["initial_read"],
                       "final_label": r["final_label"], "action": r["action"]})
    few = fit_policy(events[:3], cases, cfg, "reader_a")
    assert not few.personalised and few.threshold == cfg["policy"]["universal_threshold"]
    full = fit_policy(events, cases, cfg, "reader_a")
    assert full.n_feedback == len(events)
    assert full.expected_loss is not None and full.expected_loss <= full.universal_loss + 1e-9


def test_expected_loss_defer_everything_equals_reader_alone():
    cfg = load_config(ROOT / "configs" / "lits.yaml")
    cases = _cases()
    from radagent.policy import _arrays

    events = [{"case_id": k, "initial_read": int(v["gt_present"]), "shown": True, "ai_label": 1,
               "ai_confidence": 0.8, "final_label": 1} for k, v in cases.items()]
    arr = _arrays(events, cases)
    loss = expected_loss(arr, 0.5, -1.0, 0.0, 4.0, 0.05, cfg["policy"])   # defer all, perfect reader
    assert math.isclose(loss, cfg["policy"]["burden_weight"] * cfg["policy"]["action_cost"]["manual"])


def test_simulated_reader_behaviour():
    cfg = load_config(ROOT / "configs" / "lits.yaml")
    prof = load_profiles(cfg)
    case = {"case_id": "x", "gt_present": False, "gt_tumour_ml": 0.0, "poor_outline": False}
    wrong_ai = {"decision": "show", "ai_label": 1, "confidence": 0.95}
    u = np.random.default_rng(0).random((2000, 6))
    acc = {k: np.mean([simulate_read(p, case, wrong_ai, x)["final_label"] for x in u]) for k, p in prof.items()}
    assert acc["reader_a"] > acc["reader_b"]                     # A follows a wrong AI far more often than B


# --------------------------------------------------------------------------- triage, report, agent
def _record():
    return {"case_id": "lits_028", "split": "test", "source": "LiTS17",
            "geometry": {"shape_on_disk": [512, 512, 120], "spacing_mm": [2.5, 0.78, 0.78], "orientation_on_disk": ["L", "P", "S"]},
            "model": {"backend": "vista3d", "version": "VISTA3D 1.0", "passes": 2},
            "measurements": {"liver_found": True, "liver_volume_ml": 1500.0, "liver_mean_hu": 60.0,
                             "predicted_tumour_volume_ml": 6.2, "tumour_burden_pct": 0.41, "n_lesions": 1,
                             "largest_lesion_diameter_mm": 24.0, "liver_slice_range": [30, 90],
                             "lesions": [{"id": 1, "volume_ml": 6.2, "max_axial_diameter_mm": 24.0, "slice_range": [60, 68],
                                          "key_slice": 64, "mean_hu": 30.0, "mean_probability": 0.8}]},
            "vlm": {"status": "ok", "source": "cache", "p_yes": 0.83,
                    "model": {"backend": "plugin", "model_id": "example-vlm (research plugin)", "precision": "bf16"},
                    "text": "A hypodense lesion in the right lobe, about 4 cm. Final Answer: yes",
                    "guard": {"final_answer": 1, "unverified_sizes": ["4 cm"],
                              "notes": ["mentions sizes that the measurement tool did not produce: 4 cm"]}},
            "confidence": {"tumour_probability": 0.9, "tumour_probability_raw": 0.95, "uncertainty": 0.47,
                           "uncertainty_level": "medium", "calibrator": "fused logistic calibration",
                           "segmenter_probability": 0.88, "vlm_probability_raw": 0.83, "vlm_probability": 0.8,
                           "tools_disagree": False,
                           "uncertainty_components": {"predictive_entropy": 0.47, "expected_entropy": 0.4,
                                                      "mutual_information": 0.07}, "pass_probabilities": [0.88, 0.9, 0.92]},
            "quality": {"passed": True, "failed_reasons": []}}


def test_triage_and_report_are_deterministic():
    cfg = load_config(ROOT / "configs" / "lits.yaml")
    rec = _record()
    t = triage(rec["confidence"], rec["measurements"], rec["quality"], cfg)
    assert t["category"] == "URGENT"
    rec["triage"] = t
    rank_worklist([rec])
    a, b = draft_report(rec), draft_report(rec)
    assert a == b and "RESEARCH PREDICTION" in a and "6.2 ml" in a and "Pending" in a
    assert "example-vlm" in a and "4 cm" in a and "unverified" in a.lower()
    rv = draft_report(rec, review={"reader_id": "live", "action": "reject", "final_label": 0, "initial_read": 0,
                                   "timestamp": "2026-10-25T10:00:00"})
    assert "rejected the AI suggestion" in rv and "no liver tumour" in rv
    qc_fail = {**rec, "quality": {"passed": False, "failed_reasons": ["liver volume implausible"]}}
    assert triage(rec["confidence"], rec["measurements"], qc_fail["quality"], cfg)["category"] == "QC FAIL"


def test_extract_json_tolerates_prose_and_fences():
    assert extract_json('Sure!\n```json\n{"tool": "finish", "args": {}}\n```')["tool"] == "finish"
    assert extract_json('I pick {"tool": "load_study", "reason": "a {b} c"} ok')["tool"] == "load_study"


def test_agent_guardrails_reject_out_of_order_tools():
    from radagent.agent.runner import validate
    from radagent.agent.tools import CaseContext

    ctx = CaseContext(case_id="x", row={}, cfg={}, store=None, calibrator=Calibrator(), get_segmenter=lambda: None)
    assert validate({"tool": "draft_report", "args": {}}, ctx) is not None
    assert validate({"tool": "rm -rf", "args": {}}, ctx) is not None
    assert validate({"tool": "load_study", "args": {}}, ctx) is None
    assert validate({"tool": "load_study", "args": {"path": "/etc"}}, ctx) is not None
    ctx.done["load_study"] = {}
    assert validate({"tool": "load_study", "args": {}}, ctx) is not None


def test_triage_disagreement_never_likely_normal():
    cfg = load_config(ROOT / "configs" / "lits.yaml")
    rec = _record()
    calm = {**rec["confidence"], "tumour_probability": 0.1, "uncertainty": 0.2, "tools_disagree": False}
    meas0 = {**rec["measurements"], "predicted_tumour_volume_ml": 0.0, "largest_lesion_diameter_mm": 0.0}
    t0 = triage(calm, meas0, rec["quality"], cfg)
    t1 = triage({**calm, "tools_disagree": True}, meas0, rec["quality"], cfg)
    assert t0["category"] == "LIKELY NORMAL" and t1["category"] == "UNCERTAIN"
    assert t1["priority"] > t0["priority"]


def test_llm_planner_guardrails_with_scripted_model():
    """Scripted LLM replies: valid, out-of-order, malformed, argument injection -> runner keeps a valid plan."""
    from radagent.agent import planner as P
    from radagent.agent import runner as Rn
    from radagent.agent.tools import TOOLS, CaseContext

    replies = iter([
        '{"tool": "load_study", "args": {}, "reason": "start"}',
        '{"tool": "draft_report", "args": {}, "reason": "skip ahead"}',          # out of order -> override
        'I think we should segment now.',                                          # no JSON -> override
        '{"tool": "measure_findings", "args": {"path": "/etc/passwd"}}',          # free args -> override
    ] + ['{"tool": "finish", "args": {}}'] * 20)

    class FakeClient:
        def chat(self, messages):
            assert "never write findings" in messages[0]["content"].lower() or "never describe" in messages[0]["content"].lower()
            return next(replies)

    pl = P.LLMPlanner.__new__(P.LLMPlanner)
    pl.client, pl.model = FakeClient(), "scripted"
    saved = {k: t.fn for k, t in TOOLS.items()}
    try:
        for k, t in TOOLS.items():
            t.fn = (lambda name: (lambda ctx: {"ran": name}))(k)
        ctx = CaseContext(case_id="x", row={}, cfg={}, store=None, calibrator=Calibrator(), get_segmenter=lambda: None)
        trace = Rn.run_agent(ctx, pl, max_steps=12, save=False)
    finally:
        for k, fn in saved.items():
            TOOLS[k].fn = fn
    tools = [s["tool"] for s in trace["steps"]]
    assert trace["status"] == "complete"
    assert tools == ["load_study", "segment_liver_tumour", "measure_findings", "quality_check", "vlm_second_read",
                     "estimate_confidence", "triage_case", "draft_report", "finish"]
    assert trace["n_guardrail_overrides"] >= 3
    assert trace["steps"][0]["planned_by"] == "llm" and trace["steps"][1]["guardrail"]
