"""Tests for the reading interface (app/): wording helpers, HTML components and the full Streamlit app.

The app tests drive the real Streamlit script with ``streamlit.testing.v1.AppTest`` against a small
run built from LiTS-layout test CTs with stand-in models (no GPU, no download). They check the
safety rules of the design review as behaviour, not just looks:

* blind first read: no AI output (prediction, %, disagreement, AI priority) for unread cases, arrival order,
  no AI column; the AI evidence opens only after a saved impression, and is shown next to it;
* impressions are persisted at once and survive an interrupted session (resume); revisions never overwrite;
* session settings lock after the first saved impression or decision; only a new session unlocks them;
* reading uses only an explicitly authorised policy (named person, reason, confirmation), from the next session;
* leaving a case with unsaved work asks to save, discard or stay; nothing is discarded silently;
* every decision is stored with the recommendation on screen, session, policy and model-set versions;
* failures are explicit: QC, missing images, a failed save; "defer" records no final finding.

Run:  pytest -q tests/test_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import ui_core as U  # noqa: E402
import ui_theme as T  # noqa: E402


# --------------------------------------------------------------------------- wording and components
def _ratio(a, b):
    def lum(h):
        r, g, b_ = (int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
        f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
        return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b_)

    hi, lo = sorted([lum(a), lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_semantic_palette_meets_wcag_aa():
    pairs = [(T.NAVY, "#FFFFFF"), (T.INK, "#FFFFFF"), (T.SLATE, "#FFFFFF"), (T.SLATE, T.SURFACE),
             (T.SLATE, T.SLATE_BG), ("#FFFFFF", T.TEAL), (T.TEAL_DARK, T.TEAL_TINT), (T.RED, T.RED_BG),
             (T.AMBER, T.AMBER_BG), (T.AMBER, "#FFFFFF"), ("#7A2E0E", T.AMBER_BG), ("#DCE6F2", T.NAVY),
             ("#FDE7B0", T.NAVY), ("#D7E1EC", T.VIEWER_BG), ("#E6EDF5", "#132540")]
    for fg, bg in pairs:
        assert _ratio(fg, bg) >= 4.5, (fg, bg, round(_ratio(fg, bg), 2))


def test_overlays_visible_on_ct_and_distinct():
    rgb = {k: np.array(v, float) for k, v in T.OVERLAY.items() if k in ("liver", "lesion", "uncertainty")}
    hexes = {k: "#%02X%02X%02X" % tuple(int(x) for x in v) for k, v in rgb.items()}
    for k, h in hexes.items():
        assert _ratio(h, "#000000") >= 7.0, k                     # visible on the dark CT background / air
    keys = list(rgb)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert np.linalg.norm(rgb[keys[i]] - rgb[keys[j]]) > 120, (keys[i], keys[j])


def test_badges_codes_and_escaping():
    shapes = {v[2] for v in T.CATEGORY_STYLE.values() if v[1] != "AI withheld (QC)"}
    assert len(shapes) == len(T.CATEGORY_STYLE) - 1              # distinct shapes (withheld kinds share a family)
    for cat, (_, label, _) in T.CATEGORY_STYLE.items():
        h = T.badge(cat)
        assert label in h and "<svg" in h and 'aria-hidden="true"' in h
    assert "AI:" in T.badge("SUSPICIOUS")                        # AI outputs are labelled as such
    assert "&lt;b&gt;" in T.top_state("<b>", None, "S-1", False) and "Research prototype" in T.top_identity()
    assert "&lt;script&gt;" in T.withheld("<script>", [("a", "<i>")]) and T.esc("a & b") == "a &amp; b"
    assert "&lt;i&gt;" in T.comparison(1, "<i>", 0, "x") and "Different binary finding" in T.comparison(1, None, 0, "x")
    card = T.mobile_card(2, "lits_002", "pending", "segmentation", "Segmentation QC failed", "<b>x</b>", "Read <manually>")
    assert "Case 02" in card and "lits_002" in card and "Unread" in card and "&lt;manually&gt;" in card
    assert "ra-save-failed" in T.save_state("failed", "Save failed")


def test_stepper_meter_measures():
    s = T.progress(3, {1, 2})
    assert s.count("ra-pg-done") == 2 and 'aria-current="step"' in s and "Decision" in s
    ev = T.evidence(finding="AI finding: x", p=0.73, threshold=0.5, target="t", unc_level="high", unc_text="u",
                    qc_ok=True, agreement_kind="disagree", agreement_text="a", agreement_interp="i")
    assert ev.index("AI finding") < ev.index("73%") < ev.index("Uncertainty") < ev.index("Model agreement")
    assert "width:100.0%" in T.meter(1.7, 0.5) and "threshold 50%" in T.meter(0.2, 0.5)
    assert T.measures([("Liver volume", "1500", "mL")]).count("<dt>") == 1


def test_clinical_wording_helpers():
    assert U.humanize("p = 0.72 >= suggestion threshold 0.50") == "Probability 0.72 ≥ suggestion threshold 0.50"
    assert U.neighbours(["a", "b", "c"], "b") == ("a", "c") and U.neighbours(["a"], "z") == (None, "a")
    assert U.pct(0.734) == "73%"
    a = U.agreement(0.73, 0.91)
    assert a["kind"] == "differ" and "73% vs 91%" in a["text"]           # same class, different scores
    assert U.agreement(0.72, 0.29)["kind"] == "disagree"
    assert U.agreement(0.70, 0.78)["kind"] == "agree"
    assert U.agreement(0.70, None)["kind"] == "none" and "not a negative result" in U.agreement(0.7, None)["interpretation"]
    assert all(U.agreement(a, b)["interpretation"] for a, b in ((0.73, 0.91), (0.72, 0.29), (0.7, 0.78)))
    assert "0.84 on a 0-1 scale" in U.uncertainty_phrase(0.84)
    rec = {"quality": {"passed": False, "checks": [{"name": "tumour_inside_liver", "passed": False,
                                                    "message": "43% (167 ml) of tumour-class voxels fall outside"}]}}
    q = U.qc_status(rec)
    assert q["label"] == "Segmentation QC failed" and "withheld" in q["consequence"] and "43%" in q["detail"]
    assert "not of the CT acquisition" in q["consequence"] and "manually" in q["manual"]
    rec["quality"]["checks"].append({"name": "enough_slices", "passed": False, "message": "12 axial slices"})
    assert U.qc_status(rec)["label"] == "Image QC failed" and "series" in U.qc_status(rec)["manual"]
    assert U.qc_status({"quality": {"passed": True}})["kind"] == "ok"


def test_viewer_rendering_window_zoom_markers():
    lo, hi = U.window_to_u8(350, 75, (-100, 250))
    assert abs(lo) < 1e-6 and abs(hi - 255) < 1e-6
    arrs = {"ct": np.full((3, 64, 64), 100, np.uint8), "liver": np.zeros((3, 64, 64), np.uint8),
            "lesions": np.zeros((3, 64, 64), np.uint8), "entropy": np.zeros((3, 64, 64), np.uint8), "z": np.arange(3)}
    arrs["lesions"][1, 20:30, 20:30] = 1
    plain = U.render_slice(arrs, 1, upscale=2)
    zoomed = U.render_slice(arrs, 1, upscale=2, zoom=2.0, markers=True)
    assert plain.shape == zoomed.shape == (128, 128, 3) and not np.array_equal(plain, zoomed)
    no_fill = U.render_slice(arrs, 1, upscale=1, lesion_alpha=0.0, show_liver=False)
    filled = U.render_slice(arrs, 1, upscale=1, lesion_alpha=0.5, show_liver=False)
    assert tuple(no_fill[25, 25]) == tuple(no_fill[0, 0]) and tuple(filled[25, 25]) != tuple(filled[0, 0])
    assert U.neutral_slice_index(arrs) == 1


def test_blind_order_ignores_ai_priority():
    import pandas as pd

    wl = pd.DataFrame({"case_id": ["lits_009", "lits_002", "lits_005"], "rank": [1, 2, 3], "arrival": [9, 2, 5]})
    assert list(U.order_worklist(wl, blind=True).case_id) == ["lits_002", "lits_005", "lits_009"]
    assert list(U.order_worklist(wl, blind=False).case_id) == ["lits_009", "lits_002", "lits_005"]


def test_impressions_and_authorised_policy_persist(tmp_path):
    from radagent.config import load_config
    from radagent.store import RunStore

    store = RunStore(tmp_path / "run")
    U.save_impression(store, "live", "lits_001", 1, "S-1")
    U.save_impression(store, "live", "lits_001", 0, "S-2")                  # never replaces the original
    imp = U.load_impressions(store, "live")
    assert imp["lits_001"]["value"] == 1 and imp["lits_001"]["session_id"] == "S-1"
    cfg = load_config(str(ROOT / "configs" / "lits.yaml"))
    pol, entry = U.authorised_policy(store, cfg, "reader_a", "reader_threshold")
    assert entry is None and not pol.personalised and any("no authorised" in n for n in pol.notes)
    new = U.universal_policy(cfg)
    new.threshold, new.personalised, new.n_feedback = 0.62, True, 30
    e = U.authorise_policy(store, "reader_a", "reader_threshold", new, "Dr Test (QA lead)", "offline review", "S-1",
                           U.policy_version(pol))
    pol2, entry2 = U.authorised_policy(store, cfg, "reader_a", "reader_threshold")
    assert entry2["authorised_by"] == "Dr Test (QA lead)" and abs(pol2.threshold - 0.62) < 1e-9
    assert U.policy_version(pol2) == e["version"]
    log = (store.p("feedback") / "policy_changes.jsonl").read_text().splitlines()
    assert json.loads(log[-1])["authorised_by"] == "Dr Test (QA lead)"
    U.clear_impressions(store, "live")
    assert U.load_impressions(store, "live") == {}


# --------------------------------------------------------------------------- the Streamlit app
st_testing = pytest.importorskip("streamlit.testing.v1")


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    from radagent import pipeline as P
    from tests.test_lits_pipeline import _e2e_cfg

    cfg = _e2e_cfg(tmp_path_factory.mktemp("apprun"))
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        for step in (P.step_data, P.step_model, P.step_infer, P.step_vlm, P.step_calibrate):
            step(cfg)
        P.step_first_read(cfg, "rule")
        P.step_simulate(cfg, 2)
        P.step_figures(cfg)
    finally:
        P.release_models()
        os.chdir(cwd)
    return Path(cfg["output_root"]) / cfg["run_name"]


def _new_app(run_dir):
    at = st_testing.AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.fixture
def app(run_dir, monkeypatch):
    monkeypatch.setenv("RA_RUN", str(run_dir))
    monkeypatch.chdir(ROOT)
    d = run_dir / "feedback"
    if d.exists():
        for f in d.glob("*"):
            f.unlink()
    return _new_app(run_dir)


def _md(at) -> str:
    return "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def _records(run_dir: Path) -> dict:
    out = {}
    for f in (run_dir / "cases").glob("*.json"):
        r = json.loads(f.read_text())
        out[r["case_id"]] = r
    return out


def _events(run_dir: Path, reader: str = "live") -> list:
    from radagent.readers.feedback import FeedbackStore

    return FeedbackStore(run_dir / "feedback").events(reader)


def _order(at) -> list:
    return [b.key[5:] for b in at.button if b.key and b.key.startswith("open_")]


def _ok_case(app, run_dir, i: int = 0) -> str:
    recs = _records(run_dir)
    return [c for c in _order(app) if recs[c]["quality"]["passed"]][i]


def _impression(app, cid: str, value: str = "present") -> None:
    app.segmented_control(key=f"imp_{cid}").set_value(value).run()
    app.button(key="record_imp").click().run()
    assert not app.exception, app.exception


def test_blind_worklist_shows_no_ai_output(app, run_dir):
    assert app.session_state["view"] == "worklist" and app.session_state["reading_mode"] == "blind"
    assert len(app.sidebar.children) == 0                                  # no persistent sidebar
    md = _md(app)
    assert "AI findings hidden" in md and "AI finding</span>" not in md    # no AI column in blind mode
    summary = next(m.value for m in app.markdown if 'class="ra-summary"' in m.value)
    assert "disagree" not in summary and "urgent" not in summary.lower()
    cells = [m.value for m in app.markdown if m.value.startswith(('<div class="ra-cell"', '<div class="ra-mcard"'))]
    assert cells and all("%" not in v for v in cells)
    order = _order(app)
    assert order == sorted(order, key=lambda c: int(c.split("_")[1]))      # arrival order, not AI priority
    assert all(app.button(key=f"open_{c}").proto.type != "primary" for c in order)
    assert app.button(key="next_unread").proto.type == "primary"            # one primary action
    assert all(app.button(key=f"mopen_{c}") for c in order)                # separate mobile cards


def test_blind_journey_impression_reveal_decide_revise(app, run_dir):
    cid = _ok_case(app, run_dir)
    app.button(key=f"open_{cid}").click().run()
    assert app.session_state["view"] == "read" and app.session_state["case"] == cid
    md = _md(app)
    assert "Independent read" in md and 'class="ra-ev"' not in md and "AI overlays locked" in md
    assert all(t.disabled for t in app.toggle if t.key in ("ov_liver", "ov_lesions", "ov_unc"))
    assert not app.selectbox(key="reader").disabled
    _impression(app, cid)
    assert U.load_impressions(U.RunStore(run_dir), "live")[cid]["value"] == 1     # persisted at once
    assert app.session_state["session_locked"] and app.selectbox(key="reader").disabled
    md = _md(app)
    assert 'class="ra-ev"' in md and "Your impression" in md and "Patient-level probability" in md
    assert "Same binary finding" in md or "Different binary finding" in md
    app.segmented_control(key=f"final_{cid}").set_value("present").run()
    assert "Unsaved changes" in _md(app)
    app.button(key="save_decision").click().run()
    ev = _events(run_dir)[-1]
    assert ev["case_id"] == cid and ev["initial_read"] == 1 and ev["final_label"] == 1 and ev["supersedes"] is None
    assert ev["action"] in ("accept", "reject") and ev["recommendation"]["text"]
    assert ev["session_id"].startswith("S-") and ev["policy_version"].startswith("P-") \
        and ev["model_version"].startswith("M-") and ev["impression_at"]
    md = _md(app)
    assert "Research decision saved · report draft not signed" in md and f"case <b>{cid}</b>" in md
    app.button(key="revise").click().run()
    app.segmented_control(key=f"final_{cid}").set_value("absent").run()
    app.button(key="save_decision").click().run()
    evs = [e for e in _events(run_dir) if e["case_id"] == cid]
    assert len(evs) == 2 and evs[-1]["supersedes"] == evs[0]["timestamp"] and evs[-1]["initial_read"] == 1
    assert "Revises the decision saved" in _md(app)
    app.button(key="after_report").click().run()
    assert app.session_state["view"] == "report"
    rep = [c.value for c in app.code if "RADIOLOGIST REVIEW" in c.value]
    assert rep and "AI recommendation shown" in rep[0] and "not signed" in rep[0] and "revises" in rep[0]


def test_unsaved_work_guard(app, run_dir):
    order = _order(app)
    a, b = _ok_case(app, run_dir, 0), _ok_case(app, run_dir, 1)
    after_a = order[order.index(a) + 1]
    app.button(key=f"open_{a}").click().run()
    app.segmented_control(key=f"imp_{a}").set_value("absent").run()
    app.button(key="next_case").click().run()                              # N / next case with unsaved work
    assert app.session_state["case"] == a and app.session_state["pending_nav"]["case"] == after_a
    assert "Unsaved work on" in _md(app)
    app.button(key="pending_stay").click().run()
    assert app.session_state["case"] == a and "pending_nav" not in app.session_state
    app.button(key="next_case").click().run()
    app.button(key="pending_discard").click().run()                       # explicit discard
    assert app.session_state["case"] == after_a and a not in U.load_impressions(U.RunStore(run_dir), "live")
    app.selectbox(key="case_pick").set_value(b).run()
    assert app.session_state["case"] == b
    app.segmented_control(key=f"imp_{b}").set_value("present").run()
    app.segmented_control(key="nav").set_value("worklist").run()            # mode switch is guarded too
    assert app.session_state["view"] == "read" and app.session_state["pending_nav"]["view"] == "worklist"
    app.button(key="pending_save").click().run()                          # save and continue
    assert app.session_state["view"] == "worklist" and b in U.load_impressions(U.RunStore(run_dir), "live")


def test_interrupted_session_resumes_with_original_impression(app, run_dir, monkeypatch):
    cid = _ok_case(app, run_dir, 1)
    app.button(key=f"open_{cid}").click().run()
    _impression(app, cid, "absent")
    first_session = app.session_state["session_id"]
    again = _new_app(run_dir)                                              # browser closed and reopened
    assert again.session_state["session_id"] != first_session and not again.session_state["session_locked"]
    again.button(key=f"open_{cid}").click().run()
    md = _md(again)
    assert "Resumed." in md and first_session in md and 'class="ra-ev"' in md
    assert not [w for w in again.segmented_control if w.key == f"imp_{cid}"]   # the impression is not asked again


def test_qc_failure_and_defer(app, run_dir):
    recs = _records(run_dir)
    order = _order(app)
    qc_fail = [c for c in order if not recs[c]["quality"]["passed"]]
    if qc_fail:
        app.button(key=f"open_{qc_fail[0]}").click().run()
        md = _md(app)
        assert "AI output withheld" in md and "What it means" in md and "What you can do" in md
        assert "QC failed - AI masks not shown" in md and "not shown because quality control failed" in md
        app.segmented_control(key=f"final_{qc_fail[0]}").set_value("absent").run()
        app.button(key="save_decision").click().run()
        assert _events(run_dir)[-1]["action"] == "manual"
    ok = [c for c in order if recs[c]["quality"]["passed"]]
    app.session_state["case"], app.session_state["view"] = ok[-1], "read"
    app.run()
    _impression(app, ok[-1], "absent")
    app.radio(key=f"defer_reason_{ok[-1]}").set_value("prior_imaging").run()
    app.button(key="confirm_defer").click().run()
    ev = _events(run_dir)[-1]
    assert ev["action"] == "defer" and ev["final_label"] is None and "Prior" in ev["edit"]["defer_reason"]
    assert "not decided (deferred)" in _md(app)
    app.segmented_control(key="nav").set_value("worklist").run()
    assert "Deferred · second opinion" in _md(app)


def test_failures_are_explicit(app, run_dir, monkeypatch):
    cid = _ok_case(app, run_dir)
    order = _order(app)
    app.button(key=f"open_{cid}").click().run()
    _impression(app, cid)
    app.segmented_control(key=f"final_{cid}").set_value("absent").run()

    def broken(*a, **k):
        raise OSError("disk full")

    original = U.log_live
    monkeypatch.setattr(U, "log_live", broken)
    app.button(key="save_decision").click().run()
    md = _md(app)
    assert not app.exception and "Save failed" in md and "disk full" in md and _events(run_dir) == []
    monkeypatch.setattr(U, "log_live", original)
    other = [c for c in order if c != cid][0]
    vp = run_dir / "viewer" / f"{other}.npz"
    moved = vp.with_suffix(".bak")
    vp.rename(moved)
    try:
        app.session_state["case"] = other
        app.run()
        md = _md(app)
        assert "could not be loaded" in md and "not a negative result" in md and not app.exception
    finally:
        moved.rename(vp)


def test_report_and_audit_stay_closed_before_impression(app, run_dir):
    cid = _order(app)[0]
    app.session_state["case"], app.session_state["view"] = cid, "report"
    app.run()
    assert "available after you save your independent impression" in _md(app)
    assert not [c for c in app.code if "RADIOLOGIST REVIEW" in c.value]
    app.button(key="go_audit").click().run()
    assert app.session_state["view"] == "audit" and "opens after your impression" in _md(app)


def test_policy_needs_authorisation_and_applies_next_session(app, run_dir):
    app.selectbox(key="reader").set_value("reader_b").run()
    key = next(k for k in app.session_state["active_policies"] if k.startswith("reader_b|"))
    before = U.policy_version(app.session_state["active_policies"][key]["policy"])
    app.button(key="go_adapt").click().run()
    app.number_input(key="nmore").set_value(6).run()
    app.button(key="simn").click().run()
    assert not app.exception and 1 <= len(_events(run_dir, "reader_b")) <= 6
    assert U.policy_version(app.session_state["active_policies"][key]["policy"]) == before   # no silent change
    assert app.button(key="apply_policy").disabled                         # nobody has authorised it
    app.text_input(key="auth_by").input("Dr Test (QA lead)").run()
    app.text_input(key="auth_reason").input("offline evaluation reviewed").run()
    app.checkbox(key="auth_ok").check().run()
    btn = app.button(key="apply_policy")
    if not btn.disabled:
        btn.click().run()
        stored = json.loads((run_dir / "feedback" / "authorised_policies.json").read_text())
        entry = next(v for k, v in stored.items() if k.startswith("reader_b|"))
        assert entry["authorised_by"] == "Dr Test (QA lead)" and entry["reason"]
        log = (run_dir / "feedback" / "policy_changes.jsonl").read_text().splitlines()
        assert json.loads(log[-1])["authorised_by"] == "Dr Test (QA lead)"
        assert U.policy_version(app.session_state["active_policies"][key]["policy"]) == before   # not this session
        app.button(key="new_session").click().run()
        assert U.policy_version(app.session_state["active_policies"][key]["policy"]) == entry["version"]
    app.checkbox(key="reset_ok").check().run()
    app.button(key="reset_confirm").click().run()
    assert _events(run_dir, "reader_b") == []


def test_session_lock_and_new_session(app, run_dir):
    cid = _ok_case(app, run_dir)
    app.button(key=f"open_{cid}").click().run()
    _impression(app, cid)
    sid = app.session_state["session_id"]
    assert all(w.disabled for w in (app.selectbox(key="reader"), app.radio(key="reading_mode"),
                                    app.radio(key="policy"), app.multiselect(key="splits")))
    app.button(key="new_session").click().run()
    assert app.session_state["session_id"] != sid and not app.radio(key="reading_mode").disabled
    assert "A new session started" in _md(app)


def test_assisted_mode_and_every_view(app):
    app.radio(key="reading_mode").set_value("assisted").run()
    md = _md(app)
    assert "AI-assisted triage" in md and "AI models disagree" in md and "%" in md and "AI finding</span>" in md
    app.selectbox(key="wl_sort").set_value("arrival").run()
    assert not app.exception
    for reader in ("live", "reader_a"):
        app.selectbox(key="reader").set_value(reader).run()
        for view in ("worklist", "read"):
            app.segmented_control(key="nav").set_value(view).run()
            assert not app.exception, (reader, view, app.exception)
        for v in ("adapt", "perf", "audit", "about"):
            app.button(key=f"go_{v}").click().run()
            assert not app.exception and app.session_state["view"] == v, (reader, v, app.exception)
            assert app.session_state["nav"] is None                     # research area is outside the two modes
        app.button(key="back_wl").click().run()
        assert app.session_state["view"] == "worklist"
    app.segmented_control(key="nav").set_value(None).run()             # deselecting keeps the page
    assert app.session_state["view"] == "worklist" and not app.exception


def test_simulated_reader_in_read_view(app, run_dir):
    app.selectbox(key="reader").set_value("reader_c").run()
    app.segmented_control(key="nav").set_value("read").run()
    cid = app.session_state["case"]
    app.button(key=f"sim_{cid}").click().run()
    ev = _events(run_dir, "reader_c")
    assert not app.exception and len(ev) == 1 and ev[0]["session_id"].startswith("S-")


def test_worklist_and_viewer_controls(app, run_dir):
    app.text_input(key="wl_search").input("lits_0").run()
    assert not app.exception
    app.segmented_control(key="wl_filter").set_value("decided").run()
    assert not app.exception
    app.segmented_control(key="wl_filter").set_value("all").run()
    app.toggle(key="wl_details").set_value(True).run()
    app.radio(key="reading_mode").set_value("assisted").run()
    cid = _ok_case(app, run_dir)
    app.button(key=f"mopen_{cid}").click().run()                          # mobile card action opens the case
    assert app.session_state["view"] == "read" and app.session_state["case"] == cid
    s0 = app.session_state[f"slice_{cid}"]
    app.button(key="slice_next").click().run()
    assert app.session_state[f"slice_{cid}"] != s0 or not app.exception
    app.radio(key="window").set_value("custom").run()
    app.slider(key="zoom").set_value(2.0).run()
    assert "W 150 / L 70 HU" in _md(app)
    app.button(key="orig_btn").click().run()
    assert app.session_state["orig_ct"] and "Original CT - all AI overlays hidden" in _md(app)
    app.button(key="orig_btn").click().run()
    app.radio(key="layout").set_value("large").run()
    app.button(key="reset_view").click().run()
    assert not app.exception and app.session_state["window"] == "liver" and app.session_state["zoom"] == 1.0
    assert app.session_state["layout"] == "large"                          # layout is a session preference


@pytest.fixture(scope="module")
def single_model_run(tmp_path_factory):
    """The project's default set-up: one segmentation model and no second reader."""
    from radagent import pipeline as P
    from tests.test_lits_pipeline import _e2e_cfg

    cfg = _e2e_cfg(tmp_path_factory.mktemp("onemodel"))
    cfg["vlm"]["backend"] = "none"
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        for step in (P.step_data, P.step_model, P.step_infer, P.step_vlm, P.step_calibrate):
            step(cfg)
        P.step_first_read(cfg, "rule")
    finally:
        P.release_models()
        os.chdir(cwd)
    return Path(cfg["output_root"]) / cfg["run_name"]


def test_one_model_shows_no_second_reader_evidence(single_model_run, monkeypatch):
    monkeypatch.setenv("RA_RUN", str(single_model_run))
    monkeypatch.chdir(ROOT)
    at = _new_app(single_model_run)
    recs = _records(single_model_run)
    cid = next(c for c in _order(at) if recs[c]["quality"]["passed"])
    at.button(key=f"open_{cid}").click().run()
    assert "agreement between the two AI models" not in _md(at)
    _impression(at, cid)
    md = _md(at)
    assert 'class="ra-ev"' in md and "Model agreement" not in md and "AI models disagree" not in md
    assert not [e for e in at.expander if "Second-reader" in e.label]
    assert "pretrained, not trained in this project" in md
    at.radio(key="reading_mode")  # exists; settings are locked after the impression
    at.button(key="new_session").click().run()
    at.radio(key="reading_mode").set_value("assisted").run()
    at.segmented_control(key="nav").set_value("worklist").run()
    summary = next(m.value for m in at.markdown if 'class="ra-summary"' in m.value)
    assert "urgent by AI rule" in summary and "disagree" not in summary
    at.button(key="go_about").click().run()
    assert not at.exception and "Second-reader model" not in _md(at)


def test_onboarding_without_runs(tmp_path, monkeypatch):
    if U.find_runs(ROOT / "outputs"):
        pytest.skip("this checkout has runs in outputs/")
    monkeypatch.delenv("RA_RUN", raising=False)
    monkeypatch.chdir(tmp_path)
    at = st_testing.AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception and "No reading session yet" in _md(at)
