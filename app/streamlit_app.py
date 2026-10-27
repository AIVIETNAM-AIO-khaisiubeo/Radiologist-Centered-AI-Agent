"""ReaderAdapt-Liver - the radiologist's reading interface.

    python -m streamlit run app/streamlit_app.py -- --run outputs/lits

(or set the environment variable RA_RUN=outputs/lits). Data: LiTS17. One pretrained AI model: NVIDIA
VISTA3D (MONAI) for liver and liver-tumour segmentation; nothing is trained in this project.

Layout: one compact top bar (identity, the two working modes Worklist / Read case, a *Session*
menu and a *Research* (research and quality) menu) - no persistent sidebar, so the CT viewer gets the width.
Design notes and the responses to the design reviews: docs/UX_DESIGN.md.

Safety rules built into the flow:
* Blind first read (default): no AI output - prediction, overlays, uncertainty, model disagreement,
  AI priority - is shown for a case before the radiologist saves an independent impression; the
  worklist is in arrival order. Impressions are written to disk at once (interruption recovery) and
  the original impression is never replaced.
* Session settings (reader, reading mode, policy, worklist, run) lock after the first saved
  impression or decision; only a new session unlocks them.
* Reading sessions use only an explicitly authorised decision policy (named person, reason,
  confirmation; versioned and logged); an authorisation takes effect from the next session.
* Every decision is stored with the recommendation that was on screen and the session, policy and
  model-set versions; a revision records what it supersedes.
* Leaving a case with unsaved work asks: save, discard or stay. Nothing is discarded silently.
* Failures (images, QC, saving) are stated explicitly and never look like a negative result.

Research prototype. Every output is a research prediction, not for clinical use.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR.parent))

import ui_core as U  # noqa: E402
import ui_theme as T  # noqa: E402
from radagent.experiments import POLICY_LABELS  # noqa: E402
from radagent.plots import tool_names  # noqa: E402
from radagent.readers.feedback import FeedbackStore  # noqa: E402
from radagent.report import draft_report  # noqa: E402
from radagent.triage import CATEGORY_HELP  # noqa: E402
from radagent.vlm import vlm_enabled  # noqa: E402

PRIMARY = {"worklist": ":material/format_list_bulleted: Worklist", "read": ":material/visibility: Read case"}
RESEARCH = {"adapt": ("Reader adaptation", ":material/tune:"), "perf": ("Model performance", ":material/monitoring:"),
            "audit": ("Audit trail (this case)", ":material/fact_check:"),
            "about": ("About and intended use", ":material/info:")}
WINDOWS = {"abdomen": ("Abdomen", 350, 75), "liver": ("Liver", 150, 70), "lesion": ("Lesion", 100, 60)}
FINAL = {"present": "Tumour present", "absent": "No liver tumour"}
DEFER_REASONS = {"second_opinion": "Second opinion needed", "prior_imaging": "Prior or additional imaging needed",
                 "image_quality": "Image quality or wrong series", "other": "Other (see note)"}
LAYOUTS = {"standard": "Standard · image about 60 % of the width", "large": "Large image · about 70 % of the width"}
VIEW_DEFAULTS = {"window": "liver", "zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0, "ov_liver": True, "ov_lesions": True,
                 "ov_unc": False, "op_lesion": 0, "op_unc": 70, "wl_w": 150, "wl_l": 70, "layout": "standard",
                 "orig_ct": False}
QC_HINT = {"segmentation": "AI output withheld · images can be read manually",
           "image": "Check the series before reading"}

st.set_page_config(page_title="ReaderAdapt · Liver CT", page_icon="🩻", layout="wide",
                   initial_sidebar_state="collapsed")


def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def cols(name: str, spec, **kw):
    """Columns that reflow by the space they actually have (ui_theme.REFLOW), not only by screen width."""
    return st.container(key=f"rx_{name}").columns(spec, **kw)


def now_local() -> str:
    return datetime.now().astimezone().strftime("%H:%M")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def local_hm(ts: str | None) -> str:
    try:
        return datetime.fromisoformat(str(ts)).astimezone().strftime("%H:%M")
    except (TypeError, ValueError):
        return str(ts or "-")[:16]


def _new_session_id() -> str:
    return "S-" + datetime.now().strftime("%m%d-%H%M") + "-" + secrets.token_hex(2).upper()


# --------------------------------------------------------------------------- run selection
def _cli_run() -> str | None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--run", default=None)
    args, _ = ap.parse_known_args(sys.argv[1:])
    return args.run or os.environ.get("RA_RUN")


@st.cache_data(show_spinner=False)
def _load_viewer(path: str, mtime: float) -> dict:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def _cases_mtime(run: Path) -> float:
    files = list((run / "cases").glob("*.json"))
    return max((f.stat().st_mtime for f in files), default=0.0)


@st.cache_resource(show_spinner="Loading the reading session ...")
def _load_run_cached(run: str, mtime: float) -> dict:
    return U.load_run(Path(run))


ss = st.session_state
if "session_id" not in ss:
    ss.session_id = _new_session_id()
    ss.session_started = now_local()
ss.setdefault("session_locked", False)

runs = U.find_runs(APP_DIR.parent / "outputs") + U.find_runs("outputs")
runs = list(dict.fromkeys(p.resolve() for p in runs))
cli = _cli_run()
if cli and Path(cli).resolve() not in runs and (Path(cli) / "cases").exists():
    runs.insert(0, Path(cli).resolve())

html(T.style_tag())

if not runs:
    with st.container(key="topbar"):
        html(T.top_identity())
    html(T.page_title("No reading session yet",
                      "The website shows the results of a pipeline run on the LiTS17 liver CT data."))
    html(T.note("No finished run was found in <code>outputs/</code>.", "warn"))
    c1, c2 = cols("onboard", 2, gap="large")
    with c1:
        st.markdown("""
#### Create a run on Google Colab (GPU)
1. Open `notebooks/colab_lits_pipeline.ipynb` in Colab with a GPU runtime.
2. Add the secret `KAGGLE_API_TOKEN` (LiTS download). The pretrained NVIDIA VISTA3D weights are downloaded
   from Hugging Face automatically (no token needed).
3. Run all cells. The last cells open this website from Colab or download `lits_run.zip`.
""")
    with c2:
        st.markdown("""
#### Open a run on this computer
1. Unzip `lits_run.zip` into the `outputs/` folder of this repository.
2. Start `python -m streamlit run app/streamlit_app.py -- --run outputs/lits`.

On a GPU machine you can also run every step yourself: `python -m radagent all --config configs/lits.yaml`.
""")
    st.stop()

# --------------------------------------------------------------------------- top bar: identity · modes · session · research
locked = bool(ss.session_locked)
with st.container(key="topbar"):
    tb = cols("topbar", [2.2, 1.4, 0.62, 0.9], vertical_alignment="center")
    id_slot = tb[0].empty()
    nav_slot = tb[1].container()
    sess_pop = tb[2].popover("Session", icon=":material/badge:", width="stretch",
                             help="Reader, reading mode, decision policy, worklist and session details")
    res_pop = tb[3].popover("Research", icon=":material/science:", width="stretch",
                            help="Research and quality: reader adaptation, model performance, audit trail, intended use")

with sess_pop:
    html("<div class='ra-panel-t'>Session settings</div>")
    if locked:
        html(T.note(f"{T.icon('lock', 13)} <b>Locked for this session</b> after your first saved impression or "
                    "decision, so the reading conditions cannot change mid-session. Start a new session (below) to "
                    "change them.", "neutral"))
    default_idx = 0
    if cli:
        for i, r in enumerate(runs):
            if r == Path(cli).resolve():
                default_idx = i
    run = st.selectbox("Results run", runs, index=default_idx, format_func=lambda p: p.name, key="run_sel",
                       disabled=locked, help="A finished pipeline run (records, images, calibration, experiments).")

R = _load_run_cached(str(run), _cases_mtime(run))
cfg, store, records, cases = R["cfg"], R["store"], R["records"], R["cases"]
profiles = U.reader_profiles(cfg)
reader_options = [U.LIVE_READER] + list(profiles)
reader_names = {**{k: v.name for k, v in profiles.items()}, U.LIVE_READER: "You"}

with sess_pop:
    reader_id = st.selectbox("Reading as", reader_options, key="reader", disabled=locked,
                             format_func=lambda r: "You (live reader)" if r == U.LIVE_READER
                             else f"{reader_names[r]} (simulated)",
                             help="'You' records your own decisions. Readers A, B and C are simulated readers used "
                                  "to study adaptation; their reads are research simulations.")
    live = reader_id == U.LIVE_READER
    if live:
        mode = st.radio("Reading mode", ["blind", "assisted"], key="reading_mode", disabled=locked,
                        format_func={"blind": "Blind first read", "assisted": "AI-assisted triage"}.get,
                        help="Blind first read: no AI output is shown for a case until you save your own "
                             "impression, and the worklist is in arrival order. AI-assisted triage: the AI suggestion "
                             "and AI priority are shown from the start.")
    else:
        mode = "assisted"
        st.caption(profiles[reader_id].description)
    policy_name = st.radio("Decision policy", list(U.POLICY_CHOICES), index=3, key="policy", disabled=locked,
                           format_func=lambda p: U.POLICY_CHOICES[p],
                           help="How the AI suggestion is shown. Personalised policies are used only after they have "
                                "been authorised (Research → Reader adaptation); until then the universal "
                                "policy is used.")
    splits = st.multiselect("Worklist", ["test", "calib"], default=["test"], key="splits", disabled=locked,
                            format_func={"test": "Held-out test cases", "calib": "Calibration cases"}.get)
blind = live and mode == "blind"

# --------------------------------------------------------------------------- authorised (frozen) policy, versions, worklist
events_by_reader = U.all_events(store, reader_options)
proposed = U.current_policy(policy_name, reader_id, events_by_reader, cases, cfg)
active = ss.setdefault("active_policies", {})
pkey = f"{reader_id}|{policy_name}|{run.name}"
if pkey not in active:
    pol0, auth0 = U.authorised_policy(store, cfg, reader_id, policy_name)
    active[pkey] = {"policy": pol0, "since": now_local(), "auth": auth0}
policy = active[pkey]["policy"]
auth = active[pkey].get("auth")
pol_ver = U.policy_version(policy)
mv = U.model_version(cfg, R.get("calibration"), R.get("model_card"))
reviews = FeedbackStore(store.p("feedback")).latest_by_case(reader_id)
impressions = U.load_impressions(store, reader_id) if live else {}
wl = U.worklist(records, cases, policy, cfg, reviews, splits=tuple(splits or ["test"]), impressions=set(impressions))
reader_name = reader_names[reader_id]
n_cal = ((R.get("calibration") or {}).get("fit") or {}).get("n")


def _policy_source() -> str:
    if auth:
        return (f"Authorised by {auth.get('authorised_by')} on {str(auth.get('authorised_at'))[:10]} "
                f"({auth.get('reason')}).")
    if policy_name == "universal":
        return "Universal policy (no personalisation)."
    return "Universal policy in use: no personalised policy has been authorised for this reader yet."


# --------------------------------------------------------------------------- navigation with an unsaved-work guard
def _dirty() -> str | None:
    """What would be lost by leaving the current case now (None = nothing unsaved)."""
    if ss.get("view") != "read" or not live:
        return None
    cid = ss.get("case")
    if ss.get(f"imp_{cid}") in FINAL and cid not in impressions and cid not in reviews:
        return "an impression that is not saved yet"
    if ss.get(f"final_{cid}") is not None or str(ss.get(f"note_{cid}") or "").strip():
        if cid not in reviews or ss.get(f"revise_{cid}"):
            return "a final finding or note that is not saved yet"
    return None


def _perform(view: str | None, cid: str | None, action: str | None = None) -> None:
    ss.pop("pending_nav", None)
    if action == "new_session":
        _new_session()
        return
    if cid and cid != ss.get("case"):
        ss.case = cid
        for k in ("zoom", "pan_x", "pan_y"):          # zoom and pan are per case; window / level is per session
            ss[k] = VIEW_DEFAULTS[k]
    if view:
        ss.view = view
        if view == "read":
            ss.opened = True


def _request(view: str | None = None, cid: str | None = None, action: str | None = None) -> None:
    target = cid or ss.get("case")
    staying = action is None and view in (None, "read") and target == ss.get("case") and ss.get("view") == "read"
    what = _dirty()
    if what and not staying:
        ss.pending_nav = {"view": view, "case": target, "action": action, "what": what, "from": ss.get("case")}
        return
    _perform(view, target, action)


def _nav_changed() -> None:
    target, current = ss.nav, ss.view
    if target is None:
        if current in PRIMARY:              # clicking the active mode again must not leave the page empty
            ss.nav = current
        return
    _request(target)
    if ss.view != target:
        ss.nav = current if current in PRIMARY else None


def _pick_changed() -> None:
    target = ss.case_pick
    _request("read", target)
    if ss.case != target:
        ss.case_pick = ss.case


def _new_session() -> None:
    ss.session_id = _new_session_id()
    ss.session_started = now_local()
    ss.session_locked = False
    ss.active_policies = {}
    for k in [k for k in list(ss.keys()) if str(k).startswith("revise_")]:
        del ss[k]
    ss.pop("save_error", None)
    ss.session_note = "A new session started. Settings are unlocked and the latest authorised policy is loaded."


def _reset_feedback(rid: str) -> None:
    FeedbackStore(store.p("feedback")).reset(rid)
    U.clear_impressions(store, rid)
    ss.reset_ok = False


with sess_pop:
    st.divider()
    html("<div class='ra-panel-t'>Session details</div>"
         f"<div class='ra-panel-s'>Started {T.esc(ss.session_started)} · decision policy active since "
         f"{T.esc(active[pkey]['since'])}</div>")
    st.code(f"Session    {ss.session_id}\nModel set  {mv['id']}\nPolicy     {pol_ver}", language=None)
    st.caption(_policy_source())
    st.button("Start a new session", key="new_session", icon=":material/restart_alt:", width="stretch",
              on_click=_request, kwargs={"action": "new_session"})
    st.caption("A new session unlocks the settings and loads the latest authorised policy. Saved impressions and "
               "decisions are kept.")

with res_pop:
    html("<div class='ra-panel-t'>Research and quality</div>"
         "<div class='ra-panel-s'>Governance and evaluation - not part of reading a case.</div>")
    for v, (label, ico) in RESEARCH.items():
        st.button(label, key=f"go_{v}", icon=ico, type="tertiary", on_click=_request, args=(v,), width="stretch")
    research = st.toggle("Show the reference annotation", value=False, key="research",
                         help="Research mode: shows the dataset's reference annotation and whether each decision was "
                              "correct. Keep it off during a real read.")
    st.divider()
    n_ev = len(events_by_reader.get(reader_id, []))
    html(f"<div class='ra-panel-t'>Delete logged decisions</div><div class='ra-panel-s'>{n_ev} logged decision(s) "
         f"and the saved impressions of {T.esc(reader_name)}. Cannot be undone.</div>")
    ok_del = st.checkbox("I understand that this deletes them permanently", key="reset_ok", disabled=n_ev == 0)
    st.button("Delete decisions", key="reset_confirm", icon=":material/delete:", disabled=not ok_del or n_ev == 0,
              on_click=_reset_feedback, args=(reader_id,), width="stretch")
    st.caption(f"Run: `{run.name}`")

mode_label = None if not live else ("Blind first read" if blind else "AI-assisted triage")
state = T.top_state(reader_name if live else f"{reader_name} (simulated)", mode_label,
                    ss.session_id[:11] + "…", locked)
if research:
    state = state.replace("</div>", " · <span class='ra-t-caution'>reference shown (research)</span></div>")
id_slot.markdown(T.top_identity() + state, unsafe_allow_html=True)

if wl.empty:
    html(T.note("No cases in the selected worklist. Choose a worklist in <b>Session</b>.", "warn"))
    st.stop()

sort_mode = ss.get("wl_sort", "priority")
wlo = U.order_worklist(wl, blind, sort_mode)
order = list(wlo.case_id)
if "view" not in ss:
    qv, qc_ = st.query_params.get("view"), st.query_params.get("case")
    ss.view = qv if qv in PRIMARY or qv in RESEARCH or qv == "report" else "worklist"
    if qc_ in order:
        ss.case = qc_
if ss.get("case") not in order:
    ss.case = order[0]
if ss.view == "read":
    ss.opened = True
case_id = ss.case

ss.nav = ss.view if ss.view in PRIMARY else None
with nav_slot:
    st.segmented_control("Mode", list(PRIMARY), format_func=PRIMARY.get, key="nav", on_change=_nav_changed,
                         label_visibility="collapsed", width="stretch")
st.query_params["view"] = ss.view
st.query_params["case"] = case_id
if "session_note" in ss:
    html(T.note(T.esc(ss.pop("session_note")), "neutral"))

record = records[case_id]
case = cases[case_id]
decision = policy.decide(case["p"], case["u"], case["qc_passed"])
row = wlo.set_index("case_id").loc[case_id]
seg_name, vlm_name = tool_names(cfg)
second = vlm_enabled(cfg)          # an optional second reader; off by default (one model)
card = R.get("model_card") or {}


def _hide_ai(cid: str) -> bool:
    """Blind first read: AI output for a case stays hidden until an impression or a decision exists."""
    return blind and cid not in impressions and cid not in reviews


def _case_label(cid: str) -> str:
    r = wlo.set_index("case_id").loc[cid]
    if blind:
        return f"{r['position']}. {cid} · {U.STATUS_TEXT.get(r['status'], r['status'])}"
    return (f"{r['position']}. {cid} · {U.CATEGORY_LABEL.get(r['category'], r['category'])} · "
            f"{U.pct(r['tumour_probability'])}")


def _next_unread(after: str | None) -> str | None:
    pend = [c for c, s in zip(wlo.case_id, wlo.status) if s == "pending"]
    if not pend:
        return None
    if after in order:
        later = [c for c in pend if order.index(c) > order.index(after)]
        if later:
            return later[0]
    return pend[0]


def _research_header(title: str, subtitle: str) -> None:
    c1, c2 = cols("reportbar", [4, 1.2], vertical_alignment="bottom")
    with c1:
        html("<div class='ra-crumb'>Research and quality</div>" + T.page_title(title, subtitle))
    c2.button("Back to the worklist", key="back_wl", on_click=_request, args=("worklist",),
              icon=":material/arrow_back:", width="stretch")


# =========================================================================== Worklist
def view_worklist() -> None:
    sub = ("Blind first read · cases in arrival order · AI findings stay hidden until you save your impression."
           if blind else "AI-assisted triage · ordered by AI priority (probability, lesion size, uncertainty"
                         + (", model disagreement)." if second else ")."))
    if not live:
        sub = f"Research simulation as {reader_name} · AI triage is shown."
    html(T.page_title("Worklist", sub))
    unread = wlo[wlo.status == "pending"]
    n_dec = int(wlo.status.isin(["accept", "edit", "reject", "manual"]).sum())
    workflow = [("unread", str(len(unread)))] + [(lbl, str(n)) for lbl, n in (
        ("impression saved", int((wlo.status == "impression").sum())), ("decided", n_dec),
        ("deferred", int((wlo.status == "defer").sum()))) if n]
    groups = [("workflow", workflow)]
    n_qc = int((wlo.qc_kind != "ok").sum())
    if n_qc:
        groups.append(("technical", [("QC issues", str(n_qc))]))
    if not blind:
        n_urg, n_dis = int((unread.category == "URGENT").sum()), int((unread.agree == "disagree").sum())
        groups.append(("urgency" if n_urg else "workflow", [("urgent by AI rule", str(n_urg))]))
        if second:
            groups.append(("evidence" if n_dis else "workflow", [("AI models disagree", str(n_dis))]))
    html(T.summary_line(groups))
    qc_unread = int((unread.qc_kind != "ok").sum())
    if qc_unread:
        html(T.note(f"<b>{qc_unread} unread case(s) have a technical QC failure.</b> This is not clinical urgency. "
                    "The AI output is withheld for these cases: after a segmentation failure the images can be read "
                    "manually; after an image failure check the series first.", "technical"))
    if live and policy_name != "universal" and not blind:
        changed = wlo[(wlo.category != wlo.universal_category)]
        if len(changed):
            html(T.note(f"Your authorised decision policy ({T.esc(pol_ver)}) changes the AI category of "
                        f"{len(changed)} case(s) compared with the universal policy.", "info"))
    nxt = _next_unread(ss.get("case") if ss.get("opened") else None)
    st.button(f"Open next unread case · {nxt}" if nxt else "Every case has been read", key="next_unread",
              type="primary", disabled=nxt is None, on_click=_request, args=("read", nxt),
              icon=":material/arrow_forward:", width="content")

    t1, t2, t3, t4 = cols("toolbar", [2, 1.4, 1, 0.8], vertical_alignment="bottom")
    with t1:
        flt = st.segmented_control("Show", ["all", "unread", "decided"], default="all", key="wl_filter",
                                   format_func={"all": "All", "unread": "Unread", "decided": "Reviewed"}.get) or "all"
    with t2, st.container(key="tb_search"):
        q = st.text_input("Find case", key="wl_search", placeholder="Case ID, e.g. lits_028")
    with t3, st.container(key="tb_order"):
        if blind:
            st.selectbox("Order", ["arrival"], key="wl_sort_blind", disabled=True,
                         format_func=lambda _: "Arrival order",
                         help="Blind first read always uses arrival order, so hidden AI results cannot steer your "
                              "reading sequence.")
        else:
            st.selectbox("Order", ["priority", "arrival"], key="wl_sort",
                         format_func={"priority": "AI priority", "arrival": "Arrival order"}.get)
    with t4, st.container(key="tb_details"):
        details = st.toggle("Details", key="wl_details", help="Show QC explanations and priority reasons in each row.")

    rows = wlo
    if flt == "unread":
        rows = rows[rows.status == "pending"]
    elif flt == "decided":
        rows = rows[~rows.status.isin(["pending", "impression"])]
    if q:
        rows = rows[rows.case_id.str.contains(q.strip(), case=False, regex=False)]
    if rows.empty:
        html(T.note("No cases match." if (q or flt != "unread") else "Every case on this worklist has been read.",
                    "neutral"))
        return
    current = ss.get("case") if ss.get("opened") else None
    if current in set(rows.case_id):
        html(f"<style>.st-key-row_{current} {{ background:{T.SELECT_BG} !important; "
             f"box-shadow: inset 3px 0 0 {T.NAVY}; }}</style>")
    head = ["#", "Case", "Image / AI QC"] + ([] if blind else ["AI finding"]) + ["Reading status", ""]
    spec = [0.35, 1.25, 2.3, 2.0, 1.0] if blind else [0.35, 1.1, 1.9, 2.4, 1.9, 1.0]
    with st.container(key="desklist"):
        with st.container(key="rowhead"):
            for col, h in zip(st.columns(spec, vertical_alignment="center"), head):
                col.markdown(f"<span class='ra-th'>{h}</span>", unsafe_allow_html=True)
        for r in rows.itertuples():
            with st.container(key=f"row_{r.case_id}"):
                c = st.columns(spec, vertical_alignment="center")
                c[0].markdown(f"<span class='ra-pos'>{r.position}</span>", unsafe_allow_html=True)
                cur = "<div class='ra-cell-d'>Current case</div>" if r.case_id == current else ""
                c[1].markdown(f"<span class='ra-case'>{T.esc(r.case_id)}</span>{cur}", unsafe_allow_html=True)
                qc_html = T.qc_badge(r.qc_kind, r.qc_label, "" if _hide_ai(r.case_id) else r.qc_detail)
                if r.qc_kind != "ok":
                    qc_html += f"<div class='ra-cell-d'>{T.esc(QC_HINT[r.qc_kind])}</div>"
                    if details and not _hide_ai(r.case_id):    # numeric QC details describe the AI output
                        qc_html += f"<div class='ra-cell-d'>{T.esc(r.qc_detail)}</div>"
                c[2].markdown(T.cell("QC", qc_html), unsafe_allow_html=True)
                i = 3
                if not blind:
                    ai_html = T.badge(r.category)
                    if r.category not in ("QC FAIL", "DEFERRED"):
                        ai_html += f"<span class='ra-p'>{U.pct(r.tumour_probability)}</span>"
                        if r.agree == "disagree":
                            ai_html += f"<span class='ra-flag'>{T.icon('alert', 12)} models disagree</span>"
                        elif r.agree == "differ":
                            ai_html += f"<span class='ra-flag'>{T.icon('alert', 12)} scores differ</span>"
                    if details and r.reasons:
                        ai_html += f"<div class='ra-cell-d'>{T.esc(r.reasons)}</div>"
                    c[3].markdown(T.cell("AI", ai_html), unsafe_allow_html=True)
                    i = 4
                c[i].markdown(T.cell("Status", T.status_badge(r.status)), unsafe_allow_html=True)
                c[i + 1].button("Open", key=f"open_{r.case_id}", on_click=_request, args=("read", r.case_id),
                                icon=":material/arrow_forward:", width="stretch", help=f"Open {r.case_id}")
    with st.container(key="mobilelist"):
        for r in rows.itertuples():
            with st.container(key=f"mcard_{r.case_id}"):
                if r.qc_kind != "ok":
                    ai_line = f"<span class='ra-tag ra-t-neutral'>{T.icon('square', 14)}AI output withheld</span>"
                    hint = ("Review the technical limitation before proceeding - the images can be read manually."
                            if r.qc_kind == "segmentation" else "Check the series before reading.")
                elif _hide_ai(r.case_id):
                    ai_line, hint = T.hidden_finding(), "Independent first read"
                else:
                    ai_line = T.badge(r.category) + (f"<span class='ra-p'>{U.pct(r.tumour_probability)}</span>"
                                                     if r.category not in ("QC FAIL", "DEFERRED") else "")
                    hint = ""
                html(T.mobile_card(int(r.position), r.case_id, r.status, r.qc_kind, r.qc_label, ai_line, hint))
                st.button("Open case  →", key=f"mopen_{r.case_id}", on_click=_request, args=("read", r.case_id),
                          width="stretch")
    with st.expander("How to read this list"):
        html("<p>Separate kinds of status are used, and none implies another:</p><ul>"
             f"<li>{T.icon('square', 13)} <b>Image / AI QC</b> - technical: whether the AI output can be used. A failed "
             "check withholds the AI output; it is never a clinical finding or an urgency level.</li>"
             f"<li>{T.icon('triangle', 13)} <b>Urgent by AI rule</b> - an AI triage rule "
             f"({T.esc(U.urgent_rule(cfg))}), shown in AI-assisted triage only.</li>"
             "<li><b>Reading status</b> - workflow: unread, impression saved, decided or deferred.</li>"
             + (f"<li>{T.icon('alert', 13)} <b>Model disagreement</b> - a characteristic of the AI evidence: the two "
                "AI tools reach different conclusions.</li>" if second else "") + "</ul>")
        for k, v in CATEGORY_HELP.items():
            if k != "QC FAIL":
                html(f"<p>{T.badge(k)} &nbsp;{T.esc(v)}</p>")
        html(f"<p>{T.badge('DEFERRED')} &nbsp;This reader's decision policy withholds the AI suggestion; read "
             f"independently.</p><p>{T.badge('QC FAIL')} &nbsp;Automatic QC failed; the AI output is withheld.</p>")
        st.caption("Probabilities are calibrated patient-level probabilities that the CT shows at least one liver "
                   f"tumour. {U.UNCERTAINTY_DEFINITION}")


# =========================================================================== Read case
def _record_impression(cid: str, value: str | None) -> None:
    if value not in FINAL:
        return
    try:
        U.save_impression(store, reader_id, cid, int(value == "present"), ss.session_id)
    except Exception as e:  # noqa: BLE001 - shown to the reader, never swallowed
        ss.save_error = {"case": cid, "msg": f"Your impression was not saved ({e})."}
        return
    ss.pop("save_error", None)
    ss.session_locked = True


def _save(cid: str, action: str, final: int | None, init, audit: dict, edit: dict | None = None) -> bool:
    try:
        U.log_live(store, cases[cid], policy, init, action, final, edit=edit, audit=audit)
    except Exception as e:  # noqa: BLE001 - shown to the reader, never swallowed
        ss.save_error = {"case": cid, "msg": f"The decision was not saved ({e}). Nothing was recorded; try again."}
        return False
    ss.pop("save_error", None)
    ss.pop(f"revise_{cid}", None)
    ss.session_locked = True
    return True


def _pending_save() -> None:
    p = ss.get("pending_nav") or {}
    cid = p.get("from")
    if "impression" in p.get("what", ""):
        _record_impression(cid, ss.get(f"imp_{cid}"))
        if ss.get("save_error"):
            return
    else:
        payload = (ss.get("form_payload") or {}).get(cid)
        if not payload or not _save(cid, *payload):
            return
    _perform(p.get("view"), p.get("case"), p.get("action"))


def _pending_discard() -> None:
    p = ss.get("pending_nav") or {}
    cid = p.get("from")
    for k in (f"imp_{cid}", f"final_{cid}", f"note_{cid}", f"keep_{cid}", f"revise_{cid}"):
        ss.pop(k, None)
    _perform(p.get("view"), p.get("case"), p.get("action"))


def _reset_view(slice_key: str, start: int) -> None:
    ss[slice_key] = start
    for k, v in VIEW_DEFAULTS.items():
        if k != "layout":
            ss[k] = v


def _step_slice(slice_key: str, delta: int, nums: list) -> None:
    i = nums.index(ss[slice_key]) if ss.get(slice_key) in nums else 0
    ss[slice_key] = nums[int(np.clip(i + delta, 0, len(nums) - 1))]


def _jump(slice_key: str, idx: int) -> None:
    ss[slice_key] = int(idx)


def _toggle_orig() -> None:
    ss.orig_ct = not ss.get("orig_ct", False)


def _win_label(k: str) -> str:
    if k in WINDOWS:
        n, w, lv = WINDOWS[k]
        return f"{n} · W {w} / L {lv} HU"
    return "Custom"


def _pending_bar() -> None:
    p = ss.get("pending_nav")
    if not p or p.get("from") != case_id:
        return
    target = ("start a new session" if p.get("action") == "new_session" else
              f"open {p['case']}" if p.get("case") != case_id else f"go to {p.get('view')}")
    can_save = "impression" in p["what"] or bool((ss.get("form_payload") or {}).get(case_id))
    with st.container(key="pending"):
        html(f"<div class='ra-panel-t'>{T.icon('alert', 15)} Unsaved work on {T.esc(case_id)}</div>"
             f"<div class='ra-panel-s'>You have {T.esc(p['what'])}. Save it, discard it, or stay on this case "
             f"before you {T.esc(target)}.</div>")
        a1, a2, a3 = cols("pendingbar", 3)
        a1.button("Save and continue", key="pending_save", type="primary", disabled=not can_save,
                  on_click=_pending_save, icon=":material/save:", width="stretch",
                  help=None if can_save else "Choose a final finding first - a note alone cannot be saved.")
        a2.button("Discard and continue", key="pending_discard", on_click=_pending_discard,
                  icon=":material/delete:", width="stretch")
        a3.button("Stay on this case", key="pending_stay", on_click=lambda: ss.pop("pending_nav", None),
                  width="stretch")


def view_read() -> None:
    prev_id, next_id = U.neighbours(order, case_id)
    g = record.get("geometry", {})
    nz = int((g.get("shape_view") or [0])[0])
    sp = (g.get("spacing_mm") or [0, 0, 0])
    imp = impressions.get(case_id)
    reviewed = case_id in reviews
    revising = bool(ss.get(f"revise_{case_id}"))
    ai_withheld = decision["decision"] != "show"
    qc_failed = not record["quality"]["passed"]
    need_impression = blind and not ai_withheld and imp is None and not reviewed
    hide = blind and imp is None and not reviewed
    pos = order.index(case_id) + 1

    with st.container(key="casebar"):
        cb = cols("casebar", [2.2, 1.3], vertical_alignment="center")
        meta = (f"CT abdomen · axial · {nz} slices · {float(sp[0]):.1f} mm slice spacing · case {pos} of {len(order)}"
                + (" · AI output withheld - independent read" if ai_withheld else ""))
        prog = ""
        if blind and not ai_withheld:
            done = {1, 2, 3} if reviewed and not revising else ({1, 2} if imp else set())
            prog = T.progress(0 if len(done) == 3 else (3 if imp else 1), done)
        elif not ai_withheld and live:
            prog = T.progress(0 if reviewed and not revising else 2, {1, 2} if reviewed and not revising else {1},
                              labels=("AI evidence", "Decision"))
        cb[0].markdown(T.case_identity(case_id, row["status"], meta) + prog, unsafe_allow_html=True)
        with cb[1]:
            n1, n2, n3 = cols("casenav", [0.55, 2, 0.55], vertical_alignment="center")
            n1.button("Previous", key="prev_case", disabled=prev_id is None, on_click=_request, args=("read", prev_id),
                      icon=":material/arrow_back:", width="stretch", shortcut="P", help="Previous case (P)")
            ss.case_pick = case_id
            n2.selectbox("Go to case", order, key="case_pick", format_func=_case_label, on_change=_pick_changed,
                         label_visibility="collapsed", help="Go to case (type to search)")
            n3.button("Next", key="next_case", disabled=next_id is None, on_click=_request, args=("read", next_id),
                      icon=":material/arrow_forward:", width="stretch", shortcut="N", help="Next case (N)")
    html("<div class='ra-device-note'>" + T.note(
        "<b>Small screen.</b> This size is for checking the worklist and decisions only. Diagnostic interpretation "
        "requires a validated diagnostic display.", "warn") + "</div>")
    _pending_bar()

    vpath = store.viewer_path(case_id)
    arrs = None
    try:
        if not vpath.exists():
            raise FileNotFoundError("the image file is missing from this run")
        arrs = _load_viewer(str(vpath), vpath.stat().st_mtime)
        _ = arrs["ct"], arrs["z"]
    except Exception as e:  # noqa: BLE001 - an explicit error state, never an empty or "normal" view
        html(T.note(f"<b>The images for {T.esc(case_id)} could not be loaded</b> ({T.esc(e)}). No AI result is shown "
                    "for this case, and this is not a negative result. Try again, or go back to the worklist and "
                    "report the problem.", "technical"))
        st.button("Try loading the images again", key="retry_load", on_click=_load_viewer.clear,
                  icon=":material/refresh:")
        return
    large = ss.get("layout", "standard") == "large"
    left, right = cols("read75" if large else "read62", [3, 1] if large else [1.65, 1], gap="medium")
    m, c, q = record["measurements"], record["confidence"], record["quality"]
    qc = U.qc_status(record)
    hide_msg = ("AI overlays hidden during this independent read" if ai_withheld
                else "AI overlays locked until you save your impression")
    with left:
        _viewer(arrs, hide, qc_failed, nz, hide_msg)
    with right, st.container(key="sidepanel"):
        n = 1
        if blind and not ai_withheld:
            html(T.panel_title("01", "Independent read" if need_impression else "Impression and AI suggestion"))
            n = 2
            if need_impression:
                _impression_panel()
            else:
                if imp and imp.get("session_id") not in (None, ss.session_id) and not reviewed:
                    html(T.note(f"<b>Resumed.</b> Your impression was saved at {T.esc(imp.get('at_local'))} in session "
                                f"{T.esc(imp.get('session_id'))}. It is kept as recorded; the AI evidence is shown "
                                "below.", "info"))
                ai_txt = "Liver tumour suspected" if decision["ai_label"] else "No liver tumour suspected"
                html(T.comparison(None if imp is None else imp["value"], (imp or {}).get("at_local"),
                                  decision["ai_label"], ai_txt))
        if not need_impression:
            _evidence_panel(f"{n:02d}", m, c, q, qc, arrs)
            n += 1
        form_open = live and not need_impression and (case_id not in reviews or revising)
        extras = _decision_extras(m) if form_open else ("", None, None, False)
        with st.container(key="decision"):
            _decision_panel(f"{n:02d}", m, c, imp, need_impression, extras)


def _viewer(arrs: dict, hide: bool, qc_failed: bool, nz: int, hide_msg: str) -> None:
    nums = [int(z) + 1 for z in arrs["z"]]            # 1-based slice numbers of the original volume
    slice_key = f"slice_{case_id}"
    start = nums[U.neutral_slice_index(arrs) if hide else U.key_slice_index(arrs, record)]
    hu = cfg["preprocess"]["hu_window"]
    for k, v in VIEW_DEFAULTS.items():      # defaults via session state only (no widget default + API clash)
        ss.setdefault(k, v)
    if ss.get(slice_key) not in nums:
        ss[slice_key] = start
    masks_off = hide or qc_failed
    orig = bool(ss.get("orig_ct")) and not masks_off
    lock_help = ("Not available: automatic QC failed, so the AI masks are not shown" if qc_failed else
                 hide_msg if hide else None)
    with st.container(key="viewer"):
        v1, v2, v3, v4 = cols("viewtools", [1, 1, 1.15, 0.7], vertical_alignment="center")
        with v1.popover("Image", icon=":material/contrast:", width="stretch",
                        help="Window / level, zoom, pan and layout"):
            win = st.radio("Window / level", list(WINDOWS) + ["custom"], key="window", format_func=_win_label)
            if win == "custom":
                st.slider("Window width (HU)", 40, 350, step=5, key="wl_w")
                st.slider("Window level (HU)", -50, 200, step=5, key="wl_l")
            st.slider("Zoom", 1.0, 3.0, step=0.25, key="zoom", format="%.2f×")
            st.slider("Pan left / right", -1.0, 1.0, step=0.1, key="pan_x")
            st.slider("Pan up / down", -1.0, 1.0, step=0.1, key="pan_y")
            st.radio("Layout", list(LAYOUTS), key="layout", format_func=LAYOUTS.get)
            st.caption(f"Window / level and layout apply to every case in this session; zoom and pan reset when you "
                       f"open another case. Reset restores Liver · W 150 / L 70 HU and zoom 1×. The viewer images "
                       f"store {hu[0]} to {hu[1]} HU; values outside show as black or white. Presets are research "
                       "defaults, not clinically validated settings.")
        with v2.popover("Overlays", icon=":material/layers:", width="stretch", disabled=masks_off,
                        help=lock_help or "AI liver and lesion contours, lesion fill, uncertainty map"):
            st.toggle("AI liver contour", key="ov_liver", disabled=masks_off, help="Teal outline of the AI liver mask")
            st.toggle("AI lesion contour", key="ov_lesions", disabled=masks_off, help="Amber outline of each AI lesion")
            st.slider("Lesion fill opacity", 0, 80, step=5, key="op_lesion", format="%d%%", disabled=masks_off,
                      help="0 % shows the contour only, so the anatomy underneath stays visible.")
            st.toggle("Uncertainty map", key="ov_unc", disabled=masks_off,
                      help="Violet; stronger colour = higher voxel uncertainty")
            st.slider("Uncertainty map opacity", 0, 90, step=5, key="op_unc", format="%d%%", disabled=masks_off)
            st.caption("Contours show where the AI drew its masks - not a confirmed finding. A crisp outline does not "
                       "mean the AI is certain.")
        v3.button("Show AI overlays (O)" if orig else "Original CT (O)", key="orig_btn", on_click=_toggle_orig,
                  shortcut="O", icon=":material/layers_clear:" if not orig else ":material/layers:",
                  disabled=masks_off, width="stretch",
                  help=lock_help or "Hide every AI overlay to inspect the original CT (O)")
        v4.button("Reset", key="reset_view", on_click=_reset_view, args=(slice_key, start),
                  icon=":material/restart_alt:", width="stretch", help="Reset window, zoom, pan and overlays")
        win = ss.get("window") or "liver"
        if win == "custom":
            w_, l_, wname = float(ss.get("wl_w", 150)), float(ss.get("wl_l", 70)), "Custom"
        else:
            wname, w_, l_ = WINDOWS[win][0], float(WINDOWS[win][1]), float(WINDOWS[win][2])
        sl = nums.index(ss[slice_key])
        on = not masks_off and not orig
        show_liver, show_les, show_unc = (on and ss.get("ov_liver"), on and ss.get("ov_lesions"),
                                          on and ss.get("ov_unc"))
        kept = ss.get(f"kept_{reader_id}_{case_id}")
        img = U.render_slice(arrs, sl, show_liver=bool(show_liver), show_lesions=bool(show_les), kept_lesions=kept,
                             show_uncertainty=bool(show_unc), show_reference=research,
                             upscale=max(1, 512 // arrs["ct"].shape[1]), contrast=U.window_to_u8(w_, l_, hu),
                             lesion_alpha=ss.get("op_lesion", 0) / 100.0, unc_alpha=ss.get("op_unc", 70) / 100.0,
                             zoom=float(ss.get("zoom", 1.0)), pan=(ss.get("pan_x", 0.0), ss.get("pan_y", 0.0)),
                             markers=True)
        z_abs = int(arrs["z"][sl])
        if qc_failed:
            ov = f"<span class='ra-vwarn'>{T.icon('square', 12)} QC failed - AI masks not shown</span>"
        elif hide:
            ov = f"<span class='ra-vwarn'>{T.icon('lock', 12)} {T.esc(hide_msg)}</span>"
        elif orig:
            ov = "<span>Original CT - all AI overlays hidden</span>"
        else:
            names = [n for n, f in (("liver", show_liver), ("lesions", show_les), ("uncertainty", show_unc)) if f]
            ov = f"<span>Overlays: {', '.join(names) if names else 'none'}</span>"
            miss = []
            if show_liver and not arrs["liver"][sl].any():
                miss.append("no AI liver mask on this slice")
            if show_les and not arrs["lesions"][sl].any():
                miss.append("no AI lesion on this slice")
            if miss:
                ov += f"<span class='ra-vwarn'>{T.esc(' · '.join(miss))}</span>"
        html(f"<div class='ra-vinfo'><span>Slice <b>{z_abs + 1}</b> / {nz}</span>"
             f"<span><b>{T.esc(wname)}</b> · W {w_:.0f} / L {l_:.0f} HU · zoom {float(ss.get('zoom', 1.0)):.2f}×"
             f"</span></div><div class='ra-vinfo'>{ov}</div>")
        st.image(img, width="stretch")
        s1, s2, s3 = cols("slicenav", [0.2, 1, 0.2], vertical_alignment="center")
        s1.button("Previous slice", key="slice_prev", on_click=_step_slice, args=(slice_key, -1, nums),
                  icon=":material/remove:", width="stretch", shortcut="K", help="Previous slice (K)")
        s2.select_slider("Slice", options=nums, key=slice_key, label_visibility="collapsed",
                         help="Drag, or use J / K. Arrow keys work after clicking the slider.")
        s3.button("Next slice", key="slice_next", on_click=_step_slice, args=(slice_key, 1, nums),
                  icon=":material/add:", width="stretch", shortcut="J", help="Next slice (J)")
        items = []
        if show_liver:
            items.append(("AI liver contour", T.OVERLAY["liver"], "line"))
        if show_les:
            items.append(("AI lesion " + ("contour" if ss.get("op_lesion", 0) == 0 else "contour + fill"),
                          T.OVERLAY["lesion"], "line" if ss.get("op_lesion", 0) == 0 else "fill"))
        if research:
            items.append(("Reference (tumour solid, liver dotted)", T.OVERLAY["reference"], "dotted"))
        if items or show_unc:
            html(T.legend(items, uncertainty=bool(show_unc)))
    st.caption("Radiological view: patient's right (R) on the image's left, anterior (A) at the top. "
               "Keys: J / K slice · N / P case · O original CT. Full screen: the expand icon on the image.")


def _impression_panel() -> None:
    with st.container(key="impression"):
        html("<div class='ra-panel-s'>Record your own impression first. The AI output for this case stays hidden "
             "until you save it.</div>")
        imp = st.segmented_control("Your impression", ["present", "absent"], key=f"imp_{case_id}",
                                   format_func=FINAL.get, width="stretch")
        st.button("Save impression and reveal AI evidence", type="primary", disabled=imp is None, key="record_imp",
                  on_click=_record_impression, args=(case_id, imp), icon=":material/lock_open:", width="stretch")
        err = ss.get("save_error")
        if err and err.get("case") == case_id:
            html(T.note(T.esc(err["msg"]), "critical"))
        html(T.save_state("unsaved", "Impression not saved yet") if imp else
             T.save_state("none", "No impression recorded"))
        st.caption("Hidden until then: the AI prediction, segmentation overlays, uncertainty"
                   + (", the agreement between the two AI models" if second else "")
                   + " and the case's AI priority. Your impression is saved immediately and cannot be changed "
                     "afterwards.")


def _evidence_panel(n: str, m: dict, c: dict, q: dict, qc: dict, arrs: dict) -> None:
    html(T.panel_title(n, "Evidence and limitations", divider=n != "01"))
    v = record.get("vlm") or {}
    if not q["passed"]:
        html(T.withheld(f"AI output withheld · {qc['label']}",
                        [("What failed", qc["detail"] or "-"), ("What it means", qc["consequence"]),
                         ("What you can do", qc["manual"])]))
    elif decision["decision"] != "show":
        html(T.withheld("AI suggestion withheld for this reader",
                        [("Why", f"Your decision policy ({pol_ver}) withholds the suggestion because "
                                 f"{U.humanize(decision['reason']).lower()}."),
                         ("What you can do", "Read this case independently."
                          + ("" if _hide_ai(case_id) else " The AI measurements below are supporting evidence only."))]))
    else:
        ag = U.agreement(c.get("segmenter_probability"), c.get("vlm_probability"))
        cal_txt = f"calibrated on {n_cal} separate calibration cases" if n_cal else "calibrated"
        html(T.evidence(
            finding="AI finding: liver tumour suspected" if decision["ai_label"] else
            "AI finding: no liver tumour suspected",
            p=c["tumour_probability"], threshold=policy.threshold,
            target=f"Patient-level probability that this CT shows at least one liver tumour · {cal_txt} · "
                   f"suggestion threshold {U.pct(policy.threshold)}",
            unc_level=c["uncertainty_level"], unc_text=U.uncertainty_phrase(c["uncertainty"]), qc_ok=True,
            agreement_kind=ag["kind"] if second else None, agreement_text=ag["text"],
            agreement_interp=ag["interpretation"]))
    if _hide_ai(case_id) or not q["passed"]:
        st.caption("AI measurements" + (", the second-reader result" if second else "") + " and uncertainty details "
                   "are not shown "
                   + ("because quality control failed." if not q["passed"] else "during an independent read."))
    else:
        with st.expander("Segmentation measurements and lesions", icon=":material/straighten:"):
            html(T.measures([("Liver volume", f"{m.get('liver_volume_ml', 0):.0f}", "mL"),
                             ("AI tumour volume", f"{m.get('predicted_tumour_volume_ml', 0):.1f}", "mL"),
                             ("AI lesions", f"{m.get('n_lesions', 0)}", ""),
                             ("Largest diameter", f"{m.get('largest_lesion_diameter_mm', 0):.0f}", "mm")]))
            if m.get("lesions"):
                les = pd.DataFrame(m["lesions"])
                j1, j2 = cols("decide", 2, vertical_alignment="bottom")
                pick = j1.selectbox("Go to lesion", list(les["id"]), key=f"lesjump_{case_id}",
                                    format_func=lambda i: f"Lesion {i} · "
                                                          f"{float(les.set_index('id').loc[i, 'volume_ml']):.1f} mL")
                ks = int(les.set_index("id").loc[pick, "key_slice"])
                j2.button("Show lesion slice", key="go_lesion", on_click=_jump, disabled=_hide_ai(case_id),
                          args=(f"slice_{case_id}", int(arrs["z"][int(np.argmin(np.abs(arrs["z"] - ks)))]) + 1),
                          icon=":material/my_location:", width="stretch")
                st.dataframe(les[["id", "volume_ml", "max_axial_diameter_mm", "mean_hu", "mean_probability",
                                  "slice_range"]], hide_index=True, height=min(220, 40 + 36 * len(les)), column_config={
                    "id": "#", "volume_ml": st.column_config.NumberColumn("Volume (mL)", format="%.2f"),
                    "max_axial_diameter_mm": st.column_config.NumberColumn("Longest axial diameter (mm)", format="%.0f"),
                    "mean_hu": st.column_config.NumberColumn("Mean HU", format="%.0f"),
                    "mean_probability": st.column_config.NumberColumn("Mean voxel probability", format="%.2f"),
                    "slice_range": "Slices"})
            st.caption("Measured from the AI segmentation at the original resolution.")
        if second:
            with st.expander("Second-reader model (image and text)", icon=":material/neurology:"):
                if v.get("status") != "ok":
                    st.markdown(f"**No second-reader result for this case** ({v.get('reason', v.get('status', 'not run'))}). "
                                "A missing result is not a negative result; model agreement cannot be assessed.")
                else:
                    st.markdown(f"The second-reader model looked at {v.get('n_images')} axial slices through the predicted "
                                f"liver. Its patient-level probability of a liver tumour: **{U.pct(c.get('vlm_probability'))}"
                                f"** (calibrated; raw {U.pct(v['p_yes'])}).")
                    if "vlm_rgb" in arrs:
                        st.image([im for im in arrs["vlm_rgb"]], width=96,
                                 caption=[f"slice {int(z) + 1}" for z in arrs["vlm_z"]])
                        st.caption("Model input format: red = wide window, green = soft tissue, blue = narrow window.")
                    if v.get("text"):
                        html(T.note("<b>Unverified model text</b>, quoted as written. It is never used for measurements or "
                                    "for the suggestion.", "info"))
                        st.text(v["text"])
                    for note_ in (v.get("guard") or {}).get("notes", []):
                        html(T.note("<b>Check:</b> " + T.esc(note_), "warn"))
        with st.expander("Uncertainty and calibration", icon=":material/insights:"):
            uc = c.get("uncertainty_components", {})
            st.markdown(f"- {U.UNCERTAINTY_DEFINITION}\n"
                        f"- This case: total {c['uncertainty']:.2f}; disagreement between the model's passes on the CT and "
                        f"on its mirror image "
                        f"(epistemic) {uc.get('mutual_information', 0.0):.2f}; expected noise (aleatoric) "
                        f"{uc.get('expected_entropy', 0.0):.2f}.\n"
                        f"- Calibration: {c.get('calibrator', '-')}"
                        + (f", fitted on {n_cal} calibration cases" if n_cal else "")
                        + f". Raw segmentation score {c['tumour_probability_raw']:.2f}; calibrated segmentation "
                          f"probability {U.pct(c.get('segmenter_probability'))}.")
    if not blind and row["reasons"]:
        with st.expander("Why this case is placed here (AI priority)", icon=":material/sort:"):
            for r_ in str(row["reasons"]).split("; "):
                st.markdown(f"- {r_}")
    with st.expander("Models and versions", icon=":material/settings:"):
        model = record.get("model", {})
        st.markdown(f"- Model set **{mv['id']}** · decision policy **{pol_ver}** ({policy.summary()})\n"
                    f"- Segmentation model: {seg_name} ({model.get('version', model.get('backend', '-'))}, "
                    f"{model.get('passes', '-')} test-time passes; pretrained, not trained in this project)\n"
                    + (f"- Weights: {card.get('source')} ({card.get('revision', 'main')}) · licence: "
                       f"{card.get('license')}\n" if card.get("source") else "")
                    + (f"- Second-reader model: {(v.get('model') or {}).get('model_id', vlm_name)}\n" if second else "")
                    + f"- Session {ss.session_id}")
        if card.get("training_data_overlap"):
            st.caption("Training data: " + card["training_data_overlap"])
        st.caption(_policy_source() + (" " + " · ".join(policy.notes) if policy.notes else ""))


def _audit(c: dict, imp: dict | None, supersedes: str | None) -> dict:
    ai_shown = decision["decision"] == "show"
    rec = ({"text": ("Liver tumour suspected" if decision["ai_label"] else "No liver tumour suspected")
                    + f" ({U.pct(c['tumour_probability'])}, threshold {U.pct(policy.threshold)})",
            "probability": c["tumour_probability"], "threshold": policy.threshold, "uncertainty": c["uncertainty"],
            "agreement": U.agreement(c.get("segmenter_probability"), c.get("vlm_probability"))["kind"],
            "category": row["category"], "policy_selected": policy_name} if ai_shown else
           {"text": "AI suggestion withheld (" + ("QC failed" if not case["qc_passed"] else "reader policy") + ")",
            "category": row["category"], "policy_selected": policy_name})
    return {"session_id": ss.session_id, "policy_version": pol_ver, "model_version": mv["id"], "recommendation": rec,
            "impression_at": (imp or {}).get("at"), "revealed_at": (imp or {}).get("at"), "supersedes": supersedes}


def _relation(action: str | None) -> str:
    return {"accept": "agrees with the AI suggestion", "edit": "agrees with the AI suggestion; AI lesions modified",
            "reject": "differs from the AI suggestion", "manual": "independent read (AI suggestion withheld)",
            "defer": "deferred - no final finding yet"}.get(action or "", "-")


def _decision_extras(m: dict) -> tuple:
    """Optional parts of the decision, kept outside the sticky footer so the footer stays short."""
    ai_shown = decision["decision"] == "show"
    keep, vol, modified = None, None, False
    with st.expander("Decision note or lesion edit (optional)", icon=":material/edit_note:"):
        note_ = st.text_input("Note", key=f"note_{case_id}", placeholder="Optional free-text note")
        if ai_shown and m.get("lesions"):
            if ss.get(f"final_{case_id}") == "present":
                les_ids = [les["id"] for les in m["lesions"]]
                keep = st.multiselect("AI lesions to keep", les_ids, default=les_ids, key=f"keep_{case_id}")
                ss[f"kept_{reader_id}_{case_id}"] = keep
                vol = sum(les["volume_ml"] for les in m["lesions"] if les["id"] in keep)
                modified = set(keep) != set(les_ids)
                st.caption(f"Tumour volume after your edit: {vol:.2f} mL (from the original-resolution lesion "
                           "volumes).")
            else:
                st.caption("Choose 'Tumour present' to keep or remove individual AI lesions.")
    return note_, keep, vol, modified


def _decision_panel(n: str, m: dict, c: dict, imp: dict | None, need_impression: bool, extras: tuple) -> None:
    prev = reviews.get(case_id)
    if reader_id in profiles:
        _simulation_panel()
        return
    revising = bool(ss.get(f"revise_{case_id}"))
    err = ss.get("save_error") if (ss.get("save_error") or {}).get("case") == case_id else None
    fin_now = ss.get(f"final_{case_id}")
    note_now = str(ss.get(f"note_{case_id}") or "").strip()
    if need_impression:
        html(T.panel_title("02", "Evidence and decision", right_html=T.save_state("none", "Locked")))
        html("<div class='ra-panel-s'>The AI evidence and the final decision open after you save your impression.</div>")
        return
    if err and "impression" not in err["msg"]:
        chip = T.save_state("failed", "Save failed")
    elif prev and not revising:
        chip = T.save_state("saved", f"Saved {local_hm(prev.get('timestamp'))}")
    elif fin_now is not None or note_now:
        chip = T.save_state("unsaved", "Unsaved changes")
    else:
        chip = T.save_state("none", "Not saved")
    html(T.panel_title(n, "Decision", right_html=chip))
    nxt = _next_unread(case_id)
    if prev and not revising:
        init = prev.get("initial_read")
        fin = ("not decided (deferred)" if prev.get("final_label") is None
               else FINAL["present"] if prev["final_label"] else FINAL["absent"])
        reason = (prev.get("edit") or {}).get("defer_reason")
        lines = [f"<b>Saved {T.esc(local_hm(prev.get('timestamp')))}</b> · case <b>{T.esc(case_id)}</b>",
                 ("Impression: " + (FINAL['present'] if init else FINAL['absent']) + " → " if init is not None else "")
                 + f"Final finding: <b>{T.esc(fin)}</b>" + (f" ({T.esc(reason)})" if reason else ""),
                 "Relation to AI: " + T.esc(_relation(prev.get("action")))]
        if prev.get("supersedes"):
            lines.append(f"Revises the decision saved at {T.esc(local_hm(prev['supersedes']))} (kept in the log)")
        lines.append("<span class='ra-muted'>Research decision saved · report draft not signed</span>")
        html("<div class='ra-saved'>" + "<br>".join(lines) + "</div>")
        if research:
            st.caption("Outcome vs reference: " + U.outcome_label(prev, case["gt_present"]))
        b1, b2 = cols("after", 2)
        b1.button("Next unread case", type="primary", key="after_next", disabled=nxt is None or nxt == case_id,
                  on_click=_request, args=("read", nxt), icon=":material/arrow_forward:", width="stretch")
        b2.button("Open report", key="after_report", on_click=_request, args=("report",),
                  icon=":material/description:", width="stretch")
        st.button("Revise decision…", key="revise", type="tertiary", icon=":material/edit:",
                  on_click=lambda: ss.__setitem__(f"revise_{case_id}", True),
                  help="Record a new decision. The original decision and your impression stay in the log.")
        return
    if revising and prev:
        html(T.note(f"Revising the decision saved at {T.esc(local_hm(prev.get('timestamp')))}. The original decision "
                    "and your impression stay in the log; the new decision records what it revises.", "info"))
    ai_shown = decision["decision"] == "show"
    fin = st.segmented_control("Final finding: liver tumour", list(FINAL), key=f"final_{case_id}",
                               format_func=FINAL.get, width="stretch")
    note_, keep, vol, modified = extras
    if fin is None:
        action, final = None, None
    elif not ai_shown:
        action, final = "manual", int(fin == "present")
    else:
        final = int(fin == "present")
        action = ("edit" if modified else "accept") if final == int(decision["ai_label"]) else "reject"
    init = (imp or {}).get("value")
    if prev and revising and prev.get("initial_read") is not None:
        init = prev["initial_read"]                     # the original impression is never replaced
    audit = _audit(c, imp, prev.get("timestamp") if (prev and revising) else None)
    edit = {"kept_lesions": keep, "edited_volume_ml": round(vol, 3) if vol is not None else None, "note": note_} \
        if (modified or note_) else None
    payloads = ss.setdefault("form_payload", {})
    payloads[case_id] = (action, final, init, audit, edit) if action else None
    html(f"<div class='ra-panel-s'>Your final finding {T.esc(_relation(action))}.</div>" if action else
         "<div class='ra-panel-s'>Choose a final finding to save, or defer the case.</div>")
    if err:
        html(T.note(T.esc(err["msg"]), "critical"))
    b1, b2 = cols("savebar", [1.3, 1])
    b1.button("Save decision", type="primary", key="save_decision", disabled=action is None,
              on_click=_save, args=(case_id, action, final, init, audit, edit), icon=":material/save:",
              width="stretch")
    with b2.popover("Defer…", icon=":material/schedule:", width="stretch",
                    help="No final finding is recorded; the case is flagged for a second opinion."):
        html("<div class='ra-panel-s'>Deferring records <b>no final finding</b>. The case is flagged for a second "
             "opinion and stays open.</div>")
        reason = st.radio("Reason", list(DEFER_REASONS), key=f"defer_reason_{case_id}", format_func=DEFER_REASONS.get)
        st.button("Confirm defer", key="confirm_defer", type="primary", on_click=_save, width="stretch",
                  args=(case_id, "defer", None, init, audit, {"defer_reason": DEFER_REASONS[reason], "note": note_}))
    if revising:
        st.button("Cancel revision", key="cancel_revise", type="tertiary",
                  on_click=lambda: [ss.pop(k, None) for k in (f"revise_{case_id}", f"final_{case_id}",
                                                               f"note_{case_id}")])


def _simulation_panel() -> None:
    prof = profiles[reader_id]
    with st.container(key="simulation"):
        html(T.panel_title(None, f"Research simulation · {prof.name}",
                           "A simulated read is not a clinical action. Batch simulation is in Reader adaptation."))
        if st.button("Simulate this read", key=f"sim_{case_id}", icon=":material/play_arrow:", width="stretch"):
            ev = U.simulate_and_log(store, prof, case, policy, salt=len(events_by_reader.get(reader_id, [])),
                                    audit={"session_id": ss.session_id, "policy_version": pol_ver,
                                           "model_version": mv["id"]})
            ss.last_sim = ev
            st.rerun()
        if "last_sim" in ss:
            ev = ss.pop("last_sim")
            txt = (f"Impression <b>{'tumour' if ev['initial_read'] else 'no tumour'}</b> → action "
                   f"<b>{T.esc(ev['action'])}</b> → final <b>{'tumour' if ev['final_label'] else 'no tumour'}</b>"
                   + (" (switched to the AI)" if ev.get("switched") else ""))
            html(T.note(txt, "neutral"))
            if research:
                st.caption("Outcome vs reference: " + U.outcome_label(ev, ev["gt_present"]))


# =========================================================================== Report (part of Read case)
def view_report() -> None:
    rv = reviews.get(case_id)
    c1, c2 = cols("reportbar", [4, 1.2], vertical_alignment="bottom")
    c1.markdown(T.page_title(f"Report draft · {case_id}", "A fixed template filled with the measured values. No language "
                                                         "model writes clinical text."), unsafe_allow_html=True)
    c2.button("Back to the case", key="report_back", on_click=_request, args=("read",), icon=":material/arrow_back:",
              width="stretch")
    if _hide_ai(case_id):
        html(T.note("The draft contains AI findings, so it is available after you save your independent impression "
                    "(blind first read).", "info"))
        return
    if rv:
        act = U.STATUS_TEXT.get(rv["action"], rv["action"])
        html(T.note(f"Reviewed by <b>{T.esc(reader_name)}</b>: {T.esc(act)} at {T.esc(local_hm(rv.get('timestamp')))}."
                    " <b>Research decision saved · this report draft is not signed</b> (the prototype cannot sign "
                    "reports).", "neutral"))
    else:
        html(T.note("Draft - no decision saved yet, not signed.", "info"))
    review = {**rv, "reader_name": reader_name} if rv else None
    text = draft_report(record, review=review, policy={**decision, "personalised": policy.personalised})
    with st.container(key="report_doc"):
        st.code(text, language=None, wrap_lines=True)
    d1, d2 = cols("downloads", 2)
    d1.download_button("Download report draft (.txt)", text, file_name=f"{case_id}_report_draft.txt",
                       icon=":material/download:", width="stretch")
    d2.download_button("Download case record (.json)", json.dumps(record, indent=2, default=str),
                       file_name=f"{case_id}.json", mime="application/json", icon=":material/data_object:",
                       width="stretch")


# =========================================================================== Research and quality area
def _chart_style(ax, title: str) -> None:
    ax.set_title(title, loc="left", color=T.NAVY, fontsize=12, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(T.BORDER_STRONG)
    ax.tick_params(colors=T.SLATE, labelsize=10)
    ax.xaxis.label.set_color(T.INK)
    ax.yaxis.label.set_color(T.INK)
    ax.grid(color="#E4EAF0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.figure.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")


def _policy_log_path() -> Path:
    return store.p("feedback") / "policy_changes.jsonl"


def _authorise(pol, by: str, reason: str, previous: str) -> None:
    try:
        e = U.authorise_policy(store, reader_id, policy_name, pol, by.strip(), reason.strip(), ss.session_id, previous)
    except Exception as ex:  # noqa: BLE001
        ss.auth_msg = ("critical", f"The authorisation was not saved ({ex}).")
        return
    ss.auth_ok = False
    ss.auth_msg = ("neutral", f"Policy {e['version']} authorised by {e['authorised_by']}. It takes effect from the "
                              "next reading session; the current session keeps its policy.")


def _withhold_text(pol) -> str:
    return "never" if pol.defer_uncertainty is None else f"uncertainty ≥ {pol.defer_uncertainty:.2f}"


def view_adapt() -> None:
    evs = events_by_reader.get(reader_id, [])
    _research_header("Reader adaptation", f"How a decision policy for {reader_name} is proposed from logged decisions, "
                                          "and how it is authorised.")
    html(T.note("Reading sessions use only an <b>authorised</b> policy. A proposed update never changes a session "
                "silently: it must be authorised by a named person with a reason, it is versioned and logged, and it "
                "takes effect from the <b>next</b> session. Offline simulation never changes it. Adaptation was "
                "evaluated offline with simulated readers only; it is not validated for clinical use.", "info"))
    stored, stored_entry = U.authorised_policy(store, cfg, reader_id, policy_name)
    p1, p2 = cols("policies", 2, gap="large")
    with p1:
        html(T.panel_title(None, "Active in this session") + T.measures([
            ("Version", pol_ver, ""), ("Suggestion threshold", U.pct(policy.threshold), ""),
            ("Withhold suggestion if", _withhold_text(policy), ""), ("Since", active[pkey]["since"], "")]))
        st.caption(_policy_source())
        if stored_entry and stored_entry["version"] != pol_ver:
            html(T.note(f"Authorised for the next session: <b>{T.esc(stored_entry['version'])}</b> by "
                        f"{T.esc(stored_entry['authorised_by'])}. Start a new session (Session menu) to use it.",
                        "info"))
    with p2:
        html(T.panel_title(None, f"Proposed from {proposed.n_feedback} decisions with an impression") + T.measures([
            ("Version", U.policy_version(proposed), ""), ("Suggestion threshold", U.pct(proposed.threshold), ""),
            ("Withhold suggestion if", _withhold_text(proposed), "")]))
        if proposed.notes:
            st.caption(" · ".join(proposed.notes))
    with st.container(key="authorise"):
        html(T.panel_title(None, "Authorise the proposed policy for future sessions",
                           "Requires a named person, a reason and confirmation. Logged to policy_changes.jsonl."))
        target_ver = U.policy_version(proposed)
        same = target_ver == (stored_entry or {}).get("version", pol_ver)
        if policy_name == "universal":
            st.caption("The universal policy needs no authorisation.")
        a1, a2 = cols("authrow", 2, vertical_alignment="bottom")
        by = a1.text_input("Authorised by", key="auth_by", placeholder="Name and role")
        reason = a2.text_input("Reason", key="auth_reason", placeholder="e.g. offline evaluation reviewed")
        ok = st.checkbox("I have reviewed the offline evaluation of this policy and authorise it for future reading "
                         "sessions.", key="auth_ok")
        ready = bool(by.strip() and reason.strip() and ok) and not same and policy_name != "universal"
        st.button("No change to authorise" if same else "Authorise for future sessions", key="apply_policy",
                  type="primary", disabled=not ready, on_click=_authorise,
                  args=(proposed, by, reason, (stored_entry or {}).get("version", pol_ver)),
                  icon=":material/verified_user:", width="content")
        if "auth_msg" in ss:
            kind, msg = ss.pop("auth_msg")
            html(T.note(T.esc(msg), kind))
    if reader_id in profiles:
        with st.container(key="simulation"):
            html(T.panel_title(None, f"Offline simulation · {reader_name}",
                               "Simulated reads of unread cases; inside this simulation the policy is refitted after "
                               "every read. The authorised and active policies are not changed."))
            s1, s2 = cols("sim", 2, vertical_alignment="bottom")
            nmore = s1.number_input("Number of reads", 1, 40, 10, key="nmore")
            if s2.button(f"Simulate {int(nmore)} reads", key="simn", icon=":material/fast_forward:", width="stretch"):
                ev_list = list(evs)
                done = {e["case_id"] for e in ev_list}
                todo = [cid for cid in wl.case_id if cid not in done][: int(nmore)]
                for cid in todo:
                    pol = U.current_policy(policy_name, reader_id, {**events_by_reader, reader_id: ev_list}, cases, cfg)
                    ev_list.append(U.simulate_and_log(store, profiles[reader_id], cases[cid], pol, salt=len(ev_list),
                                                      audit={"session_id": ss.session_id,
                                                             "policy_version": U.policy_version(pol),
                                                             "model_version": mv["id"]}))
                ss.sim_done = len(todo)
                st.rerun()
            if "sim_done" in ss:
                html(T.note(f"Simulated {ss.pop('sim_done')} reads.", "neutral"))
    if evs:
        variant = policy_name if policy_name.startswith("reader") else "reader_threshold_defer"
        traj = U.policy_trajectory(evs, cases, cfg, variant)
        col1, col2 = cols("charts", 2, gap="large")
        with col1:
            fig, ax = plt.subplots(figsize=(5.8, 3.4))
            ax.step(traj.n_feedback, traj.threshold, where="post", color=T.NAVY, lw=2.2, label="Suggestion threshold")
            if traj.defer_uncertainty.notna().any():
                ax.step(traj.n_feedback, traj.defer_uncertainty, where="post", color=T.AMBER, lw=1.8, ls="--",
                        label="Withhold if uncertainty ≥")
            ax.axhline(cfg["policy"]["universal_threshold"], color=T.BORDER_STRONG, lw=1.2)
            ax.set_ylim(0, 1.02)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.set_xlabel("Decisions logged")
            _chart_style(ax, "Proposed policy as decisions accumulate")
            ax.legend(loc="lower right", frameon=False, fontsize=9)
            st.pyplot(fig, clear_figure=True)
        with col2:
            fig, ax = plt.subplots(figsize=(5.8, 3.4))
            cgrid = np.linspace(0.5, 1.0, 50)
            ax.plot(cgrid, proposed.trust(cgrid), color=T.NAVY, lw=2.2, label="Learned from decisions")
            dis = [e for e in evs if e.get("shown") and e.get("initial_read") is not None and e["ai_label"] is not None
                   and e.get("final_label") is not None and e["initial_read"] != e["ai_label"]]
            if dis:
                x = np.array([e["ai_confidence"] for e in dis])
                y = np.array([e["final_label"] == e["ai_label"] for e in dis], dtype=float)
                ax.scatter(x, y + np.random.default_rng(0).uniform(-0.03, 0.03, len(y)), s=40, color=T.NAVY_2,
                           edgecolor="#FFFFFF", linewidth=1.2, alpha=0.9, label="Disagreements (1 = followed the AI)")
            if reader_id in profiles:
                ax.plot(cgrid, [profiles[reader_id].trust(v) for v in cgrid], color=T.BORDER_STRONG, lw=1.6, ls=":",
                        label="True behaviour (simulation)")
            ax.set_ylim(-0.08, 1.08)
            ax.set_xlabel("AI confidence in its suggestion")
            ax.set_ylabel("P(follow the AI | disagree)")
            _chart_style(ax, "Trust curve")
            ax.legend(loc="upper left", frameon=False, fontsize=8.5)
            st.pyplot(fig, clear_figure=True)
        df = pd.DataFrame(evs)
        df["recommendation shown"] = [(e.get("recommendation") or {}).get("text", "") for e in evs]
        if research:
            df["reference"] = df["case_id"].map(lambda cid: int(cases[cid]["gt_present"]))
            df["outcome"] = [U.outcome_label(e, cases[e["case_id"]]["gt_present"]) for e in evs]
        ecols = [c for c in ["timestamp", "case_id", "action", "initial_read", "final_label", "recommendation shown",
                             "policy_version", "model_version", "session_id", "supersedes", "reference", "outcome",
                             "source"] if c in df.columns]
        with st.expander(f"All {len(evs)} logged decisions"):
            st.dataframe(df[ecols].iloc[::-1], hide_index=True, height=320)
    else:
        html(T.note("No decisions logged yet for this reader.", "neutral"))
    lp = _policy_log_path()
    if lp.exists():
        with st.expander("Policy change log"):
            st.dataframe(pd.DataFrame([json.loads(x) for x in lp.read_text().splitlines() if x.strip()]).iloc[::-1],
                         hide_index=True)


def view_perf() -> None:
    cal = R.get("calibration") or {}
    mc = R.get("model_card") or {}
    tools = cal.get("tools_test") or {}
    _research_header("Model performance", f"{seg_name} on the LiTS17 test cases. Reader results come from simulated "
                                          "readers and are not evidence of clinical benefit.")
    if mc.get("training_data_overlap"):
        html(T.note("<b>Not a held-out evaluation of the segmenter.</b> " + T.esc(mc["training_data_overlap"])
                    + " The calibration and the reader experiments are still split correctly (calib vs test).",
                    "warn"))
    t = []
    if cal.get("test"):
        a, b = cal["test"]["after"], cal["test"]["before"]
        t += [("Test AUROC", f"{a['auroc']:.2f}", ""), ("Calibration error (ECE)", f"{a['ece']:.3f}", f"raw {b['ece']:.3f}"),
              ("Brier score", f"{a['brier']:.3f}", f"raw {b['brier']:.3f}")]
    refs = [r.get("reference") or {} for r in records.values() if r.get("split") == "test"]
    for key, label in (("dice_liver", "Liver Dice (test)"), ("dice_tumour", "Tumour Dice (test, cases with tumour)")):
        vals = [x[key] for x in refs if x.get(key) is not None and np.isfinite(x[key])
                and (key != "dice_tumour" or x.get("tumour_present"))]
        if vals:
            t.append((label, f"{float(np.mean(vals)):.2f}", f"n={len(vals)}"))
    if t:
        html(T.measures(t))
    if tools:
        names = {"segmenter": f"Segmentation model ({seg_name})", "second_reader": f"Second-reader model ({vlm_name})",
                 "fused": "Both combined"}
        st.markdown("#### The AI tools on the test cases")
        st.dataframe(pd.DataFrame([{"Tool": names.get(k, k), "Cases": r["n"], "With tumour": r["n_pos"],
                                    "AUROC": r["auroc"], "Brier": r["brier"], "ECE": r["ece"], "NLL": r["nll"]}
                                   for k, r in tools.items()]).round(3), hide_index=True)
        st.caption(f"Each tool calibrated on the calibration cases only. Used for the worklist: "
                   f"{cal.get('calibrator', {}).get('description', '-')}.")
    s = R.get("summary")
    if s is not None:
        st.markdown("#### Decision policies with simulated readers")
        show = s.copy()
        show["policy"] = show["policy"].map(POLICY_LABELS)
        show["reader"] = show["reader"].map(lambda r: reader_names.get(r, r))
        labels = {"reader": "Reader", "policy": "Policy", "accuracy": "Accuracy", "sensitivity": "Sensitivity",
                  "specificity": "Specificity", "harmful_acceptance_rate": "Harmful acceptance",
                  "over_reliance": "Over-reliance", "deferral_rate": "Withheld rate",
                  "actions_per_case": "Actions / case", "threshold": "Threshold"}
        tab = show[list(labels)].rename(columns=labels)
        for col in tab.columns[2:]:
            tab[col] = [("-" if pd.isna(x) else f"{x:.3f}".rstrip("0").rstrip(".")) for x in tab[col]]
        st.dataframe(tab, hide_index=True)
        st.caption("Harmful acceptance: an incorrect AI suggestion was accepted. Actions per case: accept 1, reject 2, "
                   "edit 3, independent read 4.")
    figs = [(n, title) for n, title in (
        ("ai_tools.png", "Segmentation model vs second-reader model"), ("calibration_reliability.png", "Calibration"),
        ("segmentation_quality.png", "Segmentation"), ("triage_lift.png", "Worklist triage"),
        ("risk_coverage.png", "Selective prediction"), ("policy_comparison.png", "Policies × readers"),
        ("benefit_by_reader.png", "Benefit by reader (RQ1)"), ("learning_curve.png", "Learning curve (RQ3)"))
        if (store.p("figures") / n).exists()]
    if figs:
        st.markdown("#### Figures")
        for i in range(0, len(figs), 2):
            rowc = cols(f"figs{i}", 2, gap="large")
            for col, (n, title) in zip(rowc, figs[i:i + 2]):
                with col:
                    html(f"<div class='ra-panel-t'>{T.esc(title)}</div>")
                    st.image(str(store.p("figures") / n), width="stretch")


def view_audit() -> None:
    _research_header(f"Audit trail · {case_id}", "Impression, decisions with the recommendation on screen at the time, "
                                                 "and every tool the agent called.")
    if _hide_ai(case_id):
        html(T.note("This case is unread in a blind first read. Its audit trail contains AI output and opens after your "
                    "impression is saved.", "info"))
        return
    imp = impressions.get(case_id)
    if imp:
        st.markdown("#### Independent impression")
        st.dataframe(pd.DataFrame([{"Time (UTC)": imp["at"], "Impression": FINAL["present"] if imp["value"]
                                    else FINAL["absent"], "Session": imp.get("session_id")}]), hide_index=True)
    evs = [e for r_ in reader_options for e in FeedbackStore(store.p("feedback")).events(r_) if e["case_id"] == case_id]
    if evs:
        st.markdown("#### Decisions on this case (oldest first; revisions never overwrite)")
        st.dataframe(pd.DataFrame([{"Time (UTC)": e["timestamp"], "Reader": reader_names.get(e["reader_id"], e["reader_id"]),
                                    "Action": U.STATUS_TEXT.get(e["action"], e["action"]),
                                    "Impression": e.get("initial_read"), "Final": e.get("final_label"),
                                    "Recommendation on screen": (e.get("recommendation") or {}).get("text", ""),
                                    "Policy": e.get("policy_version") or e.get("policy"),
                                    "Model set": e.get("model_version"), "Session": e.get("session_id"),
                                    "Revises": e.get("supersedes") or ""}
                                   for e in evs]), hide_index=True)
    tp = store.trace_path(case_id)
    if tp.exists():
        trace = json.loads(tp.read_text())
        st.markdown("#### Agent first read")
        html(T.measures([("Planner", str(trace["planner"]), ""), ("Status", str(trace["status"]), ""),
                         ("Guardrail overrides", str(trace.get("n_guardrail_overrides", 0)), "")]))
        st.dataframe(pd.DataFrame([{"Step": s_["step"], "Tool": s_["tool"], "Planned by": s_["planned_by"],
                                    "Reason": s_.get("reason", ""), "ms": s_.get("duration_ms"),
                                    "Result": json.dumps(s_.get("summary"), default=str)[:160],
                                    "Guardrail": s_.get("guardrail") or ""} for s_ in trace["steps"]]),
                     hide_index=True)
    from radagent.data.lits import locate

    ct_ok = Path(locate(str(store.manifest().set_index("case_id").loc[case_id, "ct_path"]), cfg)).exists()
    torch_ok = importlib.util.find_spec("torch") is not None
    can_recompute = ct_ok and (torch_ok or cfg["segmenter"]["backend"] == "plugin")
    with st.expander("Run the agent again on this case", icon=":material/replay:"):
        c1, c2 = cols("rerunrow", [1.4, 1], vertical_alignment="bottom")
        planner_choice = c1.segmented_control("Planner", ["rule", "llm"], default="rule", key="planner",
                                              format_func={"rule": "Rules", "llm": "Open LLM"}.get) or "rule"
        recompute = c1.toggle("Recompute from the CT file (runs both AI models now)", value=False,
                              disabled=not can_recompute, key="recompute",
                              help="Needs the LiTS CT on this machine, PyTorch and MONAI (a GPU is recommended for VISTA3D). "
                                   + ("" if can_recompute else ("CT file not found here. " if not ct_ok else "")
                                      + ("PyTorch not installed. " if not torch_ok else "")))
        if c2.button("Run both models now" if recompute else "Re-run the agent", key="rerun_btn",
                     icon=":material/replay:", width="stretch"):
            from radagent.pipeline import run_single_case

            with st.spinner("Running the AI models (the first call loads them) ..." if recompute else "Agent running ..."):
                _, tr2 = run_single_case(cfg, case_id, planner_choice, reader_policy=policy, recompute=recompute)
            ss["rerun_trace"] = tr2
            if recompute:
                _load_run_cached.clear()
                st.rerun()
    if "rerun_trace" in ss and ss["rerun_trace"]["case_id"] == case_id:
        tr2 = ss["rerun_trace"]
        html(T.note(f"Re-run with the {T.esc(tr2['planner'])} planner: {T.esc(tr2['status'])}, "
                    f"{tr2.get('n_guardrail_overrides', 0)} guardrail override(s).", "neutral"))
        st.dataframe(pd.DataFrame([{"Step": s_["step"], "Tool": s_["tool"], "Planned by": s_["planned_by"],
                                    "Reason": s_.get("reason", ""), "Guardrail": s_.get("guardrail") or "",
                                    "Result": json.dumps(s_.get("summary"), default=str)[:160]}
                                   for s_ in tr2["steps"]]), hide_index=True)


def view_about() -> None:
    _research_header("About and intended use", "What the assistant does, what it is not for, and who decides.")
    c1, c2 = cols("about", 2, gap="large")
    with c1:
        st.markdown(f"""
#### Intended use (research)
- Organise a liver-CT worklist and pre-read each case: liver and tumour segmentation by **{seg_name}**
  (one pretrained model, used as published - nothing is trained in this project), measurements at the
  original resolution, a calibrated patient-level probability and its uncertainty.
{"- A second opinion from **" + vlm_name + "**; cases where the two AI tools disagree are flagged." if second else ""}
- Study how individual readers use the AI and whether reader-specific policies help - offline, with
  authorised, versioned and logged changes only.
""")
    with c2:
        st.markdown("""
#### Not intended for
- Clinical diagnosis or patient management - a research prototype on public, de-identified data.
- Replacing a radiologist's read: **the radiologist decides every finding**.
- Judging the segmenter on LiTS: its training data include the LiTS volumes (MSD Task 03), so LiTS
  results are optimistic. Use an external dataset to evaluate it.
""")
    st.markdown("""
#### Supported devices
- **Diagnostic interpretation:** not validated on any display. A future clinical evaluation must use validated
  diagnostic displays in the reading room.
- **Desktop and laptop screens:** the full research workflow.
- **Tablets and phones:** worklist, status and saved decisions only; the case view shows a warning on small screens.

#### When something goes wrong
- **Images cannot be loaded:** an explicit error is shown; no AI result is shown and nothing looks normal.
- **QC fails:** the AI output is withheld and the reason, the consequence and what you can still do are stated.
- **A save fails:** the decision panel shows *Save failed* and nothing is recorded; try again.
- **Interrupted session:** impressions are saved at once; reopening the case resumes with the original impression.
- **Leaving a case with unsaved work:** you choose to save, discard or stay.
""")
    st.markdown("#### How a case is pre-read")
    html(T.flow([("CT volume", "LiTS NIfTI, voxel spacing", "data"), ("Segmentation model", seg_name, "ai"),
                 ("Measure", "volumes, diameters", "step"), ("Quality control", "operational checks", "step")]
                + ([("Second-reader model", vlm_name, "ai")] if second else [])
                + [("Calibrated probability", "+ uncertainty", "step"),
                 ("Triage", "AI priority (hidden in blind read)", "step"), ("Independent read", "your impression", "human"),
                 ("Decide", "final finding or defer", "human"), ("Report draft", "fixed template, not signed", "step")]))
    st.caption(f"Run directory: {store.root}")


{"worklist": view_worklist, "read": view_read, "report": view_report, "adapt": view_adapt, "perf": view_perf,
 "audit": view_audit, "about": view_about}[ss.view]()
