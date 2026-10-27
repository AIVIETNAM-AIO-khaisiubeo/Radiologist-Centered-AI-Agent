"""UI logic without Streamlit (rendering, worklist, live policy adaptation) - unit-testable."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from radagent.confidence import uncertainty_level  # noqa: E402
from radagent.config import load_config  # noqa: E402
from radagent.policy import ReaderPolicy, case_table, fit_policy, universal_policy  # noqa: E402
from radagent.readers.feedback import FeedbackEvent, FeedbackStore  # noqa: E402
from radagent.readers.profiles import N_DRAWS, ReaderProfile, load_profiles, simulate_read  # noqa: E402
from radagent.report import suggestion_text  # noqa: E402
from radagent.store import RunStore, load_json  # noqa: E402
from radagent.triage import CATEGORY_ORDER, triage  # noqa: E402
from ui_theme import CATEGORY_STYLE, OVERLAY  # noqa: E402

LIVE_READER = "live"
POLICY_CHOICES = {
    "universal": "Universal threshold",
    "universal_defer": "Universal + deferral",
    "reader_threshold": "Reader-specific threshold",
    "reader_threshold_defer": "Reader-specific + deferral",
}
# text glyphs for places that cannot render HTML (tables, downloads); shape-coded like the badges
CATEGORY_ICON = {"QC FAIL": "■", "URGENT": "▲", "SUSPICIOUS": "◆", "UNCERTAIN": "◐", "LIKELY NORMAL": "○",
                 "DEFERRED": "●"}
CATEGORY_LABEL = {k: v[1] for k, v in CATEGORY_STYLE.items()}

# overlay colours (RGB) - see ui_theme.OVERLAY
C_LIVER = np.array(OVERLAY["liver"])
C_LESION = np.array(OVERLAY["lesion"])
C_SELECTED = np.array(OVERLAY["selected"])
C_REF = np.array(OVERLAY["reference"])
C_UNC = np.array(OVERLAY["uncertainty"])


# --------------------------------------------------------------------------- run loading
def find_runs(root: str | Path = "outputs") -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if (p / "cases").is_dir() and any((p / "cases").glob("*.json"))],
                  key=lambda p: -p.stat().st_mtime)


def load_run_config(run: Path) -> dict:
    f = run / "config_used.yaml"
    cfg = load_config(f if f.exists() else None)
    cfg["output_root"] = str(run.parent)
    cfg["run_name"] = run.name
    return cfg


def load_run(run: Path) -> dict:
    st = RunStore(run)
    records = st.records()
    out = {"store": st, "cfg": load_run_config(run), "records": {r["case_id"]: r for r in records},
           "cases": case_table(records)}
    for key, rel in (("calibration", ("calibration", "calibration_metrics.json")), ("model_card", ("model", "model_card.json")),
                     ("exp_meta", ("experiments", "meta.json"))):
        f = st.p(*rel)
        out[key] = load_json(f) if f.exists() else None
    for key in ("summary", "benefit_spread", "results", "learning_curve", "policies"):
        f = st.p("experiments", f"{key}.csv")
        out[key] = pd.read_csv(f) if f.exists() else None
    return out


# --------------------------------------------------------------------------- rendering
def _edges(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    return m & ~ndi.binary_erosion(m, iterations=1, border_value=0)


def _blend(img: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    if mask.any():
        img[mask] = (1 - alpha) * img[mask] + alpha * color


def window_to_u8(width: float, level: float, hu_window=(-100.0, 250.0)) -> tuple[float, float]:
    """Window/level in HU -> display range in the stored uint8 viewer values (stored = hu_window)."""
    lo, hi = float(hu_window[0]), float(hu_window[1])
    a, b = level - width / 2.0, level + width / 2.0
    return (a - lo) / (hi - lo) * 255.0, (b - lo) / (hi - lo) * 255.0


def _put_label(img: np.ndarray, text: str, org: tuple[int, int], scale: float) -> None:
    import cv2

    th = max(1, int(round(scale * 2)))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 241, 247), th, cv2.LINE_AA)


def orientation_markers(img: np.ndarray) -> np.ndarray:
    """Radiological axial view: patient's right on the viewer's left (R), anterior at the top (A)."""
    import cv2

    h, w = img.shape[:2]
    s = max(0.5, w / 640.0)
    (tw, tht), _ = cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, s, 2)
    pad = int(10 * s)
    _put_label(img, "R", (pad, h // 2 + tht // 2), s)
    _put_label(img, "L", (w - pad - tw, h // 2 + tht // 2), s)
    _put_label(img, "A", (w // 2 - tw // 2, pad + tht), s)
    _put_label(img, "P", (w // 2 - tw // 2, h - pad), s)
    return img


def render_slice(arrs: dict, i: int, show_liver: bool = True, show_lesions: bool = True, kept_lesions=None,
                 selected_lesion: int | None = None, show_uncertainty: bool = False, show_reference: bool = False,
                 upscale: int = 4, contrast: tuple[float, float] = (0, 255), lesion_alpha: float = 0.35,
                 unc_alpha: float = 0.85, zoom: float = 1.0, pan: tuple[float, float] = (0.0, 0.0),
                 markers: bool = False) -> np.ndarray:
    """Compose an RGB uint8 image of viewer slice ``i`` with the requested overlays.

    ``contrast`` is the display range in stored uint8 values (see :func:`window_to_u8`); ``zoom`` >= 1
    crops around the image centre shifted by ``pan`` (fractions of the half-width, -1..1).
    """
    lo, hi = contrast
    ct = arrs["ct"][i].astype(np.float32)
    ct = np.clip((ct - lo) / max(1e-3, hi - lo) * 255.0, 0, 255)
    img = np.repeat(ct[..., None], 3, axis=2)
    if show_uncertainty and unc_alpha > 0:
        h = arrs["entropy"][i].astype(np.float32) / 255.0
        a = np.clip(h * unc_alpha, 0, unc_alpha)[..., None]
        img = (1 - a) * img + a * C_UNC
    lesions = arrs["lesions"][i]
    ids_all = np.unique(lesions[lesions > 0]) if lesions.any() else []
    ids = [k for k in ids_all if kept_lesions is None or int(k) in set(kept_lesions)]
    if show_lesions and lesion_alpha > 0:
        for k in ids:
            col = C_SELECTED if selected_lesion is not None and int(k) == selected_lesion else C_LESION
            _blend(img, lesions == k, col, lesion_alpha)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if upscale > 1:
        img = np.repeat(np.repeat(img, upscale, axis=0), upscale, axis=1)

    def up(m):
        return np.repeat(np.repeat(m, upscale, axis=0), upscale, axis=1) if upscale > 1 else m

    if show_liver:
        img[_edges(up(arrs["liver"][i] > 0))] = C_LIVER
    if show_lesions:
        for k in ids:
            col = C_SELECTED if selected_lesion is not None and int(k) == selected_lesion else C_LESION
            img[_edges(up(lesions == k))] = col
    if show_reference and "reference" in arrs:
        ref = arrs["reference"][i]
        img[_edges(up(ref == 2))] = C_REF
        img[_edges(up(ref > 0)) & (np.indices(img.shape[:2]).sum(0) % 6 < 3)] = C_REF   # dotted liver outline
    if zoom > 1.0:
        import cv2

        H, W = img.shape[:2]
        ch, cw = int(round(H / zoom)), int(round(W / zoom))
        cy = H / 2 + float(pan[1]) * (H - ch) / 2
        cx = W / 2 + float(pan[0]) * (W - cw) / 2
        y0 = int(np.clip(round(cy - ch / 2), 0, H - ch))
        x0 = int(np.clip(round(cx - cw / 2), 0, W - cw))
        img = cv2.resize(np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw]), (W, H), interpolation=cv2.INTER_NEAREST)
    if markers:
        img = orientation_markers(np.ascontiguousarray(img))
    return img


def neutral_slice_index(arrs: dict) -> int:
    """Start slice that reveals nothing about the AI result: the middle of the stored slab."""
    return len(arrs["z"]) // 2


def key_slice_index(arrs: dict, record: dict) -> int:
    z = arrs["z"]
    les = record.get("measurements", {}).get("lesions", [])
    if les:
        target = les[0]["key_slice"]
        return int(np.argmin(np.abs(z - target)))
    area = (arrs["liver"] > 0).sum(axis=(1, 2))
    return int(np.argmax(area)) if area.any() else len(z) // 2


# --------------------------------------------------------------------------- policies
def all_events(store: RunStore, reader_ids: list[str]) -> dict[str, list[dict]]:
    fs = FeedbackStore(store.p("feedback"))
    return {r: fs.events(r) for r in reader_ids}


def current_policy(policy_name: str, reader_id: str, events_by_reader: dict[str, list[dict]], cases: dict,
                   cfg: dict) -> ReaderPolicy:
    if policy_name == "universal":
        return universal_policy(cfg)
    if policy_name == "universal_defer":
        pooled = [e for evs in events_by_reader.values() for e in evs]
        return fit_policy(pooled, cases, cfg, None, "universal_defer")
    return fit_policy(events_by_reader.get(reader_id, []), cases, cfg, reader_id, policy_name)


def policy_trajectory(events: list[dict], cases: dict, cfg: dict, variant: str = "reader_threshold_defer") -> pd.DataFrame:
    rows = []
    for n in range(0, len(events) + 1):
        p = fit_policy(events[:n], cases, cfg, None, variant)
        rows.append({"n_feedback": n, "threshold": p.threshold,
                     "defer_uncertainty": np.nan if p.defer_uncertainty is None else p.defer_uncertainty,
                     "personalised": p.personalised, "trust_alpha": p.trust_alpha, "trust_beta": p.trust_beta})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- clinical wording
IMAGE_CHECKS = {"voxel_spacing_plausible", "enough_slices"}
SEGMENTATION_CHECKS = {"liver_found", "liver_volume_plausible", "tumour_inside_liver"}


def pct(p) -> str:
    return "-" if p is None or (isinstance(p, float) and np.isnan(p)) else f"{100 * float(p):.0f}%"


def qc_status(record: dict) -> dict:
    """Operational readiness of the AI output - kept separate from any clinical finding or urgency."""
    q = record.get("quality") or {}
    if q.get("passed", True):
        return {"kind": "ok", "label": "QC passed", "detail": "Automatic checks passed.", "consequence": "",
                "manual": ""}
    checks = [c for c in q.get("checks", []) if not c.get("passed", True)]
    names = {c.get("name") for c in checks}
    msgs = [c.get("message", "") for c in checks] or list(q.get("failed_reasons", []))
    if names & IMAGE_CHECKS:
        label = "Image QC failed"
        consequence = ("The scan geometry is outside what the AI was built for: AI measurements may be wrong and "
                       "the AI suggestion is withheld.")
        manual = ("Check the series first. Review the images manually only if they are of diagnostic quality; "
                  "otherwise defer and request the correct series.")
    else:
        label = "Segmentation QC failed"
        consequence = ("The AI segmentation is unreliable, so its masks, measurements and suggestion are withheld. "
                       "This is a failure of the AI output, not of the CT acquisition.")
        manual = "The images can be reviewed manually."
    detail = "; ".join(m[:1].upper() + m[1:] for m in msgs if m)
    return {"kind": "image" if label.startswith("Image") else "segmentation", "label": label,
            "detail": detail, "consequence": consequence, "manual": manual}


def agreement(seg_p: float | None, vlm_p: float | None, score_gap: float = 0.15) -> dict:
    """Class agreement and score agreement are different things - say which one holds."""
    if seg_p is None or vlm_p is None:
        return {"kind": "none", "text": "No second-reader result for this case.",
                "interpretation": "Agreement cannot be assessed. A missing result is not a negative result."}
    s_pos, v_pos = seg_p >= 0.5, vlm_p >= 0.5
    flag = lambda pos: "flags a tumour" if pos else "does not flag a tumour"  # noqa: E731
    if s_pos != v_pos:
        return {"kind": "disagree", "text": f"The AI tools disagree: the segmentation model {flag(s_pos)} "
                                            f"({pct(seg_p)}), the second-reader model {flag(v_pos)} ({pct(vlm_p)}).",
                "interpretation": "The two models reach different conclusions. Neither is a reference standard - "
                                  "rely on your own reading of the images."}
    both = "Both AI tools flag a tumour" if s_pos else "Neither AI tool flags a tumour"
    if abs(seg_p - vlm_p) >= score_gap:
        return {"kind": "differ", "text": f"{both}, but their scores differ: {pct(seg_p)} vs {pct(vlm_p)}.",
                "interpretation": "Same conclusion with different confidence - treat the probability as less certain."}
    return {"kind": "agree", "text": f"{both} with similar scores ({pct(seg_p)} and {pct(vlm_p)}).",
            "interpretation": "Agreement between the models does not guarantee that they are correct."}


UNCERTAINTY_DEFINITION = ("Uncertainty is the predictive entropy of the calibrated probability, on a 0-1 scale "
                          "(0 = certain, 1 = a 50/50 call). Low < 0.50, medium 0.50-0.84, high ≥ 0.85.")


def uncertainty_phrase(u: float) -> str:
    lvl = uncertainty_level(float(u))
    return f"{lvl.capitalize()} uncertainty ({float(u):.2f} on a 0-1 scale)"


def urgent_rule(cfg: dict) -> str:
    t = cfg["triage"]
    return (f"probability ≥ {pct(t['urgent_probability'])} and a lesion ≥ {t['urgent_volume_ml']:g} mL "
            f"or ≥ {t['urgent_diameter_mm']:g} mm")


def short_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:7]


def model_version(cfg: dict, calibration: dict | None, model_card: dict | None) -> dict:
    """Identity of the model set used in this session: segmenter, second reader, calibration."""
    seg = cfg["segmenter"]
    vlm = cfg.get("vlm", {})
    cal = (calibration or {}).get("calibrator", {})
    parts = {"segmenter": seg.get("plugin") or seg["backend"],
             "weights": (model_card or {}).get("source"), "revision": (model_card or {}).get("revision"),
             "img_size": cfg["preprocess"]["img_size"],
             "vlm": (vlm.get("plugin") or vlm.get("backend")) if vlm.get("backend") not in (None, "none") else None,
             "vlm_slices": vlm.get("n_slices") if vlm.get("backend") not in (None, "none") else None,
             "calibration": {k: cal.get(k) for k in ("method", "T", "a", "b", "va", "vb", "w")}}
    return {"id": "M-" + short_hash(parts), "parts": parts}


def policy_version(policy: ReaderPolicy) -> str:
    return "P-" + short_hash({"name": policy.name, "reader": policy.reader_id, "t": round(policy.threshold, 4),
                              "d": None if policy.defer_uncertainty is None else round(policy.defer_uncertainty, 4),
                              "n": policy.n_feedback, "pers": policy.personalised})


STATUS_TEXT = {"pending": "Unread", "impression": "Impression recorded", "accept": "Decided · agreed with AI",
               "edit": "Decided · AI lesions modified", "reject": "Decided · differs from AI",
               "manual": "Decided · independent read", "defer": "Deferred · second opinion"}


def worklist(records: dict, cases: dict, policy: ReaderPolicy, cfg: dict, reviews: dict, splits=("test",),
             impressions: set | None = None) -> pd.DataFrame:
    rows = []
    for cid, r in records.items():
        if r.get("split") not in splits:
            continue
        c = r["confidence"]
        d = policy.decide(c["tumour_probability"], c["uncertainty"], r["quality"]["passed"])
        t = triage(c, r["measurements"], r["quality"], cfg, threshold=policy.threshold)
        cat = t["category"]
        qc = qc_status(r)
        if cat == "QC FAIL":
            shown_cat = "QC FAIL"
            suggestion = "Withheld - automatic QC failed"
            reasons = ["Placed early: needs a full manual read because the AI output is unreliable"]
        elif d["decision"] == "defer":
            shown_cat = "DEFERRED"
            suggestion = "Withheld for this reader"
            reasons = [f"AI suggestion withheld by this reader's policy ({d.get('reason', 'policy')})"] + \
                list(t.get("reasons", []))
        else:
            shown_cat = cat
            suggestion = suggestion_text(c["tumour_probability"], policy.threshold)
            reasons = list(t.get("reasons", []))
        rv = reviews.get(cid)
        vp = c.get("vlm_probability")
        status = rv["action"] if rv else ("impression" if impressions and cid in impressions else "pending")
        rows.append({"case_id": cid, "priority": t["priority"], "category": shown_cat,
                     "arrival": int(cases.get(cid, {}).get("number", 0) or _case_number(cid)),
                     "qc_kind": qc["kind"], "qc_label": qc["label"], "qc_detail": qc["detail"],
                     "second_reader": float(vp) if vp is not None else float("nan"),
                     "agree": agreement(c.get("segmenter_probability"), vp)["kind"],
                     "reasons": "; ".join(humanize(x) for x in reasons),
                     "unc_level": uncertainty_level(float(c["uncertainty"])),
                     "triage": f"{CATEGORY_ICON.get(shown_cat, '')} {CATEGORY_LABEL.get(shown_cat, shown_cat)}",
                     "ai_suggestion": suggestion,
                     "tumour_probability": c["tumour_probability"], "uncertainty": c["uncertainty"],
                     "tumour_ml": r["measurements"].get("predicted_tumour_volume_ml", 0.0),
                     "lesions": r["measurements"].get("n_lesions", 0),
                     "universal_category": r["triage"]["category"],
                     "status": status, "status_text": STATUS_TEXT.get(status, status)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["priority", "case_id"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def _case_number(cid: str) -> int:
    digits = "".join(ch for ch in cid if ch.isdigit())
    return int(digits) if digits else 0


def order_worklist(wl: pd.DataFrame, blind: bool, sort: str = "priority") -> pd.DataFrame:
    """Blind first read: arrival order, never an order derived from hidden AI results."""
    if blind or sort == "arrival":
        out = wl.sort_values(["arrival", "case_id"]).reset_index(drop=True)
    else:
        out = wl.sort_values(["rank"]).reset_index(drop=True)
    out["position"] = np.arange(1, len(out) + 1)
    return out


# --------------------------------------------------------------------------- simulated / live reads
def _draws(reader_id: str, case_id: str, salt: int = 0) -> np.ndarray:
    h = int(hashlib.sha256(f"{reader_id}|{case_id}|{salt}".encode()).hexdigest()[:12], 16)
    return np.random.default_rng(h).random(N_DRAWS)


def simulate_and_log(store: RunStore, profile: ReaderProfile, case: dict, policy: ReaderPolicy, salt: int = 0,
                     audit: dict | None = None) -> dict:
    d = policy.decide(case["p"], case["u"], case["qc_passed"])
    r = simulate_read(profile, case, d, _draws(profile.reader_id, case["case_id"], salt))
    ev = FeedbackEvent(reader_id=profile.reader_id, case_id=case["case_id"], policy=policy.name, shown=r["shown"],
                       ai_label=d["ai_label"], ai_probability=case["p"], ai_confidence=d["confidence"],
                       uncertainty=case["u"], threshold=policy.threshold, initial_read=r["initial_read"],
                       action=r["action"], final_label=r["final_label"], source="simulated-ui",
                       **(audit or {})).to_dict()
    FeedbackStore(store.p("feedback")).append(ev)
    return {**ev, "decision": d, "gt_present": case["gt_present"], "switched": r["switched"]}


def log_live(store: RunStore, case: dict, policy: ReaderPolicy, initial_read: int | None, action: str,
             final_label: int | None, edit: dict | None = None, audit: dict | None = None) -> dict:
    """Save a live decision with what was shown at decision time (``audit``: session, versions,
    recommendation text, impression / reveal times)."""
    d = policy.decide(case["p"], case["u"], case["qc_passed"])
    ev = FeedbackEvent(reader_id=LIVE_READER, case_id=case["case_id"], policy=policy.name,
                       shown=d["decision"] == "show", ai_label=d["ai_label"], ai_probability=case["p"],
                       ai_confidence=d["confidence"], uncertainty=case["u"], threshold=policy.threshold,
                       initial_read=initial_read, action=action,
                       final_label=None if final_label is None else int(final_label), source="live",
                       edit=edit, **(audit or {})).to_dict()
    FeedbackStore(store.p("feedback")).append(ev)
    return ev


def reader_profiles(cfg: dict) -> dict:
    return load_profiles(cfg)


def outcome_label(ev: dict, gt_present: bool) -> str:
    if ev.get("final_label") is None:
        return "deferred (no final finding yet)"
    ok = int(ev["final_label"]) == int(gt_present)
    ai_ok = ev.get("ai_label") is not None and int(ev["ai_label"]) == int(gt_present)
    if not ev["shown"]:
        return "correct (independent read)" if ok else "incorrect (independent read)"
    if ev["action"] in ("accept", "edit"):
        return "correct acceptance" if ai_ok else "HARMFUL acceptance (AI was wrong)"
    return "correct rejection (AI was wrong)" if not ai_ok else "wrong rejection (AI was right)"


def humanize(reason: str) -> str:
    """Make a triage/policy reason readable: symbols and plain words instead of code-like notation."""
    r = str(reason).replace(">=", "≥").replace("<=", "≤")
    r = r.replace("p = ", "probability ").replace("p ", "probability ", 1) if r.startswith("p ") else \
        r.replace("p = ", "probability ")
    return r[:1].upper() + r[1:]


# --------------------------------------------------------------------------- persistence: impressions, authorised policies
def _impressions_path(store: RunStore, reader_id: str) -> Path:
    return Path(store.p("feedback")) / f"{reader_id}_impressions.jsonl"


def save_impression(store: RunStore, reader_id: str, case_id: str, value: int, session_id: str) -> dict:
    """Independent impressions are written to disk at once, so an interrupted session can resume safely."""
    from datetime import datetime, timezone

    rec = {"reader_id": reader_id, "case_id": case_id, "value": int(value), "session_id": session_id,
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "at_local": datetime.now().astimezone().strftime("%H:%M")}
    p = _impressions_path(store, reader_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def load_impressions(store: RunStore, reader_id: str) -> dict:
    """First recorded impression per case (the original impression is never replaced)."""
    p = _impressions_path(store, reader_id)
    out: dict = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["case_id"], r)
    return out


def clear_impressions(store: RunStore, reader_id: str) -> None:
    p = _impressions_path(store, reader_id)
    if p.exists():
        p.unlink()


def _authorised_path(store: RunStore) -> Path:
    return Path(store.p("feedback")) / "authorised_policies.json"


def authorised_policy(store: RunStore, cfg: dict, reader_id: str, policy_name: str) -> tuple[ReaderPolicy, dict | None]:
    """The policy a reading session may use: the last explicitly authorised one, else the universal policy."""
    p = _authorised_path(store)
    data = json.loads(p.read_text()) if p.exists() else {}
    entry = data.get(f"{reader_id}|{policy_name}")
    if policy_name == "universal" or not entry:
        pol = universal_policy(cfg)
        if policy_name != "universal":
            pol.notes.append("no authorised personalisation yet - universal policy in use")
        return pol, None
    d = dict(entry["policy"])
    d["notes"] = list(d.get("notes", []))
    return ReaderPolicy(**d), entry


def authorise_policy(store: RunStore, reader_id: str, policy_name: str, policy: ReaderPolicy, by: str, reason: str,
                     session_id: str, previous: str) -> dict:
    """Record an explicit, attributed authorisation; it takes effect from the next reading session."""
    from datetime import datetime, timezone

    p = _authorised_path(store)
    data = json.loads(p.read_text()) if p.exists() else {}
    entry = {"policy": policy.to_dict(), "version": policy_version(policy), "authorised_by": by, "reason": reason,
             "authorised_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "session_id": session_id,
             "previous": previous}
    data[f"{reader_id}|{policy_name}"] = entry
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1, default=str))
    with open(Path(store.p("feedback")) / "policy_changes.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": entry["authorised_at"], "reader_id": reader_id, "policy": policy_name,
                            "from": previous, "to": entry["version"], "threshold_to": policy.threshold,
                            "withhold_to": policy.defer_uncertainty, "n_decisions": policy.n_feedback,
                            "authorised_by": by, "reason": reason, "session_id": session_id}) + "\n")
    return entry


def neighbours(order: list[str], case_id: str) -> tuple[str | None, str | None]:
    """Previous / next case in worklist order."""
    if case_id not in order:
        return None, (order[0] if order else None)
    i = order.index(case_id)
    return (order[i - 1] if i > 0 else None), (order[i + 1] if i + 1 < len(order) else None)


__all__ = ["CATEGORY_ORDER", "LIVE_READER", "POLICY_CHOICES", "find_runs", "load_run", "render_slice", "worklist",
           "current_policy", "policy_trajectory", "simulate_and_log", "log_live", "key_slice_index", "all_events"]
