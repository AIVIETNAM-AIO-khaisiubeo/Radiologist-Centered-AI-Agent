"""Visual design of the reading interface: tokens, CSS and small HTML components.

Design rationale, the radiologist journey and the responses to the design reviews: ``docs/UX_DESIGN.md``.
Everything here is plain Python returning strings (no Streamlit import), so it can be unit-tested.
All dynamic text is HTML-escaped.

Four status concepts are kept visually and verbally separate:
technical validity (QC, amber square) · clinical urgency by the AI rule (red triangle, rare) ·
workflow status (neutral text) · AI evidence characteristics such as disagreement (amber, labelled).
Colour roles: navy = structure and selection · teal = primary actions only · red = the validated urgent
rule only · amber = caution / review needed · slate = neutral and technical. No green.
"""
from __future__ import annotations

import html
import math

# --------------------------------------------------------------------------- tokens
NAVY = "#102D52"
NAVY_2 = "#163A66"
INK = "#1B2B40"
SLATE = "#526579"
TEAL = "#0F766E"
TEAL_DARK = "#0B5F59"
TEAL_TINT = "#E3F3F1"
TEAL_SOFT = "#8FD3CB"
RED = "#B42318"
RED_BG = "#FEF3F2"
AMBER = "#B54708"
AMBER_BG = "#FFF6E5"
AMBER_INK = "#7A2E0E"
SLATE_BG = "#F1F4F7"
SELECT_BG = "#EEF3F9"
HOVER_BG = "#F8FAFB"
SURFACE = "#F7F8FA"
BORDER = "#D6DEE6"
BORDER_STRONG = "#8A9BAD"
VIEWER_BG = "#0A1628"
VIEWER_INK = "#D7E1EC"
MUTED = SLATE

SPACE = {"xs": 4, "s": 8, "m": 12, "l": 16, "xl": 24, "xxl": 32}   # 8-px system

OVERLAY = {
    "liver": (64, 201, 186),       # teal contour
    "lesion": (255, 170, 51),      # amber contour (+ optional fill, off by default)
    "selected": (255, 230, 120),
    "reference": (245, 247, 250),
    "uncertainty": (199, 125, 255),  # violet, opacity = voxel uncertainty
}

_ICONS = {
    "triangle": '<path d="M8 1.6 15.2 14.4H.8z"/>',
    "diamond": '<path d="M8 .8 15.2 8 8 15.2.8 8z"/>',
    "half": '<circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" stroke-width="2"/>'
            '<path d="M8 1.8a6.2 6.2 0 0 1 0 12.4z"/>',
    "ring": '<circle cx="8" cy="8" r="5.6" fill="none" stroke="currentColor" stroke-width="2.4"/>',
    "person": '<circle cx="8" cy="4.8" r="3"/><path d="M2.4 15c0-3.3 2.5-5.6 5.6-5.6s5.6 2.3 5.6 5.6z"/>',
    "square": '<rect x="1.6" y="1.6" width="12.8" height="12.8" rx="2.2" fill="none" stroke="currentColor" '
              'stroke-width="2"/><rect x="7" y="4" width="2" height="5.4" rx=".6"/><rect x="7" y="10.9" '
              'width="2" height="2" rx=".6"/>',
    "check": '<path d="M6.3 12.6 2 8.3l1.6-1.6 2.7 2.7 6.1-6.1L14 4.9z"/>',
    "alert": '<path d="M8 1.2 15.4 14.6H.6z"/><rect x="7.1" y="5.6" width="1.8" height="4.6" fill="#fff"/>'
             '<rect x="7.1" y="11.3" width="1.8" height="1.8" fill="#fff"/>',
    "info": '<circle cx="8" cy="8" r="7"/><rect x="7.1" y="7" width="1.8" height="5" fill="#fff"/>'
            '<rect x="7.1" y="3.8" width="1.8" height="1.8" fill="#fff"/>',
    "lock": '<rect x="2.5" y="7" width="11" height="8" rx="1.6"/><path d="M5 7V5a3 3 0 0 1 6 0v2" fill="none" '
            'stroke="currentColor" stroke-width="1.8"/>',
    "dot": '<circle cx="8" cy="8" r="3.2"/>',
    "clock": '<circle cx="8" cy="8" r="6.4" fill="none" stroke="currentColor" stroke-width="1.8"/>'
             '<path d="M7.2 4h1.6v4.2l2.6 1.6-.8 1.3-3.4-2z"/>',
    "pen": '<path d="M11.6 1.6 14.4 4.4 5.2 13.6 1.6 14.4 2.4 10.8z"/>',
}

# AI category (an AI output, never a diagnosis): (css modifier, label, icon)
CATEGORY_STYLE = {
    "URGENT": ("urgent", "Urgent by AI rule", "triangle"),
    "SUSPICIOUS": ("caution", "AI: tumour suspected", "diamond"),
    "UNCERTAIN": ("caution", "AI: uncertain", "half"),
    "LIKELY NORMAL": ("neutral", "AI: no tumour suspected", "ring"),
    "DEFERRED": ("neutral", "AI withheld (policy)", "person"),
    "QC FAIL": ("neutral", "AI withheld (QC)", "square"),
}

# workflow status: (css modifier, label, icon)
STATUS_STYLE = {
    "pending": ("neutral", "Unread", "dot"),
    "impression": ("navy", "Impression saved", "pen"),
    "accept": ("navy", "Decided · agrees with AI", "check"),
    "edit": ("navy", "Decided · AI lesions modified", "check"),
    "reject": ("navy", "Decided · differs from AI", "check"),
    "manual": ("navy", "Decided · independent read", "check"),
    "defer": ("caution", "Deferred · second opinion", "clock"),
}


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def icon(name: str, size: int = 16, label: str | None = None) -> str:
    aria = f'role="img" aria-label="{esc(label)}"' if label else 'aria-hidden="true"'
    return (f'<svg class="ra-ico" width="{size}" height="{size}" viewBox="0 0 16 16" fill="currentColor" '
            f'{aria} focusable="false">{_ICONS[name]}</svg>')


def pct(p) -> str:
    return "-" if p is None or (isinstance(p, float) and math.isnan(p)) else f"{100 * float(p):.0f}%"


# --------------------------------------------------------------------------- text-level components
def badge(category: str) -> str:
    mod, label, ico = CATEGORY_STYLE.get(category, ("neutral", category.title(), "dot"))
    return f'<span class="ra-tag ra-t-{mod}">{icon(ico, 14)}<span>{esc(label)}</span></span>'


def status_badge(status: str) -> str:
    mod, label, ico = STATUS_STYLE.get(status, ("neutral", status, "dot"))
    return f'<span class="ra-tag ra-t-{mod}">{icon(ico, 14)}<span>{esc(label)}</span></span>'


def qc_badge(kind: str, label: str, detail: str = "") -> str:
    t = f' title="{esc(detail)}"' if detail else ""
    if kind == "ok":
        return f'<span class="ra-tag ra-t-neutral"{t}>{icon("check", 14)}<span>{esc(label)}</span></span>'
    return f'<span class="ra-tag ra-t-technical"{t}>{icon("square", 14)}<span>{esc(label)}</span></span>'


def hidden_finding() -> str:
    return f'<span class="ra-tag ra-t-neutral">{icon("lock", 14)}<span>AI findings hidden</span></span>'


def top_identity() -> str:
    return ('<div class="ra-id"><span class="ra-prod">ReaderAdapt</span><span class="ra-prod-sub">Liver CT</span>'
            f'<span class="ra-research">{icon("info", 13)}Research prototype · not for clinical use</span></div>')


def top_state(reader: str, mode: str | None, session_short: str, locked: bool) -> str:
    parts = [f"Reader: <b>{esc(reader)}</b>"]
    if mode:
        parts.append(esc(mode))
    parts.append(f"Session {esc(session_short)}" + (f" {icon('lock', 12, 'settings locked')}" if locked else ""))
    return f'<div class="ra-state">{" · ".join(parts)}</div>'


def summary_line(groups: list[tuple[str, list[tuple[str, str]]]]) -> str:
    """Grouped queue summary; each group is one concept: workflow / technical / urgency / evidence."""
    ico = {"workflow": "", "technical": icon("square", 13), "urgency": icon("triangle", 13),
           "evidence": icon("alert", 13)}
    out = []
    for kind, items in groups:
        txt = " · ".join(f"<b>{esc(v)}</b> {esc(lbl)}" for lbl, v in items)
        out.append(f'<span class="ra-sum ra-sum-{esc(kind)}">{ico.get(kind, "")}{txt}</span>')
    return f'<div class="ra-summary" role="status">{"".join(out)}</div>'


def progress(current: int, done: set[int], labels=("Independent read", "AI evidence", "Decision")) -> str:
    items = []
    for i, n in enumerate(labels, start=1):
        state = "done" if i in done else ("current" if i == current else "todo")
        mark = icon("check", 11) if state == "done" else str(i)
        cur = ' aria-current="step"' if state == "current" else ""
        items.append(f'<li class="ra-pg ra-pg-{state}"{cur}><span class="ra-pg-n">{mark}</span>{esc(n)}</li>')
    return f'<ol class="ra-progress" aria-label="Reading progress">{"".join(items)}</ol>'


stepper = progress


def case_identity(case_id: str, status: str, meta: str) -> str:
    return (f'<div class="ra-caseid"><span class="ra-case">{esc(case_id)}</span>{status_badge(status)}</div>'
            f'<div class="ra-meta">{esc(meta)}</div>')


def meter(p: float, threshold: float | None = None, label: str = "Tumour probability") -> str:
    p = 0.0 if p is None or (isinstance(p, float) and math.isnan(p)) else float(p)
    pc = max(0.0, min(100.0, 100.0 * p))
    thr = ""
    if threshold is not None:
        thr = (f'<span class="ra-meter-thr" style="left:{100 * threshold:.1f}%" '
               f'title="suggestion threshold {pct(threshold)}"></span>')
    return (f'<div class="ra-meter" role="img" aria-label="{esc(label)} {pct(p)}'
            + (f', threshold {pct(threshold)}' if threshold is not None else "") + '">'
            f'<div class="ra-meter-fill" style="width:{pc:.1f}%"></div>{thr}</div>')


def comparison(impression: int | None, impression_time: str | None, ai_label: int | None, ai_text: str) -> str:
    """Your impression and the AI suggestion as information - not a control."""
    if impression is None:
        mine = '<span class="ra-muted">Not recorded</span>'
    else:
        mine = f"{'Tumour present' if impression else 'No liver tumour'}"
        if impression_time:
            mine += f' <span class="ra-muted">· saved {esc(impression_time)}</span>'
    rel = ""
    if impression is not None and ai_label is not None:
        rel = ('<div class="ra-rel">Same binary finding (tumour present / absent).</div>' if impression == ai_label
               else '<div class="ra-rel ra-rel-diff">Different binary finding from your impression.</div>')
    return (f'<div class="ra-cmp"><div><div class="ra-k">Your impression</div><div class="ra-v">{mine}</div></div>'
            f'<div><div class="ra-k">AI suggestion</div><div class="ra-v">{esc(ai_text)}</div></div></div>{rel}')


def evidence(*, finding: str, p: float, threshold: float, target: str, unc_level: str, unc_text: str,
             qc_ok: bool, agreement_kind: str | None = None, agreement_text: str = "",
             agreement_interp: str = "") -> str:
    """AI evidence in reading order: finding -> probability and target -> uncertainty and validity -> agreement.

    ``agreement_kind=None`` (one model, no second reader) leaves the agreement row out."""
    unc_cls = "ra-dd-caution" if unc_level == "high" else ""
    ag_cls = "ra-dd-caution" if agreement_kind in ("disagree", "differ") else ""
    ag_ico = icon("alert", 13) if agreement_kind in ("disagree", "differ") else ""
    ag_row = "" if agreement_kind is None else (
        f'<div><dt>Model agreement</dt><dd class="{ag_cls}">{ag_ico}{esc(agreement_text)}'
        f'<span class="ra-interp">{esc(agreement_interp)}</span></dd></div>')
    return f"""
<section class="ra-ev" aria-label="AI evidence">
  <div class="ra-ev-find">{esc(finding)}</div>
  <div class="ra-ev-p"><span class="ra-ev-num">{pct(p)}</span><span class="ra-ev-target">{esc(target)}</span></div>
  {meter(p, threshold, "Patient-level tumour probability")}
  <dl class="ra-dl">
    <div><dt>Uncertainty</dt><dd class="{unc_cls}">{esc(unc_text)}</dd></div>
    <div><dt>Validity</dt><dd>{'Automatic quality checks passed.' if qc_ok else 'Quality checks failed - output withheld.'}</dd></div>
    {ag_row}
  </dl>
</section>"""


def withheld(title: str, rows: list[tuple[str, str]]) -> str:
    dl = "".join(f"<div><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>" for k, v in rows)
    return (f'<section class="ra-ev ra-ev-withheld" aria-label="AI output withheld">'
            f'<div class="ra-ev-find">{icon("square", 15)} {esc(title)}</div><dl class="ra-dl">{dl}</dl></section>')


def save_state(kind: str, text: str) -> str:
    """kind: none | unsaved | saved | failed"""
    ico = {"none": "dot", "unsaved": "pen", "saved": "check", "failed": "alert"}[kind]
    return f'<span class="ra-save ra-save-{esc(kind)}" role="status">{icon(ico, 13)}{esc(text)}</span>'


def measures(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(f'<div><dt>{esc(lbl)}</dt><dd>{esc(v)}<span>{esc(u)}</span></dd></div>' for lbl, v, u in items)
    return f'<dl class="ra-measures">{cells}</dl>'


def legend(items: list[tuple[str, tuple[int, int, int], str]], uncertainty: bool = False) -> str:
    out = []
    for label, rgb, kind in items:
        col = "rgb({},{},{})".format(*rgb)
        out.append(f'<span class="ra-leg"><i class="ra-leg-{kind}" style="--c:{col}"></i>{esc(label)}</span>')
    if uncertainty:
        c = "rgb({},{},{})".format(*OVERLAY["uncertainty"])
        out.append(f'<span class="ra-leg">Uncertainty <i class="ra-leg-grad" style="--c:{c}"></i>'
                   '<span class="ra-leg-s">low → high</span></span>')
    return f'<div class="ra-legend" aria-label="Overlay legend">{"".join(out)}</div>'


def flow(steps: list[tuple[str, str, str]]) -> str:
    li = "".join(f'<li class="ra-flow-{esc(k)}"><b>{esc(t)}</b><span>{esc(d)}</span></li>' for t, d, k in steps)
    return f'<ol class="ra-flow" aria-label="How a case flows through the assistant">{li}</ol>'


def note(text: str, kind: str = "info") -> str:
    ico = {"info": "info", "warn": "alert", "neutral": "check", "critical": "alert", "technical": "square"}[kind]
    return f'<div class="ra-note ra-note-{kind}" role="note">{icon(ico, 15)}<div>{text}</div></div>'


def page_title(title: str, subtitle: str | None = None) -> str:
    sub = f'<p class="ra-sec-sub">{esc(subtitle)}</p>' if subtitle else ""
    return f'<div class="ra-sec"><h1 class="ra-sec-t">{esc(title)}</h1>{sub}</div>'


section_title = page_title


def panel_title(n: str | None, title: str, sub: str | None = None, right_html: str = "", divider: bool = False) -> str:
    num = f'<span class="ra-pn">{esc(n)}</span>' if n else ""
    s = f'<div class="ra-panel-s">{esc(sub)}</div>' if sub else ""
    cls = "ra-panel-h ra-panel-div" if divider else "ra-panel-h"
    return f'<div class="{cls}"><div class="ra-panel-t">{num}{esc(title)}</div>{right_html}</div>{s}'


def cell(label: str, content_html: str) -> str:
    return f'<div class="ra-cell">{content_html}</div>'


def mobile_card(position: int, case_id: str, status: str, qc_kind: str, qc_label: str, ai_line: str,
                hint: str = "") -> str:
    """Case-first mobile card: identity + status, then QC, then AI visibility. One action below it."""
    return (f'<div class="ra-mcard"><div class="ra-mc-k">Case {position:02d}</div>'
            f'<div class="ra-mc-top"><span class="ra-case">{esc(case_id)}</span>{status_badge(status)}</div>'
            f'<div class="ra-mc-row">{qc_badge(qc_kind, qc_label)}</div>'
            f'<div class="ra-mc-row">{ai_line}</div>'
            + (f'<div class="ra-mc-hint">{esc(hint)}</div>' if hint else "") + '</div>')


# --------------------------------------------------------------------------- CSS
CSS = f"""
:root {{
  --ra-navy:{NAVY}; --ra-navy-2:{NAVY_2}; --ra-ink:{INK}; --ra-slate:{SLATE};
  --ra-teal:{TEAL}; --ra-teal-dark:{TEAL_DARK}; --ra-red:{RED}; --ra-red-bg:{RED_BG};
  --ra-amber:{AMBER}; --ra-amber-bg:{AMBER_BG}; --ra-amber-ink:{AMBER_INK}; --ra-slate-bg:{SLATE_BG};
  --ra-select:{SELECT_BG}; --ra-hover:{HOVER_BG}; --ra-surface:{SURFACE}; --ra-border:{BORDER};
  --ra-border-strong:{BORDER_STRONG}; --ra-viewer:{VIEWER_BG}; --ra-vink:{VIEWER_INK};
  --s1:4px; --s2:8px; --s3:12px; --s4:16px; --s5:24px; --s6:32px;
  --r-ctl:8px; --r-panel:12px; --h-ctl:44px;
}}
/* ---------- shell: light, navy structure. Type: body 15-16, table 14-15, meta 13-14, case 18, heading 22 */
[data-testid="stApp"], [data-testid="stAppViewContainer"] {{ background:#FFFFFF; color:var(--ra-ink); }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] {{ display:none !important; }}
[data-testid="stMainBlockContainer"], .block-container {{ max-width:1680px; padding:3.1rem var(--s5) var(--s6); }}
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{ font-size:0.9688rem; line-height:1.5; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ font-size:0.8438rem; color:var(--ra-slate); }}
[data-testid="stWidgetLabel"] p {{ font-size:0.875rem !important; font-weight:500; color:var(--ra-ink); }}
h1, h2, h3, h4 {{ color:var(--ra-navy) !important; letter-spacing:-0.005em; }}
[data-testid="stMarkdownContainer"] h4 {{ font-size:1rem; font-weight:700; padding:var(--s2) 0 var(--s1); }}
header[data-testid="stHeader"] {{ background:rgba(255,255,255,0.97); border-bottom:1px solid var(--ra-border); }}
a {{ color:var(--ra-teal-dark); }}
.ra-muted {{ color:var(--ra-slate); font-weight:400; }}
.ra-ico {{ flex:none; vertical-align:-2px; }}
.ra-device-note {{ display:none; }}
[data-testid="stVerticalBlock"] {{ gap:var(--s3); }}
/* markdown blocks that hold only our HTML components must not pull the next element up */
[data-testid="stMarkdownContainer"]:not(:has(> p, > ul, > ol, > h4)) {{ margin-bottom:0 !important; }}

/* ---------- top bar (one band): identity · modes · state · session · research */
.st-key-rx_topbar [data-testid="stHorizontalBlock"] {{ gap:var(--s2) var(--s3) !important; }}
.st-key-topbar {{ border-bottom:1px solid var(--ra-border); padding-bottom:var(--s3); margin-bottom:var(--s1); }}
.ra-id {{ display:flex; align-items:baseline; gap:var(--s2); flex-wrap:wrap; }}
.ra-prod {{ font-size:1.25rem; font-weight:700; color:var(--ra-navy); }}
.ra-prod-sub {{ font-size:1rem; color:var(--ra-slate); font-weight:500; }}
.ra-research {{ display:inline-flex; gap:4px; align-items:center; font-size:0.8125rem; font-weight:600;
  color:var(--ra-amber-ink); background:var(--ra-amber-bg); border-radius:6px; padding:2px 8px; }}
.ra-state {{ font-size:0.8438rem; color:var(--ra-slate); margin-top:2px; }}
.ra-state b {{ color:var(--ra-ink); font-weight:600; }}
.st-key-nav [data-testid="stWidgetLabel"] {{ display:none; }}
.st-key-nav button {{ min-height:40px; padding:0 var(--s4); font-weight:600; color:var(--ra-navy); }}
.st-key-nav [data-testid="stButtonGroup"] button[aria-checked="true"] {{
  background:var(--ra-navy) !important; color:#fff !important; border-color:var(--ra-navy) !important; }}

/* ---------- headings */
.ra-sec {{ margin:var(--s1) 0 var(--s1); }}
.ra-sec-t {{ font-size:1.375rem !important; font-weight:700; margin:0 !important; padding:0 !important; }}
.ra-sec-sub {{ color:var(--ra-slate); margin:2px 0 0; font-size:0.9375rem; }}
.ra-crumb {{ font-size:0.8438rem; color:var(--ra-slate); }}

/* ---------- queue summary: four separate concepts */
.ra-summary {{ display:flex; flex-wrap:wrap; gap:var(--s1) var(--s5); font-size:0.9375rem; padding:var(--s2) 0;
  border-bottom:1px solid var(--ra-border); }}
.ra-sum {{ display:inline-flex; gap:6px; align-items:center; color:var(--ra-ink); }}
.ra-sum b {{ font-variant-numeric:tabular-nums; }}
.ra-sum-technical {{ color:var(--ra-amber-ink); }} .ra-sum-urgency {{ color:var(--ra-red); }}
.ra-sum-evidence {{ color:var(--ra-amber-ink); }}

/* ---------- controls: one height, one radius; teal only for primary actions */
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button, [data-testid="stPopover"] button,
[data-testid="stFormSubmitButton"] button {{ min-height:var(--h-ctl); padding:0 var(--s4); border-radius:var(--r-ctl);
  font-size:0.9375rem; font-weight:600; width:100%; }}
[data-testid="stBaseButton-primary"] {{ background:var(--ra-teal); border-color:var(--ra-teal); color:#fff; }}
[data-testid="stBaseButton-primary"]:hover {{ background:var(--ra-teal-dark); border-color:var(--ra-teal-dark); color:#fff; }}
[data-testid="stBaseButton-secondary"] {{ background:#fff; color:var(--ra-navy); border:1px solid var(--ra-border-strong); }}
[data-testid="stBaseButton-secondary"]:hover {{ background:var(--ra-hover); color:var(--ra-navy); border-color:var(--ra-navy); }}
[data-testid="stBaseButton-tertiary"] {{ color:var(--ra-navy); }}
[data-testid="stButtonGroup"] button {{ min-height:var(--h-ctl); font-size:0.9375rem; padding:0 var(--s4); color:var(--ra-navy); }}
[data-testid="stButtonGroup"] button[aria-checked="true"] {{ background:var(--ra-select); color:var(--ra-navy);
  border-color:var(--ra-navy); font-weight:700; }}
[class*="st-key-imp_"], [class*="st-key-final_"] {{ width:100% !important; }}
[class*="st-key-imp_"] [role="radiogroup"], [class*="st-key-final_"] [role="radiogroup"] {{ display:flex; width:100%; }}
[class*="st-key-imp_"] [role="radiogroup"] button, [class*="st-key-final_"] [role="radiogroup"] button {{
  flex:1 1 0; min-height:48px; font-size:0.9688rem; }}
div[data-baseweb="select"] > div, [data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"],
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {{ min-height:var(--h-ctl); font-size:0.9375rem; }}
[data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"] button {{ min-width:28px; min-height:28px; display:inline-flex; align-items:center; justify-content:center; }}
[data-testid="stMainMenu"] button {{ min-width:40px; min-height:40px; }}
[data-testid="stSlider"] [role="slider"] {{ width:22px; height:22px; }}
[data-testid="stCheckbox"] label {{ min-height:40px; display:flex; align-items:center; }}
[data-testid="stExpander"] details {{ border-radius:var(--r-ctl); border:1px solid var(--ra-border); }}
[data-testid="stExpander"] summary {{ min-height:var(--h-ctl); font-weight:600; font-size:0.9375rem; color:var(--ra-navy); }}
button:focus-visible, [role="slider"]:focus-visible, input:focus-visible, summary:focus-visible, a:focus-visible,
[tabindex]:focus-visible {{ outline:3px solid var(--ra-navy) !important; outline-offset:2px !important; }}
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ animation:none !important; transition:none !important; }} }}

/* ---------- worklist table: neutral rows, one selection model, hover != selection */
.st-key-rowhead {{ border-bottom:1px solid var(--ra-border-strong); padding:var(--s1) var(--s2); }}
.ra-th {{ font-size:0.8125rem; font-weight:600; color:var(--ra-slate); text-transform:uppercase; letter-spacing:0.04em; }}
[class*="st-key-row_"] {{ border-bottom:1px solid var(--ra-border); padding:var(--s1) var(--s2); gap:0 !important; }}
[class*="st-key-row_"]:hover {{ background:var(--ra-hover); }}
[class*="st-key-row_"] [data-testid="stHorizontalBlock"], .st-key-rowhead [data-testid="stHorizontalBlock"] {{
  flex-wrap:nowrap !important; align-items:center; gap:var(--s4) !important; }}
[class*="st-key-row_"] [data-testid="stColumn"], .st-key-rowhead [data-testid="stColumn"] {{ min-width:0 !important; }}
[class*="st-key-row_"] [data-testid="stMarkdownContainer"], .st-key-rowhead [data-testid="stMarkdownContainer"] {{
  margin-bottom:0 !important; }}
.ra-row-sel {{ background:var(--ra-select) !important; box-shadow:inset 3px 0 0 var(--ra-navy); }}
.ra-pos {{ font-size:0.875rem; color:var(--ra-slate); font-variant-numeric:tabular-nums; }}
.ra-case {{ font-size:1.125rem; font-weight:700; color:var(--ra-navy); }}
.ra-cell {{ font-size:0.9375rem; line-height:1.35; }}
.ra-cell-d {{ font-size:0.8125rem; color:var(--ra-slate); margin-top:2px; }}
.ra-p {{ font-variant-numeric:tabular-nums; font-weight:600; color:var(--ra-ink); margin-left:6px; }}
.ra-flag {{ color:var(--ra-amber-ink); font-size:0.8125rem; font-weight:600; margin-left:6px; white-space:nowrap; }}
.ra-desk {{ }} .ra-mobile {{ display:none; }}
.st-key-mobilelist {{ display:none; }}

/* ---------- tags: icon + coloured text, no fills or borders (one emphasis technique) */
.ra-tag {{ display:inline-flex; align-items:center; gap:5px; font-size:0.875rem; line-height:1.3; font-weight:500; }}
.ra-t-neutral {{ color:var(--ra-slate); }} .ra-t-navy {{ color:var(--ra-navy); }}
.ra-t-caution {{ color:var(--ra-amber-ink); }} .ra-t-technical {{ color:var(--ra-amber-ink); font-weight:600; }}
.ra-t-urgent {{ color:var(--ra-red); font-weight:700; }}

/* ---------- reading view: case bar, progress */
.ra-caseid {{ display:flex; flex-wrap:wrap; align-items:baseline; gap:var(--s2) var(--s3); }}
.ra-meta {{ font-size:0.8438rem; color:var(--ra-slate); }}
ol.ra-progress {{ list-style:none; display:flex !important; flex-direction:row !important; flex-wrap:wrap; gap:var(--s1) var(--s4);
  padding:0 !important; margin:0 !important; }}
.ra-progress .ra-pg {{ margin:0 !important; padding:0 !important; display:flex; align-items:center; gap:6px; font-size:0.8438rem;
  color:var(--ra-slate); }}
.ra-pg-n {{ width:20px; height:20px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center;
  border:1.5px solid var(--ra-border-strong); font-size:0.6875rem; font-weight:700; }}
.ra-pg-current {{ color:var(--ra-navy) !important; font-weight:700; }}
.ra-pg-current .ra-pg-n {{ background:var(--ra-navy); border-color:var(--ra-navy); color:#fff; }}
.ra-pg-done .ra-pg-n {{ background:var(--ra-slate-bg); border-color:var(--ra-slate); color:var(--ra-navy); }}
.st-key-pending .ra-panel-s {{ margin-bottom:var(--s2); }}
.st-key-pending {{ border:1px solid var(--ra-amber); background:var(--ra-amber-bg); border-radius:var(--r-panel); padding:var(--s3) var(--s4); }}

/* ---------- CT viewer: dark, unobtrusive controls */
.st-key-viewer {{ background:var(--ra-viewer); border-radius:var(--r-panel); padding:var(--s3) var(--s3) var(--s4); gap:var(--s2) !important; }}
.st-key-viewer [data-testid="stMarkdownContainer"] {{ margin-bottom:0 !important; }}
.st-key-viewer [data-testid="stWidgetLabel"] p, .st-key-viewer [data-testid="stCheckbox"] p,
.st-key-viewer [data-testid="stCaptionContainer"] p {{ color:var(--ra-vink) !important; }}
.st-key-viewer button p {{ color:inherit !important; }}
.st-key-viewer [data-testid="stBaseButton-secondary"], .st-key-viewer [data-testid="stPopover"] button {{
  background:#132540; color:#E6EDF5; border:1px solid #3B5373; min-height:40px; }}
.st-key-viewer [data-testid="stBaseButton-secondary"]:hover, .st-key-viewer [data-testid="stPopover"] button:hover {{
  border-color:#C9D6E3; color:#fff; background:#1B3254; }}
.st-key-viewer [data-testid="stImage"], .st-key-viewer [data-testid="stImageContainer"] {{ background:#000; border-radius:6px; }}
.st-key-viewer img {{ border-radius:4px; max-width:100%; height:auto; }}
.st-key-viewer [data-testid="stSlider"] [role="group"] > div > div:first-child {{ filter:brightness(3.4) saturate(0.5); }}
.st-key-viewer [data-testid="stSlider"] [role="group"] > div > div:nth-child(2) {{ background-color:#E6EDF5 !important; }}
.st-key-viewer [data-testid="stSliderThumbValue"] p, .st-key-viewer [data-testid="stSliderTickBar"] p {{ color:#E6EDF5 !important; }}
.ra-vgroup {{ font-size:0.6875rem; letter-spacing:0.08em; text-transform:uppercase; color:#93A6BC; font-weight:700; }}
.ra-vinfo {{ display:flex; flex-wrap:wrap; justify-content:space-between; gap:var(--s1) var(--s3); font-size:0.8438rem;
  color:var(--ra-vink); font-variant-numeric:tabular-nums; }}
.ra-vinfo b {{ color:#fff; }}
.ra-vwarn {{ color:#FFD9A0; }}
.ra-legend {{ display:flex; flex-wrap:wrap; gap:var(--s1) var(--s4); font-size:0.8438rem; color:var(--ra-vink); }}
.ra-leg {{ display:inline-flex; align-items:center; gap:7px; }}
.ra-leg i {{ display:inline-block; width:20px; height:10px; border-radius:2px; background:#2B3A4A; }}
.ra-leg-line {{ box-shadow:inset 0 0 0 2px var(--c); }}
.ra-leg-fill {{ background:var(--c) !important; }}
.ra-leg-dotted {{ background:repeating-linear-gradient(90deg, var(--c) 0 4px, #2B3A4A 4px 7px) !important; }}
.ra-leg-grad {{ width:56px !important; background:linear-gradient(90deg, rgba(199,125,255,0.05), var(--c)) !important; }}
.ra-leg-s {{ color:#93A6BC; }}
.st-key-slice_prev [data-testid="stMarkdownContainer"], .st-key-slice_next [data-testid="stMarkdownContainer"] {{
  position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }}
.st-key-slice_prev button, .st-key-slice_next button {{ padding:0 6px !important; }}

/* ---------- side panel: scrolls on its own; the decision stays visible at its foot */
[data-testid="stLayoutWrapper"]:has(> .st-key-sidepanel) {{ flex:1 1 auto; align-self:stretch; }}
.st-key-sidepanel {{ position:sticky; top:4.25rem; max-height:calc(100vh - 14rem); min-height:22rem; overflow-y:auto; overflow-x:hidden;
  border:1px solid var(--ra-border); border-radius:var(--r-panel); padding:var(--s3) var(--s4) 0; gap:var(--s3) !important; }}
[data-testid="stLayoutWrapper"]:has(> .st-key-decision) {{ position:sticky; bottom:0; z-index:3; margin:auto calc(-1 * var(--s4)) 0;
  width:calc(100% + 2 * var(--s4)) !important; max-width:none !important; background:#fff; border-top:2px solid var(--ra-navy);
  box-shadow:0 -6px 14px rgba(16,45,82,0.08); }}
.st-key-decision {{ background:#fff; padding:var(--s3) var(--s4); }}
.st-key-impression {{ gap:var(--s3) !important; }}
.ra-panel-div {{ border-top:1px solid var(--ra-border); padding-top:var(--s3); margin-top:var(--s2); }}
.ra-panel-h {{ display:flex; justify-content:space-between; align-items:center; gap:var(--s2); flex-wrap:wrap; }}
.ra-panel-t {{ font-size:1rem; font-weight:700; color:var(--ra-navy); display:flex; align-items:center; gap:var(--s2); }}
.ra-pn {{ font-size:0.75rem; font-weight:700; color:var(--ra-slate); letter-spacing:0.04em; }}
.ra-panel-s {{ font-size:0.875rem; color:var(--ra-slate); }}
.ra-cmp {{ display:grid; grid-template-columns:1fr 1fr; gap:var(--s3); }}
.ra-k {{ font-size:0.8125rem; color:var(--ra-slate); }}
.ra-v {{ font-size:0.9688rem; font-weight:600; color:var(--ra-ink); }}
.ra-rel {{ font-size:0.875rem; color:var(--ra-slate); margin-top:var(--s1); }}
.ra-rel-diff {{ color:var(--ra-navy); font-weight:600; }}
.ra-ev-find {{ font-size:1.0625rem; font-weight:700; color:var(--ra-navy); display:flex; gap:6px; align-items:center; }}
.ra-ev-p {{ display:flex; align-items:baseline; gap:var(--s3); margin-top:var(--s1); }}
.ra-ev-num {{ font-size:1.75rem; font-weight:700; color:var(--ra-navy); font-variant-numeric:tabular-nums; line-height:1.1; }}
.ra-ev-target {{ font-size:0.8438rem; color:var(--ra-slate); }}
.ra-meter {{ position:relative; height:6px; border-radius:999px; background:#E4EAF0; margin:var(--s2) 0; }}
.ra-meter-fill {{ height:100%; border-radius:999px; background:var(--ra-navy-2); }}
.ra-meter-thr {{ position:absolute; top:-4px; width:2px; height:14px; background:var(--ra-ink); }}
dl.ra-dl {{ margin:0; display:grid; gap:var(--s2); }}
dl.ra-dl > div {{ display:grid; grid-template-columns:118px 1fr; gap:var(--s2); }}
dl.ra-dl dt {{ font-size:0.8438rem; color:var(--ra-slate); }}
dl.ra-dl dd {{ margin:0; font-size:0.9375rem; color:var(--ra-ink); }}
.ra-dd-caution {{ color:var(--ra-amber-ink) !important; font-weight:600; }}
.ra-dd-caution .ra-ico {{ margin-right:4px; }}
.ra-interp {{ display:block; font-weight:400; color:var(--ra-slate); font-size:0.8438rem; margin-top:2px; }}
.ra-ev-withheld .ra-ev-find {{ color:var(--ra-amber-ink); }}
dl.ra-measures {{ display:flex; flex-wrap:wrap; gap:var(--s2) 0; margin:0; }}
dl.ra-measures > div {{ padding:0 var(--s4) 0 0; margin-right:var(--s4); border-right:1px solid var(--ra-border); }}
dl.ra-measures > div:last-child {{ border-right:none; }}
dl.ra-measures dt {{ font-size:0.8125rem; color:var(--ra-slate); }}
dl.ra-measures dd {{ margin:0; font-size:1rem; font-weight:700; color:var(--ra-navy); font-variant-numeric:tabular-nums; }}
dl.ra-measures dd span {{ font-size:0.8438rem; font-weight:500; color:var(--ra-slate); margin-left:3px; }}
.ra-save {{ display:inline-flex; align-items:center; gap:5px; font-size:0.8438rem; font-weight:600; }}
.ra-save-none {{ color:var(--ra-slate); }} .ra-save-unsaved {{ color:var(--ra-amber-ink); }}
.ra-save-saved {{ color:var(--ra-navy); }} .ra-save-failed {{ color:var(--ra-red); }}
.ra-saved {{ font-size:0.9063rem; color:var(--ra-ink); line-height:1.5; }}
.ra-saved b {{ color:var(--ra-navy); }}

/* ---------- notes */
.ra-note {{ display:flex; gap:var(--s2); align-items:flex-start; border-radius:var(--r-ctl); padding:var(--s2) var(--s3); font-size:0.9063rem; }}
.ra-note .ra-ico {{ margin-top:3px; }}
.ra-note-info {{ background:#EEF3F9; color:#1B3A5E; }}
.ra-note-neutral {{ background:var(--ra-slate-bg); color:var(--ra-ink); }}
.ra-note-warn, .ra-note-technical {{ background:var(--ra-amber-bg); color:var(--ra-amber-ink); }}
.ra-note-critical {{ background:var(--ra-red-bg); color:#7A1A12; }}

/* ---------- report and research pages */
.st-key-report_doc pre {{ background:#FFFFFF !important; border:1px solid var(--ra-border); border-radius:var(--r-ctl);
  white-space:pre-wrap !important; word-break:break-word; }}
.st-key-report_doc code {{ font-size:0.875rem !important; color:var(--ra-ink) !important; line-height:1.55; }}
ol.ra-flow {{ list-style:none; display:flex !important; flex-direction:row; flex-wrap:wrap; gap:var(--s3) var(--s5);
  padding:0 !important; margin:var(--s2) 0 var(--s4) !important; }}
.ra-flow li {{ position:relative; margin:0 !important; padding:var(--s2) var(--s3) !important; border-radius:var(--r-ctl); min-width:130px;
  border:1px solid var(--ra-border-strong); background:#fff; display:flex; flex-direction:column; gap:2px; }}
.ra-flow li b {{ color:var(--ra-navy); font-size:0.9375rem; }}
.ra-flow li span {{ color:var(--ra-slate); font-size:0.8125rem; }}
.ra-flow li:not(:last-child)::after {{ content:"→"; position:absolute; right:-19px; top:50%; transform:translateY(-50%);
  color:var(--ra-slate); font-weight:700; }}
.ra-flow-ai {{ border-color:var(--ra-navy) !important; }}
.ra-flow-human {{ border:2px solid var(--ra-navy) !important; }}

/* ---------- mobile: a separate component, not a compressed table */
.ra-mcard {{ display:grid; gap:6px; }}
.ra-mc-k {{ font-size:0.75rem; letter-spacing:0.08em; text-transform:uppercase; color:var(--ra-slate); font-weight:600; }}
.ra-mc-top {{ display:flex; justify-content:space-between; align-items:baseline; gap:var(--s2); }}
.ra-mc-top .ra-case {{ font-size:1.25rem; }}
.ra-mcard .ra-p {{ font-size:0.875rem; }}
.ra-mc-hint {{ font-size:0.875rem; color:var(--ra-ink); }}
[class*="st-key-mcard_"] {{ border:1px solid var(--ra-border); border-radius:var(--r-panel); padding:var(--s4); gap:var(--s3) !important; }}
[class*="st-key-mcard_"] [data-testid="stBaseButton-secondary"] {{ min-height:44px; }}
[class*="st-key-mcard_"] [data-testid="stMarkdownContainer"] {{ margin-bottom:0 !important; }}
[class*="st-key-go_"] button {{ justify-content:flex-start; }}
[class*="st-key-go_"] button > div {{ justify-content:flex-start; width:100%; }}

@media (max-width: 900px) {{
  .ra-device-note {{ display:block; }}
  .st-key-sidepanel {{ position:static; max-height:none; min-height:0; overflow:visible; }}
  [data-testid="stLayoutWrapper"]:has(> .st-key-decision) {{ position:static; box-shadow:none; }}
  .st-key-decision {{ box-shadow:none; }}
  .st-key-casebar .st-key-rx_casenav [data-testid="stHorizontalBlock"] {{ flex-wrap:nowrap !important; }}
  .st-key-casebar .st-key-rx_casenav [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(odd) {{
    flex:0 0 64px !important; min-width:64px !important; }}
  .st-key-casebar .st-key-rx_casenav [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) {{
    flex:1 1 0 !important; min-width:0 !important; }}
  .st-key-prev_case [data-testid="stMarkdownContainer"], .st-key-next_case [data-testid="stMarkdownContainer"] {{
    position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }}
}}
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"], .block-container {{ padding:3rem var(--s3) var(--s5); }}
  .st-key-desklist, .st-key-rowhead {{ display:none !important; }}
  [data-testid="stColumn"]:has(.st-key-tb_search), [data-testid="stColumn"]:has(.st-key-tb_order),
  [data-testid="stColumn"]:has(.st-key-tb_details) {{ display:none !important; }}
  .st-key-mobilelist {{ display:flex !important; }}
  .st-key-topbar .st-key-rx_topbar [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) {{
    flex:1 1 100% !important; }}
  .st-key-topbar .st-key-rx_topbar [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(n+3) {{
    flex:1 1 calc(50% - 8px) !important; min-width:0 !important; }}
  .st-key-nav, .st-key-nav [data-testid="stButtonGroup"], .st-key-nav [role="radiogroup"] {{ width:100% !important; display:flex; }}
  .st-key-nav button {{ flex:1 1 0; }}
  .ra-summary {{ gap:var(--s1) var(--s3); font-size:0.875rem; }}
  ol.ra-flow {{ flex-direction:column; gap:var(--s4); }}
  .ra-flow li:not(:last-child)::after {{ content:"↓"; right:auto; left:20px; top:auto; bottom:-18px; transform:none; }}
}}
"""

# rows created with streamlit_app.cols(name, ...) reflow by the space they have: min px per column (+ grow)
REFLOW = {"decide": [150], "after": [150], "reportbar": [240], "downloads": [220], "charts": [360], "figs": [380],
          "about": [320], "onboard": [320], "rerunrow": [260], "policies": [260], "sim": [170], "authrow": [240],
          "toolbar": [240, 180, 170, 120], "viewtools": [104, 116, 176, 92], "slicenav": [44, 200, 44],
          "topbar": [240, 220, 136, 136], "casebar": [340, 470], "casenav": [140, 160, 140], "savebar": [150, 130],
          "pendingbar": [190], "read62": [540, 340], "read75": [680, 290]}
REFLOW_GROW = {"toolbar": [2, 1.4, 1, 0.6], "viewtools": [1, 1, 1.2, 0.6], "slicenav": [0, 1, 0],
               "topbar": [2.4, 1.2, 0.25, 0.25], "casebar": [2.2, 1.3], "casenav": [0, 1, 0], "savebar": [1.3, 1],
               "read62": [1.65, 1.0], "read75": [3.4, 1.0]}


def reflow_css() -> str:
    out = []
    for name, bases in REFLOW.items():
        sel = f'[class*="st-key-rx_{name}"]' if name == "figs" else f".st-key-rx_{name}"
        hbs = [f'{sel} > [data-testid="stHorizontalBlock"]', f'{sel} > * > [data-testid="stHorizontalBlock"]']
        out.append(", ".join(hbs) + " { flex-wrap:wrap !important; }")
        grows = REFLOW_GROW.get(name, [])
        for i, px in enumerate(bases, start=1):
            nth = f":nth-child({i})" if len(bases) > 1 else ""
            g = grows[i - 1] if i - 1 < len(grows) else 1
            col = ", ".join(f'{h} > [data-testid="stColumn"]{nth}' for h in hbs)
            out.append(f"{col} {{ flex:{g} 1 {px}px !important; min-width:min({px}px, 100%) !important; "
                       f"width:auto !important; }}")
    return "\n".join(out)


def style_tag(extra: str = "") -> str:
    return f"<style>{CSS}\n{reflow_css()}\n{extra}</style>"
