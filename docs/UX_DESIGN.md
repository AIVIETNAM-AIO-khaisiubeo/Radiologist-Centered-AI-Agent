# Reading interface - UX design notes

The website is the radiologist's side of ReaderAdapt-Liver. Since version 0.3 the AI is one pretrained model,
NVIDIA VISTA3D; the second-reader evidence described in the round-2 and round-3 sections appears only when a
research second reader is plugged in. These notes record who it is for, the
reading journey it supports, the design rules derived from it, and how the two external design reviews
(round 2 and round 3) were addressed. They are the reference for future UI changes.

The standard used for every decision, quoted from the final review: the interface is one *where the
radiologist can understand what is happening, recognize what is unreliable, make an independent judgment,
and recover safely when something goes wrong*.

> Research prototype. Nothing here claims clinical readiness; section 8 lists what would have to be
> shown before any clinical use.

## 1. Context of use

| | |
|---|---|
| **Users** | General radiologists and residents in a regional hospital; a QA / research lead who reviews performance and authorises adaptation. |
| **Devices** | Reading-room workstation (large screen, dimmed room) for interpretation; office desktop; tablet or phone only for checking the worklist and saved decisions. |
| **Environment** | Interruptions, end-of-shift fatigue, shared workstations, hospital networks that may block external fonts and CDNs (the bundled Source Sans font is used). |
| **Stakes** | The AI suggests; the radiologist decides. Anchoring on AI output, over-trust and under-trust are use errors with patient impact. |

## 2. Information architecture

There is **no persistent sidebar**. One compact top bar holds everything that is not the case itself:

| Area | Contents | When it is used |
|---|---|---|
| Identity | *ReaderAdapt · Liver CT*, *Research prototype · not for clinical use*, and one line of state: *Reader: You · Blind first read · Session S-1026-1432…* (a lock icon once the session is locked) | always visible, low emphasis |
| Modes | **Worklist** / **Read case** | the two working modes |
| **Session** menu | reader, reading mode, decision policy, worklist, run; session details (full session, model-set and policy ids, copyable); policy source (authorised by whom, or *universal policy in use*); **Start a new session** | at the start of a session; locked afterwards |
| **Research** menu | Reader adaptation, Model performance, Audit trail, About and intended use; *Show the reference annotation*; deleting logged decisions (explicit confirmation) | governance, not reading |

Session configuration (who reads, how, with which policy, on which cases) **locks after the first saved
impression or decision** so that the reading conditions cannot change mid-session; only a new session
unlocks it.

## 3. The reading journey

| # | Stage | Interface |
|---|---|---|
| 0 | **Arrive** | Top bar; the Session menu if anything must change before the first read. |
| 1 | **Plan** | Worklist. In *blind first read* (default) the table has **no AI column**: *# · Case · Image / AI QC · Reading status · Open*, in arrival order; nothing AI-derived is shown for unread cases (no prediction, probability, disagreement, AI priority or numeric QC details). A QC failure is a technical status with what can still be done (*images can be read manually* / *check the series*). One primary action: **Open next unread case**. |
| 2 | **Independent read** | The viewer opens with every AI overlay locked and on the middle slice of the slab, not the AI's key slice. *Tumour present* / *No liver tumour*, then **Save impression and reveal AI evidence**. The impression is written to disk at once and is never replaced. |
| 3 | **Evidence and limitations** | *01* your impression next to the AI suggestion as information (*same / different binary finding*), then *02* in reading order: AI finding → calibrated patient-level probability with its target, calibration set and threshold → uncertainty (stated scale) and validity (QC) → supporting evidence on demand (measurements and lesions, uncertainty and calibration, model and versions, including the segmenter's training-data caveat). With an optional research second reader, a *model agreement* row and its interpretation follow validity. |
| 4 | **Decide** | *03 Decision* is a footer that stays visible at the foot of the independently scrolling side panel: final finding, **Save decision**, **Defer…** (reason; records no final finding) and a save state: *Not saved · Unsaved changes · Saved hh:mm · Save failed*. Note and lesion edits are optional and sit above the footer. |
| 5 | **Confirm** | The saved summary states the case, time, impression → final finding, the relation to the AI suggestion and *research decision saved · report draft not signed*. Next: **Next unread case** (primary), **Open report**, **Revise decision…** (a new decision that names the one it revises; the original stays in the log). |
| 6 | **Recover** | Leaving a case with unsaved work asks *Save and continue / Discard and continue / Stay on this case*. Reopening a case after an interruption resumes with the saved impression (*Resumed. Your impression was saved at …*). |
| 7 | **Govern** | Research menu: the active policy next to the proposed one; a proposal is used only after an attributed **authorisation** (name and role, reason, confirmation), versioned and logged, from the next session. Audit trail: impression, decisions, revisions, the recommendation on screen, versions. |

## 4. Design rules

**Four status concepts, never inferred from each other**

| Concept | Example | Treatment |
|---|---|---|
| Technical validity | *Segmentation QC failed* | amber, square icon, plus what can still be done |
| Clinical urgency (AI rule) | *Urgent by AI rule* (probability ≥ 85 % and a lesion ≥ 5 mL or ≥ 20 mm) | red triangle - the only use of red; AI-assisted triage only |
| Workflow status | *Unread · Impression saved · Decided · Deferred* | neutral text with a small icon |
| AI evidence characteristic | high uncertainty (and, only with a research second reader, *AI models disagree*) | amber text, labelled |

**Colour has one meaning each** (contrast checked in `tests/test_app.py`):

| Token | Value | Meaning | Contrast |
|---|---|---|---|
| Navy | `#102D52` / `#163A66` | structure, active navigation, selection, toggles and sliders | 13.8 : 1 on white |
| Teal | `#0F766E` | primary actions only - never "the AI is right" | 5.5 : 1 with white text |
| Red | `#B42318` | the validated urgent rule only | 6.1 : 1 on its tint |
| Amber | `#B54708` / `#7A2E0E` | caution: QC issue, high uncertainty, model disagreement, deferred, unsaved | 5.1 : 1 on its tint |
| Slate | `#526579` | neutral and technical information, metadata | 6.0 : 1 on white |

No green is used: a positive colour next to an AI output would read as clinical approval. Emphasis uses one
technique per element (icon + coloured text); there are no filled badges.

**Type scale**: product name 20 px, page heading 22 px, case identifier 18 px (20 px on the mobile card), body
15-16 px, table and labels 14-15 px, metadata 13-14 px; the AI probability is the one large number (28 px).
**Spacing**: 4 / 8 / 12 / 16 / 24 / 32 px. **Radius**: 8 px controls, 12 px panels. **Controls**: one height
(44 px; the two clinical answers 48 px).

**Selection model** (worklist): normal rows neutral; hover a faint tint; the current case a subtle navy tint
with a 3 px accent and the text *Current case*; keyboard focus a separate 3 px navy outline.

**One primary action per context**: *Open next unread case* (worklist), *Save impression…* (independent
read), *Save decision* (decision), *Next unread case* (after saving), *Save and continue* (unsaved-work prompt).

**CT viewer**: dark canvas, grouped toolbar (**Image** · **Overlays** · **Original CT (O)** · **Reset**), a status
line with the active preset and its values (*Liver · W 150 / L 70 HU*), zoom and the overlays in use, and explicit
mask states (*locked until you save your impression*, *QC failed - AI masks not shown*, *no AI lesion on this
slice*). Overlays: teal liver contour, amber lesion contour (fill off by default so the anatomy stays visible),
violet uncertainty with a low → high legend; a caption says that a crisp contour is not certainty. Window/level and
layout apply to the whole session; zoom and pan reset per case. Presets are research defaults, not clinically
validated settings.

**Keyboard**: **J / K** slice, **N / P** case, **O** original CT - never on decision buttons, inactive while typing;
logical focus order and visible focus.

**Responsive**: rows reflow by the space they have. Below 640 px the worklist is a separate component - case cards
(*Case 01*, case id with status, QC, AI visibility, an optional hint, one *Open case* action) - and search, order
and details are hidden. The case view shows *small screen - not for diagnostic interpretation*.

**Failures are explicit**: images that cannot be loaded, failed QC, a missing second-reader result when one is
configured (*a missing result is not a negative result*) and a failed save each have a stated state; none looks like a valid negative.

## 5. Design review (round 2) - what changed

| Review point | Response |
|---|---|
| **P0** AI cues leaked into the "independent impression first" worklist | *Blind first read* mode: no AI output for unread cases, arrival order, report and audit trail closed until the impression is saved; AI evidence shown next to the impression. |
| **P0** Adaptation must not change behaviour silently | Policy frozen per session, versioned and logged (now replaced by explicit authorisation, round 3). |
| **P0** Audit what was shown | Each decision stores session, policy and model-set versions, the recommendation on screen and the impression time. |
| **P0** Ambiguous terms | *Urgent* defined by its rule; class vs score agreement; uncertainty scale; QC failure kinds kept apart from urgency; probability names its target. |
| **P1** Reading view crowded / worklist like a dashboard | Two modes, CT-first view, dense table, one-line summary. |
| **P2** Visual system, mobile | Semantic palette, restrained type, compact cards, supported-device policy. |

## 6. Design review (round 3) - what changed

| Review point | Response |
|---|---|
| **Top 1** Mobile worklist is a compressed desktop table | A separate card component (no narrow label column): overline *Case 01*, case id with status on one line, QC, AI visibility (*AI findings hidden* / *AI output withheld*), a hint for technical failures, one full-width *Open case* action; compact summary and *All / Unread / Reviewed* filter; search, order and details hidden on phones. |
| **Top 2** Too much chrome; the CT viewer must be the primary workspace | Sidebar removed (Session and Research menus in the top bar); one compact case bar with progress; viewer ≈ 60 % of the width, *Large image* ≈ 70 %; grouped viewer toolbar. |
| **Top 3** Impression, evidence, decision and saved state must be impossible to confuse | Three numbered sections; the comparison is information, not a control; the decision footer stays visible with explicit save states; the saved summary names the case, time, impression → final and says *research decision saved · report draft not signed*. |
| 2.1 Sidebar mixes configuration, metadata and navigation | Session configuration and session details in *Session*; research tools in *Research*; configuration locks after the first save. |
| 2.2 Header competes for attention | Product name and research label, then one low-emphasis state line; long ids only in the Session menu, copyable. |
| 2.3 Row structure / reserved AI column of lock messages | Blind schema without an AI column; assisted schema adds *AI finding*; QC failures state what manual review is possible; segmentation failure is explicitly *not* an acquisition failure. |
| 2.4 Selection model | Neutral rows, distinct hover, current case tint + 3 px accent + *Current case*, separate focus outline. |
| 3.1 Proportions, resizable split | 60 / 40 by default and a *Large image* layout (Streamlit has no draggable split pane; the layout choice is the tested substitute). |
| 3.2 Stacked bands; unsaved work on navigation | One case bar; unsaved-work prompt on every navigation path (buttons, shortcuts, case picker, mode switch, menus). |
| 3.3 Evidence hierarchy; agreement shown as a button | Reading order *finding → probability and target → uncertainty and validity → model agreement and interpretation → supporting evidence*; comparison as plain text; no redundant confirmation. |
| 3.4 Decision below the fold | Sticky decision footer inside an independently scrolling side panel; validation message; save failure state. |
| 4.1 Toolbar grouping | *Image* (W/L, zoom, pan, layout) · *Overlays* (visibility, opacity, explanation) · *Original CT* · *Reset*; slice navigation with *n / N* and J / K. |
| 4.2 W/L semantics | Presets show their values; the active setting is always on screen; scope and reset behaviour stated. |
| 4.3 Overlays | Contour vs fill distinguished (fill off by default), opacity, independent visibility, original CT without overlays (O), mask-unavailable states, uncertainty legend low → high, a caption against false certainty. |
| 6 Colour | Toggles and sliders navy (theme primary), teal for primary actions only, amber for every caution with explicit text, red only for the urgent rule (and only when non-zero). |
| 7 Type and spacing | Tokens in section 4. |
| 8.1 Save state and recovery | *Not saved / Unsaved changes / Saved hh:mm / Save failed*; impressions persisted at once; resume after interruption; revisions never overwrite. |
| 8.3 Session identity | Reader, session, model set and policy ids stored with every decision, shown in the Session menu. |
| 8.4 Failure handling | Explicit states for image loading, QC, missing second reader and failed saves (About lists them). |
| 8.5 Audit and versions | Impression and final decision stored separately; revisions record what they supersede; audit trail lists both. |
| 8 Adaptation needs authorisation, not only logging | A policy is used only after an attributed authorisation (name and role, reason, confirmation), versioned in `feedback/authorised_policies.json`, logged in `policy_changes.jsonl`, effective from the next session; until then the universal policy is used. Simulation never changes it. |

## 7. Validation scenarios (formative usability study)

Proposed for a small study with representative radiologists, realistic CT cases and the intended displays
(review §11). Observe completion, errors, hesitation, unnecessary navigation and whether status labels are read
correctly.

| # | Scenario | What the interface provides | Success criterion |
|---|---|---|---|
| 1 | **Routine case**: open, read independently, reveal AI evidence, decide | Open next unread case → 01 → 02 → 03 | Decision saved; the reader can say what was saved and that the report is not signed |
| 2 | **QC failure** | Technical status in the worklist, *What failed / What it means / What you can do* in the case | The reader reads manually or defers; does not treat it as urgent or as a negative result |
| 3 | **High AI uncertainty** (or model disagreement, if a research second reader is enabled) | uncertainty on a stated scale; a crisp contour is not certainty | The reader relies on the images and does not over-trust the suggestion |
| 4 | **Interruption** | Impression saved at once; *Resumed* on reopening | The reader resumes without re-entering or confusing the saved impression |
| 5 | **Navigation error** with unsaved work | *Save and continue / Discard and continue / Stay* | No silent loss; the reader chooses deliberately |
| 6 | **Display variation** | Tested at 1440 / 1024 / 820 / 390 px; small-screen warning | Readable at the target resolution and scaling; diagnostic display behaviour verified separately |

The automated tests in `tests/test_app.py` exercise scenarios 1-5 as behaviour; scenario 6 is checked with
browser screenshots at those widths. Neither replaces the study.

## 8. Before any clinical use

The interface is a research instrument. A clinical evaluation would need, at minimum: a usability-engineering
process in the spirit of IEC 62366-1 and current human-factors guidance for medical devices; formative and
summative studies with representative radiologists on realistic cases and the intended diagnostic displays
(section 7 is a starting point); validation of the viewer and of the window/level presets on diagnostic displays;
offline validation and controlled deployment of any adaptive behaviour before it is authorised; and a defined
intended use and regulatory pathway for the jurisdiction.
