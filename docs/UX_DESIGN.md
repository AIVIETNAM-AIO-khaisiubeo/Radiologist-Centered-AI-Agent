# UX Design — Radiologist Reading Interface

> **Research prototype — not for clinical use.** Model outputs, measurements, probabilities, priorities, and report drafts are not validated for patient-care decisions. The radiologist remains responsible for interpreting the CT.

## Users and context

The primary users are radiologists and residents reviewing cases on a workstation, with a research or quality lead overseeing evaluation and reader-policy authorisation. The design accounts for interruptions, fatigue, shared workstations, limited screen space, and hospital networks that may block external assets.

The AI provides evidence; it does not replace the reader's judgement. The default workflow protects the first impression from AI influence.

## Navigation and session rules

A compact top bar provides two primary modes—**Worklist** and **Read case**—and two secondary menus:

- **Session:** reader, reading mode, decision policy, worklist selection, session/model/policy identifiers, and starting a new session.
- **Research:** reader adaptation, model performance, audit trail, reference annotations, and intended use.

Session settings lock after the first saved impression or decision. A policy proposal does not become active merely because it was generated: an authorised person must record their identity, role, reason, and confirmation. An authorised policy takes effect in a new session. If no personalised policy has been authorised, the universal policy remains active.

## Reading workflow

| Stage | Interface behaviour |
|---|---|
| **1. Worklist** | In the default **Blind first read** mode, cases appear in arrival order. Unread cases do not show AI predictions, probabilities, AI priority, disagreement, or AI-derived ordering. Technical image or segmentation QC status remains visible, with an explanation of what review is still possible. |
| **2. Independent impression** | The CT viewer opens with AI overlays hidden and locked. The reader records **Tumour present** or **No liver tumour** before revealing AI evidence. The impression is persisted immediately and the original is never overwritten. |
| **3. Evidence review** | The interface presents the reader's impression alongside the AI finding, then the calibrated patient-level probability and its target, uncertainty on a stated scale, QC/validity status, and supporting measurements or lesion details. Optional second-reader evidence and model agreement appear only when that reader is configured. A clear contour is not proof of certainty. |
| **4. Decision** | The reader records a final finding, optionally edits notes or lesions, or defers with a reason. Deferral records no final finding. The decision area stays available while the evidence panel scrolls and exposes explicit save states: **Not saved**, **Unsaved changes**, **Saved**, or **Save failed**. |
| **5. Confirmation and recovery** | The saved summary identifies the case and time, the original impression and final finding, and the relation to the AI suggestion. It states that the research decision is saved and the report draft is unsigned. Revisions create a new record and retain the prior decision. Navigating away from unsaved work requires **Save and continue**, **Discard and continue**, or **Stay on this case**. Reopening a case restores the saved impression. |

The interface keeps the worklist and case-reading view separate. Reports and audit details remain unavailable until the independent impression is saved in blind-first-read mode.

## Visual and interaction principles

### Status semantics

Technical validity, AI urgency, workflow state, and AI evidence characteristics are separate concepts. One must not be inferred from another.

| Visual treatment | Meaning |
|---|---|
| **Navy** | Structure, active navigation, selection, and focus. |
| **Teal** | Primary actions; never a signal that the AI is correct. |
| **Red** | The configured AI urgent rule only. The label says “Urgent by AI rule” and is not a clinically validated urgency decision. |
| **Amber** | Caution, QC problems, high uncertainty, deferred cases, or other review-required states. |
| **Slate** | Neutral and technical information, including metadata. |

Green is avoided next to AI outputs because it can be mistaken for clinical approval. Use text and icons as well as colour so that meaning is not conveyed by colour alone.

### CT viewer

The viewer uses a dark image canvas and grouped controls for **Image**, **Overlays**, **Original CT**, and **Reset**. It shows the active window/level preset and its values, the current view settings, and which overlays are enabled. The default image layout gives roughly 60% of the reading area to the CT; **Large image** expands it to roughly 70%.

- **Liver contour:** teal.
- **Lesion contour:** amber; fill is off by default to preserve anatomical visibility.
- **Uncertainty overlay:** violet, with a low-to-high legend.

The viewer states explicitly when overlays are locked, QC has failed, or no AI lesion is present on the current slice. Window/level and layout apply to the session; zoom and pan reset for each case. Presets are research defaults, not clinically validated diagnostic settings.

### Hierarchy, controls, and accessibility

The product identity and research-prototype label remain visible but low-emphasis. Case identity, reading progress, evidence, decision controls, and save state take priority over metadata. Each context has one primary action, such as **Open next unread case**, **Save impression**, **Save decision**, or **Next unread case**.

The interface uses a compact type scale: 15–16 px body text, 14–15 px table labels, 13–14 px metadata, and a larger case heading. Controls are designed for clear targets and visible keyboard focus. Keyboard shortcuts are **J/K** for slices, **N/P** for cases, and **O** for the original CT; they are inactive while typing and do not activate decision controls.

Below 640 px, the worklist becomes a separate card layout with case identity, workflow status, QC state, AI visibility state, and one **Open case** action. Search, sorting, and secondary details are hidden on narrow screens. The case view warns that a small screen is not intended for diagnostic interpretation.

## Failure handling and auditability

Failures must be explicit and must never resemble a valid negative result.

- **Image-loading or image-QC failure:** explain the issue and available next steps.
- **Segmentation-QC failure:** withhold AI masks and state that this is a technical limitation, not evidence that a tumour is absent.
- **Missing optional second-reader result:** identify it as missing, not as a negative result or agreement.
- **High uncertainty or model disagreement:** label the evidence characteristic and explain its meaning; do not equate it with technical failure or urgency.
- **Save failure or unsaved work:** show a visible state and prevent silent loss during navigation.

The audit record keeps the original impression separate from the final decision and records revisions without erasing the earlier decision. Each decision stores the session identity, model-set and policy versions, the recommendation shown at the time, and relevant timestamps. This makes it possible to reconstruct what the reader saw and did.

## Usability evaluation

Formative usability sessions should use representative radiologists, realistic CT cases, and the intended workstation displays. Observe task completion, errors, hesitation, unnecessary navigation, and whether users correctly interpret workflow, QC, uncertainty, and urgency labels.

Key scenarios include a routine case, QC failure, high uncertainty, interruption and resumption, navigation with unsaved work, and display-size variation. Automated UI tests cover the reading workflow, persistence, policy authorisation, and failure states; responsive layouts also require visual checks at supported widths. These checks do not replace usability or clinical validation.

## Intended-use boundary

This interface is a research instrument, not a clinically validated medical device. Clinical use would require a defined intended use and regulatory pathway, representative formative and summative usability studies, validation on diagnostic displays, independent model evaluation, and controlled governance of adaptive policies. The current design and its tests do not establish clinical safety, diagnostic performance, or readiness for patient care.
