# SentinelMed P6 Frontend --- Antigravity Implementation Specification

> **Purpose:** Build a polished, integration-ready frontend scaffold for
> SentinelMed while the backend interfaces are still being developed.
>
> **Authority:** Follow the existing SentinelMed project guide and
> repository. This document specifies frontend behavior and integration
> boundaries; it does not authorize inventing missing backend
> functionality.
>
> **Current mode:** Frontend scaffold + clearly labelled demo mode. Real
> backend integration is pending confirmation of the actual interfaces.

------------------------------------------------------------------------

## 1. Can frontend work start without teammate inputs?

**Yes.** Build the frontend independently using a mock/demo adapter and
a documented integration boundary. Do not pretend that the real ML
pipeline is connected.

Work that can be completed now:

-   Streamlit page layout and visual design.
-   Image upload, validation and preview.
-   Dashboard cards and result sections.
-   Reusable display components for accepted, auto-fixed and escalated
    outcomes.
-   A typed/validated result contract based on the project guide's
    proposed fields.
-   Clearly labelled demo mode with synthetic example data.
-   Placeholder slots for original/enhanced image comparisons.
-   Frontend setup instructions and basic tests.
-   An adapter module that will later call the actual team pipeline.

Work that must wait for the real interface:

-   Calling actual model functions.
-   Claiming live inference works.
-   Claiming that image enhancement, rechecking or escalation is
    operational.
-   Showing real measured model metrics or benchmark results.
-   Guessing import paths, model file paths, function signatures or
    queue behavior.

## 2. Mandatory instructions for Antigravity

1.  **Inspect before editing.** Inspect the current repository, branch,
    files and dependencies first. Do not assume the repository is empty.
2.  Do not overwrite, move, rename or rewrite teammates' backend/model
    modules.
3.  Put all P6 frontend-owned code under `frontend/`, unless an existing
    repository convention makes this impossible. If it is impossible,
    explain why before making changes.
4.  Reuse the repository's existing environment and dependencies where
    practical. The project guide recommends Streamlit.
5.  Do not invent backend functions, model behavior, result values,
    metrics, file paths or successful integration.
6.  Keep demo data separate from live integration. Demo mode must be
    visibly labelled **DEMO DATA --- NOT LIVE INFERENCE**.
7.  If the actual pipeline interface is unknown, leave an explicit TODO
    and a small adapter boundary. Do not guess.
8.  Never silently fall back to demo data when live inference fails.
    Show a clear error and tell the user how to switch to demo mode
    intentionally.
9.  Do not label model output as a confirmed medical diagnosis. Display
    a persistent research-prototype disclaimer.
10. After implementation, report files changed, commands run, tests
    performed, what passed, and what remains unintegrated. Never claim a
    test or feature succeeded unless it was actually checked.

------------------------------------------------------------------------

## 3. Product and workflow

SentinelMed is a medical-image AI reliability **research prototype**.
The intended workflow is:

1.  A user uploads a scan image.
2.  The backend inspects image quality.
3.  The backend runs prediction and uncertainty estimation.
4.  A conformal gate decides whether the prediction can be accepted.
5.  If flagged, the backend diagnoses the cause and calculates a
    Recoverability Score where supported.
6.  If the scan is recoverable, the backend enhances it and re-runs
    quality inspection, prediction/uncertainty and the conformal gate.
7.  Accept the post-enhancement result only if the recheck passes;
    otherwise escalate for human review.
8.  The frontend displays the outcome, evidence and measured latency
    returned by the backend.

If the existing backend already implements this workflow, call its entry
point rather than duplicating the logic in the frontend.

**Critical:** the frontend must not implement medical ML logic or
independently decide that a scan is safe to accept.

## 4. Visual design system

Create a minimalist, attractive, professional interface appropriate for
a technical healthcare reliability dashboard. Avoid a generic, colourful
template.

### Palette

Use these design tokens consistently:

-   **Warm off-white page background:** `#F7F6F2`
-   **White / near-white panels:** `#FFFFFF`
-   **Primary text, deep navy:** `#18313B`
-   **Secondary text, slate:** `#65747B`
-   **Primary accent, muted teal:** `#147D78`
-   **Teal hover/darker accent:** `#0F625F`
-   **Borders and dividers:** `#E3E6E2`
-   **Accepted status:** `#238568`
-   **Auto-fixed / attention status:** `#C88725`
-   **Escalated / error status:** `#C94C43`
-   **Subtle secondary surface:** `#EFEEE8`

Keep status colours restrained and use text labels/icons as well as
colour so meaning is not conveyed by colour alone. Maintain readable
contrast. Do not use gradients, neon colours, excessive shadows, emojis
as the main icon system, or unnecessary animation.

### Layout and interaction

-   Use a clear page title and concise subtitle.
-   Use a compact sidebar or simple navigation for **Single Scan**,
    **Scan Queue**, and **About / Limitations**, if these pages fit the
    current app structure.
-   Keep generous whitespace, consistent alignment, subtle borders and
    modest corner rounding.
-   Use clear hierarchy: final verdict first, supporting evidence
    second, detailed technical values in expandable sections.
-   Keep the primary action obvious: **Analyse scan**.
-   Disable or guard duplicate submissions while a scan is processing.
-   Provide a spinner/progress state, empty state, validation errors and
    backend failure state.
-   Do not add charts with invented data.
-   Avoid unnecessary animation and excessive dashboard cards.
-   Use Streamlit-native components unless a justified dependency
    already exists.

## 5. Recommended `frontend/` structure

Start with this structure, adapting only when the existing repository
requires it:

``` text
frontend/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── components/
│   ├── __init__.py
│   ├── header.py
│   ├── upload_panel.py
│   ├── result_panel.py
│   ├── image_comparison.py
│   └── metrics_panel.py
├── services/
│   ├── __init__.py
│   ├── pipeline_adapter.py
│   ├── result_schema.py
│   └── demo_adapter.py
└── tests/
    ├── test_result_schema.py
    └── test_demo_adapter.py
```

Do not create empty modules merely to make the tree look complete.
Implement only modules that are actually used. If a smaller structure is
clearer, prefer the smaller structure.

### Responsibilities

-   `app.py`: page configuration, navigation, page state and top-level
    composition only.
-   `components/`: presentation-only UI components. Components must not
    contain ML inference logic.
-   `services/pipeline_adapter.py`: one small boundary for calling the
    real backend later. Do not invent the import or function signature.
    Until confirmed, return a clear "real backend not configured"
    result/error rather than fake live results.
-   `services/result_schema.py`: validate and normalize result data for
    rendering. Do not silently fabricate missing metrics.
-   `services/demo_adapter.py`: deterministic demo examples, explicitly
    marked as synthetic.
-   `tests/`: tests for schema validation and demo data behavior.
-   `.streamlit/config.toml`: configure the supported Streamlit theme
    using the palette above, if compatible with the installed Streamlit
    version.

Keep frontend imports local to `frontend/` where practical. Avoid
circular imports and avoid importing backend/model modules at
module-import time if this would prevent demo mode from starting.

## 6. Single-scan page requirements

Build a usable page with these sections:

### A. Upload panel

-   Accept PNG, JPG and JPEG files, subject to the real backend's
    supported format.
-   Display a preview after successful decoding.
-   Validate that the uploaded file is a readable image.
-   Apply a reasonable configurable size limit and explain rejection
    clearly.
-   Do not assume that a file extension alone guarantees a valid image.
-   Do not persist uploaded images unless the backend genuinely requires
    it.
-   Use the preprocessing and colour/channel format required by the
    actual model once confirmed. Do not invent preprocessing.

### B. Analysis action

-   Require an explicit **Analyse scan** action.
-   Display a clear processing indicator.
-   Prevent accidental duplicate execution caused by Streamlit reruns.
-   Show errors safely and concisely; detailed tracebacks belong in
    logs/developer diagnostics, not the ordinary UI.
-   Distinguish "backend unavailable" from "analysis completed."

### C. Result summary

Render these fields only when supplied and validated:

-   Scan ID, if supplied.
-   Predicted class/label and confidence.
-   Final reliability verdict.
-   Total decision latency.
-   Explanation/reason, if supplied.

Do not convert a numeric class index into a human-readable medical label
unless the label mapping is confirmed by the team.

### D. Quality and uncertainty

Display available fields only, for example:

-   Blur, contrast, noise and brightness.
-   Entropy, MC-dropout variance and model disagreement.
-   Conformal gate outcome.

Do not assume all fields exist. Use clear units and definitions from the
team. Do not invent thresholds or label a metric "good" or "bad" without
a supplied definition.

### E. Conditional recovery details

For an auto-fixed result, show:

-   Original image and the actual enhanced image returned by the
    backend.
-   Diagnosed cause.
-   Recoverability Score, if returned.
-   Recheck outcome and final verdict.

Never label the original image twice as "before" and "after." If the
enhanced image is absent, state that it was not supplied.

For an escalated result, show the supplied cause/reason and a clear
**Human review required** status. Do not claim a human has reviewed it
unless that status is explicitly returned.

### F. Latency

Show total and per-stage latency only when measured values are supplied.
The workflow diagram mentions a target under one second, **to be
measured**. Treat this as a target, not a verified result. Do not
generate sample latency figures and present them as measured.

## 7. Proposed result contract: guidance, not proof of implementation

The project guide proposes a shared result dictionary similar to:

``` python
{
    "pred": None,
    "confidence": None,
    "quality": {},
    "uncertainty": {},
    "gate": None,
    "cause": None,
    "recoverability": None,
    "verdict": None,
    "latency_ms": {},
    "explanation": None,
}
```

This is a **schema sketch only**, not a claim that the backend already
returns these values or supports these exact types. Inspect the
repository and confirm the actual schema with the team before connecting
it.

The guide names these expected verdicts:

-   `accepted`
-   `auto_fixed`
-   `escalated`

The guide mentions example cause labels including `quality`, `ood`,
`ambiguous` and `disagree`. Treat these as proposed values until
confirmed against the code.

The schema validator should: - Validate required fields and allowed
verdicts based on the confirmed contract. - Treat optional values as
optional. - Reject or report malformed output with a helpful message. -
Never fill missing confidence, cause, score, latency or images with
made-up values. - Keep unknown extra fields available in a collapsible
technical-details section when safe to display.

## 8. Demo mode

The frontend must be demonstrable before the actual backend is ready.

-   Add an explicit mode selector or configuration setting for **Demo
    mode** versus **Live backend**.
-   Demo mode may use a small, deterministic set of synthetic result
    dictionaries.
-   Mark every demo result prominently as **DEMO DATA --- NOT LIVE
    INFERENCE**.
-   Do not use fabricated results in screenshots/slides as if they were
    measured results.
-   Do not let live mode silently fall back to demo mode.
-   If no actual scan images are available, use a neutral placeholder
    state instead of fabricating medical imagery or implying that a
    synthetic result is real.
-   Keep the demo adapter separate so it can be removed or disabled
    without changing the result components.

## 9. Scan queue and replay page

Prepare a simple frontend structure for a queue, but do not claim that a
live queue works until the P5 interface is available.

When queue integration is not yet defined: - Show a clear
integration-pending state in Live Queue. - Optionally allow a small,
explicitly labelled demo queue using the demo adapter. - Do not invent a
queue-generator import, API endpoint, generator signature or file path.

Once the P5 contract is confirmed, the queue should: - Process scans one
at a time according to the existing queue implementation. - Show scan
ID, verdict, cause, recoverability when available, and actual latency. -
Avoid duplicate processing due to reruns. - Handle empty queue,
end-of-queue and per-scan failures. - Support replay from real saved
results if the team supplies them. - Support the project's unattended
three-minute demo requirement. - Clearly distinguish replayed results
from live inference.

## 10. Easy future backend integration

The UI must depend on one adapter boundary, not on model modules
scattered throughout presentation code.

Desired conceptual flow:

``` text
Streamlit UI
    |
    v
pipeline_adapter
    |
    v
Existing SentinelMed pipeline / P5 queue interface
    |
    v
Validated result dictionary
    |
    v
Reusable result components
```

When the team supplies the real function signature:

1.  Implement the actual import/call only inside
    `services/pipeline_adapter.py`.
2.  Convert the backend output into the confirmed result contract at
    that boundary if necessary.
3.  Keep all ML logic in the existing backend.
4.  Keep presentation components unaware of model implementation
    details.
5.  Add a small adapter test using a representative backend output.
6.  Test one real image end-to-end before enabling live mode by default.

Do not add FastAPI, Flask, React, Node.js or HTTP endpoints merely to
connect two Python modules in the same application. If the existing
project already uses a separate service, inspect and use its actual
documented interface instead.

## 11. Safety, privacy and truthful UI

-   Display a persistent disclaimer: **Research prototype only --- not
    for clinical diagnosis or treatment decisions.**
-   Do not state that a scan is medically safe, disease-free or
    clinically validated based solely on the reliability verdict.
-   Do not upload scans to external services.
-   Avoid logging image contents or sensitive metadata.
-   Do not persist uploaded images without an explicit project
    requirement.
-   Keep exception messages user-friendly and avoid exposing secrets,
    credentials or sensitive file paths.

## 12. Dependencies and run instructions

Inspect the current Python environment before writing dependencies. Keep
`frontend/requirements.txt` minimal and compatible with the repository.
Do not blindly pin arbitrary versions or duplicate the entire backend
dependency list.

The intended startup command from the repository root is:

``` bash
python -m streamlit run frontend/app.py
```

If the environment requires installing frontend dependencies separately,
document the exact command in `frontend/README.md`. Verify the startup
command in the actual environment before claiming it works.

## 13. Acceptance criteria

The frontend scaffold is complete when:

-   [ ] All P6-owned application files are under `frontend/`.
-   [ ] The UI has the off-white/navy/teal professional design.
-   [ ] A valid image can be uploaded and previewed.
-   [ ] Invalid/unreadable images show a helpful error.
-   [ ] Demo mode is explicit and visibly labelled.
-   [ ] Demo data is isolated from the live adapter.
-   [ ] Accepted, auto-fixed and escalated result layouts can render
    validated example dictionaries.
-   [ ] Missing optional values do not crash the UI or become fabricated
    values.
-   [ ] Before/after layout only labels an image as enhanced when a real
    enhanced image is supplied.
-   [ ] Live mode reports "not configured" clearly when the backend
    contract is missing.
-   [ ] The frontend does not import guessed model or queue modules.
-   [ ] The research-prototype disclaimer is visible.
-   [ ] The app startup command and setup instructions are documented.
-   [ ] Basic schema/demo tests are run and their outcomes reported.
-   [ ] Existing teammate code remains unchanged.

## 14. Required Antigravity execution order

1.  Inspect the repository and summarize existing structure and
    constraints.
2.  Identify any conflict between this specification and existing
    project conventions.
3.  Implement the minimal working `frontend/` scaffold and demo mode.
4.  Add result validation and reusable UI components.
5.  Add tests and run them.
6.  Run the documented startup command if possible.
7.  Report changed files, tests/commands actually run, and any
    unresolved issues.

Do not begin by implementing or rewriting the ML pipeline. The
deliverable at this stage is an attractive, working, isolated frontend
that is easy to connect once P5's actual interface is confirmed.
