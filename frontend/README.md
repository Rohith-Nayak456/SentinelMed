# SentinelMed P6 Frontend Scaffold

This directory contains the **P6 Frontend** implementation for SentinelMed, built with **Streamlit** following the minimalist off-white, deep navy, and muted teal technical design system.

The frontend is built as an **isolated, integration-ready scaffold** with a working upload panel, schema validation, and a clearly labelled demo mode. Live backend integration remains strictly decoupled in the pipeline adapter so that teammate ML modules (P1–P5) can be connected without modifying the UI layer.

---

## 1. Directory Structure

```text
frontend/
├── app.py                      # Main Streamlit app: routing, state, and page layout
├── requirements.txt            # Minimal frontend dependencies
├── README.md                   # This documentation and run instructions
├── SentinelMed_P6_Antigravity_Spec.md  # Official P6 specification
├── .streamlit/
│   └── config.toml             # Palette and Streamlit UI theme configuration
├── components/
│   ├── __init__.py
│   ├── header.py               # Headers, design tokens, custom CSS, disclaimers
│   ├── upload_panel.py         # Image upload, file size & format validation, preview
│   ├── result_panel.py         # Primary verdict hero banner & decision evidence
│   ├── image_comparison.py     # Side-by-side original vs enhanced restoration view
│   └── metrics_panel.py        # Quality, uncertainty, latency breakdown & raw schema
├── services/
│   ├── __init__.py
│   ├── result_schema.py        # Typed contract validation (accepted, auto_fixed, escalated)
│   ├── pipeline_adapter.py     # Isolated backend boundary (raises clear error until wired)
│   └── demo_adapter.py         # Deterministic synthetic data generator (DEMO DATA)
└── tests/
    ├── __init__.py
    ├── test_result_schema.py   # Validation and error checking tests
    ├── test_demo_adapter.py    # Demo data and synthetic image tests
    └── test_pipeline_adapter.py# Adapter isolation and error propagation tests
```

---

## 2. Design System & Palette Tokens

The interface uses a calm, professional healthcare aesthetic designed for technical clarity:

| Token | Hex Value | Usage |
| :--- | :--- | :--- |
| **Warm Page Background** | `#F7F6F2` | App canvas background |
| **Panel Surface** | `#FFFFFF` | Cards and evidence containers |
| **Subtle Secondary Surface**| `#EFEEE8` | Secondary metric cards and placeholders |
| **Primary Text** | `#18313B` | Deep navy headings and labels |
| **Secondary Text** | `#65747B` | Slate descriptive copy and captions |
| **Primary Accent** | `#147D78` | Muted teal action buttons |
| **Accent Hover** | `#0F625F` | Darker teal button hover |
| **Borders & Dividers** | `#E3E6E2` | Hairline dividers and container outlines |
| **Accepted Status** | `#238568` | Conformal gate pass badge & banner |
| **Auto-Fixed Status** | `#C88725` | Recovered scan / attention badge & banner |
| **Escalated Status** | `#C94C43` | Triage / specialist human review required |

---

## 3. Installation & Setup

From the repository root:

```bash
# Install frontend requirements
python -m pip install -r frontend/requirements.txt
```

---

## 4. Running the Frontend Application

Run the application using the Streamlit CLI:

```bash
python -m streamlit run frontend/app.py
```

The application will launch on `http://localhost:8501`.

---

## 5. Execution Modes

### Demo Mode (Synthetic Data)
- **Active by default.**
- Prominently labeled with **DEMO DATA — NOT LIVE INFERENCE**.
- Allows exploring all three verdict paths:
  1. **Accepted:** High quality, high confidence, gate passed.
  2. **Auto-Fixed:** Degraded contrast detected, automated enhancement applied, recheck verified. Shows original vs enhanced image side-by-side.
  3. **Escalated:** Out-of-distribution anomaly, high epistemic uncertainty, low recoverability. Flags mandatory human specialist review.
- Supports batch inspection testing in the **Scan Queue** tab.

### Live Backend Mode
- Routes requests to `frontend/services/pipeline_adapter.py`.
- If teammate modules are not yet plugged in, it **never** silently falls back to demo data. Instead, it informs the operator with a clear `BackendNotConfiguredError` and instructions on connecting the pipeline.

---

## 6. How Teammates Connect the Live Backend (P1–P5)

All backend integration is isolated to `frontend/services/pipeline_adapter.py`. To connect the real pipeline:

1. Open `frontend/services/pipeline_adapter.py`.
2. Import your team's pipeline entrypoint (e.g., from `backend.pipeline import run_sentinel_pipeline`).
3. Set `self._is_connected = True` in `PipelineAdapter.__init__`.
4. In `PipelineAdapter.run_inference(self, image, scan_id=None)`:
   - Call your pipeline.
   - Return a dictionary conforming to the SentinelMed result contract:
     ```python
     {
         "verdict": "accepted" | "auto_fixed" | "escalated",
         "scan_id": "SCAN-001",
         "pred": "Class Label or Index",
         "confidence": 0.94,            # Float in [0.0, 1.0]
         "gate": "passed",              # Conformal gate status
         "cause": None,                 # e.g., "quality", "ood", "ambiguous"
         "recoverability": None,        # Float in [0.0, 1.0] if evaluated
         "quality": {                   # Dict of image quality metrics
             "blur_score": 142.0,
             "contrast_ratio": 68.2,
         },
         "uncertainty": {               # Dict of epistemic uncertainty metrics
             "entropy": 0.12,
             "mc_dropout_var": 0.015,
         },
         "latency_ms": {                # Measured execution latencies
             "total": 412.0,
             "quality": 45.0,
             "inference": 320.0,
         },
         "explanation": "Decision reasoning string...",
         "enhanced_image": pil_image,   # Attached only if restored!
     }
     ```
5. Run the test suite:
   ```bash
   python -m pytest frontend/tests
   ```

---

## 7. Running Tests

Execute the automated test suite with pytest:

```bash
python -m pytest frontend/tests
```

All 16 unit tests validate schema normalization, out-of-bounds guards, demo isolation, and adapter error propagation.

---

## 8. Clinical Disclaimer

**RESEARCH PROTOTYPE ONLY — NOT FOR CLINICAL DIAGNOSIS OR TREATMENT DECISIONS.**

Verdicts produced by SentinelMed evaluate image quality thresholds and model conformal confidence. An **Accepted** verdict does not constitute medical certification of health or disease absence.
