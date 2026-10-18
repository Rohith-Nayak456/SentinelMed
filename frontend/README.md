# SentinelMed Modern Clinical Workflow & AI Reliability Dashboard

A modern, responsive, interactive medical workflow dashboard and conformal AI reliability scaffold for **SentinelMed**. Built with **Streamlit** following a clean, accessible healthcare software design system.

---

## 1. Design System & Palette Tokens

The interface implements a high-contrast, professional medical software aesthetic:

| Token | Hex Value | Usage |
| :--- | :--- | :--- |
| **Page Canvas Background** | `#F7F8FA` | Crisp, modern clinical background |
| **Main Surface / Cards** | `#FFFFFF` | Elevated cards, tables, and metric containers |
| **Subtle Surface** | `#F1F5F9` | Secondary cards, pill containers, and neutral badges |
| **Primary Text** | `#172B3A` | Deep slate-navy headings, labels, and primary values |
| **Secondary Text** | `#64748B` | Slate descriptive copy, captions, and secondary telemetry |
| **Primary Accent** | `#147D78` | Muted teal action buttons, active navigation, indicators |
| **Accent Hover** | `#0F625F` | Darker teal button hover state |
| **Borders & Dividers** | `#E2E8F0` | Subtle hairline dividers and card outlines |
| **Success / Accepted** | `#238568` | Conformal gate passed, normal baseline, doctor approved |
| **Warning / Auto-Fixed** | `#C88725` | Degraded contrast restored, medium priority, retake |
| **Critical / Escalated** | `#C94C43` | Epistemic uncertainty flagged, doctor triage required |

### Priority Tokens
- **Critical:** `#991B1B` on `#FEF2F2` (border `#FCA5A5`)
- **High:** `#C2410C` on `#FFF7ED` (border `#FDBA74`)
- **Medium:** `#B45309` on `#FFFBEB` (border `#FCD34D`)
- **Low:** `#475569` on `#F8FAFC` (border `#CBD5E1`)

---

## 2. Key Pages & Features

### 1. 📊 Executive Overview & Analytics
- **Live KPI Stat Cards:** Total Scans, Awaiting Doctor Review, Accepted (Gate Passed), Auto-Fixed (Restored), and Average Pipeline Latency.
- **Interactive Visualizations (Altair):**
  - **Verdict Distribution:** Responsive Donut chart showing Accepted vs Auto-Fixed vs Escalated breakdown.
  - **Triage Priority Distribution:** Urgency breakdown across Critical, High, Medium, and Low cases.
  - **Pipeline Stage Latency:** Measured runtime across Quality Inspection, Inference, Gate, and Recovery.
- **Recent Scans Activity Feed:** Real-time stream of incoming cases with quick status indicators.

### 2. 🔬 Single Scan Analysis
- **Upload Panel:** Drag-and-drop file uploader supporting PNG, JPG, and JPEG with PIL-decoded image verification and maximum 15 MB limit.
- **Clinical Metadata:** Scan ID, Imaging Modality (Radiograph, CT, MRI, Ultrasound), and Initial Triage Priority.
- **Demo Scan Loader:** Instant neutral synthetic scan generator for testing without personal medical imagery.
- **Analysis Execution:** Explicit **Analyse scan** action triggering reliability inspection. Automatically records analyzed scans into the session queue.
- **Verdict Hero Banner:** High-visibility banner for `ACCEPTED`, `AUTO-FIXED`, or `ESCALATED`.
- **Image Restoration Side-by-Side:** Side-by-side comparison for auto-fixed scans. Never duplicates the original scan as "after".
- **Quality & Uncertainty Telemetry:** Epistemic uncertainty meters, quality metrics, and latency breakdown.

### 3. 👨‍⚕️ Clinician Review & Priority Queue
- **Real-Time Search & Multi-Attribute Filters:** Filter by Review Status (Pending Review, Resolved, Accepted, Auto-Fixed) and Priority (Critical, High, Medium, Low).
- **Sortable Queue:** Sort by Priority (Critical → Low), Submission Time, or Execution Latency.
- **Case Review Drawer:** Diagnostic telemetry, model output, and conformal uncertainty rationale.
- **Priority Management:** Priority adjustment interface with audit trail logging.
- **Doctor Actions:**
  - **Approve Scan:** Clinician gate override.
  - **Order Retake:** Image defect triage.
  - **Confirm Triage:** Escalation to senior radiologist.
- **Case Audit Trail:** Immutable timestamped record of human decisions and clinical rationale.

### 4. 🔄 Batch Stream Queue
- Simulated automated scan stream showing unattended queue progression, status badges, and per-scan execution latencies.

### 5. ℹ️ System Architecture & About
- Research prototype safety guidelines, multi-stage gate workflow, P1–P5 integration boundaries, and schema documentation.

---

## 3. Directory Layout

```text
frontend/
├── app.py                      # Main Streamlit app: 5-page navigation, session state
├── requirements.txt            # Minimal frontend dependencies
├── README.md                   # Project documentation and run instructions
├── SentinelMed_P6_Antigravity_Spec.md  # Official P6 specification
├── .streamlit/
│   └── config.toml             # Modern healthcare theme configuration
├── components/
│   ├── __init__.py
│   ├── header.py               # Custom CSS design tokens, persistent banners
│   ├── dashboard_overview.py   # KPI cards, charts grid, and recent activity stream
│   ├── doctor_queue.py         # Clinician inbox, priority triage, actions, audit trail
│   ├── analytics_charts.py     # Responsive Altair charts (donut, priority bar, latency)
│   ├── upload_panel.py         # File validation, preview, modality & priority metadata
│   ├── result_panel.py         # Verdict hero banner & primary evidence metrics
│   ├── image_comparison.py     # Side-by-side restoration visual audit
│   └── metrics_panel.py        # Quality, uncertainty, measured latency & raw schema
├── services/
│   ├── __init__.py
│   ├── result_schema.py        # Typed contract validation (verdicts, priorities, doctor status)
│   ├── queue_store.py          # Session-state repository, statistics aggregation, audit log
│   ├── pipeline_adapter.py     # Isolated backend boundary (BackendNotConfiguredError)
│   └── demo_adapter.py         # Deterministic synthetic data generator (DEMO DATA)
└── tests/
    ├── __init__.py
    ├── test_result_schema.py   # Schema validation, bounds, and priority tests
    ├── test_queue_store.py     # Case lifecycle, statistics, and doctor review tests
    ├── test_demo_adapter.py    # Demo data and synthetic image tests
    └── test_pipeline_adapter.py# Adapter isolation and error propagation tests
```

---

## 4. Installation & Local Execution

From the repository root:

```bash
# 1. Install frontend requirements
python -m pip install -r frontend/requirements.txt

# 2. Run the application
python -m streamlit run frontend/app.py
```

The application will launch on `http://localhost:8501`.

---

## 5. Running the Automated Test Suite

Run the full pytest suite:

```bash
python -m pytest frontend/tests
```

**24 comprehensive unit tests** validate:
- Result schema validation and allowed verdicts/priorities
- Out-of-bounds guards on confidence and recoverability
- Pipeline adapter isolation and error raising
- Demo adapter determinism and synthetic enhancement isolation
- Queue store case lifecycle, live statistics calculation, and doctor review audit trail

---

## 6. How Teammates Connect Live Backend Modules (P1–P5)

All backend integration is isolated to [`frontend/services/pipeline_adapter.py`](file:///c:/Users/Vikhyath%20Shetty/SentinelMed/frontend/services/pipeline_adapter.py):

1. Open `frontend/services/pipeline_adapter.py`.
2. Import your team's pipeline entry point (e.g., `from backend.pipeline import run_sentinel_pipeline`).
3. Set `self._is_connected = True`.
4. Implement `PipelineAdapter.run_inference(self, image, scan_id=None)` to call your ML pipeline and return a dictionary matching the SentinelMed contract:
   ```python
   {
       "verdict": "accepted" | "auto_fixed" | "escalated",
       "scan_id": "SCAN-001",
       "pred": "Class 0 (Normal)",
       "confidence": 0.94,            # Float in [0.0, 1.0]
       "gate": "passed",              # Conformal gate status
       "cause": None,                 # e.g., "quality", "ood", "ambiguous"
       "recoverability": None,        # Float in [0.0, 1.0] if evaluated
       "priority": "Medium",          # "Critical", "High", "Medium", "Low"
       "quality": {                   # Quality metrics dict
           "blur_score": 142.0,
           "contrast_ratio": 68.2,
       },
       "uncertainty": {               # Epistemic uncertainty dict
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

---

## 7. Medical Disclaimer

**RESEARCH PROTOTYPE ONLY — NOT FOR CLINICAL DIAGNOSIS OR AUTONOMOUS TREATMENT DECISIONS.**

SentinelMed reliability verdicts evaluate image quality and conformal uncertainty bounds. Clinician review is the final authority for all patient management decisions.
