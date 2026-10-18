# SentinelMed
Medical image quality and reliability assessment prototype

**IBM Z Datathon 2026**

SentinelMed is an enterprise-grade medical imaging reliability and quality assurance platform designed for clinical decision support on chest X-ray scans (PneumoniaMNIST).

---

## Architecture & System Overview

SentinelMed operates across an end-to-end reliability and triage pipeline:

- **P1 — Classifier & Uncertainty:** Deep neural network classification (ResNet-18 / MobileNet-V2) and predictive uncertainty estimation (entropy, margin).
- **P2 — Data & Image Quality:** Dataset validation, manifest management, and image quality assessment (contrast, brightness, sharpness, noise).
- **P3 — Conformal Reliability Gate:** Risk-controlled prediction sets and coverage guarantees to catch unreliable predictions before clinical exposure.
- **P4 — Core Innovation / Reliability Engine:** Self-healing reliability triage featuring failure cause diagnosis, baseline recoverability scoring, cause-specific enhancement (CLAHE, Non-Local Means, Unsharp Masking), and post-recovery verification loops.
- **P5 — Clinical Review Queue:** Human-in-the-loop review triage, priority ranking, and audit logging.
- **P6 — Frontend Dashboard:** Streamlit clinical interface and doctor verification portal.

---

## P4: Core Innovation / Reliability Engine

The P4 reliability engine is located in [`reliability/`](reliability/) with complete standalone execution capabilities in [`run_p4.py`](run_p4.py) and documentation in [`docs/P4_RELIABILITY_ENGINE.md`](docs/P4_RELIABILITY_ENGINE.md).

### Core Capabilities
1. **Input Validation:** Strict data contract enforcement, shape/channel checking, probability axiom validation.
2. **Failure Cause Diagnosis:** Multi-cause diagnostic rules detecting contrast degradation, spatial blur, high-frequency noise, brightness anomalies, prediction ambiguity, and conformal gate failures.
3. **Recoverability Baseline:** Pre-recovery triage scoring to prevent fruitless compute and patient delay.
4. **Targeted Image Recovery:** Cause-specific enhancements (CLAHE for contrast, NLM for noise, Unsharp Masking for blur) preserving original raw artifacts.
5. **Post-Recovery Verification Loop:** Comparative pre/post verification enforcing measurable quality improvements and rejecting confidence-only illusions.
6. **Clinical Review Handoff:** Standardized JSON and CSV handoff structures for downstream P5 triage.

### Running P4 Tests
```powershell
python -m pytest tests/ -v
```

### Running P4 Pipeline
```powershell
python run_p4.py
```
