# SentinelMed — P2: Data and Image Quality

**IBM Datathon 2026 · Role: P2 — Data and Image Quality**

This folder contains all the work for the **P2** role in the SentinelMed project.
P2 is responsible for ensuring the PneumoniaMNIST dataset is clean, well-documented,
and consistently preprocessed before it is passed on to teammates (P1, P3, P4).

---

## Dataset at a Glance

| Split          | Images    | Resolution     | Format        |
|----------------|-----------|----------------|---------------|
| Training       | 4,708     | 28 × 28 pixels | Grayscale PNG |
| Validation     | 524       | 28 × 28 pixels | Grayscale PNG |
| Test           | 624       | 28 × 28 pixels | Grayscale PNG |
| **Total**      | **5,856** |                |               |

- **Source:** [MedMNIST v2 — PneumoniaMNIST](https://medmnist.com/)
- **Cache file:** `C:\Users\<you>\.medmnist\pneumoniamnist.npz` (3.98 MB)
- **Labels:** `0` = Normal (no pneumonia), `1` = Pneumonia
- **Pixel scale:** 0–255 (uint8). No normalisation applied to the raw PNG files.

---

## Folder Structure

```
SentinelMed/
|
+-- README.md                                <- This file.
|
+-- sentinelmed_p2_original_dataset.zip      <- Shareable package for teammates (4.4 MB).
|
+-- scripts/                                 <- All P2 Python scripts (run from here).
|   +-- check_dataset.py                     <- Verifies MedMNIST cache and split counts.
|   +-- quality_inspector.py                 <- Quality metrics for all 5,856 images.
|   +-- create_manifest.py                   <- Builds the full original-image manifest.
|   +-- export_original_images.py            <- Exports images from cache as PNGs.
|   +-- package_dataset.py                   <- Bundles images + reports into a ZIP.
|
+-- data/
|   +-- raw/                                 <- Original images from the MedMNIST cache.
|   |   |                                       READ-ONLY. Do not edit these files.
|   |   +-- train/   (4,708 PNG files)       <- train_0.png ... train_4707.png
|   |   +-- val/     (524 PNG files)         <- val_0.png   ... val_523.png
|   |   +-- test/    (624 PNG files)         <- test_0.png  ... test_623.png
|   |
|   +-- processed/                           <- Degraded copies go here (not yet generated).
|
+-- outputs/
|   +-- manifests/
|   |   +-- dataset_manifest_originals.csv   <- Full manifest; image_path left blank.
|   |   +-- dataset_manifest_with_paths.csv  <- Full manifest with relative PNG paths.
|   |
|   +-- quality_reports/
|   |   +-- quality_report_full.csv          <- Quality metrics for all 5,856 images.
|   |
|   +-- degraded_images/                     <- Approved degraded images (pending).
|   |
|   +-- validation/
|       +-- export_validation_report.csv     <- Per-image PNG export verification.
|
+-- docs/                                    <- Handoff documents for P1/P3/P4 (pending).
```

> **Pilot files** from early exploration remain in the root folder and are not part of
> the main pipeline: `manifest.csv` (10 images), `quality_report.csv` (100 images),
> `data/processed/` (30 degraded pilot images). Preserved for reference only.

---

## Completed Work

### 1. Dataset Verification — `scripts/check_dataset.py`

- Confirmed MedMNIST cache at `C:\Users\<you>\.medmnist\pneumoniamnist.npz`.
- Verified split counts: 4,708 / 524 / 624 match the official PneumoniaMNIST dataset.
- All images confirmed 28 × 28 pixels. medmnist v3.0.2. Python 3.13.
- **Result: PASSED. Zero errors.**

---

### 2. Full-Dataset Quality Inspection — `scripts/quality_inspector.py`

Computed four metrics for every one of the 5,856 original images:

| Metric       | Definition                                              | Observed range |
|--------------|---------------------------------------------------------|----------------|
| `brightness` | Mean grayscale pixel value (0–255)                      | 64.9 – 225.5   |
| `contrast`   | Standard deviation of pixel values                      | 6.3 – 87.4     |
| `blur_score` | Variance of the Laplacian (OpenCV) — higher = sharper   | 19.5 – 6,882.5 |
| `noise_score`| Mean absolute diff vs. 3×3 Gaussian blur                | 0.7 – 12.4     |

> **Limitation:** `blur_score` and `noise_score` are proxies affected by image content,
> not only by technical degradation. A naturally smooth image can score low even if
> it is clean. Compare only images of similar content.

Output: `outputs/quality_reports/quality_report_full.csv` (5,856 rows · 396 KB)
**Result: PASSED. Zero validation errors.**

---

### 3. Original-Image Manifest — `scripts/create_manifest.py`

- One record per original image across all three splits.
- Stable IDs: `train_0` … `train_4707`, `val_0` … `val_523`, `test_0` … `test_623`.
- Labels read directly from the MedMNIST dataset — never hardcoded.
- Cross-checked against the quality report: all IDs, labels, and dimensions match.

Output: `outputs/manifests/dataset_manifest_originals.csv` (5,856 rows · 730 KB)
**Result: PASSED. Zero mismatches.**

---

### 4. PNG Export — `scripts/export_original_images.py`

- All 5,856 images exported from the compressed NPZ cache as individual grayscale PNGs.
- Pixel values preserved exactly (uint8, 0–255, no normalisation).
- Every exported PNG was re-opened and pixel-matched against the source dataset.
- `image_path` column populated in the updated manifest with relative paths.

Outputs:
- `data/raw/train/` — 4,708 PNGs
- `data/raw/val/` — 524 PNGs
- `data/raw/test/` — 624 PNGs
- `outputs/manifests/dataset_manifest_with_paths.csv` (5,856 rows · 890 KB)
- `outputs/validation/export_validation_report.csv` (5,856 rows · 684 KB)

**Result: PASSED. 5,856 / 5,856 images verified. Zero failures.**

---

### 5. Dataset Package — `scripts/package_dataset.py`

- `sentinelmed_p2_original_dataset.zip` (4.4 MB) — ready to share with teammates.
- Contains: 5,856 PNGs + 3 CSV reports + this README (5,860 entries total).
- Folder structure inside the ZIP mirrors the project layout exactly.

**Result: PASSED. ZIP contents verified. Zero errors.**

---

## Pending Work

- [ ] **Controlled image degradation** — Apply blur, Gaussian noise, low brightness,
      and low contrast to all 5,856 images with reproducible random seeds (`42 + index`).
      Save to `data/processed/` and extend the manifest with degraded-image records.
- [ ] **Shared preprocessing pipeline** — Agree normalisation and augmentation
      parameters with P1, P3, and P4 before producing final model inputs.
- [ ] **Final handover document** — Write `docs/handoff.md` for P1, P3, and P4
      covering all outputs, file formats, and metric definitions.
- [ ] **Calibration / test allocation** — Not a P2 decision. Decided at team level.

---

## Output Files Reference

| File | Rows | Size | Status |
|------|------|------|--------|
| `outputs/manifests/dataset_manifest_originals.csv` | 5,856 | 730 KB | Done |
| `outputs/manifests/dataset_manifest_with_paths.csv` | 5,856 | 890 KB | Done |
| `outputs/quality_reports/quality_report_full.csv` | 5,856 | 396 KB | Done |
| `outputs/validation/export_validation_report.csv` | 5,856 | 684 KB | Done |
| `sentinelmed_p2_original_dataset.zip` | — | 4.4 MB | Done |

### Manifest column guide

| Column          | Example value                              | Notes |
|-----------------|--------------------------------------------|-------|
| `image_id`      | `train_0`                                  | Stable unique ID: `<split>_<index>` |
| `original_id`   | `train_0`                                  | Same as `image_id` for original images |
| `label`         | `1`                                        | `0` = Normal, `1` = Pneumonia |
| `split`         | `train`                                    | `train`, `val`, or `test` |
| `degradation`   | `original`                                 | Always `original` in this manifest |
| `image_path`    | `data/raw/train/train_0.png`               | **Relative to the project root** — see note below |
| `source_dataset`| `medmnist::PneumoniaMNIST::train::0`       | Traceable back to the MedMNIST NPZ cache |
| `width`         | `28`                                       | Pixels |
| `height`        | `28`                                       | Pixels |

> **Important for teammates using `image_path`:**
> All paths are **relative to the project root folder** — the folder that contains
> this README. For example, `data/raw/train/train_0.png` resolves to:
>
> `C:\Users\<you>\Downloads\SentinelMed\data\raw\train\train_0.png`
>
> If you extract `sentinelmed_p2_original_dataset.zip`, the same relative paths apply
> from whichever folder you extract into.

---

## Label Distribution (verified from actual dataset)

| Split     | Normal (label 0) | Pneumonia (label 1) | Total |
|-----------|-----------------|---------------------|-------|
| train     | 1,214           | 3,494               | 4,708 |
| val       | 135             | 389                 | 524   |
| test      | 234             | 390                 | 624   |
| **Total** | **1,583**       | **4,273**           | **5,856** |

> The dataset is **imbalanced** — Pneumonia images outnumber Normal images by roughly
> 2.7 : 1. Models trained on this data should account for this.

---

## How to Run the P2 Scripts (Windows)

Open **Command Prompt** or **PowerShell** in the `SentinelMed` folder, then run:

```powershell
# Step 1 — Verify dataset cache and counts (fast, no download needed)
python scripts/check_dataset.py

# Step 2 — Compute quality metrics for all 5,856 images (~1-2 minutes)
python scripts/quality_inspector.py

# Step 3 — Build the full original-image manifest
python scripts/create_manifest.py

# Step 4 — Export all images as individual PNGs (~3-5 minutes)
python scripts/export_original_images.py

# Step 5 — Package everything into a ZIP for sharing
python scripts/package_dataset.py
```

> **Tip:** If you see a `ModuleNotFoundError`, install the required libraries first:
> ```powershell
> pip install medmnist pandas numpy opencv-python matplotlib Pillow
> ```

---

## Notes for Teammates

- **Do not re-download** the dataset — MedMNIST reads from a local cache at
  `C:\Users\<you>\.medmnist\pneumoniamnist.npz`.
- **Treat `data/raw/` as read-only.** These are the ground-truth originals.
  Any processed or degraded copies go into `data/processed/`.
- **Labels:** `0` = Normal, `1` = Pneumonia. The dataset is imbalanced.
- **Random seed convention (pending):** `42 + image_index` for reproducible degradation.
- **Allocation decisions** (which images go to model training vs. calibration vs.
  held-out testing) are **NOT** decided by P2. That is a team-level decision.
- All quality metrics are computed on the **grayscale** version of each image (0–255 scale).
  No normalisation has been applied to the exported PNG files.
