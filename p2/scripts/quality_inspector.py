"""
quality_inspector.py  (full-dataset version)
---------------------------------------------
P2 Quality Inspection — SentinelMed / IBM Datathon 2026

PURPOSE
-------
Compute four image-quality metrics for every original image in
PneumoniaMNIST (train + val + test = 5,856 images total) and save
the results to:
    outputs/quality_reports/quality_report_full.csv

A separate validation report is written to:
    outputs/quality_reports/quality_validation_report.csv

This script does NOT:
  - Download the dataset (reads from the existing MedMNIST cache)
  - Generate degraded images
  - Change any labels or splits
  - Overwrite the earlier pilot report (quality_report.csv in the root)

HOW TO RUN (from the SentinelMed folder)
-----------------------------------------
    python scripts/quality_inspector.py

METRIC DEFINITIONS
------------------
All metrics are computed on the GRAYSCALE version of each image
(pixel values 0–255, integer scale).

  brightness
    Formula : mean(gray)
    Meaning : Average pixel intensity across the image.
              0 = pure black, 255 = pure white.
    Range   : 0–255

  contrast
    Formula : std(gray)
    Meaning : Spread of pixel intensities around the mean.
              A flat, uniform image has contrast ≈ 0.
              A high-contrast image has large contrast.
    Range   : 0–127 (approx)

  blur_score
    Formula : var( Laplacian(gray) )   [OpenCV Laplacian, CV_64F]
    Meaning : The Laplacian detects edges. Blurry images have weak
              edges, so the variance of the Laplacian is low.
              Sharp images have high blur_score.
    LIMITATION: This is a PROXY. A naturally smooth image (e.g.
              a plain background) can have a low blur_score even
              though it was never blurred. Always compare images
              of similar content.
    Range   : > 0 (no upper bound; typical range for 28×28 chest
              X-rays: 50–3000)

  noise_score
    Formula : mean( abs( gray - GaussianBlur(gray, 3×3) ) )
    Meaning : Measures how much the image differs from a smoothed
              version of itself. High values suggest more
              high-frequency variation (noise or fine detail).
    LIMITATION: This is also a PROXY. Fine anatomical detail
              raises noise_score even in a clean image. Low values
              mean the image is smooth, not necessarily clean.
    Range   : 0–255 (typical range for chest X-rays: 1–15)

OUTPUT COLUMNS
--------------
  image_id     : Unique ID for this record  (e.g. train_0)
  original_id  : ID of the source image     (same as image_id for originals)
  label        : 0 = Normal, 1 = Pneumonia
  split        : train | val | test
  degradation  : "original" for all records in this script
  brightness   : float, rounded to 2 decimal places
  contrast     : float, rounded to 2 decimal places
  blur_score   : float, rounded to 2 decimal places
  noise_score  : float, rounded to 2 decimal places
  width        : int, image width in pixels  (expected: 28)
  height       : int, image height in pixels (expected: 28)
"""

import os
import sys
import math

# ── Guard: confirm we can import everything before starting work ─────────────
missing = []
for pkg in ("medmnist", "numpy", "cv2", "pandas", "PIL"):
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    # cv2 is installed via opencv-python, PIL via Pillow
    install_names = {
        "cv2": "opencv-python",
        "PIL": "Pillow",
    }
    pkgs = " ".join(install_names.get(p, p) for p in missing)
    print(f"[ERROR] Missing packages: {', '.join(missing)}")
    print(f"Install them with:\n  pip install {pkgs}")
    sys.exit(1)

import numpy as np
import cv2
import pandas as pd
from medmnist import PneumoniaMNIST

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "quality_reports")
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "quality_report_full.csv")
VALIDATION_CSV = os.path.join(OUTPUT_DIR, "quality_validation_report.csv")

# Guard: do not overwrite the pilot report in the root folder
PILOT_CSV = os.path.join(PROJECT_ROOT, "quality_report.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Expected values ──────────────────────────────────────────────────────────
SPLITS = [
    ("train", 4708),
    ("val",   524),
    ("test",  624),
]
EXPECTED_WIDTH  = 28
EXPECTED_HEIGHT = 28
VALID_LABELS    = {0, 1}

# ── Helper: compute metrics for one grayscale numpy array ────────────────────

def compute_metrics(gray: np.ndarray) -> dict:
    """
    Given a 2-D uint8 numpy array (grayscale, 0-255), return a dict
    with brightness, contrast, blur_score, and noise_score.

    Returns NaN values if anything goes wrong so errors are visible
    rather than silently replaced with zeros.
    """
    try:
        brightness  = float(np.mean(gray))
        contrast    = float(np.std(gray))
        blur_score  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        smooth      = cv2.GaussianBlur(gray, (3, 3), 0)
        noise_score = float(
            np.mean(np.abs(gray.astype(float) - smooth.astype(float)))
        )
        return {
            "brightness":  brightness,
            "contrast":    contrast,
            "blur_score":  blur_score,
            "noise_score": noise_score,
        }
    except Exception as exc:
        return {
            "brightness":  float("nan"),
            "contrast":    float("nan"),
            "blur_score":  float("nan"),
            "noise_score": float("nan"),
            "_error":      str(exc),
        }


# ── Main loop ────────────────────────────────────────────────────────────────

print("=" * 65)
print("SentinelMed — P2 Full-Dataset Quality Inspection")
print("=" * 65)
print(f"\nOutput file      : {OUTPUT_CSV}")
print(f"Validation file  : {VALIDATION_CSV}")
print(f"Pilot report kept: {PILOT_CSV}  (untouched)")
print()

records    = []   # one row per image
error_rows = []   # rows that failed a check

for split_name, expected_count in SPLITS:
    print(f"Processing split: {split_name}  (expected {expected_count} images)")

    dataset = PneumoniaMNIST(split=split_name, download=False)
    actual_count = len(dataset)

    if actual_count != expected_count:
        print(f"  [WARNING] Got {actual_count} images, expected {expected_count}.")

    for i in range(actual_count):
        image, label_arr = dataset[i]
        label = int(label_arr[0])
        image_id = f"{split_name}_{i}"

        # ── Dimension check ──────────────────────────────────────────────────
        width, height = image.size   # PIL Image .size → (width, height)

        # ── Convert to grayscale numpy array ─────────────────────────────────
        gray = np.array(image.convert("L"))

        # ── Compute metrics ──────────────────────────────────────────────────
        metrics = compute_metrics(gray)

        # ── Build the row ────────────────────────────────────────────────────
        row = {
            "image_id":    image_id,
            "original_id": image_id,       # same: this is an original image
            "label":       label,
            "split":       split_name,
            "degradation": "original",
            "brightness":  round(metrics["brightness"],  2),
            "contrast":    round(metrics["contrast"],    2),
            "blur_score":  round(metrics["blur_score"],  2),
            "noise_score": round(metrics["noise_score"], 2),
            "width":       width,
            "height":      height,
        }
        records.append(row)

        # ── Validation checks ────────────────────────────────────────────────
        issues = []

        if label not in VALID_LABELS:
            issues.append(f"invalid label: {label}")

        if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
            issues.append(
                f"unexpected size: {width}x{height} "
                f"(expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT})"
            )

        for metric_name in ("brightness", "contrast", "blur_score", "noise_score"):
            val = metrics[metric_name]
            if not math.isfinite(val):
                issues.append(f"{metric_name} is not finite: {val}")

        if "_error" in metrics:
            issues.append(f"metric computation error: {metrics['_error']}")

        if issues:
            error_rows.append({
                "image_id": image_id,
                "split":    split_name,
                "index":    i,
                "issues":   "; ".join(issues),
            })

    print(f"  Done — {actual_count} images processed.")

# ── Save main quality report ──────────────────────────────────────────────────
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)

# ── Save validation report ────────────────────────────────────────────────────
if error_rows:
    df_errors = pd.DataFrame(error_rows)
else:
    # Create an empty validation report so the file always exists
    df_errors = pd.DataFrame(
        columns=["image_id", "split", "index", "issues"]
    )

df_errors.to_csv(VALIDATION_CSV, index=False)

# ── Final summary and PASS/FAIL ───────────────────────────────────────────────
print()
print("=" * 65)
print("RESULTS SUMMARY")
print("=" * 65)

total_expected = sum(count for _, count in SPLITS)
total_actual   = len(records)
total_errors   = len(error_rows)

for split_name, expected_count in SPLITS:
    actual = df[df["split"] == split_name].shape[0]
    status = "OK" if actual == expected_count else "MISMATCH"
    print(f"  {split_name:<6}  {actual:>5} images processed  "
          f"(expected {expected_count})  [{status}]")

print(f"\n  Total images processed : {total_actual}")
print(f"  Total expected         : {total_expected}")
print(f"  Validation errors      : {total_errors}")
print()
print(f"  Quality report   : {OUTPUT_CSV}")
print(f"  Validation log   : {VALIDATION_CSV}")
print()

count_ok    = total_actual == total_expected
metrics_ok  = total_errors == 0

if count_ok and metrics_ok:
    print("OVERALL STATUS: PASSED")
    print("All 5,856 images processed. All metrics are finite and valid.")
else:
    print("OVERALL STATUS: FAILED")
    if not count_ok:
        print(f"  Image count mismatch: got {total_actual}, "
              f"expected {total_expected}.")
    if not metrics_ok:
        print(f"  {total_errors} image(s) had validation errors.")
        print(f"  See: {VALIDATION_CSV}")

print("=" * 65)
