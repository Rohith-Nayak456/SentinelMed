"""
create_manifest.py  (full-dataset version)
-------------------------------------------
P2 Manifest Creation — SentinelMed / IBM Datathon 2026

PURPOSE
-------
Build a complete manifest CSV covering every original image in
PneumoniaMNIST across all three splits (train + val + test = 5,856
images) and cross-check it against the quality report produced by
scripts/quality_inspector.py.

OUTPUT
------
    outputs/manifests/dataset_manifest_originals.csv

This script does NOT:
  - Download the dataset (reads from the existing MedMNIST cache)
  - Generate or save degraded images
  - Overwrite the pilot manifest  (manifest.csv in the root folder)
  - Overwrite the pilot quality report  (quality_report.csv in root)
  - Decide training / calibration / test allocations (P2 does not do that)

HOW TO RUN (from the SentinelMed folder)
-----------------------------------------
    python scripts/create_manifest.py

COLUMN DEFINITIONS
------------------
  image_id        Unique stable ID for this image, e.g. train_0, val_3, test_621.
                  Built from  <split>_<index>  where index is 0-based within
                  that split, matching the order PneumoniaMNIST returns images.

  original_id     For original (non-degraded) images this equals image_id.
                  Degraded copies (created by a separate script) will have a
                  different image_id but the same original_id as their source.

  label           Integer class label read directly from the dataset.
                    0 = Normal (no pneumonia)
                    1 = Pneumonia

  split           Which dataset partition the image belongs to:
                    train | val | test

  degradation     Type of modification applied to this image.
                  Always "original" in this manifest — no degraded copies here.

  image_path      Where a saved PNG of this image would be stored if it were
                  extracted.  Since PneumoniaMNIST images live inside the
                  compressed NPZ cache file rather than as separate PNGs, no
                  individual file has been extracted yet.  This column is left
                  blank for now and will be populated by a future extraction
                  step if the team decides to export individual PNGs.

  source_dataset  A documented reference to the origin of the image so
                  teammates can trace it back to the raw data.
                  Format:
                    medmnist::PneumoniaMNIST::<split>::<index>
                  Example:
                    medmnist::PneumoniaMNIST::train::0
                  This uniquely identifies the image within the MedMNIST
                  framework without depending on any file path.

  source_npz      Absolute path to the NPZ cache file on this machine.
                  Useful for debugging and reproducibility checks.

  width           Image width in pixels (expected: 28).
  height          Image height in pixels (expected: 28).

CROSS-CHECK WITH QUALITY REPORT
---------------------------------
After building the manifest this script loads
    outputs/quality_reports/quality_report_full.csv
and verifies:
  - Every manifest image_id appears in the quality report.
  - Labels match between the two files.
  - Dimensions match.
Any mismatch is recorded and the script exits with FAIL status.
"""

import os
import sys
import math

# ── Dependency guard ─────────────────────────────────────────────────────────
import importlib.util as _ilu
REQUIRED = {"medmnist": "medmnist", "pandas": "pandas", "PIL": "Pillow"}
missing = [pkg for pkg in REQUIRED if _ilu.find_spec(pkg) is None]
if missing:
    pkgs = " ".join(REQUIRED[p] for p in missing)
    print(f"[ERROR] Missing packages: {', '.join(missing)}")
    print(f"Install with:  pip install {pkgs}")
    sys.exit(1)

import pandas as pd
from medmnist import PneumoniaMNIST

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "manifests")
OUTPUT_CSV   = os.path.join(OUTPUT_DIR,   "dataset_manifest_originals.csv")

QR_CSV       = os.path.join(PROJECT_ROOT, "outputs", "quality_reports",
                             "quality_report_full.csv")

# Files that must NOT be overwritten
PROTECTED = [
    os.path.join(PROJECT_ROOT, "manifest.csv"),
    os.path.join(PROJECT_ROOT, "quality_report.csv"),
]

# MedMNIST cache location
CACHE_DIR  = os.path.join(os.path.expanduser("~"), ".medmnist")
CACHE_FILE = os.path.join(CACHE_DIR, "pneumoniamnist.npz")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Expected counts ──────────────────────────────────────────────────────────
SPLITS = [
    ("train", 4708),
    ("val",   524),
    ("test",  624),
]
EXPECTED_TOTAL  = 5856
EXPECTED_W      = 28
EXPECTED_H      = 28
VALID_LABELS    = {0, 1}

# ── Header ───────────────────────────────────────────────────────────────────
SEP = "=" * 65
print(SEP)
print("SentinelMed -- P2 Full-Dataset Manifest Creation")
print(SEP)
print(f"\nOutput manifest  : {OUTPUT_CSV}")
print(f"Quality report   : {QR_CSV}")
print(f"NPZ cache        : {CACHE_FILE}")
print()
for path in PROTECTED:
    print(f"Protected (will NOT be overwritten): {path}")
print()

# ── Guard: output must not already be a protected file (sanity check) ────────
if OUTPUT_CSV in PROTECTED:
    print("[ERROR] Output path conflicts with a protected file. Aborting.")
    sys.exit(1)

# ── Guard: cache must exist ──────────────────────────────────────────────────
if not os.path.isfile(CACHE_FILE):
    print("[ERROR] MedMNIST cache not found:", CACHE_FILE)
    print("Run download_dataset.py first.")
    sys.exit(1)

# ── Guard: quality report must exist ─────────────────────────────────────────
if not os.path.isfile(QR_CSV):
    print("[ERROR] Quality report not found:", QR_CSV)
    print("Run scripts/quality_inspector.py first.")
    sys.exit(1)

# ── Build manifest ────────────────────────────────────────────────────────────
print("-" * 65)
print("Building manifest from MedMNIST cache (no download)...")
print("-" * 65)

records = []

for split_name, expected_count in SPLITS:
    dataset = PneumoniaMNIST(split=split_name, download=False)
    actual  = len(dataset)
    print(f"\n  Split: {split_name}  |  loaded {actual} images"
          f"  (expected {expected_count})"
          f"  [{'OK' if actual == expected_count else 'MISMATCH'}]")

    for i in range(actual):
        image, label_arr = dataset[i]
        label    = int(label_arr[0])
        image_id = f"{split_name}_{i}"
        w, h     = image.size          # PIL (width, height)

        records.append({
            "image_id":       image_id,
            "original_id":    image_id,
            "label":          label,
            "split":          split_name,
            "degradation":    "original",
            "image_path":     "",       # not extracted to PNG yet
            "source_dataset": f"medmnist::PneumoniaMNIST::{split_name}::{i}",
            "source_npz":     CACHE_FILE,
            "width":          w,
            "height":         h,
        })

    print(f"  Done -- {actual} records added for '{split_name}'.")

df = pd.DataFrame(records)

# ── Validation block 1: counts ───────────────────────────────────────────────
print()
print("-" * 65)
print("Validation (manifest internal checks)...")
print("-" * 65)

issues = []

# 1a. Total count
if len(df) != EXPECTED_TOTAL:
    issues.append(
        f"Total record count: got {len(df)}, expected {EXPECTED_TOTAL}."
    )

# 1b. Per-split count
for split_name, expected_count in SPLITS:
    actual = df[df["split"] == split_name].shape[0]
    status = "OK" if actual == expected_count else "MISMATCH"
    print(f"  {split_name:<6} : {actual:>5} records  (expected {expected_count})  [{status}]")
    if actual != expected_count:
        issues.append(f"Split '{split_name}': got {actual}, expected {expected_count}.")

# 1c. Unique IDs
dup_ids = df[df.duplicated("image_id")]["image_id"].tolist()
if dup_ids:
    issues.append(f"Duplicate image_ids found: {dup_ids[:5]} (showing first 5).")
else:
    print(f"  Unique IDs     : {len(df)}  [OK]")

# 1d. Valid labels
invalid_labels = df[~df["label"].isin(VALID_LABELS)]["image_id"].tolist()
if invalid_labels:
    issues.append(
        f"Invalid labels on {len(invalid_labels)} image(s): "
        f"{invalid_labels[:5]} (showing first 5)."
    )
else:
    print(f"  Label values   : all in {{0, 1}}  [OK]")

# 1e. Dimensions
bad_dims = df[(df["width"] != EXPECTED_W) | (df["height"] != EXPECTED_H)]
if not bad_dims.empty:
    issues.append(
        f"{len(bad_dims)} image(s) have unexpected dimensions "
        f"(expected {EXPECTED_W}x{EXPECTED_H})."
    )
else:
    print(f"  Dimensions     : all {EXPECTED_W}x{EXPECTED_H}  [OK]")

# ── Validation block 2: cross-check with quality report ─────────────────────
print()
print("-" * 65)
print("Cross-checking against quality report...")
print("-" * 65)

qr = pd.read_csv(QR_CSV)

# Keep only original rows from the quality report (all rows are original here,
# but this is defensive in case the report is extended later)
qr_orig = qr[qr["degradation"] == "original"].copy()

manifest_ids = set(df["image_id"])
qr_ids       = set(qr_orig["image_id"])

# 2a. IDs in manifest but missing from quality report
only_manifest = manifest_ids - qr_ids
if only_manifest:
    sample = sorted(only_manifest)[:5]
    issues.append(
        f"{len(only_manifest)} manifest ID(s) missing from quality report: "
        f"{sample} (showing first 5)."
    )
else:
    print(f"  All {len(manifest_ids)} manifest IDs found in quality report  [OK]")

# 2b. IDs in quality report but not in manifest
only_qr = qr_ids - manifest_ids
if only_qr:
    sample = sorted(only_qr)[:5]
    issues.append(
        f"{len(only_qr)} quality-report ID(s) not in manifest: "
        f"{sample} (showing first 5)."
    )
else:
    print(f"  No extra IDs in quality report  [OK]")

# 2c. Label mismatches on shared IDs
shared_ids = manifest_ids & qr_ids
if shared_ids:
    df_shared  = df[df["image_id"].isin(shared_ids)].set_index("image_id")
    qr_shared  = qr_orig[qr_orig["image_id"].isin(shared_ids)].set_index("image_id")
    label_diff = (df_shared["label"] != qr_shared["label"])
    bad_labels = label_diff[label_diff].index.tolist()
    if bad_labels:
        issues.append(
            f"Label mismatch between manifest and quality report on "
            f"{len(bad_labels)} image(s): {bad_labels[:5]} (showing first 5)."
        )
    else:
        print(f"  Labels match quality report on all shared IDs  [OK]")

    # 2d. Dimension mismatches on shared IDs
    dim_diff = (
        (df_shared["width"]  != qr_shared["width"]) |
        (df_shared["height"] != qr_shared["height"])
    )
    bad_dims_shared = dim_diff[dim_diff].index.tolist()
    if bad_dims_shared:
        issues.append(
            f"Dimension mismatch between manifest and quality report on "
            f"{len(bad_dims_shared)} image(s): "
            f"{bad_dims_shared[:5]} (showing first 5)."
        )
    else:
        print(f"  Dimensions match quality report on all shared IDs  [OK]")

# ── Save manifest ─────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
size_kb = round(os.path.getsize(OUTPUT_CSV) / 1024, 1)

# ── Final summary ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("RESULTS SUMMARY")
print(SEP)
print(f"  Total records written  : {len(df)}")
for split_name, expected_count in SPLITS:
    n = df[df["split"] == split_name].shape[0]
    label_counts = df[df["split"] == split_name]["label"].value_counts().sort_index()
    lc_str = "  ".join(f"label {k}={v}" for k, v in label_counts.items())
    print(f"  {split_name:<6}  {n:>5} images   ({lc_str})")
print(f"\n  Output file            : {OUTPUT_CSV}  ({size_kb} KB)")
print(f"  Columns                : {list(df.columns)}")
print(f"\n  Validation issues      : {len(issues)}")
if issues:
    for issue in issues:
        print(f"    [ISSUE] {issue}")

print()
if not issues:
    print("OVERALL STATUS: PASSED")
    print("All 5,856 records written. Cross-check with quality report: clean.")
else:
    print("OVERALL STATUS: FAILED")
    print("Review the issues listed above before sharing this manifest.")

print(SEP)
