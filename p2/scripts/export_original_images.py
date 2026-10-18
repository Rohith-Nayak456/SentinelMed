"""
export_original_images.py
--------------------------
P2 Image Export — SentinelMed / IBM Datathon 2026

PURPOSE
-------
Export every original PneumoniaMNIST image from the existing MedMNIST
cache as a grayscale PNG file.  Creates three sub-folders under data/raw/:

    data/raw/train/   <- 4,708 PNG files  (train_0.png ... train_4707.png)
    data/raw/val/     <- 524 PNG files    (val_0.png   ... val_523.png)
    data/raw/test/    <- 624 PNG files    (test_0.png  ... test_623.png)

Then it produces two output files:

    outputs/manifests/dataset_manifest_with_paths.csv
        A copy of dataset_manifest_originals.csv with image_path filled in.
        The original manifest is NEVER modified.

    outputs/validation/export_validation_report.csv
        Per-image verification: re-opens every saved PNG and checks that
        the dimensions are 28x28 and that the pixel values match the
        source dataset.

This script does NOT:
  - Download the dataset (reads only from the existing MedMNIST cache)
  - Generate degraded images
  - Modify or overwrite the source NPZ file
  - Overwrite existing PNG files silently (reports what it finds)
  - Modify manifest.csv, quality_report.csv, or any other existing file
  - Decide calibration/training allocations

HOW TO RUN (from the SentinelMed folder)
-----------------------------------------
    python scripts/export_original_images.py

If any PNG files already exist, the script will report them and skip
re-exporting them by default.  To force a fresh export, delete the
data/raw/train/, data/raw/val/, and data/raw/test/ sub-folders first.

IMAGE FORMAT NOTE
------------------
PneumoniaMNIST images are stored as single-channel (grayscale) uint8
arrays in the NPZ file.  When saved via PIL as mode "L" PNG files, the
original pixel values (0-255) are preserved exactly with no
normalization or rescaling.
"""

import os
import sys

# ── Dependency guard ─────────────────────────────────────────────────────────
import importlib.util as _ilu

REQUIRED = {
    "medmnist": "medmnist",
    "pandas":   "pandas",
    "PIL":      "Pillow",
    "numpy":    "numpy",
}
missing = [pkg for pkg in REQUIRED if _ilu.find_spec(pkg) is None]
if missing:
    pkgs = " ".join(REQUIRED[p] for p in missing)
    print(f"[ERROR] Missing packages: {', '.join(missing)}")
    print(f"Install with:  pip install {pkgs}")
    sys.exit(1)

import numpy as np
import pandas as pd
from PIL import Image
from medmnist import PneumoniaMNIST

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Input manifest (read-only — never modified)
SOURCE_MANIFEST = os.path.join(
    PROJECT_ROOT, "outputs", "manifests", "dataset_manifest_originals.csv"
)

# Output files (new — must not exist as protected files)
OUTPUT_MANIFEST = os.path.join(
    PROJECT_ROOT, "outputs", "manifests", "dataset_manifest_with_paths.csv"
)
VALIDATION_CSV  = os.path.join(
    PROJECT_ROOT, "outputs", "validation", "export_validation_report.csv"
)

# data/raw sub-folder per split
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
SPLIT_DIRS = {
    "train": os.path.join(RAW_DIR, "train"),
    "val":   os.path.join(RAW_DIR, "val"),
    "test":  os.path.join(RAW_DIR, "test"),
}

# MedMNIST cache
CACHE_FILE = os.path.join(os.path.expanduser("~"), ".medmnist", "pneumoniamnist.npz")

# Files that must NOT be overwritten under any circumstances
PROTECTED = [
    os.path.join(PROJECT_ROOT, "manifest.csv"),
    os.path.join(PROJECT_ROOT, "quality_report.csv"),
    os.path.join(PROJECT_ROOT, "outputs", "manifests", "dataset_manifest_originals.csv"),
    os.path.join(PROJECT_ROOT, "outputs", "quality_reports", "quality_report_full.csv"),
]

# ── Expected values ───────────────────────────────────────────────────────────
SPLITS = [
    ("train", 4708),
    ("val",   524),
    ("test",  624),
]
EXPECTED_TOTAL = 5856
EXPECTED_W     = 28
EXPECTED_H     = 28
VALID_LABELS   = {0, 1}

SEP = "=" * 65

# ── Preflight checks ──────────────────────────────────────────────────────────
print(SEP)
print("SentinelMed -- P2 Original Image Export")
print(SEP)

# Confirm outputs will not collide with protected files
for path in [OUTPUT_MANIFEST, VALIDATION_CSV]:
    if path in PROTECTED:
        print(f"[ERROR] Output path conflicts with a protected file: {path}")
        sys.exit(1)

# Source manifest must exist
if not os.path.isfile(SOURCE_MANIFEST):
    print(f"[ERROR] Source manifest not found: {SOURCE_MANIFEST}")
    print("Run scripts/create_manifest.py first.")
    sys.exit(1)

# NPZ cache must exist
if not os.path.isfile(CACHE_FILE):
    print(f"[ERROR] MedMNIST cache not found: {CACHE_FILE}")
    print("Run download_dataset.py first.")
    sys.exit(1)

print(f"\nSource manifest  : {SOURCE_MANIFEST}")
print(f"Output manifest  : {OUTPUT_MANIFEST}")
print(f"Validation report: {VALIDATION_CSV}")
print(f"PNG destination  : {RAW_DIR}")
print(f"NPZ cache        : {CACHE_FILE}")
print()
print("Protected files (will NOT be touched):")
for p in PROTECTED:
    print(f"  {p}")

# ── Create output directories ─────────────────────────────────────────────────
print()
for split_name, split_dir in SPLIT_DIRS.items():
    os.makedirs(split_dir, exist_ok=True)
    print(f"Directory ready: {split_dir}")

# ── Load source manifest ──────────────────────────────────────────────────────
print()
df_source = pd.read_csv(SOURCE_MANIFEST)
print(f"Source manifest loaded: {len(df_source)} rows")

# Build a lookup: image_id -> row index in df_source for fast access
source_lookup = df_source.set_index("image_id")

# ── Pre-scan: count already-existing files ────────────────────────────────────
print()
print("-" * 65)
print("Scanning for already-exported files...")
print("-" * 65)

already_exist   = []   # (image_id, path) tuples
to_export       = []   # (split_name, index, image_id) tuples

for split_name, expected_count in SPLITS:
    split_dir  = SPLIT_DIRS[split_name]
    exist_count = 0
    for i in range(expected_count):
        image_id = f"{split_name}_{i}"
        png_path = os.path.join(split_dir, f"{image_id}.png")
        if os.path.isfile(png_path):
            already_exist.append((image_id, png_path))
            exist_count += 1
        else:
            to_export.append((split_name, i, image_id))
    print(f"  {split_name:<6}: {exist_count:>5} already exist, "
          f"{expected_count - exist_count:>5} to export")

print(f"\n  Total already exist : {len(already_exist)}")
print(f"  Total to export     : {len(to_export)}")

if already_exist:
    print(f"\n  NOTE: {len(already_exist)} file(s) already exist and will be SKIPPED.")
    print("  To force a fresh export, delete data/raw/train/, data/raw/val/,")
    print("  and data/raw/test/ sub-folders and re-run this script.")

# ── Export loop ───────────────────────────────────────────────────────────────
print()
print("-" * 65)
print("Exporting images...")
print("-" * 65)

# Load all three split datasets once (avoids reloading inside the loop)
print("Loading dataset splits from cache (no download)...")
datasets = {
    split_name: PneumoniaMNIST(split=split_name, download=False)
    for split_name, _ in SPLITS
}
print("All splits loaded.")
print()

exported_count  = 0
skipped_count   = len(already_exist)
export_errors   = []   # (image_id, error message)

for split_name, i, image_id in to_export:
    png_path = os.path.join(SPLIT_DIRS[split_name], f"{image_id}.png")
    try:
        image, _ = datasets[split_name][i]
        # Convert to grayscale to guarantee single-channel L mode.
        # PneumoniaMNIST returns PIL Images; .convert("L") is a no-op
        # if it's already grayscale, and ensures consistency.
        gray_image = image.convert("L")
        gray_image.save(png_path, format="PNG")
        exported_count += 1
    except Exception as exc:
        export_errors.append((image_id, str(exc)))

# Progress report
for split_name, expected_count in SPLITS:
    split_dir   = SPLIT_DIRS[split_name]
    saved       = len([f for f in os.listdir(split_dir) if f.endswith(".png")])
    print(f"  {split_name:<6}: {saved:>5} PNG files now in {split_dir}")

print()
print(f"  Newly exported : {exported_count}")
print(f"  Skipped (exist): {skipped_count}")
print(f"  Export errors  : {len(export_errors)}")

if export_errors:
    print("\n  [EXPORT ERRORS]")
    for image_id, msg in export_errors[:10]:
        print(f"    {image_id}: {msg}")
    if len(export_errors) > 10:
        print(f"    ... and {len(export_errors) - 10} more (see validation report)")

# ── Verification loop ─────────────────────────────────────────────────────────
print()
print("-" * 65)
print("Verifying every exported PNG...")
print("-" * 65)

validation_rows = []
verify_errors   = 0

for split_name, expected_count in SPLITS:
    split_dir = SPLIT_DIRS[split_name]
    dataset   = datasets[split_name]

    for i in range(expected_count):
        image_id = f"{split_name}_{i}"
        png_path = os.path.join(split_dir, f"{image_id}.png")

        row = {
            "image_id":    image_id,
            "split":       split_name,
            "index":       i,
            "png_path":    png_path,
            "file_exists": False,
            "width_ok":    False,
            "height_ok":   False,
            "pixel_match": False,
            "label_valid": False,
            "status":      "FAIL",
            "notes":       "",
        }

        # Check file exists
        if not os.path.isfile(png_path):
            row["notes"] = "PNG file missing"
            validation_rows.append(row)
            verify_errors += 1
            continue

        row["file_exists"] = True

        try:
            # Re-open the saved PNG
            saved_img  = Image.open(png_path).convert("L")
            saved_arr  = np.array(saved_img, dtype=np.uint8)
            w, h       = saved_img.size

            # Dimension check
            row["width_ok"]  = (w == EXPECTED_W)
            row["height_ok"] = (h == EXPECTED_H)

            # Pixel-level match against the source dataset
            source_img, source_label_arr = dataset[i]
            source_arr = np.array(source_img.convert("L"), dtype=np.uint8)
            source_label = int(source_label_arr[0])

            pixel_match      = np.array_equal(saved_arr, source_arr)
            row["pixel_match"] = pixel_match
            row["label_valid"] = source_label in VALID_LABELS

            issues = []
            if not row["width_ok"]:
                issues.append(f"width={w} (expected {EXPECTED_W})")
            if not row["height_ok"]:
                issues.append(f"height={h} (expected {EXPECTED_H})")
            if not pixel_match:
                issues.append("pixel values differ from source")
            if not row["label_valid"]:
                issues.append(f"invalid label {source_label}")

            if issues:
                row["notes"]  = "; ".join(issues)
                row["status"] = "FAIL"
                verify_errors += 1
            else:
                row["status"] = "PASS"

        except Exception as exc:
            row["notes"]  = f"Verification error: {exc}"
            row["status"] = "FAIL"
            verify_errors += 1

        validation_rows.append(row)

    pass_count = sum(1 for r in validation_rows if r["split"] == split_name and r["status"] == "PASS")
    fail_count = sum(1 for r in validation_rows if r["split"] == split_name and r["status"] == "FAIL")
    print(f"  {split_name:<6}: {pass_count} PASS  |  {fail_count} FAIL")

# ── Build updated manifest with image_path filled in ─────────────────────────
print()
print("-" * 65)
print("Building updated manifest with image_path populated...")
print("-" * 65)

df_new = df_source.copy()

# The source manifest has image_path as all-NaN (float64).
# Cast to object (string) first so pandas accepts string assignments.
df_new["image_path"] = df_new["image_path"].astype(object)

# Build all paths in one vectorised step — avoids row-by-row dtype issues.
df_new["image_path"] = df_new.apply(
    lambda row: f"data/raw/{row['split']}/{row['image_id']}.png",
    axis=1
)

df_new.to_csv(OUTPUT_MANIFEST, index=False)
size_kb = round(os.path.getsize(OUTPUT_MANIFEST) / 1024, 1)
print(f"  Saved: {OUTPUT_MANIFEST}  ({size_kb} KB)")

# Confirm original manifest is unchanged
original_size = os.path.getsize(SOURCE_MANIFEST)
print(f"  Source manifest unchanged: {SOURCE_MANIFEST}  ({round(original_size/1024,1)} KB)")

# ── Save validation report ────────────────────────────────────────────────────
df_val = pd.DataFrame(validation_rows)
df_val.to_csv(VALIDATION_CSV, index=False)
val_size_kb = round(os.path.getsize(VALIDATION_CSV) / 1024, 1)
print(f"  Validation report: {VALIDATION_CSV}  ({val_size_kb} KB)")

# ── Final summary ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("RESULTS SUMMARY")
print(SEP)

total_png = sum(
    len([f for f in os.listdir(SPLIT_DIRS[s]) if f.endswith(".png")])
    for s, _ in SPLITS
)
total_pass = sum(1 for r in validation_rows if r["status"] == "PASS")
total_fail = sum(1 for r in validation_rows if r["status"] == "FAIL")

print(f"  Splits processed  : {len(SPLITS)}")
for split_name, expected_count in SPLITS:
    n_png = len([f for f in os.listdir(SPLIT_DIRS[split_name]) if f.endswith(".png")])
    ok    = "OK" if n_png == expected_count else "MISMATCH"
    print(f"    {split_name:<6}: {n_png:>5} PNG files  (expected {expected_count})  [{ok}]")

print(f"\n  Total PNG files    : {total_png}  (expected {EXPECTED_TOTAL})")
print(f"  Verification PASS  : {total_pass}")
print(f"  Verification FAIL  : {total_fail}")
print(f"  Export errors      : {len(export_errors)}")
print()
print(f"  Output manifest    : {OUTPUT_MANIFEST}  ({size_kb} KB)")
print(f"  Validation report  : {VALIDATION_CSV}  ({val_size_kb} KB)")
print()

all_ok = (
    total_png   == EXPECTED_TOTAL and
    total_pass  == EXPECTED_TOTAL and
    total_fail  == 0              and
    len(export_errors) == 0
)

if all_ok:
    print("OVERALL STATUS: PASSED")
    print("All 5,856 images exported and verified successfully.")
else:
    print("OVERALL STATUS: FAILED")
    if total_png != EXPECTED_TOTAL:
        print(f"  PNG count mismatch: got {total_png}, expected {EXPECTED_TOTAL}.")
    if total_fail > 0:
        print(f"  {total_fail} image(s) failed verification.")
        print(f"  See: {VALIDATION_CSV}")
    if export_errors:
        print(f"  {len(export_errors)} image(s) failed during export.")

print(SEP)
