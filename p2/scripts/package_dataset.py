"""
package_dataset.py
-------------------
P2 Dataset Packaging — SentinelMed / IBM Datathon 2026

Bundles the original PNG images, manifests, quality report,
validation report, and a teammate-facing README into a single
ZIP file for sharing.

HOW TO RUN (from the SentinelMed folder):
    python scripts/package_dataset.py
"""

import os
import sys
import zipfile
import hashlib

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

OUTPUT_ZIP   = os.path.join(PROJECT_ROOT, "sentinelmed_p2_original_dataset.zip")

# Files / folders to include (path on disk -> path inside ZIP)
SPLITS = [
    ("train", 4708),
    ("val",   524),
    ("test",  624),
]
EXPECTED_TOTAL = 5856

CSV_FILES = {
    os.path.join(PROJECT_ROOT, "outputs", "manifests",
                 "dataset_manifest_with_paths.csv"):
        "outputs/manifests/dataset_manifest_with_paths.csv",

    os.path.join(PROJECT_ROOT, "outputs", "quality_reports",
                 "quality_report_full.csv"):
        "outputs/quality_reports/quality_report_full.csv",

    os.path.join(PROJECT_ROOT, "outputs", "validation",
                 "export_validation_report.csv"):
        "outputs/validation/export_validation_report.csv",

    os.path.join(PROJECT_ROOT, "README.md"):
        "README.md",
}

SEP = "=" * 65

# ── Pre-flight: verify all source files exist ─────────────────────────────────
print(SEP)
print("SentinelMed -- P2 Dataset Packaging")
print(SEP)
print(f"\nOutput ZIP : {OUTPUT_ZIP}\n")

all_ok = True

print("Pre-flight checks:")
for src, arc in CSV_FILES.items():
    if os.path.isfile(src):
        kb = round(os.path.getsize(src) / 1024, 1)
        print(f"  OK   {arc}  ({kb} KB)")
    else:
        print(f"  MISSING  {src}")
        all_ok = False

for split_name, expected_count in SPLITS:
    split_dir = os.path.join(PROJECT_ROOT, "data", "raw", split_name)
    if not os.path.isdir(split_dir):
        print(f"  MISSING  data/raw/{split_name}/")
        all_ok = False
        continue
    pngs = [f for f in os.listdir(split_dir) if f.endswith(".png")]
    ok   = len(pngs) == expected_count
    mark = "OK  " if ok else "MISMATCH"
    print(f"  {mark} data/raw/{split_name}/  ({len(pngs)} PNGs, expected {expected_count})")
    if not ok:
        all_ok = False

if not all_ok:
    print("\n[ERROR] Pre-flight failed. Fix the missing files above and re-run.")
    sys.exit(1)

print("\nAll source files verified. Proceeding to package.\n")

# ── Build the ZIP ─────────────────────────────────────────────────────────────
print("-" * 65)
print("Building ZIP...")
print("-" * 65)

png_counts   = {s: 0 for s, _ in SPLITS}
total_added  = 0

with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:

    # 1. CSV files and README
    for src, arc in CSV_FILES.items():
        zf.write(src, arc)
        print(f"  Added: {arc}")

    # 2. PNG images, split by split
    for split_name, _ in SPLITS:
        split_dir = os.path.join(PROJECT_ROOT, "data", "raw", split_name)
        pngs = sorted(f for f in os.listdir(split_dir) if f.endswith(".png"))

        for fname in pngs:
            src_path = os.path.join(split_dir, fname)
            arc_path = f"data/raw/{split_name}/{fname}"
            zf.write(src_path, arc_path)
            png_counts[split_name] += 1
            total_added += 1

        print(f"  Added: data/raw/{split_name}/  ({png_counts[split_name]} PNGs)")

zip_size_mb = round(os.path.getsize(OUTPUT_ZIP) / (1024 * 1024), 1)
print(f"\nZIP written: {OUTPUT_ZIP}  ({zip_size_mb} MB)")

# ── Verify ZIP contents ───────────────────────────────────────────────────────
print()
print("-" * 65)
print("Verifying ZIP contents...")
print("-" * 65)

verify_errors = []

with zipfile.ZipFile(OUTPUT_ZIP, "r") as zf:
    names = set(zf.namelist())

    # Check CSV files and README
    for _, arc in CSV_FILES.items():
        if arc in names:
            print(f"  OK   {arc}")
        else:
            print(f"  MISSING in ZIP: {arc}")
            verify_errors.append(f"Missing: {arc}")

    # Check PNG counts per split
    for split_name, expected_count in SPLITS:
        prefix     = f"data/raw/{split_name}/"
        split_pngs = [n for n in names if n.startswith(prefix) and n.endswith(".png")]
        ok         = len(split_pngs) == expected_count
        mark       = "OK  " if ok else "MISMATCH"
        print(f"  {mark} data/raw/{split_name}/  ({len(split_pngs)} PNGs, "
              f"expected {expected_count})")
        if not ok:
            verify_errors.append(
                f"Split '{split_name}': {len(split_pngs)} PNGs, expected {expected_count}"
            )

    total_pngs_in_zip = sum(
        1 for n in names if n.startswith("data/raw/") and n.endswith(".png")
    )
    total_entries = len(names)

# ── Final summary ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("RESULTS SUMMARY")
print(SEP)
print(f"  ZIP file          : {OUTPUT_ZIP}")
print(f"  ZIP size          : {zip_size_mb} MB")
print(f"  Total entries     : {total_entries}")
print(f"  PNG images        : {total_pngs_in_zip}  (expected {EXPECTED_TOTAL})")
for split_name, expected_count in SPLITS:
    print(f"    {split_name:<6} : {png_counts[split_name]}  (expected {expected_count})")
print(f"  Verification errors: {len(verify_errors)}")
if verify_errors:
    for e in verify_errors:
        print(f"    [ERROR] {e}")
print()

if not verify_errors and total_pngs_in_zip == EXPECTED_TOTAL:
    print("OVERALL STATUS: PASSED")
    print("ZIP is complete and verified. Ready to share with teammates.")
else:
    print("OVERALL STATUS: FAILED")
    print("Review the errors above.")

print(SEP)
