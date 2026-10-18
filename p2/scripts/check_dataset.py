"""
check_dataset.py
----------------
P2 Diagnostic Script — SentinelMed / IBM Datathon 2026

What this script does:
  1. Checks that the medmnist package is installed.
  2. Locates the PneumoniaMNIST cache file on disk (no download).
  3. Verifies the image counts for train, val, and test splits.
  4. Confirms the image size (should be 28 x 28 pixels).
  5. Prints a summary you can share with teammates.

How to run (from the SentinelMed folder):
  python scripts/check_dataset.py
"""

import os
import sys

# ── 1. Check Python version ──────────────────────────────────────────────────
print("=" * 60)
print("SentinelMed — P2 Dataset Diagnostic")
print("=" * 60)
print(f"\nPython version : {sys.version}")

# ── 2. Check medmnist is installed ───────────────────────────────────────────
try:
    import medmnist
    print(f"medmnist version: {medmnist.__version__}")
except ImportError:
    print("\n[ERROR] medmnist is NOT installed.")
    print("Run this command to install it:")
    print("  pip install medmnist")
    sys.exit(1)

# ── 3. Locate the cache file ─────────────────────────────────────────────────
cache_dir = os.path.join(os.path.expanduser("~"), ".medmnist")
cache_file = os.path.join(cache_dir, "pneumoniamnist.npz")

print(f"\nExpected cache folder : {cache_dir}")
print(f"Expected cache file   : {cache_file}")

if not os.path.isfile(cache_file):
    print("\n[ERROR] Cache file not found.")
    print("Run download_dataset.py first to download the dataset.")
    sys.exit(1)

size_mb = os.path.getsize(cache_file) / (1024 * 1024)
print(f"Cache file found      : YES  ({size_mb:.2f} MB)")

# ── 4. Load splits and verify counts ─────────────────────────────────────────
print("\n" + "-" * 60)
print("Loading splits from cache (no download)...")
print("-" * 60)

from medmnist import PneumoniaMNIST

EXPECTED = {"train": 4708, "val": 524, "test": 624}
all_ok = True

for split, expected_count in EXPECTED.items():
    dataset = PneumoniaMNIST(split=split, download=False)
    actual_count = len(dataset)

    # Get the image and label for the first sample
    first_image, first_label = dataset[0]

    # Image size (PIL Image)
    width, height = first_image.size
    label_value = int(first_label[0])  # 0 = Normal, 1 = Pneumonia

    status = "OK" if actual_count == expected_count else "MISMATCH"
    if status == "MISMATCH":
        all_ok = False

    print(f"\n  Split      : {split}")
    print(f"  Count      : {actual_count}  (expected {expected_count})  [{status}]")
    print(f"  Image size : {width} x {height} px")
    print(f"  First label: {label_value}  (0=Normal, 1=Pneumonia)")

# ── 5. Final summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
total = sum(EXPECTED.values())
print(f"Total images across all splits : {total}")

if all_ok:
    print("Dataset integrity check        : PASSED")
    print("\nAll counts match. The dataset is ready for P2 processing.")
else:
    print("Dataset integrity check        : FAILED — see MISMATCH above")
    print("Please re-download the dataset using download_dataset.py")

print("=" * 60)
