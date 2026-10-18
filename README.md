# SentinelMed

Medical image quality and reliability assessment prototype.

## P1 - Medical-Image Classification

P1 develops and evaluates ResNet-18 and MobileNetV2 classifiers, with supporting work for calibration, uncertainty estimation, Grad-CAM visualizations, and robustness evaluation.

## Dataset Information

- Dataset name, labels, and image dimensions: pending confirmation from P2.
- Preprocessing and normalization: follow the team's agreed implementation.
- Training and validation splits: coordinate with P2.
- Calibration data: coordinate with P2 and P3.
- Final test set: keep untouched until final evaluation.

## Leakage Prevention

- Keep images derived from the same original image within the same split.
- Do not evaluate the final classifier on training images.
- Do not independently recreate dataset splits in P1.

## Status

Dataset details are pending confirmation from P2. Reported model metrics are available in the `outputs/` directory.
