
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter

from src.p1_classifier.inference import (
    CLASS_NAMES,
    load_classifier,
    predict_image,
)
from src.p1_classifier.uncertainty import (
    mc_dropout_predict,
    predictive_entropy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    r"C:\Users\DiyaUser79\Downloads\sentinelmed_p2_original_dataset_extracted"
)
MANIFEST_PATH = DATA_ROOT / "outputs" / "manifests" / "dataset_manifest_with_paths.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "robustness_results.csv"

# Start small to verify the pipeline; set to None to use all validation images.
MAX_IMAGES = None
MC_SAMPLES = 20
SEED = 42

def create_variants(image):
    """Return controlled variants of a grayscale PIL image."""
    image = image.convert("L")

    # Fixed random seed makes noise reproducible.
    rng = np.random.default_rng(SEED)

    pixels = np.asarray(image, dtype=np.float32)

    noisy_pixels = np.clip(
        pixels + rng.normal(0, 12, pixels.shape),
        0,
        255,
    ).astype(np.uint8)

    return {
        "original": image,
        "blur": image.filter(ImageFilter.GaussianBlur(radius=1.0)),
        "noise": Image.fromarray(noisy_pixels),
        "brightness_low": ImageEnhance.Brightness(image).enhance(0.7),
        "brightness_high": ImageEnhance.Brightness(image).enhance(1.3),
        "contrast_low": ImageEnhance.Contrast(image).enhance(0.7),
        "contrast_high": ImageEnhance.Contrast(image).enhance(1.3),
    }

def image_to_tensor(image, device):
    """Match the inference preprocessing: grayscale, 28x28, values [0, 1]."""
    image = image.convert("L").resize((28, 28))
    array = np.array(image, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).float().div(255.0)
    return tensor.unsqueeze(0).unsqueeze(0).to(device)

def predict_variant(image, model, device, model_version):
    """Evaluate one in-memory image variant."""
    tensor = image_to_tensor(image, device)

    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0

    mc = mc_dropout_predict(
        model=model,
        image_tensor=tensor,
        device=device,
        num_samples=MC_SAMPLES,
    )

    predicted_index = int(np.argmax(mc["mean_probabilities"]))
    mc_mean = mc["mean_probabilities"]
    variance = mc["probability_variance"]

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "normal_probability": float(mc_mean[0]),
        "pneumonia_probability": float(mc_mean[1]),
        "deterministic_normal_probability": float(probabilities[0]),
        "deterministic_pneumonia_probability": float(probabilities[1]),
        "confidence": float(max(mc_mean)),
        "predictive_entropy": predictive_entropy(mc_mean),
        "mc_variance_normal": float(variance[0]),
        "mc_variance_pneumonia": float(variance[1]),
        "model_version": model_version,
        "inference_latency_ms": float(latency_ms),
    }

def main():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)
    validation = manifest[
        (manifest["split"] == "val")
        & (manifest["degradation"] == "original")
    ].copy()

    if MAX_IMAGES is not None:
        validation = validation.head(MAX_IMAGES)

    if validation.empty:
        raise ValueError("No original validation images found in manifest.")

    models = {}
    for name in ("resnet18", "mobilenetv2"):
        model, device, version = load_classifier(name)
        models[name] = (model, device, version)

    results = []

    for _, row in validation.iterrows():
        image_id = str(row["image_id"])
        original_id = str(row["original_id"])
        label = int(row["label"])
        image_path = DATA_ROOT / Path(row["image_path"])

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as source:
            original = source.convert("L").copy()

        variants = create_variants(original)

        for variant_name, variant_image in variants.items():
            for model_name, (model, device, version) in models.items():
                prediction = predict_variant(
                    variant_image,
                    model,
                    device,
                    version,
                )

                results.append({
                    "original_id": original_id,
                    "image_id": image_id,
                    "variant": variant_name,
                    "label": label,
                    "model_name": model_name,
                    "correct": (
                        CLASS_NAMES.index(prediction["predicted_class"]) == label
                    ),
                    **prediction,
                })

        print(f"Processed {image_id}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(results)} rows to: {OUTPUT_PATH}")
    print("Images tested:", len(validation))
    print("Variants per image:", len(create_variants(original)))
    print("Models:", ", ".join(models.keys()))

if __name__ == "__main__":
    main()
