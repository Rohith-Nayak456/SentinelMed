
import argparse
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from src.p1_classifier.data_loader import ManifestImageDataset
from src.p1_classifier.models import (
    create_mobilenet_v2,
    create_resnet18,
)
from src.p1_classifier.utils import get_device


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = (
    Path.home() / "Downloads" /
    "sentinelmed_p2_original_dataset_extracted"
)

CLASS_NAMES = ["Normal", "Pneumonia"]


def build_model(name):
    if name == "resnet18":
        return create_resnet18(num_classes=2)
    if name == "mobilenetv2":
        return create_mobilenet_v2(num_classes=2)
    raise ValueError(f"Unsupported model: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["resnet18", "mobilenetv2"],
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("batch-size must be positive.")

    manifest = args.manifest or (
        args.data_root
        / "outputs"
        / "manifests"
        / "dataset_manifest_with_paths.csv"
    )
    checkpoint_path = (
        PROJECT_ROOT / "artifacts" / "checkpoints"
        / f"{args.model}_best.pth"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train the model first."
        )

    device = get_device()
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = build_model(args.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load only the final test split. Never use it for training or selection.
    test_dataset = ManifestImageDataset(
        manifest_path=str(manifest),
        data_root=str(args.data_root),
        split="test",
        image_size=None,
        num_channels=1,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_labels = []
    all_predictions = []
    all_probabilities = []
    prediction_records = []
    batch_latencies_ms = []

    image_offset = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)

            start = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            batch_latencies_ms.append(elapsed_ms)

            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            labels_list = labels.tolist()
            predictions_list = predictions.cpu().tolist()
            probabilities_list = probabilities.cpu().tolist()

            batch_size = len(labels_list)
            batch_samples = test_dataset.samples[
                image_offset:image_offset + batch_size
            ]

            for sample, true_label, pred_label, probs in zip(
                batch_samples,
                labels_list,
                predictions_list,
                probabilities_list,
            ):
                image_path, _, image_id = sample

                prediction_records.append({
                    "image_id": image_id,
                    "predicted_class": CLASS_NAMES[pred_label],
                    "true_class": CLASS_NAMES[true_label],
                    "class_probabilities": {
                        CLASS_NAMES[0]: probs[0],
                        CLASS_NAMES[1]: probs[1],
                    },
                    "confidence": max(probs),
                    "model_version": (
                        f"{args.model}-epoch-{checkpoint['epoch']}"
                    ),
                })

            all_labels.extend(labels_list)
            all_predictions.extend(predictions_list)
            all_probabilities.extend(probabilities_list)
            image_offset += batch_size

    metrics = {
        "model": args.model,
        "checkpoint_epoch": checkpoint["epoch"],
        "test_images": len(test_dataset),
        "accuracy": accuracy_score(all_labels, all_predictions),
        "macro_f1": f1_score(
            all_labels,
            all_predictions,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            all_labels, all_predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "mean_batch_inference_ms": (
            sum(batch_latencies_ms) / len(batch_latencies_ms)
        ),
        "mean_inference_ms_per_image": (
            sum(batch_latencies_ms) / len(test_dataset)
        ),
        "latency_note": (
            "Model forward-pass timing; excludes image loading and "
            "preprocessing. Per-image latency is averaged across batches."
        ),
        "probability_note": (
            "Softmax probabilities from the uncalibrated model."
        ),
    }

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"{args.model}_test_metrics.json"
    predictions_path = output_dir / f"{args.model}_test_predictions.json"

    metrics_path.write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    predictions_path.write_text(
        json.dumps(prediction_records, indent=2), encoding="utf-8"
    )

    print(f"Model: {args.model}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("Confusion matrix [rows=true, columns=predicted]:")
    print(metrics["confusion_matrix"])
    print(f"Mean inference latency per image: "
          f"{metrics['mean_inference_ms_per_image']:.3f} ms")
    print(f"Metrics saved: {metrics_path}")
    print(f"Predictions saved: {predictions_path}")


if __name__ == "__main__":
    main()
