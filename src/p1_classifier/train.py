
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from src.p1_classifier.data_loader import create_image_datasets
from src.p1_classifier.models import (
    create_mobilenet_v2,
    create_resnet18,
)
from src.p1_classifier.utils import get_device, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = (
    Path.home() / "Downloads" /
    "sentinelmed_p2_original_dataset_extracted"
)

CLASS_NAMES = ["Normal", "Pneumonia"]


def build_model(name: str):
    if name == "resnet18":
        return create_resnet18(num_classes=2)
    if name == "mobilenetv2":
        return create_mobilenet_v2(num_classes=2)
    raise ValueError(f"Unsupported model: {name}")


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / len(loader.dataset)

    return {
        "loss": average_loss,
        "accuracy": accuracy_score(all_labels, all_predictions),
        "macro_f1": f1_score(
            all_labels, all_predictions,
            labels=[0, 1], average="macro", zero_division=0
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["resnet18", "mobilenetv2"],
        default="resnet18",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive.")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive.")

    manifest = args.manifest or (
        args.data_root
        / "outputs"
        / "manifests"
        / "dataset_manifest_with_paths.csv"
    )

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Manifest: {manifest}")

    # Only train and validation datasets are loaded here.
    train_dataset, val_dataset = create_image_datasets(
        str(manifest),
        str(args.data_root),
        image_size=None,
        num_channels=1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    # Inverse-frequency class weights, calculated from training labels only.
    counts = np.bincount(
        [label for _, label, _ in train_dataset.samples],
        minlength=2,
    )
    if np.any(counts == 0):
        raise ValueError(f"Missing training class. Counts: {counts.tolist()}")

    class_weights = len(train_dataset) / (2.0 * counts)
    weights_tensor = torch.tensor(
        class_weights, dtype=torch.float32, device=device
    )

    print(f"Training class counts [Normal, Pneumonia]: {counts.tolist()}")
    print(f"Class weights: {class_weights.tolist()}")

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate
    )

    checkpoint_dir = PROJECT_ROOT / "artifacts" / "checkpoints"
    output_dir = PROJECT_ROOT / "outputs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{args.model}_best.pth"
    metrics_path = output_dir / f"{args.model}_training_metrics.json"

    best_macro_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_dataset)
        val_metrics = evaluate(
            model, val_loader, device, criterion
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": val_metrics,
        }
        history.append(record)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_accuracy={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "num_classes": 2,
                    "class_mapping": {
                        "0": "Normal",
                        "1": "Pneumonia",
                    },
                    "input_config": {
                        "channels": 1,
                        "height": 28,
                        "width": 28,
                        "pixel_range": [0.0, 1.0],
                    },
                    "epoch": epoch,
                    "validation_macro_f1": best_macro_f1,
                    "seed": args.seed,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")

    results = {
        "model": args.model,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "training_class_counts": counts.tolist(),
        "best_validation_macro_f1": best_macro_f1,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    metrics_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(f"Training metrics saved: {metrics_path}")
    print("Training run complete. Test split was not loaded.")


if __name__ == "__main__":
    main()
