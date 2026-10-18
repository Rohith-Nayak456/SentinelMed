import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.p1_classifier.data_loader import ManifestImageDataset
from src.p1_classifier.inference import load_classifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = (
    Path.home()
    / "Downloads"
    / "sentinelmed_p2_original_dataset_extracted"
)
MANIFEST = (
    DATA_ROOT
    / "outputs"
    / "manifests"
    / "dataset_manifest_with_paths.csv"
)


def collect_validation_logits(model, dataset, device):
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    all_logits = []
    all_labels = []

    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_logits), torch.cat(all_labels)


def expected_calibration_error(logits, labels, bins=15):
    probabilities = torch.softmax(logits, dim=1)
    confidences, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels).float()

    ece = 0.0
    edges = torch.linspace(0.0, 1.0, bins + 1)

    for index in range(bins):
        if index == 0:
            mask = (
                (confidences >= edges[index])
                & (confidences <= edges[index + 1])
            )
        else:
            mask = (
                (confidences > edges[index])
                & (confidences <= edges[index + 1])
            )

        if mask.any():
            bin_accuracy = correct[mask].mean()
            bin_confidence = confidences[mask].mean()
            fraction = mask.float().mean()
            ece += fraction * torch.abs(
                bin_accuracy - bin_confidence
            )

    return float(ece.item())


def fit_temperature(logits, labels):
    # Optimize log(T) so temperature remains positive.
    log_temperature = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(-3, 3).exp()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    return float(log_temperature.detach().clamp(-3, 3).exp().item())


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["resnet18", "mobilenetv2"],
        default="resnet18",
    )
    args = parser.parse_args()

    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    dataset = ManifestImageDataset(
        str(MANIFEST),
        str(DATA_ROOT),
        split="val",
        image_size=None,
        num_channels=1,
    )

    model, device, version = load_classifier(args.model)

    logits, labels = collect_validation_logits(
        model, dataset, device
    )

    indices = np.arange(len(labels))
    fit_indices, assessment_indices = train_test_split(
        indices,
        test_size=0.5,
        random_state=42,
        stratify=labels.numpy(),
    )

    fit_indices = torch.tensor(fit_indices, dtype=torch.long)
    assessment_indices = torch.tensor(
        assessment_indices, dtype=torch.long
    )

    fit_logits = logits[fit_indices]
    fit_labels = labels[fit_indices]
    assessment_logits = logits[assessment_indices]
    assessment_labels = labels[assessment_indices]

    temperature = fit_temperature(fit_logits, fit_labels)

    before_nll = F.cross_entropy(
        assessment_logits, assessment_labels
    ).item()
    after_nll = F.cross_entropy(
        assessment_logits / temperature, assessment_labels
    ).item()

    before_ece = expected_calibration_error(
        assessment_logits, assessment_labels
    )
    after_ece = expected_calibration_error(
        assessment_logits / temperature, assessment_labels
    )

    output_dir = PROJECT_ROOT / "artifacts" / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model}_temperature.json"

    result = {
        "model_name": args.model,
        "model_version": version,
        "temperature": temperature,
        "fit_images": len(fit_indices),
        "assessment_images": len(assessment_indices),
        "assessment_nll_before": before_nll,
        "assessment_nll_after": after_nll,
        "assessment_ece_before": before_ece,
        "assessment_ece_after": after_ece,
        "ece_bins": 15,
        "validation_only": True,
        "test_split_used": False,
    }

    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=4))
    print(f"\nSaved calibration result: {output_path}")


if __name__ == "__main__":
    main()
