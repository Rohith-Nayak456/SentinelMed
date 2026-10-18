import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.p1_classifier.inference import load_classifier, load_image_tensor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = (
    Path.home()
    / "Downloads"
    / "sentinelmed_p2_original_dataset_extracted"
)


def get_target_layer(model, model_name):
    if model_name == "resnet18":
        return model.layer4[-1]
    if model_name == "mobilenetv2":
        return model.features[18]
    raise ValueError(f"Unsupported model: {model_name}")


def generate_gradcam(model, image_tensor, device, target_layer):
    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)

        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = int(probabilities.argmax(dim=1).item())

        # Explain the class predicted by the model.
        score = logits[0, predicted_class]
        score.backward()

        feature_maps = activations[0]
        feature_gradients = gradients[0]

        # Average gradients over spatial positions to get channel weights.
        weights = feature_gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps.
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)

        return (
            cam.detach().cpu().numpy(),
            predicted_class,
            probabilities[0].detach().cpu().numpy(),
        )
    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default=str(DATA_ROOT / "data" / "raw" / "val" / "val_0.png"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["resnet18", "mobilenetv2"],
        default=["resnet18", "mobilenetv2"],
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    original = Image.open(image_path).convert("L")
    image_tensor = load_image_tensor(str(image_path), "cpu")
    output_dir = PROJECT_ROOT / "outputs" / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = ["Normal", "Pneumonia"]

    for model_name in args.models:
        model, device, version = load_classifier(model_name)
        target_layer = get_target_layer(model, model_name)

        heatmap, predicted_class, probabilities = generate_gradcam(
            model, image_tensor, device, target_layer
        )

        output_path = output_dir / f"{image_path.stem}_{model_name}_gradcam.png"

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")

        axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM heatmap")

        axes[2].imshow(original, cmap="gray")
        axes[2].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axes[2].set_title("Overlay")

        for axis in axes:
            axis.axis("off")

        fig.suptitle(
            f"{model_name} | {version} | "
            f"Predicted: {class_names[predicted_class]} "
            f"({probabilities[predicted_class]:.3f})"
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        print(f"\nModel: {model_name}")
        print(f"Version: {version}")
        print(f"Predicted class: {class_names[predicted_class]}")
        print(f"Probabilities: Normal={probabilities[0]:.6f}, "
              f"Pneumonia={probabilities[1]:.6f}")
        print(f"Saved Grad-CAM: {output_path}")


if __name__ == "__main__":
    main()
