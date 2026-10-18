
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.p1_classifier.models import (
    create_mobilenet_v2,
    create_resnet18,
)
from src.p1_classifier.uncertainty import (
    mc_dropout_predict,
    predictive_entropy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = ["Normal", "Pneumonia"]


def load_classifier(model_name: str, checkpoint_path=None, device=None):
    """Load a trained classifier and return model, device, and version."""

    if model_name == "resnet18":
        model = create_resnet18(num_classes=2)

    elif model_name == "mobilenetv2":
        model = create_mobilenet_v2(num_classes=2)

    else:
        raise ValueError(
            "model_name must be 'resnet18' or 'mobilenetv2'."
        )

    if checkpoint_path is None:
        checkpoint_path = (
            PROJECT_ROOT
            / "artifacts"
            / "checkpoints"
            / f"{model_name}_best.pth"
        )

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    # Support checkpoints saved as either a full training checkpoint
    # or a plain model state dictionary.
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")

    model_version = f"{model_name}-epoch-{epoch}"

    return model, device, model_version


def load_image_tensor(image_path, device):
    """
    Load an image using the agreed preprocessing:
    grayscale, 28x28, and pixel values scaled to [0, 1].
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    try:
        with Image.open(image_path) as image:
            image = image.convert("L").resize((28, 28))

            image_array = np.array(image, copy=True)

    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Cannot read a valid image: {image_path}"
        ) from exc

    image_tensor = (
        torch.from_numpy(image_array)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    return image_tensor.to(device)


def predict_image(
    image_path,
    model,
    device,
    model_version,
    num_mc_samples=20,
):
    """
    Predict one image and return the standardized P1 output.

    Predictive entropy uses the deterministic class probabilities.
    MC Dropout variance is reported separately for each class.

    inference_latency_ms measures the deterministic model forward pass;
    it excludes the additional MC Dropout passes.
    """

    image_path = Path(image_path)

    image_tensor = load_image_tensor(image_path, device)

    model.eval()

    # Measure deterministic classifier inference latency.
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

    if device.type == "cuda":
        torch.cuda.synchronize()

    latency_ms = (time.perf_counter() - start) * 1000.0

    predicted_index = max(
        range(len(probabilities)),
        key=probabilities.__getitem__,
    )

    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    # Calculate predictive entropy from deterministic probabilities.
    entropy = predictive_entropy(probabilities)

    # Estimate class-wise probability variance using MC Dropout.
    mc_result = mc_dropout_predict(
        model=model,
        image_tensor=image_tensor,
        device=device,
        num_samples=num_mc_samples,
    )

    variance_values = mc_result["probability_variance"]

    mc_variance_by_class = {
        name: float(variance_values[index])
        for index, name in enumerate(CLASS_NAMES)
    }

    prediction = {
        "image_id": image_path.stem,
        "predicted_class": predicted_class,
        "probabilities": {
            name: float(probabilities[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "confidence": confidence,
        "uncertainty": {
            "predictive_entropy": float(entropy),
            "mc_dropout_variance": mc_variance_by_class,
        },
        "model_version": model_version,
        "inference_latency_ms": float(latency_ms),
    }

    return prediction
