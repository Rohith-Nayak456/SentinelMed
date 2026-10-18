
import math

import torch
import torch.nn as nn


def predictive_entropy(class_probabilities) -> float:
    """Calculate predictive entropy in nats."""
    if isinstance(class_probabilities, dict):
        class_probabilities = list(class_probabilities.values())

    probabilities = torch.as_tensor(
        class_probabilities,
        dtype=torch.float64,
    )

    if probabilities.ndim != 1 or probabilities.numel() < 2:
        raise ValueError("Provide at least two class probabilities.")

    if not torch.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")

    if torch.any(probabilities < 0) or torch.any(probabilities > 1):
        raise ValueError("Probabilities must be between 0 and 1.")

    if not math.isclose(
        probabilities.sum().item(),
        1.0,
        rel_tol=1e-5,
        abs_tol=1e-5,
    ):
        raise ValueError("Class probabilities must sum to 1.")

    nonzero = probabilities > 0

    entropy = -torch.sum(
        probabilities[nonzero] * torch.log(probabilities[nonzero])
    )

    return float(entropy.item())


def normalized_predictive_entropy(class_probabilities) -> float:
    """Normalize predictive entropy to the range [0, 1]."""
    if isinstance(class_probabilities, dict):
        num_classes = len(class_probabilities)
    else:
        num_classes = torch.as_tensor(class_probabilities).numel()

    if num_classes < 2:
        raise ValueError("At least two classes are required.")

    return predictive_entropy(class_probabilities) / math.log(num_classes)


def enable_mc_dropout(model):
    """Enable Dropout while keeping BatchNorm layers in evaluation mode."""
    model.eval()
    dropout_found = False

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
            dropout_found = True

    if not dropout_found:
        raise ValueError("This model has no supported Dropout layers.")


def mc_dropout_predict(
    model,
    image_tensor,
    device,
    num_samples=20,
):
    """
    Estimate mean class probabilities and their variance
    over repeated stochastic forward passes.
    """
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2.")

    enable_mc_dropout(model)
    image_tensor = image_tensor.to(device)

    probability_samples = []

    try:
        with torch.inference_mode():
            for _ in range(num_samples):
                logits = model(image_tensor)
                probabilities = torch.softmax(logits, dim=1)
                probability_samples.append(probabilities)

        # Each sample has shape [1, 2].
        # Concatenation produces shape [num_samples, 2].
        samples = torch.cat(probability_samples, dim=0)

        # Both results have shape [2]: one value per class.
        mean_probabilities = samples.mean(dim=0)
        probability_variance = samples.var(dim=0, unbiased=True)

        # IMPORTANT: Do not use [0] here.
        # Indexing [0] would return only the Normal-class scalar.
        return {
            "mean_probabilities": mean_probabilities.cpu().tolist(),
            "probability_variance": probability_variance.cpu().tolist(),
            "num_mc_samples": num_samples,
        }

    finally:
        # Restore normal evaluation mode after MC Dropout.
        model.eval()
