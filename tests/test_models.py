
import torch

from src.p1_classifier.models import (
    create_resnet18,
    create_mobilenet_v2,
)


def test_resnet18():
    model = create_resnet18(num_classes=2)
    model.eval()

    sample = torch.randn(1, 1, 224, 224)

    with torch.no_grad():
        output = model(sample)

    assert output.shape == (1, 2)
    print("ResNet18 test passed:", output.shape)


def test_mobilenet_v2():
    model = create_mobilenet_v2(num_classes=2)
    model.eval()

    sample = torch.randn(1, 1, 224, 224)

    with torch.no_grad():
        output = model(sample)

    assert output.shape == (1, 2)
    print("MobileNetV2 test passed:", output.shape)


if __name__ == "__main__":
    test_resnet18()
    test_mobilenet_v2()
