
import torch
import torch.nn as nn
from torchvision import models


def create_resnet18(
    num_classes: int = 2,
    pretrained: bool = False,
    dropout: float = 0.2,
):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    # Dropout before the final classification layer.
    model.dropout = nn.Dropout(p=dropout)
    original_forward = model.forward

    def forward_with_dropout(x):
        # Use ResNet's normal feature extraction, then apply dropout.
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.dropout(x)
        return model.fc(x)

    model.forward = forward_with_dropout
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_mobilenet_v2(
    num_classes: int = 2,
    pretrained: bool = False,
):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    old_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    return model
