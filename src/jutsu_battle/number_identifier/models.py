"""Neural-network models for number-gesture classification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from jutsu_battle.number_identifier.spec import CLASS_LABELS


class ConvBlock(nn.Sequential):
    """Convolution, normalization, activation, and optional downsampling."""

    def __init__(self, in_channels: int, out_channels: int, *, pool: bool) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2))
        super().__init__(*layers)


class MiniNumberCNN(nn.Module):
    """Compact four-block baseline CNN trained from scratch."""

    def __init__(self, class_count: int = len(CLASS_LABELS)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, pool=True),
            ConvBlock(32, 64, pool=True),
            ConvBlock(64, 128, pool=True),
            ConvBlock(128, 256, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, class_count),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return unnormalized class logits."""
        return cast(Tensor, self.classifier(self.pool(self.features(inputs))))


class LandmarkNumberMLP(nn.Module):
    """Small classifier for two normalized 21-point hand skeletons."""

    def __init__(
        self,
        input_features: int = 128,
        class_count: int = len(CLASS_LABELS),
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, class_count),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return unnormalized class logits."""
        return cast(Tensor, self.network(inputs))


def create_model(
    model_name: str,
    *,
    class_count: int = len(CLASS_LABELS),
    pretrained: bool = False,
) -> nn.Module:
    """Construct a model by its stable checkpoint identifier."""
    if model_name == "mini_cnn":
        return MiniNumberCNN(class_count=class_count)
    if model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(input_features, class_count)
        return cast(nn.Module, model)
    if model_name == "landmark_mlp":
        return LandmarkNumberMLP(class_count=class_count)
    raise ValueError(f"Unsupported model: {model_name}")


def parameter_count(model: nn.Module) -> int:
    """Return the number of trainable and frozen parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def load_model_checkpoint(
    path: str,
    *,
    device: torch.device,
) -> tuple[nn.Module, Mapping[str, Any]]:
    """Reconstruct a model from a trusted project checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_name = str(checkpoint["model_name"])
    class_labels = tuple(str(label) for label in checkpoint["class_labels"])
    if not class_labels or len(set(class_labels)) != len(class_labels):
        raise ValueError(f"Checkpoint contains invalid classes: {class_labels!r}")
    model = create_model(model_name, class_count=len(class_labels), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint
