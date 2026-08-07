"""Tests for MobileNet and landmark comparison models."""

import numpy as np
import torch

from jutsu_battle.number_identifier.hand_detector import (
    DetectedHand,
    HandDetection,
    LandmarkPoint,
)
from jutsu_battle.number_identifier.landmarks import (
    LANDMARK_FEATURE_SIZE,
    normalized_landmark_features,
)
from jutsu_battle.number_identifier.models import (
    LandmarkNumberMLP,
    create_model,
)


def test_mobilenet_output_shape_without_weight_download() -> None:
    """The transfer architecture exposes the same twelve logits."""
    model = create_model("mobilenet_v3_small", pretrained=False)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 160, 160))
    assert logits.shape == (1, 12)


def test_landmark_feature_shape_and_presence_masks() -> None:
    """Two normalized hands produce 128 finite features."""
    landmarks = tuple(
        LandmarkPoint(0.2 + index * 0.01, 0.3 + index * 0.005, index * 0.001)
        for index in range(21)
    )
    hand = DetectedHand(landmarks=landmarks, handedness="Left", score=0.9)
    features = normalized_landmark_features(HandDetection((hand, hand)))
    assert features.shape == (LANDMARK_FEATURE_SIZE,)
    assert np.isfinite(features).all()
    assert features[63] == 1.0
    assert features[127] == 1.0


def test_landmark_mlp_output_shape() -> None:
    """The landmark baseline emits one logit for each class."""
    model = LandmarkNumberMLP()
    model.eval()
    assert model(torch.zeros(2, LANDMARK_FEATURE_SIZE)).shape == (2, 12)
