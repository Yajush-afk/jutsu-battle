"""Normalized MediaPipe landmark features and their dataset cache."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from jutsu_battle.number_identifier.dataset import NumberSample
from jutsu_battle.number_identifier.hand_detector import HandDetection, HandDetector
from jutsu_battle.number_identifier.preprocessing import read_bgr_image
from jutsu_battle.number_identifier.spec import class_to_index

FEATURES_PER_HAND = 64
LANDMARK_FEATURE_SIZE = FEATURES_PER_HAND * 2


def normalized_landmark_features(detection: HandDetection) -> np.ndarray[Any, Any]:
    """Encode at most two hands relative to each wrist and hand scale."""
    ordered_hands = sorted(detection.hands, key=lambda hand: hand.landmarks[0].x)[:2]
    hand_features: list[float] = []
    for hand in ordered_hands:
        points = np.asarray(
            [(point.x, point.y, point.z) for point in hand.landmarks],
            dtype=np.float32,
        )
        centered = points - points[0]
        scale = float(np.linalg.norm(centered[:, :2], axis=1).max())
        if scale > 1e-6:
            centered /= scale
        hand_features.extend(centered.flatten().tolist())
        hand_features.append(1.0)
    while len(hand_features) < LANDMARK_FEATURE_SIZE:
        hand_features.extend([0.0] * FEATURES_PER_HAND)
    return np.asarray(hand_features[:LANDMARK_FEATURE_SIZE], dtype=np.float32)


def extract_landmark_cache(
    manifest_path: Path,
    capture_root: Path,
    model_path: Path,
    output_path: Path,
) -> Path:
    """Extract deterministic landmark features for one image manifest."""
    samples: list[NumberSample] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        samples = [NumberSample.from_mapping(row) for row in csv.DictReader(handle)]
    if not samples:
        raise ValueError(
            f"Cannot extract landmarks from empty manifest: {manifest_path}"
        )
    features: list[np.ndarray[Any, Any]] = []
    labels: list[int] = []
    label_indices = class_to_index()
    with HandDetector(model_path=model_path, max_hands=2) as detector:
        for sample in samples:
            image = read_bgr_image(capture_root / sample.relative_path)
            features.append(normalized_landmark_features(detector.detect(image)))
            labels.append(label_indices[sample.label])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=np.stack(features),
        labels=np.asarray(labels, dtype=np.int64),
    )
    return output_path


class LandmarkFeatureDataset(Dataset[tuple[Tensor, int]]):
    """Torch dataset loaded from a deterministic compressed feature cache."""

    def __init__(self, cache_path: Path) -> None:
        with np.load(cache_path) as cache:
            self.features = torch.from_numpy(
                cast(np.ndarray[Any, Any], cache["features"]).copy()
            ).float()
            self.labels = torch.from_numpy(
                cast(np.ndarray[Any, Any], cache["labels"]).copy()
            ).long()
        if self.features.ndim != 2 or self.features.shape[1] != LANDMARK_FEATURE_SIZE:
            raise ValueError(f"Invalid landmark feature cache: {cache_path}")
        if len(self.features) != len(self.labels):
            raise ValueError(f"Feature/label length mismatch: {cache_path}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self.features[index], int(self.labels[index].item())
